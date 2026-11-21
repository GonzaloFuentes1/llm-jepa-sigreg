"""Full fine-tuning with NTP, LLM-JEPA and SIGReg terms, then evaluation on the test split.

Optimization follows the official run (HF Trainer defaults as used by finetune.py):
bf16 weights, AdamW (betas 0.9/0.999, eps 1e-8, no weight decay), linear decay to
zero without warmup, gradient clipping at 1.0, global batch 128 (8 GPUs x 4 x 4
accumulation in run.sh), 4 epochs. No checkpoint is written: the model is evaluated
in-process and only metrics, logs and predictions are saved.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from .data import DEFAULT_DATE_STRING, collate, encode_example, load_split
from .evaluate import embedding_stats, generate
from .model import compute_losses, load_model_and_tokenizer, make_sigreg

LOSS_KEYS = ("total", "ntp", "jepa", "cos", "sigreg")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--revision", default=None)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--train-split", default="synth_train")
    p.add_argument("--test-split", default="synth_test")
    p.add_argument("--no-verify-data", action="store_true", help="skip the sha256 check")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="run", help="condition name used by aggregate.py")
    p.add_argument("--seed", type=int, default=82)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128, help="examples per optimizer step")
    p.add_argument("--grad-accum", type=int, default=1, help="micro-batches per step")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None, help="stop early (pilot runs)")
    p.add_argument("--lbd", type=float, default=0.0, help="JEPA weight (lambda)")
    p.add_argument("--beta", type=float, default=0.0, help="SIGReg weight")
    p.add_argument("--predictors", type=int, default=1, help="number of predictor tokens k")
    p.add_argument("--last-token", type=int, default=-2, help="-2 for Llama (token before eot)")
    p.add_argument("--num-slices", type=int, default=1024)
    p.add_argument("--pad-as-eos", action="store_true", help="official pad = eos behaviour")
    p.add_argument("--label-scope", choices=["all", "assistant"], default="all")
    p.add_argument("--date-string", default=DEFAULT_DATE_STRING)
    p.add_argument("--max-eval", type=int, default=None, help="evaluate on the first N tests")
    p.add_argument(
        "--holdout",
        type=int,
        default=0,
        help="train on all but the last N training examples and evaluate on those N "
        "(validation for hyperparameter selection; the test split is not used)",
    )
    p.add_argument("--eval-batch-size", type=int, default=250)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    return p.parse_args(argv)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _environment(device: torch.device) -> dict:
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
    return env


def load_data(args) -> tuple[list, list, str]:
    """Return (train conversations, evaluation conversations, evaluation split name)."""
    verify = not args.no_verify_data
    train_set = load_split(args.train_split, args.data_dir, verify=verify)
    if args.holdout > 0:
        return (
            train_set[: -args.holdout],
            train_set[-args.holdout :],
            f"{args.train_split}[-{args.holdout}:]",
        )
    eval_set = load_split(args.test_split, args.data_dir, verify=verify)
    if args.max_eval is not None:
        eval_set = eval_set[: args.max_eval]
    return train_set, eval_set, args.test_split


def train(args, model, tokenizer, device, conversations) -> tuple[list[dict], dict]:
    encoded = [
        encode_example(
            tokenizer,
            m,
            predictors=args.predictors,
            label_scope=args.label_scope,
            date_string=args.date_string,
        )
        for m in conversations
    ]
    with_views = args.lbd > 0 or args.beta > 0
    steps_per_epoch = math.ceil(len(encoded) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: max(0.0, 1.0 - s / max(1, total_steps))
    )
    sigreg = make_sigreg(args.num_slices).to(device)
    generator = torch.Generator().manual_seed(args.seed)

    model.train()
    log, step, start = [], 0, time.time()
    for epoch in range(args.epochs):
        order = torch.randperm(len(encoded), generator=generator).tolist()
        for first in range(0, len(order), args.batch_size):
            if step >= total_steps:
                break
            indices = order[first : first + args.batch_size]
            micro = math.ceil(len(indices) / args.grad_accum)
            sums = dict.fromkeys(LOSS_KEYS, 0.0)
            for m in range(0, len(indices), micro):
                chunk = indices[m : m + micro]
                batch = collate(
                    [encoded[i] for i in chunk],
                    pad_id=tokenizer.pad_token_id,
                    with_views=with_views,
                    last_token=args.last_token,
                )
                batch = {k: v.to(device) for k, v in batch.items()}
                losses = compute_losses(model, batch, lbd=args.lbd, beta=args.beta, sigreg=sigreg)
                weight = len(chunk) / len(indices)
                (losses["total"] * weight).backward()
                for key in LOSS_KEYS:
                    if key in losses:
                        sums[key] += float(losses[key].detach()) * weight
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            lr = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            record = {"step": step, "epoch": epoch, "lr": lr, "grad_norm": float(grad_norm)}
            record.update({k: v for k, v in sums.items() if k in losses})
            record["elapsed_s"] = round(time.time() - start, 2)
            log.append(record)
            step += 1
    summary = {"steps": step, "train_seconds": round(time.time() - start, 1)}
    if log:
        tail = log[-10:]
        for key in LOSS_KEYS:
            if key in tail[-1]:
                summary[f"final_{key}"] = sum(r[key] for r in tail) / len(tail)
    return log, summary


def evaluate(args, model, tokenizer, conversations, split) -> tuple[list[dict], dict]:
    model.eval()
    model.config.use_cache = True
    start = time.time()
    predictions = generate(
        model,
        tokenizer,
        conversations,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        date_string=args.date_string,
    )
    encoded = [
        encode_example(
            tokenizer,
            m,
            predictors=args.predictors,
            label_scope=args.label_scope,
            date_string=args.date_string,
        )
        for m in conversations
    ]
    stats = embedding_stats(
        model,
        encoded,
        pad_id=tokenizer.pad_token_id,
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        last_token=args.last_token,
    )
    n = len(predictions)
    metrics = {
        "split": split,
        "n": n,
        "exact_match": sum(p["correct"] for p in predictions) / n,
        "eot_rate": sum(p["terminated"] for p in predictions) / n,
        "eval_seconds": round(time.time() - start, 1),
        **stats,
    }
    return predictions, metrics


def main(argv=None) -> dict:
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_everything(args.seed)
    dtype = getattr(torch, args.dtype)
    model, tokenizer = load_model_and_tokenizer(
        args.model_id, pad_as_eos=args.pad_as_eos, dtype=dtype, revision=args.revision
    )
    model.to(device)
    model.config.use_cache = False
    model.activation_checkpointing_enable({"use_reentrant": False})

    train_set, eval_set, eval_split = load_data(args)
    log, train_summary = train(args, model, tokenizer, device, train_set)
    predictions, eval_metrics = evaluate(args, model, tokenizer, eval_set, eval_split)
    if device.type == "cuda":
        train_summary["max_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)

    metrics = {
        "tag": args.tag,
        "config": vars(args),
        "tokenizer": {
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "vocab_size": len(tokenizer),
        },
        "environment": _environment(device),
        "train": train_summary,
        "eval": eval_metrics,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "train_log.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in log), encoding="utf-8"
    )
    with gzip.open(out / "predictions.jsonl.gz", "wt", encoding="utf-8") as f:
        for i, p in enumerate(predictions):
            f.write(json.dumps({"i": i, **p}) + "\n")
    print(json.dumps({"tag": args.tag, "seed": args.seed, **eval_metrics}), flush=True)
    return metrics


if __name__ == "__main__":
    main()
