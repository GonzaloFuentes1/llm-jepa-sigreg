"""Exact-match evaluation (official rule) and embedding diagnostics on the test set."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

from .data import DEFAULT_DATE_STRING, collate, encode_prompt
from .model import embeddings, make_sigreg


def exact_match(prediction: str, reference: str) -> bool:
    """Official NL-RX rule: stripped generation equals the reference regex."""
    response = prediction.strip()
    if response.endswith("<|end|>"):
        response = response[: -len("<|end|>")].strip()
    return response == reference


def effective_rank(x: torch.Tensor) -> float:
    """Effective rank (Roy & Vetterli, 2007) of the centered rows of ``x``."""
    x = x.double()
    s = torch.linalg.svdvals(x - x.mean(0))
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def generate(
    model,
    tokenizer,
    conversations: list[list[dict]],
    *,
    batch_size: int,
    max_new_tokens: int,
    date_string: str = DEFAULT_DATE_STRING,
) -> list[dict]:
    """Greedy decoding, batched with left padding; prompts sorted by length."""
    device = next(model.parameters()).device
    prompts = [encode_prompt(tokenizer, m, date_string) for m in conversations]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    config = GenerationConfig(
        do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=eos, pad_token_id=pad
    )
    results: list[dict | None] = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        length = max(len(prompts[i]) for i in chunk)
        input_ids = torch.tensor([[pad] * (length - len(prompts[i])) + prompts[i] for i in chunk])
        mask = torch.tensor(
            [[0] * (length - len(prompts[i])) + [1] * len(prompts[i]) for i in chunk]
        )
        out = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=mask.to(device),
            generation_config=config,
        )
        for row, i in enumerate(chunk):
            ids = out[row, length:].tolist()
            terminated = eos in ids
            if terminated:
                ids = ids[: ids.index(eos)]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            reference = conversations[i][2]["content"]
            results[i] = {
                "prediction": text,
                "reference": reference,
                "correct": exact_match(text, reference),
                "terminated": terminated,
            }
    return results


@torch.no_grad()
def embedding_stats(
    model, encoded: list[dict], *, pad_id: int, batch_size: int, num_slices: int, last_token: int
) -> dict:
    """Cosine, SIGReg statistic and effective rank of test-set JEPA embeddings.

    The SIGReg statistic is averaged over chunks of ``batch_size`` pairs
    (2 * batch_size embeddings), the same sample size seen in training, with a
    fresh, deterministic set of slices.
    """
    device = next(model.parameters()).device
    preds, targets = [], []
    for start in range(0, len(encoded), batch_size):
        batch = collate(
            encoded[start : start + batch_size],
            pad_id=pad_id,
            with_views=True,
            last_token=last_token,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        hidden = model.get_decoder()(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state
        e_pred, e_target = embeddings(hidden, batch)
        preds.append(e_pred)
        targets.append(e_target)
    e_pred, e_target = torch.cat(preds), torch.cat(targets)
    sigreg = make_sigreg(num_slices).to(device)
    stats = [
        float(sigreg(torch.cat([e_pred[s : s + batch_size], e_target[s : s + batch_size]])))
        for s in range(0, len(e_pred), batch_size)
    ]
    return {
        "cos_mean": float(F.cosine_similarity(e_pred, e_target, dim=-1).mean()),
        "sigreg_stat": sum(stats) / len(stats),
        "erank_pred": effective_rank(e_pred),
        "erank_target": effective_rank(e_target),
        "erank_joint": effective_rank(torch.cat([e_pred, e_target])),
        "norm_pred": float(e_pred.norm(dim=-1).mean()),
        "norm_target": float(e_target.norm(dim=-1).mean()),
    }
