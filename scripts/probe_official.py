"""Run the official LLM-JEPA data pipeline and report what actually reaches the model.

Needs a local clone of https://github.com/galilai-group/llm-jepa (not vendored here).
For two examples it prints, under the official ``pad = eos`` fallback and under the
reserved ``<|finetune_right_pad_id|>``:

* the token at the embedding index returned by ``RepresentationTrainer._last_token_index``
  (``last_token=-2``) for the text and code views;
* which positions of the full conversation keep a label after the official
  ``DataCollatorForLanguageModeling(mlm=False)``, and the labels of the genuine
  ``<|eot_id|>`` tokens.

The Meta tokenizer files are read from a public copy whose git blob ids equal those of
meta-llama/Llama-3.2-1B-Instruct (see verify_mirror.py); like Meta's, it has no pad token.

Usage: python scripts/probe_official.py OFFICIAL_CLONE DATA_JSONL [--tokenizer ID]
"""

import argparse
import sys
import types

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("official_clone")
    p.add_argument("data_jsonl")
    p.add_argument("--tokenizer", default="alpindale/Llama-3.2-1B-Instruct")
    p.add_argument("--n", type=int, default=2)
    args = p.parse_args()

    torch.cuda.current_device = lambda: 0  # the official code calls it unconditionally
    sys.path.insert(0, args.official_clone)
    import finetune as official
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling

    eot = "<|eot_id|>"
    for pad in ("eos (official fallback)", "<|finetune_right_pad_id|>"):
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        assert tok.pad_token is None, "expected the Meta tokenizer (no pad token)"
        new = [t for t in ("<|predictor_1|>", "<|perception|>") if t not in tok.vocab]
        tok.add_special_tokens({"additional_special_tokens": new})
        tok.pad_token = tok.eos_token if pad.startswith("eos") else pad
        data = official.load_and_prepare_dataset(
            args.data_jsonl, tok, "meta-llama/Llama-3.2-1B-Instruct", max_length=512, predictors=1
        )
        batch = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)(
            [data[i] for i in range(args.n)]
        )
        trainer = types.SimpleNamespace(last_token=-2, debug=0)
        index = {
            view: official.RepresentationTrainer._last_token_index(
                trainer,
                batch[f"input_ids_{view}"],
                batch[f"labels_{view}"],
                batch[f"attention_mask_{view}"],
            )
            for view in ("user", "assistant")
        }
        print(f"===== pad = {pad} (pad_token_id={tok.pad_token_id}, eos_id={tok.eos_token_id})")
        name = tok.convert_ids_to_tokens
        for i in range(args.n):
            for view in ("user", "assistant"):
                ids = batch[f"input_ids_{view}"][i]
                j = int(index[view][i])
                at, after = name(int(ids[j])), name(int(ids[j + 1]))
                print(f"example {i} {view:9s} view: index {j} -> {at!r} (next {after!r})")
            ids, labels = batch["input_ids"][i], batch["labels"][i]
            real = batch["attention_mask"][i].bool()
            supervised = (labels != -100).nonzero().flatten()
            eot_pos = ((ids == tok.convert_tokens_to_ids(eot)) & real).nonzero().flatten().tolist()
            print(
                f"example {i} full view: {int(real.sum())} real tokens, "
                f"{len(supervised)} supervised (first {name(int(ids[supervised[0]]))!r}); "
                f"labels at the {len(eot_pos)} genuine {eot}: {[int(labels[k]) for k in eot_pos]}"
            )
            for view, label in (("user", "text"), ("assistant", "code")):
                mask = batch[f"attention_mask_{view}"][i].bool()
                print(f"  {label} view: {tok.decode(batch[f'input_ids_{view}'][i][mask])!r}")


if __name__ == "__main__":
    main()
