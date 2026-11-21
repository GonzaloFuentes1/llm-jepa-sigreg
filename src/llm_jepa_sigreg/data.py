"""NL-RX-SYNTH loading, the three LLM-JEPA views, label masks and batching.

The views and tokenization mirror the official LLM-JEPA code for Llama models
(galilai-group/llm-jepa, finetune.py):

* full view:   system + user + assistant, rendered with the chat template;
* text view:   the user turn alone (the template inserts its default system header),
               with k predictor tokens appended to the user content;
* code view:   the assistant turn alone, keeping the ``assistant`` role.

As in the official code, the rendered template (which already starts with
``<|begin_of_text|>``) is tokenized with ``add_special_tokens=True``, so every
sequence starts with two BOS tokens. Training and evaluation use the same
convention.
"""

from __future__ import annotations

import copy
import hashlib
import json
import urllib.request
from pathlib import Path

import torch

IGNORE_INDEX = -100
DEFAULT_DATE_STRING = "26 Jul 2024"

OFFICIAL_COMMIT = "ea0017c654ad917066ff32afc88276bea8ca5f7e"
DATA_URL = (
    "https://raw.githubusercontent.com/galilai-group/llm-jepa/{commit}/datasets/{split}.jsonl"
)
SHA256 = {
    "synth_train": "ca959f108fb5a9c363910bca46cd9de69ead3bac016ed2564c74be39ba634e46",
    "synth_test": "acfce67c9284bb97dc0b89ad9412b00f6464faf2dfbd2581dd4930140716e45d",
}


def load_split(split: str, data_dir: str | Path, verify: bool = True) -> list[list[dict]]:
    """Return the conversations of ``split``, downloading it from the official repo if needed."""
    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = DATA_URL.format(commit=OFFICIAL_COMMIT, split=split)
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    if verify:
        digest = hashlib.sha256(raw).hexdigest()
        if digest != SHA256[split]:
            raise ValueError(f"{path}: sha256 {digest} does not match the pinned {SHA256[split]}")
    return [json.loads(line)["messages"] for line in raw.decode("utf-8").splitlines() if line]


def build_views(messages: list[dict], predictors: int) -> tuple[list, list, list]:
    """Return (full, text, code) message lists for one system/user/assistant conversation."""
    full = copy.deepcopy(messages)
    user = copy.deepcopy(messages[1:2])
    user[0]["content"] += "".join(f"<|predictor_{i}|>" for i in range(predictors, 0, -1))
    target = copy.deepcopy(messages[2:3])
    return full, user, target


def _render(tokenizer, messages, date_string, add_generation_prompt=False) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        date_string=date_string,
    )
    return list(tokenizer(text, add_special_tokens=True)["input_ids"])


def encode_prompt(tokenizer, messages: list[dict], date_string: str = DEFAULT_DATE_STRING):
    """Token ids of the generation prompt (system + user + assistant header)."""
    return _render(tokenizer, messages[:2], date_string, add_generation_prompt=True)


def encode_example(
    tokenizer,
    messages: list[dict],
    *,
    predictors: int,
    label_scope: str,
    date_string: str = DEFAULT_DATE_STRING,
) -> dict[str, list[int]]:
    """Tokenize the three views and build next-token labels for the full view.

    ``label_scope="all"`` reproduces what the official pipeline trains on: its
    ``DataCollatorForLanguageModeling(mlm=False)`` replaces the prepared labels by
    ``input_ids`` and masks only tokens equal to ``pad_token_id``.
    ``label_scope="assistant"`` supervises only the answer and its closing token.
    In both cases tokens equal to ``pad_token_id`` are masked, so with
    ``pad_token == eos_token`` every genuine ``<|eot_id|>`` target is dropped.
    """
    full, user, target = build_views(messages, predictors)
    full_ids = _render(tokenizer, full, date_string)
    labels = list(full_ids)
    if label_scope == "assistant":
        prompt = encode_prompt(tokenizer, messages, date_string)
        if full_ids[: len(prompt)] != prompt:
            raise ValueError("generation prompt is not a prefix of the full conversation")
        labels[: len(prompt)] = [IGNORE_INDEX] * len(prompt)
    elif label_scope != "all":
        raise ValueError(f"unknown label_scope {label_scope!r}")
    pad_id = tokenizer.pad_token_id
    labels = [
        IGNORE_INDEX if tok == pad_id else lab for tok, lab in zip(full_ids, labels, strict=True)
    ]
    return {
        "full_ids": full_ids,
        "full_labels": labels,
        "user_ids": _render(tokenizer, user, date_string),
        "target_ids": _render(tokenizer, target, date_string),
    }


def _right_pad(rows: list[list[int]], length: int, value: int) -> torch.Tensor:
    return torch.tensor([row + [value] * (length - len(row)) for row in rows], dtype=torch.long)


def last_token_index(attention_mask: torch.Tensor, offset: int) -> torch.Tensor:
    """Index ``unpadded_length + offset`` per row, like the official ``_last_token_index``.

    The official code measures the unpadded length with the attention mask, not with
    the token ids, so the index does not depend on which token is used for padding.
    With Llama and ``offset=-2`` it points at the token right before the final
    ``<|eot_id|>``.
    """
    lengths = attention_mask.sum(-1)
    positions = torch.arange(attention_mask.shape[-1], device=attention_mask.device)
    if not torch.equal(attention_mask.bool(), positions < lengths[:, None]):
        raise ValueError("expected right padding")
    return lengths + offset


def collate(
    examples: list[dict], *, pad_id: int, with_views: bool, last_token: int = -2
) -> dict[str, torch.Tensor]:
    """Right-pad a batch. With views, rows are [full; text; code] (3B rows)."""
    rows = [ex["full_ids"] for ex in examples]
    if with_views:
        rows += [ex["user_ids"] for ex in examples] + [ex["target_ids"] for ex in examples]
    length = max(len(r) for r in rows)
    input_ids = _right_pad(rows, length, pad_id)
    attention_mask = _right_pad([[1] * len(r) for r in rows], length, 0)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": _right_pad([ex["full_labels"] for ex in examples], length, IGNORE_INDEX),
    }
    if with_views:
        b = len(examples)
        batch["user_index"] = last_token_index(attention_mask[b : 2 * b], last_token)
        batch["target_index"] = last_token_index(attention_mask[2 * b :], last_token)
    return batch
