"""Model and tokenizer setup, and the LLM-JEPA (+ SIGReg) objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import IGNORE_INDEX

# Same list as the official finetune.py; tokens already in the vocabulary are skipped.
OFFICIAL_SPECIAL_TOKENS = [f"<|predictor_{i}|>" for i in range(1, 11)] + [
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|perception|>",
]
DEDICATED_PAD = "<|finetune_right_pad_id|>"


def add_special_tokens(tokenizer) -> list[str]:
    vocab = tokenizer.get_vocab()
    new = [t for t in OFFICIAL_SPECIAL_TOKENS if t not in vocab]
    if new:
        tokenizer.add_tokens(new, special_tokens=True)
    return new


def configure_padding(tokenizer, *, pad_as_eos: bool) -> None:
    """Use the reserved Llama pad token, or reproduce the official ``pad = eos`` fallback."""
    if pad_as_eos:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        if DEDICATED_PAD not in tokenizer.get_vocab():
            raise ValueError(f"tokenizer has no {DEDICATED_PAD} token")
        tokenizer.pad_token = DEDICATED_PAD
    tokenizer.padding_side = "right"


def load_model_and_tokenizer(
    model_id: str, *, pad_as_eos: bool, dtype: torch.dtype, revision: str | None = None
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    add_special_tokens(tokenizer)
    configure_padding(tokenizer, pad_as_eos=pad_as_eos)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, revision=revision)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def make_sigreg(num_slices: int = 1024) -> SlicingUnivariateTest:
    """SIGReg as in the LeJEPA reference: sliced Epps-Pulley test against N(0, I)."""
    return SlicingUnivariateTest(EppsPulley(t_max=3.0, n_points=17), num_slices=num_slices)


def ntp_loss(lm_head, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over supervised positions (logits only where needed)."""
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    logits = lm_head(hidden[:, :-1][mask]).float()
    return F.cross_entropy(logits, targets[mask])


def embeddings(hidden: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Last-layer states of the text view (e_pred) and code view (e_target), in fp32."""
    b = batch["user_index"].shape[0]
    rows = torch.arange(b, device=hidden.device)
    e_pred = hidden[b : 2 * b][rows, batch["user_index"]].float()
    e_target = hidden[2 * b : 3 * b][rows, batch["target_index"]].float()
    return e_pred, e_target


def compute_losses(model, batch: dict, *, lbd: float, beta: float, sigreg) -> dict:
    """total = NTP + lbd * (1 - cos(e_pred, e_target)) + beta * SIGReg([e_pred; e_target]).

    The three views share one forward pass. No stop-gradient is applied. When
    ``beta == 0`` the SIGReg statistic is still reported but computed without gradient.
    """
    hidden = model.get_decoder()(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    ).last_hidden_state
    b = batch["labels"].shape[0]
    ntp = ntp_loss(model.get_output_embeddings(), hidden[:b], batch["labels"])
    out = {"ntp": ntp, "total": ntp}
    if "user_index" not in batch:
        return out
    e_pred, e_target = embeddings(hidden, batch)
    cos = F.cosine_similarity(e_pred, e_target, dim=-1).mean()
    stacked = torch.cat([e_pred, e_target])
    if beta > 0:
        sig = sigreg(stacked)
    else:
        with torch.no_grad():
            sig = sigreg(stacked.detach())
    out.update(cos=cos, jepa=1.0 - cos, sigreg=sig)
    total = ntp + lbd * out["jepa"]
    if beta > 0:
        total = total + beta * sig
    out["total"] = total
    return out
