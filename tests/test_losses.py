import pytest
import torch
from conftest import MESSAGES, MESSAGES_2

from llm_jepa_sigreg.data import collate, encode_example
from llm_jepa_sigreg.model import compute_losses, make_sigreg


def _batch(tokenizer, with_views=True, label_scope="all"):
    examples = [
        encode_example(tokenizer, m, predictors=1, label_scope=label_scope)
        for m in (MESSAGES, MESSAGES_2, MESSAGES, MESSAGES_2)
    ]
    return collate(examples, pad_id=tokenizer.pad_token_id, with_views=with_views)


def test_hidden_state_used_is_hf_last_hidden_state(tokenizer, tiny_model):
    batch = _batch(tokenizer)
    with torch.no_grad():
        ref = tiny_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        ).hidden_states[-1]
        ours = tiny_model.get_decoder()(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state
    assert torch.allclose(ref, ours)


@pytest.mark.parametrize("label_scope", ["all", "assistant"])
def test_loss_reduces_to_ntp_when_lambda_and_beta_are_zero(tokenizer, tiny_model, label_scope):
    batch = _batch(tokenizer, label_scope=label_scope)
    b = batch["labels"].shape[0]
    with torch.no_grad():
        hf = tiny_model(
            input_ids=batch["input_ids"][:b],
            attention_mask=batch["attention_mask"][:b],
            labels=batch["labels"],
        ).loss
        losses = compute_losses(tiny_model, batch, lbd=0.0, beta=0.0, sigreg=make_sigreg(64))
    assert torch.allclose(losses["total"], hf, atol=1e-5)
    assert torch.allclose(losses["ntp"], hf, atol=1e-5)
    # Views are still encoded, so the monitored terms are available.
    assert 0.0 <= float(losses["jepa"]) <= 2.0
    assert float(losses["sigreg"]) > 0.0


def test_regular_batch_without_views_gives_plain_ntp(tokenizer, tiny_model):
    batch = _batch(tokenizer, with_views=False)
    with torch.no_grad():
        hf = tiny_model(**{k: batch[k] for k in ("input_ids", "attention_mask", "labels")}).loss
        losses = compute_losses(tiny_model, batch, lbd=1.0, beta=1.0, sigreg=make_sigreg(64))
    assert torch.allclose(losses["total"], hf, atol=1e-5)
    assert set(losses) == {"total", "ntp"}


def test_total_is_ntp_plus_lambda_jepa_plus_beta_sigreg(tokenizer, tiny_model):
    batch = _batch(tokenizer)
    with torch.no_grad():
        losses = compute_losses(tiny_model, batch, lbd=0.7, beta=0.3, sigreg=make_sigreg(64))
    expected = losses["ntp"] + 0.7 * losses["jepa"] + 0.3 * losses["sigreg"]
    assert torch.allclose(losses["total"], expected, atol=1e-5)
    assert torch.allclose(losses["jepa"], 1.0 - losses["cos"])


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def test_gradient_flows_through_sigreg(tokenizer, tiny_model):
    batch = _batch(tokenizer)
    sigreg = make_sigreg(64)
    sig = compute_losses(tiny_model, batch, lbd=0.0, beta=1.0, sigreg=sigreg)["sigreg"]
    assert sig.requires_grad
    sig.backward()
    grads = _grads(tiny_model)
    assert grads, "SIGReg produced no parameter gradients"
    assert any(g.abs().sum() > 0 for g in grads.values())


def test_sigreg_changes_the_gradient_only_when_beta_is_positive(tokenizer, tiny_model):
    batch = _batch(tokenizer)

    def total_grad(beta):
        tiny_model.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        compute_losses(tiny_model, batch, lbd=1.0, beta=beta, sigreg=make_sigreg(64))[
            "total"
        ].backward()
        return _grads(tiny_model)

    g0, g0_again, g1 = total_grad(0.0), total_grad(0.0), total_grad(1.0)
    name = "model.norm.weight"
    assert torch.allclose(g0[name], g0_again[name])
    assert not torch.allclose(g0[name], g1[name])


def test_sigreg_is_small_for_gaussian_and_large_for_collapsed_embeddings():
    torch.manual_seed(0)
    sigreg = make_sigreg(256)
    gaussian = sigreg(torch.randn(512, 32))
    collapsed = sigreg(torch.randn(512, 1).expand(512, 32).contiguous() + 3.0)
    assert float(collapsed) > 100 * float(gaussian)
