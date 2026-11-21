import json

import pytest
import torch
from conftest import MESSAGES, MESSAGES_2

from llm_jepa_sigreg.evaluate import effective_rank, exact_match
from llm_jepa_sigreg.train import main


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("(a)*", "(a)*", True),
        ("  (a)*\n", "(a)*", True),
        ("(a)*<|end|>", "(a)*", True),
        ("(a)* (a)*", "(a)*", False),
        ("", "(a)*", False),
    ],
)
def test_exact_match_follows_official_rule(prediction, reference, expected):
    assert exact_match(prediction, reference) is expected


def test_effective_rank_bounds():
    torch.manual_seed(0)
    assert effective_rank(torch.randn(4000, 8)) == pytest.approx(8.0, rel=0.05)
    rank_one = torch.randn(100, 1) * torch.randn(1, 8)
    assert effective_rank(rank_one) == pytest.approx(1.0, abs=1e-3)


def _setup(tmp_path, tokenizer, tiny_model) -> list[str]:
    model_dir = tmp_path / "model"
    tiny_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split, n in (("synth_train", 6), ("synth_test", 3)):
        with open(data_dir / f"{split}.jsonl", "w") as f:
            for i in range(n):
                f.write(json.dumps({"messages": (MESSAGES, MESSAGES_2)[i % 2]}) + "\n")
    return [
        "--model-id", str(model_dir), "--data-dir", str(data_dir),
        "--out-dir", str(tmp_path / "out"),
        "--epochs", "1", "--batch-size", "4", "--max-new-tokens", "4",
        "--eval-batch-size", "2", "--dtype", "float32", "--no-verify-data", "--tag", "smoke",
    ]  # fmt: skip


@pytest.mark.parametrize("args", [["--lbd", "0", "--beta", "0"], ["--lbd", "1", "--beta", "0.1"]])
def test_train_cli_smoke(tmp_path, tokenizer, tiny_model, pad_as_eos, args):
    extra = ["--pad-as-eos"] if pad_as_eos else []
    main([*_setup(tmp_path, tokenizer, tiny_model), *args, *extra])
    out = tmp_path / "out"
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["config"]["pad_as_eos"] is pad_as_eos
    assert metrics["train"]["steps"] == 2  # ceil(6 / 4)
    assert 0.0 <= metrics["eval"]["exact_match"] <= 1.0
    assert metrics["eval"]["n"] == 3
    assert metrics["eval"]["split"] == "synth_test"
    for key in ("sigreg_stat", "erank_pred", "erank_target", "cos_mean", "eot_rate"):
        assert key in metrics["eval"]
    log = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert len(log) == 2 and all("ntp" in r and "lr" in r for r in log)
    assert (out / "predictions.jsonl.gz").exists()
    assert not any(p.suffix in {".safetensors", ".bin", ".pt"} for p in out.rglob("*"))


def test_holdout_trains_on_head_and_evaluates_on_tail_of_train(tmp_path, tokenizer, tiny_model):
    main([*_setup(tmp_path, tokenizer, tiny_model), "--holdout", "2"])
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["train"]["steps"] == 1  # 4 training examples, batch 4
    assert metrics["eval"]["n"] == 2
    assert metrics["eval"]["split"] == "synth_train[-2:]"


def test_grad_accum_keeps_one_optimizer_step_per_batch(tmp_path, tokenizer, tiny_model):
    main([*_setup(tmp_path, tokenizer, tiny_model), "--lbd", "1", "--grad-accum", "2"])
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["train"]["steps"] == 2
    log = [json.loads(x) for x in (tmp_path / "out" / "train_log.jsonl").read_text().splitlines()]
    assert all({"ntp", "jepa", "sigreg"} <= set(r) for r in log)
