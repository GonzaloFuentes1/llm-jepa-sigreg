import pytest
import torch
from conftest import MESSAGES, MESSAGES_2

from llm_jepa_sigreg.data import IGNORE_INDEX, build_views, collate, encode_example


def _tokens(tokenizer, ids):
    return tokenizer.convert_ids_to_tokens([int(i) for i in ids])


def test_build_views_match_official_llama_layout():
    full, user, target = build_views(MESSAGES, predictors=2)
    assert full == MESSAGES
    # Text view: only the user turn (the template adds its default system header),
    # with k predictor tokens appended in descending order as in the official code.
    assert user == [
        {"role": "user", "content": MESSAGES[1]["content"] + "<|predictor_2|><|predictor_1|>"}
    ]
    # Code view: the assistant turn keeps its role for Llama.
    assert target == [MESSAGES[2]]
    assert MESSAGES[1]["content"].endswith("times")  # input is not mutated


@pytest.mark.parametrize("predictors", [0, 1, 2])
def test_last_token_index_is_token_before_final_eot(tokenizer, pad_as_eos, predictors):
    examples = [
        encode_example(tokenizer, m, predictors=predictors, label_scope="all")
        for m in (MESSAGES, MESSAGES_2)
    ]
    batch = collate(examples, pad_id=tokenizer.pad_token_id, with_views=True)
    b = len(examples)
    user_rows = batch["input_ids"][b : 2 * b]
    target_rows = batch["input_ids"][2 * b :]
    for i, messages in enumerate((MESSAGES, MESSAGES_2)):
        u = int(batch["user_index"][i])
        t = int(batch["target_index"][i])
        assert _tokens(tokenizer, [user_rows[i, u + 1]]) == ["<|eot_id|>"]
        assert _tokens(tokenizer, [target_rows[i, t + 1]]) == ["<|eot_id|>"]
        # The embedding position is the last real token, never padding.
        assert batch["attention_mask"][b + i, u] == 1
        assert batch["attention_mask"][2 * b + i, t] == 1
        if predictors > 0:
            assert _tokens(tokenizer, [user_rows[i, u]]) == ["<|predictor_1|>"]
        else:
            last_user_char = messages[1]["content"][-1]
            assert tokenizer.decode([int(user_rows[i, u])]) == last_user_char
        assert tokenizer.decode([int(target_rows[i, t])]) == messages[2]["content"][-1]


def test_index_does_not_depend_on_pad_token(tokenizer):
    ex = encode_example(tokenizer, MESSAGES, predictors=1, label_scope="all")
    ex2 = encode_example(tokenizer, MESSAGES_2, predictors=1, label_scope="all")
    eos_batch = collate([ex, ex2], pad_id=tokenizer.eos_token_id, with_views=True)
    pad_batch = collate([ex, ex2], pad_id=tokenizer.pad_token_id, with_views=True)
    assert torch.equal(eos_batch["user_index"], pad_batch["user_index"])
    assert torch.equal(eos_batch["target_index"], pad_batch["target_index"])


def test_double_bos_is_kept_like_the_official_pipeline(tokenizer):
    ex = encode_example(tokenizer, MESSAGES, predictors=1, label_scope="all")
    for key in ("full_ids", "user_ids", "target_ids"):
        assert _tokens(tokenizer, ex[key][:2]) == ["<|begin_of_text|>", "<|begin_of_text|>"]


def test_label_scope_all_with_dedicated_pad_supervises_every_token(tokenizer, pad_as_eos):
    ex = encode_example(tokenizer, MESSAGES, predictors=1, label_scope="all")
    ids, labels = ex["full_ids"], ex["full_labels"]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eot_positions = [i for i, t in enumerate(ids) if t == eot]
    assert len(eot_positions) == 3  # system, user and assistant turns
    if pad_as_eos:
        # Official behaviour: the collator masks every label equal to pad_id == eos_id.
        assert all(labels[i] == IGNORE_INDEX for i in eot_positions)
        assert all(labels[i] == ids[i] for i in range(len(ids)) if i not in eot_positions)
    else:
        assert labels == ids


def test_label_scope_assistant_supervises_answer_and_its_eot(tokenizer, pad_as_eos):
    ex = encode_example(tokenizer, MESSAGES, predictors=1, label_scope="assistant")
    ids, labels = ex["full_ids"], ex["full_labels"]
    supervised = [i for i, lab in enumerate(labels) if lab != IGNORE_INDEX]
    text = tokenizer.decode([ids[i] for i in supervised])
    if pad_as_eos:
        assert text == MESSAGES[2]["content"]
    else:
        assert text == MESSAGES[2]["content"] + "<|eot_id|>"
    assert supervised == list(range(supervised[0], supervised[-1] + 1))
    assert supervised[-1] in (len(ids) - 1, len(ids) - 2)


def test_collate_right_pads_and_never_supervises_padding(tokenizer):
    examples = [
        encode_example(tokenizer, m, predictors=1, label_scope="all")
        for m in (MESSAGES, MESSAGES_2)
    ]
    batch = collate(examples, pad_id=tokenizer.pad_token_id, with_views=True)
    b = len(examples)
    assert batch["input_ids"].shape[0] == 3 * b
    assert batch["labels"].shape == (b, batch["input_ids"].shape[1])
    mask = batch["attention_mask"]
    # Right padding: every row is a block of ones followed by zeros.
    lengths = mask.sum(-1)
    for row, n in zip(mask, lengths, strict=True):
        assert row[:n].all() and not row[n:].any()
    assert (batch["labels"][mask[:b] == 0] == IGNORE_INDEX).all()
    assert (batch["input_ids"][mask == 0] == tokenizer.pad_token_id).all()


def test_collate_without_views_only_contains_full_conversation(tokenizer):
    examples = [encode_example(tokenizer, MESSAGES, predictors=1, label_scope="all")]
    batch = collate(examples, pad_id=tokenizer.pad_token_id, with_views=False)
    assert batch["input_ids"].shape[0] == 1
    assert "user_index" not in batch
