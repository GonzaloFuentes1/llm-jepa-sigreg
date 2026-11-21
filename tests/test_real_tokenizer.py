"""Checks against the real Llama-3.2 tokenizer; skipped unless LLM_JEPA_TOKENIZER is set.

Example: LLM_JEPA_TOKENIZER=unsloth/Llama-3.2-1B-Instruct pytest tests/test_real_tokenizer.py
"""

import os

import pytest
from conftest import MESSAGES, MESSAGES_2

from llm_jepa_sigreg.data import IGNORE_INDEX, collate, encode_example
from llm_jepa_sigreg.model import add_special_tokens, configure_padding

TOKENIZER = os.environ.get("LLM_JEPA_TOKENIZER")
pytestmark = pytest.mark.skipif(not TOKENIZER, reason="LLM_JEPA_TOKENIZER not set")


@pytest.fixture
def real_tokenizer(pad_as_eos):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    add_special_tokens(tok)
    configure_padding(tok, pad_as_eos=pad_as_eos)
    return tok


def test_real_tokenizer_embedding_tokens_and_eot_labels(real_tokenizer, pad_as_eos):
    tok = real_tokenizer
    assert tok.pad_token_id == (128009 if pad_as_eos else 128004)
    examples = [
        encode_example(tok, m, predictors=1, label_scope="all") for m in (MESSAGES, MESSAGES_2)
    ]
    batch = collate(examples, pad_id=tok.pad_token_id, with_views=True)
    b = len(examples)
    for i, ex in enumerate(examples):
        user_tok = tok.convert_ids_to_tokens(int(batch["input_ids"][b + i, batch["user_index"][i]]))
        target_id = int(batch["input_ids"][2 * b + i, batch["target_index"][i]])
        assert user_tok == "<|predictor_1|>"
        assert ex["target_ids"][-1] == 128009
        assert target_id == ex["target_ids"][-2]
        eot_labels = [
            lab for t, lab in zip(ex["full_ids"], ex["full_labels"], strict=True) if t == 128009
        ]
        assert len(eot_labels) == 3
        assert all(lab == (IGNORE_INDEX if pad_as_eos else 128009) for lab in eot_labels)
