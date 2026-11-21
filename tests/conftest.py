"""Offline fixtures: a byte-level tokenizer with a Llama-3-style chat template and a tiny Llama."""

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from llm_jepa_sigreg.model import add_special_tokens, configure_padding

# Same structure as the Llama-3.2-Instruct template for plain text messages.
LLAMA3_LIKE_TEMPLATE = (
    "{{- bos_token }}"
    "{%- if not date_string is defined %}{%- set date_string = '26 Jul 2024' %}{%- endif %}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{%- set system_message = messages[0]['content']|trim %}{%- set messages = messages[1:] %}"
    "{%- else %}{%- set system_message = '' %}{%- endif %}"
    "{{- '<|start_header_id|>system<|end_header_id|>\\n\\n' }}"
    "{{- 'Cutting Knowledge Date: December 2023\\n' }}"
    "{{- 'Today Date: ' + date_string + '\\n\\n' }}"
    "{{- system_message }}{{- '<|eot_id|>' }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
    " + message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{%- endif %}"
)

MESSAGES = [
    {"role": "system", "content": "Convert natural language to regular expression."},
    {"role": "user", "content": "lines with a vowel, 3 or more times"},
    {"role": "assistant", "content": "(([AEIOUaeiou])){3,}"},
]

MESSAGES_2 = [
    {"role": "system", "content": "Convert natural language to regular expression."},
    {"role": "user", "content": "lines containing the string 'dog'"},
    {"role": "assistant", "content": ".*dog.*"},
]


def _base_tokenizer() -> PreTrainedTokenizerFast:
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    specials = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|finetune_right_pad_id|>",
    ]
    backend.add_special_tokens(specials)
    bos_id = backend.token_to_id("<|begin_of_text|>")
    backend.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A", special_tokens=[("<|begin_of_text|>", bos_id)]
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>"
    )
    tok.chat_template = LLAMA3_LIKE_TEMPLATE
    return tok


@pytest.fixture(params=[False, True], ids=["dedicated_pad", "pad_as_eos"])
def pad_as_eos(request):
    return request.param


@pytest.fixture
def tokenizer(pad_as_eos):
    tok = _base_tokenizer()
    add_special_tokens(tok)
    configure_padding(tok, pad_as_eos=pad_as_eos)
    return tok


@pytest.fixture
def tiny_model(tokenizer):
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return LlamaForCausalLM(config).float()
