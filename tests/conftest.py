"""Shared test fixtures.

`toy_tokenizer` is a small byte-level BPE tokenizer with a Qwen-style chat
template, so the prompt code can be tested without downloading the model.
"""

import pytest

# Same shape as Qwen2.5's template: a default system turn, then the user turn.
TOY_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

TOY_CORPUS = [
    "You are a financial analyst. Use only the report below.",
    "### Report\n### Question\nAnswer:",
    "List the numbers from the report that you use, then give the final answer.",
    "revenue | 2009 | 4,215\ncost of sales | 2009 | ( 12 )",
    "the company 's revenue was $ 4215 million in 2009 .",
    "what was the change in revenue from 2008 to 2009 ?",
] * 20


@pytest.fixture(scope="session")
def toy_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = ["<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        TOY_CORPUS,
        trainers.BpeTrainer(vocab_size=400, special_tokens=specials,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    fast.add_special_tokens({"additional_special_tokens": specials})
    fast.chat_template = TOY_CHAT_TEMPLATE
    return fast
