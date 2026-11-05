import pytest

import re

from common.prompt import (
    build_context, build_prompt, encode, encode_answer, table_to_text, tokens_in_span,
)

TABLE = [["revenue", "2009", "4,215"], ["cost of sales", "2009", "( 12 )"]]
PRE = ["the company 's revenue was $ 4215 million in 2009 .", ".", "  "]
POST = [".", "cost of sales fell ."]
QUESTION = "what was the change in revenue from 2008 to 2009 ?"


def test_table_to_text():
    assert table_to_text(TABLE) == "revenue | 2009 | 4,215\ncost of sales | 2009 | ( 12 )"


def test_table_to_text_strips_cells():
    assert table_to_text([[" a ", "b  "]]) == "a | b"


def test_build_context_order_and_junk_lines():
    assert build_context(PRE, TABLE, POST) == (
        "the company 's revenue was $ 4215 million in 2009 .\n"
        "revenue | 2009 | 4,215\n"
        "cost of sales | 2009 | ( 12 )\n"
        "cost of sales fell ."
    )


def test_build_context_keeps_lines_with_digits():
    # A line like "( 1 )" has no letters but is real content.
    assert build_context(["( 1 )"], [["a"]], []) == "( 1 )\na"


def test_build_prompt_matches_readme():
    expected = (
        "You are a financial analyst. Use only the report below.\n\n"
        "### Report\n"
        "CTX\n\n"
        "### Question\n"
        "Q?\n\n"
        "List the numbers from the report that you use, then give the final answer.\n"
        "Answer:"
    )
    assert build_prompt(" Q? ", "CTX") == expected


def test_tokens_in_span():
    offsets = [(0, 3), (3, 4), (4, 4), (4, 9), (9, 12)]  # (4, 4) is an empty special token
    assert tokens_in_span(offsets, 3, 9) == [1, 3]
    assert tokens_in_span(offsets, 2, 4) == [0, 1]     # partial overlap counts
    assert tokens_in_span(offsets, 12, 15) == []


@pytest.mark.parametrize("chat_template", [True, False])
def test_encode_context_span(toy_tokenizer, chat_template):
    context = build_context(PRE, TABLE, POST)
    enc = encode(toy_tokenizer, QUESTION, context, chat_template)
    ids, s, e = enc["input_ids"], enc["ctx_start"], enc["ctx_end"]
    assert 0 < s < e < len(ids)
    c0, c1 = enc["ctx_chars"]
    assert enc["text"][c0:c1] == context

    # The context tokens decode to the report (plus at most a straddling newline).
    assert toy_tokenizer.decode(ids[s:e]).strip() == context
    # The tokens just outside the span are the prompt around the report.
    assert toy_tokenizer.decode(ids[:s]).endswith("### Report\n")
    assert toy_tokenizer.decode(ids[e:]).lstrip("\n").startswith("### Question")


def test_encode_chat_template(toy_tokenizer):
    context = build_context(PRE, TABLE, POST)
    with_chat = encode(toy_tokenizer, QUESTION, context, chat_template=True)
    plain = encode(toy_tokenizer, QUESTION, context, chat_template=False)

    assert with_chat["text"].startswith("<|im_start|>system")
    assert with_chat["text"].endswith("<|im_start|>assistant\n")
    assert plain["text"] == build_prompt(QUESTION, context)
    # The chat template adds tokens before the report, so the span shifts but keeps its length.
    assert with_chat["ctx_start"] > plain["ctx_start"]
    assert (with_chat["ctx_end"] - with_chat["ctx_start"]) == (plain["ctx_end"] - plain["ctx_start"])
    # Tokenizing the text equals what apply_chat_template itself produces.
    direct = toy_tokenizer.apply_chat_template(
        [{"role": "user", "content": build_prompt(QUESTION, context)}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )
    assert with_chat["input_ids"] == list(direct)


def test_encode_with_real_model_tokenizer():
    """Runs only where the Qwen tokenizer can be downloaded (e.g. Colab)."""
    try:
        import yaml
        from transformers import AutoTokenizer
        with open("config.yaml") as f:
            model = yaml.safe_load(f)["model"]
        tok = AutoTokenizer.from_pretrained(model)
    except Exception as exc:  # offline or blocked
        pytest.skip(f"model tokenizer not available: {exc.__class__.__name__}")

    context = build_context(PRE, TABLE, POST)
    enc = encode(tok, QUESTION, context, chat_template=True)
    ids, s, e = enc["input_ids"], enc["ctx_start"], enc["ctx_end"]
    assert tok.decode(ids[s:e]).strip() == context
    direct = tok.apply_chat_template(
        [{"role": "user", "content": build_prompt(QUESTION, context)}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )
    assert ids == list(direct)


class NewlineMergingTokenizer:
    """One token per character, except a run of newlines is one token (like Qwen's)."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [m.span() for m in re.finditer(r"\n+|.", text, re.S)]
        out = {"input_ids": [text[s:e] for s, e in spans]}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


def test_encode_answer(toy_tokenizer):
    context = build_context(PRE, TABLE, POST)
    enc = encode(toy_tokenizer, QUESTION, context)
    answer = "revenue was 4,215 in 2009 ."
    full = encode_answer(toy_tokenizer, enc["text"], answer, enc["input_ids"])
    n = full["answer_start"]
    assert n == len(enc["input_ids"])
    assert full["input_ids"][:n] == enc["input_ids"]
    assert full["answer_char_start"] == len(enc["text"])
    assert toy_tokenizer.decode(full["input_ids"][n:]) == answer
    assert all(s >= len(enc["text"]) for s, _ in full["offsets"][n:])
    # Same result when the prompt ids are not passed in.
    assert encode_answer(toy_tokenizer, enc["text"], answer) == full


def test_encode_answer_detects_changed_prompt_tokens():
    tok = NewlineMergingTokenizer()
    assert encode_answer(tok, "abc\n", "xyz")["answer_start"] == 4
    with pytest.raises(ValueError):
        encode_answer(tok, "abc\n", "\nxyz")  # "\n" + "\n" becomes one token
