"""Tests for the model-free parts of analysis/b0_attention_plot.py."""

import numpy as np
import pytest

from analysis.b0_attention_plot import (
    answer_numbers, choose_number, parse_heads, plot_attention,
    plot_lookback, top_heads,
)
from common.numbers import extract_numbers
from common.prompt import build_context, encode

TABLE = [["revenue", "2009", "4,215"], ["cost of sales", "2009", "( 12 )"]]
CONTEXT = build_context(["the company 's revenue was $ 4215 million in 2009 ."], TABLE, [])
QUESTION = "what was the change in revenue from 2008 to 2009 ?"


def test_top_heads_and_parse_heads():
    ctx_att = np.zeros((3, 4, 5))
    ctx_att[2, 1] = 0.2   # mass 1.0
    ctx_att[0, 3] = 0.1   # mass 0.5
    assert top_heads(ctx_att, k=2) == [(2, 1), (0, 3)]
    assert parse_heads("10:3,20:5") == [(10, 3), (20, 5)]


def _numbers(tokenizer, answer):
    enc = encode(tokenizer, QUESTION, CONTEXT)
    full = tokenizer(enc["text"] + answer, add_special_tokens=False, return_offsets_mapping=True)
    nums = answer_numbers(answer, full["offset_mapping"], len(enc["text"]))
    return nums, full, len(enc["input_ids"])


def test_answer_numbers_tokens_cover_digits(toy_tokenizer):
    nums, full, _ = _numbers(toy_tokenizer, "growth 7% and revenue $4,215 million")
    assert [n.text for n, _ in nums] == ["7%", "$4,215 million"]
    for n, toks in nums:
        decoded = toy_tokenizer.decode([full["input_ids"][t] for t in toks])
        assert n.text[n.num_start - n.start:n.num_end - n.start] in decoded


def test_choose_number(toy_tokenizer):
    nums, _, n_prompt = _numbers(toy_tokenizer, "growth 7% and revenue $4,215 million")
    report = extract_numbers(CONTEXT)
    assert choose_number(nums, report, n_prompt, 0.01)[0].value == 4.215e9   # the one in the report
    assert choose_number(nums, report, n_prompt, 0.01, index=0)[0].value == 7


def test_choose_number_needs_an_answer_token_before(toy_tokenizer):
    nums, _, n_prompt = _numbers(toy_tokenizer, "4,215 in total")
    assert nums[0][1][0] == n_prompt  # the number is the first answer token
    with pytest.raises(ValueError):
        choose_number(nums, extract_numbers(CONTEXT), n_prompt, 0.01, index=0)


def test_plots_write_files(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    ctx_att = rng.random((4, 3, 50)) / 100
    lookback = rng.random((4, 3))
    plot_attention(tmp_path / "a.png", ctx_att, [(0, 1), (3, 2)], lookback, [10, 11],
                   "title", lambda p: f"tok {p}")
    plot_lookback(tmp_path / "b.png", lookback, "title")
    assert (tmp_path / "a.png").stat().st_size > 0
    assert (tmp_path / "b.png").stat().st_size > 0
