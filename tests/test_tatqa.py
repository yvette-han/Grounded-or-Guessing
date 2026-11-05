"""Tests for common/tatqa.py. Expected values are worked out by hand."""

import pytest

from common.tatqa import _decimals, convert, derivation_steps, evaluate


@pytest.mark.parametrize("derivation, steps", [
    ("44.1-56.7", [-12.6]),
    ("(44.1-56.7)/56.7", [-12.6, -12.6 / 56.7]),
    ("4.00 - 1.90 ", [2.1]),
    ("[(2.9+2.9)/2] - [(2.7+2.7)/2]", [5.8, 2.9, 5.4, 2.7, 0.2]),   # [ ] are brackets
    ("($166.3-$513.3)/$513.3", [-347.0, -347.0 / 513.3]),
    ("-(3+2)", [5.0, -5.0]),                                          # the final value is added
    ("4,215", [4215.0]),
    ("159 millions / 1,068,645 thousands", []),                       # words: not plain arithmetic
    ("5/0", []),
])
def test_evaluate(derivation, steps):
    assert evaluate(derivation) == pytest.approx(steps)


def test_decimals():
    assert (_decimals(0.39), _decimals(14491), _decimals(0.2)) == (2, 0, 1)


@pytest.mark.parametrize("derivation, answer, scale, last", [
    ("(44.1-56.7)/56.7", -22.22, "percent", -12.6 / 56.7),   # percent answer = value x 100
    ("(1.7%+1.5%+1.5%)/3", 1.57, "percent", 4.7 / 3),        # "%" just part of the number
    ("1,027/11%", 9336.36, "", 1027 / 0.11),                 # "%" means / 100
    ("7,572 - (6,919) ", 14491.0, "", 14491.0),              # (6,919) is negative
    ("(213+73)/1,607 ", 0.18, "", 286 / 1607),               # answer rounded to 2 decimals
])
def test_derivation_steps_reproduces_answer(derivation, answer, scale, last):
    steps, ok = derivation_steps(derivation, answer, scale)
    assert ok
    assert steps[-1] == pytest.approx(last)


def test_derivation_steps_not_reproduced():
    assert derivation_steps("159 millions / 1,068,645 thousands", 14.88, "percent") == ([], False)
    steps, ok = derivation_steps("10-4", 99.0, "")
    assert not ok and steps == [6.0]


def test_convert():
    item = {
        "table": {"uid": "T1", "table": [["", "2019", "2018"], ["Revenue", "$ 4,215", "$ 3,900"]]},
        "paragraphs": [{"uid": "p2", "order": 2, "text": "Second paragraph."},
                       {"uid": "p1", "order": 1, "text": "First paragraph."}],
        "questions": [
            {"uid": "q1", "question": " What is the change in revenue? ", "answer": 315, "derivation": "4,215-3,900",
             "answer_type": "arithmetic", "scale": "million"},
            {"uid": "q2", "question": "What is revenue?", "answer": ["$ 4,215"], "derivation": "",
             "answer_type": "span", "scale": "million"},
            {"uid": "q3", "question": "When?", "answer": "31 January 2029", "derivation": "",
             "answer_type": "arithmetic", "scale": ""},
        ],
    }
    records, not_reproduced = convert([item])
    assert not_reproduced == 0
    assert len(records) == 1                  # only q1: arithmetic with a numeric answer
    r = records[0]
    assert r["context_text"] == " | 2019 | 2018\nRevenue | $ 4,215 | $ 3,900\nFirst paragraph.\nSecond paragraph."
    assert (r["id"], r["report_key"], r["split"], r["question"]) == ("q1", "tatqa/T1", "test",
                                                                     "What is the change in revenue?")
    assert (r["gold_answer"], r["gold_program"], r["gold_steps"], r["gold_scale"]) == (315.0, "4,215-3,900",
                                                                                       [315.0], "million")
