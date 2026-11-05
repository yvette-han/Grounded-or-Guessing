"""Tests for analysis/stage1.py. Expected values are worked out by hand."""

import json

import pytest

from analysis.stage1 import (
    bootstrap_ci, build_table, distinct_count, final_number, gold_anywhere, is_correct,
    same_final, setting_metrics, write_outputs,
)

TOL = 0.01
QUESTIONS = {
    "A/2009/page_1.pdf-1": {"report_key": "A/2009", "gold_answer": 0.0807692},
    "A/2009/page_1.pdf-2": {"report_key": "A/2009", "gold_answer": 100.0},
    "B/2010/page_3.pdf-1": {"report_key": "B/2010", "gold_answer": 5.0},
}
IDS = sorted(QUESTIONS)
Q1, Q2, Q3 = "A/2009/page_1.pdf-1", "A/2009/page_1.pdf-2", "B/2010/page_3.pdf-1"


def answer(text, tokens=10, cut=False):
    return {"answer_text": text, "n_new_tokens": tokens, "hit_max_tokens": cut}


GREEDY = {
    Q1: answer("4,215 and 3,900. Final answer: 8.1%.", 10),   # correct (gold 0.0808)
    Q2: answer("Answer: 90", 20, cut=True),                   # wrong
    Q3: answer("Answer: 5", 30),                              # correct
}
RECORDS = [
    {"id": Q1, "label": "copied"}, {"id": Q1, "label": "copied"}, {"id": Q1, "label": "computed"},
    {"id": Q2, "label": "fabricated"},
    {"id": Q3, "label": "computed"},
]


def test_final_number_and_correctness():
    assert final_number("4,215 and 3,900. Final answer: 8.1%.").value == 8.1
    assert final_number("In 2009 nothing changed.") is None
    assert is_correct("Final answer: 8.1%", 0.0807692, TOL)
    assert not is_correct("Final answer: 9.5%", 0.0807692, TOL)
    # only the last number counts for accuracy; gold_anywhere looks at all of them
    assert not is_correct("8.1%, so revenue is 4,215", 0.0807692, TOL)
    assert gold_anywhere("8.1%, so revenue is 4,215", 0.0807692, TOL)


def test_same_final_and_distinct_count():
    assert same_final(None, None, TOL)
    assert not same_final(None, final_number("5"), TOL)
    finals = [final_number(t) for t in ["8.1%", "0.081", "no number", "9.5%", "none either"]]
    assert distinct_count(finals, TOL) == 3   # {8.1% = 0.081}, {no number}, {9.5%}


def test_bootstrap_ci():
    assert bootstrap_ci([(1, 1), (2, 2)]) == (1.0, 1.0)
    assert bootstrap_ci([(1, 2), (2, 4)]) == (0.5, 0.5)
    lo, hi = bootstrap_ci([(0, 3), (3, 3), (1, 2), (2, 4)], seed=1)
    assert 0 <= lo <= 6 / 12 <= hi <= 1


def test_setting_metrics_by_hand():
    row = setting_metrics("greedy", GREEDY, RECORDS, QUESTIONS, IDS, TOL)
    assert row["questions"] == 3
    assert row["accuracy"] == pytest.approx(2 / 3)
    assert row["accuracy_ci_low"] <= row["accuracy"] <= row["accuracy_ci_high"]
    assert row["gold_anywhere"] == pytest.approx(2 / 3)
    assert row["numbers"] == 5
    assert row["fabricated_share"] == pytest.approx(1 / 5)
    assert row["answers_with_fabricated"] == pytest.approx(1 / 3)
    assert row["copied_share"] == pytest.approx(2 / 5)
    assert row["computed_share"] == pytest.approx(2 / 5)
    assert row["question_share"] == 0 and row["wrong_calc_share"] == 0
    assert row["numbers_per_answer"] == pytest.approx(5 / 3)
    assert row["mean_new_tokens"] == pytest.approx(20)
    assert row["cut_at_max"] == pytest.approx(1 / 3)


def test_differs_from_greedy():
    greedy_finals = {i: final_number(GREEDY[i]["answer_text"]) for i in IDS}
    other = {Q1: answer("Final answer: 8.1%"),   # same as greedy
             Q2: answer("Answer: 100"),          # greedy said 90
             Q3: answer("no idea")}              # greedy said 5
    row = setting_metrics("temp12", other, None, QUESTIONS, IDS, TOL, greedy_finals)
    assert row["differs_from_greedy"] == pytest.approx(2 / 3)
    assert "fabricated_share" not in row       # no labels for this setting


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_build_table_uses_common_questions(tmp_path):
    _write_jsonl(tmp_path / "subset.jsonl", [dict(id=i, **QUESTIONS[i]) for i in IDS])
    _write_jsonl(tmp_path / "answers_greedy.jsonl", [dict(id=i, **GREEDY[i]) for i in IDS])
    _write_jsonl(tmp_path / "answers_temp12.jsonl", [dict(id=i, **GREEDY[i]) for i in IDS[:2]])
    _write_jsonl(tmp_path / "numbers_greedy.jsonl", RECORDS)
    cfg = {"label_tolerance": TOL, "decoding": {"greedy": {}, "beam5": {}, "temp12": {}}}
    rows, info = build_table(cfg, tmp_path / "subset.jsonl", tmp_path, tmp_path)
    assert [r["setting"] for r in rows] == ["greedy", "temp12"]
    assert info["common_questions"] == 2
    assert info["missing_answers"] == ["beam5"]
    assert info["missing_labels"] == ["temp12"]
    assert info["mean_distinct_finals"] == pytest.approx(1.0)  # identical answers

    pytest.importorskip("matplotlib")
    write_outputs(rows, info, tmp_path / "out")
    for name in ("stage1_table.csv", "stage1_table.md", "stage1_bars.png"):
        assert (tmp_path / "out" / name).stat().st_size > 0
