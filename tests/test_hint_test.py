"""Tests for analysis/hint_test.py (the parts that need no model). Expected values worked out by hand."""

import pytest

from analysis.hint_test import (
    find_counterpart, focus_number, gold_input_lines, gold_program_args, hinted_question, is_cut_tail,
    left_context, pick_candidates,
)


def test_hinted_question():
    assert hinted_question("what was the change ? ", " Use the 2009 row.") == "what was the change ? Use the 2009 row."
    with pytest.raises(ValueError):
        hinted_question("q", "  ")


def test_gold_program_args_skips_steps_and_constants():
    assert gold_program_args("divide(-14611, 739187)") == [-14611.0, 739187.0]
    assert gold_program_args("subtract(2.36, 1.96), divide(#0, 1.96)") == [2.36, 1.96, 1.96]
    assert gold_program_args("add(const_4, 44), multiply(#0, const_100)") == [44.0]


def test_gold_input_lines():
    context = "intro 2008 text\nadjusted cash flow | 739187\nnon-tower | $ -14611 ( 14611 )\nother | 741565"
    # 741565 is within 1% of 739187 but not equal, so its line is not included
    assert gold_input_lines(context, "divide(-14611, 739187)") == [
        "adjusted cash flow | 739187", "non-tower | $ -14611 ( 14611 )"]


def _answer(text, cut):
    return {"answer_text": text, "hit_max_tokens": cut}


def test_cut_tail_and_focus():
    a = _answer("x 100. Sum: 37. Average: 12.33. Then $1", cut=True)
    recs = [(4, {"label": "fabricated", "char_start": 2, "char_end": 5, "value": 100.0}),
            (5, {"label": "copied", "char_start": 12, "char_end": 14, "value": 37.0}),
            (6, {"label": "fabricated", "char_start": 25, "char_end": 30, "value": 12.33}),
            (7, {"label": "fabricated", "char_start": 37, "char_end": 39, "value": 1.0})]
    assert is_cut_tail(recs[3][1], a) and not is_cut_tail(recs[2][1], a)
    assert focus_number(recs, a)[0] == 6                                  # skips the 100 and the cut-off tail
    assert focus_number(recs, a, skip_values=())[0] == 4                  # --keep-100
    assert focus_number(recs[3:], a) is None                              # only a cut-off tail
    assert focus_number(recs[3:], _answer(a["answer_text"], cut=False))[0] == 7   # not cut: a real number


def test_find_counterpart_by_left_context():
    before = "Revenue 2009: $4,215\nAdjusted cash flow: $73,918,700\nNon-tower: $14,611"
    after = "Revenue 2009: $4,215\nNon-tower: $14,611\nAdjusted cash flow: $739,187"
    after_recs = [{"char_start": after.index("$4,215")}, {"char_start": after.index("$14,611")},
                  {"char_start": after.index("$739,187")}]
    i, ratio = find_counterpart(before, before.index("$73,918,700"), after, after_recs)
    assert i == 2 and ratio == 1.0          # same line text: "Adjusted cash flow: "
    assert find_counterpart(before, 10, "no numbers", []) == (None, 0.0)
    assert left_context("abcdef", 4, width=2) == "cd"
    assert left_context("total\nsum: 5", 11) == "sum: "            # stops at the line start
    assert left_context("total\n5", 6) == "total\n"                # number starts its line: plain window


def test_pick_candidates_is_seeded_and_marks_the_number():
    questions = {f"q{i}": {"split": "test" if i % 2 else "train", "question": "?", "gold_answer": 1.0,
                           "gold_program": "add(1, 2)", "context_text": "a | 1\nb | 2"} for i in range(6)}
    answers = {qid: {"answer_text": "x 9 y", "hit_max_tokens": False} for qid in questions}
    records = [{"id": qid, "setting": "greedy", "number_text": "9", "value": 9.0, "char_start": 2, "char_end": 3,
                "label": "fabricated" if i < 4 else "copied"} for i, qid in enumerate(questions)]
    rows = pick_candidates(records, answers, questions, n=3, seed=1)
    assert len(rows) == 3 and rows == pick_candidates(records, answers, questions, n=3, seed=1)
    assert {r["id"] for r in rows} <= {"q0", "q1", "q2", "q3"}           # only answers with a hallucinated number
    assert rows[0]["answer"] == "x ⟦9⟧ y" and rows[0]["hint"] == ""
    assert rows[0]["gold_input_lines"] == "a | 1 / b | 2"
    test_rows = pick_candidates(records, answers, questions, n=10, split="test")
    assert {r["id"] for r in test_rows} == {"q1", "q3"}


def test_main_follows_extends_and_run_name(tmp_path, monkeypatch):
    # `candidates` with an `extends` config reads labels/<run_name>/ (and doesn't fail on a missing key).
    import sys
    from pathlib import Path

    from analysis import hint_test
    root = Path(hint_test.__file__).resolve().parents[1]
    cfg = tmp_path / "other.yaml"
    cfg.write_text(f"extends: {root / 'config.yaml'}\nmodel: some/other-model\nrun_name: zz_test_run\n")
    monkeypatch.setattr(sys, "argv", ["hint_test.py", "candidates", "--config", str(cfg)])
    with pytest.raises(FileNotFoundError) as err:
        hint_test.main()
    assert "labels/zz_test_run" in str(err.value)
