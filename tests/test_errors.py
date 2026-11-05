"""Tests for analysis/errors.py. Expected values are worked out by hand."""

import csv
import json

import numpy as np
import pytest

from analysis.errors import (
    choose_detector, gold_steps_matching, mark, pick_errors, report_lines_with, run_all, score_summary,
)
from common.numbers import extract_numbers
from detector.train_eval import DETECTORS

TOL = 0.01


def test_choose_detector_uses_val_auroc_and_skips_random():
    rows = [{"detector": "lookback", "val_auroc": "0.80"}, {"detector": "hf", "val_auroc": "0.85"},
            {"detector": "lookback_hf", "val_auroc": "0.83"}, {"detector": "confidence", "val_auroc": "0.70"},
            {"detector": "random", "val_auroc": "0.99"}]
    assert choose_detector(rows) == "hf"


def _ts():
    ts = {"label": np.array(["copied", "computed", "fabricated", "fabricated", "copied", "question", "computed"]),
          "lookback": np.array([0.9, 0.2, 0.3, 0.8, 0.1, 0.95, 0.7]),
          "lookback_flagged": np.array([1, 0, 0, 1, 0, 1, 1])}
    return ts


def test_pick_errors_by_hand():
    # false positives: copied/computed flagged -> 0 (0.9), 6 (0.7); the flagged question (5) doesn't count
    # false negatives: fabricated not flagged -> 2
    fp, fn = pick_errors(_ts(), "lookback", n=10, pick="extreme")
    assert list(fp) == [0, 6] and list(fn) == [2]
    fp, _ = pick_errors(_ts(), "lookback", n=1, pick="extreme")
    assert list(fp) == [0]                                  # the most confident one
    a, _ = pick_errors(_ts(), "lookback", n=1, seed=3)
    b, _ = pick_errors(_ts(), "lookback", n=1, seed=3)
    assert list(a) == list(b) and a[0] in (0, 6)            # random, reproducible
    with pytest.raises(ValueError):
        pick_errors(_ts(), "lookback", pick="worst")


def test_mark():
    assert mark("growth was 9.5% .", 11, 15) == "growth was ⟦9.5%⟧ ."


def test_report_lines_with():
    context = "revenue | 2009 | 4215\ncost | 2009 | 3,100\nrevenue was $ 4215 million ."
    number = extract_numbers("$4,215")[0]
    # the bare table value and "4215 million" (in millions) both match
    assert report_lines_with(context, number, TOL) == ["revenue | 2009 | 4215", "revenue was $ 4215 million ."]


def test_gold_steps_matching():
    q = {"gold_answer": 0.0807692, "gold_steps": [315.0, 0.0807692, None]}
    assert gold_steps_matching(extract_numbers("8.1%")[0], q, TOL) == ["answer", "step 1"]
    assert gold_steps_matching(extract_numbers("315")[0], q, TOL) == ["step 0"]
    assert gold_steps_matching(extract_numbers("999")[0], q, TOL) == []


def test_score_summary_by_hand():
    ts = {"label": np.array(["copied", "copied", "fabricated"]), "lookback": np.array([0.1, 0.3, 0.9]),
          "lookback_flagged": np.array([0, 0, 1])}
    rows = {r["label"]: r for r in score_summary(ts, "lookback")}
    assert rows["copied"]["n"] == 2
    assert rows["copied"]["mean"] == pytest.approx(0.2)
    assert (rows["copied"]["q25"], rows["copied"]["q75"]) == pytest.approx((0.15, 0.25))
    assert rows["fabricated"]["flagged"] == 1.0
    assert "wrong_calc" not in rows


def test_run_all_end_to_end(tmp_path):
    qid = "X/2009/page_1.pdf-1"
    answer = "Numbers: 4,215 and 3,900. Growth: 9.5%."
    records = [  # labels/numbers_greedy.jsonl
        {"id": qid, "setting": "greedy", "number_text": "4,215", "value": 4215.0, "char_start": 9, "char_end": 14,
         "tok_start": 3, "tok_end": 5, "label": "copied"},
        {"id": qid, "setting": "greedy", "number_text": "3,900", "value": 3900.0, "char_start": 19, "char_end": 24,
         "tok_start": 6, "tok_end": 8, "label": "copied"},
        {"id": qid, "setting": "greedy", "number_text": "9.5%", "value": 9.5, "char_start": 34, "char_end": 38,
         "tok_start": 11, "tok_end": 13, "label": "fabricated"},
    ]
    (tmp_path / "numbers_greedy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (tmp_path / "answers_greedy.jsonl").write_text(json.dumps({"id": qid, "answer_text": answer}) + "\n")
    (tmp_path / "subset.jsonl").write_text(json.dumps({
        "id": qid, "question": "growth?", "gold_answer": 0.0807692, "gold_program": "subtract(4215, 3900), divide(#0, 3900)",
        "gold_steps": [315.0, 0.0807692], "context_text": "revenue | 2009 | 4215\nrevenue | 2008 | 3900"}) + "\n")
    with open(tmp_path / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["detector", "val_auroc", "threshold"])
        w.writeheader()
        w.writerows({"detector": d, "val_auroc": 0.9 if d == "lookback" else 0.5, "threshold": 0.5} for d in DETECTORS)
    with open(tmp_path / "test_scores.csv", "w", newline="") as f:
        cols = ["setting", "row", "id", "report", "label"] + [c for d in DETECTORS for c in (d, f"{d}_flagged")]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row, (score, label) in enumerate([(0.8, "copied"), (0.2, "copied"), (0.1, "fabricated")]):
            r = {"setting": "greedy", "row": row, "id": qid, "report": "X/2009", "label": label}
            r.update({c: (score if not c.endswith("_flagged") else int(score >= 0.5)) for c in cols[5:]})
            w.writerow(r)

    cfg = {"label_tolerance": TOL, "seed": 0}
    detector, examples, summary = run_all(cfg, tmp_path / "subset.jsonl", tmp_path, tmp_path, tmp_path,
                                          tmp_path / "out")
    assert detector == "lookback"
    fp, fn = [e for e in examples if e["kind"] == "fp"], [e for e in examples if e["kind"] == "fn"]
    assert [e["number"] for e in fp] == ["4,215"] and [e["number"] for e in fn] == ["9.5%"]
    assert fp[0]["answer_marked"] == "Numbers: ⟦4,215⟧ and 3,900. Growth: 9.5%."
    assert fp[0]["report_lines"] == "revenue | 2009 | 4215"
    assert fn[0]["answer_labels"] == "4,215 copied · 3,900 copied · 9.5% fabricated"
    md = (tmp_path / "out" / "b5_examples.md").read_text()
    assert "False positive 1" in md and "False negative 1" in md
    pytest.importorskip("matplotlib")
    assert (tmp_path / "out" / "b5_scores_by_label.png").stat().st_size > 0
