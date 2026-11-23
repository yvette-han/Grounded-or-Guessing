"""Tests for detector/three_way.py. Expected values are worked out by hand."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from detector.three_way import (  # noqa: E402
    CLASSES, class_index, confusion, macro_f1, per_class_f1, recall_per_class, run,
)
from detector.train_eval import DETECTORS, pool  # noqa: E402


def test_class_index():
    labels = ["copied", "computed", "fabricated", "wrong_calc", "question"]
    assert list(class_index(labels)) == [0, 1, 2, 2, -1]
    assert CLASSES == ["copied", "computed", "hallucinated"]


def test_f1_and_confusion_by_hand():
    y = np.array([0, 0, 0, 1, 2, 2])
    pred = np.array([0, 0, 2, 1, 2, 0])
    # copied: TP 2, FP 1 (the last), FN 1 -> 4/6; computed: 2/2 = 1; hallucinated: TP 1, FP 1, FN 1 -> 2/4
    np.testing.assert_allclose(per_class_f1(y, pred), [2 / 3, 1.0, 0.5])
    assert macro_f1(y, pred) == pytest.approx((2 / 3 + 1 + 0.5) / 3)
    m = confusion(y, pred)
    np.testing.assert_array_equal(m, [[2, 0, 1], [0, 1, 0], [1, 0, 1]])
    np.testing.assert_allclose(recall_per_class(m), [2 / 3, 1.0, 0.5])
    # a class missing from y doesn't count in the macro average (computed here)
    assert macro_f1(np.array([0, 2]), np.array([0, 2])) == 1.0


def _synthetic(n_reports=60, per_report=15, seed=4):
    """Lookback head (0, 0) is high for copied, (0, 1) high for computed; hallucinated is low on both."""
    rng = np.random.default_rng(seed)
    n = n_reports * per_report
    labels = rng.choice(["copied", "computed", "fabricated", "wrong_calc", "question"], size=n,
                        p=[0.45, 0.2, 0.2, 0.1, 0.05])
    lb = rng.normal(size=(n, 2, 3))
    lb[:, 0, 0] += 4 * (labels == "copied")
    lb[:, 0, 1] += 4 * (labels == "computed")
    reports = np.repeat([f"R{i}/2009" for i in range(n_reports)], per_report)
    ids = np.array([f"{r}/page_1.pdf-{k}" for k, r in enumerate(reports)])
    d = {"lookback": lb, "hf_ctx": rng.normal(size=(n, 2, 2, 3)), "hf_gen": rng.normal(size=(n, 2, 2, 3)),
         "confidence": rng.normal(size=(n, 3)), "label": labels, "id": ids, "row": np.arange(n),
         "hf_cutoffs": np.array([0.25, 0.45])}
    split = {r: ("train" if i < 42 else "val" if i < 51 else "test") for i, r in enumerate(dict.fromkeys(reports))}
    return pool({"greedy": d}, {i: split[r] for i, r in zip(ids, reports)}, dict(zip(ids, reports)))


def test_run_on_synthetic_data():
    data = _synthetic()
    results, conf_rows, columns, test = run(data, n_boot=50, seed=0)
    by = {r["detector"]: r for r in results}
    assert [r["detector"] for r in results] == DETECTORS
    assert by["lookback"]["macro_f1"] > 0.85 and by["lookback_hf"]["macro_f1"] > 0.8
    assert by["confidence"]["macro_f1"] < 0.5 and by["random"]["macro_f1"] < 0.5
    for r in results:
        assert r["macro_f1_ci_low"] <= r["macro_f1"] <= r["macro_f1_ci_high"]
    # question numbers are left out; the confusion counts add up to the test numbers
    assert "question" not in set(data["label"][test])
    assert sum(r["n"] for r in conf_rows if r["detector"] == "lookback") == by["lookback"]["n_test"] == test.sum()
    assert set(columns["lookback"]) <= set(CLASSES) and len(columns["lookback"]) == test.sum()
    n_hal = int(np.isin(data["label"][test], ["fabricated", "wrong_calc"]).sum())
    assert by["lookback"]["n_test_hallucinated"] == n_hal
