"""Tests for detector/train_eval.py. Expected values are worked out by hand."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from detector.train_eval import (  # noqa: E402
    DETECTORS, best_f1_threshold, bootstrap_ci, f1_at, features_for, finite_mask, load_setting,
    pool, run, subset, targets,
)


def test_targets():
    labels = ["fabricated", "copied", "computed", "wrong_calc", "question"]
    assert list(targets(labels)) == [1, 0, 0, -1, -1]


def test_f1_at_by_hand():
    y, s = np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])
    # threshold 0.75 flags the first two: TP 1, FP 1, FN 1 -> 2 / (2 + 1 + 1) = 0.5
    assert f1_at(y, s, 0.75) == pytest.approx(0.5)
    assert f1_at(np.array([0, 0]), np.array([0.1, 0.2]), 0.5) == 0.0


def test_best_f1_threshold_by_hand():
    y, s = np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])
    # >=0.9: F1 2/3; >=0.8: 0.5; >=0.7: TP 2, FP 1 -> 4/5 = 0.8; >=0.1: 2/3
    assert best_f1_threshold(y, s) == pytest.approx((0.7, 0.8))
    # ties are flagged together: >=0.5 flags both 0.5s (TP 1, FP 1): 0.5; >=0.2: 0.8
    assert best_f1_threshold(np.array([1, 0, 1]), np.array([0.5, 0.5, 0.2])) == pytest.approx((0.2, 0.8))


def test_bootstrap_ci_perfect_detector():
    from sklearn.metrics import roc_auc_score
    y = np.array([0, 1] * 10)
    groups = np.repeat(np.arange(10), 2)       # each report has one of each
    assert bootstrap_ci(y, y.astype(float), groups, roc_auc_score, n_boot=50) == (1.0, 1.0)


def _data(n=4, layers=2, heads=3, cutoffs=(0.25, 0.45)):
    rng = np.random.default_rng(0)
    return {
        "lookback": rng.random((n, layers, heads)), "hf_ctx": rng.random((n, len(cutoffs), layers, heads)),
        "hf_gen": rng.random((n, len(cutoffs), layers, heads)), "confidence": rng.random((n, 3)),
        "label": np.array(["copied"] * n), "hf_cutoffs": np.array(cutoffs),
    }


def test_features_for_shapes():
    d = _data()
    assert features_for(d, "lookback").shape == (4, 6)
    assert features_for(d, "hf", 1).shape == (4, 12)
    assert features_for(d, "lookback_hf", 0).shape == (4, 18)
    assert features_for(d, "confidence").shape == (4, 3)
    np.testing.assert_array_equal(features_for(d, "hf", 1)[:, :6], d["hf_ctx"][:, 1].reshape(4, 6))


def test_finite_mask_and_subset():
    d = _data()
    d["hf_gen"][2, 0, 1, 1] = np.nan
    keep = finite_mask(d)
    assert list(keep) == [True, True, False, True]
    assert len(subset(d, keep)["label"]) == 3


def test_load_setting_joins_on_row(tmp_path):
    d = _data()
    np.savez(tmp_path / "f.npz", lookback=d["lookback"], hf_ctx=d["hf_ctx"], hf_gen=d["hf_gen"],
             hf_cutoffs=d["hf_cutoffs"], label=np.array(["copied", "fabricated", "copied", "computed"]),
             id=np.array(["a", "a", "b", "b"]), row=np.array([0, 1, 2, 3]))
    conf_rows = np.array([3, 1, 0])            # different order, row 2 missing
    np.savez(tmp_path / "c.npz", mean_logprob=np.array([-3.0, -1.0, -0.0]),
             min_logprob=np.array([-4.0, -2.0, -0.5]), mean_entropy=np.array([3.0, 1.0, 0.1]),
             label=np.array(["computed", "fabricated", "copied"]), id=np.array(["b", "a", "a"]), row=conf_rows)
    s = load_setting(tmp_path / "f.npz", tmp_path / "c.npz")
    assert list(s["row"]) == [0, 1, 3]
    assert s["confidence"][:, 0].tolist() == [-0.0, -1.0, -3.0]   # matched by row, not position
    np.testing.assert_allclose(s["lookback"], d["lookback"][[0, 1, 3]], rtol=1e-6)
    # a label mismatch for the same row is an error
    np.savez(tmp_path / "c2.npz", mean_logprob=np.zeros(3), min_logprob=np.zeros(3), mean_entropy=np.zeros(3),
             label=np.array(["copied", "fabricated", "copied"]), id=np.array(["b", "a", "a"]), row=conf_rows)
    with pytest.raises(ValueError):
        load_setting(tmp_path / "f.npz", tmp_path / "c2.npz")


def _synthetic(n_reports=60, per_report=12, seed=1):
    """Lookback (and hf at cutoff index 1) separate fabricated from the rest; everything else is noise."""
    rng = np.random.default_rng(seed)
    n = n_reports * per_report
    labels = rng.choice(["fabricated", "copied", "computed", "wrong_calc"], size=n, p=[0.3, 0.5, 0.15, 0.05])
    fab = (labels == "fabricated")[:, None, None]
    d = {
        "lookback": rng.normal(size=(n, 2, 3)) - 1.5 * fab,
        "hf_ctx": rng.normal(size=(n, 2, 2, 3)),
        "hf_gen": rng.normal(size=(n, 2, 2, 3)),
        "confidence": rng.normal(size=(n, 3)),
        "label": labels,
        "hf_cutoffs": np.array([0.25, 0.45]),
    }
    d["hf_ctx"][:, 1] += 1.5 * fab
    reports = np.repeat([f"R{i}/2009" for i in range(n_reports)], per_report)
    split_of_report = {r: ("train" if i < 42 else "val" if i < 51 else "test")
                       for i, r in enumerate(dict.fromkeys(reports))}
    return d, reports, split_of_report


def test_run_on_synthetic_data():
    d, reports, split_of_report = _synthetic()
    ids = np.array([f"{r}/page_1.pdf-{k}" for k, r in enumerate(reports)])
    d.update(id=ids, row=np.arange(len(ids)))
    data = pool({"greedy": d}, {i: split_of_report[r] for i, r in zip(ids, reports)},
                {i: r for i, r in zip(ids, reports)})
    results, columns, test_all = run(data, n_boot=100, seed=0)
    by = {r["detector"]: r for r in results}
    assert [r["detector"] for r in results] == DETECTORS
    assert by["lookback"]["auroc"] > 0.9 and by["lookback_hf"]["auroc"] > 0.9
    assert by["hf"]["hf_cutoff"] == 0.45                 # the cutoff that carries the signal
    assert by["confidence"]["auroc"] < 0.7 and 0.3 < by["random"]["auroc"] < 0.7
    for r in results:
        assert r["auroc_ci_low"] <= r["auroc"] <= r["auroc_ci_high"]
    # every test number is scored, including wrong_calc (left out of training and metrics)
    assert len(columns["lookback"]) == test_all.sum()
    assert "wrong_calc" in set(data["label"][test_all])
    assert by["lookback"]["n_test"] == int((test_all & (targets(data["label"]) >= 0)).sum())
