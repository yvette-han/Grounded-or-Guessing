"""Tests for detector/transfer.py, on synthetic data with known signal."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from detector.train_eval import DETECTORS, pool  # noqa: E402
from detector.transfer import check_compatible, transfer  # noqa: E402


def _data(n_reports, signal, split_of_index, seed, per_report=12):
    """Lookback is lower for fabricated numbers by `signal`."""
    rng = np.random.default_rng(seed)
    n = n_reports * per_report
    labels = rng.choice(["fabricated", "copied", "computed", "wrong_calc"], size=n, p=[0.3, 0.5, 0.15, 0.05])
    fab = (labels == "fabricated")[:, None, None]
    reports = np.repeat([f"R{seed}-{i}" for i in range(n_reports)], per_report)
    ids = np.array([f"{r}-{k}" for k, r in enumerate(reports)])
    d = {"lookback": rng.normal(size=(n, 2, 3)) - signal * fab, "hf_ctx": rng.normal(size=(n, 2, 2, 3)),
         "hf_gen": rng.normal(size=(n, 2, 2, 3)), "confidence": rng.normal(size=(n, 3)), "label": labels,
         "id": ids, "row": np.arange(n), "hf_cutoffs": np.array([0.25, 0.45])}
    split = {r: split_of_index(i) for i, r in enumerate(dict.fromkeys(reports))}
    return pool({"greedy": d}, {i: split[r] for i, r in zip(ids, reports)}, dict(zip(ids, reports)))


def _finqa():
    return _data(60, 1.5, lambda i: "train" if i < 42 else "val" if i < 51 else "test", seed=1)


def test_transfer_keeps_a_shared_signal():
    tatqa = _data(30, 1.5, lambda i: "test", seed=2)
    rows, columns = transfer(_finqa(), tatqa, n_boot=50)
    by = {r["detector"]: r for r in rows}
    assert [r["detector"] for r in rows] == DETECTORS
    assert by["lookback"]["finqa_test_auroc"] > 0.9 and by["lookback"]["tatqa_auroc"] > 0.9
    assert 0.3 < by["random"]["tatqa_auroc"] < 0.7
    assert by["lookback"]["tatqa_auroc_ci_low"] <= by["lookback"]["tatqa_auroc"] <= by["lookback"]["tatqa_auroc_ci_high"]
    assert len(columns["lookback"]) == len(tatqa["label"])     # every TAT-QA number is scored, all labels
    assert by["lookback"]["n_tatqa"] == int(np.isin(tatqa["label"], ["fabricated", "copied", "computed"]).sum())


def test_transfer_without_the_signal_is_near_chance():
    tatqa = _data(30, 0.0, lambda i: "test", seed=3)
    rows, _ = transfer(_finqa(), tatqa, n_boot=50)
    assert {r["detector"]: r for r in rows}["lookback"]["tatqa_auroc"] < 0.7


def test_check_compatible(tmp_path):
    np.savez(tmp_path / "a.npz", row_mode=np.array("number"), hf_cutoffs=np.array([0.25, 0.45]))
    np.savez(tmp_path / "b.npz", row_mode=np.array("predicting"), hf_cutoffs=np.array([0.25, 0.45]))
    np.savez(tmp_path / "c.npz", row_mode=np.array("number"), hf_cutoffs=np.array([0.25, 0.35]))
    check_compatible(tmp_path / "a.npz", tmp_path / "a.npz")
    with pytest.raises(ValueError):
        check_compatible(tmp_path / "a.npz", tmp_path / "b.npz")
    with pytest.raises(ValueError):
        check_compatible(tmp_path / "a.npz", tmp_path / "c.npz")
