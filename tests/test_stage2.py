"""Tests for analysis/stage2.py. Expected values are worked out by hand."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from analysis.stage2 import (  # noqa: E402
    bridge_table, flag_share, head_importance, head_matrix, load_lookback, pair_auroc, rank_heads,
    rq2_table, rq3,
)
from detector.train_eval import DETECTORS, pool  # noqa: E402


def test_pair_auroc_by_hand():
    labels = np.array(["fabricated", "copied", "computed", "fabricated", "copied"])
    scores = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
    groups = np.arange(5)
    auc, lo, hi, n = pair_auroc(labels, scores, groups, "copied", n_boot=50)
    assert (auc, n) == (1.0, 4)                 # both fabricated above both copied
    auc, _, _, n = pair_auroc(labels, scores, groups, "computed", n_boot=50)
    assert (auc, n) == (0.5, 3)                 # 0.9 > 0.8 but 0.7 < 0.8
    assert pair_auroc(labels, scores, groups, "question")[0] is None   # no question numbers


def test_flag_share_by_hand():
    labels = np.array(["computed", "computed", "computed", "copied"])
    flagged = np.array([1, 0, 0, 1])
    share, lo, hi, n = flag_share(labels, flagged, np.array(["a", "a", "b", "b"]), "computed", n_boot=50)
    assert share == pytest.approx(1 / 3) and n == 3
    assert 0 <= lo <= share <= hi <= 1
    assert flag_share(labels, flagged, np.arange(4), "wrong_calc")[0] is None


def test_rq2_table_has_every_detector():
    rng = np.random.default_rng(0)
    labels = np.array(["fabricated", "copied", "computed", "wrong_calc"] * 10)
    ts = {"label": labels, "report": np.repeat(np.arange(10), 4)}
    for det in DETECTORS:
        ts[det] = rng.random(len(labels))
        ts[f"{det}_flagged"] = (ts[det] > 0.5).astype(int)
    rows = rq2_table(ts, n_boot=20)
    assert [r["detector"] for r in rows] == DETECTORS
    assert rows[0]["n_fab_vs_copied"] == 20 and rows[0]["n_wrong_calc"] == 10


def _small(n=5, layers=2, heads=3, cutoffs=2):
    rng = np.random.default_rng(1)
    return {"lookback": rng.random((n, layers, heads)), "hf_ctx": rng.random((n, cutoffs, layers, heads)),
            "hf_gen": rng.random((n, cutoffs, layers, heads))}


def test_head_matrix_picks_the_right_columns():
    d = _small()
    X = head_matrix(d, "lookback", [(1, 2), (0, 0)])
    np.testing.assert_array_equal(X, np.stack([d["lookback"][:, 1, 2], d["lookback"][:, 0, 0]], 1))
    X = head_matrix(d, "lookback_hf", [(1, 2), (0, 0)], cutoff_index=1)
    assert X.shape == (5, 6)
    np.testing.assert_array_equal(X[:, 2], d["hf_ctx"][:, 1, 1, 2])   # second block, first head
    np.testing.assert_array_equal(X[:, 5], d["hf_gen"][:, 1, 0, 0])   # third block, second head


def test_head_importance_and_rank():
    imp = head_importance(np.arange(6) - 3.0, "lookback", 2, 3)       # |[-3,-2,-1,0,1,2]|
    np.testing.assert_array_equal(imp, [[3, 2, 1], [0, 1, 2]])
    coef = np.concatenate([np.ones(6), -2 * np.ones(6), np.zeros(6)])  # three blocks
    np.testing.assert_array_equal(head_importance(coef, "lookback_hf", 2, 3), np.full((2, 3), 3.0))
    with pytest.raises(ValueError):
        head_importance(np.ones(5), "lookback", 2, 3)
    assert rank_heads(np.array([[0.1, 0.5, 0.2], [0.9, 0.0, 0.3]])) == [
        (1, 0), (0, 1), (1, 2), (0, 2), (0, 0), (1, 1)]


def _synthetic(n_reports=60, per_report=12, layers=3, heads=4, seed=2):
    """Only heads (1, 0) and (1, 2) carry signal."""
    rng = np.random.default_rng(seed)
    n = n_reports * per_report
    labels = rng.choice(["fabricated", "copied", "computed", "wrong_calc"], size=n, p=[0.3, 0.5, 0.15, 0.05])
    fab = labels == "fabricated"
    lb = rng.normal(size=(n, layers, heads))
    lb[:, 1, 0] -= 2 * fab
    lb[:, 1, 2] -= 2 * fab
    reports = np.repeat([f"R{i}/2009" for i in range(n_reports)], per_report)
    ids = np.array([f"{r}/page_1.pdf-{k}" for k, r in enumerate(reports)])
    d = {"lookback": lb, "hf_ctx": rng.normal(size=(n, 2, layers, heads)),
         "hf_gen": rng.normal(size=(n, 2, layers, heads)), "confidence": rng.normal(size=(n, 3)),
         "label": labels, "id": ids, "row": np.arange(n), "hf_cutoffs": np.array([0.25, 0.45])}
    split = {r: ("train" if i < 42 else "val" if i < 51 else "test") for i, r in enumerate(dict.fromkeys(reports))}
    return pool({"greedy": d}, {i: split[r] for i, r in zip(ids, reports)}, dict(zip(ids, reports)))


def test_rq3_finds_the_signal_heads():
    data = _synthetic()
    layer_rows, head_rows, topk_rows, imp = rq3(data, "lookback", C=1.0, cutoff_index=None,
                                                top_k=[2, 5], n_boot=50)
    aucs = [r["auroc"] for r in layer_rows]
    assert int(np.argmax(aucs)) == 1 and aucs[1] > 0.9
    assert {(r["layer"], r["head"]) for r in head_rows[:2]} == {(1, 0), (1, 2)}
    by_k = {r["heads"]: r["auroc"] for r in topk_rows}
    assert set(by_k) == {2, 5, 12}               # 12 = all heads (the full detector)
    assert by_k[2] > 0.9 and abs(by_k[2] - by_k[12]) < 0.05
    assert imp.shape == (3, 4)


def test_bridge_table_by_hand():
    lb = {"setting": np.array(["greedy"] * 3 + ["temp12"] * 2),
          "label": np.array(["copied", "copied", "fabricated", "copied", "fabricated"]),
          "report": np.array(["a", "b", "a", "a", "b"]),
          "mean_lookback": np.array([0.8, 0.6, 0.4, 0.5, 0.1])}
    rows = {(r["setting"], r["label"]): r for r in bridge_table(lb, ["greedy", "temp12"], n_boot=50)}
    assert rows[("greedy", "all")]["mean_lookback"] == pytest.approx(0.6)       # (0.8 + 0.6 + 0.4) / 3
    assert rows[("greedy", "copied")]["mean_lookback"] == pytest.approx(0.7)
    assert rows[("temp12", "fabricated")]["mean_lookback"] == pytest.approx(0.1)
    assert ("greedy", "wrong_calc") not in rows                              # no such numbers


def test_load_lookback_skips_numbers_without_lookback(tmp_path):
    lookback = np.array([[[0.2, 0.4]], [[np.nan, np.nan]], [[0.6, np.nan]]])   # [3 numbers, 1 layer, 2 heads]
    np.savez(tmp_path / "greedy.npz", lookback=lookback, label=np.array(["copied", "copied", "fabricated"]),
             id=np.array(["q1", "q1", "q2"]))
    lb = load_lookback(tmp_path, ["greedy"], {"q1": "A/2009", "q2": "B/2010"})
    assert lb["mean_lookback"] == pytest.approx([0.3, 0.6])                  # NaN heads ignored
    assert list(lb["report"]) == ["A/2009", "B/2010"]
