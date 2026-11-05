"""Tests for baseline/confidence.py. Expected values are worked out by hand."""

import math

import numpy as np
import pytest

from baseline.confidence import (
    answer_logit_rows, group_records, next_token_stats, number_stats, save_npz,
)


def test_answer_logit_rows_align_with_answer_tokens():
    # 10 tokens: positions 0-6 are the prompt, 7-9 the answer.
    n_total, n_prompt = 10, 7
    positions = np.arange(n_total)        # stand-in for one logit row per position
    # Model returned every position.
    rows = positions[answer_logit_rows(n_total, n_total, n_prompt)]
    assert list(rows) == [6, 7, 8]        # these predict tokens 7, 8, 9: the answer
    # Model returned only the last n_answer + 1 positions (logits_to_keep).
    kept = positions[-4:]
    assert list(kept[answer_logit_rows(len(kept), n_total, n_prompt)]) == [6, 7, 8]
    with pytest.raises(ValueError):
        answer_logit_rows(5, n_total, n_prompt)


def test_number_stats_by_hand():
    logprob = np.array([-0.1, -2.0, -0.5, -1.0])
    entropy = np.array([0.2, 1.0, 0.4, 0.6])
    stats = number_stats(logprob, entropy, [(1, 3), (3, 4)])
    # tokens 1-2: mean (-2.0 - 0.5) / 2 = -1.25, min -2.0, entropy (1.0 + 0.4) / 2 = 0.7
    assert stats[0] == pytest.approx((-1.25, -2.0, 0.7))
    assert stats[1] == pytest.approx((-1.0, -1.0, 0.6))
    with pytest.raises(ValueError):
        number_stats(logprob, entropy, [(2, 5)])   # past the end of the answer


def test_group_records_keeps_order_and_rows():
    records = [{"id": "a"}, {"id": "a"}, {"id": "b"}, {"id": "a"}]
    groups = group_records(records)
    assert list(groups) == ["a", "b"]
    assert [row for row, _ in groups["a"]] == [0, 1, 3]
    assert [row for row, _ in groups["b"]] == [2]


def test_next_token_stats_matches_numpy():
    torch = pytest.importorskip("torch")   # runs on Colab
    # By hand: two equal logits -> each has probability 1/2.
    lp, ent = next_token_stats(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))
    assert lp.item() == pytest.approx(-math.log(2), abs=1e-6)
    assert ent.item() == pytest.approx(math.log(2), abs=1e-6)
    # Against a numpy reference on random logits.
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 50)).astype(np.float32)
    targets = rng.integers(0, 50, size=5)
    ref_logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    lp, ent = next_token_stats(torch.tensor(logits), torch.tensor(targets))
    np.testing.assert_allclose(lp.numpy(), ref_logp[np.arange(5), targets], rtol=1e-5)
    np.testing.assert_allclose(ent.numpy(), -(np.exp(ref_logp) * ref_logp).sum(-1), rtol=1e-5)


def test_save_npz_roundtrip(tmp_path):
    rows = [{"mean_logprob": -0.5, "min_logprob": -1.0, "mean_entropy": 0.3,
             "label": "copied", "id": "A/2009/page_1.pdf-1", "row": 0},
            {"mean_logprob": -2.0, "min_logprob": -3.0, "mean_entropy": 1.5,
             "label": "fabricated", "id": "A/2009/page_1.pdf-1", "row": 1}]
    save_npz(tmp_path / "c.npz", rows)
    data = np.load(tmp_path / "c.npz")   # no pickle needed
    assert list(data["label"]) == ["copied", "fabricated"]
    assert list(data["row"]) == [0, 1]
    assert data["mean_logprob"] == pytest.approx([-0.5, -2.0])
    assert set(data.files) == {"mean_logprob", "min_logprob", "mean_entropy", "label", "id", "row"}
