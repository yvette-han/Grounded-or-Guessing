"""Tests for features/attention.py and features/extract.py. Values are worked out by hand."""

import numpy as np
import pytest

from common.prompt import build_context, encode, encode_answer
from features.attention import attention_row, hf_energy, lookback_ratio, number_hf, number_lookback
from features.extract import answer_features, number_positions, rows_needed, save_npz
from labels.label import label_answer

CTX = slice(2, 6)       # report = positions 2-5
ANSWER_START = 7        # answer = positions 7 onward


def test_attention_row():
    assert attention_row(8, "number") == 8
    assert attention_row(8, "predicting") == 7
    with pytest.raises(ValueError):
        attention_row(8, "previous")


def test_lookback_ratio_by_hand():
    att = np.zeros((1, 2, 10))
    att[0, 0, :] = 0.1          # head 0: uniform -> report mean = answer mean -> 0.5
    att[0, 1, 2:6] = 0.25       # head 1: everything on the report -> 1.0
    lb = lookback_ratio(att, ctx=CTX, gen=slice(7, 10))
    assert lb[0, 0] == pytest.approx(0.5)
    assert lb[0, 1] == pytest.approx(1.0)


def _two_token_rows():
    """Rows for a number at positions 8 and 9 (one layer, one head)."""
    rows = np.zeros((1, 1, 2, 10))
    rows[0, 0, 0, 2:6] = 0.1                 # token 8: report mean 0.1
    rows[0, 0, 0, 7] = 0.3                   #          answer tokens before it: {7}, mean 0.3
    rows[0, 0, 1, 2:6] = 0.15                # token 9: report mean 0.15
    rows[0, 0, 1, 7:9] = [0.05, 0.25]        #          answer tokens before it: {7, 8}, mean 0.15
    return rows


def test_number_lookback_averages_tokens():
    # token 8: 0.1 / (0.1 + 0.3) = 0.25; token 9: 0.15 / (0.15 + 0.15) = 0.5; mean 0.375
    lb = number_lookback(_two_token_rows(), [8, 9], CTX, ANSWER_START)
    assert lb.shape == (1, 1)
    assert lb[0, 0] == pytest.approx(0.375)


def test_number_lookback_skips_first_answer_token():
    rows = _two_token_rows()
    # A number at positions 7, 8: token 7 has no answer token before it and is skipped.
    # Token 8 uses row 1 here, whose answer part is positions 7..7: mean 0.05.
    lb = number_lookback(rows, [7, 8], CTX, ANSWER_START)
    assert lb[0, 0] == pytest.approx(0.15 / (0.15 + 0.05))
    assert np.isnan(number_lookback(rows[:, :, :1], [7], CTX, ANSWER_START)).all()


def test_number_positions_and_rows_needed():
    assert number_positions(100, 3, 5) == [103, 104]
    nums = [[8, 9], [10]]
    assert rows_needed(nums, "number") == ([8, 9, 10], [[0, 1], [2]])
    # predicting rows are one earlier; row 9 serves both numbers' needs once
    assert rows_needed(nums, "predicting") == ([7, 8, 9], [[0, 1], [2]])


def test_hf_energy_by_hand():
    cutoffs = [0.0, 0.25, 0.45]
    # Flat attention is all "DC": nothing above any cutoff above 0.
    flat = hf_energy(np.full(4, 0.25), cutoffs)
    assert flat[1:] == pytest.approx([0, 0], abs=1e-12)
    # Alternating 0.1, 0.3: normalized 0.125, 0.375 = mean 0.25 plus +-0.125 at frequency 0.5.
    # Above 0.25 and 0.45 only the +-0.125 part is left: L2 = sqrt(4 * 0.125^2) = 0.25.
    alt = hf_energy(np.array([0.1, 0.3, 0.1, 0.3]), cutoffs)
    assert alt[1:] == pytest.approx([0.25, 0.25])
    assert hf_energy(np.array([0.2, 0.6, 0.2, 0.6]), cutoffs) == pytest.approx(alt)  # scale doesn't matter
    # A spike [1, 0, 0, 0]: cutoff 0 keeps everything (L2 = 1); cutoff 0.25 removes only the
    # mean -> [0.75, -0.25, -0.25, -0.25], L2 = sqrt(0.75); cutoff 0.45 keeps only frequency 0.5
    # -> 0.25 * [1, -1, 1, -1], L2 = 0.5.
    spike = hf_energy(np.array([1.0, 0, 0, 0]), cutoffs)
    assert spike == pytest.approx([1.0, np.sqrt(0.75), 0.5])
    # Shapes: [..., n] -> [cutoffs, ...]
    assert hf_energy(np.ones((2, 3, 4)), cutoffs).shape == (3, 2, 3)


def _hf_rows():
    """Rows for a number at positions 8 and 9 (one layer, one head)."""
    rows = np.zeros((1, 1, 2, 10))
    rows[0, 0, 0, 2:6] = [0.1, 0.3, 0.1, 0.3]   # token 8: report hf (cutoff 0.25) = 0.25
    rows[0, 0, 0, 7] = 0.3                      #          answer part {7}: one value -> 0
    rows[0, 0, 1, 2:6] = 0.15                   # token 9: flat report -> 0
    rows[0, 0, 1, 7:9] = [0.05, 0.25]           #          answer part {7, 8}: [1/6, 5/6] -> +-1/3
    return rows


def test_number_hf_by_hand():
    hf_ctx, hf_gen = number_hf(_hf_rows(), [8, 9], CTX, ANSWER_START, [0.25])
    assert hf_ctx.shape == hf_gen.shape == (1, 1, 1)
    assert hf_ctx[0, 0, 0] == pytest.approx((0.25 + 0) / 2)
    assert hf_gen[0, 0, 0] == pytest.approx((0 + np.sqrt(2 / 9)) / 2)
    # A number starting at the first answer token: hf_ctx uses both tokens, hf_gen only the second.
    hf_ctx, hf_gen = number_hf(_hf_rows(), [7, 8], CTX, ANSWER_START, [0.25])
    assert hf_ctx[0, 0, 0] == pytest.approx(0.125)
    assert hf_gen[0, 0, 0] == pytest.approx(0.0)        # row 1's answer part is {7}: one value
    _, hf_gen = number_hf(_hf_rows()[:, :, :1], [7], CTX, ANSWER_START, [0.25])
    assert np.isnan(hf_gen).all()


def test_answer_features():
    rows = np.concatenate([_two_token_rows(), _two_token_rows()[:, :, :1]], axis=2)  # 3 rows
    feats = answer_features(rows, [[0, 1], [2]], [[8, 9], [8]], CTX, ANSWER_START, [0.25, 0.45])
    assert feats[0]["lookback"][0, 0] == pytest.approx(0.375)
    assert feats[1]["lookback"][0, 0] == pytest.approx(0.25)
    assert feats[0]["hf_ctx"].shape == feats[0]["hf_gen"].shape == (2, 1, 1)


def test_positions_from_labels_point_at_the_digits(toy_tokenizer):
    context = build_context(["revenue grew ."], [["revenue", "4,215"]], [])
    question = {"question": "what was revenue ?", "context_text": context,
                "gold_answer": 4215.0, "gold_steps": [4215.0]}
    enc = encode(toy_tokenizer, question["question"], context)
    answer = "Revenue was $4,215 million, up 7.5%."
    row = {"id": "X/2009/page_1.pdf-1", "setting": "greedy", "prompt": enc["text"],
           "prompt_len": len(enc["input_ids"]), "answer_text": answer}
    records = label_answer(row, question, toy_tokenizer, 0.01)
    full = encode_answer(toy_tokenizer, row["prompt"], answer)
    for r, digits in zip(records, ["4,215", "7.5"]):
        pos = number_positions(full["answer_start"], r["tok_start"], r["tok_end"])
        assert digits in toy_tokenizer.decode([full["input_ids"][p] for p in pos])


def test_save_npz_roundtrip(tmp_path):
    hf = np.full((2, 3, 2), 0.1)   # 2 cutoffs, 3 layers, 2 heads
    rows = [{"id": "q1", "row": 0, "label": "copied", "lookback": np.full((3, 2), 0.5),
             "hf_ctx": hf, "hf_gen": hf},
            {"id": "q1", "row": 1, "label": "fabricated", "lookback": np.full((3, 2), np.nan),
             "hf_ctx": hf, "hf_gen": hf * np.nan}]
    save_npz(tmp_path / "f.npz", rows, "number", 3, 2, [0.25, 0.45])
    data = np.load(tmp_path / "f.npz")
    assert data["lookback"].shape == (2, 3, 2)
    assert data["hf_ctx"].shape == data["hf_gen"].shape == (2, 2, 3, 2)
    assert data["hf_ctx"].dtype == np.float16
    assert list(data["hf_cutoffs"]) == pytest.approx([0.25, 0.45])
    assert list(data["label"]) == ["copied", "fabricated"]
    assert list(data["row"]) == [0, 1]
    assert str(data["row_mode"]) == "number" and int(data["n_layers"]) == 3
