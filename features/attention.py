"""Attention features for number tokens (README 6.4, B1, B2), shared by B0 and extract.py.

Positions are absolute token indices in prompt + answer. For a number token at
position t:
  context tokens    the report: positions ctx_start .. ctx_end - 1
  generated tokens  answer tokens before t: positions answer_start .. t - 1
  the attention row we read:
    row mode "number"      row t: the number token's own row (README B0/B1 wording)
    row mode "predicting"  row t - 1: the row that predicts token t (Lookback Lens);
                           its own token t - 1 is one of the generated tokens
A token with no answer token before it (t == answer_start) has no lookback
ratio and no hf_gen and is skipped for those (README 6.4); hf_ctx still uses it.
"""

import numpy as np

ROW_MODES = ("number", "predicting")


def attention_row(t, mode):
    """The attention row used for the number token at position t."""
    if mode not in ROW_MODES:
        raise ValueError(f"row mode must be one of {ROW_MODES}")
    return t if mode == "number" else t - 1


def lookback_ratio(att_row, ctx, gen):
    """att_row: [..., seq] attention from one row; ctx, gen: slices of positions.

    Mean attention on context tokens / (mean on context + mean on generated
    tokens), per head. README B2.
    """
    on_ctx = att_row[..., ctx].mean(-1)
    on_gen = att_row[..., gen].mean(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return on_ctx / (on_ctx + on_gen)


def number_lookback(rows, positions, ctx, answer_start):
    """Lookback ratio for one number, averaged over its tokens.

    rows       [layers, heads, n_tokens, seq]: rows[:, :, k] is the attention
               row used for the number token at positions[k]
    positions  absolute positions of the number's tokens
    Returns [layers, heads]; all NaN if no token has an answer token before it.
    """
    per_token = [lookback_ratio(rows[:, :, k], ctx, slice(answer_start, t))
                 for k, t in enumerate(positions) if t > answer_start]
    if not per_token:
        return np.full(rows.shape[:2], np.nan, dtype=np.float32)
    return np.mean(per_token, axis=0).astype(np.float32)


def hf_energy(att, cutoffs):
    """High-frequency energy of attention over a span of positions (README B2).

    att: [..., n] attention weights over n consecutive positions.
    1. divide by their sum (so span length and total mass don't matter)
    2. numpy.fft.rfft
    3. zero every coefficient whose frequency is below the cutoff; frequencies
       are numpy.fft.rfftfreq(n): cycles per token, from 0 to 0.5
    4. inverse FFT, L2 norm
    Returns [len(cutoffs), ...].
    """
    n = att.shape[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = att / att.sum(-1, keepdims=True)
    spectrum = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(n)
    energies = []
    for cutoff in cutoffs:
        high = np.where(freqs >= cutoff, spectrum, 0)
        energies.append(np.linalg.norm(np.fft.irfft(high, n=n, axis=-1), axis=-1))
    return np.stack(energies)


def number_hf(rows, positions, ctx, answer_start, cutoffs):
    """High-frequency energy for one number, averaged over its tokens.

    rows, positions as in number_lookback. Returns (hf_ctx, hf_gen), each
    [len(cutoffs), layers, heads]:
      hf_ctx  over the report; every token counts
      hf_gen  over the answer tokens before each token; a token with none is
              skipped, and all NaN if no token has any
    """
    shape = (len(cutoffs),) + rows.shape[:2]
    hf_ctx = hf_energy(rows[:, :, :, ctx], cutoffs).mean(-1)  # [C, L, H, tokens] -> [C, L, H]
    per_token = [hf_energy(rows[:, :, k, answer_start:t], cutoffs)
                 for k, t in enumerate(positions) if t > answer_start]
    hf_gen = np.mean(per_token, axis=0) if per_token else np.full(shape, np.nan)
    return hf_ctx.astype(np.float32), hf_gen.astype(np.float32)
