"""B0: plot one number token's attention over the report (README 8, step B0).

Run on a GPU (Colab) from the repo root, after A1 has built data/finqa_subset.jsonl:
    python analysis/b0_attention_plot.py
    python analysis/b0_attention_plot.py --id "ADI/2009/page_49.pdf-1" --number 2

Steps:
  1. Generate a greedy answer for one question (A3's answers don't exist yet).
  2. Tokenize prompt + answer as one string and run one teacher-forced forward
     pass with output_attentions=True (the same way B1 will).
  3. Pick one number in the answer and keep the attention row of its first
     digit token, for every layer and head.

Saves to analysis/figures/:
  b0_attention.png  that row's attention over the report tokens for a few heads,
                    with the report positions holding the same value marked
  b0_lookback.png   that token's lookback ratio for every layer and head
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `common` imports when run as a script

from common.numbers import extract_numbers, find_match, same_value  # noqa: E402
from common.prompt import encode, encode_answer, tokens_in_span  # noqa: E402
from features.attention import attention_row, lookback_ratio  # noqa: E402

# Colors (light chart surface, one blue for the attention line, orange for marks).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# Pure functions (tested in tests/test_b0.py) -------------------------------

def answer_numbers(answer, offsets, answer_char_start):
    """Numbers in the answer, each with the token indices of its digits."""
    out = []
    for n in extract_numbers(answer):
        toks = tokens_in_span(offsets, answer_char_start + n.num_start, answer_char_start + n.num_end)
        if toks:
            out.append((n, toks))
    return out


def choose_number(numbers, report_numbers, n_prompt, tol, index=None):
    """The (Number, tokens) to plot.

    `index` picks one directly. Otherwise: the first number that also appears
    in the report. Either way, the number's first digit token needs at least
    one answer token before it (README 6.4).
    """
    usable = [i for i, (_, toks) in enumerate(numbers) if toks[0] > n_prompt]
    if index is not None:
        if index not in usable:
            raise ValueError(f"number {index} has no answer token before it, or does not exist")
        return numbers[index]
    if not usable:
        raise ValueError("no number in the answer has an answer token before it")
    in_report = [i for i in usable if find_match(numbers[i][0], report_numbers, tol) is not None]
    return numbers[(in_report or usable)[0]]


def top_heads(ctx_att, k=4):
    """The k (layer, head) pairs with the most total attention on the report."""
    mass = ctx_att.sum(-1)
    flat = np.argsort(mass, axis=None)[::-1][:k]
    return [tuple(int(x) for x in np.unravel_index(i, mass.shape)) for i in flat]


def parse_heads(text):
    """'10:3,20:5' -> [(10, 3), (20, 5)]."""
    return [tuple(int(x) for x in pair.split(":")) for pair in text.split(",")]


# Plots ---------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_attention(path, ctx_att, heads, lookback, marks, title, snippet):
    """One panel per head: attention weight over report token positions.

    ctx_att  [layers, heads, n_ctx]  attention from the chosen token to the report
    marks    report token positions (0-based within the report) holding the same value
    snippet  function(position) -> short report text around that token
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(heads), 1, figsize=(10, 1.9 * len(heads) + 0.8),
                             sharex=True, facecolor=SURFACE, squeeze=False)
    x = np.arange(ctx_att.shape[-1])
    for ax, (layer, head) in zip(axes[:, 0], heads):
        _style(ax)
        y = ctx_att[layer, head]
        for j, m in enumerate(marks):
            ax.axvline(m, color=ORANGE, linewidth=0.8, alpha=0.7, zorder=1,
                       label="same value in the report" if j == 0 else None)
        ax.plot(x, y, color=BLUE, linewidth=1.5, zorder=2)
        peak = int(y.argmax())
        ax.annotate(snippet(peak), (peak, y[peak]), xytext=(6, -2), textcoords="offset points",
                    fontsize=8, color=INK_2, va="top", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.9))
        ax.set_ylim(0, max(y.max() * 1.15, 1e-6))
        ax.set_title(f"Layer {layer}, head {head}   lookback {lookback[layer, head]:.2f}",
                     loc="left", fontsize=9, color=INK)
        ax.set_ylabel("attention", fontsize=8, color=INK_2)
    if marks:
        axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_2)
    axes[-1, 0].set_xlabel("report token position", fontsize=8, color=INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_lookback(path, lookback, title):
    """Heatmap of the lookback ratio, layers x heads, on a fixed 0-1 scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    n_layers, n_heads = lookback.shape
    fig, ax = plt.subplots(figsize=(0.35 * n_heads + 2.2, 0.22 * n_layers + 1.2), facecolor=SURFACE)
    mesh = ax.pcolormesh(lookback, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=1)
    ax.invert_yaxis()  # layer 0 at the top
    ax.set_xticks(np.arange(n_heads) + 0.5, np.arange(n_heads))
    ax.set_yticks(np.arange(0, n_layers, 2) + 0.5, np.arange(0, n_layers, 2))
    ax.set_xlabel("head", fontsize=8, color=INK_2)
    ax.set_ylabel("layer", fontsize=8, color=INK_2)
    ax.tick_params(colors=MUTED, labelsize=7, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.06, pad=0.03)
    bar.set_label("lookback ratio", fontsize=8, color=INK_2)
    bar.ax.tick_params(colors=MUTED, labelsize=7)
    bar.outline.set_visible(False)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# Model run -----------------------------------------------------------------

def load_example(subset_path, example_id):
    with open(subset_path) as f:
        rows = [json.loads(line) for line in f]
    if example_id:
        return next(r for r in rows if r["id"] == example_id)
    return next(r for r in rows if r["split"] == "val")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--subset", type=Path, default=ROOT / "data" / "finqa_subset.jsonl")
    ap.add_argument("--id", help="FinQA id (default: the first val question)")
    ap.add_argument("--number", type=int,
                    help="index of the answer number to plot, as printed (default: first one found in the report)")
    ap.add_argument("--row", choices=["number", "predicting"], default="number",
                    help="'number': the row of the number's first digit token (README B0). "
                         "'predicting': the row of the token before it, which predicts that digit")
    ap.add_argument("--heads", help="heads to plot as layer:head, e.g. 10:3,20:5 "
                                    "(default: the 4 with the most attention on the report)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "figures")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ex = load_example(args.subset, args.id)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager",  # needed for output_attentions
    ).to(device).eval()
    print(f"{cfg['model']}: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} heads, on {device}")

    enc = encode(tokenizer, ex["question"], ex["context_text"], cfg.get("chat_template", True))
    n_prompt = len(enc["input_ids"])

    # 1. Greedy answer.
    prompt_ids = torch.tensor([enc["input_ids"]], device=device)
    with torch.no_grad():
        gen = model.generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                             max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(gen[0, n_prompt:], skip_special_tokens=True)
    print(f"\n{ex['id']}  ({ex['split']})\nQ: {ex['question']}\ngold: {ex['gold_answer']}"
          f"  {ex['gold_program']}\n\nanswer:\n{answer}\n")

    # 2. Teacher forcing over prompt + answer, tokenized as one string.
    try:
        full = encode_answer(tokenizer, enc["text"], answer, enc["input_ids"])
    except ValueError as err:
        sys.exit(f"{err}; pick another --id")
    ids, offsets = full["input_ids"], full["offsets"]

    # 3. Choose a number and the attention row to plot.
    tol = cfg["label_tolerance"]
    report_numbers = extract_numbers(ex["context_text"])
    numbers = answer_numbers(answer, offsets, len(enc["text"]))
    for i, (n, toks) in enumerate(numbers):
        where = "in report" if find_match(n, report_numbers, tol) is not None else "not in report"
        print(f"  [{i}] {n.text!r:<22} value {n.value:<14g} {where:<14} tokens {toks[0]}-{toks[-1]}")
    num, toks = choose_number(numbers, report_numbers, n_prompt, tol, args.number)
    row = attention_row(toks[0], args.row)
    row_text = tokenizer.decode([ids[row]])
    print(f"\nplotting the row of token {row} {row_text!r} for {num.text!r}")

    with torch.no_grad():
        out = model(torch.tensor([ids], device=device), output_attentions=True)
    att = torch.stack([a[0, :, row, :] for a in out.attentions]).float().cpu().numpy()
    del out  # all layers' full attention maps; free GPU memory
    print(f"attention rows sum to {att.sum(-1).min():.3f}-{att.sum(-1).max():.3f} (should be ~1)")

    ctx = slice(enc["ctx_start"], enc["ctx_end"])
    lookback = lookback_ratio(att, ctx, slice(n_prompt, toks[0]))  # answer tokens before the number
    print(f"lookback ratio over all heads: min {lookback.min():.2f}, "
          f"mean {lookback.mean():.2f}, max {lookback.max():.2f}")

    c0 = enc["ctx_chars"][0]
    marks = sorted({t - enc["ctx_start"]
                    for r in report_numbers if same_value(num, r, tol)
                    for t in tokens_in_span(offsets, c0 + r.num_start, c0 + r.num_end)})

    def snippet(pos):
        p = enc["ctx_start"] + pos
        return "…" + tokenizer.decode(ids[max(enc["ctx_start"], p - 4):p + 5]).replace("\n", " / ") + "…"

    heads = parse_heads(args.heads) if args.heads else top_heads(att[:, :, ctx])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    which = "first digit" if args.row == "number" else "token before the first digit"
    plot_attention(args.out_dir / "b0_attention.png", att[:, :, ctx], heads, lookback, marks,
                   f"{ex['id']}: attention from {row_text!r} ({which} of {num.text!r} in the answer) "
                   f"over the report", snippet)
    plot_lookback(args.out_dir / "b0_lookback.png", lookback,
                  f"Lookback ratio of {row_text!r} in {num.text!r}, every layer and head")
    print(f"saved {args.out_dir / 'b0_attention.png'} and {args.out_dir / 'b0_lookback.png'}")


if __name__ == "__main__":
    main()
