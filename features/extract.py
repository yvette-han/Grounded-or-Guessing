"""B1-B2: attention features for every labeled number (README 8, steps B1 and B2).

Run on a GPU (Colab) from the repo root, after A4:
    python features/extract.py --limit 20       # B1 check: first 20 answers of each setting
    python features/extract.py                  # every setting that has labels
    python features/extract.py --setting greedy --row predicting

For each answer: one teacher-forced forward pass over prompt + answer with
output_attentions=True (eager attention, float16, batch size 1). Right away only
the attention rows of the number tokens are kept (all layers and heads), and
the rest is dropped. Per number, averaged over its digit tokens
(tok_start..tok_end from A4), per layer and head:
  lookback  attention on the report / (on the report + on earlier answer tokens)
  hf_ctx    high-frequency energy of the attention over the report, per cutoff
  hf_gen    the same over the earlier answer tokens
(definitions in features/attention.py; cutoffs from hf_cutoffs in config.yaml)

Output: features/{setting}.npz (compressed) with arrays
    lookback [numbers, layers, heads]                  float32
    hf_ctx, hf_gen [numbers, cutoffs, layers, heads]   float16, to keep the file small
    hf_cutoffs [cutoffs], label, id, row
and the scalars row_mode, n_layers, n_heads. `row` is the line in
labels/numbers_{setting}.jsonl, the same join key as baseline/confidence_{setting}.npz.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `common` and `features` import when run as a script

from common.io import group_records, read_jsonl  # noqa: E402
from common.prompt import encode_answer  # noqa: E402
from features.attention import ROW_MODES, attention_row, number_hf, number_lookback  # noqa: E402


# Pure functions (tested in tests/test_features.py) -------------------------

def number_positions(answer_start, tok_start, tok_end):
    """Absolute positions of a number's tokens, from its span within the answer."""
    return list(range(answer_start + tok_start, answer_start + tok_end))


def rows_needed(numbers_positions, mode):
    """Sorted attention rows to keep for all numbers of one answer, and for each
    number the indices of its rows in that list."""
    rows = sorted({attention_row(t, mode) for pos in numbers_positions for t in pos})
    where = {r: i for i, r in enumerate(rows)}
    return rows, [[where[attention_row(t, mode)] for t in pos] for pos in numbers_positions]


def answer_features(rows_att, index_per_number, positions_per_number, ctx, answer_start, cutoffs):
    """rows_att [layers, heads, n_rows, seq] -> one dict of features per number."""
    feats = []
    for idx, pos in zip(index_per_number, positions_per_number):
        rows = rows_att[:, :, idx]
        hf_ctx, hf_gen = number_hf(rows, pos, ctx, answer_start, cutoffs)
        feats.append({"lookback": number_lookback(rows, pos, ctx, answer_start),
                      "hf_ctx": hf_ctx, "hf_gen": hf_gen})
    return feats


def save_npz(path, rows, row_mode, n_layers, n_heads, cutoffs):
    np.savez_compressed(
        path,
        lookback=np.stack([r["lookback"] for r in rows]).astype(np.float32),
        hf_ctx=np.stack([r["hf_ctx"] for r in rows]).astype(np.float16),
        hf_gen=np.stack([r["hf_gen"] for r in rows]).astype(np.float16),
        hf_cutoffs=np.array(cutoffs, dtype=np.float32),
        label=np.array([r["label"] for r in rows], dtype=str),
        id=np.array([r["id"] for r in rows], dtype=str),
        row=np.array([r["row"] for r in rows], dtype=np.int64),
        row_mode=np.array(row_mode),
        n_layers=np.array(n_layers),
        n_heads=np.array(n_heads),
    )


# Model run -----------------------------------------------------------------

def run_setting(setting, model, tokenizer, args, device, cutoffs):
    import torch
    from tqdm import tqdm

    records = read_jsonl(args.labels_dir / f"numbers_{setting}.jsonl")
    answers = {r["id"]: r for r in read_jsonl(args.answers_dir / f"answers_{setting}.jsonl")}
    groups = list(group_records(records).items())
    if args.limit:
        groups = groups[:args.limit]

    out_rows, skipped = [], []
    row_sum_min, row_sum_max = np.inf, -np.inf
    for qid, recs in tqdm(groups, desc=setting, unit="answer"):
        a = answers[qid]
        try:
            full = encode_answer(tokenizer, a["prompt"], a["answer_text"])
        except ValueError as err:
            skipped.append((qid, str(err)))
            continue
        answer_start = full["answer_start"]
        if answer_start != a["prompt_len"]:
            skipped.append((qid, "prompt length differs from gen/"))
            continue
        positions = [number_positions(answer_start, r["tok_start"], r["tok_end"]) for _, r in recs]
        rows, index = rows_needed(positions, args.row)

        ids = torch.tensor([full["input_ids"]], device=device)
        with torch.inference_mode():
            out = model(input_ids=ids, output_attentions=True)
            # [layers, heads, n_rows, seq]: keep only the rows we need, drop the rest
            rows_att = torch.stack([layer[0][:, rows, :] for layer in out.attentions]).float().cpu().numpy()
            del out
        sums = rows_att.sum(-1)
        row_sum_min, row_sum_max = min(row_sum_min, sums.min()), max(row_sum_max, sums.max())

        ctx = slice(a["ctx_start"], a["ctx_end"])
        feats_per_number = answer_features(rows_att, index, positions, ctx, answer_start, cutoffs)
        for (row, r), feats in zip(recs, feats_per_number):
            out_rows.append({"id": qid, "row": row, "label": r["label"], **feats})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{setting}.npz"
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    if not out_rows:
        print(f"{setting}: no numbers to save")
        return
    save_npz(path, out_rows, args.row, n_layers, n_heads, cutoffs)

    lb = np.stack([r["lookback"] for r in out_rows])
    nan_numbers = int(np.isnan(lb).all(axis=(1, 2)).sum())
    print(f"{setting}: {len(out_rows)} numbers from {len(groups) - len(skipped)} answers -> {path}")
    print(f"  attention row sums: {row_sum_min:.3f} to {row_sum_max:.3f} (should be ~1)")
    print(f"  lookback: {np.nanmin(lb):.3f} to {np.nanmax(lb):.3f} (should be within 0-1); "
          f"numbers without a lookback (no answer token before them): {nan_numbers}")
    for name in ("hf_ctx", "hf_gen"):
        v = np.stack([r[name] for r in out_rows])
        print(f"  {name} (cutoffs {cutoffs}): {np.nanmin(v):.3f} to {np.nanmax(v):.3f}, "
              f"NaN values: {int(np.isnan(v).sum())}")
    if skipped:
        print(f"  skipped {len(skipped)} answers, e.g. {skipped[:3]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--setting", nargs="+", help="default: every setting with a labels file")
    ap.add_argument("--row", choices=ROW_MODES, default="number",
                    help="'number': each number token's own attention row (default). "
                         "'predicting': the row before it, which predicts that token")
    ap.add_argument("--limit", type=int, help="only the first N answers of each setting (B1 check)")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--answers-dir", type=Path, default=ROOT / "gen")
    ap.add_argument("--labels-dir", type=Path, default=ROOT / "labels")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "features")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    settings = args.setting or [s for s in cfg["decoding"]
                                if (args.labels_dir / f"numbers_{s}.jsonl").exists()]
    if not settings:
        sys.exit(f"no labels/numbers_*.jsonl found in {args.labels_dir}; run labels/label.py first")

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
          f"{model.config.num_attention_heads} heads, on {device}; row mode {args.row}")
    for s in settings:
        run_setting(s, model, tokenizer, args, device, cfg["hf_cutoffs"])


if __name__ == "__main__":
    main()
