"""A7: confidence baseline for every labeled number (README 8, step A7).

Run on a GPU (Colab) from the repo root, after A4:
    python baseline/confidence.py                    # every setting that has labels
    python baseline/confidence.py --setting greedy   # one or more settings

For each answer, one teacher-forced forward pass over prompt + answer (no
attention needed). For each number in labels/numbers_{setting}.jsonl, over its
digit tokens (tok_start..tok_end from A4):
  mean_logprob   mean log-probability the model gave those tokens
  min_logprob    the lowest of them
  mean_entropy   mean entropy (in nats) of the next-token distribution each was drawn from
The distribution is the model's own (temperature 1, no top-k/top-p) for every
setting, so the scores are comparable across settings.

Output: baseline/confidence_{setting}.npz with arrays
    mean_logprob, min_logprob, mean_entropy, label, id, row
`row` is the line index in labels/numbers_{setting}.jsonl, to join with features/.
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `common` imports when run as a script

from common.prompt import encode_answer  # noqa: E402


# Pure functions (tested in tests/test_confidence.py) -----------------------

def answer_logit_rows(n_rows, n_total, n_prompt):
    """Slice of the returned logits whose row k predicts answer token k.

    Logits at position p predict token p + 1, so answer token k (at position
    n_prompt + k) is predicted by position n_prompt + k - 1.
    n_rows: how many logit rows the model returned, either all n_total
    positions or only the last (n_total - n_prompt + 1) of them.
    """
    n_answer = n_total - n_prompt
    if n_rows == n_total:
        return slice(n_prompt - 1, n_total - 1)
    if n_rows == n_answer + 1:
        return slice(0, n_answer)
    raise ValueError(f"unexpected number of logit rows: {n_rows}")


def number_stats(tok_logprob, tok_entropy, spans):
    """Per number: (mean log-prob, min log-prob, mean entropy) over its tokens.

    tok_logprob[k], tok_entropy[k]: for answer token k.
    spans: (tok_start, tok_end) half-open, counted from the first answer token.
    """
    n = len(tok_logprob)
    out = []
    for start, end in spans:
        if not 0 <= start < end <= n:
            raise ValueError(f"span {(start, end)} outside the answer's {n} tokens")
        lp, ent = tok_logprob[start:end], tok_entropy[start:end]
        out.append((float(np.mean(lp)), float(np.min(lp)), float(np.mean(ent))))
    return out


def group_records(records):
    """id -> list of (row index, record), keeping file order."""
    groups = OrderedDict()
    for row, r in enumerate(records):
        groups.setdefault(r["id"], []).append((row, r))
    return groups


def next_token_stats(logits, targets):
    """torch: logits [n, vocab] (row k predicts targets[k]) -> (log-prob of each
    target, entropy of each row's distribution), both float32 [n]."""
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logprob = logp.gather(1, targets[:, None])[:, 0]
    entropy = -(logp.exp() * logp).nansum(dim=-1)
    return tok_logprob, entropy


def save_npz(path, rows):
    """rows: dicts with mean_logprob, min_logprob, mean_entropy, label, id, row."""
    np.savez(
        path,
        mean_logprob=np.array([r["mean_logprob"] for r in rows], dtype=np.float32),
        min_logprob=np.array([r["min_logprob"] for r in rows], dtype=np.float32),
        mean_entropy=np.array([r["mean_entropy"] for r in rows], dtype=np.float32),
        label=np.array([r["label"] for r in rows], dtype=str),
        id=np.array([r["id"] for r in rows], dtype=str),
        row=np.array([r["row"] for r in rows], dtype=np.int64),
    )


# Model run -----------------------------------------------------------------

def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_setting(setting, model, tokenizer, answers_dir, labels_dir, out_dir, device):
    import torch
    from tqdm import tqdm

    records = _read_jsonl(labels_dir / f"numbers_{setting}.jsonl")
    answers = {r["id"]: r for r in _read_jsonl(answers_dir / f"answers_{setting}.jsonl")}
    rows, skipped = [], []
    for qid, recs in tqdm(group_records(records).items(), desc=setting, unit="answer"):
        a = answers[qid]
        try:
            full = encode_answer(tokenizer, a["prompt"], a["answer_text"])
        except ValueError as err:
            skipped.append((qid, str(err)))
            continue
        n_prompt, n_total = full["answer_start"], len(full["input_ids"])
        if n_prompt != a["prompt_len"]:
            skipped.append((qid, "prompt length differs from gen/"))
            continue

        ids = torch.tensor([full["input_ids"]], device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids, logits_to_keep=n_total - n_prompt + 1).logits[0]
            logits = logits[answer_logit_rows(logits.shape[0], n_total, n_prompt)]
            tok_logprob, entropy = next_token_stats(logits, ids[0, n_prompt:])
        stats = number_stats(tok_logprob.cpu().numpy(), entropy.cpu().numpy(),
                             [(r["tok_start"], r["tok_end"]) for _, r in recs])
        for (row, r), (mean_lp, min_lp, mean_ent) in zip(recs, stats):
            rows.append({"mean_logprob": mean_lp, "min_logprob": min_lp, "mean_entropy": mean_ent,
                         "label": r["label"], "id": qid, "row": row})

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"confidence_{setting}.npz"
    save_npz(path, rows)
    lp = np.array([r["mean_logprob"] for r in rows])
    print(f"{setting}: {len(rows)} of {len(records)} numbers -> {path}")
    if rows:
        print(f"  mean_logprob range {lp.min():.2f} to {lp.max():.2f}, NaNs: {int(np.isnan(lp).sum())}")
    if skipped:
        print(f"  skipped {len(skipped)} answers, e.g. {skipped[:3]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--setting", nargs="+", help="default: every setting with a labels file")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--answers-dir", type=Path, default=ROOT / "gen")
    ap.add_argument("--labels-dir", type=Path, default=ROOT / "labels")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "baseline")
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
        cfg["model"], dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    for s in settings:
        run_setting(s, model, tokenizer, args.answers_dir, args.labels_dir, args.out_dir, device)


if __name__ == "__main__":
    main()
