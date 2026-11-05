"""B5: error-analysis material (README 8, step B5).

Run from the repo root after B3 (no GPU needed):
    python analysis/errors.py
    python analysis/errors.py --detector lookback --pick extreme
or open analysis/errors.ipynb, which calls the same functions.

For one detector (default: the learned detector with the best val AUROC in
detector/results.csv, so the choice doesn't depend on test results), on the test split:
  false positives  copied or computed numbers it flags (score >= its val-tuned threshold)
  false negatives  fabricated numbers it doesn't flag
  --pick random (default): a seeded random sample of each; --pick extreme: the
  highest-scoring false positives and lowest-scoring false negatives.
Each example shows the question, gold answer and steps, the answer with the
number marked, the labels of every number in that answer, and the report lines
where the number's value appears.
Where wrong_calc falls: score summary per label (n, quartiles, share flagged)
and a box plot of scores per label with the threshold.

Writes to analysis/figures/:
  b5_examples.md, b5_examples.csv   the false positives and false negatives
  b5_scores_by_label.csv, .png      score distribution per label
The explanations are ours to write (README 3), in analysis/errors.ipynb.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.stage2 import read_test_scores  # noqa: E402
from common.io import read_jsonl  # noqa: E402
from common.numbers import extract_numbers, find_match, same_value  # noqa: E402
from labels.label import LABELS  # noqa: E402

LEARNED = ["lookback", "hf", "lookback_hf", "confidence"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


# Pure functions (tested in tests/test_errors.py) ---------------------------

def choose_detector(results):
    """The learned detector with the best val AUROC (results: rows of detector/results.csv)."""
    rows = [r for r in results if r["detector"] in LEARNED]
    return max(rows, key=lambda r: float(r["val_auroc"]))["detector"]


def pick_errors(ts, detector, n=10, seed=0, pick="random"):
    """(false positive indices, false negative indices) into the test-score arrays."""
    labels, flagged, scores = ts["label"], ts[f"{detector}_flagged"], ts[detector]
    fp = np.flatnonzero(np.isin(labels, ["copied", "computed"]) & (flagged == 1))
    fn = np.flatnonzero((labels == "fabricated") & (flagged == 0))
    if pick == "extreme":
        return fp[np.argsort(-scores[fp], kind="stable")][:n], fn[np.argsort(scores[fn], kind="stable")][:n]
    if pick != "random":
        raise ValueError("pick must be 'random' or 'extreme'")
    rng = np.random.default_rng(seed)

    def sample(idx):
        if len(idx) == 0:
            return idx
        return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))

    return sample(fp), sample(fn)


def mark(text, start, end):
    """The text with [start, end) wrapped in ⟦ ⟧."""
    return text[:start] + "⟦" + text[start:end] + "⟧" + text[end:]


def report_lines_with(context, number, tol):
    """Report lines that contain a number equal to `number`."""
    return [line for line in context.split("\n") if find_match(number, extract_numbers(line), tol) is not None]


def gold_steps_matching(number, question, tol):
    """Which gold values the number equals: 'answer' and/or 'step k'."""
    hits = ["answer"] if same_value(number, question["gold_answer"], tol) else []
    hits += [f"step {k}" for k, v in enumerate(question["gold_steps"]) if v is not None and same_value(number, v, tol)]
    return hits


def score_summary(ts, detector):
    """Per label: n, mean, quartiles of the score, and share flagged."""
    rows = []
    for label in LABELS:
        keep = ts["label"] == label
        if not keep.any():
            continue
        s = ts[detector][keep]
        q25, q50, q75 = np.percentile(s, [25, 50, 75])
        rows.append({"label": label, "n": int(keep.sum()), "mean": float(s.mean()), "q25": float(q25),
                     "median": float(q50), "q75": float(q75),
                     "flagged": float(ts[f"{detector}_flagged"][keep].mean())})
    return rows


# Building the examples -----------------------------------------------------

def build_examples(ts, idx, kind, detector, threshold, labels_by_setting, answers_by_setting, questions, tol):
    """One dict per example, with everything needed to judge it."""
    out = []
    for k, i in enumerate(idx, 1):
        setting, qid = ts["setting"][i], ts["id"][i]
        rec = labels_by_setting[setting][int(ts["row"][i])]
        answer = answers_by_setting[setting][qid]["answer_text"]
        q = questions[qid]
        number = extract_numbers(answer[rec["char_start"]:rec["char_end"]])
        number = number[0] if number else rec["value"]
        same_answer = [r for r in labels_by_setting[setting] if r["id"] == qid]
        out.append({
            "kind": kind, "n": k, "detector": detector, "score": float(ts[detector][i]), "threshold": threshold,
            "setting": setting, "id": qid, "label": rec["label"], "number": rec["number_text"],
            "question": q["question"], "gold_answer": q["gold_answer"], "gold_program": q["gold_program"],
            "gold_steps": ", ".join("n/a" if v is None else f"{v:g}" for v in q["gold_steps"]),
            "matches_gold": ", ".join(gold_steps_matching(number, q, tol)),
            "answer_marked": mark(answer, rec["char_start"], rec["char_end"]),
            "answer_labels": " · ".join(f"{r['number_text']} {r['label']}" for r in same_answer),
            "report_lines": " / ".join(report_lines_with(q["context_text"], number, tol)),
        })
    return out


def examples_markdown(examples, detector, threshold):
    lines = [f"# B5 examples · detector `{detector}` · threshold {threshold:.3f}", "",
             "False positives: copied or computed numbers the detector flags. False negatives: fabricated "
             "numbers it misses. Test split. Explanations go in `analysis/errors.ipynb`.", ""]
    for ex in examples:
        title = "False positive" if ex["kind"] == "fp" else "False negative"
        lines += [
            f"## {title} {ex['n']} · {ex['label']} `{ex['number']}` · score {ex['score']:.3f} · "
            f"{ex['setting']} · {ex['id']}", "",
            f"- **Question:** {ex['question']}",
            f"- **Gold:** {ex['gold_answer']:g} · `{ex['gold_program']}` · steps: {ex['gold_steps']}"
            + (f" · this number equals the gold {ex['matches_gold']}" if ex["matches_gold"] else ""),
            f"- **Numbers in this answer:** {ex['answer_labels']}",
            f"- **Report lines with this value:** {ex['report_lines'] or 'none'}", "",
            "> " + ex["answer_marked"].replace("\n", "\n> "), "",
        ]
    return "\n".join(lines)


def plot_scores(path, ts, detector, threshold):
    import matplotlib.pyplot as plt
    present = [lab for lab in LABELS if (ts["label"] == lab).any()]
    data = [ts[detector][ts["label"] == lab] for lab in present]
    fig, ax = plt.subplots(figsize=(1.3 * len(present) + 2.5, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    box = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                     medianprops={"color": INK, "linewidth": 1.5},
                     whiskerprops={"color": INK_2}, capprops={"color": INK_2})
    for patch in box["boxes"]:
        patch.set(facecolor=BLUE, alpha=0.35, edgecolor=BLUE)
    rng = np.random.default_rng(0)
    for k, d in enumerate(data, 1):
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(d)), d, s=4, color=BLUE, alpha=0.35, linewidths=0, zorder=3)
    ax.axhline(threshold, color=ORANGE, linewidth=1.2, linestyle="--", zorder=4)
    ax.text(0.55, threshold, "threshold", color=ORANGE, fontsize=8, va="bottom", ha="left", zorder=5)
    ax.set_xticks(range(1, len(present) + 1), [f"{lab}\n(n={len(d)})" for lab, d in zip(present, data)])
    ax.set_ylabel(f"{detector} score", fontsize=8, color=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(f"Test scores by label · detector {detector}", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_all(cfg, subset_path, answers_dir, labels_dir, detector_dir, out_dir, detector=None, n=10, pick="random"):
    """Write every B5 output. Returns (detector, examples, score summary)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(detector_dir) / "results.csv", newline="") as f:
        results = list(csv.DictReader(f))
    detector = detector or choose_detector(results)
    threshold = float(next(r for r in results if r["detector"] == detector)["threshold"])
    ts = read_test_scores(Path(detector_dir) / "test_scores.csv")

    settings = sorted(set(ts["setting"]))
    labels_by_setting = {s: read_jsonl(Path(labels_dir) / f"numbers_{s}.jsonl") for s in settings}
    answers_by_setting = {s: {a["id"]: a for a in read_jsonl(Path(answers_dir) / f"answers_{s}.jsonl")}
                          for s in settings}
    questions = {q["id"]: q for q in read_jsonl(subset_path)}
    tol = cfg["label_tolerance"]

    fp, fn = pick_errors(ts, detector, n, cfg["seed"], pick)
    examples = (build_examples(ts, fp, "fp", detector, threshold, labels_by_setting, answers_by_setting, questions, tol)
                + build_examples(ts, fn, "fn", detector, threshold, labels_by_setting, answers_by_setting,
                                 questions, tol))
    summary = score_summary(ts, detector)
    if examples:
        _write_csv(out_dir / "b5_examples.csv", examples)
    (out_dir / "b5_examples.md").write_text(examples_markdown(examples, detector, threshold))
    _write_csv(out_dir / "b5_scores_by_label.csv", summary)
    plot_scores(out_dir / "b5_scores_by_label.png", ts, detector, threshold)
    return detector, examples, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--detector", choices=LEARNED, help="default: best val AUROC")
    ap.add_argument("--pick", choices=["random", "extreme"], default="random")
    ap.add_argument("-n", type=int, default=10, help="examples of each kind (README: 10)")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--subset", type=Path, default=ROOT / "data" / "finqa_subset.jsonl")
    ap.add_argument("--answers-dir", type=Path, default=ROOT / "gen")
    ap.add_argument("--labels-dir", type=Path, default=ROOT / "labels")
    ap.add_argument("--detector-dir", type=Path, default=ROOT / "detector")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "figures")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    detector, examples, summary = run_all(cfg, args.subset, args.answers_dir, args.labels_dir, args.detector_dir,
                                          args.out_dir, args.detector, args.n, args.pick)
    n_fp = sum(e["kind"] == "fp" for e in examples)
    print(f"detector {detector}: {n_fp} false positives, {len(examples) - n_fp} false negatives "
          f"-> {args.out_dir / 'b5_examples.md'}")
    print(f"{'label':<11} {'n':>5} {'median':>7} {'q25':>6} {'q75':>6} {'flagged':>8}")
    for r in summary:
        print(f"{r['label']:<11} {r['n']:>5} {r['median']:7.3f} {r['q25']:6.3f} {r['q75']:6.3f} {r['flagged']:8.0%}")


if __name__ == "__main__":
    main()
