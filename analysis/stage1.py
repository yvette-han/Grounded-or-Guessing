"""A6: Stage 1 results table and bar chart, per decoding setting (README 8, step A6; RQ0).

Run from the repo root after A3 and A4 (any settings that exist are used):
    python analysis/stage1.py
or open analysis/stage1.ipynb, which calls the same functions.

Writes to analysis/figures/:
  stage1_table.csv   every measure below, one row per setting
  stage1_table.md    the main columns, formatted, for the report/poster
  stage1_bars.png    numeric accuracy and fabricated-number share per setting, with 95% CIs

Measures (all over the questions answered in every available setting):
  accuracy               the answer's last number equals the gold answer (within label_tolerance)
  gold_anywhere          some number in the answer equals the gold answer
  fabricated_share       numbers labeled fabricated / all labeled numbers
  answers_with_fabricated  answers with at least one fabricated number
  <label>_share          the same for copied, computed, question, wrong_calc
  differs_from_greedy    the last number differs from greedy's last number for the same question
  mean_new_tokens, cut_at_max   answer length (A3), and answers cut at max_new_tokens
  distinct finals (printed): mean number of different last numbers per question across settings
CIs: 95% bootstrap over reports (company/year), as in README B3.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io import read_jsonl  # noqa: E402
from common.numbers import extract_numbers, find_match, same_value  # noqa: E402
from labels.label import LABELS  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


# Pure functions (tested in tests/test_stage1.py) ---------------------------

def final_number(answer):
    """The last number in the answer (the model's final answer), or None."""
    numbers = extract_numbers(answer)
    return numbers[-1] if numbers else None


def is_correct(answer, gold, tol):
    n = final_number(answer)
    return n is not None and same_value(n, gold, tol)


def gold_anywhere(answer, gold, tol):
    return find_match(gold, extract_numbers(answer), tol) is not None


def same_final(x, y, tol):
    """Two final answers (Number or None) are the same; two Nones count as the same."""
    if x is None or y is None:
        return x is None and y is None
    return same_value(x, y, tol)


def distinct_count(finals, tol):
    """How many different final answers; 'no number' (None) counts as one answer."""
    groups = []
    for f in finals:
        if not any(same_final(f, g, tol) for g in groups):
            groups.append(f)
    return len(groups)


def bootstrap_ci(per_report, n_boot=1000, seed=0, level=0.95):
    """CI of sum(numerators) / sum(denominators), resampling reports with replacement.

    per_report: list of (numerator, denominator), one pair per report.
    """
    arr = np.asarray(per_report, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    num, den = arr[idx, 0].sum(1), arr[idx, 1].sum(1)
    ratios = num[den > 0] / den[den > 0]
    lo, hi = np.percentile(ratios, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def _ratio_with_ci(pairs_by_report, seed):
    pairs = list(pairs_by_report.values())
    num, den = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    if den == 0:
        return None, None, None
    lo, hi = bootstrap_ci(pairs, seed=seed)
    return num / den, lo, hi


def setting_metrics(setting, answers, records, questions, ids, tol, greedy_finals=None, seed=0):
    """One table row. answers: id -> A3 row; records: A4 rows (or None if not labeled);
    questions: id -> subset row; ids: the questions to use."""
    row = {"setting": setting, "questions": len(ids)}

    correct = defaultdict(lambda: [0, 0])
    for i in ids:
        r = correct[questions[i]["report_key"]]
        r[0] += is_correct(answers[i]["answer_text"], questions[i]["gold_answer"], tol)
        r[1] += 1
    row["accuracy"], row["accuracy_ci_low"], row["accuracy_ci_high"] = _ratio_with_ci(correct, seed)
    row["gold_anywhere"] = sum(gold_anywhere(answers[i]["answer_text"], questions[i]["gold_answer"], tol)
                               for i in ids) / len(ids)

    if records is not None:
        idset = set(ids)
        recs = [r for r in records if r["id"] in idset]
        counts = Counter(r["label"] for r in recs)
        fab = defaultdict(lambda: [0, 0])
        for r in recs:
            f = fab[questions[r["id"]]["report_key"]]
            f[0] += r["label"] == "fabricated"
            f[1] += 1
        row["numbers"] = len(recs)
        row["fabricated_share"], row["fabricated_ci_low"], row["fabricated_ci_high"] = _ratio_with_ci(fab, seed)
        row["answers_with_fabricated"] = len({r["id"] for r in recs if r["label"] == "fabricated"}) / len(ids)
        for label in LABELS:
            if label != "fabricated":
                row[f"{label}_share"] = counts[label] / len(recs) if recs else None
        row["numbers_per_answer"] = len(recs) / len(ids)

    if greedy_finals is not None:
        row["differs_from_greedy"] = sum(
            not same_final(final_number(answers[i]["answer_text"]), greedy_finals[i], tol) for i in ids) / len(ids)
    row["mean_new_tokens"] = sum(answers[i]["n_new_tokens"] for i in ids) / len(ids)
    row["cut_at_max"] = sum(bool(answers[i]["hit_max_tokens"]) for i in ids) / len(ids)
    return row


# Loading and output --------------------------------------------------------

def build_table(cfg, subset, answers_dir, labels_dir, seed=0):
    """(rows, info) for every setting in the config that has an answers file."""
    tol = cfg["label_tolerance"]
    questions = {r["id"]: r for r in read_jsonl(subset)}
    answers, records, missing = {}, {}, []
    for s in cfg["decoding"]:
        path = Path(answers_dir) / f"answers_{s}.jsonl"
        if not path.exists():
            missing.append(s)
            continue
        answers[s] = {r["id"]: r for r in read_jsonl(path)}
        lpath = Path(labels_dir) / f"numbers_{s}.jsonl"
        records[s] = read_jsonl(lpath) if lpath.exists() else None
    if not answers:
        raise FileNotFoundError(f"no answers_*.jsonl in {answers_dir}")

    ids = sorted(set.intersection(*(set(a) for a in answers.values())))
    greedy_finals = ({i: final_number(answers["greedy"][i]["answer_text"]) for i in ids}
                     if "greedy" in answers else None)
    rows = [setting_metrics(s, answers[s], records[s], questions, ids, tol, greedy_finals, seed)
            for s in answers]
    finals = [[final_number(answers[s][i]["answer_text"]) for s in answers] for i in ids]
    info = {
        "settings": list(answers),
        "missing_answers": missing,
        "missing_labels": [s for s in answers if records[s] is None],
        "common_questions": len(ids),
        "mean_distinct_finals": sum(distinct_count(f, tol) for f in finals) / len(ids),
    }
    return rows, info


def write_outputs(rows, info, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "stage1_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def pct(v):
        return "–" if v is None else f"{100 * v:.1f}%"

    def ci(r, a, b):
        return "–" if r.get(a) is None else f"{100 * r[a]:.1f}–{100 * r[b]:.1f}"

    lines = [
        f"Questions answered in every setting: {info['common_questions']}. "
        f"95% CIs: bootstrap over reports.",
        "",
        "| Setting | Accuracy | 95% CI | Fabricated numbers | 95% CI | Answers with a fabricated number "
        "| Differs from greedy | Mean tokens | Cut at max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['setting']} | {pct(r['accuracy'])} | {ci(r, 'accuracy_ci_low', 'accuracy_ci_high')} "
            f"| {pct(r.get('fabricated_share'))} | {ci(r, 'fabricated_ci_low', 'fabricated_ci_high')} "
            f"| {pct(r.get('answers_with_fabricated'))} | {pct(r.get('differs_from_greedy'))} "
            f"| {r['mean_new_tokens']:.0f} | {pct(r['cut_at_max'])} |")
    (out_dir / "stage1_table.md").write_text("\n".join(lines) + "\n")
    plot_bars(out_dir / "stage1_bars.png", rows, info)


def plot_bars(path, rows, info):
    """Grouped bars: accuracy and fabricated-number share per setting, with 95% CIs."""
    import matplotlib.pyplot as plt

    names = [r["setting"] for r in rows]
    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(1.3 * len(rows) + 2.5, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # (value column, CI low column, CI high column, legend label, color, bar offset)
    series = [
        ("accuracy", "accuracy_ci_low", "accuracy_ci_high",
         "Numeric accuracy", BLUE, -width / 2 - 0.01),
        ("fabricated_share", "fabricated_ci_low", "fabricated_ci_high",
         "Fabricated numbers (share of all numbers)", ORANGE, width / 2 + 0.01),
    ]
    for key, key_lo, key_hi, label, color, dx in series:
        vals = [r.get(key) for r in rows]
        if all(v is None for v in vals):
            continue  # e.g. no labels yet
        heights = [100 * v if v is not None else 0 for v in vals]
        lo = [100 * (v - r[key_lo]) if v is not None else 0 for v, r in zip(vals, rows)]
        hi = [100 * (r[key_hi] - v) if v is not None else 0 for v, r in zip(vals, rows)]
        ax.bar(x + dx, heights, width, color=color, label=label, zorder=2)
        ax.errorbar(x + dx, heights, yerr=[lo, hi], fmt="none", ecolor=INK_2, elinewidth=1,
                    capsize=3, zorder=3)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% ", color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0, 1.13), ncol=2)
    ax.set_title(f"By decoding setting · {info['common_questions']} questions · "
                 f"error bars: 95% CI over reports", loc="left", fontsize=9, color=INK, pad=26)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--subset", type=Path, default=ROOT / "data" / "finqa_subset.jsonl")
    ap.add_argument("--answers-dir", type=Path, default=ROOT / "gen")
    ap.add_argument("--labels-dir", type=Path, default=ROOT / "labels")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "figures")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rows, info = build_table(cfg, args.subset, args.answers_dir, args.labels_dir, seed=cfg["seed"])
    write_outputs(rows, info, args.out_dir)
    print((args.out_dir / "stage1_table.md").read_text())
    print(f"mean number of different final answers per question across "
          f"{len(info['settings'])} settings: {info['mean_distinct_finals']:.2f}")
    if info["missing_answers"]:
        print(f"no answers yet for: {', '.join(info['missing_answers'])}")
    if info["missing_labels"]:
        print(f"no labels yet for: {', '.join(info['missing_labels'])} (run labels/label.py)")
    print(f"wrote {args.out_dir}/stage1_table.csv, stage1_table.md, stage1_bars.png")


if __name__ == "__main__":
    main()
