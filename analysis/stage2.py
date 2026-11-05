"""B4: Stage 2 analysis tables and figures (README 8, step B4): RQ2, RQ3, bridge.

Run from the repo root after B3 (no GPU needed):
    python analysis/stage2.py
or open analysis/stage2.ipynb, which calls the same functions.

RQ2, from detector/test_scores.csv (test split):
  per detector, AUROC of fabricated vs copied and of fabricated vs computed,
  separately, and the share of each label's numbers it flags (score >= its
  val-tuned threshold). The computed share = correct computed numbers flagged.
RQ3, for the lookback and lookback_hf detectors, on the same numbers as B3:
  per layer   a detector on that layer's heads only (C tuned on val); test AUROC
  per head    importance = |coefficient| of the head's standardized feature in
              the full detector refit on train with B3's C and cutoff (summed
              over its 3 features for lookback_hf)
  top-k heads a detector on the k most important heads (ranked on train only);
              test AUROC, next to the full detector's
Bridge: mean lookback ratio (over all layers and heads) per decoding setting,
  for all numbers and per label, all splits, from features/.
All CIs: 95%, bootstrap over reports.

Writes to analysis/figures/: stage2_rq2.csv/.md, stage2_layers.csv/.png,
stage2_heads.csv/.png, stage2_topk.csv, stage2_bridge.csv/.png
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.stage1 import bootstrap_ci as ratio_ci  # noqa: E402
from common.io import read_jsonl  # noqa: E402
from detector.train_eval import (  # noqa: E402
    C_GRID, DETECTORS, bootstrap_ci, features_for, finite_mask, fit_logreg, load_setting, pool,
    subset, targets,
)
from labels.label import LABELS  # noqa: E402

RQ3_DETECTORS = ["lookback", "lookback_hf"]
TOP_K = [10, 25, 50, 100, 200]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical, fixed order
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# RQ2 -----------------------------------------------------------------------

def read_test_scores(path):
    """detector/test_scores.csv as a dict of arrays."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {k: np.array([r[k] for r in rows]) for k in ("setting", "id", "report", "label")}
    for det in DETECTORS:
        out[det] = np.array([float(r[det]) for r in rows])
        out[f"{det}_flagged"] = np.array([int(r[f"{det}_flagged"]) for r in rows])
    return out


def pair_auroc(labels, scores, groups, neg, pos="fabricated", n_boot=1000, seed=0):
    """AUROC of pos (1) vs neg (0), only numbers with those labels; with a CI."""
    from sklearn.metrics import roc_auc_score

    keep = (labels == pos) | (labels == neg)
    y = (labels[keep] == pos).astype(int)
    if len(np.unique(y)) < 2:
        return None, None, None, int(keep.sum())
    lo, hi = bootstrap_ci(y, scores[keep], groups[keep], roc_auc_score, n_boot, seed)
    return float(roc_auc_score(y, scores[keep])), lo, hi, int(keep.sum())


def flag_share(labels, flagged, groups, label, n_boot=1000, seed=0):
    """Share of `label` numbers that are flagged, with a CI over reports."""
    keep = labels == label
    if not keep.any():
        return None, None, None, 0
    per_report = defaultdict(lambda: [0, 0])
    for g, f in zip(groups[keep], flagged[keep]):
        per_report[g][0] += int(f)
        per_report[g][1] += 1
    lo, hi = ratio_ci(list(per_report.values()), n_boot=n_boot, seed=seed)
    return float(flagged[keep].mean()), lo, hi, int(keep.sum())


def rq2_table(ts, n_boot=1000, seed=0):
    rows = []
    for det in DETECTORS:
        row = {"detector": det}
        for neg in ("copied", "computed"):
            auc, lo, hi, n = pair_auroc(ts["label"], ts[det], ts["report"], neg, n_boot=n_boot, seed=seed)
            row.update({f"auroc_fab_vs_{neg}": auc, f"auroc_fab_vs_{neg}_ci_low": lo,
                        f"auroc_fab_vs_{neg}_ci_high": hi, f"n_fab_vs_{neg}": n})
        for label in LABELS:
            share, lo, hi, n = flag_share(ts["label"], ts[f"{det}_flagged"], ts["report"], label, n_boot, seed)
            row.update({f"flagged_{label}": share, f"flagged_{label}_ci_low": lo,
                        f"flagged_{label}_ci_high": hi, f"n_{label}": n})
        rows.append(row)
    return rows


# RQ3 -----------------------------------------------------------------------

def head_matrix(data, feature_set, heads, cutoff_index=None):
    """Features of the given (layer, head) pairs only: [numbers, k] for lookback,
    [numbers, 3k] for lookback_hf (lookback, hf_ctx, hf_gen blocks)."""
    layers = np.array([h[0] for h in heads])
    hs = np.array([h[1] for h in heads])
    blocks = [data["lookback"][:, layers, hs]]
    if feature_set == "lookback_hf":
        blocks += [data["hf_ctx"][:, cutoff_index, layers, hs], data["hf_gen"][:, cutoff_index, layers, hs]]
    elif feature_set != "lookback":
        raise ValueError(f"RQ3 feature set must be one of {RQ3_DETECTORS}")
    return np.concatenate(blocks, axis=1)


def layer_heads(layer, n_heads):
    return [(layer, h) for h in range(n_heads)]


def head_importance(coef, feature_set, n_layers, n_heads):
    """|coefficients| of features_for(...) mapped back to [layers, heads].
    lookback_hf has three blocks of layers x heads; a head's importance is their sum."""
    blocks = 1 if feature_set == "lookback" else 3
    if len(coef) != blocks * n_layers * n_heads:
        raise ValueError("coefficient count doesn't match the feature set")
    return np.abs(np.asarray(coef)).reshape(blocks, n_layers, n_heads).sum(0)


def rank_heads(importance):
    """(layer, head) pairs, most important first."""
    order = np.argsort(-importance, axis=None, kind="stable")
    return [tuple(int(x) for x in np.unravel_index(i, importance.shape)) for i in order]


def _masks(data):
    y = targets(data["label"])
    split = data["split"]
    return y, (split == "train") & (y >= 0), (split == "val") & (y >= 0), (split == "test") & (y >= 0)


def _test_auroc(model, X, y, test, groups, n_boot, seed):
    from sklearn.metrics import roc_auc_score
    scores = model.predict_proba(X[test])[:, 1]
    lo, hi = bootstrap_ci(y[test], scores, groups, roc_auc_score, n_boot, seed)
    return float(roc_auc_score(y[test], scores)), lo, hi


def rq3(data, feature_set, C, cutoff_index, Cs=C_GRID, top_k=TOP_K, n_boot=1000, seed=0):
    """(layer rows, head rows, top-k rows, importance [layers, heads]) for one feature set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y, train, val, test = _masks(data)
    groups = data["report"][test]
    n_layers, n_heads = data["lookback"].shape[1:]

    X = features_for(data, feature_set, cutoff_index)
    full = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=seed))
    full.fit(X[train], y[train])
    importance = head_importance(full[-1].coef_[0], feature_set, n_layers, n_heads)
    full_auc = _test_auroc(full, X, y, test, groups, n_boot, seed)

    layer_rows = []
    for layer in range(n_layers):
        Xl = head_matrix(data, feature_set, layer_heads(layer, n_heads), cutoff_index)
        val_auc, C_l, model = fit_logreg(Xl[train], y[train], Xl[val], y[val], Cs, seed)
        auc, lo, hi = _test_auroc(model, Xl, y, test, groups, n_boot, seed)
        layer_rows.append({"feature_set": feature_set, "layer": layer, "auroc": auc, "auroc_ci_low": lo,
                           "auroc_ci_high": hi, "val_auroc": val_auc, "C": C_l})

    ranked = rank_heads(importance)
    head_rows = [{"feature_set": feature_set, "rank": k + 1, "layer": l, "head": h,
                  "importance": float(importance[l, h])} for k, (l, h) in enumerate(ranked)]

    topk_rows = []
    for k in [k for k in top_k if k < len(ranked)]:
        Xk = head_matrix(data, feature_set, ranked[:k], cutoff_index)
        val_auc, C_k, model = fit_logreg(Xk[train], y[train], Xk[val], y[val], Cs, seed)
        auc, lo, hi = _test_auroc(model, Xk, y, test, groups, n_boot, seed)
        topk_rows.append({"feature_set": feature_set, "heads": k, "auroc": auc, "auroc_ci_low": lo,
                          "auroc_ci_high": hi, "val_auroc": val_auc, "C": C_k})
    topk_rows.append({"feature_set": feature_set, "heads": len(ranked), "auroc": full_auc[0],
                      "auroc_ci_low": full_auc[1], "auroc_ci_high": full_auc[2], "val_auroc": None, "C": C})
    return layer_rows, head_rows, topk_rows, importance


# Bridge --------------------------------------------------------------------

def load_lookback(features_dir, settings, report_of):
    """Per number: mean lookback over layers and heads, setting, label, report.
    Numbers with no lookback (all NaN) are left out."""
    out = defaultdict(list)
    for s in settings:
        f = np.load(Path(features_dir) / f"{s}.npz")
        n = len(f["label"])
        flat = f["lookback"].reshape(n, -1)
        ok = np.isfinite(flat).any(1)
        with np.errstate(invalid="ignore"):
            out["mean_lookback"].append(np.nanmean(flat[ok], axis=1))
        out["setting"].append(np.array([s] * int(ok.sum())))
        out["label"].append(f["label"][ok])
        out["report"].append(np.array([report_of[i] for i in f["id"][ok]]))
    return {k: np.concatenate(v) for k, v in out.items()}


def bridge_table(lb, settings, n_boot=1000, seed=0):
    """Mean lookback per setting, for all numbers and per label, with CIs over reports."""
    rows = []
    for s in settings:
        for label in ["all"] + LABELS:
            keep = (lb["setting"] == s) & ((lb["label"] == label) if label != "all" else True)
            if not keep.any():
                continue
            per_report = defaultdict(lambda: [0.0, 0])
            for g, v in zip(lb["report"][keep], lb["mean_lookback"][keep]):
                per_report[g][0] += float(v)
                per_report[g][1] += 1
            lo, hi = ratio_ci(list(per_report.values()), n_boot=n_boot, seed=seed)
            rows.append({"setting": s, "label": label, "mean_lookback": float(lb["mean_lookback"][keep].mean()),
                         "ci_low": lo, "ci_high": hi, "n": int(keep.sum())})
    return rows


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


def plot_layers(path, layer_rows):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.text(0, 0.505, "chance", color=MUTED, fontsize=8, va="bottom")
    for color, fs in zip(SERIES, RQ3_DETECTORS):
        rows = [r for r in layer_rows if r["feature_set"] == fs]
        if not rows:
            continue
        x = [r["layer"] for r in rows]
        y = [r["auroc"] for r in rows]
        ax.fill_between(x, [r["auroc_ci_low"] for r in rows], [r["auroc_ci_high"] for r in rows],
                        color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=3, label=fs, zorder=3)
    ax.set_xlabel("layer", fontsize=8, color=INK_2)
    ax.set_ylabel("test AUROC", fontsize=8, color=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.13))
    ax.set_title("Detector on one layer's heads only · shaded: 95% CI over reports",
                 loc="left", fontsize=9, color=INK, pad=24)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_heads(path, importances):
    """One heatmap per feature set: head importance, layers x heads."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    n = len(importances)
    first = next(iter(importances.values()))
    n_layers, n_heads = first.shape
    fig, axes = plt.subplots(1, n, figsize=(0.35 * n_heads * n + 2.5 * n, 0.22 * n_layers + 1.4),
                             facecolor=SURFACE, squeeze=False)
    for ax, (fs, imp) in zip(axes[0], importances.items()):
        mesh = ax.pcolormesh(imp, cmap=cmap, vmin=0, edgecolors=SURFACE, linewidth=1)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(n_heads) + 0.5, np.arange(n_heads))
        ax.set_yticks(np.arange(0, n_layers, 2) + 0.5, np.arange(0, n_layers, 2))
        ax.tick_params(colors=MUTED, labelsize=7, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel("head", fontsize=8, color=INK_2)
        ax.set_ylabel("layer", fontsize=8, color=INK_2)
        ax.set_title(f"{fs}: head importance", loc="left", fontsize=9, color=INK)
        bar = fig.colorbar(mesh, ax=ax, fraction=0.06, pad=0.03)
        bar.set_label("|coefficient| (standardized)", fontsize=8, color=INK_2)
        bar.ax.tick_params(colors=MUTED, labelsize=7)
        bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_bridge(path, bridge_rows, settings):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.2 * len(settings) + 3, 3.8), facecolor=SURFACE)
    _style(ax)
    groups = [("all", "all numbers", INK)] + [
        (lab, lab, c) for lab, c in zip(["copied", "computed", "fabricated"], SERIES)]
    width = 0.8 / len(groups)
    for j, (label, name, color) in enumerate(groups):
        rows = {r["setting"]: r for r in bridge_rows if r["label"] == label}
        xs = [i + (j - (len(groups) - 1) / 2) * width for i, s in enumerate(settings) if s in rows]
        ys = [rows[s]["mean_lookback"] for s in settings if s in rows]
        err = [[rows[s]["mean_lookback"] - rows[s]["ci_low"] for s in settings if s in rows],
               [rows[s]["ci_high"] - rows[s]["mean_lookback"] for s in settings if s in rows]]
        ax.errorbar(xs, ys, yerr=err, fmt="o", color=color, ecolor=color, markersize=6,
                    elinewidth=1.2, capsize=0, label=name, zorder=3)
    ax.set_xticks(range(len(settings)), settings)
    ax.set_ylabel("mean lookback ratio", fontsize=8, color=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.14))
    ax.set_title("Mean lookback (all layers and heads) by decoding setting · bars: 95% CI over reports",
                 loc="left", fontsize=9, color=INK, pad=24)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# Run -----------------------------------------------------------------------

def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def _rq2_markdown(rows):
    def pct(v):
        return "–" if v is None else f"{100 * v:.0f}%"

    def auc(r, k):
        return "–" if r[k] is None else f"{r[k]:.3f} ({r[k + '_ci_low']:.3f}–{r[k + '_ci_high']:.3f})"

    lines = ["| Detector | AUROC fabricated vs copied | AUROC fabricated vs computed "
             "| Flagged: copied | computed | wrong_calc | question | fabricated |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['detector']} | {auc(r, 'auroc_fab_vs_copied')} | {auc(r, 'auroc_fab_vs_computed')} "
                     f"| {pct(r['flagged_copied'])} | {pct(r['flagged_computed'])} | {pct(r['flagged_wrong_calc'])} "
                     f"| {pct(r['flagged_question'])} | {pct(r['flagged_fabricated'])} |")
    return "\n".join(lines) + "\n"


def run_all(cfg, subset_path, features_dir, baseline_dir, detector_dir, out_dir, n_boot=1000, top_k=TOP_K):
    """Compute every B4 table and figure. Returns a dict of the tables."""
    seed = cfg["seed"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = {q["id"]: q for q in read_jsonl(subset_path)}
    report_of = {i: q["report_key"] for i, q in questions.items()}
    tables = {}

    # RQ2
    ts = read_test_scores(Path(detector_dir) / "test_scores.csv")
    tables["rq2"] = rq2_table(ts, n_boot, seed)
    _write_csv(out_dir / "stage2_rq2.csv", tables["rq2"])
    (out_dir / "stage2_rq2.md").write_text(_rq2_markdown(tables["rq2"]))

    # RQ3: same numbers and settings as B3
    settings = [s for s in cfg["decoding"]
                if (Path(features_dir) / f"{s}.npz").exists() and (Path(baseline_dir) / f"confidence_{s}.npz").exists()]
    per_setting = {s: load_setting(Path(features_dir) / f"{s}.npz", Path(baseline_dir) / f"confidence_{s}.npz")
                   for s in settings}
    data = pool(per_setting, {i: q["split"] for i, q in questions.items()}, report_of)
    data = subset(data, finite_mask(data))
    with open(Path(detector_dir) / "results.csv", newline="") as f:
        b3 = {r["detector"]: r for r in csv.DictReader(f)}
    layer_rows, head_rows, topk_rows, importances = [], [], [], {}
    for fs in RQ3_DETECTORS:
        cut = b3[fs]["hf_cutoff"]
        ci = int(np.flatnonzero(np.isclose(data["hf_cutoffs"], float(cut)))[0]) if cut else None
        lr, hr, tr, imp = rq3(data, fs, float(b3[fs]["C"]), ci, top_k=top_k, n_boot=n_boot, seed=seed)
        layer_rows += lr
        head_rows += hr
        topk_rows += tr
        importances[fs] = imp
    tables.update(layers=layer_rows, heads=head_rows, topk=topk_rows)
    _write_csv(out_dir / "stage2_layers.csv", layer_rows)
    _write_csv(out_dir / "stage2_heads.csv", head_rows)
    _write_csv(out_dir / "stage2_topk.csv", topk_rows)
    plot_layers(out_dir / "stage2_layers.png", layer_rows)
    plot_heads(out_dir / "stage2_heads.png", importances)

    # Bridge: every setting with features
    fsettings = [s for s in cfg["decoding"] if (Path(features_dir) / f"{s}.npz").exists()]
    tables["bridge"] = bridge_table(load_lookback(features_dir, fsettings, report_of), fsettings, n_boot, seed)
    _write_csv(out_dir / "stage2_bridge.csv", tables["bridge"])
    plot_bridge(out_dir / "stage2_bridge.png", tables["bridge"], fsettings)
    return tables


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--subset", type=Path, default=ROOT / "data" / "finqa_subset.jsonl")
    ap.add_argument("--features-dir", type=Path, default=ROOT / "features")
    ap.add_argument("--baseline-dir", type=Path, default=ROOT / "baseline")
    ap.add_argument("--detector-dir", type=Path, default=ROOT / "detector")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "figures")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tables = run_all(cfg, args.subset, args.features_dir, args.baseline_dir, args.detector_dir,
                     args.out_dir, args.n_boot)

    print("RQ2 (test split)\n" + _rq2_markdown(tables["rq2"]))
    print("RQ3 top-k heads (test AUROC)")
    for r in tables["topk"]:
        print(f"  {r['feature_set']:<12} {r['heads']:>4} heads  {r['auroc']:.3f} "
              f"({r['auroc_ci_low']:.3f}-{r['auroc_ci_high']:.3f})")
    print("Bridge: mean lookback per setting (all numbers)")
    for r in tables["bridge"]:
        if r["label"] == "all":
            print(f"  {r['setting']:<8} {r['mean_lookback']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f})  n={r['n']}")
    print(f"wrote stage2_* tables and figures to {args.out_dir}")


if __name__ == "__main__":
    main()
