"""B3 (3-way): copied vs computed vs hallucinated numbers (README 8, step B3; classes in 6.5).

Run from the repo root after B2 and A7, next to train_eval.py (no GPU needed):
    python detector/three_way.py

Classes: copied, computed, hallucinated (fabricated or wrong_calc). question
numbers are left out. Same numbers, splits and feature sets as
detector/train_eval.py (numbers with a NaN feature dropped).

Per feature set (lookback, hf, lookback_hf, confidence): multinomial logistic
regression, standardized (scaler fit on train only), L2 penalty, class weights
"balanced" (each class counts equally in the loss, because copied numbers far
outnumber computed ones). C and, for hf / lookback_hf, the HF cutoff are chosen
by val macro-F1. The predicted class is the one with the highest probability.
Baseline `random`: each number gets one of the 3 classes uniformly at random.

Test metrics: macro-F1 (the mean of the per-class F1s, over the classes present)
with a 95% CI from bootstrap resamples of test reports, accuracy, recall per
class, and the confusion matrix.

Outputs:
  detector/three_way_results.csv       one row per detector
  detector/three_way_confusion.csv     detector, true class, predicted class, count (test)
  detector/three_way_predictions.csv   every test number in the task: label, class, and
                                       each detector's predicted class
  analysis/figures/three_way_confusion.png   confusion matrices, each row as shares of its true class
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.train_eval import (  # noqa: E402
    C_GRID, DETECTORS, TASKS, bootstrap_ci, features_for, load_dataset, write_csv,
)

CLASSES = ["copied", "computed", "hallucinated"]
HALLUCINATED = TASKS["hallucinated"][0]
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE_RAMP = ["#f2f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# Pure functions (tested in tests/test_three_way.py) ------------------------

def class_index(labels):
    """0 copied, 1 computed, 2 hallucinated (fabricated or wrong_calc), -1 left out."""
    return np.array([0 if lab == "copied" else 1 if lab == "computed" else 2 if lab in HALLUCINATED else -1
                     for lab in labels])


def per_class_f1(y, pred, n_classes=len(CLASSES)):
    """F1 of each class (NaN for a class absent from y and never predicted)."""
    out = []
    for c in range(n_classes):
        tp = int(((pred == c) & (y == c)).sum())
        denom = int((pred == c).sum()) + int((y == c).sum())   # 2TP + FP + FN
        out.append(2 * tp / denom if denom else np.nan)
    return np.array(out)


def macro_f1(y, pred):
    """Mean F1 over the classes present in y."""
    f1 = per_class_f1(y, pred)
    present = np.unique(y)
    return float(np.mean(f1[present]))


def confusion(y, pred, n_classes=len(CLASSES)):
    """Counts [true class, predicted class]."""
    m = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(m, (y, pred), 1)
    return m


def recall_per_class(m):
    """Share of each true class predicted correctly (NaN if the class has no numbers)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.diag(m) / m.sum(1)


def fit_multinomial(X_train, y_train, X_val, y_val, Cs, seed=0):
    """(val macro-F1, C, fitted pipeline) for the C with the best val macro-F1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best = None
    for C in Cs:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, class_weight="balanced", max_iter=5000, random_state=seed))
        model.fit(X_train, y_train)   # the scaler is fit on train only
        score = macro_f1(y_val, model.predict(X_val))
        if best is None or score > best[0]:
            best = (score, C, model)
    return best


def masks(data):
    """(y, train, val, test), each split keeping only numbers in the 3 classes."""
    y = class_index(data["label"])
    split = data["split"]
    return y, (split == "train") & (y >= 0), (split == "val") & (y >= 0), (split == "test") & (y >= 0)


def fit_three_way(data, Cs=C_GRID, seed=0):
    """{detector: {model, cutoff_index, hf_cutoff, C, val_macro_f1, n_features}}, fit on train, tuned on val."""
    y, train, val, _ = masks(data)
    for name, m in (("train", train), ("val", val)):
        if len(np.unique(y[m])) < 3:
            raise ValueError(f"the {name} split needs copied, computed and hallucinated numbers")
    fitted = {}
    for det in DETECTORS:
        if det == "random":
            fitted[det] = {"model": None, "cutoff_index": None, "hf_cutoff": None, "C": None, "n_features": 0}
        else:
            best = None
            for ci in (range(len(data["hf_cutoffs"])) if det in ("hf", "lookback_hf") else [None]):
                X = features_for(data, det, ci)
                score, C, model = fit_multinomial(X[train], y[train], X[val], y[val], Cs, seed)
                if best is None or score > best[0]:
                    best = (score, C, ci, model, X.shape[1])
            _, C, ci, model, n_features = best
            fitted[det] = {"model": model, "cutoff_index": ci, "C": C, "n_features": n_features,
                           "hf_cutoff": None if ci is None else round(float(data["hf_cutoffs"][ci]), 6)}
        fitted[det]["val_macro_f1"] = macro_f1(y[val], predict(det, fitted[det], data, seed)[val])
    return fitted


def predict(det, fit, data, seed=0):
    """Predicted class of every number in data."""
    if det == "random":
        return np.random.default_rng(seed).integers(0, len(CLASSES), size=len(data["label"]))
    return fit["model"].predict(features_for(data, det, fit["cutoff_index"]))


def run(data, Cs=C_GRID, n_boot=1000, seed=0):
    """(result rows, confusion rows, prediction columns, test mask) for every detector."""
    y, train, val, test = masks(data)
    if len(np.unique(y[test])) < 2:
        raise ValueError("the test split needs at least two of the classes")
    groups = data["report"][test]
    fitted = fit_three_way(data, Cs, seed)
    results, conf_rows, columns = [], [], {}
    for det in DETECTORS:
        fit = fitted[det]
        pred = predict(det, fit, data, seed)
        yt, pt = y[test], pred[test]
        m = confusion(yt, pt)
        rec = recall_per_class(m)
        lo, hi = bootstrap_ci(yt, pt, groups, macro_f1, n_boot, seed)
        results.append({
            "detector": det, "macro_f1": macro_f1(yt, pt), "macro_f1_ci_low": lo, "macro_f1_ci_high": hi,
            "accuracy": float((yt == pt).mean()),
            **{f"recall_{c}": (None if np.isnan(r) else float(r)) for c, r in zip(CLASSES, rec)},
            "val_macro_f1": fit["val_macro_f1"], "C": fit["C"], "hf_cutoff": fit["hf_cutoff"],
            "n_features": fit["n_features"], "n_train": int(train.sum()), "n_val": int(val.sum()),
            "n_test": int(test.sum()), **{f"n_test_{c}": int((yt == k).sum()) for k, c in enumerate(CLASSES)},
        })
        conf_rows += [{"detector": det, "true": CLASSES[a], "predicted": CLASSES[b], "n": int(m[a, b])}
                      for a in range(len(CLASSES)) for b in range(len(CLASSES))]
        columns[det] = np.array(CLASSES)[pt]
    return results, conf_rows, columns, test


# Plot ----------------------------------------------------------------------

def plot_confusions(path, conf_rows):
    """One matrix per detector; cells colored by the share of the true class, labeled with counts."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    dets = list(dict.fromkeys(r["detector"] for r in conf_rows))
    k = len(CLASSES)
    fig, axes = plt.subplots(1, len(dets), figsize=(2.9 * len(dets), 3.4), facecolor=SURFACE, squeeze=False)
    for ax, det in zip(axes[0], dets):
        m = np.zeros((k, k), dtype=int)
        for r in conf_rows:
            if r["detector"] == det:
                m[CLASSES.index(r["true"]), CLASSES.index(r["predicted"])] = r["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.nan_to_num(m / m.sum(1, keepdims=True))
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for a in range(k):
            for b in range(k):
                ax.text(b, a, f"{m[a, b]}\n{100 * share[a, b]:.0f}%", ha="center", va="center", fontsize=7,
                        color=SURFACE if share[a, b] > 0.55 else INK)
        ax.set_xticks(range(k), CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(k), CLASSES if ax is axes[0][0] else [""] * k)
        ax.tick_params(colors=MUTED, labelsize=7, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(det, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("predicted", fontsize=8, color=INK_2)
    axes[0][0].set_ylabel("true", fontsize=8, color=INK_2)
    fig.suptitle("3-way confusion on test · count and share of each true class", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# Run -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--subset", type=Path, default=ROOT / "data" / "finqa_subset.jsonl")
    ap.add_argument("--features-dir", type=Path, default=ROOT / "features")
    ap.add_argument("--baseline-dir", type=Path, default=ROOT / "baseline")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "detector")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "analysis" / "figures")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    try:
        data, settings, dropped = load_dataset(cfg, args.subset, args.features_dir, args.baseline_dir)
    except FileNotFoundError as err:
        sys.exit(str(err))
    print(f"settings: {', '.join(settings)}; dropped {len(dropped)} numbers with a NaN feature")

    results, conf_rows, columns, test = run(data, n_boot=args.n_boot, seed=cfg["seed"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "three_way_results.csv", results)
    write_csv(args.out_dir / "three_way_confusion.csv", conf_rows)
    pred_rows = []
    for k, i in enumerate(np.flatnonzero(test)):
        row = {"setting": data["setting"][i], "row": int(data["row"][i]), "id": data["id"][i],
               "report": data["report"][i], "label": data["label"][i],
               "class": CLASSES[class_index([data["label"][i]])[0]]}
        row.update({det: col[k] for det, col in columns.items()})
        pred_rows.append(row)
    write_csv(args.out_dir / "three_way_predictions.csv", pred_rows)
    plot_confusions(args.fig_dir / "three_way_confusion.png", conf_rows)

    r0 = results[0]
    print(f"train {r0['n_train']}, val {r0['n_val']}, test {r0['n_test']} numbers (test: "
          + ", ".join(f"{c} {r0[f'n_test_{c}']}" for c in CLASSES) + ")")
    print(f"{'detector':<12} {'macro-F1':>8}  {'95% CI':<13} {'acc':>5}  "
          + "  ".join(f"{'recall ' + c:>19}" for c in CLASSES))
    for r in results:
        recalls = "  ".join(f"{'–' if r[f'recall_{c}'] is None else format(r[f'recall_{c}'], '.2f'):>19}"
                            for c in CLASSES)
        print(f"{r['detector']:<12} {r['macro_f1']:8.3f}  {r['macro_f1_ci_low']:.3f}-{r['macro_f1_ci_high']:.3f}  "
              f"{r['accuracy']:5.2f}  {recalls}")
    print(f"wrote three_way_results.csv, three_way_confusion.csv, three_way_predictions.csv to {args.out_dir} "
          f"and {args.fig_dir / 'three_way_confusion.png'}")


if __name__ == "__main__":
    main()
