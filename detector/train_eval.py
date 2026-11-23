"""B3: train and compare detectors of hallucinated numbers (README 8, step B3; tasks in 6.5).

Run from the repo root after B2 and A7 (no GPU needed):
    python detector/train_eval.py
    python detector/train_eval.py --features-dir features/predicting --out-dir detector/predicting

Tasks (README 6.5), each trained and tested separately:
  hallucinated (main)  fabricated or wrong_calc = 1; copied and computed = 0
  fabricated (extra)   fabricated = 1; copied and computed = 0; wrong_calc left out
question numbers are in neither task. Numbers outside a task are left out of
its training, tuning and metrics, but every test number is still scored (for
B4 and B5). The 3-way version (copied / computed / hallucinated) is
detector/three_way.py.

Data: every setting that has both features/{setting}.npz (B2) and
baseline/confidence_{setting}.npz (A7), pooled, joined on (setting, row), and
split by report with the splits in data/finqa_subset.jsonl. A number with a
NaN in any feature (e.g. no answer token before it) is dropped for every
detector, so all detectors see the same numbers.

Detectors:
  lookback     logistic regression on lookback [layers x heads]
  hf           ... on hf_ctx and hf_gen at one cutoff
  lookback_hf  ... on lookback, hf_ctx and hf_gen at one cutoff
  confidence   ... on mean_logprob, min_logprob, mean_entropy (A7)
  random       uniform random scores
Each logistic regression: standardize (fit on train only), L2 penalty; C and,
for hf / lookback_hf, the cutoff are chosen by val AUROC. Threshold: the one
with the best F1 on val. Test: AUROC and F1 at that threshold, with 95% CIs
from bootstrap resamples of test reports.

Outputs in detector/:
  results.csv      one row per task and detector
  test_scores.csv  every test number (all labels): each detector's score and flag.
                   Main task: columns `{detector}`, `{detector}_flagged`;
                   other tasks: `{task}_{detector}`, `{task}_{detector}_flagged`
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io import load_config, read_jsonl, run_dir  # noqa: E402

DETECTORS = ["lookback", "hf", "lookback_hf", "confidence", "random"]
NEGATIVE = frozenset({"copied", "computed"})
TASKS = {  # task: (positive labels, negative labels), README 6.5
    "hallucinated": (frozenset({"fabricated", "wrong_calc"}), NEGATIVE),
    "fabricated": (frozenset({"fabricated"}), NEGATIVE),
}
MAIN_TASK = "hallucinated"
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


# Pure functions (tested in tests/test_detector.py) -------------------------

def targets(labels, task=MAIN_TASK):
    """1 for the task's positive labels, 0 for copied/computed, -1 for labels left out (README 6.5)."""
    positive, negative = TASKS[task]
    return np.array([1 if lab in positive else 0 if lab in negative else -1 for lab in labels])


def column_name(task, detector):
    """Column of one task's scores in test_scores.csv: plain for the main task."""
    return detector if task == MAIN_TASK else f"{task}_{detector}"


def load_setting(features_path, confidence_path):
    """One setting's features and confidence, joined on `row`."""
    f, c = np.load(features_path), np.load(confidence_path)
    c_index = {int(r): i for i, r in enumerate(c["row"])}
    keep_f = np.array([i for i, r in enumerate(f["row"]) if int(r) in c_index], dtype=int)
    keep_c = np.array([c_index[int(f["row"][i])] for i in keep_f], dtype=int)
    if not np.array_equal(f["label"][keep_f], c["label"][keep_c]):
        raise ValueError(f"labels differ between {features_path} and {confidence_path}")
    conf = np.stack([c["mean_logprob"], c["min_logprob"], c["mean_entropy"]], axis=1)
    return {
        "lookback": f["lookback"][keep_f].astype(np.float32),
        "hf_ctx": f["hf_ctx"][keep_f].astype(np.float32),
        "hf_gen": f["hf_gen"][keep_f].astype(np.float32),
        "confidence": conf[keep_c].astype(np.float32),
        "label": f["label"][keep_f],
        "id": f["id"][keep_f],
        "row": f["row"][keep_f],
        "hf_cutoffs": f["hf_cutoffs"],
    }


def pool(settings_data, split_of, report_of):
    """Concatenate settings; add setting, split and report per number."""
    keys = ["lookback", "hf_ctx", "hf_gen", "confidence", "label", "id", "row"]
    data = {k: np.concatenate([d[k] for d in settings_data.values()]) for k in keys}
    data["setting"] = np.concatenate([[s] * len(d["label"]) for s, d in settings_data.items()])
    data["split"] = np.array([split_of[i] for i in data["id"]])
    data["report"] = np.array([report_of[i] for i in data["id"]])
    data["hf_cutoffs"] = next(iter(settings_data.values()))["hf_cutoffs"]
    return data


def finite_mask(data):
    """Numbers with no NaN/inf in any feature."""
    n = len(data["label"])
    return np.all([np.isfinite(data[k].reshape(n, -1)).all(1)
                   for k in ("lookback", "hf_ctx", "hf_gen", "confidence")], axis=0)


def subset(data, mask):
    return {k: (v[mask] if k != "hf_cutoffs" else v) for k, v in data.items()}


def features_for(data, detector, cutoff_index=None):
    """The feature matrix [numbers, features] for one detector."""
    n = len(data["label"])
    lookback = data["lookback"].reshape(n, -1)
    if detector == "confidence":
        return data["confidence"]
    if detector == "lookback":
        return lookback
    hf = [data["hf_ctx"][:, cutoff_index].reshape(n, -1), data["hf_gen"][:, cutoff_index].reshape(n, -1)]
    if detector == "hf":
        return np.concatenate(hf, axis=1)
    if detector == "lookback_hf":
        return np.concatenate([lookback] + hf, axis=1)
    raise ValueError(f"unknown detector {detector!r}")


def f1_at(y, scores, threshold):
    """F1 when every score >= threshold is flagged."""
    flagged = scores >= threshold
    tp = int((flagged & (y == 1)).sum())
    denom = int(flagged.sum()) + int((y == 1).sum())   # TP + FP + (TP + FN)
    return 2 * tp / denom if denom else 0.0


def best_f1_threshold(y, scores):
    """(threshold, F1): the threshold with the highest F1, trying every distinct score."""
    order = np.argsort(-scores, kind="stable")
    s, yy = scores[order], y[order]
    tp, fp = np.cumsum(yy == 1), np.cumsum(yy == 0)
    last_of_run = np.r_[s[1:] != s[:-1], True]   # flagging "score >= s" includes all ties
    tp, fp, s = tp[last_of_run], fp[last_of_run], s[last_of_run]
    f1 = 2 * tp / (tp + fp + (y == 1).sum())
    k = int(np.argmax(f1))
    return float(s[k]), float(f1[k])


def bootstrap_ci(y, scores, groups, metric, n_boot=1000, seed=0, level=0.95):
    """CI of metric(y, scores), resampling groups (reports) with replacement.
    Resamples with only one class are skipped (AUROC is undefined there)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(metric(y[idx], scores[idx]))
    lo, hi = np.percentile(values, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def fit_logreg(X_train, y_train, X_val, y_val, Cs, seed=0):
    """(val AUROC, C, fitted pipeline) for the C with the best val AUROC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best = None
    for C in Cs:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=seed))
        model.fit(X_train, y_train)   # the scaler is fit on train only
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        if best is None or auc > best[0]:
            best = (float(auc), C, model)
    return best


def split_masks(data, task=MAIN_TASK):
    """(y, train, val, test) where train/val/test keep only numbers in the task (y >= 0)."""
    y = targets(data["label"], task)
    split = data["split"]
    return y, (split == "train") & (y >= 0), (split == "val") & (y >= 0), (split == "test") & (y >= 0)


def score_detector(det, fit, data, seed=0):
    """Scores of one fitted detector on every number of `data`."""
    if det == "random":
        return np.random.default_rng(seed).uniform(size=len(data["label"]))
    return fit["model"].predict_proba(features_for(data, det, fit["cutoff_index"]))[:, 1]


def fit_detectors(data, Cs=C_GRID, seed=0, task=MAIN_TASK):
    """Fit every detector for one task on the train split and tune it on val.

    Returns {detector: {model, cutoff_index, hf_cutoff, C, val_auroc, threshold, n_features}};
    the threshold is the one with the best F1 on val.
    """
    from sklearn.metrics import roc_auc_score

    y, train, val, _ = split_masks(data, task)
    for name, m in (("train", train), ("val", val)):
        if len(np.unique(y[m])) < 2:
            raise ValueError(f"task {task}: the {name} split needs both {'/'.join(sorted(TASKS[task][0]))} "
                             "and copied/computed numbers")
    fitted = {}
    for det in DETECTORS:
        if det == "random":
            fit = {"model": None, "cutoff_index": None, "C": None, "n_features": 0}
        else:
            cut_options = range(len(data["hf_cutoffs"])) if det in ("hf", "lookback_hf") else [None]
            best = None
            for ci in cut_options:
                X = features_for(data, det, ci)
                val_auc, C, model = fit_logreg(X[train], y[train], X[val], y[val], Cs, seed)
                if best is None or val_auc > best[0]:
                    best = (val_auc, C, ci, model, X.shape[1])
            _, C, ci, model, n_features = best
            fit = {"model": model, "cutoff_index": ci, "C": C, "n_features": n_features}
        fit["hf_cutoff"] = (None if fit["cutoff_index"] is None
                            else round(float(data["hf_cutoffs"][fit["cutoff_index"]]), 6))  # stored as float32
        scores = score_detector(det, fit, data, seed)
        fit["val_auroc"] = float(roc_auc_score(y[val], scores[val]))
        fit["threshold"], _ = best_f1_threshold(y[val], scores[val])
        fitted[det] = fit
    return fitted


def test_metrics(y, scores, groups, threshold, n_boot=1000, seed=0):
    """AUROC and F1 at the threshold, each with a CI over groups (reports)."""
    from sklearn.metrics import roc_auc_score

    auc_lo, auc_hi = bootstrap_ci(y, scores, groups, roc_auc_score, n_boot, seed)
    f1_lo, f1_hi = bootstrap_ci(y, scores, groups, lambda yy, ss: f1_at(yy, ss, threshold), n_boot, seed)
    return {"auroc": float(roc_auc_score(y, scores)), "auroc_ci_low": auc_lo, "auroc_ci_high": auc_hi,
            "f1": f1_at(y, scores, threshold), "f1_ci_low": f1_lo, "f1_ci_high": f1_hi}


def run(data, Cs=C_GRID, n_boot=1000, seed=0, task=MAIN_TASK):
    """Train, tune and test every detector for one task.

    Returns (results rows, test score columns, mask of the test numbers): each
    column has one value per test number, in data order, for every label.
    Columns are named with column_name(task, detector).
    """
    y, train, val, test = split_masks(data, task)
    test_all = data["split"] == "test"
    if len(np.unique(y[test])) < 2:
        raise ValueError(f"task {task}: the test split needs both positive and copied/computed numbers")
    groups = data["report"][test]
    test_labels = data["label"][test]

    fitted = fit_detectors(data, Cs, seed, task)
    results, columns = [], {}
    for det in DETECTORS:
        fit = fitted[det]
        scores = score_detector(det, fit, data, seed)
        results.append({
            "task": task, "detector": det,
            **test_metrics(y[test], scores[test], groups, fit["threshold"], n_boot, seed),
            "threshold": fit["threshold"], "val_auroc": fit["val_auroc"], "C": fit["C"],
            "hf_cutoff": fit["hf_cutoff"], "n_features": fit["n_features"],
            "n_train": int(train.sum()), "n_val": int(val.sum()), "n_test": int(test.sum()),
            "n_test_positive": int((y[test] == 1).sum()),
            "n_test_fabricated": int((test_labels == "fabricated").sum()),
            "n_test_wrong_calc": int((test_labels == "wrong_calc").sum()),
        })
        name = column_name(task, det)
        columns[name] = scores[test_all]
        columns[f"{name}_flagged"] = (scores[test_all] >= fit["threshold"]).astype(int)
    return results, columns, test_all


def load_dataset(cfg, subset_path, features_dir, baseline_dir):
    """(data, settings, dropped labels): pooled features + confidence for every
    setting that has both files, without the numbers that have a NaN feature."""
    questions = {q["id"]: q for q in read_jsonl(subset_path)}
    settings = [s for s in cfg["decoding"]
                if (Path(features_dir) / f"{s}.npz").exists() and (Path(baseline_dir) / f"confidence_{s}.npz").exists()]
    if not settings:
        raise FileNotFoundError(f"no setting has both {features_dir}/<setting>.npz and "
                                f"{baseline_dir}/confidence_<setting>.npz")
    per_setting = {s: load_setting(Path(features_dir) / f"{s}.npz", Path(baseline_dir) / f"confidence_{s}.npz")
                   for s in settings}
    data = pool(per_setting, {i: q["split"] for i, q in questions.items()},
                {i: q["report_key"] for i, q in questions.items()})
    keep = finite_mask(data)
    return subset(data, keep), settings, data["label"][~keep]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--subset", type=Path, default=ROOT / "data" / "finqa_subset.jsonl")
    ap.add_argument("--features-dir", type=Path,
                    help="default: features/, or features/<run_name>/ if the config sets run_name")
    ap.add_argument("--baseline-dir", type=Path,
                    help="default: baseline/, or baseline/<run_name>/ if the config sets run_name")
    ap.add_argument("--out-dir", type=Path,
                    help="default: detector/, or detector/<run_name>/ if the config sets run_name")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    args.features_dir = args.features_dir or run_dir(cfg, ROOT, "features")
    args.baseline_dir = args.baseline_dir or run_dir(cfg, ROOT, "baseline")
    args.out_dir = args.out_dir or run_dir(cfg, ROOT, "detector")

    try:
        data, settings, dropped = load_dataset(cfg, args.subset, args.features_dir, args.baseline_dir)
    except FileNotFoundError as err:
        sys.exit(str(err))
    print(f"settings: {', '.join(settings)}; {len(data['label']) + len(dropped)} numbers, dropped {len(dropped)} "
          f"with a NaN feature ({', '.join(f'{lab} {n}' for lab, n in zip(*np.unique(dropped, return_counts=True)))})")

    results, columns, test_all = [], {}, None
    for task in TASKS:
        rows, cols, test_all = run(data, n_boot=args.n_boot, seed=cfg["seed"], task=task)
        results += rows
        columns.update(cols)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "results.csv", results)
    test_rows = []
    for k, i in enumerate(np.flatnonzero(test_all)):
        row = {"setting": data["setting"][i], "row": int(data["row"][i]), "id": data["id"][i],
               "report": data["report"][i], "label": data["label"][i]}
        row.update({name: (float(col[k]) if not name.endswith("_flagged") else int(col[k]))
                    for name, col in columns.items()})
        test_rows.append(row)
    write_csv(args.out_dir / "test_scores.csv", test_rows)

    for task in TASKS:
        rows = [r for r in results if r["task"] == task]
        r0 = rows[0]
        print(f"\ntask {task}{' (main)' if task == MAIN_TASK else ''}: train {r0['n_train']}, val {r0['n_val']}, "
              f"test {r0['n_test']} numbers ({r0['n_test_positive']} positive in test: "
              f"{r0['n_test_fabricated']} fabricated, {r0['n_test_wrong_calc']} wrong_calc)")
        print(f"{'detector':<12} {'AUROC':>6}  {'95% CI':<13} {'F1':>5}  {'95% CI':<13} {'C':>6} {'cutoff':>6}")
        for r in rows:
            print(f"{r['detector']:<12} {r['auroc']:6.3f}  {r['auroc_ci_low']:.3f}-{r['auroc_ci_high']:.3f}  "
                  f"{r['f1']:5.3f}  {r['f1_ci_low']:.3f}-{r['f1_ci_high']:.3f}  "
                  f"{'' if r['C'] is None else r['C']:>6} {'' if r['hf_cutoff'] is None else r['hf_cutoff']:>6}")
    print(f"wrote {args.out_dir / 'results.csv'} and {args.out_dir / 'test_scores.csv'}")


if __name__ == "__main__":
    main()
