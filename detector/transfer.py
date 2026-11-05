"""R1: detectors trained on FinQA, tested on TAT-QA (README 8, R1).

Run from the repo root after the FinQA inputs of B3 and the TAT-QA runs of
MANUAL_STEPS.md step 12 exist (no GPU needed):
    python detector/transfer.py

Every detector is fit exactly as in B3: FinQA train split; C, HF cutoff and
F1 threshold chosen on FinQA val. Then every TAT-QA number is scored. Per
detector: FinQA test AUROC (for reference), and TAT-QA AUROC and F1 at the
FinQA threshold, with 95% CIs from bootstrap over TAT-QA tables. Task as in
README 6.5 (fabricated vs copied + computed); numbers with a NaN feature are
dropped, as in B3.

FinQA uses every setting with features/ and baseline/ files; TAT-QA every
setting with tatqa/features/ and tatqa/baseline/ files. Both must use the same
attention row mode and HF cutoffs.

Outputs in detector/:
  transfer_results.csv  one row per detector
  tatqa_scores.csv      every TAT-QA number (all labels): each detector's score and flag
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io import read_jsonl  # noqa: E402
from detector.train_eval import (  # noqa: E402
    C_GRID, DETECTORS, finite_mask, fit_detectors, load_setting, pool, score_detector, split_masks,
    subset, targets, test_metrics, write_csv,
)


def load_dataset(cfg, subset_path, features_dir, baseline_dir):
    """(data, settings): pooled features + confidence for every setting that has both."""
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
    return subset(data, finite_mask(data)), settings


def check_compatible(finqa_features, tatqa_features):
    """Both feature files must come from the same row mode and HF cutoffs."""
    a, b = np.load(finqa_features), np.load(tatqa_features)
    if str(a["row_mode"]) != str(b["row_mode"]):
        raise ValueError(f"row modes differ: FinQA {a['row_mode']}, TAT-QA {b['row_mode']}")
    if not np.allclose(a["hf_cutoffs"], b["hf_cutoffs"]):
        raise ValueError("HF cutoffs differ between FinQA and TAT-QA features")


def transfer(finqa, tatqa, Cs=C_GRID, n_boot=1000, seed=0):
    """(result rows, TAT-QA score columns): fit on FinQA as in B3, test on all of TAT-QA."""
    from sklearn.metrics import roc_auc_score

    y_f, _, _, test_f = split_masks(finqa)
    y_t = targets(tatqa["label"])
    keep = y_t >= 0
    if len(np.unique(y_t[keep])) < 2:
        raise ValueError("TAT-QA needs both fabricated and copied/computed numbers")
    groups = tatqa["report"][keep]

    fitted = fit_detectors(finqa, Cs, seed)
    rows, columns = [], {}
    for det in DETECTORS:
        fit = fitted[det]
        finqa_scores = score_detector(det, fit, finqa, seed)
        scores = score_detector(det, fit, tatqa, seed)
        metrics = test_metrics(y_t[keep], scores[keep], groups, fit["threshold"], n_boot, seed)
        rows.append({
            "detector": det,
            "finqa_test_auroc": float(roc_auc_score(y_f[test_f], finqa_scores[test_f])),
            **{f"tatqa_{k}": v for k, v in metrics.items()},
            "threshold": fit["threshold"], "C": fit["C"], "hf_cutoff": fit["hf_cutoff"],
            "n_tatqa": int(keep.sum()), "n_tatqa_fabricated": int((y_t[keep] == 1).sum()),
        })
        columns[det] = scores
        columns[f"{det}_flagged"] = (scores >= fit["threshold"]).astype(int)
    return rows, columns


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--finqa-subset", type=Path, default=ROOT / "data" / "finqa_subset.jsonl")
    ap.add_argument("--features-dir", type=Path, default=ROOT / "features")
    ap.add_argument("--baseline-dir", type=Path, default=ROOT / "baseline")
    ap.add_argument("--tatqa-subset", type=Path, default=ROOT / "data" / "tatqa_subset.jsonl")
    ap.add_argument("--tatqa-features-dir", type=Path, default=ROOT / "tatqa" / "features")
    ap.add_argument("--tatqa-baseline-dir", type=Path, default=ROOT / "tatqa" / "baseline")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "detector")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    finqa, f_settings = load_dataset(cfg, args.finqa_subset, args.features_dir, args.baseline_dir)
    tatqa, t_settings = load_dataset(cfg, args.tatqa_subset, args.tatqa_features_dir, args.tatqa_baseline_dir)
    check_compatible(args.features_dir / f"{f_settings[0]}.npz", args.tatqa_features_dir / f"{t_settings[0]}.npz")
    print(f"FinQA settings: {', '.join(f_settings)}; TAT-QA settings: {', '.join(t_settings)}")

    rows, columns = transfer(finqa, tatqa, n_boot=args.n_boot, seed=cfg["seed"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "transfer_results.csv", rows)
    score_rows = []
    for i in range(len(tatqa["label"])):
        r = {"setting": tatqa["setting"][i], "row": int(tatqa["row"][i]), "id": tatqa["id"][i],
             "report": tatqa["report"][i], "label": tatqa["label"][i]}
        r.update({k: (int(v[i]) if k.endswith("_flagged") else float(v[i])) for k, v in columns.items()})
        score_rows.append(r)
    write_csv(args.out_dir / "tatqa_scores.csv", score_rows)

    print(f"TAT-QA: {rows[0]['n_tatqa']} numbers in the task ({rows[0]['n_tatqa_fabricated']} fabricated)")
    print(f"{'detector':<12} {'FinQA AUROC':>11} {'TAT-QA AUROC':>12}  {'95% CI':<13} {'F1':>5}  {'95% CI'}")
    for r in rows:
        print(f"{r['detector']:<12} {r['finqa_test_auroc']:11.3f} {r['tatqa_auroc']:12.3f}  "
              f"{r['tatqa_auroc_ci_low']:.3f}-{r['tatqa_auroc_ci_high']:.3f}  {r['tatqa_f1']:5.3f}  "
              f"{r['tatqa_f1_ci_low']:.3f}-{r['tatqa_f1_ci_high']:.3f}")
    print(f"wrote {args.out_dir / 'transfer_results.csv'} and {args.out_dir / 'tatqa_scores.csv'}")


if __name__ == "__main__":
    main()
