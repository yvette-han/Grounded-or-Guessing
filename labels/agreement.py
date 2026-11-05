"""A5: compare the hand labels with each other and with the auto labels (README 8, step A5).

Workflow:
  1. python labels/label.py --setting greedy --handcheck 50
     -> labels/handcheck.csv (to fill in) and labels/handcheck_key.csv (auto labels)
  2. Each of us fills our own column (label_A or label_B) in our own copy of
     handcheck.csv, without looking at handcheck_key.csv or the other column.
     Use the label names of README 6.3: copied, computed, question, wrong_calc, fabricated.
  3. python labels/agreement.py --sheet-a <A's copy> --sheet-b <B's copy>
     (both default to labels/handcheck.csv, for when both columns are in one file)

Prints agreement rates and writes labels/handcheck_disagreements.csv with every
row where any two of A, B and the auto label differ. Rows are matched by the
`row` column, so sorting the sheet is fine.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labels.label import LABELS  # noqa: E402


def normalize_label(text):
    """' Copied ' -> 'copied'; '' -> None (not labeled yet). Unknown names raise ValueError."""
    label = text.strip().lower().replace(" ", "_").replace("-", "_")
    if not label:
        return None
    if label not in LABELS:
        raise ValueError(f"unknown label {text!r}; use one of {LABELS}")
    return label


def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length label lists: (p_o - p_e) / (1 - p_e)."""
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    count_a, count_b = Counter(a), Counter(b)
    p_e = sum(count_a[k] * count_b[k] for k in set(a) | set(b)) / n ** 2
    if p_e == 1:
        return 1.0  # both used one and the same label everywhere
    return (p_o - p_e) / (1 - p_e)


def rate(pairs):
    """(agreeing, total) for a list of (x, y) pairs."""
    return sum(x == y for x, y in pairs), len(pairs)


def compare(rows):
    """rows: dicts with row, id, number, a, b, auto (a/b may be None).

    Returns a dict of results; every rate is (agreeing, total) over the rows
    where both labels exist.
    """
    both = [r for r in rows if r["a"] and r["b"]]
    consensus = [r for r in both if r["a"] == r["b"]]
    return {
        "n_rows": len(rows),
        "n_a": sum(1 for r in rows if r["a"]),
        "n_b": sum(1 for r in rows if r["b"]),
        "a_vs_b": rate([(r["a"], r["b"]) for r in both]),
        "kappa_a_b": cohen_kappa([r["a"] for r in both], [r["b"] for r in both]) if both else None,
        "a_vs_auto": rate([(r["a"], r["auto"]) for r in rows if r["a"]]),
        "b_vs_auto": rate([(r["b"], r["auto"]) for r in rows if r["b"]]),
        "consensus_vs_auto": rate([(r["a"], r["auto"]) for r in consensus]),
        "confusion": Counter((r["a"], r["auto"]) for r in consensus),  # (hand, auto) -> count
        "disagreements": [r for r in rows
                          if len({x for x in (r["a"], r["b"], r["auto"]) if x}) > 1],
    }


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_rows(sheet_a, sheet_b, key):
    """Join A's labels, B's labels and the auto labels by the `row` column."""
    a = {r["row"]: r for r in read_csv(sheet_a)}
    b = {r["row"]: r for r in read_csv(sheet_b)}
    rows, errors = [], []
    for k in read_csv(key):
        row = k["row"]
        if row not in a or row not in b:
            errors.append(f"row {row}: missing from a sheet")
            continue
        if not (a[row]["id"] == b[row]["id"] == k["id"]):
            errors.append(f"row {row}: ids differ between the sheets and the key")
            continue
        entry = {"row": row, "id": k["id"], "number": k["number"], "auto": k["auto_label"]}
        for who, sheet, col in (("a", a, "label_A"), ("b", b, "label_B")):
            try:
                entry[who] = normalize_label(sheet[row].get(col, ""))
            except ValueError as err:
                errors.append(f"row {row}, {col}: {err}")
                entry[who] = None
        rows.append(entry)
    return rows, errors


def _pct(pair):
    agree, total = pair
    return f"{agree}/{total} = {agree / total:.0%}" if total else "n/a (no rows)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet-a", type=Path, default=ROOT / "labels" / "handcheck.csv")
    ap.add_argument("--sheet-b", type=Path, default=ROOT / "labels" / "handcheck.csv")
    ap.add_argument("--key", type=Path, default=ROOT / "labels" / "handcheck_key.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "labels" / "handcheck_disagreements.csv")
    args = ap.parse_args()

    rows, errors = load_rows(args.sheet_a, args.sheet_b, args.key)
    if errors:
        sys.exit("fix these first:\n  " + "\n  ".join(errors))
    res = compare(rows)

    print(f"{res['n_rows']} numbers; labeled by A: {res['n_a']}, by B: {res['n_b']}")
    kappa = f"{res['kappa_a_b']:.2f}" if res["kappa_a_b"] is not None else "n/a"
    print(f"A vs B:                 {_pct(res['a_vs_b'])}   (Cohen's kappa {kappa})")
    print(f"A vs auto:              {_pct(res['a_vs_auto'])}")
    print(f"B vs auto:              {_pct(res['b_vs_auto'])}")
    print(f"A = B rows vs auto:     {_pct(res['consensus_vs_auto'])}")

    if res["confusion"]:
        print("\nrows where A = B: hand label (rows) vs auto label (columns)")
        print(" " * 12 + "".join(f"{c:>12}" for c in LABELS))
        for h in LABELS:
            print(f"{h:>12}" + "".join(f"{res['confusion'][(h, c)]:>12}" for c in LABELS))

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["row", "id", "number", "a", "b", "auto"])
        w.writeheader()
        w.writerows(res["disagreements"])
    print(f"\n{len(res['disagreements'])} rows with any disagreement -> {args.out}")


if __name__ == "__main__":
    main()
