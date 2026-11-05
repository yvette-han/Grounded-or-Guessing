"""Train/val/test split by report (README 6.2).

All questions from one report (company + year) go to the same split.
"""

import random
import re
from collections import defaultdict


def report_key(example_id):
    """'ADI/2009/page_49.pdf-1' -> 'ADI/2009'."""
    parts = example_id.split("/")
    if len(parts) < 3 or not parts[0] or not re.fullmatch(r"\d{4}", parts[1]):
        raise ValueError(f"not a FinQA id: {example_id!r}")
    return f"{parts[0]}/{parts[1]}"


def assign_splits(ids, ratios, seed):
    """Map each question id to a split name, keeping each report in one split.

    ratios: e.g. {"train": 0.70, "val": 0.15, "test": 0.15}; splits are filled
            in this order and the fractions refer to questions, not reports.
    seed:   reports are shuffled with random.Random(seed).

    Reports are taken in shuffled order. A report goes to the first split
    whose question quota is not yet full, so a split can go over its quota
    by at most one report.
    """
    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids")
    names = list(ratios)
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")

    by_report = defaultdict(list)
    for i in ids:
        by_report[report_key(i)].append(i)
    reports = sorted(by_report)
    random.Random(seed).shuffle(reports)

    # Question count at which each split is full.
    limits, total = [], 0.0
    for name in names:
        total += ratios[name]
        limits.append(total * len(ids))

    out, assigned = {}, 0
    for rep in reports:
        k = next((j for j, lim in enumerate(limits) if assigned < lim), len(names) - 1)
        for i in by_report[rep]:
            out[i] = names[k]
        assigned += len(by_report[rep])
    return out
