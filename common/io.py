"""Small file helpers shared by the scripts."""

import json
from collections import OrderedDict


def read_jsonl(path):
    """All records of a .jsonl file, skipping blank lines."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def group_records(records):
    """id -> list of (row index, record), keeping file order.

    The row index is the record's line in the file: the `row` join key of
    baseline/ and features/ outputs (README 7).
    """
    groups = OrderedDict()
    for row, r in enumerate(records):
        groups.setdefault(r["id"], []).append((row, r))
    return groups
