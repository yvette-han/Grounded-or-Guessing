from collections import Counter

import pytest

from common.splits import assign_splits, report_key

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def make_ids(n_reports=60, per_report=5):
    return [f"CO{r}/{2000 + r % 20}/page_{p}.pdf-{p % 3}"
            for r in range(n_reports) for p in range(per_report)]


@pytest.mark.parametrize("example_id, key", [
    ("ADI/2009/page_49.pdf-1", "ADI/2009"),
    ("ADI/2009/page_49.pdf-12", "ADI/2009"),
    ("BRK/B/2004/page_1.pdf-3", None),  # not company/year: should fail loudly
])
def test_report_key(example_id, key):
    if key is None:
        with pytest.raises(ValueError):
            report_key(example_id)
    else:
        assert report_key(example_id) == key


def test_every_id_gets_one_split():
    ids = make_ids()
    out = assign_splits(ids, RATIOS, seed=354)
    assert set(out) == set(ids)
    assert set(out.values()) == set(RATIOS)


def test_no_report_in_two_splits():
    out = assign_splits(make_ids(), RATIOS, seed=354)
    splits_per_report = {}
    for i, s in out.items():
        splits_per_report.setdefault(report_key(i), set()).add(s)
    assert all(len(s) == 1 for s in splits_per_report.values())


def test_split_sizes_close_to_ratios():
    ids = make_ids(n_reports=200, per_report=5)  # 1000 questions
    counts = Counter(assign_splits(ids, RATIOS, seed=354).values())
    # Each split can go over by at most one report (5 questions).
    assert 700 <= counts["train"] <= 705
    assert 150 <= counts["val"] <= 155
    assert counts["test"] == 1000 - counts["train"] - counts["val"]


def test_same_seed_same_split_and_order_does_not_matter():
    ids = make_ids()
    a = assign_splits(ids, RATIOS, seed=354)
    b = assign_splits(list(reversed(ids)), RATIOS, seed=354)
    assert a == b


def test_different_seed_different_split():
    ids = make_ids()
    assert assign_splits(ids, RATIOS, seed=1) != assign_splits(ids, RATIOS, seed=2)


def test_bad_ratios_and_duplicates():
    with pytest.raises(ValueError):
        assign_splits(make_ids(), {"train": 0.7, "val": 0.2, "test": 0.2}, seed=0)
    with pytest.raises(ValueError):
        assign_splits(["A/2009/page_1.pdf-1"] * 2, RATIOS, seed=0)
