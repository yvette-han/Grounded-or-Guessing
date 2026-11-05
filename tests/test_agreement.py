"""Tests for labels/agreement.py. Expected values are worked out by hand."""

import csv

import pytest

from labels.agreement import cohen_kappa, compare, load_rows, normalize_label


def test_normalize_label():
    assert normalize_label(" Copied ") == "copied"
    assert normalize_label("wrong calc") == "wrong_calc"
    assert normalize_label("Wrong-Calc") == "wrong_calc"
    assert normalize_label("") is None
    with pytest.raises(ValueError):
        normalize_label("copy")


def test_cohen_kappa_by_hand():
    # p_o = 3/4; p_e = 0.5*0.25 + 0.5*0.75 = 0.5; kappa = (0.75 - 0.5) / 0.5 = 0.5
    assert cohen_kappa(["x", "x", "y", "y"], ["x", "y", "y", "y"]) == pytest.approx(0.5)
    assert cohen_kappa(["x", "y"], ["x", "y"]) == pytest.approx(1.0)
    assert cohen_kappa(["x", "x"], ["x", "x"]) == 1.0   # one label everywhere


def test_compare():
    rows = [
        {"row": "1", "id": "q1", "number": "4", "a": "copied", "b": "copied", "auto": "copied"},
        {"row": "2", "id": "q1", "number": "5", "a": "computed", "b": "computed", "auto": "fabricated"},
        {"row": "3", "id": "q2", "number": "6", "a": "copied", "b": "wrong_calc", "auto": "copied"},
        {"row": "4", "id": "q2", "number": "7", "a": "fabricated", "b": None, "auto": "fabricated"},
    ]
    res = compare(rows)
    assert res["a_vs_b"] == (2, 3)               # rows 1-3 have both labels; 1 and 2 agree
    assert res["a_vs_auto"] == (3, 4)            # rows 1, 3, 4
    assert res["b_vs_auto"] == (1, 3)            # B labeled rows 1-3; only row 1 matches auto
    assert res["consensus_vs_auto"] == (1, 2)    # A = B on rows 1, 2; auto agrees on row 1
    assert res["confusion"][("computed", "fabricated")] == 1
    assert [r["row"] for r in res["disagreements"]] == ["2", "3"]


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_load_rows_joins_by_row_even_when_sorted(tmp_path):
    key = [{"row": "1", "id": "q1", "number": "4", "auto_label": "copied"},
           {"row": "2", "id": "q2", "number": "5", "auto_label": "computed"}]
    sheet_a = [{"row": "2", "id": "q2", "label_A": "Computed", "label_B": ""},   # sorted differently
               {"row": "1", "id": "q1", "label_A": "copied", "label_B": ""}]
    sheet_b = [{"row": "1", "id": "q1", "label_A": "", "label_B": "copied"},
               {"row": "2", "id": "q2", "label_A": "", "label_B": "fabricated"}]
    _write(tmp_path / "key.csv", key)
    _write(tmp_path / "a.csv", sheet_a)
    _write(tmp_path / "b.csv", sheet_b)
    rows, errors = load_rows(tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "key.csv")
    assert errors == []
    assert [(r["row"], r["a"], r["b"], r["auto"]) for r in rows] == [
        ("1", "copied", "copied", "copied"), ("2", "computed", "fabricated", "computed")]


def test_load_rows_reports_bad_labels(tmp_path):
    _write(tmp_path / "key.csv", [{"row": "1", "id": "q1", "number": "4", "auto_label": "copied"}])
    _write(tmp_path / "s.csv", [{"row": "1", "id": "q1", "label_A": "copy", "label_B": ""}])
    _, errors = load_rows(tmp_path / "s.csv", tmp_path / "s.csv", tmp_path / "key.csv")
    assert len(errors) == 1 and "label_A" in errors[0]
