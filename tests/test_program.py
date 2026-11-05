import pytest

from common.program import parse_value, run_program, split_steps

TABLE = [
    ["", "2009", "2008"],
    ["revenue", "$ 4,215", "$ 3,900"],
    ["loss", "( 12 )", "5"],
]


def test_split_steps():
    assert split_steps("subtract(5829, 5735), divide(#0, 5735)") == [
        ("subtract", "5829", "5735"), ("divide", "#0", "5735")]
    # row names may contain parentheses and commas
    assert split_steps("table_average(net revenue ( loss ), none)") == [
        ("table_average", "net revenue ( loss )", "none")]
    assert split_steps("table_sum(income , net, none), add(#0, 1)") == [
        ("table_sum", "income , net", "none"), ("add", "#0", "1")]


@pytest.mark.parametrize("text, value", [
    ("4,215", 4215.0), ("23.6%", 0.236), ("const_100", 100.0), ("const_m1", -1.0),
    ("-3.5", -3.5), ("#0", None), ("none", None),
])
def test_parse_value(text, value):
    assert parse_value(text) == (pytest.approx(value) if value is not None else None)


@pytest.mark.parametrize("program, steps", [
    ("subtract(5829, 5735), divide(#0, 5735)", [94.0, 94 / 5735]),
    ("divide(9896, 23.6%)", [9896 / 0.236]),
    ("add(1, const_m1), multiply(#0, const_100)", [0.0, 0.0]),
    ("exp(1.05, const_2)", [1.1025]),
    ("table_average(revenue, none)", [4057.5]),
    ("table_max(revenue, none), subtract(#0, 100)", [4215.0, 4115.0]),
    ("table_sum(loss, none), add(#0, 1)", [None, None]),   # "( 12 )" is not a number to FinQA
    ("table_min(missing row, none)", [None]),
    ("divide(5, 0)", [None]),
    ("greater(5, 3)", [None]),                             # yes/no, not a number
])
def test_run_program(program, steps):
    got = run_program(program, TABLE)
    assert len(got) == len(steps)
    for g, s in zip(got, steps):
        assert g == (pytest.approx(s) if s is not None else None)
