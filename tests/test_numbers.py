"""Tests for common/numbers.py.

README A2 asks for ~30 hand-made strings. Each case below is
(text, expected numbers), where each expected number is (value, is_percent).
Check every expected value by hand before trusting the parser (README 3).
"""

import pytest

from common.numbers import extract_numbers, find_match, same_value

P, N = True, False  # percent / not percent

CASES = [
    # README examples
    ("$4.2 billion", [(4.2e9, N)]),
    ("(12.5)", [(-12.5, N)]),
    ("12%", [(12, P)]),
    ("in 2009", []),
    # commas, decimals
    ("4,215", [(4215, N)]),
    ("1,234,567.89", [(1234567.89, N)]),
    ("0.12", [(0.12, N)]),
    (".5", [(0.5, N)]),
    # FinQA's spaced style
    ("$ 4215 million", [(4.215e9, N)]),
    ("( 12 )", [(-12, N)]),
    ("23.6 %", [(23.6, P)]),
    ("revenue | 2009 | 4,215", [(4215, N)]),
    # negatives
    ("-3.5%", [(-3.5, P)]),
    ("−7", [(-7, N)]),  # unicode minus sign
    ("-$1,200", [(-1200, N)]),
    ("$(1,200)", [(-1200, N)]),
    ("($4.2 million)", [(-4.2e6, N)]),
    ("the 2008-2009 change was -5", [(-5, N)]),  # hyphen in a range is not a minus
    # percent and scale words
    ("12 percent", [(12, P)]),
    ("$500M", [(5e8, N)]),
    ("$1.5bn", [(1.5e9, N)]),
    ("3k", [(3000, N)]),
    ("1.5x", [(1.5, N)]),
    ("12 months", [(12, N)]),
    ("rose 5% to $4.2 billion in 2009.", [(5, P), (4.2e9, N)]),
    # years: 1990-2030, plain integer
    ("from 2008 to 2009", []),
    ("2009.5", [(2009.5, N)]),   # decimal point: not a year
    ("$2009", [(2009, N)]),      # currency: not a year
    ("2,009", [(2009, N)]),      # comma: not a year
    ("1989 and 2031", [(1989, N), (2031, N)]),  # outside the year range
    # list markers, dates, words
    ("1. revenue was 100", [(100, N)]),
    ("notes:\n2) cost was 7.5", [(7.5, N)]),
    ("(1) revenue of 4", [(4, N)]),
    ("december 31 , 2009", []),
    ("dec . 29 2012", []),           # FinQA's spaced abbreviation
    ("dec. 29, 2012", []),
    ("Q4 and the 3rd quarter", []),
]


@pytest.mark.parametrize("text, expected", CASES)
def test_extract_values(text, expected):
    got = [(n.value, n.is_percent) for n in extract_numbers(text)]
    assert len(got) == len(expected), got
    for (v, p), (ev, ep) in zip(got, expected):
        assert v == pytest.approx(ev)
        assert p == ep


@pytest.mark.parametrize("text, mention, digits", [
    ("revenue fell to ($4.2 million) in 2009", "($4.2 million)", "4.2"),
    ("growth of 23.6 % .", "23.6 %", "23.6"),
    ("a loss of -$1,200 .", "-$1,200", "1,200"),
    ("net ( 12 ) .", "( 12 )", "12"),
])
def test_spans(text, mention, digits):
    (n,) = extract_numbers(text)
    assert n.text == mention == text[n.start:n.end]
    assert text[n.num_start:n.num_end] == digits


def one(text):
    (n,) = extract_numbers(text)
    return n


TOL = 0.01


@pytest.mark.parametrize("a, b, equal", [
    (4.2, 4.21, True),                                 # README: 4.2 matches 4.21
    (4.2, 4.3, False),
    (one("12%"), 0.12, True),                          # README: 12% equals 0.12
    (one("12%"), 12.0, True),
    (one("12%"), one("0.12%"), False),                 # two percents: compared as written
    (one("$4.2 billion"), one("4,200 million"), True),  # README: 4.2 billion = 4,200 million
    (one("$4.2 billion"), one("4215"), True),          # bare table number in millions
    (one("4.2 million"), one("4.2 billion"), False),   # two scale words: compared as written
    (one("(12)"), 12.0, True),                         # sign ignored by default
    (0.14134, one("14.1%"), True),                     # gold ratio vs answer percent
    (one("14%"), 0.15, False),
    (0.0, 0.0, True),
    (0.0, 0.001, False),
])
def test_same_value(a, b, equal):
    assert same_value(a, b, TOL) is equal
    assert same_value(b, a, TOL) is equal  # symmetric


def test_same_value_with_sign():
    assert not same_value(one("(12)"), 12.0, TOL, ignore_sign=False)
    assert same_value(one("(12)"), -12.0, TOL, ignore_sign=False)


def test_find_match():
    report = extract_numbers("revenue | 2009 | 4,215\ncost | 2009 | 3,100")
    assert find_match(one("$4.2 billion"), report, TOL).value == 4215
    assert find_match(one("$5 billion"), report, TOL) is None
