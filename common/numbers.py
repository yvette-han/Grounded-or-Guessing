"""Find numbers in text and decide when two numbers are equal (README 6.3).

extract_numbers(text)   every number mention, with its value and character span
same_value(a, b, tol)   whether two numbers count as the same for labeling
find_match(x, pool, tol)
                        the first number in `pool` equal to x, or None

Both the report and the model's answer go through extract_numbers, so the
same rules apply on both sides.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Number:
    text: str          # the whole mention as written, e.g. "($4.2 billion)"
    start: int         # character span of the mention in the source text
    end: int
    num_start: int     # character span of just the digits, e.g. "4.2"
    num_end: int
    value: float       # signed, scale applied: "$4.2 billion" -> 4.2e9, "(12)" -> -12
    is_percent: bool   # "12%" -> value 12.0 with is_percent True
    has_scale: bool    # a scale word or abbreviation was written: thousand, million, bn, ...


# The digits: 4215, 4,215, 4.2, .5. Not inside a word or identifier (Q4, page_49).
_DIGITS = re.compile(r"(?<![\w.])(?:(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?|\.\d+)")

# Just before the digits: "(", a minus sign and "$", in the orders they occur:
# "-$12", "$-12", "($12)", "$(12)", "( 12 )". A minus right after a letter or
# digit is a hyphen ("2008-2009", "COVID-19"), not a sign.
_BEFORE = re.compile(
    r"(?P<open>\(\s*)?(?P<sign>(?<![\w)])[-−])?(?P<cur>\$\s?)?"
    r"(?P<open2>\(\s*)?(?P<sign2>(?<![\w)])[-−])?\Z"
)

# Just after the digits: a scale word ("4.2 billion"), an abbreviation
# ("$500M", "$1.5bn") or the multiplier "x" ("1.5x", value unchanged).
_SUFFIX = re.compile(
    r"\s?(?P<word>thousand|million|billion|trillion)s?\b"
    r"|\s?(?P<abbr>bn|mn|mm)\b"
    r"|(?P<short>[kmb]|x)\b",
    re.IGNORECASE,
)
_SCALES = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "m": 1e6, "mm": 1e6, "mn": 1e6,
    "billion": 1e9, "b": 1e9, "bn": 1e9,
    "trillion": 1e12,
}
_PERCENT = re.compile(r"\s?(?:%|percent\b|per cent\b)", re.IGNORECASE)
_CLOSE = re.compile(r"\s*\)")
_MONTH_BEFORE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\Z", re.IGNORECASE
)


def extract_numbers(text):
    """All number mentions in `text`, in order. Years, days in dates and list markers are skipped."""
    numbers = []
    for m in _DIGITS.finditer(text):
        n = _read_number(text, m)
        if n is not None:
            numbers.append(n)
    return numbers


def _read_number(text, m):
    digits = m.group()
    num_start, num_end = m.span()

    # After the digits: scale or "x", then "%", then ")".
    pos = num_end
    scale, has_scale = 1.0, False
    suffix = _SUFFIX.match(text, pos)
    if suffix:
        name = (suffix.group("word") or suffix.group("abbr") or suffix.group("short")).lower()
        if name != "x":
            scale, has_scale = _SCALES[name], True
        pos = suffix.end()
    if pos < len(text) and text[pos].isalpha():
        return None  # part of a word: "4th", "3rd", "10yr"
    percent = _PERCENT.match(text, pos)
    if percent:
        pos = percent.end()
    close = _CLOSE.match(text, pos)

    # Before the digits: "(", minus, "$".
    before = _BEFORE.search(text, max(0, num_start - 8), num_start)
    opened = "open" if before.group("open") else "open2" if before.group("open2") else None
    paren = bool(opened and close)  # "(12)" in accounting means -12
    sign = before.group("sign") or before.group("sign2")
    currency = before.group("cur")

    # Skip rules for plain integers.
    if "." not in digits and "," not in digits and not (currency or sign or has_scale or percent):
        v = int(digits)
        if 1990 <= v <= 2030:
            return None  # a year
        if 1 <= v <= 31 and not paren and _MONTH_BEFORE.search(text, max(0, num_start - 12), num_start):
            return None  # the day in "december 31"
        if _is_list_marker(text, num_start, num_end):
            return None  # "1." / "1)" / "(1)" at the start of a line

    value = float(digits.replace(",", "")) * scale
    if sign or paren:
        value = -value

    # The mention covers the decorations we used.
    starts = [num_start] + [before.start(g) for g in ("sign", "cur", "sign2") if before.group(g)]
    if paren:
        starts.append(before.start(opened))
    start = min(starts)
    end = close.end() if paren else pos
    return Number(text[start:end], start, end, num_start, num_end,
                  value, bool(percent), has_scale)


def _is_list_marker(text, num_start, num_end):
    """'1.', '1)' or '(1)' at the start of a line, followed by whitespace."""
    line_start = text.rfind("\n", 0, num_start) + 1
    if not re.fullmatch(r"\s*\(?", text[line_start:num_start]):
        return False
    after = text[num_end:num_end + 2]
    return len(after) == 2 and after[0] in ".)" and after[1].isspace()


# Comparing numbers ----------------------------------------------------------

_UNITS = (1.0, 1e3, 1e6, 1e9)  # ones, thousands, millions, billions


def _as_number(x):
    """A plain float (e.g. a gold answer) is a bare number: no %, no scale word."""
    if isinstance(x, Number):
        return x
    return Number(str(x), -1, -1, -1, -1, float(x), False, False)


def _readings(x, other):
    """The values x could stand for when compared with `other`."""
    values = [x.value]
    if x.is_percent and not other.is_percent:
        values.append(x.value / 100)  # 12% vs 0.12
    if x.has_scale and not other.has_scale:
        # "$4.2 billion" vs a bare "4215" from a table in millions: try each unit.
        values = [v / unit for v in values for unit in _UNITS]
    return values


def _close(a, b, tol):
    return abs(a - b) <= tol * max(abs(a), abs(b))


def same_value(a, b, tol, ignore_sign=True):
    """Whether a and b (Number or float) are the same number, within relative `tol`.

    - Two percents, or two numbers with scale words, are compared as written.
    - A percent vs a non-percent: 12% equals both 12 and 0.12.
    - A number with a scale word vs a bare number: the bare number may be in
      ones, thousands, millions or billions ("$4.2 billion" equals 4200).
    - ignore_sign=True compares absolute values ("a decrease of 12" vs "(12)").
    """
    a, b = _as_number(a), _as_number(b)
    pairs = [(x, y) for x in _readings(a, b) for y in _readings(b, a)]
    if ignore_sign:
        pairs = [(abs(x), abs(y)) for x, y in pairs]
    return any(_close(x, y, tol) for x, y in pairs)


def find_match(x, pool, tol, ignore_sign=True):
    """The first item of `pool` with the same value as x, or None."""
    return next((p for p in pool if same_value(x, p, tol, ignore_sign)), None)
