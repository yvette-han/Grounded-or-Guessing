"""Execute a FinQA gold program and return every step's value.

Follows FinQA's own executor (code/evaluate/evaluate.py in the FinQA repo):
  - numbers may have commas; "23.6%" means 0.236
  - const_100 means 100, const_m1 means -1
  - "#k" is the result of step k
  - table_sum/average/max/min(row name, none) use the table row whose first
    cell is that name; each other cell has "$" removed and anything from "("
    on dropped, and the step fails if any cell is not a number

Example: "subtract(5829, 5735), divide(#0, 5735)" -> [94.0, 0.01639...]
"""

import re

ARITHMETIC = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "exp": lambda a, b: a ** b,
}
TABLE = {
    "table_sum": sum,
    "table_average": lambda xs: sum(xs) / len(xs),
    "table_max": max,
    "table_min": min,
}

# Steps are separated by "), " followed by the next op name and "(".
# (Row names can contain commas and parentheses, so a simple split won't do.)
_STEP_BOUNDARY = re.compile(r"\)\s*,\s*(?=(?:%s)\()" % "|".join(list(ARITHMETIC) + list(TABLE) + ["greater"]))


def parse_value(text):
    """A program argument as a number, or None. FinQA's str_to_num."""
    text = text.strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        pass
    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100
        except ValueError:
            return None
    if text.startswith("const_"):
        rest = text[len("const_"):]
        return -1.0 if rest == "m1" else _float_or_none(rest)
    return None


def _float_or_none(text):
    try:
        return float(text)
    except ValueError:
        return None


def _row_values(cells):
    """FinQA's process_row: all cells as numbers, or None if any isn't one."""
    values = []
    for cell in cells:
        v = parse_value(cell.replace("$", "").split("(")[0])
        if v is None:
            return None
        values.append(v)
    return values or None


def split_steps(program):
    """'subtract(5829, 5735), divide(#0, 5735)' -> [('subtract', '5829', '5735'), ('divide', '#0', '5735')]."""
    steps = []
    for part in _STEP_BOUNDARY.split(program.strip()):
        op, _, args = part.partition("(")
        args = args.rstrip().removesuffix(")")
        arg1, _, arg2 = args.rpartition(",")  # the 2nd argument never contains a comma
        steps.append((op.strip(), arg1.strip(), arg2.strip()))
    return steps


def run_program(program, table):
    """The value of every step (None where a step can't be computed).

    `table` is FinQA's `table` field: a list of rows, each a list of cell strings.
    A step that uses a failed step's result also gives None.
    """
    rows = {row[0]: row[1:] for row in table}  # later rows win, as in FinQA
    results = []

    def arg_value(a):
        if a.startswith("#"):
            k = int(a[1:])
            return results[k] if k < len(results) else None
        return parse_value(a)

    for op, a1, a2 in split_steps(program):
        value = None
        if op in ARITHMETIC:
            x, y = arg_value(a1), arg_value(a2)
            if x is not None and y is not None:
                try:
                    value = float(ARITHMETIC[op](x, y))
                except (ZeroDivisionError, OverflowError):
                    value = None
        elif op in TABLE:
            cells = rows.get(a1)
            nums = _row_values(cells) if cells is not None else None
            value = float(TABLE[op](nums)) if nums else None
        results.append(value)  # "greater" and unknown ops give None
    return results
