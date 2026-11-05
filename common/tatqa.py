"""TAT-QA in the same format as data/finqa_subset.jsonl (README 8, R1).

TAT-QA (Zhu et al., 2021): each item is one table plus paragraphs from a
report, with several questions. We keep the `arithmetic` questions: their
answer is a number, with a `scale` (percent, thousand, million, billion or
none) and a `derivation`, a formula such as "(44.1-56.7)/56.7".

Conversion to the FinQA subset fields:
  id            question uid
  report_key    "tatqa/" + table uid (questions about one table stay together)
  split         "test" (we only test on TAT-QA)
  context_text  the table (one row per line, cells joined by " | "), then the paragraphs
  gold_answer   the answer as written (scale not applied: 4.2 with scale "million")
  gold_program  the derivation
  gold_steps    the value of every operation in the derivation (derivation_steps)
  gold_scale    TAT-QA's scale (extra field)
"""

import ast
import operator
import re

from common.prompt import build_context

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _clean(derivation, percent_as_fraction, paren_negative):
    """The derivation as a Python arithmetic expression."""
    s = derivation.replace("$", "").replace("×", "*").replace("÷", "/")
    s = s.replace("[", "(").replace("]", ")")                    # [ ] are used as brackets too
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)                      # 1,027 -> 1027
    if paren_negative:                                           # accounting: (6919) -> -6919
        s = re.sub(r"\(\s*(\d+(?:\.\d+)?)\s*\)", r"(-\1)", s)
    if percent_as_fraction:                                      # 11% -> 0.11
        s = re.sub(r"(\d+(?:\.\d+)?)\s*%", lambda m: repr(float(m.group(1)) / 100), s)
    else:                                                        # 11% -> 11
        s = s.replace("%", "")
    return s.strip()


def evaluate(derivation, percent_as_fraction=False, paren_negative=False):
    """Values of every +, -, *, / in the derivation, innermost first, then the
    value of the whole expression if it isn't the last of those. [] if it
    can't be evaluated (not plain arithmetic, or division by zero)."""
    try:
        tree = ast.parse(_clean(derivation, percent_as_fraction, paren_negative), mode="eval")
    except SyntaxError:
        return []
    steps = []

    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = walk(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            v = _OPS[type(node.op)](walk(node.left), walk(node.right))
            steps.append(float(v))
            return v
        raise ValueError("not plain arithmetic")

    try:
        value = float(walk(tree.body))
    except (ValueError, ZeroDivisionError, TypeError):
        return []
    if not steps or steps[-1] != value:
        steps.append(value)
    return steps


def _decimals(x):
    """Decimal places of a number as written: 0.39 -> 2, 14491 -> 0."""
    text = repr(float(x))
    return 0 if text.endswith(".0") or "e" in text else len(text.split(".")[1])


def _close(value, answer, tol):
    """Within relative tol, or equal once rounded like the answer (answers are rounded)."""
    if abs(value - answer) <= tol * max(abs(value), abs(answer)):
        return True
    return abs(round(value, _decimals(answer)) - answer) < 1e-9


def matches_answer(value, answer, scale, tol=0.01):
    """The derivation's value equals the answer (a percent answer may be value x 100)."""
    return _close(value, answer, tol) or (scale == "percent" and _close(100 * value, answer, tol))


def derivation_steps(derivation, answer, scale, tol=0.01):
    """(steps, ok): the derivation's step values under the first reading that
    reproduces the answer. Readings: "11%" as 11 or as 0.11, and a lone number
    in parentheses "(6,919)" as 6919 or as -6919 (accounting). ok is False if
    none does; then the plain reading's steps are returned ([] if unparseable)."""
    readings = [evaluate(derivation, pct, neg) for neg in (False, True) for pct in (False, True)]
    for steps in readings:
        if steps and matches_answer(steps[-1], answer, scale, tol):
            return steps, True
    return readings[0], False


def convert(items):
    """(records, n_not_reproduced): one record per arithmetic question with a numeric answer."""
    records, not_reproduced = [], 0
    for item in items:
        table = item["table"]["table"]
        paragraphs = [p["text"] for p in sorted(item["paragraphs"], key=lambda p: p["order"])]
        context = build_context([], table, paragraphs)
        for q in item["questions"]:
            if q.get("answer_type") != "arithmetic":
                continue
            try:
                answer = float(q["answer"])
            except (TypeError, ValueError):
                continue
            steps, ok = derivation_steps(q["derivation"], answer, q["scale"])
            not_reproduced += not ok
            records.append({
                "id": q["uid"],
                "report_key": f"tatqa/{item['table']['uid']}",
                "split": "test",
                "question": q["question"].strip(),
                "context_text": context,
                "gold_answer": answer,
                "gold_program": q["derivation"].strip(),
                "gold_steps": steps,
                "gold_scale": q["scale"],
            })
    return records, not_reproduced
