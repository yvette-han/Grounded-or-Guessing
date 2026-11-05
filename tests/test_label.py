"""Tests for labels/label.py. Expected labels are worked out by hand from README 6.3."""

import pytest

from common.numbers import extract_numbers
from common.prompt import build_context, encode, encode_answer
from labels.label import handcheck_rows, label_answer, label_numbers

CONTEXT = build_context(["revenue grew this year ."],
                        [["", "2009", "2008"], ["revenue", "4,215", "3,900"]], [])
QUESTION = {
    "id": "X/2009/page_1.pdf-1",
    "question": "what was the growth in revenue if costs rose 7% ?",
    "context_text": CONTEXT,
    "gold_answer": 0.0807692,              # (4215 - 3900) / 3900
    "gold_program": "subtract(4215, 3900), divide(#0, 3900)",
    "gold_steps": [315.0, 0.0807692],
}
TOL = 0.01


def labels_for(answer, context=CONTEXT):
    gold = [QUESTION["gold_answer"]] + QUESTION["gold_steps"]
    return label_numbers(extract_numbers(answer), extract_numbers(context), gold,
                         extract_numbers(QUESTION["question"]), TOL)


@pytest.mark.parametrize("answer, expected", [
    ("Numbers used: 4,215 and 3,900. Change: 315. Growth: 8.1%.",
     ["copied", "copied", "computed", "computed"]),
    ("4,215 and 3,900. Growth: 9.5%.", ["copied", "copied", "wrong_calc"]),
    ("Growth was 12%.", ["fabricated"]),                       # nothing before it
    ("Revenue 5,000 and growth 9.5%.", ["fabricated", "fabricated"]),
    ("Costs rose 7%, revenue 4,215, so growth 20%.", ["question", "copied", "wrong_calc"]),
    ("4,215 then 999 then 20%.", ["copied", "wrong_calc", "fabricated"]),  # 20% builds on a wrong number
    ("4,215, 3,900, 315, then 7.5%.", ["copied", "copied", "computed", "wrong_calc"]),
    ("In 2009 revenue was 4,215.", ["copied"]),                 # the year is skipped
])
def test_label_numbers(answer, expected):
    assert labels_for(answer) == expected


def test_copied_wins_over_computed():
    # 315 is a gold step, but if the report also has it, it is copied (README 6.3).
    context = CONTEXT + "\nthe change was 315 ."
    assert labels_for("Change: 315.", context) == ["copied"]


def _answer_row(tokenizer, answer):
    enc = encode(tokenizer, QUESTION["question"], CONTEXT)
    return {"id": QUESTION["id"], "setting": "greedy", "prompt": enc["text"],
            "prompt_len": len(enc["input_ids"]), "answer_text": answer}


def test_label_answer_spans(toy_tokenizer):
    answer = "Numbers used: $4,215 million and 3,900. Growth: 8.1%."
    row = _answer_row(toy_tokenizer, answer)
    records = label_answer(row, QUESTION, toy_tokenizer, TOL)
    assert [r["number_text"] for r in records] == ["$4,215 million", "3,900", "8.1%"]
    assert [r["label"] for r in records] == ["copied", "copied", "computed"]

    full = encode_answer(toy_tokenizer, row["prompt"], answer)
    answer_ids = full["input_ids"][full["answer_start"]:]
    digits = ["4,215", "3,900", "8.1"]
    for r, d in zip(records, digits):
        assert answer[r["char_start"]:r["char_end"]] == r["number_text"]
        assert 0 <= r["tok_start"] < r["tok_end"] <= len(answer_ids)
        assert d in toy_tokenizer.decode(answer_ids[r["tok_start"]:r["tok_end"]])


def test_label_answer_rejects_changed_prompt(toy_tokenizer):
    row = _answer_row(toy_tokenizer, "4,215")
    row["prompt_len"] += 1
    with pytest.raises(ValueError):
        label_answer(row, QUESTION, toy_tokenizer, TOL)


def test_handcheck_hides_auto_labels(toy_tokenizer):
    answer = "Growth: 9.5%."
    row = _answer_row(toy_tokenizer, answer)
    records = label_answer(row, QUESTION, toy_tokenizer, TOL)
    sheet, key = handcheck_rows(records, {row["id"]: row}, {QUESTION["id"]: QUESTION}, 5, seed=354)
    assert len(sheet) == len(key) == 1
    assert sheet[0]["answer (number marked ⟦ ⟧)"] == "Growth: ⟦9.5%⟧."
    assert not any("label" in k and k not in ("label_A", "label_B") for k in sheet[0])
    assert key[0]["auto_label"] == "fabricated"
