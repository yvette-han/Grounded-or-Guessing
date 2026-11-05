"""A4: label every number in one setting's answers (README 8, step A4; rules in 6.3).

Run from the repo root after A3 (needs the model's tokenizer, e.g. on Colab):
    python labels/label.py --setting greedy
    python labels/label.py --setting greedy --handcheck 50   # also writes the A5 hand-check sheet

Labels, checked in this order:
  copied      the value appears in the report
  computed    not in the report, but equals the gold answer or a step of the gold program
  question    not in the report or the gold program, but appears in the question
              (not in README 6.3; see LOG.md)
  wrong_calc  none of the above, but every number before it in the answer is
              copied, computed or question, and at least one is copied: the
              model used the right inputs and got the math wrong
  fabricated  anything else: the first number of an answer that matches
              nothing, or any number after a wrong_calc or fabricated one

Output: labels/numbers_{setting}.jsonl, one number per line:
    id, setting, number_text, value, char_start, char_end, tok_start, tok_end, label
  char_start, char_end  the whole mention ("$4.2 billion") in answer_text
  tok_start, tok_end    half-open token span of just its digits ("4.2"), counted
                        from the first answer token (README 7: "within the answer")

With --handcheck N: labels/handcheck.csv (N random numbers for both of us to
label by hand, auto labels hidden) and labels/handcheck_key.csv (the auto labels).
An existing handcheck.csv is never overwritten.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `common` imports when run as a script

from common.numbers import extract_numbers, find_match  # noqa: E402
from common.prompt import encode_answer, tokens_in_span  # noqa: E402

LABELS = ["copied", "computed", "question", "wrong_calc", "fabricated"]
GOOD_INPUTS = {"copied", "computed", "question"}  # what may come before a wrong_calc


def label_numbers(numbers, report_numbers, gold_values, question_numbers, tol):
    """Labels for one answer's numbers, in answer order (rules in the module docstring)."""
    labels = []
    for n in numbers:
        if find_match(n, report_numbers, tol) is not None:
            label = "copied"
        elif find_match(n, gold_values, tol) is not None:
            label = "computed"
        elif find_match(n, question_numbers, tol) is not None:
            label = "question"
        elif "copied" in labels and all(x in GOOD_INPUTS for x in labels):
            label = "wrong_calc"
        else:
            label = "fabricated"
        labels.append(label)
    return labels


def gold_values(question):
    """The gold answer and every computable step of the gold program."""
    return [question["gold_answer"]] + [v for v in question["gold_steps"] if v is not None]


def label_answer(answer_row, question, tokenizer, tol):
    """One record per number in this answer. Raises ValueError if the
    answer's tokens can't be lined up with the prompt (see encode_answer)."""
    answer = answer_row["answer_text"]
    full = encode_answer(tokenizer, answer_row["prompt"], answer)
    if full["answer_start"] != answer_row["prompt_len"]:
        raise ValueError("prompt length differs from gen/: was the tokenizer or config changed?")

    numbers = extract_numbers(answer)
    labels = label_numbers(numbers, extract_numbers(question["context_text"]), gold_values(question),
                           extract_numbers(question["question"]), tol)
    records = []
    for n, label in zip(numbers, labels):
        c0 = full["answer_char_start"]
        toks = tokens_in_span(full["offsets"], c0 + n.num_start, c0 + n.num_end)
        records.append({
            "id": answer_row["id"],
            "setting": answer_row["setting"],
            "number_text": n.text,
            "value": n.value,
            "char_start": n.start,
            "char_end": n.end,
            "tok_start": toks[0] - full["answer_start"],
            "tok_end": toks[-1] + 1 - full["answer_start"],
            "label": label,
        })
    return records


def handcheck_rows(records, answers, questions, n, seed):
    """(sheet, key): n random numbers to label by hand, and their auto labels."""
    sample = random.Random(seed).sample(records, min(n, len(records)))
    sheet, key = [], []
    for k, r in enumerate(sample, 1):
        a = answers[r["id"]]["answer_text"]
        q = questions[r["id"]]
        marked = a[:r["char_start"]] + "⟦" + a[r["char_start"]:r["char_end"]] + "⟧" + a[r["char_end"]:]
        sheet.append({
            "row": k, "id": r["id"], "number": r["number_text"],
            "answer (number marked ⟦ ⟧)": marked,
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "gold_program": q["gold_program"],
            "gold_steps": ", ".join("n/a" if v is None else f"{v:g}" for v in q["gold_steps"]),
            "report": q["context_text"],
            "label_A": "", "label_B": "",
        })
        key.append({"row": k, "id": r["id"], "number": r["number_text"], "auto_label": r["label"]})
    return sheet, key


def write_csv(path, rows):
    # utf-8-sig so Excel shows the ⟦ ⟧ marks correctly; Google Sheets reads it too
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--setting", required=True)
    ap.add_argument("--handcheck", type=int, metavar="N", help="also write an N-number hand-check sheet")
    ap.add_argument("--tokenizer", help="tokenizer name or path (default: the config model)")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--subset", type=Path, default=ROOT / "data" / "finqa_subset.jsonl")
    ap.add_argument("--answers-dir", type=Path, default=ROOT / "gen")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "labels")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tol = cfg["label_tolerance"]
    with open(args.subset) as f:
        questions = {r["id"]: r for r in map(json.loads, f)}
    with open(args.answers_dir / f"answers_{args.setting}.jsonl") as f:
        answers = {r["id"]: r for r in map(json.loads, f)}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or cfg["model"])

    records, skipped = [], []
    for qid, a in answers.items():
        try:
            records += label_answer(a, questions[qid], tokenizer, tol)
        except ValueError as err:
            skipped.append((qid, str(err)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"numbers_{args.setting}.jsonl"
    with open(out_path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)

    counts = Counter(r["label"] for r in records)
    with_numbers = len({r["id"] for r in records})
    print(f"{args.setting}: {len(answers)} answers, {with_numbers} with numbers, "
          f"{len(records)} numbers -> {out_path}")
    for label in LABELS:
        print(f"  {label:<11} {counts[label]:>6}  {counts[label] / max(1, len(records)):6.1%}")
    if skipped:
        print(f"  skipped {len(skipped)} answers whose tokens don't line up, e.g. {skipped[:3]}")

    if args.handcheck:
        sheet_path, key_path = args.out_dir / "handcheck.csv", args.out_dir / "handcheck_key.csv"
        if sheet_path.exists():
            sys.exit(f"{sheet_path} already exists (it may hold hand labels); move it away first")
        sheet, key = handcheck_rows(records, answers, questions, args.handcheck, cfg["seed"])
        write_csv(sheet_path, sheet)
        write_csv(key_path, key)
        print(f"hand-check sheet: {sheet_path} ({len(sheet)} numbers); auto labels: {key_path}")


if __name__ == "__main__":
    main()
