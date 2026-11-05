"""A1: build the FinQA question subset (README 8, step A1).

Run from the repo root:
    python data/build_subset.py

Steps:
  1. Download FinQA train/dev/test (pinned commit) into data/raw/ if missing.
  2. Pool all three files. FinQA's own splits share reports, so we re-split.
  3. Drop yes/no questions (no numeric gold answer).
  4. Build the report text and prompt; drop prompts longer than
     max_context_tokens (counted on the full model input).
  5. Sample n_questions with the config seed.
  6. Assign train/val/test by report (common/splits.py).
  7. Execute each gold program (common/program.py) and keep every step's value.

Output: data/finqa_subset.jsonl, one question per line, fields
    id, report_key, split, question, context_text, gold_answer, gold_program, gold_steps
"""

import argparse
import json
import random
import statistics
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `common` imports when run as a script

from common.program import run_program  # noqa: E402
from common.prompt import build_context, encode  # noqa: E402
from common.splits import assign_splits, report_key  # noqa: E402

FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/{commit}/dataset/{name}.json"
FINQA_FILES = ["train", "dev", "test"]


def download_finqa(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FINQA_FILES:
        path = raw_dir / f"{name}.json"
        if path.exists():
            continue
        url = FINQA_URL.format(commit=FINQA_COMMIT, name=name)
        print(f"downloading {url}")
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(url, part)
        part.rename(path)


def load_finqa(raw_dir):
    examples = []
    for name in FINQA_FILES:
        with open(raw_dir / f"{name}.json") as f:
            examples += json.load(f)
    return examples


def is_numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def summary(values):
    return (f"mean {statistics.mean(values):.0f}, median {statistics.median(values):.0f}, "
            f"min {min(values)}, max {max(values)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "finqa_subset.jsonl")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer name or local path (default: the config model)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or cfg["model"])
    chat_template = cfg.get("chat_template", True)
    max_tokens, n_questions, seed = cfg["max_context_tokens"], cfg["n_questions"], cfg["seed"]

    download_finqa(args.raw_dir)
    examples = load_finqa(args.raw_dir)
    print(f"FinQA train+dev+test: {len(examples)} questions, "
          f"{len({report_key(e['id']) for e in examples})} reports")

    numeric = [e for e in examples if is_numeric(e["qa"]["exe_ans"])]
    print(f"dropped {len(examples) - len(numeric)} yes/no questions -> {len(numeric)}")

    eligible = []
    for e in numeric:
        context = build_context(e["pre_text"], e["table"], e["post_text"])
        enc = encode(tokenizer, e["qa"]["question"], context, chat_template)
        n_prompt = len(enc["input_ids"])
        if n_prompt > max_tokens:
            continue
        eligible.append({
            "id": e["id"],
            "report_key": report_key(e["id"]),
            "question": e["qa"]["question"].strip(),
            "context_text": context,
            "gold_answer": float(e["qa"]["exe_ans"]),
            "gold_program": e["qa"]["program"],
            "gold_steps": run_program(e["qa"]["program"], e["table"]),
            "_prompt_tokens": n_prompt,
            "_context_tokens": enc["ctx_end"] - enc["ctx_start"],
        })
    print(f"dropped {len(numeric) - len(eligible)} prompts over {max_tokens} tokens "
          f"-> {len(eligible)} eligible")
    if len(eligible) < n_questions:
        sys.exit(f"only {len(eligible)} eligible questions, need {n_questions}")

    eligible.sort(key=lambda r: r["id"])
    sample = random.Random(seed).sample(eligible, n_questions)
    splits = assign_splits([r["id"] for r in sample], cfg["split"], seed)
    for r in sample:
        r["split"] = splits[r["id"]]
    sample.sort(key=lambda r: r["id"])

    off = [r["id"] for r in sample if r["gold_steps"][-1] is None
           or abs(r["gold_steps"][-1] - r["gold_answer"]) > 1e-4 * max(1.0, abs(r["gold_answer"]))]
    print(f"gold programs whose last step differs from FinQA's answer: {len(off)} {off[:5]}")

    fields = ["id", "report_key", "split", "question", "context_text",
              "gold_answer", "gold_program", "gold_steps"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in sample:
            f.write(json.dumps({k: r[k] for k in fields}) + "\n")

    print(f"\nwrote {len(sample)} questions to {args.out}")
    print(f"tokenizer: {args.tokenizer or cfg['model']}, chat_template: {chat_template}")
    print(f"prompt tokens:  {summary([r['_prompt_tokens'] for r in sample])}")
    print(f"context tokens: {summary([r['_context_tokens'] for r in sample])}")
    questions = Counter(r["split"] for r in sample)
    report_split = {r["report_key"]: r["split"] for r in sample}  # one split per report
    reports = Counter(report_split.values())
    for name in cfg["split"]:
        print(f"{name:>5}: {questions[name]} questions, {reports[name]} reports")


if __name__ == "__main__":
    main()
