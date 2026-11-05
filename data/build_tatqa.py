"""R1: build the TAT-QA test subset, in the same format as A1 (README 8, R1).

Run from the repo root (needs the model's tokenizer, e.g. on Colab):
    python data/build_tatqa.py

Steps:
  1. Download TAT-QA train/dev/test_gold (pinned commit) into data/raw/tatqa/ if missing.
  2. Keep the arithmetic questions with a numeric answer (common/tatqa.py).
  3. Drop prompts longer than max_context_tokens (same prompt and count as A1).
  4. Sample tatqa_n_questions with the config seed. All go to split "test":
     we only test on TAT-QA, after training on FinQA.

Output: data/tatqa_subset.jsonl with the fields of data/finqa_subset.jsonl plus
gold_scale. Then run A3/A4/A7/B1-B2 on it with the --subset / --*-dir flags
(MANUAL_STEPS.md, step 12) and detector/transfer.py.
"""

import argparse
import json
import random
import statistics
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `common` imports when run as a script

from common.prompt import encode  # noqa: E402
from common.tatqa import convert  # noqa: E402

TATQA_COMMIT = "870accc41953dcde885aabeb963d94aabdc0fbc3"
TATQA_URL = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{commit}/dataset_raw/{name}.json"
TATQA_FILES = ["tatqa_dataset_train", "tatqa_dataset_dev", "tatqa_dataset_test_gold"]  # the ones with answers


def download(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in TATQA_FILES:
        path = raw_dir / f"{name}.json"
        if path.exists():
            continue
        url = TATQA_URL.format(commit=TATQA_COMMIT, name=name)
        print(f"downloading {url}")
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(url, part)
        part.rename(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "tatqa")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "tatqa_subset.jsonl")
    ap.add_argument("--tokenizer", default=None, help="tokenizer name or local path (default: the config model)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or cfg["model"])
    n_questions = cfg["tatqa_n_questions"]

    download(args.raw_dir)
    items = []
    for name in TATQA_FILES:
        with open(args.raw_dir / f"{name}.json") as f:
            items += json.load(f)
    records, not_reproduced = convert(items)
    print(f"TAT-QA: {sum(len(it['questions']) for it in items)} questions; {len(records)} arithmetic "
          f"with a numeric answer; derivation doesn't reproduce the answer for {not_reproduced}")

    eligible = []
    for r in records:
        enc = encode(tokenizer, r["question"], r["context_text"], cfg.get("chat_template", True))
        if len(enc["input_ids"]) <= cfg["max_context_tokens"]:
            eligible.append((r, len(enc["input_ids"]), enc["ctx_end"] - enc["ctx_start"]))
    print(f"dropped {len(records) - len(eligible)} prompts over {cfg['max_context_tokens']} tokens "
          f"-> {len(eligible)} eligible")
    if len(eligible) < n_questions:
        sys.exit(f"only {len(eligible)} eligible questions, need {n_questions}")

    eligible.sort(key=lambda e: e[0]["id"])
    sample = sorted(random.Random(cfg["seed"]).sample(eligible, n_questions), key=lambda e: e[0]["id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r, _, _ in sample:
            f.write(json.dumps(r) + "\n")

    prompt_tokens = [p for _, p, _ in sample]
    context_tokens = [c for _, _, c in sample]
    print(f"\nwrote {len(sample)} questions from {len({r['report_key'] for r, _, _ in sample})} tables to {args.out}")
    print(f"prompt tokens:  mean {statistics.mean(prompt_tokens):.0f}, max {max(prompt_tokens)}")
    print(f"context tokens: mean {statistics.mean(context_tokens):.0f}, max {max(context_tokens)}")


if __name__ == "__main__":
    main()
