"""A3: generate answers for one decoding setting (README 8, step A3).

Run on a GPU (Colab) from the repo root, after A1:
    python gen/generate.py --setting greedy --limit 50   # first 50, for B (README: by Nov 12)
    python gen/generate.py --setting greedy              # the rest; resumes where it stopped
    python gen/generate.py --setting temp12

Output: gen/answers_{setting}.jsonl, one answer per line, fields
    id, setting, prompt, answer_text, prompt_len, ctx_start, ctx_end,
    n_new_tokens, hit_max_tokens

Decoding uses only the setting from config.yaml: everything else is set to
"off" (no top-k, no top-p, no repetition penalty), so the model's own
default sampling settings never leak in.

Questions are processed in a fixed shuffled order, so `--limit 50` takes
the same 50 every time and a later run continues with the rest. If the
output file already exists, ids in it are skipped (`--overwrite` restarts).
"""

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `common` imports when run as a script

from common.numbers import extract_numbers  # noqa: E402
from common.prompt import encode, encode_answer  # noqa: E402

ALLOWED_KEYS = {"do_sample", "num_beams", "temperature", "top_p", "top_k"}


# Pure functions (tested in tests/test_generate.py) -------------------------

def generation_kwargs(setting, max_new_tokens):
    """Keyword arguments for model.generate for one config.yaml decoding setting.

    Anything the setting doesn't name is switched off rather than left to the
    model's defaults: no repetition penalty, one beam, and when sampling no
    top-k, no top-p and temperature 1.
    """
    unknown = set(setting) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown decoding keys {sorted(unknown)}")
    if "do_sample" not in setting:
        raise ValueError("each decoding setting must say do_sample: true or false")
    kwargs = {"num_beams": 1, "repetition_penalty": 1.0, "max_new_tokens": max_new_tokens}
    if setting["do_sample"]:
        kwargs.update(temperature=1.0, top_k=0, top_p=1.0)
    kwargs.update(setting)
    return kwargs


def question_order(ids, seed):
    """Ids in a fixed shuffled order (sorted first, so file order doesn't matter)."""
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    return ids


def batch_seed(seed, setting, first_id):
    """A seed for one batch that doesn't depend on which batches ran before it."""
    return zlib.crc32(f"{seed}|{setting}|{first_id}".encode())


def left_pad(seqs, pad_id):
    """Left-pad token lists to one length. Returns (input_ids, attention_mask)."""
    width = max(len(s) for s in seqs)
    ids = [[pad_id] * (width - len(s)) + list(s) for s in seqs]
    mask = [[0] * (width - len(s)) + [1] * len(s) for s in seqs]
    return ids, mask


def count_new_tokens(tokens, stop_ids):
    """Tokens generated before the first stop/pad token, and whether none was reached."""
    for i, t in enumerate(tokens):
        if t in stop_ids:
            return i, False
    return len(tokens), True


def chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def read_done(path):
    """Ids already in the output file. A half-written last line (the run was
    cut off mid-write) is removed so that new lines append cleanly."""
    if not path.exists():
        return set()
    with open(path) as f:
        lines = f.read().splitlines()
    good = []
    for line in lines:
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if len(good) != len([line for line in lines if line.strip()]):
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in good)
    return {r["id"] for r in good}


# Model run -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--setting", required=True, help="a key under `decoding` in config.yaml")
    ap.add_argument("--limit", type=int, help="stop when the output file has this many answers")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true", help="start the output file from scratch")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--subset", type=Path, default=ROOT / "data" / "finqa_subset.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "gen")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.setting not in cfg["decoding"]:
        sys.exit(f"unknown setting {args.setting!r}; choose from {list(cfg['decoding'])}")
    gen_kwargs = generation_kwargs(cfg["decoding"][args.setting], cfg["max_new_tokens"])
    chat_template = cfg.get("chat_template", True)

    with open(args.subset) as f:
        questions = {r["id"]: r for r in map(json.loads, f)}
    out_path = args.out_dir / f"answers_{args.setting}.jsonl"
    if args.overwrite and out_path.exists():
        out_path.unlink()
    done = read_done(out_path)
    order = question_order(questions, cfg["seed"])
    if args.limit is not None:
        order = order[:args.limit]
    todo = [i for i in order if i not in done]
    print(f"{args.setting}: {len(done)} answers already in {out_path}, {len(todo)} to generate")
    if not todo:
        return

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    # Replace the model's default decoding settings (Qwen's: temperature 0.7,
    # top_p 0.8, top_k 20, repetition_penalty 1.05) with special tokens only.
    shipped = model.generation_config
    pad_id = shipped.pad_token_id if shipped.pad_token_id is not None else tokenizer.pad_token_id
    model.generation_config = GenerationConfig(
        bos_token_id=shipped.bos_token_id, eos_token_id=shipped.eos_token_id, pad_token_id=pad_id)
    eos = shipped.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos]) | {pad_id}
    print(f"model {cfg['model']} on {device}; generate({gen_kwargs})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    retok_problems = 0
    with open(out_path, "a") as out:
        for batch in tqdm(chunks(todo, args.batch_size), unit="batch"):
            encs = [encode(tokenizer, questions[i]["question"], questions[i]["context_text"], chat_template)
                    for i in batch]
            ids, mask = left_pad([e["input_ids"] for e in encs], pad_id)
            set_seed(batch_seed(cfg["seed"], args.setting, batch[0]))
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=torch.tensor(ids, device=device),
                    attention_mask=torch.tensor(mask, device=device),
                    **gen_kwargs,
                )
            new_tokens = generated[:, len(ids[0]):].tolist()
            for qid, enc, toks in zip(batch, encs, new_tokens):
                n_new, hit_max = count_new_tokens(toks, stop_ids)
                answer = tokenizer.decode(toks[:n_new], skip_special_tokens=True)
                try:
                    encode_answer(tokenizer, enc["text"], answer, enc["input_ids"])
                except ValueError:
                    retok_problems += 1
                out.write(json.dumps({
                    "id": qid,
                    "setting": args.setting,
                    "prompt": enc["text"],
                    "answer_text": answer,
                    "prompt_len": len(enc["input_ids"]),
                    "ctx_start": enc["ctx_start"],
                    "ctx_end": enc["ctx_end"],
                    "n_new_tokens": n_new,
                    "hit_max_tokens": hit_max,
                }) + "\n")
            out.flush()

    with open(out_path) as f:
        rows = [json.loads(line) for line in f]
    n = len(rows)
    n_nums = [len(extract_numbers(r["answer_text"])) for r in rows]
    print(f"\n{out_path}: {n} answers")
    print(f"  mean new tokens {sum(r['n_new_tokens'] for r in rows) / n:.1f}, "
          f"cut at max_new_tokens {sum(r['hit_max_tokens'] for r in rows)}")
    print(f"  numbers per answer: mean {sum(n_nums) / n:.1f}, answers with none {n_nums.count(0)}")
    print(f"  answers whose re-tokenization changes the prompt tokens (this run): {retok_problems}")


if __name__ == "__main__":
    main()
