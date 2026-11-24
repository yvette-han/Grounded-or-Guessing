"""R2 (stretch): add a hint to a question the model hallucinated on, rerun, and
compare attention before and after on the same instance (README 8, R2).

Step 1, no GPU: list candidate answers that contain a hallucinated number
(fabricated or wrong_calc):
    python analysis/hint_test.py candidates --setting greedy -n 20
  writes hints/candidates.csv. Copy it to hints/hints.csv and, for the rows you
  want, write a `hint` by hand: a sentence appended to the question that points
  to the right place, e.g. "Use the row 'adjusted consolidated cash flow'."
  `focus_row` is the hallucinated number to follow: the first one that isn't
  cut off at the end of the answer and isn't the value 100 (mostly the "x 100"
  of a percent formula; --keep-100 keeps it). Change it if you prefer another.

Step 2, on a GPU (Colab), after B1-B2 and B4:
    python analysis/hint_test.py run
  For each row with a hint:
  1. question + " " + hint -> the README 6.1 prompt. The report comes before the
     question, so its tokens and positions are identical to the original prompt
     (checked), and attention over the report can be compared position by position.
  2. Generate one answer with the same decoding setting (seed from the id). Use
     greedy: with a sampling setting, a different answer can also come from the
     random draw, not only from the hint.
  3. Label its numbers with the A4 rules (the hint is part of the question, so
     its numbers count as question numbers).
  4. One teacher-forced pass with attention for the original answer and one for
     the new one: lookback per number (as in B1-B2), and the attention row of
     each number's first token over the report.
  5. Counterpart: the number in the new answer that takes the focus number's
     place, chosen as the one whose preceding text on its line (up to 40
     characters) looks most like the focus number's (difflib ratio). Check it in hint_results.csv; to choose
     another, put its index (0 = first number of the new answer) in
     `counterpart_index` and run again.

Outputs:
  hints/answers.jsonl       the new answers (fields as gen/answers_*.jsonl, plus k, hint, question)
  hints/numbers.jsonl       their labeled numbers (fields as labels/numbers_*.jsonl, plus k)
  hints/pair_{k}.npz        attention rows over the report (focus, counterpart) and
                            per-number lookback, before and after
  hints/hint_results.csv    one row per hint: labels, accuracy, counterpart, lookback before/after
  analysis/figures/hint_{k}.png      attention over the report at one head, before vs after,
                                     and mean lookback per layer
  analysis/figures/hint_summary.png  mean lookback before -> after for every pair
The head in the plots: --head L,H, or by default the top lookback head of B4
(analysis/figures/stage2_heads.csv), the same head for every pair.
"""

import argparse
import csv
import difflib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.stage1 import is_correct  # noqa: E402
from common.io import check_model, group_records, load_config, read_jsonl, run_dir  # noqa: E402
from common.numbers import extract_numbers  # noqa: E402
from common.prompt import encode, encode_answer, tokens_in_span  # noqa: E402
from detector.train_eval import TASKS  # noqa: E402

HALLUCINATED = TASKS["hallucinated"][0]
CONTEXT_CHARS = 40
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
CANDIDATE_FIELDS = ["k", "setting", "id", "split", "focus_row", "focus_number", "focus_label", "question",
                    "gold_answer", "gold_program", "gold_input_lines", "answer", "hint", "counterpart_index"]


# Pure functions (tested in tests/test_hint_test.py) ------------------------

def hinted_question(question, hint):
    """The question with the hint appended."""
    if not hint or not hint.strip():
        raise ValueError("empty hint")
    return f"{question.strip()} {hint.strip()}"


def gold_program_args(program):
    """Numbers written in a FinQA program, e.g. divide(-14611, 739187) -> [-14611, 739187].
    Step references (#0) and constants (const_100) are not included."""
    return [float(x) for x in re.findall(r"(?<![#\w.])-?\d+(?:\.\d+)?", program)]


def gold_input_lines(context, program):
    """Report lines that contain one of the gold program's numbers (exactly)."""
    args = [abs(a) for a in gold_program_args(program)]
    return [line for line in context.split("\n")
            if any(abs(abs(n.value) - a) < 1e-9 for n in extract_numbers(line) for a in args)]


def is_cut_tail(record, answer):
    """The last number of an answer cut at max_new_tokens (often a fragment)."""
    return answer["hit_max_tokens"] and record["char_end"] >= len(answer["answer_text"].rstrip()) - 2


def focus_number(recs, answer, skip_values=(100.0,)):
    """(row, record) of the first hallucinated number that isn't a cut-off tail
    or one of skip_values, or None."""
    for row, r in recs:
        if r["label"] in HALLUCINATED and not is_cut_tail(r, answer) and r["value"] not in skip_values:
            return row, r
    return None


def pick_candidates(records, answers, questions, n, seed=0, split=None, skip_values=(100.0,)):
    """Up to n seeded-random answers with a hallucinated number, as candidates.csv rows."""
    pool = []
    for qid, recs in group_records(records).items():
        if split and questions[qid]["split"] != split:
            continue
        focus = focus_number(recs, answers[qid], skip_values)
        if focus is not None:
            pool.append((qid, focus))
    rows = []
    for k, (qid, (row, r)) in enumerate(random.Random(seed).sample(pool, min(n, len(pool))), 1):
        q, a = questions[qid], answers[qid]["answer_text"]
        rows.append({
            "k": k, "setting": r["setting"], "id": qid, "split": q["split"], "focus_row": row,
            "focus_number": r["number_text"], "focus_label": r["label"], "question": q["question"],
            "gold_answer": q["gold_answer"], "gold_program": q["gold_program"],
            "gold_input_lines": " / ".join(gold_input_lines(q["context_text"], q["gold_program"])),
            "answer": a[:r["char_start"]] + "⟦" + a[r["char_start"]:r["char_end"]] + "⟧" + a[r["char_end"]:],
            "hint": "", "counterpart_index": "",
        })
    return rows


def left_context(text, start, width=CONTEXT_CHARS):
    """Up to `width` characters before position start, within the same line; the
    plain window if the number (nearly) starts its line."""
    line_start = text.rfind("\n", 0, start) + 1
    same_line = text[max(line_start, start - width):start]
    return same_line if len(same_line.strip()) >= 3 else text[max(0, start - width):start]


def find_counterpart(before_text, before_start, after_text, after_records):
    """(index, similarity) of the after-answer number whose preceding text looks most
    like the focus number's; (None, 0.0) if the new answer has no numbers."""
    target = left_context(before_text, before_start).lower()
    best = (None, 0.0)
    for i, r in enumerate(after_records):
        ratio = difflib.SequenceMatcher(None, target, left_context(after_text, r["char_start"]).lower()).ratio()
        if ratio > best[1]:
            best = (i, ratio)
    return best


def head_from_csv(path):
    """The top lookback head (layer, head) in analysis/figures/stage2_heads.csv."""
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["feature_set"] == "lookback":
                return int(r["layer"]), int(r["head"])
    raise ValueError(f"no lookback heads in {path}")


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # Excel shows ⟦ ⟧ correctly
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# Model run -----------------------------------------------------------------

def attention_for(model, tokenizer, prompt_text, answer_text, records, row_mode, device):
    """Teacher-forced pass over prompt + answer. Returns (rows_att [layers, heads, rows, seq],
    each number's indices into rows, each number's token positions, answer_start), as in B1."""
    import torch
    from features.capture import attention_rows
    from features.extract import number_positions, rows_needed

    full = encode_answer(tokenizer, prompt_text, answer_text)
    s0 = full["answer_start"]
    positions = [number_positions(s0, r["tok_start"], r["tok_end"]) for r in records]
    rows, index = rows_needed(positions, row_mode)
    rows_att = attention_rows(model, torch.tensor([full["input_ids"]], device=device), rows)
    return rows_att, index, positions, s0


def run_hints(cfg, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from features.extract import answer_features
    from gen.generate import batch_seed, count_new_tokens, generation_kwargs, plain_generation_config
    from labels.label import label_answer

    hints = [h for h in read_csv(args.hints) if h.get("hint", "").strip()]
    if not hints:
        sys.exit(f"no rows with a hint in {args.hints}")
    head = tuple(int(x) for x in args.head.split(",")) if args.head else head_from_csv(args.heads_csv)
    tol, cutoffs = cfg["label_tolerance"], cfg["hf_cutoffs"]
    questions = {q["id"]: q for q in read_jsonl(args.subset)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=getattr(torch, args.dtype),  # float16 overflows into NaN attention
        attn_implementation="eager").to(device).eval()
    pad_id, stop_ids = plain_generation_config(model, tokenizer)
    print(f"{cfg['model']} on {device}; {len(hints)} hints; plot head layer {head[0]} head {head[1]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    results, new_answers, new_numbers = [], [], []
    for h in hints:
        k, setting, qid = int(h["k"]), h["setting"], h["id"]
        if setting not in cache:
            answers_path = args.answers_dir / f"answers_{setting}.jsonl"
            setting_answers = read_jsonl(answers_path)
            check_model(setting_answers, cfg["model"], answers_path)
            cache[setting] = (read_jsonl(args.labels_dir / f"numbers_{setting}.jsonl"),
                              {a["id"]: a for a in setting_answers})
        records, answers = cache[setting]
        q, before = questions[qid], answers[qid]
        before_recs = [r for r in records if r["id"] == qid]
        before_rows = [i for i, r in enumerate(records) if r["id"] == qid]
        focus_i = before_rows.index(int(h["focus_row"]))
        focus = before_recs[focus_i]

        # 1-2. hinted prompt (same report tokens) and one new answer
        hq = hinted_question(q["question"], h["hint"])
        enc = encode(tokenizer, hq, q["context_text"], cfg.get("chat_template", True))
        old = tokenizer(before["prompt"], add_special_tokens=False)["input_ids"]
        same_report = (enc["ctx_start"] == before["ctx_start"] and enc["ctx_end"] == before["ctx_end"]
                       and enc["input_ids"][enc["ctx_start"]:enc["ctx_end"]] == old[before["ctx_start"]:before["ctx_end"]])
        if not same_report:
            print(f"  hint {k}: the report's tokens changed; skipped")
            continue
        gen_kwargs = generation_kwargs(cfg["decoding"][setting], cfg["max_new_tokens"])
        set_seed(batch_seed(cfg["seed"], f"{setting}+hint", qid))
        with torch.inference_mode():
            out = model.generate(input_ids=torch.tensor([enc["input_ids"]], device=device),
                                 attention_mask=torch.ones(1, len(enc["input_ids"]), dtype=torch.long, device=device),
                                 **gen_kwargs)
        toks = out[0, len(enc["input_ids"]):].tolist()
        n_new, hit_max = count_new_tokens(toks, stop_ids)
        after_text = tokenizer.decode(toks[:n_new], skip_special_tokens=True)
        after = {"k": k, "id": qid, "setting": setting, "model": cfg["model"], "hint": h["hint"], "question": hq,
                 "prompt": enc["text"],
                 "answer_text": after_text, "prompt_len": len(enc["input_ids"]), "ctx_start": enc["ctx_start"],
                 "ctx_end": enc["ctx_end"], "n_new_tokens": n_new, "hit_max_tokens": hit_max}
        # 3. labels with the A4 rules (one misaligned answer is skipped, as in labels/label.py)
        try:
            after_recs = label_answer(after, dict(q, question=hq), tokenizer, tol)
        except ValueError as err:
            print(f"  hint {k}: could not label the new answer ({err}); skipped")
            continue
        new_answers.append(after)
        new_numbers += [dict(r, k=k) for r in after_recs]

        # 4. attention, before and after
        ctx = slice(before["ctx_start"], before["ctx_end"])
        feats, first_rows = {}, {}
        for name, prompt, text, recs in (("before", before["prompt"], before["answer_text"], before_recs),
                                         ("after", enc["text"], after_text, after_recs)):
            if not recs:
                feats[name], first_rows[name] = [], np.zeros((0,) + model_shape(model) + (ctx.stop - ctx.start,))
                continue
            rows_att, index, positions, s0 = attention_for(model, tokenizer, prompt, text, recs, args.row, device)
            feats[name] = answer_features(rows_att, index, positions, ctx, s0, cutoffs)
            first_rows[name] = np.stack([rows_att[:, :, idx[0], ctx] for idx in index])

        # 5. counterpart
        if h.get("counterpart_index", "").strip():
            c_i, ratio = int(h["counterpart_index"]), None
        else:
            c_i, ratio = find_counterpart(before["answer_text"], focus["char_start"], after_text, after_recs)
        lb_before = np.stack([f["lookback"] for f in feats["before"]])
        lb_after = np.stack([f["lookback"] for f in feats["after"]]) if feats["after"] else np.zeros((0,) + lb_before.shape[1:])
        np.savez_compressed(args.out_dir / f"pair_{k}.npz", focus_row=first_rows["before"][focus_i],
                            counterpart_row=(first_rows["after"][c_i] if c_i is not None else np.array([])),
                            lookback_before=lb_before, lookback_after=lb_after,
                            labels_before=np.array([r["label"] for r in before_recs]),
                            labels_after=np.array([r["label"] for r in after_recs]),
                            focus_index=focus_i, counterpart_index=-1 if c_i is None else c_i)

        cp = after_recs[c_i] if c_i is not None else None
        results.append({
            "k": k, "setting": setting, "id": qid, "hint": h["hint"],
            "focus_number": focus["number_text"], "focus_label": focus["label"],
            "counterpart_index": "" if c_i is None else c_i,
            "counterpart_number": cp["number_text"] if cp else "", "counterpart_label": cp["label"] if cp else "",
            "counterpart_similarity": "" if ratio is None else round(ratio, 3),
            "counterpart_context": left_context(after_text, cp["char_start"]) if cp else "",
            "n_hallucinated_before": sum(r["label"] in HALLUCINATED for r in before_recs),
            "n_hallucinated_after": sum(r["label"] in HALLUCINATED for r in after_recs),
            "correct_before": int(is_correct(before["answer_text"], q["gold_answer"], tol)),
            "correct_after": int(is_correct(after_text, q["gold_answer"], tol)),
            "lookback_before": float(np.nanmean(lb_before[focus_i])),
            "lookback_after": float(np.nanmean(lb_after[c_i])) if cp else "",
            "lookback_head_before": float(lb_before[focus_i][head]),
            "lookback_head_after": float(lb_after[c_i][head]) if cp else "",
            "answer_after": after_text,
        })
        plot_pair(args.fig_dir / f"hint_{k}.png", results[-1], first_rows["before"][focus_i],
                  first_rows["after"][c_i] if cp else None, lb_before[focus_i], lb_after[c_i] if cp else None,
                  head, report_marks(tokenizer, before, q))
        print(f"  hint {k} ({qid}): {focus['label']} {focus['number_text']!r} -> "
              f"{cp['label'] + ' ' + repr(cp['number_text']) if cp else 'no number'}")

    with open(args.out_dir / "answers.jsonl", "w") as f:
        f.writelines(json.dumps(a) + "\n" for a in new_answers)
    with open(args.out_dir / "numbers.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in new_numbers)
    if results:
        write_csv(args.out_dir / "hint_results.csv", results)
        plot_summary(args.fig_dir / "hint_summary.png", results)
    print(f"wrote {len(results)} pairs to {args.out_dir / 'hint_results.csv'} and figures to {args.fig_dir}")


def model_shape(model):
    return model.config.num_hidden_layers, model.config.num_attention_heads


def report_marks(tokenizer, answer, question):
    """Report-token positions (from the report's first token) of the gold program's numbers."""
    args = [abs(a) for a in gold_program_args(question["gold_program"])]
    text = answer["prompt"]
    c0 = text.index(question["context_text"])
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    marks = set()
    for n in extract_numbers(question["context_text"]):
        if any(abs(abs(n.value) - a) < 1e-9 for a in args):
            marks |= {t - answer["ctx_start"] for t in tokens_in_span(offsets, c0 + n.num_start, c0 + n.num_end)}
    return sorted(marks)


# Plots ---------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def _spans(ix):
    out = []
    for i in ix:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def plot_pair(path, res, row_before, row_after, lb_before, lb_after, head, marks):
    """Left: attention over the report at one head, before vs after. Right: mean lookback per layer."""
    import matplotlib.pyplot as plt
    layer, h = head
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.8), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [2.4, 1]})
    for ax in (ax1, ax2):
        _style(ax)
    for j, (a0, a1) in enumerate(_spans(marks)):
        ax1.axvspan(a0 - 0.5, a1 + 0.5, color=MUTED, alpha=0.3, linewidth=0, zorder=1,
                    label="a gold-program input" if j == 0 else None)
    before_name = f"before: {res['focus_label']} “{res['focus_number']}”"
    ax1.plot(row_before[layer, h], color=BLUE, linewidth=1.3, label=before_name, zorder=3)
    if row_after is not None:
        after_name = f"after: {res['counterpart_label']} “{res['counterpart_number']}”"
        ax1.plot(row_after[layer, h], color=ORANGE, linewidth=1.3, label=after_name, zorder=2)
    ax1.set_xlabel("report token position", fontsize=8, color=INK_2)
    ax1.set_ylabel(f"attention (layer {layer}, head {h})", fontsize=8, color=INK_2)
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)

    layers = np.arange(lb_before.shape[0])
    ax2.plot(layers, np.nanmean(lb_before, axis=1), color=BLUE, linewidth=2, marker="o", markersize=3)
    if lb_after is not None:
        ax2.plot(layers, np.nanmean(lb_after, axis=1), color=ORANGE, linewidth=2, marker="o", markersize=3)
    ax2.set_xlabel("layer", fontsize=8, color=INK_2)
    ax2.set_ylabel("lookback (mean over heads)", fontsize=8, color=INK_2)
    ax2.set_ylim(0, 1)
    fig.suptitle(f"Hint {res['k']} · {res['id']} ({res['setting']}) · hint: “{res['hint']}”",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_summary(path, results):
    """Mean lookback of the focus number (before) and its counterpart (after), one line per pair."""
    import matplotlib.pyplot as plt
    rows = [r for r in results if r["lookback_after"] != ""]
    fig, ax = plt.subplots(figsize=(4.8, 4), facecolor=SURFACE)
    _style(ax)
    for r in rows:
        still = r["counterpart_label"] in HALLUCINATED
        ax.plot([0, 1], [r["lookback_before"], r["lookback_after"]], color=BLUE if still else ORANGE,
                linewidth=1.5, marker="o", markersize=4, alpha=0.8)
    ax.plot([], [], color=ORANGE, label="after: not hallucinated")
    ax.plot([], [], color=BLUE, label="after: still hallucinated")
    ax.set_xticks([0, 1], ["before (focus number)", "after (counterpart)"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("lookback (mean over layers and heads)", fontsize=8, color=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0, 1.15), ncol=2)
    ax.set_title(f"Hint test · {len(rows)} pairs", loc="left", fontsize=9, color=INK, pad=22)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("candidates", help="list answers with a hallucinated number (no GPU)")
    c.add_argument("--setting", default="greedy")
    c.add_argument("-n", type=int, default=20)
    c.add_argument("--split", choices=["train", "val", "test"], help="only questions from this split")
    c.add_argument("--keep-100", action="store_true", help="allow the value 100 as the focus number")
    r = sub.add_parser("run", help="rerun with the hints and compare attention (GPU)")
    r.add_argument("--hints", type=Path, default=ROOT / "hints" / "hints.csv")
    r.add_argument("--head", help="layer,head for the plots (default: top lookback head of B4)")
    r.add_argument("--heads-csv", type=Path,
                   help="default: stage2_heads.csv in analysis/figures/ (or analysis/figures/<run_name>/)")
    r.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                   help="model weights dtype (default float32; float16 gives NaN attention)")
    r.add_argument("--row", choices=["number", "predicting"], default="number",
                   help="attention row mode; use the one B1 used")
    r.add_argument("--fig-dir", type=Path,
                   help="default: analysis/figures/, or analysis/figures/<run_name>/ if the config sets run_name")
    for p in (c, r):
        p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
        p.add_argument("--subset", type=Path, default=ROOT / "data" / "finqa_subset.jsonl")
        p.add_argument("--answers-dir", type=Path,
                       help="default: gen/, or gen/<run_name>/ if the config sets run_name")
        p.add_argument("--labels-dir", type=Path,
                       help="default: labels/, or labels/<run_name>/ if the config sets run_name")
        p.add_argument("--out-dir", type=Path,
                       help="default: hints/, or hints/<run_name>/ if the config sets run_name")
    args = ap.parse_args()
    cfg = load_config(args.config)
    args.answers_dir = args.answers_dir or run_dir(cfg, ROOT, "gen")
    args.labels_dir = args.labels_dir or run_dir(cfg, ROOT, "labels")
    args.out_dir = args.out_dir or run_dir(cfg, ROOT, "hints")
    if args.command == "run":
        args.fig_dir = args.fig_dir or run_dir(cfg, ROOT, "analysis/figures")
        args.heads_csv = args.heads_csv or run_dir(cfg, ROOT, "analysis/figures") / "stage2_heads.csv"

    if args.command == "candidates":
        questions = {q["id"]: q for q in read_jsonl(args.subset)}
        records = read_jsonl(args.labels_dir / f"numbers_{args.setting}.jsonl")
        answers_path = args.answers_dir / f"answers_{args.setting}.jsonl"
        answer_list = read_jsonl(answers_path)
        check_model(answer_list, cfg["model"], answers_path)
        answers = {a["id"]: a for a in answer_list}
        rows = pick_candidates(records, answers, questions, args.n, cfg["seed"], args.split,
                               () if args.keep_100 else (100.0,))
        if not rows:
            sys.exit("no answer with a hallucinated number")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.out_dir / "candidates.csv", rows, CANDIDATE_FIELDS)
        print(f"wrote {len(rows)} candidates to {args.out_dir / 'candidates.csv'}; copy it to hints.csv "
              f"and write the hints by hand")
    else:
        run_hints(cfg, args)


if __name__ == "__main__":
    main()
