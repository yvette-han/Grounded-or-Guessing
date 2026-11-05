# Grounded or Guessing?

**Detecting hallucinated numbers in financial QA from attention patterns**
CSE 354 Final Project · Poster **Dec 9** (NCS atrium) · Report **Dec 11**

A language model reads part of a company's financial report and answers a question with numbers. Some numbers are **copied** from the report, some are **computed** from it (correctly or not), and some are **made up**. We test whether the model's own attention can tell these apart.

**Status (Nov 5):** the code for every step (A1–A7, B0–B5, R1) is written and passes its tests (`python -m pytest -q`). Nothing has run on real data yet. Next: the Colab runs in [`MANUAL_STEPS.md`](MANUAL_STEPS.md), starting with A1 and the 50 greedy answers due Nov 12.

---

## Contents

1. [Research questions](#1-research-questions)
2. [Team and ownership](#2-team-and-ownership)
3. [AI use policy](#3-ai-use-policy)
4. [Setup](#4-setup)
5. [Repo layout](#5-repo-layout)
6. [Shared definitions (read before writing code)](#6-shared-definitions-read-before-writing-code)
7. [File contracts](#7-file-contracts)
8. [Pipeline: step by step](#8-pipeline-step-by-step)
9. [Timeline](#9-timeline)
10. [Known pitfalls](#10-known-pitfalls)
11. [References](#11-references)

---

## 1. Research questions

| # | Question | Who answers it |
|---|---|---|
| RQ0 | How does the decoding strategy (greedy, beam, temperature, top-p) change numeric accuracy and the fabrication rate? | Stage 1 (works as a standalone project) |
| RQ1 | Do attention features (lookback ratio, high-frequency energy) detect fabricated numbers better than the model's own confidence? | Stage 2 |
| RQ2 | Do the detectors confuse **computed** numbers with **fabricated** ones? | Stage 2 (our main contribution) |
| RQ3 | Which layers and heads carry the signal? | Stage 2 |
| Stretch | Does a detector trained on FinQA transfer to TAT-QA? Does the lookback ratio fall as temperature rises? | Both |

**Hypothesis for RQ2:** attention detectors mostly measure "is the model copying from the document?" They should separate copied from fabricated numbers well, but may wrongly flag correct computed numbers.

---

## 2. Team and ownership

| | Person A: **Stage 1** | Person B: **Stage 2** |
|---|---|---|
| Name | Eduardo | Yvette |
| Owns | Data, generation, decoding settings, labeling, confidence baseline, Stage 1 results | Attention extraction, features, detectors, layer/head analysis |
| Folders | `data/`, `gen/`, `labels/`, `baseline/`, `analysis/stage1.*` | `features/`, `detector/`, `analysis/b0_attention_plot.py`, `analysis/stage2.*`, `analysis/errors.*` |
| Presents at poster | Dataset, labeling, decoding results | Attention method, detector results, analysis |

**Both:** `config.yaml`, `common/`, `tests/`, the shared definitions in §6, the label hand-check, the report and the poster. **Each person must be able to explain the other's half.**

**GPU runs are split by decoding setting** so we can run at the same time: Eduardo runs `greedy` and `beam5`; Yvette runs `temp07`, `temp12` and `topp09`. Each runs generation, labeling, confidence and features for their own settings. Checklists: `MANUAL_STEPS.md`.

---

## 3. AI use policy

Professor's rule: AI use is **allowed for the coding**, but *"the analysis and the knowledge that you produce must be your own."*

What that means for us:

- ✅ **Fine:** AI writes or debugs code, explains a library, writes plotting boilerplate, suggests a regex.
- ✅ **Required:** before trusting an AI-written function, test it on a small example where you already know the answer (e.g. the number parser on 20 hand-written strings).
- ✅ **Required:** each of us can explain every file in our folders line by line. If you can't explain it, rewrite it until you can.
- ❌ **Not fine:** AI writes the analysis, interprets the results, or chooses the conclusions in the report or poster. We write those ourselves from our own results.
- 📝 **`LOG.md`:** one dated line per decision or dead end ("Nov 14: switched tolerance to 1% because 'about 12%' was being labeled fabricated"). We also show it to professor at meetings.
- 🧭 **`MANUAL_STEPS.md`:** what each of us runs and does by hand, step by step.
- 📋 **`ANALYSIS_TODO.md`:** the analysis that is ours to do, the choices in AI-written code to review, and the files we must be able to explain.

---

## 4. Setup

**Compute:** Google Colab with a GPU (a T4 is enough for a 1.5B model). Both of us use the same runtime type.

```bash
git clone https://github.com/yvette-han/Grounded-or-Guessing && cd Grounded-or-Guessing
pip install -r requirements.txt
```

**On Colab:** open Colab with a GPU runtime and run these in one session. Put `!` before shell lines and use `%cd` to change folders.

```bash
git clone -b main https://github.com/yvette-han/Grounded-or-Guessing
cd Grounded-or-Guessing && pip install -r requirements.txt && python -m pytest -q
python data/build_subset.py
python analysis/b0_attention_plot.py
```

For everything else (who runs which settings, in which order, and how we share files), follow [`MANUAL_STEPS.md`](MANUAL_STEPS.md).

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (fallback: `meta-llama/Llama-3.2-3B-Instruct`). Read the layer and head counts from `model.config.num_hidden_layers` and `model.config.num_attention_heads`; don't hard-code them.

**Data:** FinQA (Chen et al., 2021), from the official GitHub repo (`dataset/train.json`, `dev.json`, `test.json`). Each example has the text before the table (`pre_text`), the `table`, the text after it (`post_text`), and a `qa` block with the question, the gold `program` (calculation steps) and the executed answer.

**Data for R1 (stretch):** TAT-QA (Zhu et al., 2021), from its official GitHub repo. Both datasets are pinned to one commit and downloaded into `data/raw/`, which is not in git.

**`config.yaml`** (one shared file; every script reads it):

```yaml
model: Qwen/Qwen2.5-1.5B-Instruct
seed: 354
n_questions: 1000
max_context_tokens: 1500
max_new_tokens: 128
split: {train: 0.70, val: 0.15, test: 0.15}     # split BY REPORT, see §6
decoding:
  greedy:  {do_sample: false}
  beam5:   {do_sample: false, num_beams: 5}
  temp07:  {do_sample: true, temperature: 0.7}
  temp12:  {do_sample: true, temperature: 1.2}
  topp09:  {do_sample: true, temperature: 1.0, top_p: 0.9}
label_tolerance: 0.01        # relative; 4.2 matches 4.21
hf_cutoffs: [0.25, 0.35, 0.45]   # normalized frequency; tune on val
chat_template: true          # send the 6.1 prompt as the user turn of the model's chat template
tatqa_n_questions: 500     # R1 stretch: TAT-QA test questions (data/build_tatqa.py)
```

---

## 5. Repo layout

```
config.yaml             shared settings; every script reads it
requirements.txt        pip install -r requirements.txt
LOG.md                  decisions and dead ends, dated; run notes per person at the bottom
ANALYSIS_TODO.md        analysis, reviews and explanations that are ours to do (§3)
MANUAL_STEPS.md         who runs what, in which order: Colab checklists for A and B
CLAUDE.md               notes for Claude (AI-use rule, keeping the trackers current)
common/                 both: prompt.py     prompt + token spans (§6.1, §6.4)
                              numbers.py    find and compare numbers (§6.3)
                              splits.py     split by report (§6.2)
                              program.py    run FinQA gold programs
                              tatqa.py      TAT-QA conversion (R1)
                              io.py         file helpers
data/                   A: build_subset.py  → finqa_subset.jsonl
                        R1: build_tatqa.py  → tatqa_subset.jsonl
gen/                    A: generate.py      → answers_{setting}.jsonl
labels/                 A: label.py         → numbers_{setting}.jsonl, handcheck.csv
                           agreement.py     → hand-check agreement (A5)
baseline/               A: confidence.py    → confidence_{setting}.npz
features/               B: attention.py     lookback and HF energy
                           extract.py       → {setting}.npz (kept in our shared Drive folder)
detector/               B: train_eval.py    → results.csv, test_scores.csv
                        R1: transfer.py     → transfer_results.csv, tatqa_scores.csv
analysis/               A: stage1.py + stage1.ipynb (A6)
                        B: b0_attention_plot.py (B0), stage2.py + stage2.ipynb (B4), errors.py + errors.ipynb (B5)
                           figures/ for the poster and report; notes/ for our B5 write-ups
tatqa/                  R1: gen/, labels/, baseline/ for TAT-QA (same scripts, folder flags)
tests/                  unit tests for every module: python -m pytest -q
```

`common/` is shared code. Tell the other person before changing it: both stages depend on it.

---

## 6. Shared definitions (read before writing code)

### 6.1 Prompt template

The prompt asks the model to **show its work**, so each answer contains copied input numbers *and* a computed or fabricated result. Without this, almost every number would be a single final answer, and there would be no "copied" class.

```
You are a financial analyst. Use only the report below.

### Report
{pre_text}
{table_as_text}
{post_text}

### Question
{question}

List the numbers from the report that you use, then give the final answer.
Answer:
```

- Tables become text one row per line, cells joined with ` | ` (e.g. `revenue | 2009 | 4,215`).
- Each `pre_text`/`post_text` sentence goes on its own line; FinQA's lines that are only `.` are dropped.
- With `chat_template: true`, this text is the user turn of the model's chat template, and the answer starts right after the assistant header. `common/prompt.py` (`encode`) builds the exact model input and finds the report's token span.

### 6.2 Train/val/test split: by report

FinQA ids look like `ADI/2009/page_49.pdf-1`. The **report key** is company + year (`ADI/2009`). All questions from one report go in the same split, so the detector can't memorize a report it will be tested on. `common/splits.py` computes the split once, and it is saved in `finqa_subset.jsonl`.

### 6.3 Number labels (check in this order)

| Order | Label | Rule |
|---|---|---|
| 1 | **copied** | The value appears in the report (within `label_tolerance` after normalization) |
| 2 | **computed** | Not in the report, but matches the gold answer or an intermediate step of the gold `program` |
| 3 | **question** | Not in the report or the gold program, but appears in the question (added Nov 5, see `LOG.md`) |
| 4 | **wrong_calc** | Not correct, but the inputs were right: the model looked in the right place and did the math wrong. In code: every number before it in the answer is copied, computed or question, and at least one is copied |
| 5 | **fabricated** | Anything else |

The order matters: a gold answer that happens to equal a report number is **copied**.

**Normalization** (`common/numbers.py`):

- Handle `%`, `$`, commas, negatives, parentheses (`(12)` = −12) and scale words ("4.2 billion" = 4,200 million).
- `12%` equals `0.12`.
- **Skip** years (1990–2030 written without a decimal point), list markers like `1.` at the start of a line, the day in dates (`december 31`), ordinals (`3rd`) and digits inside words (`Q4`). Matching rules: see `LOG.md`.

### 6.4 "Context" for attention features

- **Context tokens** = the report text only, between `### Report` and `### Question`. The instruction and the question count as neither context nor generated text.
- **Generated tokens** = answer tokens before the current number token *t*. With `--row predicting`, we read row *t − 1*, and its own token counts as generated (as in Lookback Lens). Code: `features/attention.py`.
- If a number token has **no** generated tokens before it, skip it for the lookback ratio (it would be undefined).

### 6.5 Positive class for detection

Main task: **fabricated = 1**, copied + computed = 0. `wrong_calc` and `question` are left out of the main training set; `wrong_calc` is analyzed in §8, step B5.

---

## 7. File contracts

Both stages read and write these exact formats. If one changes, update this table **and** `LOG.md`.

| File | Writer | One row/record = | Fields |
|---|---|---|---|
| `data/finqa_subset.jsonl` | A | question | `id, report_key, split, question, context_text, gold_answer, gold_program, gold_steps` (each program step's value; `null` if it can't be computed) |
| `gen/answers_{setting}.jsonl` | A | generated answer | `id, setting, prompt, answer_text, prompt_len, ctx_start, ctx_end, n_new_tokens, hit_max_tokens` (`ctx_*`: the report's token span, half-open: `ids[ctx_start:ctx_end]`) |
| `labels/numbers_{setting}.jsonl` | A | number in an answer | `id, setting, number_text, value, char_start, char_end, tok_start, tok_end, label` (`tok_*`: the digits' tokens **within the answer**, half-open; `char_*`: the whole mention in `answer_text`) |
| `baseline/confidence_{setting}.npz` | A | number | `mean_logprob, min_logprob, mean_entropy, label, id, row` (`row`: line in `labels/numbers_{setting}.jsonl`, the join key) |
| `features/{setting}.npz` | B | number | `lookback[L×H], hf_ctx[C×L×H], hf_gen[C×L×H], hf_cutoffs, label, id, row` (C = one per `hf_cutoffs` value, float16; same `row` as the confidence file). Kept in our shared Drive folder, not git |
| `detector/results.csv` | B | detector | `detector, auroc, auroc_ci_low, auroc_ci_high, f1, f1_ci_low, f1_ci_high, threshold, val_auroc, C, hf_cutoff, n_features, n_train, n_val, n_test, n_test_fabricated` |
| `detector/test_scores.csv` | B | test number (every label) | `setting, row, id, report, label`, then per detector its score and `{detector}_flagged` (score ≥ val-tuned threshold) |
| `data/tatqa_subset.jsonl` | R1 | TAT-QA question | the `finqa_subset.jsonl` fields (`gold_program` = TAT-QA's derivation, `split` = test) plus `gold_scale` |
| `detector/transfer_results.csv` | R1 | detector | `detector, finqa_test_auroc, tatqa_auroc (+ CI), tatqa_f1 (+ CI), threshold, C, hf_cutoff, n_tatqa, n_tatqa_fabricated` |
| `detector/tatqa_scores.csv` | R1 | TAT-QA number (every label) | same columns as `test_scores.csv` |

All token indices come from the **same tokenizer and the same prompt string**. Always tokenize with `common/prompt.py`.

---

## 8. Pipeline: step by step

What to run, what it produces, and when it's done. 🔁 marks a step the other person needs, so ship a small version early.

### Stage 1 (Person A: Eduardo)

**A1. Build the subset** → `python data/build_subset.py`
- Load FinQA, convert tables to text, drop examples whose prompt exceeds `max_context_tokens`, sample `n_questions`, assign splits by report.
- ✅ Done when `finqa_subset.jsonl` exists and `LOG.md` has the count, the average context length and the split sizes.

**A2. Number parser** → `common/numbers.py` + `tests/test_numbers.py`
- Write 30 hand-made test strings first (`"$4.2 billion"`, `"(12.5)"`, `"12%"`, `"in 2009"`…), then the parser.
- ✅ Done when all tests pass.

**A3. Generate** → `python gen/generate.py --setting greedy` (repeat for each setting)
- Fixed seed; `max_new_tokens` from config; save `ctx_start`/`ctx_end`.
- Only the parameters in the setting apply. Everything else is off (no top-k, top-p or repetition penalty), so the model's default sampling settings don't leak in.
- 🔁 **Run greedy on 50 questions first and send them to B by Nov 12** (`--limit 50`; a later run without `--limit` continues with the rest).
- ✅ Done when every setting has 1,000 answers.

**A4. Label** → `python labels/label.py --setting greedy`
- Apply §6.3 in order, and print the count of each label per setting.
- `--handcheck 50` also writes `labels/handcheck.csv` for A5 (auto labels kept apart in `handcheck_key.csv`).
- ✅ Done when the label counts are in `LOG.md`.

**A5. Hand-check labels (both, Nov 12–14)** → `labels/handcheck.csv`
- Each person labels the same 50 numbers by hand without seeing the auto labels; then compare and report the agreement rate.
- Each fills `label_A` or `label_B` in their own copy; `python labels/agreement.py --sheet-a A.csv --sheet-b B.csv` prints the agreement rates and writes `labels/handcheck_disagreements.csv`.
- If agreement is under ~90%, fix the rules in §6.3 **before** anything else runs.

**A6. Stage 1 results (RQ0)** → `analysis/stage1.ipynb`
- Per setting: numeric accuracy vs gold, fabrication rate, distinct answers across settings, answer length. One table and one bar chart.
- The notebook calls `analysis/stage1.py`, which writes `analysis/figures/stage1_table.csv`, `stage1_table.md` and `stage1_bars.png`. Definitions are at the top of that file.

**A7. Confidence baseline** → `python baseline/confidence.py`
- One teacher-forced pass (no attention needed). For each number's digit tokens: the mean and min log-probability and the mean entropy of the next-token distribution.
- Scored under the model's own distribution (temperature 1, no top-k/top-p) for every setting.
- ✅ Done when `confidence_{setting}.npz` exists for all settings.

### Stage 2 (Person B: Yvette)

**B0. Meeting plot (before the Nov 2–11 meeting)** → `python analysis/b0_attention_plot.py`
- One FinQA example: load the model with `attn_implementation="eager"`, run one forward pass with `output_attentions=True`, and plot one number token's attention over the report tokens for a few heads.

**B1. Attention extraction** → `features/extract.py`
- **Teacher forcing:** one forward pass over `prompt + answer_text` gives the attention from every answer token. No generation needed.
- Load in float16, batch size 1. Keep only the **rows for number tokens** and discard the rest right away.
- Start with `output_attentions=True`. If long reports run out of GPU memory, switch to forward hooks on each attention layer that compute the features and drop the weights.
- ✅ Done when the features for the first 20 examples look sane: attention rows sum to ~1, and lookback values lie between 0 and 1. `python features/extract.py --limit 20` prints both.

**B2. Features** (per head, per number token, averaged over the number's tokens)
- **Lookback ratio** = mean attention on context tokens ÷ (mean on context + mean on previously generated tokens).
- **High-frequency energy:** take the attention over context positions, divide by its sum (so report length and total mass don't matter), apply `numpy.fft.rfft`, zero the coefficients below the cutoff, invert, and take the L2 norm. Do the same for the attention over generated tokens, for each cutoff in `hf_cutoffs`.
- Frequencies are in `numpy.fft.rfftfreq` units (cycles per token, 0 to 0.5), so cutoffs 0.25 / 0.35 / 0.45 keep the top 50% / 30% / 10% of the range. Computed in the same pass as B1.
- ✅ Done when `features/{setting}.npz` exists for every setting.

**B3. Detectors** → `python detector/train_eval.py`
- Pool all settings, with the same report split. Standardize the features (fit on train only). Logistic regression with L2; tune `C` and the HF cutoff on **val**.
- Compare five detectors: lookback only, HF only, lookback + HF, confidence baseline (from A7), random.
- Metrics on **test**: AUROC (main) and F1 at the val-tuned threshold, with 95% confidence intervals by bootstrap over test **reports**.
- Numbers with a NaN feature are dropped for every detector, so all five see the same numbers. `test_scores.csv` keeps every test number's scores, including `wrong_calc` and `question`, for B4 and B5.
- ✅ Done when `results.csv` has one row per detector.

**B4. Analysis** → `analysis/stage2.ipynb`
- **RQ2:** AUROC for copied-vs-fabricated and for computed-vs-fabricated, separately. What fraction of correct computed numbers get flagged?
- **RQ3:** AUROC per layer; rank heads by coefficient size; does a top-100-head detector keep most of the AUROC?
- **Bridge:** mean lookback ratio per decoding setting. Does it fall as temperature rises?
- The notebook calls `analysis/stage2.py`, which writes the `analysis/figures/stage2_*` tables and figures. Definitions are at the top of that file.

**B5. Error analysis (both)**
- 10 false positives (correct numbers flagged), 10 false negatives (fabrications missed), and where `wrong_calc` numbers fall on the detector's score. Write what we think explains each pattern, in our own words.
- `analysis/errors.ipynb` (or `python analysis/errors.py`) picks the examples and writes `analysis/figures/b5_*`. Each example shows the question, the gold answer and steps, the answer with the number marked, the labels of the answer's other numbers, and the report lines with that value.

### Both

**R1. Stretch:** TAT-QA in the same format; train on FinQA, test on TAT-QA.
- `python data/build_tatqa.py` → `data/tatqa_subset.jsonl`: TAT-QA's arithmetic questions (table first, then paragraphs), `tatqa_n_questions` sampled, all as test. `common/tatqa.py` evaluates each derivation into `gold_steps`.
- Then the A3, A4, A7 and B1–B2 scripts with `--subset data/tatqa_subset.jsonl` and `tatqa/...` folders, and `python detector/transfer.py`, which fits the detectors on FinQA exactly as in B3 and scores TAT-QA. Steps: `MANUAL_STEPS.md`, section "R1 (optional)".

**R2. Poster:** dataset + labeling (A), decoding chart (A), attention diagram (B), AUROC table (B), head/layer heatmap (B).

**R3. Report:** each person writes their own sections; we both edit the full draft.

---

## 9. Timeline

| Dates | Eduardo (A, Stage 1) | Yvette (B, Stage 2) | Checkpoint |
|---|---|---|---|
| Nov 2–8 | A1 | B0, B1 on hand-made prompts | Meeting with professor (bring B0 plot + A1 stats) |
| Nov 9–15 | A3, A4 · **50 greedy examples to B by Nov 12** | B1, B2 on the first batch | **A5 hand-check Nov 12–14** |
| Nov 16–22 | A6, A7 | B2 on all settings, B3 | Stage 1 finished = safe floor |
| Nov 23–29 | Stage 1 error analysis | B4 (Thanksgiving week) | Main results table frozen |
| Nov 30–Dec 6 | R1 prep, report sections | B5, R1, report sections | Poster draft by Dec 6 |
| Dec 7–8 | Rehearse | Rehearse | Each explains the other's half |
| **Dec 9 / Dec 11** | | | **Poster / report** |

**Syncs:** 15 minutes twice a week. Update `LOG.md` and the results table before each one.

Both of us run GPU steps in parallel (§2). The day-by-day plan with hand-offs is in [`MANUAL_STEPS.md`](MANUAL_STEPS.md#timeline).

---

## 10. Known pitfalls

| Pitfall | Prevention |
|---|---|
| Few fabricated numbers under greedy | Pool all decoding settings for training; temperature 1.2 produces more |
| Detector memorizes reports | Split by report (§6.2), and fit the feature scaler on train only |
| Token indices drift between files | One tokenizer, one prompt builder (`common/prompt.py`) |
| A year like "2009" counted as a number | Skip rule in §6.3 |
| Long reports raise high-frequency energy automatically | Normalize each attention row before the FFT (B2) |
| Wrong labels ruin everything downstream | Hand-check early (A5), before features are computed at scale |
| AI-written code with a silent bug | Unit tests on known examples (§3) |

---

## 11. References

- Qi et al. (2026). *Detecting Contextual Hallucinations in LLMs with Frequency-Aware Attention.* arXiv:2602.18145
- Chuang et al. (2024). *Lookback Lens: Detecting and Mitigating Contextual Hallucinations in LLMs Using Only Attention Maps.* EMNLP 2024
- Chen et al. (2021). *FinQA: A Dataset of Numerical Reasoning over Financial Data.* EMNLP 2021
- Zhu et al. (2021). *TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance.* ACL 2021
