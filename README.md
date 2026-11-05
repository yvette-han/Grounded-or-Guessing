# Grounded or Guessing?

**Detecting hallucinated numbers in financial QA from attention patterns**
CSE 354 Final Project · Poster **Dec 9** (NCS atrium) · Report **Dec 11**

A language model reads part of a company's financial report and answers a question with numbers. Some numbers are **copied** from the report, some are **computed** from it (correctly or not), and some are **made up**. We test whether the model's own attention can tell these apart.

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
| Name | _fill in_ | _fill in_ |
| Owns | Data, generation, decoding settings, labeling, confidence baseline, Stage 1 results | Attention extraction, features, detectors, layer/head analysis |
| Folders | `data/`, `gen/`, `labels/`, `baseline/` | `features/`, `detector/`, `analysis/` |
| Presents at poster | Dataset, labeling, decoding results | Attention method, detector results, analysis |

**Both:** `config.yaml`, the shared definitions in §6, the label hand-check, the report and the poster. **Each person must be able to explain the other person's half.**

---

## 3. AI use policy

The professor's rule (Lecture 1): AI use is **allowed for the coding**, but *"the analysis and the knowledge that you produce must be your own."*

What that means for us:

- ✅ **Fine:** AI writes or debugs code, explains a library, writes plotting boilerplate, suggests a regex.
- ✅ **Required:** test every AI-written function on a small example where you already know the answer before you trust it (e.g. the number parser on 20 hand-written strings).
- ✅ **Required:** each of us can explain every file in our folders line by line. If you can't explain it, rewrite it until you can.
- ❌ **Not fine:** AI writes the analysis, interprets the results, or chooses the conclusions in the report or poster. We write those ourselves from our own results.
- 📝 **Keep `LOG.md`:** one line per decision or dead end ("Nov 14: switched tolerance to 1% because 'about 12%' was being labeled fabricated"). This is also what we show the professor at meetings.

---

## 4. Setup

**Compute:** Google Colab with a GPU (T4 is enough for a 1.5B model). Use the same notebook runtime type for both of us.

```bash
git clone <repo-url> && cd grounded-or-guessing
pip install torch transformers accelerate datasets numpy scipy scikit-learn pandas matplotlib pyyaml tqdm
```

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (fallback: `meta-llama/Llama-3.2-3B-Instruct`). Read the layer and head counts from `model.config.num_hidden_layers` and `model.config.num_attention_heads`; don't hard-code them.

**Data:** FinQA (Chen et al., 2021), from the official GitHub repo (`dataset/train.json`, `dev.json`, `test.json`). Each example has the text before the table (`pre_text`), the `table`, the text after it (`post_text`), and a `qa` block with the question, the gold `program` (calculation steps) and the executed answer. Check the exact field names when you first load it.

**`config.yaml`** (one shared file; every script reads from it):

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
```

---

## 5. Repo layout

```
config.yaml
LOG.md                  decisions and dead ends, dated
data/                   A: build_subset.py  → finqa_subset.jsonl
gen/                    A: generate.py      → answers_{setting}.jsonl
labels/                 A: label.py         → numbers_{setting}.jsonl, handcheck.csv
baseline/               A: confidence.py    → confidence_{setting}.npz
features/               B: extract.py       → {setting}.npz
detector/               B: train_eval.py    → results.csv
analysis/               B (+A): notebooks and figures for the poster
common/                 both: numbers.py (parse/normalize), prompt.py, splits.py, metrics.py
```

`common/` is shared code. Change it only after telling the other person, because both stages depend on it.

---

## 6. Shared definitions (read before writing code)

### 6.1 Prompt template

The prompt asks the model to **show its work**, so each answer contains copied input numbers *and* a computed or fabricated result. Without this, almost every number would be a single final answer and there would be no "copied" class.

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

Tables become text one row per line, cells joined with ` | ` (e.g. `revenue | 2009 | 4,215`).

### 6.2 Train/val/test split: by report

FinQA ids look like `ADI/2009/page_49.pdf-1`. The **report key** is company + year (`ADI/2009`). All questions from one report go in the same split. This prevents the detector from memorizing a report it will be tested on. The split is computed once in `common/splits.py` and saved in `finqa_subset.jsonl`.

### 6.3 Number labels (check in this order)

| Order | Label | Rule |
|---|---|---|
| 1 | **copied** | The value appears in the report (within `label_tolerance` after normalization) |
| 2 | **computed** | Not in the report, but matches the gold answer or an intermediate step of the gold `program` |
| 3 | **wrong_calc** | Not in the report and not correct, but every input number the answer listed is **copied**. The model looked in the right place and did the math wrong |
| 4 | **fabricated** | Anything else |

The order matters: a gold answer that happens to equal a report number is **copied**.

**Normalization** (in `common/numbers.py`): handle `%`, `$`, commas, negatives and parentheses `(12)` = −12, and scale words ("4.2 billion" = 4,200 million). Compare `12%` with `0.12` as equal. **Skip** years (1990–2030 written without a decimal point) and list markers like `1.` at the start of a line.

### 6.4 "Context" for attention features

- **Context tokens** = the report text only, between `### Report` and `### Question`. The instruction and the question count as neither context nor generated text.
- **Generated tokens** = answer tokens before the current token.
- If a number token has **no** generated tokens before it, skip it for the lookback ratio (it would be undefined).

### 6.5 Positive class for detection

Main task: **fabricated = 1**, copied + computed = 0. `wrong_calc` is analyzed separately (§8, step B5) and left out of the main training set.

---

## 7. File contracts

Both stages read and write these exact formats. If one changes, update this table **and** `LOG.md`.

| File | Writer | One row/record = | Fields |
|---|---|---|---|
| `data/finqa_subset.jsonl` | A | question | `id, report_key, split, question, context_text, gold_answer, gold_program` |
| `gen/answers_{setting}.jsonl` | A | generated answer | `id, setting, prompt, answer_text, prompt_len, ctx_start, ctx_end` (token indices of the report span) |
| `labels/numbers_{setting}.jsonl` | A | number in an answer | `id, setting, number_text, value, tok_start, tok_end, label` (token indices **within the answer**) |
| `baseline/confidence_{setting}.npz` | A | number | `mean_logprob, min_logprob, mean_entropy, label, id` |
| `features/{setting}.npz` | B | number | `lookback[L×H], hf_ctx[L×H], hf_gen[L×H], label, id` |

Token indices come from the **same tokenizer and the same prompt string**. Always tokenize with `common/prompt.py`.

---

## 8. Pipeline: step by step

Each step lists what to run, what it produces, and how to know it's done. Steps marked 🔁 feed the other person, so ship a small version early.

### Stage 1 (Person A)

**A1. Build the subset** → `python data/build_subset.py`
- Load FinQA, convert tables to text, drop examples whose prompt exceeds `max_context_tokens`, sample `n_questions`, assign splits by report.
- ✅ Done when `finqa_subset.jsonl` exists, and `LOG.md` records the count, the average context length and the split sizes.

**A2. Number parser** → `common/numbers.py` + `tests/test_numbers.py`
- Write 30 hand-made test strings first (`"$4.2 billion"`, `"(12.5)"`, `"12%"`, `"in 2009"`…), then the parser.
- ✅ Done when all tests pass.

**A3. Generate** → `python gen/generate.py --setting greedy` (repeat for each setting)
- Fixed seed, `max_new_tokens` from config. Save `ctx_start`/`ctx_end`.
- 🔁 **Run greedy on 50 questions first and send them to B by Nov 12.**
- ✅ Done when every setting has 1,000 answers.

**A4. Label** → `python labels/label.py --setting greedy`
- Apply §6.3 in order. Print the count of each label per setting.
- ✅ Done when label counts are in `LOG.md`.

**A5. Hand-check labels (both, Nov 12–14)** → `labels/handcheck.csv`
- Each person labels the same 50 numbers by hand without looking at the auto label; compare. Report the agreement rate.
- If agreement is under ~90%, fix the rules in §6.3 **before** anything else runs.

**A6. Stage 1 results (RQ0)** → `analysis/stage1.ipynb`
- Per setting: numeric accuracy vs gold, fabrication rate, distinct answers across settings, answer length. One table + one bar chart.

**A7. Confidence baseline** → `python baseline/confidence.py`
- Teacher-forced pass (no attention needed): for each number span, the mean and min log-probability and the mean entropy of the next-token distribution.
- ✅ Done when `confidence_{setting}.npz` exists for all settings.

### Stage 2 (Person B)

**B0. Meeting plot (before the Nov 2–11 meeting)**
- One FinQA example: load the model with `attn_implementation="eager"`, run one forward pass with `output_attentions=True`, plot one number token's attention over the report tokens for a few heads.

**B1. Attention extraction** → `features/extract.py`
- **Teacher forcing:** one forward pass over `prompt + answer_text` gives the attention from every answer token. No generation needed here.
- Load in float16, batch size 1. Immediately keep only the **rows for number tokens** and discard the rest.
- Start with `output_attentions=True`. If you run out of GPU memory on long reports, switch to forward hooks on each attention layer that compute the features and drop the weights.
- ✅ Done when features for the first 20 examples look sane: attention rows sum to ~1, and lookback values lie between 0 and 1.

**B2. Features** (per head, per number token, averaged over the number's tokens)
- **Lookback ratio** = mean attention on context tokens ÷ (mean on context + mean on previously generated tokens).
- **High-frequency energy:** take the attention over context positions, divide by its sum (so report length and total mass don't matter), apply `numpy.fft.rfft`, zero the coefficients below the cutoff, invert, take the L2 norm. Do the same for attention over generated tokens. Compute for each cutoff in `hf_cutoffs`.
- ✅ Done when `features/{setting}.npz` exists for every setting.

**B3. Detectors** → `python detector/train_eval.py`
- Pool all settings, using the same report split. Standardize features (fit on train only). Logistic regression with L2; tune `C` and the HF cutoff on **val**.
- Compare five detectors: lookback only, HF only, lookback + HF, confidence baseline (from A7), random.
- Metrics on **test**: AUROC (main), F1 at the val-tuned threshold. Add 95% confidence intervals by bootstrap over test **reports**.
- ✅ Done when `results.csv` has one row per detector.

**B4. Analysis** → `analysis/stage2.ipynb`
- **RQ2:** AUROC for copied-vs-fabricated and for computed-vs-fabricated, separately. What fraction of correct computed numbers get flagged?
- **RQ3:** AUROC per layer; rank heads by coefficient size; does a top-100-head detector keep most of the AUROC?
- **Bridge:** mean lookback ratio per decoding setting. Does it fall as temperature rises?

**B5. Error analysis (both)**
- 10 false positives (correct numbers flagged), 10 false negatives (fabrications missed), and where `wrong_calc` numbers fall on the detector's score. Write what we think explains each pattern, in our own words.

### Both

**R1. Stretch:** TAT-QA in the same format; train on FinQA, test on TAT-QA.
**R2. Poster:** dataset + labeling (A), decoding chart (A), attention diagram (B), AUROC table (B), head/layer heatmap (B).
**R3. Report:** each person writes their own sections; we both edit the full draft.

---

## 9. Timeline

| Dates | Person A (Stage 1) | Person B (Stage 2) | Checkpoint |
|---|---|---|---|
| Nov 2–8 | A1, A2 | B0, B1 on hand-made prompts | Meeting with Niranjan (bring B0 plot + A1 stats) |
| Nov 9–15 | A3, A4 · **50 greedy examples to B by Nov 12** | B1, B2 on the first batch | **A5 hand-check Nov 12–14** |
| Nov 16–22 | A6, A7 | B2 on all settings, B3 | Stage 1 finished = safe floor |
| Nov 23–29 | Stage 1 error analysis | B4 (Thanksgiving week: lighter) | Main results table frozen |
| Nov 30–Dec 6 | R1 prep, report sections | B5, R1, report sections | Poster draft by Dec 6 |
| Dec 7–8 | Rehearse | Rehearse | Each explains the other's half |
| **Dec 9 / Dec 11** | | | **Poster / report** |

**Syncs:** 15 minutes twice a week. Update `LOG.md` and the results table before each one.

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
