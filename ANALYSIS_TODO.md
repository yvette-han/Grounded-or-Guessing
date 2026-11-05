# Analysis to do (ours, not AI's)

README §3: AI may write the code, but *"the analysis and the knowledge that you produce must be your own."* This file lists that work, plus the AI-written code we must understand. Claude adds items as it writes code; we tick them off. How to run each step: `MANUAL_STEPS.md`.

_Last updated: Nov 5, with Yvette's tasks for this week._

## This week: Yvette (Thu Nov 5 – Sun Nov 8)

Goal: the B0 plot and the A1 stats for the meeting with professor. The runs are in `MANUAL_STEPS.md`, B-0 to B-2. The items below point to sections 1–3; tick them there.

1. **The B0 plot** (section 3, row B0): after the run, write a few sentences in your own words on what the heads attend to when the model writes the number. Put them in your `LOG.md` run notes and bring them to the meeting.
2. **Be able to explain** (section 2): `analysis/b0_attention_plot.py` and `features/attention.py`, line by line. Check the hand-worked values in `tests/test_b0.py` and `tests/test_features.py`.
3. **Start these reviews** (section 1, before Nov 12):
   - **B0, B1:** the row choice (`number` or `predicting`). You decide it on Nov 12 (B-4); this week's B0 runs are a first look.
   - **B0:** which heads and which example the plot shows.
   - **B1:** how "generated tokens" are defined, and the per-number averaging.
   - **B2:** the frequency units. Check how Qi et al. (2026) normalize frequency.
4. **The A1 stats** (section 3, row A1): only if you run A1 yourself because Eduardo hasn't yet.

## 1. Choices made in AI-written code, to review

Each choice is also in `LOG.md`. Keep it, change it, or replace it; once we decide, it's our decision. **Who reviews:** A items → Eduardo, B items → Yvette, R1 → both.

**Stage 1 (Eduardo)**

- [ ] **A1:** pool FinQA train+dev+test and re-split by report (243 of FinQA's 278 test reports are also in its train file)
- [ ] **A1:** drop the 154 yes/no questions
- [ ] **A1:** drop FinQA's `"."`-only lines; one sentence per line in the report text
- [ ] **A1:** send the prompt through the chat template; `max_context_tokens` counts the full templated prompt
- [ ] **A1:** split rule: shuffle reports, then fill train → val → test by question count; a split can go over by one report
- [ ] **A1:** the subset stores `gold_steps`, each gold program step's value (`common/program.py` matches FinQA's own answer on all 8,127 questions)
- [ ] **A1 → A3, B1:** the report's token span is half-open (`ids[ctx_start:ctx_end]`); a token that straddles the report's edge counts as context
- [ ] **A2:** a number with a scale word vs a bare number: the bare one is tried in ones, thousands, millions and billions (`$4.2 billion` = table `4215`). Lenient: a few made-up numbers may count as copied
- [ ] **A2:** sign is ignored when matching (`a decrease of 12` = `(12)`); `ignore_sign=False` turns this off
- [ ] **A2:** a percent matches both readings: `12%` = `0.12` and `12%` = `12`
- [ ] **A2:** extra skip rules beyond README §6.3: the day after a month name (`december 31`), ordinals (`3rd`), digits inside words (`Q4`, `page_49`); list markers also include `1)` and `(1)`
- [ ] **A2:** `(12)` = −12 also turns FinQA footnote markers like `( 1 )` into −1
- [ ] **A2 → A4:** `tok_start`/`tok_end` cover only the digits (`4.2`); the whole mention (`$4.2 billion`) is in `char_start`/`char_end`
- [ ] **A3:** decoding parameters a setting doesn't name are off (top-k 0, top-p 1, temperature 1, repetition penalty 1), not Qwen's defaults (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05)
- [ ] **A3:** batches of 8 with left padding and one seed per batch, so sampled answers depend on the batch size, and fp16 batching can change a greedy answer slightly
- [ ] **A3:** questions run in a fixed shuffled order; the 50 for B are the first 50 of that order
- [ ] **A3:** two extra fields in `gen/answers_*.jsonl`: `n_new_tokens` and `hit_max_tokens`; `answer_text` is the decoded tokens up to the stop token, not stripped
- [ ] **A4:** a new label, `question`: a number found only in the question. Left out of main training, like `wrong_calc`
- [ ] **A4:** `wrong_calc` = every number before it in the answer is copied, computed or question, and at least one is copied. A number after a `wrong_calc` or `fabricated` one is `fabricated`
- [ ] **A4:** `copied` is checked against every report number, so a wrong value can be `copied` by coincidence (in a test on fake answers, 3 of 14 wrong values matched within 1%, e.g. `3.473` vs `3.5%`). Watch for this in the hand-check
- [ ] **A4:** the hand-check sheet is 50 random numbers from one setting (not balanced by label); answers whose tokens don't line up with the prompt are skipped and counted
- [ ] **A5 (not decided yet):** which agreement does README A5's "~90%" mean? `labels/agreement.py` prints four: A vs B (with Cohen's kappa), A vs auto, B vs auto, and auto vs the rows where A = B
- [ ] **A6:** accuracy = the answer's **last** number equals the gold answer within `label_tolerance` (sign ignored; `12%` = `0.12`). The table also has `gold_anywhere`: gold equals any number in the answer
- [ ] **A6:** fabrication rate two ways: share of numbers labeled `fabricated` (in the chart), and share of answers with at least one
- [ ] **A6:** "distinct answers across settings" = per setting, % of questions whose last number differs from greedy's; overall, the mean number of different last numbers per question
- [ ] **A6:** settings are compared on the questions every setting answered, all splits pooled; 95% CIs by bootstrap over reports
- [ ] **A7:** confidence is computed over the digit tokens only (not `$`, sign or scale word), the same span as B1
- [ ] **A7:** confidence uses the model's own distribution (temperature 1, no top-k/top-p) for every setting, not the one each setting sampled from, so scores are comparable across settings
- [ ] **A7 → B1, B3:** confidence and feature files carry `row` (line in `labels/numbers_{setting}.jsonl`) so B3 can join them

**Stage 2 (Yvette)**

- [ ] **B0, B1 (not decided yet):** which attention row stands for a number token: its own row (default) or the row before it, which predicts it (`--row predicting`). Both scripts share the code in `features/attention.py`
- [ ] **B0:** the plot shows the 4 heads with the most attention on the report (`--heads` picks others); the example is the first val question and the first answer number found in the report (`--id`, `--number` pick others)
- [ ] **B1:** "generated tokens" (README §6.4) = answer tokens before the number token. With `--row predicting`, that includes the row's own token, as in Lookback Lens
- [ ] **B1:** a number's lookback = the mean of its digit tokens' ratios; a token with no answer token before it is skipped; a number with none left is NaN (B3 drops it)
- [ ] **B2:** "normalized frequency" = numpy's `rfftfreq` units (0 to 0.5), so 0.25 / 0.35 / 0.45 keep the top 50% / 30% / 10%. If Qi et al. normalize the top to 1, the same cutoffs would keep 75% / 65% / 55%
- [ ] **B2:** `hf_ctx` averages over all the number's tokens (the README §6.4 skip rule is only for lookback); `hf_gen` skips tokens with no answer token before them, and a one-token span gives 0
- [ ] **B2:** HF features are stored as float16, and `features/*.npz` stays out of git (~5 MB per 1,000 numbers); they live in the shared Drive folder and can be rebuilt
- [ ] **B3:** a number with a NaN in any feature is dropped for all detectors, so every detector is scored on the same numbers
- [ ] **B3:** the confidence baseline is a logistic regression on mean/min log-prob and mean entropy, tuned on val like the others (not one raw score)
- [ ] **B3:** C is picked from 0.001, 0.01, 0.1, 1, 10 by val AUROC; no class weighting; `hf` and `lookback_hf` each pick their own cutoff
- [ ] **B3:** F1 threshold = best F1 on val; CIs = 1,000 bootstrap resamples of test reports (resamples with one class are skipped)
- [ ] **B3:** settings are pooled with each (setting, number) as one example, so a question can appear up to five times; its copies share a report, so they share a split
- [ ] **B4 RQ2:** each pairwise AUROC uses only test numbers with those two labels; "flagged" = score at or above B3's val-tuned threshold; "correct computed numbers" = numbers labeled `computed`
- [ ] **B4 RQ3:** per-layer AUROC = a detector on that layer's heads only (12 features for lookback, 36 for lookback_hf), C tuned on val. Head importance = absolute standardized coefficient in the full detector (B3's C and cutoff, train only), summed over a head's 3 features for lookback_hf. Top-k = 10, 25, 50, 100, 200 and all
- [ ] **B4 bridge:** a number's lookback = mean over all layers and heads; then the mean per setting, all splits. Shown per label too, because settings produce different label mixes
- [ ] **B5:** the detector analyzed is the learned one with the best **val** AUROC (`--detector` / `DETECTOR` picks another)
- [ ] **B5:** the 10 + 10 examples are a seeded random sample of the test errors; `--pick extreme` / `PICK = "extreme"` takes the most confident errors instead
- [ ] **B5:** false positive = a flagged `copied` or `computed` number (a flagged `question` number doesn't count); false negative = a `fabricated` number that isn't flagged

**R1 (both)**

- [ ] **R1:** only TAT-QA's `arithmetic` questions with a numeric answer (train, dev and test_gold, all treated as test); 500 sampled after the A1 length filter
- [ ] **R1:** TAT-QA context = the table first, then the paragraphs in order (FinQA: text, table, text)
- [ ] **R1:** `gold_answer` is TAT-QA's answer as written (scale not applied; a percent answer like −22.22 is already ×100). `gold_steps` = every operation in the derivation; `%` and a lone `(n)` are read whichever way reproduces the answer. 18 of 6,959 derivations (0.3%) contain words and keep only the gold answer
- [ ] **R1:** the transfer uses FinQA's C, cutoff and F1 threshold unchanged, and pools whichever TAT-QA settings were run

## 2. AI-written code we must be able to explain

README §3: each of us can explain every file in our folders line by line, and has checked its tests' expected values by hand. By Dec 9, each must also be able to explain the other's files.

**Both**

- [ ] `common/prompt.py`: check `tests/test_prompt.py` (including `encode_answer`, used by A7 and B1)
- [ ] `common/splits.py`: check `tests/test_splits.py`
- [ ] `common/numbers.py`: check every expected value in `tests/test_numbers.py`, and add a few strings of our own in the styles the model actually uses (README A2)
- [ ] `common/io.py`: the shared file helpers
- [ ] `labels/agreement.py`: check the hand-worked kappa and rates in `tests/test_agreement.py`

**Eduardo (A)**

- [ ] `data/build_subset.py`: sanity-check the counts it prints on Colab
- [ ] `common/program.py`: check `tests/test_program.py`
- [ ] `gen/generate.py`: check `tests/test_generate.py`, especially the decoding arguments each setting turns into
- [ ] `labels/label.py`: check every expected label in `tests/test_label.py` against README §6.3
- [ ] `baseline/confidence.py`: explain why the logits at position *t* score token *t + 1*; check `tests/test_confidence.py`
- [ ] `analysis/stage1.py` and `analysis/stage1.ipynb`: check the hand-worked example in `tests/test_stage1.py`

**Yvette (B)**

- [ ] `analysis/b0_attention_plot.py`: check `tests/test_b0.py`
- [ ] `features/attention.py` and `features/extract.py`: check the hand-worked lookback and HF values in `tests/test_features.py`
- [ ] `detector/train_eval.py`: check the hand-worked F1 and threshold cases in `tests/test_detector.py`
- [ ] `analysis/stage2.py` and `analysis/stage2.ipynb`: check `tests/test_stage2.py`
- [ ] `analysis/errors.py` and `analysis/errors.ipynb`: check `tests/test_errors.py`

**R1 (both)**

- [ ] `common/tatqa.py`, `data/build_tatqa.py`, `detector/transfer.py`: check `tests/test_tatqa.py` and `tests/test_transfer.py`

## 3. Analysis and writing, by step

⏳ waiting on an earlier step · 🟢 ready to start · ✅ done

Claude makes the plots and tables; reading them and drawing conclusions is ours. A = Eduardo, B = Yvette. Order and details: `MANUAL_STEPS.md`.

| Step | What's ours | Who | Status |
|---|---|---|---|
| A1 | Run `data/build_subset.py`; fill the A1 stats line in `LOG.md` (a run) | A | 🟢 ready |
| A3 | Run `gen/generate.py` for your settings; greedy `--limit 50` first (a run) | A: greedy, beam5 · B: temp07, temp12, topp09 | 🟢 ready to run |
| B0 | Run `analysis/b0_attention_plot.py` (after A1); explain the plot to professor: what the heads attend to | B | 🟢 ready to run |
| A4 | Run `labels/label.py` for your settings; label counts into `LOG.md` (a run) | A: greedy, beam5 · B: temp07, temp12, topp09 | ⏳ needs A3 |
| A5 | Hand-label the same 50 numbers without seeing the auto labels; run `labels/agreement.py`; go through the disagreements; if under ~90%, decide how §6.3 changes | both | ⏳ needs the A3 greedy run |
| A6 | RQ0: what decoding does to accuracy and fabrication, and why ("Our reading" in `analysis/stage1.ipynb`) | A | ⏳ needs A3, A4 |
| A6 | Stage 1 error analysis (Nov 23–29) | A | ⏳ needs A6 results |
| A7 | Run `baseline/confidence.py` for your settings (a run) | A: greedy, beam5 · B: temp07, temp12, topp09 | ⏳ needs A4 |
| B1–B2 | `features/extract.py --limit 20`: row sums ~1, lookback within 0–1 (B1 done); then the full run for your settings (B2 done) | B: the check · A, B: full runs | ⏳ needs A4 |
| B3 | Read `detector/results.csv`: which detectors beat the confidence baseline, given the CIs | B | ⏳ needs A7, B2 |
| B4 | **RQ2 (main contribution):** do detectors flag correct computed numbers, and does that support the hypothesis? (RQ2 cell in `analysis/stage2.ipynb`) | B | ⏳ needs B3 |
| B4 | RQ3: which layers and heads carry the signal; does a top-100-head detector keep most of the AUROC? (RQ3 cell) | B | ⏳ needs B3 |
| B4 | Bridge: does the lookback ratio fall as temperature rises? (bridge cell) | B | ⏳ needs B2 |
| B5 | Explain 10 false positives, 10 false negatives, and where `wrong_calc` falls, in our own words (notes files, then `analysis/errors.ipynb`) | A: false positives, `wrong_calc` · B: false negatives | ⏳ needs B3 |
| R1 | Optional: hand-check ~30 TAT-QA labels, then interpret the transfer (`detector/transfer_results.csv`) | A: TAT-QA subset, greedy · B: temp12, transfer · both: hand-check | ⏳ needs B3 and the R1 runs (`MANUAL_STEPS.md`, "R1 (optional)") |
| R2 | Poster text and takeaways | A: dataset, labeling, decoding chart · B: attention diagram, AUROC table, heatmap | ⏳ |
| R3 | Report: own sections, then both edit the draft | both | ⏳ |
| Dec 7–8 | Rehearse: each explains the other's half | both | ⏳ |
