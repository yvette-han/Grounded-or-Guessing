# Analysis to do (ours, not AI's)

README §3: AI may write the code, but *"the analysis and the knowledge that you produce must be your own."* This file tracks the work in that second group, plus the AI-written code we still need to understand. Claude updates it whenever it adds code or results; we tick items off. How to do each run by hand is in `MANUAL_STEPS.md`.

_Last updated: Nov 5, after the R1 code._

## 1. Choices made in AI-written code, to review

Each one is also in `LOG.md`. Keep it, change it or replace it; once we've decided, it's our decision.

- [ ] **A1:** pool FinQA train+dev+test and re-split by report (243 of FinQA's 278 test reports are also in its train file)
- [ ] **A1:** drop the 154 yes/no questions
- [ ] **A1:** drop FinQA's `"."`-only lines; one sentence per line in the report text
- [ ] **A1:** send the prompt through the chat template; `max_context_tokens` counts the full templated prompt
- [ ] **A1:** split fill rule: reports shuffled, then train → val → test by question count; a split can go over by one report
- [ ] **A1 → A3, B1:** the report's token span is half-open (`ids[ctx_start:ctx_end]`); a token straddling the report's edge counts as context
- [ ] **A2:** a number with a scale word vs a bare number tries the bare one in ones, thousands, millions and billions (`$4.2 billion` = table `4215`). Lenient: it may call a few made-up numbers copied
- [ ] **A2:** sign is ignored when matching by default (`a decrease of 12` = `(12)`); `ignore_sign=False` turns this off
- [ ] **A2:** a percent matches both readings: `12%` = `0.12` and `12%` = `12`
- [ ] **A2:** extra skip rules beyond README 6.3: the day after a month name (`december 31`), ordinals (`3rd`), digits inside words (`Q4`, `page_49`); list markers also include `1)` and `(1)`
- [ ] **A2:** `(12)` = −12 also turns FinQA footnote markers like `( 1 )` in reports into −1
- [ ] **A2 → A4:** each number has two spans, the whole mention (`$4.2 billion`) and just the digits (`4.2`). A4 uses the digits for `tok_start`/`tok_end` and keeps the mention in `char_start`/`char_end`
- [ ] **B0, B1 (not decided yet):** which attention row stands for a number token: its own row (default in both, README wording) or the row before it, which predicts it (`--row predicting`). B0 and B1 share the code in `features/attention.py`
- [ ] **B0:** the plot shows the 4 heads with the most attention on the report from that token (`--heads` picks others); the example is the first val question and the first answer number found in the report unless `--id` / `--number` are given
- [ ] **A3:** decoding parameters a setting doesn't name are switched off (top-k 0, top-p 1, temperature 1, repetition penalty 1) instead of taking Qwen's defaults (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05)
- [ ] **A3:** batches of 8 with left padding, one seed per batch from (seed, setting, first id): sampled answers depend on the batch size, and fp16 batching can change a greedy answer slightly compared with batch size 1
- [ ] **A1:** the subset now has `gold_steps`, each gold program step's value from `common/program.py` (matches FinQA's own answer on all 8,127 questions)
- [ ] **A4:** new label `question` (not in the original README 6.3): a number that is only in the question. Left out of main training like `wrong_calc` (README 6.3 and 6.5 updated)
- [ ] **A4:** `wrong_calc` in code: every number before it in the answer is copied, computed or question, and at least one is copied. A number after a `wrong_calc` or `fabricated` one is `fabricated`
- [ ] **A4:** `copied` is checked against every number in the report, so a wrong value can be labeled `copied` by coincidence. In the smoke test with fake answers, 3 of 14 deliberately wrong values matched a report number within 1% (`3.473` vs `3.5%`). Watch for this in the hand-check
- [ ] **A4:** the hand-check sheet is 50 numbers drawn at random from one setting (not balanced by label); answers whose tokens don't line up with the prompt are skipped and counted
- [ ] **A5 (not decided yet):** README A5 says "if agreement is under ~90%" but not which agreement. `labels/agreement.py` prints four: A vs B (with Cohen's kappa), A vs auto, B vs auto, and auto vs the rows where A = B. Decide which one the ~90% applies to
- [ ] **A6:** numeric accuracy = the answer's **last** number equals the gold answer within `label_tolerance` (sign ignored, `12%` = `0.12`). `gold_anywhere` (gold equals any number in the answer) is in the table too
- [ ] **A6:** fabrication rate two ways: share of all numbers labeled `fabricated` (the one in the chart) and share of answers with at least one
- [ ] **A6:** "distinct answers across settings" = per setting, % of questions whose last number differs from greedy's; overall, mean number of different last numbers per question across settings
- [ ] **A6:** every setting is measured on the questions answered in all available settings, all splits pooled (Stage 1 trains nothing); 95% CIs bootstrap over reports
- [ ] **A7:** confidence is computed over the number's digit tokens only (not `$`, sign or scale word), the same span B1 will use
- [ ] **A7:** confidence uses the model's own next-token distribution (temperature 1, no top-k/top-p) for every setting, not the distribution each setting actually sampled from (e.g. temperature 1.2), so scores are comparable across settings
- [ ] **A7 → B1, B3:** confidence and feature files carry `row` (line in `labels/numbers_{setting}.jsonl`) so B3 can join them (README §7 updated for both)
- [ ] **B1:** "generated tokens" (README 6.4) = answer tokens before the number token. With `--row predicting` that includes the row's own token, as in Lookback Lens (B0 now uses the same definition)
- [ ] **B1:** a number's lookback = mean of the per-token ratios over its digit tokens; a token with no answer token before it is skipped, and a number with none left is NaN (B3 must drop those)
- [ ] **B2:** "normalized frequency" for `hf_cutoffs` = numpy's `rfftfreq` units (cycles per token, 0 to 0.5), so 0.25 / 0.35 / 0.45 keep the top 50% / 30% / 10% of the range. If the paper (Qi et al.) normalizes so the top is 1, the same cutoffs would keep 75% / 65% / 55%
- [ ] **B2:** `hf_ctx` averages over all the number's tokens (the README 6.4 skip rule is for the lookback ratio); `hf_gen` skips tokens with no answer token before them, and a one-token answer span gives 0 (a single value has no high frequencies)
- [ ] **B2:** HF features are stored as float16 and `features/*.npz` stays out of git (about 5 MB per 1,000 numbers); they live in the shared Drive folder and can be rebuilt from the code and answers
- [ ] **B3:** a number with a NaN in any feature (e.g. at the very start of an answer) is dropped for all detectors, so every detector is scored on the same numbers
- [ ] **B3:** the confidence baseline is a logistic regression on mean/min log-prob and mean entropy, tuned on val like the attention detectors (not a single raw score)
- [ ] **B3:** C is picked from 0.001, 0.01, 0.1, 1, 10 by val AUROC; no class weighting; `hf` and `lookback_hf` each pick their own cutoff by val AUROC
- [ ] **B3:** F1 threshold = the one with the best F1 on val; CIs = 1,000 bootstrap resamples of test reports (resamples with only one class are skipped)
- [ ] **B3:** settings are pooled with each (setting, number) as its own example, so one question can appear up to five times; all its copies share a report and so a split
- [ ] **B4 RQ2:** each pairwise AUROC uses only test numbers with those two labels; "flagged" = score at or above the detector's val-tuned threshold from B3; "correct computed numbers" = numbers labeled `computed`
- [ ] **B4 RQ3:** per-layer AUROC = a detector trained on that layer's heads only (12 features for lookback, 36 for lookback_hf), C tuned on val. Head importance = absolute standardized coefficient in the full detector refit on train with B3's C and cutoff, summed over a head's 3 features for lookback_hf. Heads are ranked on train only; top-k = 10, 25, 50, 100, 200 and all
- [ ] **B4 bridge:** a number's lookback = mean over all layers and heads; then the mean over numbers per setting, all splits, numbers without a lookback left out. Shown per label too, because the settings differ in how many numbers of each label they produce
- [ ] **B5:** the detector analyzed is, by default, the learned one with the best **val** AUROC (`--detector` / `DETECTOR` picks another)
- [ ] **B5:** the 10 + 10 examples are a seeded random sample of the test errors (representative); `--pick extreme` / `PICK = "extreme"` takes the most confident errors instead
- [ ] **B5:** false positive = a `copied` or `computed` number that is flagged (a flagged `question` number isn't counted); false negative = a `fabricated` number that isn't flagged
- [ ] **R1:** only TAT-QA's `arithmetic` questions with a numeric answer (from its train, dev and test_gold files, all treated as test); `tatqa_n_questions` = 500 sampled after the same length filter as A1
- [ ] **R1:** TAT-QA context = the table first, then the paragraphs in order (FinQA is text, table, text)
- [ ] **R1:** `gold_answer` is TAT-QA's answer as written (scale not applied; a percent answer like -22.22 is already x100). `gold_steps` = every operation's value in the derivation; `%` and a lone `(n)` are read whichever way reproduces the answer. 18 of 6,959 derivations (0.3%) contain words and keep only the gold answer
- [ ] **R1:** the transfer uses FinQA's C, cutoff and F1 threshold unchanged, and pools whichever TAT-QA settings were run
- [ ] **A3:** questions run in a fixed shuffled order; the 50 for B are the first 50 of that order
- [ ] **A3:** two extra fields in `gen/answers_*.jsonl`: `n_new_tokens`, `hit_max_tokens` (README §7 updated); `answer_text` is the decoded tokens up to the stop token, not stripped

## 2. AI-written code we must be able to explain

README §3: each of us can explain every file in our folders line by line, and each function is checked on an example where we already know the answer.

- [ ] `common/prompt.py` (both): explain it; check the expected strings in `tests/test_prompt.py` by hand
- [ ] `common/splits.py` (both): explain it; check the expected values in `tests/test_splits.py` by hand
- [ ] `data/build_subset.py` (A): explain it; sanity-check the counts it prints on the Colab run
- [ ] `common/numbers.py` (both): explain it; check every expected value in `tests/test_numbers.py` by hand. README A2 asks for hand-made strings, so also add a few of your own, especially styles you expect from the model's answers
- [ ] `analysis/b0_attention_plot.py` (B): explain it; check the hand-worked lookback values in `tests/test_b0.py`
- [ ] `common/program.py` (A): explain it; check `tests/test_program.py` by hand
- [ ] `labels/label.py` (A): explain it; check every expected label in `tests/test_label.py` by hand against README 6.3
- [ ] `labels/agreement.py` (both): explain it; check the hand-worked kappa and rates in `tests/test_agreement.py`
- [ ] `analysis/stage1.py` and `analysis/stage1.ipynb` (A): explain them; check the hand-worked example in `tests/test_stage1.py`
- [ ] `baseline/confidence.py` (A): explain it, especially why the logits at position *t* score token *t + 1*; check `tests/test_confidence.py` by hand
- [ ] `features/attention.py` and `features/extract.py` (B): explain them; check the hand-worked lookback and high-frequency values in `tests/test_features.py`
- [ ] `common/io.py` (both): the shared file helpers
- [ ] `detector/train_eval.py` (B): explain it; check the hand-worked F1 and threshold cases in `tests/test_detector.py`
- [ ] `analysis/stage2.py` and `analysis/stage2.ipynb` (B): explain them; check the hand-worked cases in `tests/test_stage2.py`
- [ ] `analysis/errors.py` and `analysis/errors.ipynb` (B, both use): explain them; check the hand-worked cases in `tests/test_errors.py`
- [ ] `common/tatqa.py`, `data/build_tatqa.py`, `detector/transfer.py` (R1): explain them; check `tests/test_tatqa.py` and `tests/test_transfer.py`. `detector/train_eval.py` was split into `fit_detectors` / `score_detector` / `test_metrics` for this (same results)
- [ ] `gen/generate.py` (A): explain it; check `tests/test_generate.py`, especially the decoding arguments each setting turns into
- [ ] `encode_answer` in `common/prompt.py` (both): prompt + answer as one string for teacher forcing (A7, B1); check its tests in `tests/test_prompt.py`

## 3. Analysis and writing, by step

⏳ waiting on an earlier step · 🟢 ready to start · ✅ done

Claude can make the plots and tables for these; reading them and drawing conclusions is ours.

| Step | What's ours | Status |
|---|---|---|
| A1 | Run `data/build_subset.py` with the Qwen tokenizer and fill the stats line in `LOG.md` (a run, not analysis) | 🟢 ready |
| A3 | Run `gen/generate.py` for every setting on Colab, `--setting greedy --limit 50` first for B (a run, not analysis) | 🟢 ready to run |
| B0 | Run `analysis/b0_attention_plot.py` on Colab (after A1), then explain the plot to the professor: what the heads attend to | 🟢 ready to run |
| A4 | Run `labels/label.py` for every setting and put the label counts in `LOG.md` (a run, not analysis) | ⏳ needs the A3 run |
| A5 | Hand-label the same 50 numbers each without seeing the auto labels (sheet: `labels/label.py --setting greedy --handcheck 50`); run `labels/agreement.py`; go through `handcheck_disagreements.csv` together; if under ~90%, decide how to change §6.3 | ⏳ needs the A3 greedy run |
| A6 | RQ0: what decoding does to accuracy and fabrication rate, and why. Write it in the "Our reading" cell of `analysis/stage1.ipynb` | ⏳ needs the A3 and A4 runs, then `python analysis/stage1.py` |
| A6 | Stage 1 error analysis (Nov 23–29) | ⏳ needs A6 results |
| A7 | Run `baseline/confidence.py` on Colab (a run, not analysis) | ⏳ needs the A4 run |
| B1–B2 | Run `features/extract.py --limit 20` and check row sums ~1 and lookback within 0–1 (README B1 done), then the full run for every setting (README B2 done) (a run, not analysis) | ⏳ needs the A4 run |
| B3 | Read the results table (`detector/results.csv`): which detectors beat the confidence baseline, given the confidence intervals | ⏳ needs the A7 and B2 runs, then `python detector/train_eval.py` |
| B4 | **RQ2 (main contribution):** do detectors flag correct computed numbers, and does that support the hypothesis? Write it in the RQ2 "Our reading" cell of `analysis/stage2.ipynb` | ⏳ needs the B3 run, then `python analysis/stage2.py` |
| B4 | RQ3: which layers and heads carry the signal, does a top-100-head detector keep most of the AUROC, and what that means (RQ3 cell) | ⏳ needs the B3 run |
| B4 | Bridge: does the lookback ratio fall as temperature rises, and what that says (bridge cell) | ⏳ needs the B2 run |
| B5 | 10 false positives, 10 false negatives, where `wrong_calc` falls: explain each pattern in our own words, in the "Our reading" cells of `analysis/errors.ipynb` (both of us) | ⏳ needs the B3 run |
| R1 | Stretch: hand-check ~30 TAT-QA labels (its answers and derivations differ from FinQA's), then interpret the FinQA → TAT-QA transfer (`detector/transfer_results.csv`) | ⏳ optional; needs the B3 inputs and the R1 runs (`MANUAL_STEPS.md` step 12) |
| R2 | Poster text and takeaways | ⏳ |
| R3 | Report: each writes their own sections, both edit the draft | ⏳ |
| Dec 7–8 | Rehearse: each explains the other's half | ⏳ |
