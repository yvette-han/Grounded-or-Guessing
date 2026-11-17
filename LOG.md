# LOG

One dated line per decision or dead end (README §3).
Example: `Nov 14: switched tolerance to 1% because 'about 12%' was being labeled fabricated`

- Nov 5: Repo created with README, `config.yaml` and `requirements.txt` from the project plan.
- Nov 5: A1 pools FinQA train+dev+test (8,281 questions, 799 company/year reports) and re-splits by report, because 243 of FinQA's 278 test reports also appear in its train file. FinQA pinned to commit `0f16e28`.
- Nov 5: A1 drops the 154 yes/no questions (no numeric gold answer) → 8,127 left.
- Nov 5: Report text drops FinQA's 13,057 lines that are only "." and puts each sentence on its own line.
- Nov 5: The prompt goes through the chat template as the user turn (`chat_template: true`). The `max_context_tokens` filter counts the full templated prompt.
- Nov 5: `ctx_start`/`ctx_end` are a half-open token span; a token that straddles the report's edge (e.g. "." merged with "\n\n") counts as context.
- Nov 5: Split rule: shuffle reports with the seed, then fill train, val, test by question count; a split can go over by one report.
- A1 stats (fill in after running with the Qwen tokenizer): questions after length filter = ?, avg context tokens = ?, split sizes = ?
- Nov 5: A2 `common/numbers.py` also skips the day after a month name (`december 31`), ordinals (`3rd`), digits inside words (`Q4`), and list markers `1.`, `1)`, `(1)` at a line start.
- Nov 5: A2 matching: relative tolerance; `12%` equals `12` and `0.12`; a scale-word number vs a bare number tries ones, thousands, millions and billions; sign ignored by default.
- Nov 5: A2 check: `extract_numbers` + `find_match` find 99.57% (18,658 / 18,738) of the numeric arguments of FinQA's gold programs in their reports. The 80 misses are years and days (skipped by rule) and constants not in the report (`1000000`, `3`).
- Nov 5: B0 generates its own greedy answer (A3 hasn't run), then runs one teacher-forced pass over prompt + answer as one string. Default row: the number's first digit token; `--row predicting` uses the token before it.
- Nov 5: `common/prompt.py` gained `tokens_in_span` (character span → token indices, for A4 and B1); `encode` also returns `ctx_chars`.
- Nov 5: A3: transformers fills any decoding parameter we leave unset from the model's own config (Qwen: temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05). So A3 replaces that config and switches off everything a setting doesn't name.
- Nov 5: A3 runs batches of 8 (left padding) with one seed per batch, in a fixed shuffled question order, and resumes from an existing output file. New answer fields: `n_new_tokens`, `hit_max_tokens`.
- Nov 5: `common/prompt.py` gained `encode_answer` (prompt + answer as one string; checks the prompt's tokens don't change). B0 uses it.
- Nov 5: A1 stores `gold_steps`, each gold program step's value from `common/program.py` (follows FinQA's own executor). The last step equals FinQA's `exe_ans` on all 8,127 numeric questions.
- Nov 5: A2 fix: the day in FinQA's spaced dates (`dec . 29`) is skipped too.
- Nov 5: A4 adds the label `question` (a number only in the question; ~10% of FinQA questions contain numbers), between computed and wrong_calc, left out of main training. `wrong_calc` = every earlier number in the answer is copied/computed/question and at least one is copied.
- Nov 5: A4 token span = the number's digits (`tok_start`/`tok_end`, counted from the first answer token); the whole mention is `char_start`/`char_end`. `--handcheck 50` samples 50 random numbers from one setting and never overwrites an existing sheet.
- Nov 5: A4 test on fake answers: 3 of 14 deliberately wrong values matched some report number within 1% and came out `copied` (e.g. `3.473` vs `3.5%`). Check how often this happens on real answers in A5.
- Nov 5: A5 `labels/agreement.py` joins the two hand-labeled sheets and the key by `row`; prints A vs B (percent and Cohen's kappa), A vs auto, B vs auto, A=B rows vs auto and a confusion table; writes `labels/handcheck_disagreements.csv`. Hand-check CSVs are UTF-8 with BOM so Excel shows ⟦ ⟧.
- Nov 5: A6 `analysis/stage1.py`: accuracy = last number vs gold (within `label_tolerance`), plus gold-anywhere; fabrication = share of numbers labeled fabricated, plus share of answers with one; distinct answers = % differing from greedy and mean distinct finals per question; length from `n_new_tokens`. Settings compared on their common questions; 95% CIs by bootstrap over reports.
- Nov 5: A7 `baseline/confidence.py`: one teacher-forced pass per answer; per number, mean and min log-prob and mean entropy (nats) over its digit tokens, under the model's own distribution. New join key `row` (line in `labels/numbers_{setting}.jsonl`) in the confidence and feature contracts.
- Nov 5: B1 `features/extract.py`: one teacher-forced pass per answer with output_attentions (eager, fp16); keeps only the number tokens' rows; saves per-number lookback [layers, heads] (mean over digit tokens) with `label`, `id`, `row`. Lookback and row choice live in `features/attention.py`, shared with B0.
- Nov 5: Definition: the generated tokens for number token t = the answer tokens before t. B0 now follows it too (it used to exclude the predicting row's own token). File helpers moved to `common/io.py`.
- Nov 5: B2 (same pass as B1): `hf_ctx`, `hf_gen` [cutoffs, layers, heads] per number. Frequencies in `numpy.fft.rfftfreq` units (0 to 0.5); coefficients below the cutoff zeroed. `hf_ctx` averages over all the number's tokens; `hf_gen` skips tokens with no answer token before them.
- Nov 5: Feature files are ~5 MB per 1,000 numbers (HF as float16, compressed), so `features/*.npz` is git-ignored and kept in the shared Drive folder.
- Nov 5: B3 `detector/train_eval.py`: pools settings that have both features and confidence (joined on setting + row); drops numbers with a NaN feature; L2 logistic regressions (scaler fit on train) for lookback, hf, lookback_hf, confidence, plus random; C (0.001 to 10) and HF cutoff by val AUROC; threshold = best val F1; test AUROC and F1 with 1,000 report-bootstrap CIs. Writes `results.csv` and `test_scores.csv` (all test numbers, all labels).
- Nov 5: B4 `analysis/stage2.py`: RQ2 pairwise AUROCs (fabricated vs copied, vs computed) and per-label flag shares; RQ3 per-layer detectors, head importance = absolute standardized coefficient of the full detector (B3's C and cutoff, train only), top-k head detectors; bridge = mean lookback per setting and label. CIs by bootstrap over reports.
- Nov 5: B5 `analysis/errors.py`: for the detector with the best val AUROC, a seeded random 10 false positives (copied/computed flagged) and 10 false negatives (fabricated missed) with full context, plus test scores per label (quartiles, share flagged, box plot with the threshold). `read_test_scores` also returns `row`.
- Nov 5: R1 `common/tatqa.py` converts TAT-QA's arithmetic questions to the A1 format (table, then paragraphs; split test; extra `gold_scale`). Evaluating derivations (with brackets, `%` and accounting `(n)` readings) reproduces the answer for 6,941 of 6,959; the other 18 contain words. TAT-QA pinned to commit `870accc`. `data/build_tatqa.py` samples `tatqa_n_questions` (500).
- Nov 5: R1 `detector/transfer.py`: B3's fitting split into `fit_detectors`, `score_detector`, `test_metrics` (B3 output unchanged); detectors fit on FinQA as in B3, then score every TAT-QA number; CIs by bootstrap over TAT-QA tables. TAT-QA outputs live in `tatqa/`.
- Nov 5: Manual work split so both GPUs run in parallel: Person A runs greedy and beam5 and the A steps; Person B runs temp07, temp12, topp09 and the B steps (`MANUAL_STEPS.md`). Labeling waits for the A5 decision; feature files go to the shared Drive folder `GoG-shared`.
- Nov 5: Roles: Person A (Stage 1) = Eduardo, Person B (Stage 2) = Yvette.
- Nov 11: Office hours with the professor. What he said:
  - The project is a good fit for the class. Start simple: one model, a small set of instances with and without hallucinations, attention features, a simple classifier. Add the other ideas only if time allows.
  - Read the related papers to see how they aggregate attention patterns. Start from their features (replicate them on our data), and add our own only if they improve the predictions. The point is to explore and understand how these methods work, not the best score.
  - Hallucinations come in two kinds: made up with no support (our `fabricated`), or the right data with a wrong computation (our `wrong_calc`). With labels for these, the task could be 3-way: in the report, computed, hallucinated.
  - Analyze how good the classifier is and which attention features are most indicative of hallucination; plot attention patterns on single examples and on groups of instances.
  - Optional: for a question where the model hallucinates, add information to the question that points to the right place, rerun, and compare the attention on the same instance before and after.
  - The report must describe what we learned about this area (from the papers and our experiments), backed by our experiments and analysis. The report is graded and required; a poster session would not be graded.
- Nov 17: The poster session is cancelled (confirmed). The report (Dec 11) is the graded deliverable; the poster is removed from README, `MANUAL_STEPS.md` and `ANALYSIS_TODO.md`.
- Nov 17: Detector tasks (Yvette, after office hours): main = hallucinated (`fabricated` or `wrong_calc`) vs copied + computed; extra = fabricated vs copied + computed (the main task until now); `question` in neither. `detector/results.csv` gains a `task` column; the extra task's scores in `test_scores.csv` are `fabricated_{detector}`. B4 RQ2 = hallucinated vs copied and vs computed; B5 false negatives = missed hallucinated numbers.
- Nov 17: 3-way classifier `detector/three_way.py` (copied / computed / hallucinated): multinomial logistic regression with balanced class weights; C and HF cutoff by val macro-F1; test macro-F1 with a CI over reports, recall per class, confusion matrix.
- Nov 17: R2 hint test `analysis/hint_test.py`, to run only if time allows. The hint is appended to the question (the report's tokens don't change); greedy; the new answer is labeled with the A4 rules; attention over the report is compared before and after. The counterpart number is matched by the text before it on its line. The focus number skips the cut-off last number of a truncated answer and the value 100.
- Nov 17: Scope (1,000 questions × 5 settings) to be decided together after timing the first 50 greedy answers on a T4 (`MANUAL_STEPS.md` A-3).

## Run notes: Person A

_Person A (Eduardo) adds run notes and counts here (one line each, dated)._

## Run notes: Person B

_Person B (Yvette) adds run notes and counts here (one line each, dated)._
