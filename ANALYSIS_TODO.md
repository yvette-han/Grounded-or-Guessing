# Analysis to do (ours, not AI's)

README §3: AI may write the code, but *"the analysis and the knowledge that you produce must be your own."* This file lists that work, plus the AI-written code we must understand. Claude adds items as it writes code; we tick them off. How to run each step: `MANUAL_STEPS.md`.

_Last updated: Nov 23 (A-2 code check by Claude: section 2 (A)/(both) files checked, `Claude check` lines under A1–A7, edge cases in `tests/test_numbers.py`; earlier: merged main into `eduardo`; review fixes: LaTeX `\%`, month skip, comma constants, `run_name` in three_way/hint_test; provisional greedy/beam5 labels, Stage 1 table and confidence; float32 for attention features)._

## This week: Yvette (Thu Nov 5 – Sun Nov 8)

Goal: the B0 plot and the A1 stats for the meeting with professor. The runs are in `MANUAL_STEPS.md`, B-0 to B-2. The items below point to sections 1–3; tick them there.

1. **The B0 plot** (section 3, row B0): after the run, write a few sentences in your own words on what the heads attend to when the model writes the number. Put them in your `LOG.md` run notes and bring them to the meeting.
2. **Be able to explain** (section 2): `analysis/b0_attention_plot.py` and `features/attention.py`, line by line. Check the hand-worked values in `tests/test_b0.py` and `tests/test_features.py`.
3. **Start these reviews** (section 1, before Nov 12):
   - **B0, B1:** the row choice (`number` or `predicting`). You decide it on Nov 12 (B-4); this week's B0 runs are a first look.
   - **B0:** which heads and which example the plot shows.
   - **B1:** how "generated tokens" are defined, and the per-number averaging.
   - **B2:** the frequency units. Check how Qi et al. (2026) normalize frequency.
4. **The A1 stats** (section 3, row A1): done by Eduardo on Nov 6; pull `main` instead of running A1.

## 1. Choices made in AI-written code, to review

Each choice is also in `LOG.md`. Keep it, change it, or replace it; once we decide, it's our decision. **Who reviews:** A items → Eduardo, B and R2 items → Yvette, R1 → both.

**Stage 1 (Eduardo)**

- [ ] **A1:** pool FinQA train+dev+test and re-split by report (243 of FinQA's 278 test reports are also in its train file)
  Claude check (Nov 23): yes: `load_finqa` concatenates train, dev and test; `assign_splits` re-splits by `report_key`. Rerun: 278 FinQA test reports, 243 also in train. No unit test of the pooling (script only); the by-report split is tested in `tests/test_splits.py`.
- [ ] **A1:** drop the 154 yes/no questions
  Claude check (Nov 23): yes: `is_numeric(exe_ans)` keeps int/float answers only; rerun gives 154 dropped, 8,127 left. No unit test.
- [ ] **A1:** drop FinQA's `"."`-only lines; one sentence per line in the report text
  Claude check (Nov 23): yes: `_is_junk` drops lines with no letter or digit (13,057 lines are exactly "."), and `build_context` joins one sentence per line. Tested: `test_build_context_order_and_junk_lines`, `test_build_context_keeps_lines_with_digits`.
- [ ] **A1:** send the prompt through the chat template; `max_context_tokens` counts the full templated prompt
  Claude check (Nov 23): yes: `to_model_text` applies the chat template; `build_subset.py` filters on `len(encode(...)["input_ids"])`, the full templated prompt. Tested: `test_encode_chat_template`, `test_encode_with_real_model_tokenizer`.
- [ ] **A1:** split rule: shuffle reports, then fill train → val → test by question count; a split can go over by one report
  Claude check (Nov 23): yes: reports sorted, shuffled with the seed, each goes to the first split under its question limit. Tested: `test_split_sizes_close_to_ratios` (overflow at most one report) and the order/seed tests. The rerun reproduces the subset's 1,000 ids and splits.
- [ ] **A1:** the subset stores `gold_steps`, each gold program step's value (`common/program.py` matches FinQA's own answer on all 8,127 questions)
  Claude check (Nov 23): yes: `gold_steps = run_program(...)`; rerun: last step equals `exe_ans` on all 8,127 numeric questions. Tested: `tests/test_program.py` (10 programs).
- [ ] **A1 → A3, B1:** the report's token span is half-open (`ids[ctx_start:ctx_end]`); a token that straddles the report's edge counts as context
  Claude check (Nov 23): yes: `tokens_in_span` keeps any token overlapping the report's characters and `ctx_end` = last + 1. Tested: `test_tokens_in_span` (partial overlap counts), `test_encode_context_span`.
- [ ] **A2:** a number with a scale word vs a bare number: the bare one is tried in ones, thousands, millions and billions (`$4.2 billion` = table `4215`). Lenient: a few made-up numbers may count as copied
  Claude check (Nov 23): yes: `_readings` divides a scaled number by 1, 1e3, 1e6, 1e9 only when the other side has no scale word. Tested: `$4.2 billion` vs `4215` in `test_same_value` and `test_find_match`.
- [ ] **A2:** sign is ignored when matching (`a decrease of 12` = `(12)`); `ignore_sign=False` turns this off
  Claude check (Nov 23): yes: `same_value(..., ignore_sign=True)` compares absolute values; `label_numbers` uses the default. Tested: `test_same_value` (`(12)` vs 12) and `test_same_value_with_sign`.
- [ ] **A2:** a percent matches both readings: `12%` = `0.12` and `12%` = `12`
  Claude check (Nov 23): yes: `_readings` adds value/100 for a percent vs a non-percent; two percents are compared as written. Tested: `12%` vs 0.12, vs 12, vs `0.12%` in `test_same_value`.
- [ ] **A2:** extra skip rules beyond README §6.3: the day after a month name (`december 31`), ordinals (`3rd`), digits inside words (`Q4`, `page_49`); list markers also include `1)` and `(1)`
  Claude check (Nov 23): yes for all four: `_MONTH_BEFORE` (month name before a day 1–31), a letter right after the digits (`3rd`), the `(?<![\w.])` lookbehind (`Q4`, `page_49`), `_is_list_marker` (`1.`, `1)`, `(1)` + space at a line start). Tested in `CASES`. Not skipped (strict xfails added): `10-K`, `COVID-19`, `30 june`, `**1.**`.
- [ ] **A2:** `(12)` = −12 also turns FinQA footnote markers like `( 1 )` into −1
  Claude check (Nov 23): yes: `( 1 )` mid-line or at a line start reads as -1 (a line-start `(1) ` with no inner spaces is a list marker instead). Now pinned in `EDGE_CASES` (`net income ( 1 ) was 4,215`).
- [ ] **A2 → A4:** `tok_start`/`tok_end` cover only the digits (`4.2`); the whole mention (`$4.2 billion`) is in `char_start`/`char_end`
  Claude check (Nov 23): yes: `label_answer` takes tokens from `num_start`/`num_end` and chars from `start`/`end`. Tested: `test_label_answer_spans`, `test_spans`, `test_latex_percent_span`.
- [ ] **A3:** decoding parameters a setting doesn't name are off (top-k 0, top-p 1, temperature 1, repetition penalty 1), not Qwen's defaults (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05)
  Claude check (Nov 23): yes: `generation_kwargs` sets num_beams 1, repetition_penalty 1, and top_k 0 / top_p 1 / temperature 1 when sampling; `plain_generation_config` replaces the shipped config with special tokens only. Tested: `test_generation_kwargs`, `test_every_config_setting_is_valid`; `plain_generation_config` has no unit test (needs the model).
- [ ] **A3:** batches of 8 with left padding and one seed per batch, so sampled answers depend on the batch size, and fp16 batching can change a greedy answer slightly
  Claude check (Nov 23): yes: `--batch-size` default 8, `left_pad`, `set_seed(batch_seed(seed, setting, first id))`, so the seed depends on which id starts the batch. Tested: `test_left_pad`, `test_batch_seed`; the batch-size dependence itself is not tested.
- [ ] **A3:** questions run in a fixed shuffled order; the 50 for B are the first 50 of that order
  Claude check (Nov 23): yes: `question_order` sorts then shuffles with the seed; `--limit` takes `order[:limit]`. Tested: `test_question_order_is_fixed`.
- [ ] **A3:** two extra fields in `gen/answers_*.jsonl`: `n_new_tokens` and `hit_max_tokens`; `answer_text` is the decoded tokens up to the stop token, not stripped
  Claude check (Nov 23): yes: `n_new_tokens`/`hit_max_tokens` from `count_new_tokens`; `answer_text` = `decode(toks[:n_new], skip_special_tokens=True)` with no strip. Tested: `test_count_new_tokens`.
- [ ] **A3 (Nov 6):** `max_new_tokens` = 512, up from 128, because 128 cut 44 of the first 50 greedy answers before the final answer. Trade-off: longer answers mean more numbers per answer (9 vs 4) and slower feature extraction; 384 would cut 1 of 50 too
  Claude check (Nov 23): yes: `config.yaml` has 512. No test pins the value.
- [ ] **A3 (Nov 7):** runs with other models: a config with `extends: config.yaml` + `model` + `run_name` writes to `<folder>/<run_name>/`; the FinQA subset (questions, splits, the A1 length filter counted with Qwen's tokenizer) stays shared. Review: whether a second model should reuse the same 1,000 questions, and which model(s) to try
  Claude check (Nov 23): yes: `load_config` (`extends`, top-level override) and `run_dir`; `--subset` defaults to `data/finqa_subset.jsonl` for every model. Tested: `tests/test_io.py`. The reuse/which-model question is a decision.
- [ ] **A3 (Nov 7):** `gen/answers_*.jsonl` carries `model`; label, confidence, features and a resumed generate stop on a mismatch. Added to the existing greedy and beam5 files after the fact
  Claude check (Nov 23): yes: `check_model` is called in `label.py` (unless `--tokenizer`), `confidence.py`, `features/extract.py` and `read_done`; both answer files have `model` = Qwen/Qwen2.5-1.5B-Instruct on all 1,000 rows. Tested: `test_check_model`, `test_generate_resume_refuses_other_model`.
- [ ] **A4:** a new label, `question`: a number found only in the question. Left out of every detector task (README §6.5)
  Claude check (Nov 23): yes: checked after computed in `label_numbers`; `detector/train_eval.py` TASKS has `question` in neither class. Tested: the `Costs rose 7%` case in `test_label_numbers`; `tests/test_detector.py` checks question numbers are scored but left out.
- [ ] **A4 (Nov 6):** a new label, `constant`, checked **before** `copied`: a bare integer that is a FinQA program constant (1–10, 100, 1,000, …, 1,000,000), like the 100 in "multiply by 100". Left out of training, counts as a good input for `wrong_calc`. Review: the constant set, whether it should come before `copied` (a bare `100` copied from a table becomes `constant`), and whether the hand-check agrees
  Claude check (Nov 23): yes: `is_constant` runs first; the set is 1–10 and 10^2–10^6, mention must be digits only (commas allowed). Tested: `test_constant_is_checked_before_copied`, `test_constant_with_comma`, and the decorated `$100`/`100%`/`100 million` case.
- [ ] **A2 (Nov 6, not changed):** a bare number vs a bare report number is matched as written, so a value the model rescaled (`$739,187,000` for a table cell `739187` in thousands) is `fabricated`. Decide in the hand-check whether that is right, or whether bare-vs-bare should also try ×1,000 and ×1,000,000
  Claude check (Nov 23): yes: `same_value("$739,187,000", "739187")` is False (no scale word on either side, so no unit readings). No test pins this.
- [ ] **A2/A4 (Nov 23, review fixes):** LaTeX `\%` counts as a percent; the date skip needs a real month name; `1,000`-style constants are `constant`. These changed 276 greedy and 304 beam5 labels (mostly wrong_calc → computed). Not changed, decide at the hand-check: LaTeX scale words (`24 \text{ million}`) and the 1% tolerance on small rounded ratios (`1.6%` vs 0.0164)
  Claude check (Nov 23): the three fixes are in the code and tested (`\approx 17.14\%`, `growth = 8.1 \%`, `revenue decreased 12`, `margin of 5`, `test_constant_with_comma`, `test_latex_percent_is_a_percent`). Open: `24 \text{ million}` is now a strict xfail; `same_value("1.6%", 0.0164)` is False (2.4% apart, tolerance 1%), no test.
- [ ] **A4:** `wrong_calc` = every number before it in the answer is copied, computed or question, and at least one is copied. A number after a `wrong_calc` or `fabricated` one is `fabricated`
  Claude check (Nov 23): yes: `"copied" in labels and all(x in GOOD_INPUTS ...)`, with constant also a good input. Tested: the wrong_calc cases in `test_label_numbers`, including `4,215 then 999 then 20%` (fabricated after wrong_calc).
- [ ] **A4:** `copied` is checked against every report number, so a wrong value can be `copied` by coincidence (in a test on fake answers, 3 of 14 wrong values matched within 1%, e.g. `3.473` vs `3.5%`). Watch for this in the hand-check
  Claude check (Nov 23): yes: `find_match(n, report_numbers, tol)` over all report numbers. No test of the coincidence rate (the 3 of 14 is from a LOG Nov 5 trial).
- [ ] **A4:** the hand-check sheet is 50 random numbers from one setting (not balanced by label); answers whose tokens don't line up with the prompt are skipped and counted
  Claude check (Nov 23): yes: `random.Random(seed).sample(records, n)` over one setting's records, no balancing; skipped answers are counted and printed in `main`. Tested: `test_handcheck_hides_auto_labels`.
- [ ] **A5 (not decided yet):** which agreement does README A5's "~90%" mean? `labels/agreement.py` prints four: A vs B (with Cohen's kappa), A vs auto, B vs auto, and auto vs the rows where A = B
  Claude check (Nov 23): yes: `compare` returns `a_vs_b` + `kappa_a_b`, `a_vs_auto`, `b_vs_auto`, `consensus_vs_auto`; all printed. Tested: `test_compare`, `test_cohen_kappa_by_hand`.
- [ ] **A6:** accuracy = the answer's **last** number equals the gold answer within `label_tolerance` (sign ignored; `12%` = `0.12`). The table also has `gold_anywhere`: gold equals any number in the answer
  Claude check (Nov 23): yes: `is_correct` = `same_value(final_number, gold, tol)` (sign ignored, percent both readings); `gold_anywhere` uses `find_match`. Tested: `test_final_number_and_correctness`, `test_setting_metrics_by_hand`.
- [ ] **A6:** fabrication rate two ways: share of numbers labeled `fabricated` (in the chart), and share of answers with at least one
  Claude check (Nov 23): yes: `fabricated_share` (plotted in `plot_bars`) and `answers_with_fabricated`. Tested: `test_setting_metrics_by_hand` (1/5 and 1/3).
- [ ] **A6:** "distinct answers across settings" = per setting, % of questions whose last number differs from greedy's; overall, the mean number of different last numbers per question
  Claude check (Nov 23): yes: `differs_from_greedy` and `mean_distinct_finals` ("no number" counts as one answer). Tested: `test_differs_from_greedy`, `test_same_final_and_distinct_count`.
- [ ] **A6:** settings are compared on the questions every setting answered, all splits pooled; 95% CIs by bootstrap over reports
  Claude check (Nov 23): yes: `build_table` intersects ids across settings with no split filter; `bootstrap_ci` resamples `report_key` groups (1,000 draws). Tested: `test_build_table_uses_common_questions`, `test_bootstrap_ci`.
- [ ] **A7:** confidence is computed over the digit tokens only (not `$`, sign or scale word), the same span as B1
  Claude check (Nov 23): yes: `number_stats` uses the labels' `tok_start`/`tok_end` (digits only), the same spans `features/extract.py` reads. Tested: `test_number_stats_by_hand`.
- [ ] **A7:** confidence uses the model's own distribution (temperature 1, no top-k/top-p) for every setting, not the one each setting sampled from, so scores are comparable across settings
  Claude check (Nov 23): yes: plain `log_softmax` of the forward-pass logits, no temperature or top-k/top-p, for every setting. Tested: `test_next_token_stats_matches_numpy`.
- [ ] **A7 → B1, B3:** confidence and feature files carry `row` (line in `labels/numbers_{setting}.jsonl`) so B3 can join them
  Claude check (Nov 23): yes: `row` = index from `group_records` (line in the labels file), saved by `save_npz` and in `features/extract.py`. Tested: `test_group_records_keeps_order_and_rows`, `test_save_npz_roundtrip`.

**Stage 2 (Yvette)**

- [ ] **B0, B1 (not decided yet):** which attention row stands for a number token: its own row (default) or the row before it, which predicts it (`--row predicting`). Both scripts share the code in `features/attention.py`
- [ ] **B0:** the plot shows the 4 heads with the most attention on the report (`--heads` picks others); the example is the first val question and the first answer number found in the report (`--id`, `--number` pick others)
- [ ] **B1:** "generated tokens" (README §6.4) = answer tokens before the number token. With `--row predicting`, that includes the row's own token, as in Lookback Lens
- [ ] **B1:** a number's lookback = the mean of its digit tokens' ratios; a token with no answer token before it is skipped; a number with none left is NaN (B3 drops it)
- [ ] **B2:** "normalized frequency" = numpy's `rfftfreq` units (0 to 0.5), so 0.25 / 0.35 / 0.45 keep the top 50% / 30% / 10%. If Qi et al. normalize the top to 1, the same cutoffs would keep 75% / 65% / 55%
- [ ] **B2:** `hf_ctx` averages over all the number's tokens (the README §6.4 skip rule is only for lookback); `hf_gen` skips tokens with no answer token before them, and a one-token span gives 0
- [ ] **B2:** HF features are stored as float16, and `features/*.npz` stays out of git (~5 MB per 1,000 numbers); they live in the shared Drive folder and can be rebuilt
- [x] **B3 (decided Nov 17, Yvette, after office hours):** main task = hallucinated (`fabricated` or `wrong_calc`) vs copied + computed; extra task = fabricated vs copied + computed (the old main task); `question` in neither
- [ ] **B3:** a number with a NaN in any feature is dropped for all detectors, so every detector is scored on the same numbers
- [ ] **B3:** the confidence baseline is a logistic regression on mean/min log-prob and mean entropy, tuned on val like the others (not one raw score)
- [ ] **B3:** C is picked from 0.001, 0.01, 0.1, 1, 10 by val AUROC; no class weighting; `hf` and `lookback_hf` each pick their own cutoff
- [ ] **B3:** F1 threshold = best F1 on val; CIs = 1,000 bootstrap resamples of test reports (resamples with one class are skipped)
- [ ] **B3:** settings are pooled with each (setting, number) as one example, so a question can appear up to five times; its copies share a report, so they share a split
- [ ] **B3 3-way:** multinomial logistic regression on the same feature sets, with **balanced class weights** (copied numbers far outnumber computed ones); C and HF cutoff chosen by val **macro-F1**; prediction = the most probable class; `random` = a uniformly random class
- [ ] **B3 3-way:** macro-F1 averages over the classes present, so a bootstrap resample with no computed numbers averages over two classes; CIs over test reports as in B3
- [ ] **B4 RQ2:** each pairwise AUROC (hallucinated vs copied, hallucinated vs computed) uses only test numbers with those labels, scored by the main task's detectors; "flagged" = score at or above B3's val-tuned threshold; "correct computed numbers" = numbers labeled `computed`
- [ ] **B4 RQ3:** per-layer AUROC = a detector on that layer's heads only (12 features for lookback, 36 for lookback_hf), C tuned on val. Head importance = absolute standardized coefficient in the full detector (B3's C and cutoff, train only), summed over a head's 3 features for lookback_hf. Top-k = 10, 25, 50, 100, 200 and all
- [ ] **B4 bridge:** a number's lookback = mean over all layers and heads; then the mean per setting, all splits. Shown per label too, because settings produce different label mixes
- [ ] **B5:** the detector analyzed is the learned one with the best **val** AUROC (`--detector` / `DETECTOR` picks another)
- [ ] **B5:** the 10 + 10 examples are a seeded random sample of the test errors; `--pick extreme` / `PICK = "extreme"` takes the most confident errors instead
- [ ] **B5:** false positive = a flagged `copied` or `computed` number (a flagged `question` number doesn't count); false negative = a hallucinated (`fabricated` or `wrong_calc`) number that isn't flagged

**R1 (both)**

- [ ] **R1:** only TAT-QA's `arithmetic` questions with a numeric answer (train, dev and test_gold, all treated as test); 500 sampled after the A1 length filter
- [ ] **R1:** TAT-QA context = the table first, then the paragraphs in order (FinQA: text, table, text)
- [ ] **R1:** `gold_answer` is TAT-QA's answer as written (scale not applied; a percent answer like −22.22 is already ×100). `gold_steps` = every operation in the derivation; `%` and a lone `(n)` are read whichever way reproduces the answer. 18 of 6,959 derivations (0.3%) contain words and keep only the gold answer
- [ ] **R1:** the transfer uses FinQA's C, cutoff and F1 threshold unchanged (main task), and pools whichever TAT-QA settings were run

**R2 (Yvette)**

- [ ] **R2:** the hint is appended to the question after a space; the report comes first in the prompt, so its tokens and positions don't change (the script checks), and attention over the report is compared position by position
- [ ] **R2:** candidates = a seeded random sample of answers with a hallucinated number (default greedy); the focus number = the first hallucinated one that isn't the cut-off last number of a truncated answer and isn't the value 100 (mostly the "× 100" of a percent formula; `--keep-100` allows it)
- [ ] **R2:** the new answer uses the same decoding setting with a seed from the id; with a sampling setting a different answer can also come from the draw, so greedy is the default
- [ ] **R2:** the new answer's numbers are labeled with the A4 rules, with the hint as part of the question (numbers in the hint count as `question`)
- [ ] **R2:** counterpart = the new answer's number whose text before it on its line (up to 40 characters) is most similar to the focus number's (difflib ratio); `counterpart_index` overrides it
- [ ] **R2:** the plots use the same head for every pair: B4's top lookback head (`--head` picks another); the right panel is lookback per layer, averaged over heads

## 2. AI-written code we must be able to explain

README §3: each of us can explain every file in our folders line by line, and has checked its tests' expected values by hand. By Dec 11, each must also be able to explain the other's files.

**Both**

- [x] `common/prompt.py`: check `tests/test_prompt.py` (including `encode_answer`, used by A7 and B1) (checked Nov 23 by Claude: all 12 tests pass, including the real Qwen tokenizer test (cached locally); `tokens_in_span` offsets, junk-line dropping and the template text recomputed against README 6.1)
- [x] `common/splits.py`: check `tests/test_splits.py` (checked Nov 23 by Claude: 9 tests pass; the 700/150 limits and the at-most-one-report overflow recomputed by hand)
- [x] `common/numbers.py`: check every expected value in `tests/test_numbers.py`, and add a few strings of our own in the styles the model actually uses (README A2). Nov 23: LaTeX `\%` read as a percent; the month-day skip matches real month names only (checked Nov 23 by Claude: 66 tests pass; every value and percent flag in CASES, the spans and the `same_value` pairs recomputed by hand (e.g. 4,200 vs 4,215 is 0.36%, 0.141 vs 0.14134 is 0.24%); added 4 passing edge cases, a LaTeX-percent span test and 5 strict xfails for known limits (10-K/COVID-19, `( 2.3 ) %`, `30 june`, `24 \text{ million}`, `**1.**`))
- [x] `common/io.py`: the shared file helpers, plus `load_config` (`extends`), `run_dir` (`run_name` folders) and `check_model` added Nov 7; check `tests/test_io.py` (checked Nov 23 by Claude: 6 tests pass; `extends` is a shallow, top-level override (a sub-config that names `decoding` replaces the whole block); `read_jsonl`/`group_records` are tested in `tests/test_confidence.py`)
- [x] `labels/agreement.py`: check the hand-worked kappa and rates in `tests/test_agreement.py` (checked Nov 23 by Claude: 5 tests pass; kappa (p_o 3/4, p_e 8/16, kappa 0.5) and the four rates (2/3, 3/4, 1/3, 1/2) recomputed by hand)

**Eduardo (A)**

- [x] `data/build_subset.py`: sanity-check the counts it prints on Colab (checked Nov 23 by Claude: not on Colab: reran its logic locally with the cached Qwen tokenizer; 8,281 questions / 799 reports, 154 yes/no, 7,347 eligible (780 dropped), prompt tokens mean 995 / median 1,006 / 241–1,496, context 906 / 916 / 146–1,407, splits 700/329, 151/66, 149/62 all match LOG Nov 6; the same 1,000 ids and splits as `data/finqa_subset.jsonl`)
- [x] `common/program.py`: check `tests/test_program.py` (checked Nov 23 by Claude: 18 tests pass; each step value recomputed (94/5735, 9896/0.236, 1.05^2 = 1.1025, (4215+3900)/2 = 4057.5); the last step equals FinQA's `exe_ans` on all 8,127 numeric questions (rerun))
- [x] `gen/generate.py`: check `tests/test_generate.py`, especially the decoding arguments each setting turns into (`plain_generation_config`, moved out of `main` on Nov 17, is also used by R2) (checked Nov 23 by Claude: 12 tests pass; the kwargs for all four setting shapes match README A3 (num_beams 1, repetition penalty 1, and top_k 0 / top_p 1 / temperature 1 only when sampling); `plain_generation_config` needs the model and has no unit test)
- [x] `labels/label.py`: check every expected label in `tests/test_label.py` against README §6.3, including the `constant` cases added Nov 6 and the comma constant and LaTeX-percent cases added Nov 23 (checked Nov 23 by Claude: 20 tests pass; all 13 parametrized cases and the constant, comma-constant, LaTeX-percent and copied-before-computed tests worked through README 6.3 by hand (e.g. 7.5% vs the question's 7% is 6.7% off, so wrong_calc))
- [ ] `baseline/confidence.py`: explain why the logits at position *t* score token *t + 1*; check `tests/test_confidence.py` (checked Nov 23 by Claude: 5 tests pass; `answer_logit_rows` slices [6, 7, 8] for both return shapes and `number_stats` (-1.25, -2.0, 0.7) recomputed by hand; the torch test runs locally. Left open: the explanation of the t / t + 1 shift is Eduardo's)
- [x] `analysis/stage1.py` and `analysis/stage1.ipynb`: check the hand-worked example in `tests/test_stage1.py` (checked Nov 23 by Claude: 6 tests pass; accuracy 2/3, fabricated 1/5, answers with one 1/3, shares 2/5, tokens 20, cut 1/3, differs 2/3 and 3 distinct finals recomputed by hand; the notebook only calls `build_table` and `write_outputs`)

**Yvette (B)**

- [ ] `analysis/b0_attention_plot.py`: check `tests/test_b0.py`
- [ ] `features/attention.py` and `features/extract.py`: check the hand-worked lookback and HF values in `tests/test_features.py`. Nov 23: the model now loads in float32 by default (`--dtype`; float16 eager attention gave NaN in layers 1–27), and the row-sum check counts NaN/inf rows and stops on the first answer if it has any. Same `--dtype` in `analysis/b0_attention_plot.py` and `analysis/hint_test.py run`
- [ ] `detector/train_eval.py`: check the hand-worked F1 and threshold cases and the two tasks' targets in `tests/test_detector.py`
- [ ] `detector/three_way.py`: check the hand-worked per-class F1, macro-F1 and confusion matrix in `tests/test_three_way.py`; Nov 23: config via `load_config`, folders via `run_dir`
- [ ] `analysis/stage2.py` and `analysis/stage2.ipynb`: check `tests/test_stage2.py`
- [ ] `analysis/errors.py` and `analysis/errors.ipynb`: check `tests/test_errors.py`; Nov 23: stops if a test score's row no longer matches the labels file
- [ ] `analysis/hint_test.py` (R2): check `tests/test_hint_test.py` (gold program numbers, focus choice, counterpart matching); Nov 23: `load_config`/`run_dir`, `model` check and field, unlabelable new answers skipped

**R1 (both)**

- [ ] `common/tatqa.py`, `data/build_tatqa.py`, `detector/transfer.py`: check `tests/test_tatqa.py` and `tests/test_transfer.py`

## 3. Analysis and writing, by step

⏳ waiting on an earlier step · 🟢 ready to start · ✅ done

Claude makes the plots and tables; reading them and drawing conclusions is ours. A = Eduardo, B = Yvette. Order and details: `MANUAL_STEPS.md`.

| Step | What's ours | Who | Status |
|---|---|---|---|
| A1 | Run `data/build_subset.py`; fill the A1 stats line in `LOG.md` (a run) | A | ✅ done Nov 6 (`data/finqa_subset.jsonl`, stats in `LOG.md`) |
| A3 | Run `gen/generate.py` for your settings; greedy `--limit 50` first (a run) | A: greedy, beam5 · B: temp07, temp12, topp09 | A: ✅ greedy and beam5 done (1,000 each, Nov 6–7; stats in `LOG.md`) · B: 🟢 ready to run |
| B0 | Run `analysis/b0_attention_plot.py` (after A1); explain the plot to professor: what the heads attend to | B | 🟢 ready to run |
| A4 | Run `labels/label.py` for your settings; label counts into `LOG.md` (a run) | A: greedy, beam5 · B: temp07, temp12, topp09 | A: 🟢 greedy and beam5 labeled Nov 23, **provisional** (pre-hand-check rules; counts in `LOG.md`) · B: ⏳ needs B-3 |
| A5 | Hand-label the same 50 numbers without seeing the auto labels; run `labels/agreement.py`; go through the disagreements; if under ~90%, decide how §6.3 changes | both | 🟢 `labels/handcheck.csv` ready (Nov 6) |
| A6 | RQ0: what decoding does to accuracy and fabrication, and why ("Our reading" in `analysis/stage1.ipynb`) | A | 🟢 provisional greedy + beam5 table, chart and executed notebook (Nov 23); final needs B-6 and the hand-check rules |
| A6 | Stage 1 error analysis (Nov 23–29) | A | ⏳ needs A6 results |
| A7 | Run `baseline/confidence.py` for your settings (a run) | A: greedy, beam5 · B: temp07, temp12, topp09 | A: 🟢 greedy and beam5 done Nov 23, **provisional** (fixed-code labels; UNLV GPU) · B: ⏳ needs B-6 |
| B1–B2 | `features/extract.py --limit 20`: row sums ~1, lookback within 0–1 (B1 done); then the full run for your settings (B2 done) | B: the check · A, B: full runs | ⏳ needs A4 |
| Papers | Read the related papers: how they aggregate attention into features for hallucination detection, and which features we replicate. What we learned about the area goes in the report (office hours, `LOG.md`) | both | 🟢 ready |
| B3 | Read `detector/results.csv` (both tasks): which detectors beat the confidence baseline, given the CIs; and the 3-way confusion matrix (`three_way_confusion.png`) | B | ⏳ needs A7, B2 |
| B4 | **RQ2 (main contribution):** do detectors flag correct computed numbers, and does that support the hypothesis? (RQ2 cell in `analysis/stage2.ipynb`) | B | ⏳ needs B3 |
| B4 | RQ3: which layers and heads carry the signal; does a top-100-head detector keep most of the AUROC? (RQ3 cell) | B | ⏳ needs B3 |
| B4 | Bridge: does the lookback ratio fall as temperature rises? (bridge cell) | B | ⏳ needs B2 |
| B5 | Explain 10 false positives, 10 false negatives, and where `wrong_calc` falls, in our own words (notes files, then `analysis/errors.ipynb`) | A: false positives, `wrong_calc` · B: false negatives | ⏳ needs B3 |
| R1 | Optional: hand-check ~30 TAT-QA labels, then interpret the transfer (`detector/transfer_results.csv`) | A: TAT-QA subset, greedy · B: temp12, transfer · both: hand-check | ⏳ needs B3 and the R1 runs (`MANUAL_STEPS.md`, "R1 (optional)") |
| R2 | Optional: write the hints by hand; check the counterparts; then, in our own words, what changes in the attention when a hallucination goes away (`hint_*.png`, `hint_results.csv`) | B | ⏳ needs B4 |
| R3 | Report: own sections, then both edit the draft | both | ⏳ |
| Dec 7–10 | Report edits; each can explain the other's half | both | ⏳ |
