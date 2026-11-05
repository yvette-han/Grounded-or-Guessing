# LOG

One line per decision or dead end, dated (README §3).
Example: `Nov 14: switched tolerance to 1% because 'about 12%' was being labeled fabricated`

- Nov 5: Repo created with README, `config.yaml` and `requirements.txt` from the project plan.
- Nov 5: A1 pools FinQA train+dev+test (8,281 questions, 799 company/year reports) and re-splits by report, because 243 of the 278 reports in FinQA's test file also appear in its train file. FinQA is pinned to commit `0f16e28`.
- Nov 5: A1 drops the 154 yes/no questions (`exe_ans` is "yes"/"no", no numeric gold answer) → 8,127 left.
- Nov 5: Report text drops the 13,057 FinQA lines that are only "." and puts each sentence on its own line.
- Nov 5: Prompt is sent through the chat template as the user turn (`chat_template: true`). The `max_context_tokens` filter counts the full templated prompt.
- Nov 5: `ctx_start`/`ctx_end` are a half-open token span; a token that straddles the edge of the report (e.g. "." merged with "\n\n") counts as context.
- Nov 5: Split assignment: shuffle reports with the seed, fill train then val then test by question count; a split can go over by one report.
- A1 stats (fill in after running with the Qwen tokenizer): questions after length filter = ?, avg context tokens = ?, split sizes = ?
- Nov 5: A2 `common/numbers.py`: on top of README 6.3 it skips the day after a month name (`december 31`), ordinals (`3rd`), digits inside words (`Q4`), and list markers `1.`, `1)`, `(1)` at a line start.
- Nov 5: A2 matching: relative tolerance; `12%` equals `12` and `0.12`; a scale-word number vs a bare number tries ones/thousands/millions/billions; sign ignored by default.
- Nov 5: A2 check: 99.57% (18,658 / 18,738) of the numeric arguments in FinQA's gold programs are found in their report by `extract_numbers` + `find_match`. The 80 misses are years and days (skipped by rule) and constants not in the report (`1000000`, `3`).
- Nov 5: B0 `analysis/b0_attention_plot.py` generates its own greedy answer (A3 hasn't run), then does one teacher-forced pass over prompt + answer tokenized as one string. Default row: the number's first digit token; `--row predicting` uses the token before it.
- Nov 5: `common/prompt.py` gained `tokens_in_span` (character span → token indices, for A4 and B1) and `encode` now also returns `ctx_chars`.
- Nov 5: A3 `gen/generate.py`: transformers fills any decoding parameter we leave unset from the model's own generation config (Qwen: temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05), so A3 replaces it and switches off everything a setting doesn't name.
- Nov 5: A3 runs batches of 8 (left padding) with one seed per batch from (seed, setting, first id), in a fixed shuffled question order; it resumes from an existing output file. Added `n_new_tokens`, `hit_max_tokens` to the answers file.
- Nov 5: `common/prompt.py` gained `encode_answer` (prompt + answer as one string, checks the prompt's tokens don't change); B0 now uses it.
- Nov 5: A1 now stores `gold_steps`: each gold program step's value, from `common/program.py`, which follows FinQA's own executor. Its last step equals FinQA's `exe_ans` on all 8,127 numeric questions.
- Nov 5: A2 fix: FinQA writes dates as `dec . 29`; the day is now skipped there too.
- Nov 5: A4 adds a label `question` (number only in the question, ~10% of FinQA questions contain numbers) between computed and wrong_calc; left out of main training. `wrong_calc` = every earlier number in the answer is copied/computed/question and at least one is copied.
- Nov 5: A4 token span = the number's digits (`tok_start`/`tok_end`, from the first answer token); whole mention kept as `char_start`/`char_end`. `--handcheck 50` samples 50 random numbers from one setting and never overwrites an existing sheet.
- Nov 5: A4 smoke test on fake answers: 3 of 14 deliberately wrong values matched some report number within 1% and came out `copied` (e.g. `3.473` vs `3.5%`). Check how often this happens on real answers in A5.
- Nov 5: A5 `labels/agreement.py`: joins the two hand-labeled sheets and the key by `row`, prints A vs B (percent + Cohen's kappa), A vs auto, B vs auto, A=B rows vs auto, a hand-vs-auto confusion table, and writes the disagreements to `labels/handcheck_disagreements.csv`. Hand-check CSVs are written as UTF-8 with BOM so Excel shows ⟦ ⟧.
- Nov 5: A6 `analysis/stage1.py` + `analysis/stage1.ipynb`: accuracy = last number in the answer vs gold (within `label_tolerance`), plus gold-anywhere; fabrication = share of numbers labeled fabricated, plus share of answers with one; distinct answers = % differing from greedy and mean distinct finals per question; answer length from `n_new_tokens`. Settings compared on their common questions; 95% CIs bootstrap over reports.
- Nov 5: A7 `baseline/confidence.py`: one teacher-forced pass per answer; per number, mean and min log-prob and mean entropy (nats) over its digit tokens, under the model's own distribution for every setting. Added `row` (line in `labels/numbers_{setting}.jsonl`) to the confidence and feature file contracts as the join key.
