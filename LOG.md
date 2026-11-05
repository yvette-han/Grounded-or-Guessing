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
