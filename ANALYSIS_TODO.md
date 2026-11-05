# Analysis to do (ours, not AI's)

README §3: AI may write the code, but *"the analysis and the knowledge that you produce must be your own."* This file tracks the work in that second group, plus the AI-written code we still need to understand. Claude updates it whenever it adds code or results; we tick items off.

_Last updated: Nov 5, after the A1 code._

## 1. Choices made in AI-written code, to review

Each one is also in `LOG.md`. Keep it, change it or replace it; once we've decided, it's our decision.

- [ ] **A1:** pool FinQA train+dev+test and re-split by report (243 of FinQA's 278 test reports are also in its train file)
- [ ] **A1:** drop the 154 yes/no questions
- [ ] **A1:** drop FinQA's `"."`-only lines; one sentence per line in the report text
- [ ] **A1:** send the prompt through the chat template; `max_context_tokens` counts the full templated prompt
- [ ] **A1:** split fill rule: reports shuffled, then train → val → test by question count; a split can go over by one report
- [ ] **A1 → A3, B1:** the report's token span is half-open (`ids[ctx_start:ctx_end]`); a token straddling the report's edge counts as context

## 2. AI-written code we must be able to explain

README §3: each of us can explain every file in our folders line by line, and each function is checked on an example where we already know the answer.

- [ ] `common/prompt.py` (both): explain it; check the expected strings in `tests/test_prompt.py` by hand
- [ ] `common/splits.py` (both): explain it; check the expected values in `tests/test_splits.py` by hand
- [ ] `data/build_subset.py` (A): explain it; sanity-check the counts it prints on the Colab run

## 3. Analysis and writing, by step

⏳ waiting on an earlier step · 🟢 ready to start · ✅ done

Claude can make the plots and tables for these; reading them and drawing conclusions is ours.

| Step | What's ours | Status |
|---|---|---|
| A1 | Run `data/build_subset.py` with the Qwen tokenizer and fill the stats line in `LOG.md` (a run, not analysis) | 🟢 ready |
| B0 | Explain the meeting plot to Niranjan: what the heads attend to | ⏳ needs plot code |
| A5 | Hand-label the same 50 numbers each without seeing the auto labels; compute agreement; if under ~90%, decide how to change §6.3 | ⏳ needs A3, A4 |
| A6 | RQ0: what decoding does to accuracy and fabrication rate, and why | ⏳ needs A3, A4 |
| A6 | Stage 1 error analysis (Nov 23–29) | ⏳ needs A6 results |
| B3 | Read the results table: which detectors beat the confidence baseline, given the confidence intervals | ⏳ needs A7, B2 |
| B4 | **RQ2 (main contribution):** do detectors flag correct computed numbers, and does that support the hypothesis? | ⏳ needs B3 |
| B4 | RQ3: which layers and heads carry the signal, and what that means | ⏳ needs B3 |
| B4 | Bridge: does the lookback ratio fall as temperature rises, and what that says | ⏳ needs B2 |
| B5 | 10 false positives, 10 false negatives, where `wrong_calc` falls: explain each pattern in our own words | ⏳ needs B3 |
| R1 | Stretch: interpret the FinQA → TAT-QA transfer | ⏳ needs B3 + TAT-QA |
| R2 | Poster text and takeaways | ⏳ |
| R3 | Report: each writes their own sections, both edit the draft | ⏳ |
| Dec 7–8 | Rehearse: each explains the other's half | ⏳ |
