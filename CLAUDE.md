# Notes for Claude

- Course AI-use rule (README §3): write code, tests, plots and tables, but never the analysis. Don't interpret results, draw conclusions, or write report or poster text.
- Keep `ANALYSIS_TODO.md` current in the same commit as the work:
  - new or changed code → list it in section 2
  - a choice made in code → add it to section 1 and to `LOG.md`
  - a step's data, plots or tables are ready → mark its analysis row 🟢 ready
  - update the "Last updated" line
- Keep `MANUAL_STEPS.md` current too: when new code needs a manual run, a hand check or a decision, add the step (commands, what to check, what to record in `LOG.md`, what to commit), and update its "Last updated" line.
