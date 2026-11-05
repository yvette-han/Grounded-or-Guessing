# Notes for Claude

- **Branches:** push work directly to `yvette` (never a new `claude/...` branch). Pull `yvette` first and merge in anything new. Push to `main` only when the user asks; otherwise it's updated through pull requests.
- **Roles:** Person A (Stage 1) is Eduardo; Person B (Stage 2) is Yvette, the user.
- **AI-use rule (README §3):** write code, tests, plots and tables, never the analysis. Don't interpret results, draw conclusions, or write report or poster text.
- **`ANALYSIS_TODO.md`:** update it in the same commit as the work:
  - new or changed code → section 2
  - a choice made in code → section 1, and a line in `LOG.md`
  - data, plots or tables ready for a step → mark its section 3 row 🟢
  - update the "Last updated" line
- **`MANUAL_STEPS.md`:** when new code needs a manual run, a hand check or a decision, add it to the right person's checklist (command, what to check, what to record in `LOG.md`, what to push). Keep both people able to work in parallel, and update the timeline table and the "Last updated" line.
- **`LOG.md`:** add decision lines above the `## Run notes` headings; those sections belong to Person A and Person B.
