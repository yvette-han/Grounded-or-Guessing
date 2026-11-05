# Manual steps

Claude writes the code. We run it on Colab, check the output, and write the analysis. Tick items off in `ANALYSIS_TODO.md` as you finish them.

_Last updated: Nov 5._

**Contents:** [Who does what](#who-does-what) · [Timeline](#timeline) · [Setup and sharing](#setup-and-sharing) · [Eduardo (Person A)](#eduardo-person-a) · [Yvette (Person B)](#yvette-person-b) · [Hand-check](#hand-check) · [Script options and checks](#script-options-and-checks) · [R1 (optional)](#r1-optional)

On Colab, each code block below is one cell. Shell commands start with `!`; change folders with `%cd`.

---

## Who does what

Each of us runs the full chain (generate → label → confidence → features) for our own decoding settings, so both GPUs work at the same time.

| | Eduardo (Person A, Stage 1) | Yvette (Person B, Stage 2) |
|---|---|---|
| Decoding settings | `greedy`, `beam5` (beam is ~2x slower) | `temp07`, `temp12`, `topp09` |
| Runs and writes up | A1, A6, Stage 1 error analysis | B0, B1 check, B3, B4 |
| Together | A5 hand-check | A5 hand-check |
| B5 | the 10 false positives, where `wrong_calc` falls | the 10 false negatives |
| Reviews in `ANALYSIS_TODO.md` | items A1–A7, files marked (A) | items B0–B5, files marked (B) |
| Poster | dataset, labeling, decoding chart | attention diagram, AUROC table, head/layer heatmap |

Running a script doesn't change who owns it. By Dec 9, each of us must be able to explain the other's half.

### Timeline

🔁 = hand-off: push and merge it, because the other person needs it next.

| When | Eduardo (A) | Yvette (B) |
|---|---|---|
| Nov 5–8 | A-0 setup · A-1 subset 🔁 · A-2 reviews | B-0 setup · B-1 B0 plot for the meeting · B-2 reviews |
| Nov 9–12 | A-3 first 50 greedy answers + hand-check sheet, **by Nov 12** 🔁 · then generate `greedy`, `beam5` | B-3 generate `temp07`, `temp12`, `topp09` · B-4 B1 check (after A-3) |
| Nov 12–14 | A-4 hand-label 🔁 · run the agreement · decide the rules together | B-5 hand-label 🔁 · decide the rules together |
| Nov 14–20 | A-5 label, confidence, features for your settings 🔁 | B-6 label, confidence, features for your settings 🔁 |
| Nov 16–22 | A-6 Stage 1 results (needs B-6) | B-7 detectors (needs A-5) |
| Nov 23–29 | A-7 Stage 1 error analysis | B-8 Stage 2 analysis + B5 examples 🔁 |
| Nov 30–Dec 6 | A-8 B5 false positives · poster, report · optional R1 | B-9 B5 false negatives · poster, report · optional R1 |
| Dec 7–8 | Rehearse Yvette's half | Rehearse Eduardo's half |

**Label after Nov 14, not before.** If the hand-check changes the rules (README §6.3), every label, confidence score and feature file must be redone. Generating answers doesn't depend on the rules, so generate first. The one exception is the first 50 greedy answers, which the hand-check needs.

---

## Setup and sharing

**Once:**

1. Yvette adds Eduardo under the repo's **Settings → Collaborators**.
2. Eduardo creates a branch, e.g. `eduardo`: on GitHub, open the branch menu, type the name, and choose "Create branch … from main". Yvette uses `yvette`.
3. Each makes a GitHub token (GitHub → Settings → Developer settings):
   - Yvette (owner): **fine-grained**, this repo only, "Contents: read and write".
   - Eduardo (collaborator): **classic**, `repo` scope.

   Save it in Colab's 🔑 Secrets panel as `GITHUB_TOKEN`, with "Notebook access" on.
4. Yvette creates the Drive folder `GoG-shared` and shares it with Eduardo as Editor. Eduardo: "Shared with me" → right-click it → **Organize → Add shortcut → My Drive**.
5. Each clones into their own Drive. Never let two Colab sessions use the same clone.

```python
from google.colab import drive, userdata
drive.mount("/content/drive")
token = userdata.get("GITHUB_TOKEN")
```

```python
%cd /content/drive/MyDrive
!git clone -b main https://{token}@github.com/yvette-han/Grounded-or-Guessing
```

**Every session.** First set **Runtime → Change runtime type → T4 GPU**, then:

```python
from google.colab import drive
drive.mount("/content/drive")
SHARED = "/content/drive/MyDrive/GoG-shared"
```

```python
%cd /content/drive/MyDrive/Grounded-or-Guessing
!git config user.name "Your Name" && git config user.email "you@example.com"
!git fetch origin && git checkout <your-branch> && git pull origin main
!pip install -q -r requirements.txt
!python -m pytest -q
```

All tests should pass, including the one that downloads the Qwen tokenizer. Free Colab gives a few GPU hours a day; long runs resume, so spread them over several days.

**Where files go:**

- **Small files** (subset, answers, labels, confidence, results, figures, notebooks) go in git. Push to your branch, open a pull request into `main` on GitHub, and merge it (or ask Claude to). Each setting has its own files, so merges don't conflict.
- **Feature files** (`.npz` from `features/extract.py`, ~5 MB per 1,000 numbers) go in `{SHARED}/features`, not git.
- **`LOG.md`:** write your run notes under your own heading at the bottom (`## Run notes: Person A` or `## Run notes: Person B`). Shared decisions go above those headings.

**Push from Colab:**

```python
!git add <files> LOG.md
!git commit -m "<what you did>" && git push origin HEAD:<your-branch>
```

Without git: download the files from Colab's Files panel, then on GitHub open your branch → **Add file → Upload files**.

---

## Eduardo (Person A)

Your settings: `greedy`, `beam5`.

**A-0 · Setup** · Nov 5–8. Do [Setup and sharing](#setup-and-sharing).

**A-1 · Build the subset** · do this first; Yvette needs it. (If Yvette already ran it, skip this: it's the same file.)

```python
!python data/build_subset.py
```

- Check: `gold programs whose last step differs from FinQA's answer: 0`.
- In `LOG.md`, replace the `A1 stats (fill in ...)` line with the printed counts.
- Push `data/finqa_subset.jsonl` and `LOG.md`; merge; tell Yvette. 🔁

**A-2 · Reviews** · any time. In `ANALYSIS_TODO.md`: section 1 items A1–A7, and section 2 files marked (A) or (both). Add a few test strings of your own to `tests/test_numbers.py`.

**A-3 · First 50 greedy answers** · due Nov 12.

```python
!python gen/generate.py --setting greedy --limit 50
!python labels/label.py --setting greedy --handcheck 50
```

- Don't open `labels/handcheck_key.csv` or `labels/numbers_greedy.jsonl` before the hand-check: they show the auto labels.
- Push `gen/answers_greedy.jsonl`, `labels/numbers_greedy.jsonl` and `labels/handcheck.csv` (not the key yet); merge; tell Yvette. 🔁

Then start the full runs. They take hours and resume after a disconnect:

```python
!python gen/generate.py --setting greedy
!python gen/generate.py --setting beam5
```

- If `beam5` runs out of GPU memory, add `--batch-size 4`.
- Push each `gen/answers_*.jsonl` when it's done.

**A-4 · Hand-check** · Nov 12–14. Fill `label_A` (see [Hand-check](#hand-check)) and push `labels/handcheck_A.csv`. 🔁 When Yvette's sheet is merged, run the agreement (step 5 there) and decide the rules with Yvette.

**A-5 · Label, confidence, features** · after the rules are settled. Use the row choice from B-4: add `--row predicting` to the last line only if Yvette chose it.

```python
!python labels/label.py --setting greedy
!python labels/label.py --setting beam5
!python baseline/confidence.py --setting greedy beam5
!python features/extract.py --setting greedy beam5 --out-dir {SHARED}/features
```

- Check each output (see [Script options and checks](#script-options-and-checks)). Copy the label counts into `LOG.md`.
- Push `labels/numbers_greedy.jsonl`, `labels/numbers_beam5.jsonl`, `baseline/confidence_greedy.npz`, `baseline/confidence_beam5.npz` and `LOG.md`; merge. 🔁

**A-6 · Stage 1 results** · after B-6 is merged. Pull `main`. Open `analysis/stage1.ipynb` (File → Open notebook → Google Drive) and run all, or run `!python analysis/stage1.py`. Write the "Our reading" cell. Push the notebook and `analysis/figures/stage1_*`.

**A-7 · Stage 1 error analysis** · Nov 23–29. Write it in our own words in `analysis/stage1.ipynb`, below "Our reading".

**A-8 · B5: false positives and `wrong_calc`** · after B-8. Pull `main`. Read `analysis/figures/b5_examples.md` and `b5_scores_by_label.png`. In `analysis/notes/b5_A.md`, explain each of the 10 false positives, then the pattern, then where `wrong_calc` falls. Push.

**A-9 · Poster and report.** Your parts: dataset and labeling, the decoding chart (`stage1_bars.png`), the Stage 1 results. Then edit the full draft together.

**A-10 · Rehearse** · Dec 7–8. Be able to explain Yvette's half: `features/`, `detector/`, `analysis/stage2.py`, and her readings.

---

## Yvette (Person B)

Your settings: `temp07`, `temp12`, `topp09`. **This week (Nov 5–8): B-0, B-1, B-2.**

**B-0 · Setup** · Thu–Fri Nov 5–6. Do [Setup and sharing](#setup-and-sharing): add Eduardo as a collaborator, make your fine-grained token, create and share `GoG-shared`, and clone (your branch: `yvette`).

**B-1 · Subset and B0 plot for the meeting** · Fri–Sun Nov 6–8.

- **Subset:** if Eduardo has merged A-1, pull `main`. If not, run A-1 yourself: `!python data/build_subset.py`. Check it, fill the A1 stats line in `LOG.md`, push, merge, and tell Eduardo to skip A-1.
- **Plot:**

```python
!python analysis/b0_attention_plot.py
```

- Check: `attention rows sum to` about 1.
- The script lists the answer's numbers as "in report" or "not in report". Also plot one that is not in the report (`--number <its index>`), and try `--row predicting` once.
- Each run overwrites `b0_attention.png` and `b0_lookback.png`, so rename the ones you want to keep first:

```python
!mv analysis/figures/b0_attention.png analysis/figures/b0_attention_in_report.png
!mv analysis/figures/b0_lookback.png analysis/figures/b0_lookback_in_report.png
```

- Push the PNGs you keep.
- Before the meeting: write a few sentences, in your own words, on what the plot shows (in your `LOG.md` run notes). Be able to explain `analysis/b0_attention_plot.py` and `features/attention.py`. Bring the plots and the A1 stats.

**B-2 · Reviews** · before Nov 12. In `ANALYSIS_TODO.md`: section 1 items B0–B5, and section 2 files marked (B) or (both). Start with the row choice (you decide it in B-4), the B1 definitions, and the B2 frequency units: check how Qi et al. (2026) normalize frequency before Nov 14.

**B-3 · Generate your settings** · from Mon Nov 9, or earlier if you have spare GPU time. Needs only the subset.

```python
!python gen/generate.py --setting temp07
!python gen/generate.py --setting temp12
!python gen/generate.py --setting topp09
```

- Run one per cell; each resumes after a disconnect.
- Push each `gen/answers_*.jsonl` when it's done. Don't label yet.

**B-4 · B1 check** · Nov 12, after A-3 is merged. This writes to a local scratch folder:

```python
!git pull origin main
!python features/extract.py --setting greedy --limit 20 --out-dir features/check
!python features/extract.py --setting greedy --limit 20 --row predicting --out-dir features/check_predicting
```

- Check: row sums about 1; lookback and HF values within 0–1. Note the result in `LOG.md` (README B1 is then done).
- Decide the row choice (`number` or `predicting`) and tell Eduardo. Everyone uses it from now on.
- Don't open `labels/numbers_greedy.jsonl` before the hand-check.

**B-5 · Hand-check** · Nov 12–14. Fill `label_B` (see [Hand-check](#hand-check)) and push `labels/handcheck_B.csv`. 🔁 Then go through the disagreements with Eduardo.

**B-6 · Label, confidence, features** · after the rules are settled. Add `--row predicting` to the last line only if you chose it.

```python
!python labels/label.py --setting temp07
!python labels/label.py --setting temp12
!python labels/label.py --setting topp09
!python baseline/confidence.py --setting temp07 temp12 topp09
!python features/extract.py --setting temp07 temp12 topp09 --out-dir {SHARED}/features
```

- Check each output (see [Script options and checks](#script-options-and-checks)). Copy the label counts into `LOG.md`.
- Push `labels/numbers_temp07.jsonl`, `numbers_temp12.jsonl`, `numbers_topp09.jsonl`, `baseline/confidence_temp07.npz`, `confidence_temp12.npz`, `confidence_topp09.npz` and `LOG.md`; merge. 🔁

**B-7 · Detectors (B3)** · after A-5 is merged. Pull `main`, and check that `{SHARED}/features` has all five settings.

```python
!python detector/train_eval.py --features-dir {SHARED}/features
```

- Check: the settings line, the dropped counts, and that the test split has fabricated numbers. Note the chosen C and cutoffs in `LOG.md`.
- Push `detector/results.csv` and `detector/test_scores.csv`.

**B-8 · Stage 2 analysis (B4) and the B5 examples** · Nov 23–29.

- Open `analysis/stage2.ipynb`, set `FEATURES_DIR = Path(SHARED) / "features"` in the first code cell, run all, and write the three "Our reading" cells. (Or run `!python analysis/stage2.py --features-dir {SHARED}/features`.)
- Make the B5 examples for both of you: `!python analysis/errors.py`
- Push `analysis/stage2.ipynb`, `analysis/figures/stage2_*` and `analysis/figures/b5_*`; tell Eduardo. 🔁

**B-9 · B5: false negatives.** In `analysis/notes/b5_B.md`, explain each of the 10 false negatives, then the pattern. Push. At the end, one of us copies `b5_A.md` and `b5_B.md` into `analysis/errors.ipynb`.

**B-10 · Poster and report.** Your parts: the attention diagram (`b0_attention.png`), the AUROC table (`detector/results.csv`), the head/layer heatmap (`stage2_heads.png`), the Stage 2 results. Then edit the full draft together.

**B-11 · Rehearse** · Dec 7–8. Be able to explain Eduardo's half: `data/`, `gen/`, `labels/`, `baseline/`, `analysis/stage1.py`, and Eduardo's readings.

---

## Hand-check

Both of us, Nov 12–14 (README A5).

1. Copy `labels/handcheck.csv` into your own Google Sheet (File → Import → Upload) or open it in Excel.
2. Fill only your column (`label_A` or `label_B`) with `copied`, `computed`, `question`, `wrong_calc` or `fabricated` (README §6.3). Each row shows the answer with the number marked ⟦like this⟧, the question, the gold answer, program and steps, and the report.
3. Don't look at the key, the labels file, or the other person's sheet.
4. Download it as CSV and save it as `labels/handcheck_A.csv` or `labels/handcheck_B.csv`. Keep the `row` and `id` columns; sorting is fine.
5. Eduardo, once both sheets are merged:

```python
!python labels/agreement.py --sheet-a labels/handcheck_A.csv --sheet-b labels/handcheck_B.csv
```

6. Put the agreement rates in `LOG.md`. Push `labels/handcheck_key.csv` and `labels/handcheck_disagreements.csv`. Go through the disagreements together.
7. Decide which agreement the ~90% in README A5 refers to, and whether §6.3 changes. If it does, tell Claude, wait for the fix, then each of us relabels our settings.

---

## Script options and checks

| Script | Useful options | Check in the output |
|---|---|---|
| `data/build_subset.py` | | `gold programs whose last step differs ...: 0`; counts for `LOG.md` |
| `analysis/b0_attention_plot.py` | `--number k`, `--id <id>`, `--row predicting`, `--heads 10:3,20:5` | `attention rows sum to` about 1 |
| `gen/generate.py` | `--setting`, `--limit N`, `--batch-size 4` (if out of memory), `--overwrite` | `answers whose re-tokenization changes the prompt tokens: 0`; how many were cut at max tokens |
| `labels/label.py` | `--setting`, `--handcheck N` (never overwrites an existing sheet) | label counts (into `LOG.md`); no skipped answers |
| `labels/agreement.py` | `--sheet-a`, `--sheet-b`, `--key`, `--out` | agreement rates (into `LOG.md`) |
| `baseline/confidence.py` | `--setting ...` (default: every labeled setting) | number count matches the labels file; `NaNs: 0`; no skipped answers |
| `features/extract.py` | `--setting ...`, `--limit N`, `--row predicting`, `--out-dir` | row sums about 1; lookback, `hf_ctx`, `hf_gen` within 0–1; NaN only for numbers at the start of an answer. Needs more GPU memory than generation |
| `detector/train_eval.py` | `--features-dir`, `--out-dir` | settings used; dropped counts by label; the test split has fabricated numbers |
| `analysis/stage1.py` | | every setting is in the table; missing ones are named |
| `analysis/stage2.py` | `--features-dir` | one row per detector; every setting in the bridge |
| `analysis/errors.py` | `--detector`, `--pick extreme`, `-n` | 10 false positives and 10 false negatives |
| `data/build_tatqa.py` | | about 0.3% of derivations don't reproduce their answer |
| `detector/transfer.py` | `--features-dir`, `--tatqa-features-dir` | refuses features made with a different `--row` or HF cutoffs |

**To compare both row choices:** extract with `--row predicting --out-dir {SHARED}/features_predicting`, then run `detector/train_eval.py --features-dir {SHARED}/features_predicting --out-dir detector/predicting`.

---

## R1 (optional)

Only if time allows, after B-7. TAT-QA outputs go to `tatqa/` and `{SHARED}/tatqa_features`, so no FinQA file is touched. Use the same `--row` as for FinQA.

**Eduardo:** build the subset (Yvette waits for it), then run the `greedy` chain:

```python
!python data/build_tatqa.py
!python gen/generate.py --setting greedy --subset data/tatqa_subset.jsonl --out-dir tatqa/gen
!python labels/label.py --setting greedy --subset data/tatqa_subset.jsonl --answers-dir tatqa/gen --out-dir tatqa/labels --handcheck 30
!python baseline/confidence.py --setting greedy --answers-dir tatqa/gen --labels-dir tatqa/labels --out-dir tatqa/baseline
!python features/extract.py --setting greedy --answers-dir tatqa/gen --labels-dir tatqa/labels --out-dir {SHARED}/tatqa_features
```

Push `data/tatqa_subset.jsonl` and `tatqa/`, and record the `build_tatqa.py` counts in `LOG.md`. 🔁

**Yvette:** after pulling Eduardo's subset, run the `temp12` chain:

```python
!git pull origin main
!python gen/generate.py --setting temp12 --subset data/tatqa_subset.jsonl --out-dir tatqa/gen
!python labels/label.py --setting temp12 --subset data/tatqa_subset.jsonl --answers-dir tatqa/gen --out-dir tatqa/labels
!python baseline/confidence.py --setting temp12 --answers-dir tatqa/gen --labels-dir tatqa/labels --out-dir tatqa/baseline
!python features/extract.py --setting temp12 --answers-dir tatqa/gen --labels-dir tatqa/labels --out-dir {SHARED}/tatqa_features
```

Once Eduardo's TAT-QA files are merged, run the transfer:

```python
!python detector/transfer.py --features-dir {SHARED}/features --tatqa-features-dir {SHARED}/tatqa_features
```

Push `tatqa/`, `detector/transfer_results.csv` and `detector/tatqa_scores.csv`.

**Both:** hand-check `tatqa/labels/handcheck.csv` the same way as the [Hand-check](#hand-check), and compare with:

```python
!python labels/agreement.py --sheet-a tatqa/labels/handcheck_A.csv --sheet-b tatqa/labels/handcheck_B.csv --key tatqa/labels/handcheck_key.csv --out tatqa/labels/handcheck_disagreements.csv
```
