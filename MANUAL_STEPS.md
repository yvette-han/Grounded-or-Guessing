# Manual steps (what we do by hand)

Claude writes the code. These steps need a GPU, our judgment, or our own words, so we do them ourselves. Tick items off in `ANALYSIS_TODO.md` as they're done.

_Last updated: Nov 5, split between Person A (Eduardo) and Person B (Yvette) so both can work at the same time; Yvette's plan for this week added. Every step with code is here; the poster and report (R2, R3) are ours._

In a Colab cell, shell commands start with `!` and folder changes use `%cd`. Every block below is one Colab cell.

**Contents:** [Who does what](#who-does-what) · [How we share files](#how-we-share-files) · [Person A checklist](#person-a-checklist) · [Person B checklist](#person-b-checklist) ([this week](#this-week-thu-nov-5--sun-nov-8)) · [Reference: step details](#reference-step-details)

---

## Who does what

Two people, two Colab GPUs, working at the same time. The GPU work is split **by decoding setting**: each person runs the whole chain (generate → label → confidence → features) for their own settings.

| | Person A: Eduardo (Stage 1 owner) | Person B: Yvette (Stage 2 owner) |
|---|---|---|
| Decoding settings | `greedy`, `beam5` (beam search is about 2x slower) | `temp07`, `temp12`, `topp09` |
| Runs and writes up | A1, A5 (with B), A6, Stage 1 error analysis | B0, B1 check, B3, B4 |
| B5 | the 10 false positives and where `wrong_calc` falls | the 10 false negatives |
| Reviews in `ANALYSIS_TODO.md` | items starting A1–A7, and the A files | items starting B0–B5, and the B files |
| Poster (README R2) | dataset + labeling, decoding chart | attention diagram, AUROC table, head/layer heatmap |

Running a script doesn't change who owns it: each of us still explains our own folders (README §2), and before Dec 9 each must be able to explain the other's half.

### Timeline

🔁 = a hand-off: the other person needs this pushed and merged before their next step.

| When | Person A | Person B |
|---|---|---|
| Nov 5–8 | A-0 setup · **A-1 build the subset first** 🔁 · A-2 reviews | B-0 setup + shared folder · B-1 B0 plot (after A-1) · B-2 reviews |
| Nov 9–12 | **A-3 first 50 greedy answers + labels + hand-check sheet, by Nov 12** 🔁 · then start generating `greedy`, `beam5` | B-3 generate `temp07`, `temp12`, `topp09` (no need to wait) · B-4 B1 check on A's 50 (after A-3) |
| Nov 12–14 | A-4 hand-label `label_A` 🔁 · run the agreement · decide the rules together | B-5 hand-label `label_B` 🔁 · decide the rules together |
| Nov 14–20 | A-5 label → confidence → features for `greedy`, `beam5` 🔁 | B-6 label → confidence → features for `temp07`, `temp12`, `topp09` 🔁 |
| Nov 16–22 | A-6 Stage 1 results + reading (needs B's labels) | B-7 detectors (B3) (needs A's confidence + features) |
| Nov 23–29 | A-7 Stage 1 error analysis | B-8 Stage 2 analysis (B4) + readings · makes the B5 examples 🔁 |
| Nov 30–Dec 6 | A-8 B5 false positives + `wrong_calc` · poster/report sections · optional R1 part | B-9 B5 false negatives · poster/report sections · optional R1 part |
| Dec 7–8 | Rehearse: explain B's half | Rehearse: explain A's half |

**Why labeling waits until Nov 14:** if the hand-check (A5) changes the rules in README §6.3, every label, confidence score and feature file has to be redone. Generating answers doesn't depend on the rules, so both of us generate first and label after the decision. The only early labels are A's first 50 greedy answers, which the hand-check needs.

---

## How we share files

**Each person has their own clone and their own branch.** Never run two Colab sessions in the same clone folder.

- Yvette (Person B) works on the branch `yvette`. Eduardo (Person A) makes their own branch, e.g. `eduardo`: on GitHub, open the branch menu, type the name and choose "Create branch … from main". Yvette must first add Eduardo under the repo's **Settings → Collaborators**.
- **Small files go through git:** subset, answers, labels, confidence scores, results, figures, notebooks. After each step: push to your branch, then on GitHub open a **pull request into `main`** and merge it (or ask Claude to merge). Before each session: pull `main` (step 0). We write different files (each setting has its own file names), so merges don't conflict.
- **`LOG.md`:** add your run notes under your own heading at the bottom (`## Run notes: Person A` or `## Run notes: Person B`), never in the other's, so merges stay clean. Decisions we make together go above those headings.
- **Big files go in a shared Drive folder:** feature files (`features/*.npz`, ~5 MB per 1,000 numbers) are too big for git. Person B creates a Drive folder `GoG-shared` and shares it with Person A as Editor. Person A opens "Shared with me", right-clicks it → **Organize → Add shortcut → My Drive**. Then both see it at `/content/drive/MyDrive/GoG-shared`, and all feature runs write into `GoG-shared/features/`.

Put this cell after the setup cell in every session, so the commands below can use `{SHARED}`:

```python
SHARED = "/content/drive/MyDrive/GoG-shared"
```

---

## Person A checklist

**Eduardo.** Settings: **`greedy`, `beam5`**. Details of each check are in the reference section ([step N](#reference-step-details)).

**A-0. Setup (once, then the session cell every time).** Do [step 0](#0-colab-setup-each-new-session), with your token. Set `SHARED` as above (after B shares the folder).

**A-1. Build the subset (do this first; B is waiting for it).** [Step 1](#1-a1-build-the-subset).

```python
!python data/build_subset.py
```

- Check `gold programs whose last step differs from FinQA's answer: 0`.
- In `LOG.md`, replace the line `A1 stats (fill in ...)` with the printed counts.
- Push `data/finqa_subset.jsonl` and `LOG.md`, merge into `main`, and tell B. 🔁

**A-2. Reviews (any time, in between runs).** In `ANALYSIS_TODO.md`: every section 1 item starting with A1–A7, every section 2 file marked (A) or (both), and add a few test strings of your own to `tests/test_numbers.py`.

**A-3. The first 50 greedy answers for B (due Nov 12).** [Steps 3–4](#3-a3-generate-answers).

```python
!python gen/generate.py --setting greedy --limit 50
!python labels/label.py --setting greedy --handcheck 50
```

- **Don't open** `labels/handcheck_key.csv` or `labels/numbers_greedy.jsonl` until the hand-check is done (they show the auto labels). Don't commit the key yet.
- Push `gen/answers_greedy.jsonl`, `labels/numbers_greedy.jsonl`, `labels/handcheck.csv`; merge; tell B. 🔁
- Then start the full generation. It runs for hours and resumes after a disconnect, so keep going with A-4 meanwhile:

```python
!python gen/generate.py --setting greedy
!python gen/generate.py --setting beam5
```

  (`beam5` out of GPU memory → add `--batch-size 4`.) Push each `gen/answers_*.jsonl` when it's complete.

**A-4. Hand-check (Nov 12–14).** [Step 5](#5-a5-hand-check-the-labels).

- Copy `labels/handcheck.csv` into Google Sheets, fill **only `label_A`**, download as CSV, save as `labels/handcheck_A.csv`, push. 🔁
- When B's `labels/handcheck_B.csv` is merged, pull and run:

```python
!python labels/agreement.py --sheet-a labels/handcheck_A.csv --sheet-b labels/handcheck_B.csv
```

- Put the rates in `LOG.md`, push `labels/handcheck_key.csv` and `labels/handcheck_disagreements.csv`, and go through the disagreements with B. Decide together whether README §6.3 changes. If it does, tell Claude and wait for the fix before A-5.

**A-5. Your settings, after the rules are settled.** [Steps 4, 7, 8](#4-a4-label-the-numbers). Use the row choice B picked in B-4 (`--row predicting` only if B chose it).

```python
!python labels/label.py --setting greedy
!python labels/label.py --setting beam5
!python baseline/confidence.py --setting greedy beam5
!python features/extract.py --setting greedy beam5 --out-dir {SHARED}/features
```

- Checks: the label counts (copy into `LOG.md`), `NaNs: 0` from confidence, row sums ~1 and values within 0–1 from features.
- Push `labels/numbers_greedy.jsonl`, `labels/numbers_beam5.jsonl`, `baseline/confidence_greedy.npz`, `baseline/confidence_beam5.npz`, `LOG.md`; merge; tell B. 🔁 The feature files are already in the shared folder.

**A-6. Stage 1 results (after B's labels are merged).** [Step 6](#6-a6-stage-1-results). Pull `main`, then run `analysis/stage1.ipynb` (or `!python analysis/stage1.py`). Write its **"Our reading"** cell. Push the notebook and `analysis/figures/stage1_*`.

**A-7. Stage 1 error analysis (Nov 23–29).** From the Stage 1 table and the answers, in our own words, in `analysis/stage1.ipynb` below "Our reading".

**A-8. B5, your part (after B pushes the B5 examples).** [Step 11](#11-b5-error-analysis). Pull `main`, read `analysis/figures/b5_examples.md` and `b5_scores_by_label.png`. Write the **false positives** (each of the 10, then the pattern) and **where `wrong_calc` falls** in a new file `analysis/notes/b5_A.md`. Push it. (Separate files so we don't edit the same notebook at once; at the end one of us copies both into `analysis/errors.ipynb`.)

**A-9. Poster and report.** Your sections: dataset + labeling, the decoding chart (`stage1_bars.png`), the Stage 1 results. Then edit the full draft together.

**A-10. Rehearse (Dec 7–8).** Be able to explain B's half: `features/`, `detector/`, `analysis/stage2.py`, and the B readings.

**Optional R1, your part.** [Step 12](#12-r1-optional-stretch-finqa--tat-qa-transfer). Build the TAT-QA subset and push it (🔁 B waits for it), then the `greedy` chain:

```python
!python data/build_tatqa.py
!python gen/generate.py --setting greedy --subset data/tatqa_subset.jsonl --out-dir tatqa/gen
!python labels/label.py --setting greedy --subset data/tatqa_subset.jsonl --answers-dir tatqa/gen --out-dir tatqa/labels --handcheck 30
!python baseline/confidence.py --setting greedy --answers-dir tatqa/gen --labels-dir tatqa/labels --out-dir tatqa/baseline
!python features/extract.py --setting greedy --answers-dir tatqa/gen --labels-dir tatqa/labels --out-dir {SHARED}/tatqa_features
```

Push `data/tatqa_subset.jsonl` and the `tatqa/` files; hand-label the 30 TAT-QA numbers (`label_A`) as in A-4.

---

## Person B checklist

**Yvette.** Settings: **`temp07`, `temp12`, `topp09`**. Details of each check are in the reference section ([step N](#reference-step-details)).

### This week (Thu Nov 5 – Sun Nov 8)

Goal: the B0 plot and the A1 stats for the meeting with professor (README §9), and everything set up so B-3 can start on Mon Nov 9.

**Thu Nov 5 – Fri Nov 6: set up (B-0)**

1. GitHub → the repo → **Settings → Collaborators** → add Eduardo, so Eduardo can make their branch and push.
2. Make your token: GitHub → Settings → Developer settings → **Fine-grained tokens**, access to this repo only, "Contents: read and write". Save it in Colab's 🔑 Secrets panel as `GITHUB_TOKEN` with "Notebook access" on.
3. Google Drive: create the folder `GoG-shared` and share it with Eduardo as Editor.
4. Colab with a T4 GPU: do [step 0](#0-colab-setup-each-new-session) (clone once, then the session cell; your branch is `yvette`), then set `SHARED`. All tests should pass, including the one that downloads the Qwen tokenizer; that also shows Colab can reach Hugging Face.

**Fri Nov 6: get the subset**

- If Eduardo has already merged A-1: `!git pull origin main`.
- If not yet, run it yourself. It's the same file whoever runs it (fixed seed), and the meeting needs its stats:

```python
!python data/build_subset.py
```

  Check `gold programs whose last step differs from FinQA's answer: 0`. Fill the `A1 stats (fill in ...)` line in `LOG.md` with the printed counts, push `data/finqa_subset.jsonl` and `LOG.md`, merge into `main`, and tell Eduardo to skip A-1.

**Sat Nov 7: the B0 plot (B-1)**

```python
!python analysis/b0_attention_plot.py
```

- Check `attention rows sum to` about 1.
- It prints the answer's numbers, each marked "in report" or "not in report". For a comparison, also plot one marked "not in report" (`--number <its index>`) and try `--row predicting` once.
- Each run overwrites `analysis/figures/b0_attention.png` and `b0_lookback.png`, so rename the ones to keep before the next run, e.g.:

```python
!mv analysis/figures/b0_attention.png analysis/figures/b0_attention_in_report.png
!mv analysis/figures/b0_lookback.png analysis/figures/b0_lookback_in_report.png
```

- Push the PNGs you keep.

**Sun Nov 8, or before the meeting: prepare, in your own words**

- A few sentences on what the plot shows: which report tokens the heads look at when the model writes the number. These have to be yours (README §3); keep them under your run notes in `LOG.md` or in your meeting notes.
- Be able to explain `analysis/b0_attention_plot.py` and `features/attention.py` line by line (`ANALYSIS_TODO.md` section 2).
- Bring the B0 plot(s) and the A1 stats from `LOG.md`.

**Reviews to do before Nov 12** (`ANALYSIS_TODO.md` section 1, your items):

- The row choice (`number` or `predicting`): you decide it in B-4 on Nov 12; the B0 runs above are a first look.
- B1: how "generated tokens" is defined, and the per-number averaging.
- B2: the frequency units. Check how Qi et al. (2026) normalize frequency before features are computed at scale (from Nov 14).

**If you have GPU time left:** start B-3 early with `!python gen/generate.py --setting temp07`. Generation doesn't depend on the label rules, and stopping is fine because it resumes.

Next week: B-3 (generate your settings) from Mon Nov 9, then B-4 and B-5 on Thu Nov 12, once Eduardo has pushed the first 50 greedy answers.

**B-0. Setup (once, then the session cell every time).** Do [step 0](#0-colab-setup-each-new-session), with your own token and clone. Create the shared Drive folder `GoG-shared`, share it with A as Editor (see [How we share files](#how-we-share-files)), and set `SHARED`.

**B-1. B0 plot for the meeting (after A-1 is merged).** [Step 2](#2-b0-attention-plot-for-the-meeting). Pull `main`, then:

```python
!python analysis/b0_attention_plot.py
```

- Check `attention rows sum to` ~1. Push the two PNGs and prepare to explain them to professor.

**B-2. Reviews (any time, in between runs).** In `ANALYSIS_TODO.md`: every section 1 item starting with B0–B5, and every section 2 file marked (B) or (both).

**B-3. Generate your settings (start as soon as the subset is merged; you don't need A's answers).** [Step 3](#3-a3-generate-answers).

```python
!python gen/generate.py --setting temp07
!python gen/generate.py --setting temp12
!python gen/generate.py --setting topp09
```

- One per cell; each takes a while and resumes after a disconnect. Push each `gen/answers_*.jsonl` when it's complete. Don't label yet (see "Why labeling waits").

**B-4. B1 check on A's first 50 (after A-3 is merged, Nov 12).** [Step 8](#8-b1b2-attention-features). This writes to a local scratch folder, not the shared one:

```python
!git pull origin main
!python features/extract.py --setting greedy --limit 20 --out-dir features/check
!python features/extract.py --setting greedy --limit 20 --row predicting --out-dir features/check_predicting
```

- Check row sums ~1, lookback and HF values within 0–1. Note the result in `LOG.md` (README B1 done).
- Decide the row choice (`number` or `predicting`, `ANALYSIS_TODO.md` B0/B1 item) and tell A, so all feature runs use the same one.
- Don't open `labels/numbers_greedy.jsonl` before the hand-check is done.

**B-5. Hand-check (Nov 12–14).** [Step 5](#5-a5-hand-check-the-labels). Copy `labels/handcheck.csv` into Google Sheets, fill **only `label_B`**, download as CSV, save as `labels/handcheck_B.csv`, push. 🔁 Then go through the disagreements with A.

**B-6. Your settings, after the rules are settled.** [Steps 4, 7, 8](#4-a4-label-the-numbers). Add `--row predicting` to the last line only if that's what you chose in B-4.

```python
!python labels/label.py --setting temp07
!python labels/label.py --setting temp12
!python labels/label.py --setting topp09
!python baseline/confidence.py --setting temp07 temp12 topp09
!python features/extract.py --setting temp07 temp12 topp09 --out-dir {SHARED}/features
```

- Checks: label counts (copy into `LOG.md`), `NaNs: 0`, feature row sums ~1.
- Push `labels/numbers_temp07.jsonl`, `numbers_temp12.jsonl`, `numbers_topp09.jsonl`, `baseline/confidence_temp07.npz`, `confidence_temp12.npz`, `confidence_topp09.npz`, `LOG.md`; merge. 🔁

**B-7. Detectors (B3), after A-5 is merged.** [Step 9](#9-b3-detectors). Pull `main`; check `{SHARED}/features` has all five settings; then:

```python
!python detector/train_eval.py --features-dir {SHARED}/features
```

- Check the settings line, the dropped counts, and that test has fabricated numbers. Note the chosen C and cutoffs in `LOG.md`. Push `detector/results.csv`, `detector/test_scores.csv`.

**B-8. Stage 2 analysis (B4) and the B5 examples (Nov 23–29).** [Steps 10–11](#10-b4-stage-2-analysis).

- In `analysis/stage2.ipynb`, set `FEATURES_DIR = Path(SHARED) / "features"` in the first code cell (or run `!python analysis/stage2.py --features-dir {SHARED}/features`), run all, and write the three **"Our reading"** cells (RQ2, RQ3, bridge).
- Then make the B5 examples for both of you:

```python
!python analysis/errors.py
```

- Push `analysis/stage2.ipynb`, `analysis/figures/stage2_*`, `analysis/figures/b5_*`; tell A. 🔁

**B-9. B5, your part.** Write the **false negatives** (each of the 10, then the pattern) in a new file `analysis/notes/b5_B.md` and push it. At the end, one of us copies `b5_A.md` and `b5_B.md` into the reading cells of `analysis/errors.ipynb`.

**B-10. Poster and report.** Your sections: the attention diagram (`b0_attention.png`), the AUROC table (`detector/results.csv`), the head/layer heatmap (`stage2_heads.png`), the Stage 2 results. Then edit the full draft together.

**B-11. Rehearse (Dec 7–8).** Be able to explain A's half: `data/`, `gen/`, `labels/`, `baseline/`, `analysis/stage1.py`, and the A readings.

**Optional R1, your part (after A pushes `data/tatqa_subset.jsonl`).** The `temp12` chain, then the transfer once A's TAT-QA files are merged:

```python
!git pull origin main
!python gen/generate.py --setting temp12 --subset data/tatqa_subset.jsonl --out-dir tatqa/gen
!python labels/label.py --setting temp12 --subset data/tatqa_subset.jsonl --answers-dir tatqa/gen --out-dir tatqa/labels
!python baseline/confidence.py --setting temp12 --answers-dir tatqa/gen --labels-dir tatqa/labels --out-dir tatqa/baseline
!python features/extract.py --setting temp12 --answers-dir tatqa/gen --labels-dir tatqa/labels --out-dir {SHARED}/tatqa_features
```

```python
!python detector/transfer.py --features-dir {SHARED}/features --tatqa-features-dir {SHARED}/tatqa_features
```

Push the `tatqa/` files and `detector/transfer_results.csv`, `detector/tatqa_scores.csv`; hand-label the 30 TAT-QA numbers (`label_B`).

---

# Reference: step details

What each step checks and commits. The checklists above say who runs which settings.

## 0. Colab setup (each new session)

1. **Runtime → Change runtime type → T4 GPU.** (Free Colab gives a few GPU hours a day; long runs resume, so spread them over days.)
2. Work inside **your own** Google Drive, so files survive a disconnect and long runs can resume:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive
```

3. **First time only**, clone the repo with a GitHub token. Make the token on GitHub → Settings → Developer settings: a **fine-grained token** with "Contents: read and write" for this repo if you own it (Yvette), or a **classic token** with the `repo` scope if you're a collaborator (Eduardo). Save it in Colab's 🔑 Secrets panel (left sidebar) as `GITHUB_TOKEN`, with "Notebook access" switched on:

```python
from google.colab import userdata
token = userdata.get("GITHUB_TOKEN")
!git clone -b main https://{token}@github.com/yvette-han/Grounded-or-Guessing
```

4. **Every session**: go into the folder, switch to your branch, bring in `main`, install, test:

```python
%cd /content/drive/MyDrive/Grounded-or-Guessing
!git config user.name "Your Name" && git config user.email "you@example.com"
!git fetch origin && git checkout <your-branch> && git pull origin main
!pip install -q -r requirements.txt
!python -m pytest -q
```

All tests should pass, including the one that uses the real Qwen tokenizer (Claude's container can't download it, so it was always skipped there).

### Saving results to GitHub

Generated files only exist in your Drive until you push them:

```python
!git add <the files the step lists> LOG.md
!git commit -m "A-5: greedy and beam5 labeled" && git push origin HEAD:<your-branch>
```

Then on GitHub open a pull request from your branch into `main` and merge it (or ask Claude to merge). Without git: download the files from Colab's Files panel and upload them on GitHub to your branch (**Add file → Upload files**).

---

## 1. A1: build the subset

```python
!python data/build_subset.py
```

- Check the output says `gold programs whose last step differs from FinQA's answer: 0`.
- In `LOG.md`, replace the line starting `A1 stats (fill in ...)` with the printed numbers: questions left after the length filter, mean context tokens, and the three split sizes (questions and reports).
- **Commit:** `data/finqa_subset.jsonl`, `LOG.md`.

## 2. B0: attention plot for the meeting

```python
!python analysis/b0_attention_plot.py
```

- It prints the generated answer and a numbered list of its numbers. To plot a different one: `--number 2`. Another question: `--id "<an id from data/finqa_subset.jsonl>"`. The row that *predicts* the digit instead of the digit's own row: `--row predicting`.
- Check the printed `attention rows sum to ...` is about 1.
- Look at `analysis/figures/b0_attention.png` and `b0_lookback.png` and prepare to explain them in our own words.
- **Commit:** the two PNGs.

## 3. A3: generate answers

`!python gen/generate.py --setting <setting>` for each of your settings; `--limit 50` for the first greedy batch.

- If Colab disconnects, rerun the same command: it continues where it stopped.
- If `beam5` runs out of GPU memory, add `--batch-size 4`.
- Each run ends with a summary. Note anything odd in your `LOG.md` section, especially a non-zero `answers whose re-tokenization changes the prompt tokens`.
- **Commit:** your `gen/answers_*.jsonl`.

## 4. A4: label the numbers

`!python labels/label.py --setting <setting>` for each of your settings, once all its answers exist. Person A adds `--handcheck 50` the first time (greedy).

- Copy each setting's label counts into `LOG.md` (README A4 is done when all five are there).
- **Until both of us have finished the hand-check, don't open `labels/handcheck_key.csv` or `labels/numbers_greedy.jsonl`**: they show the auto labels.
- **Commit:** your `labels/numbers_*.jsonl` (and A: `labels/handcheck.csv`; the key only after the hand-check).

## 5. A5: hand-check the labels

1. Each makes their own copy of `labels/handcheck.csv` (Google Sheets: File → Import → Upload; or open it in Excel).
2. Fill **only your own column**: Person A `label_A`, Person B `label_B`. Use one of `copied`, `computed`, `question`, `wrong_calc`, `fabricated`, following README §6.3. Each row has the answer with the number marked ⟦like this⟧, the question, the gold answer, program and steps, and the report.
3. Don't look at the key, the labels file, or the other person's sheet.
4. Download your copy as CSV and save it as `labels/handcheck_A.csv` or `labels/handcheck_B.csv` (keep the `row` and `id` columns; sorting is fine).
5. Compare (Person A, once both sheets are merged):

```python
!python labels/agreement.py --sheet-a labels/handcheck_A.csv --sheet-b labels/handcheck_B.csv
```

6. Put the agreement rates in `LOG.md`. Go through `labels/handcheck_disagreements.csv` together.
7. Decide which agreement the README's ~90% applies to, and whether §6.3 needs to change. If it does, tell Claude what to change; then each of us relabels our settings (step 4) before going on.

- **Commit:** your sheet; A also `labels/handcheck_key.csv`, `labels/handcheck_disagreements.csv`, `LOG.md`.

## 6. A6: Stage 1 results

After every setting is generated and labeled (both people's):

```python
!python analysis/stage1.py
```

or open `analysis/stage1.ipynb` (in Colab: File → Open notebook → Google Drive, then the repo folder) and run all cells.

- Write the **"Our reading"** cell of the notebook in our own words: what decoding does to accuracy and fabrication, and why.
- **Commit:** `analysis/figures/stage1_table.csv`, `stage1_table.md`, `stage1_bars.png`, `analysis/stage1.ipynb`.

---

## 7. A7: confidence baseline

`!python baseline/confidence.py --setting <your settings>` after step 4 has labeled them.

- One teacher-forced pass per answer.
- Check each setting's line: the number count matches its labels file, `NaNs: 0`, and no skipped answers.
- **Commit:** your `baseline/confidence_*.npz`.

---

## 8. B1–B2: attention features

The README B1 check (Person B, on the first greedy answers) writes to a local scratch folder:

```python
!python features/extract.py --setting greedy --limit 20 --out-dir features/check
```

- Check each setting's lines: `attention row sums` about 1, and `lookback` between 0 and 1. Note the result in `LOG.md` (README B1 is done when this looks sane).
- `hf_ctx` and `hf_gen` are printed too: values should be between 0 and 1. NaN values in `hf_gen` should only come from numbers at the very start of an answer.

The full run, by each person for their settings, into the shared folder (uses more GPU memory than A3 because it keeps all attention maps for one answer at a time):

```python
!python features/extract.py --setting <your settings> --out-dir {SHARED}/features
```

- Everyone uses the same `--row` (B decides in B-4). To keep both versions: `--row predicting --out-dir {SHARED}/features_predicting`.
- **Don't commit** feature files (git ignores them): they live in `GoG-shared`. Commit only the `LOG.md` notes.

---

## 9. B3: detectors

After steps 7 and 8 for all five settings (both people's). No GPU needed:

```python
!python detector/train_eval.py --features-dir {SHARED}/features
```

- Check the first line: which settings were used and how many numbers were dropped for a NaN feature (by label).
- Check the train/val/test counts and that the test split has fabricated numbers.
- If you also made `features_predicting`: `!python detector/train_eval.py --features-dir {SHARED}/features_predicting --out-dir detector/predicting`.
- Note in `LOG.md` the chosen C and cutoff per detector (they're in `results.csv`).
- **Commit:** `detector/results.csv`, `detector/test_scores.csv` (and `detector/predicting/` if made).

---

## 10. B4: Stage 2 analysis

After step 9. No GPU needed:

```python
!python analysis/stage2.py --features-dir {SHARED}/features
```

or open `analysis/stage2.ipynb`, set `FEATURES_DIR = Path(SHARED) / "features"` in the first code cell, and run all cells. It refits detectors per layer and per top-k heads, so it takes a few minutes.

- Check the RQ2 table has a row per detector and the bridge has every setting.
- Write the three **"Our reading"** cells (RQ2, RQ3, bridge) in our own words.
- **Commit:** `analysis/figures/stage2_*` and `analysis/stage2.ipynb`.

---

## 11. B5: error analysis

After step 9. No GPU needed. Person B runs it once so we both read the same examples:

```python
!python analysis/errors.py
```

- By default it uses the detector with the best val AUROC and a random 10 + 10 examples. To look at the most confident errors too: `--pick extreme` (or `PICK = "extreme"` in `analysis/errors.ipynb`).
- Read each example in `analysis/figures/b5_examples.md`: the marked number, the labels of the other numbers in that answer, and the report lines with that value.
- Person A writes the false positives and `wrong_calc` in `analysis/notes/b5_A.md`; Person B writes the false negatives in `analysis/notes/b5_B.md`. At the end, one of us copies both into the three **"Our reading"** cells of `analysis/errors.ipynb`.
- **Commit:** `analysis/figures/b5_*`, the notes files, and `analysis/errors.ipynb`.

---

## 12. R1 (optional stretch): FinQA → TAT-QA transfer

Only if time allows, after steps 7–9. TAT-QA outputs go into `tatqa/` (and `{SHARED}/tatqa_features`) so nothing of FinQA is overwritten. Use the same `--row` for features as for FinQA. The commands are in the checklists: A builds the subset and runs `greedy`, B runs `temp12` and the transfer (greedy alone may have too few fabricated numbers).

- `build_tatqa.py` prints how many derivations don't reproduce their answer (expect about 0.3%). Record its counts in `LOG.md`.
- Both hand-check `tatqa/labels/handcheck.csv` as in step 5, then compare with `labels/agreement.py --sheet-a ... --sheet-b ... --key tatqa/labels/handcheck_key.csv --out tatqa/labels/handcheck_disagreements.csv`, because TAT-QA's answers and derivations differ from FinQA's.
- `transfer.py` refuses features made with a different `--row` or different HF cutoffs than FinQA's.
- **Commit:** `data/tatqa_subset.jsonl`, `tatqa/gen/`, `tatqa/labels/`, `tatqa/baseline/`, `detector/transfer_results.csv`, `detector/tatqa_scores.csv` (TAT-QA features stay in `GoG-shared`).

---

## Any time: reviews and explanations

- **`ANALYSIS_TODO.md` section 1:** read each choice Claude made in code (items starting A… are Person A's, B… Person B's, R1 both). Tick it if we keep it, or tell Claude what to change.
- **Section 2:** each of us must be able to explain every AI-written file in our folders line by line, and check the expected values in its tests by hand (README §3). For A2, Person A also adds a few test strings of their own to `tests/test_numbers.py`, in the styles the model's answers actually use.
