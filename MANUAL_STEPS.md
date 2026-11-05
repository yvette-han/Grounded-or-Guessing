# Manual steps (what we do by hand)

Claude writes the code. These steps need a GPU, our judgment, or our own words, so we do them ourselves, in this order. Tick items off in `ANALYSIS_TODO.md` as they're done.

_Last updated: Nov 5, after the R1 code. Every step with code is now here; R2–R3 (poster, report) are ours._

In a Colab cell, shell commands start with `!` and folder changes use `%cd`. Every block below is one Colab cell.

---

## 0. Colab setup (each new session)

1. **Runtime → Change runtime type → T4 GPU.**
2. Work inside Google Drive, so files survive a disconnect and long runs can resume:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive
```

3. **First time only**, clone the repo (if it's private, see "Private repo" below):

```python
!git clone -b main https://github.com/yvette-han/Grounded-or-Guessing
```

4. **Every session**, go into the folder, update and test:

```python
%cd /content/drive/MyDrive/Grounded-or-Guessing
!git pull
!pip install -q -r requirements.txt
!python -m pytest -q
```

All tests should pass, including the one that uses the real Qwen tokenizer (Claude's container can't download it, so it was always skipped there).

**Private repo:** make a GitHub fine-grained token with access to this repo (GitHub → Settings → Developer settings → Fine-grained tokens, "Contents: read and write"), save it in Colab's 🔑 Secrets panel (left sidebar) as `GITHUB_TOKEN` with "Notebook access" switched on, then clone with:

```python
from google.colab import userdata
token = userdata.get("GITHUB_TOKEN")
!git clone -b main https://{token}@github.com/yvette-han/Grounded-or-Guessing
```

## Saving results to GitHub

Generated files only exist in Drive until we commit them. After each step below, either:

- **From Colab** (needs the token clone above, once per session set who you are):

```python
!git config user.name "Your Name" && git config user.email "you@example.com"
!git add <the files the step lists> LOG.md
!git commit -m "A1: subset built" && git push origin HEAD:yvette
```

  Then tell Claude, which merges `yvette` into its branch before the next update of `main`.

- **Or by hand:** download the files from Colab's Files panel, then on GitHub open the `yvette` branch → **Add file → Upload files**.

---

## 1. A1: build the subset (Nov 2–8)

```python
!python data/build_subset.py
```

- Check the output says `gold programs whose last step differs from FinQA's answer: 0`.
- In `LOG.md`, replace the line starting `A1 stats (fill in ...)` with the printed numbers: questions left after the length filter, mean context tokens, and the three split sizes (questions and reports).
- **Commit:** `data/finqa_subset.jsonl`, `LOG.md`.

## 2. B0: attention plot for the meeting (before the meeting with the professor)

```python
!python analysis/b0_attention_plot.py
```

- It prints the generated answer and a numbered list of its numbers. To plot a different one: `--number 2`. Another question: `--id "<an id from data/finqa_subset.jsonl>"`. The row that *predicts* the digit instead of the digit's own row: `--row predicting`.
- Check the printed `attention rows sum to ...` is about 1.
- Look at `analysis/figures/b0_attention.png` and `b0_lookback.png` and prepare to explain them in our own words.
- **Commit:** the two PNGs.

## 3. A3: generate answers (Nov 9–15)

First the 50 greedy answers for Stage 2 (**due Nov 12**):

```python
!python gen/generate.py --setting greedy --limit 50
```

Then all 1,000 for every setting (run one per cell; each takes a while):

```python
!python gen/generate.py --setting greedy
!python gen/generate.py --setting beam5
!python gen/generate.py --setting temp07
!python gen/generate.py --setting temp12
!python gen/generate.py --setting topp09
```

- If Colab disconnects, rerun the same command: it continues where it stopped.
- If `beam5` runs out of GPU memory, add `--batch-size 4`.
- Each run ends with a summary. Note anything odd in `LOG.md`, especially a non-zero `answers whose re-tokenization changes the prompt tokens`.
- **Commit:** `gen/answers_*.jsonl`.

## 4. A4: label the numbers

The first time, also make the hand-check sheet (it can come from the first 50 greedy answers):

```python
!python labels/label.py --setting greedy --handcheck 50
```

Then each setting once all its answers exist (this includes greedy again after its full 1,000):

```python
!python labels/label.py --setting greedy
!python labels/label.py --setting beam5
!python labels/label.py --setting temp07
!python labels/label.py --setting temp12
!python labels/label.py --setting topp09
```

- Copy each setting's label counts into `LOG.md` (README A4 is done when they're there).
- **Until both of us have finished the hand-check, don't open `labels/handcheck_key.csv` or `labels/numbers_greedy.jsonl`**: they show the auto labels.
- **Commit:** `labels/numbers_*.jsonl`, `labels/handcheck.csv`. Commit `labels/handcheck_key.csv` only after the hand-check.

## 5. A5: hand-check the labels (Nov 12–14, both of us)

1. Each makes their own copy of `labels/handcheck.csv` (Google Sheets: File → Import → Upload; or open it in Excel).
2. Fill **only your own column**: Person A `label_A`, Person B `label_B`. Use one of `copied`, `computed`, `question`, `wrong_calc`, `fabricated`, following README §6.3. Each row has the answer with the number marked ⟦like this⟧, the question, the gold answer, program and steps, and the report.
3. Don't look at the key, the labels file, or the other person's sheet.
4. Download your copy as CSV and put both in `labels/`, e.g. `handcheck_A.csv` and `handcheck_B.csv` (keep the `row` and `id` columns; sorting is fine).
5. Compare:

```python
!python labels/agreement.py --sheet-a labels/handcheck_A.csv --sheet-b labels/handcheck_B.csv
```

6. Put the agreement rates in `LOG.md`. Go through `labels/handcheck_disagreements.csv` together.
7. Decide which agreement the README's ~90% applies to, and whether §6.3 needs to change. If it does, tell Claude what to change, then rerun step 4 for every setting (and step 6).

Person A can do their half before Person B joins; run step 5 once both sheets exist.

- **Commit:** both sheets, `labels/handcheck_key.csv`, `labels/handcheck_disagreements.csv`, `LOG.md`.

## 6. A6: Stage 1 results (Nov 16–22)

After every setting is generated and labeled:

```python
!python analysis/stage1.py
```

or open `analysis/stage1.ipynb` (in Colab: File → Open notebook → Google Drive, then the repo folder) and run all cells.

- Write the **"Our reading"** cell of the notebook in our own words: what decoding does to accuracy and fabrication, and why.
- **Commit:** `analysis/figures/stage1_table.csv`, `stage1_table.md`, `stage1_bars.png`, `analysis/stage1.ipynb`.

---

## 7. A7: confidence baseline (Nov 16–22)

After step 4 has labeled the settings:

```python
!python baseline/confidence.py
```

- It scores every setting that has a labels file (or name them: `--setting greedy temp12`). One teacher-forced pass per answer.
- Check each setting's line: the number count matches its labels file, `NaNs: 0`, and no skipped answers.
- **Commit:** `baseline/confidence_*.npz`.

---

## 8. B1–B2: attention features (Nov 9–22)

Needs the labels from step 4. First the README B1 check on 20 answers per setting:

```python
!python features/extract.py --limit 20
```

- Check each setting's lines: `attention row sums` about 1, and `lookback` between 0 and 1. Note the result in `LOG.md` (README B1 is done when this looks sane).
- `hf_ctx` and `hf_gen` are printed too: values should be between 0 and 1. NaN values in `hf_gen` should only come from numbers at the very start of an answer.
- Then the full run (every setting that has labels; uses more GPU memory than A3 because it keeps all attention maps for one answer at a time):

```python
!python features/extract.py
```

- To try the other row choice: `--row predicting --out-dir features/predicting` (keeps both versions).
- **Don't commit** `features/*.npz` (git ignores them): at about 5 MB per 1,000 numbers they are too big for GitHub. They stay in the Drive folder; share that folder with your teammate so both of you can run B3. Commit only the `LOG.md` notes.

---

## 9. B3: detectors (Nov 16–22)

After steps 7 (A7) and 8 (B1–B2) for every setting. No GPU needed, but run it in the same Drive folder so it finds `features/`:

```python
!python detector/train_eval.py
```

- Check the first line: which settings were used and how many numbers were dropped for a NaN feature (by label).
- Check the train/val/test counts and that the test split has fabricated numbers.
- If you also made `features/predicting/` in step 8: `!python detector/train_eval.py --features-dir features/predicting --out-dir detector/predicting`.
- Note in `LOG.md` the chosen C and cutoff per detector (they're in `results.csv`).
- **Commit:** `detector/results.csv`, `detector/test_scores.csv` (and `detector/predicting/` if made).

---

## 10. B4: Stage 2 analysis (Nov 23–29)

After step 9 (B3). No GPU needed; run it in the Drive folder so it finds `features/`:

```python
!python analysis/stage2.py
```

or open `analysis/stage2.ipynb` and run all cells. It refits detectors per layer and per top-k heads, so it takes a few minutes.

- Check the RQ2 table has a row per detector and the bridge has every setting.
- Write the three **"Our reading"** cells (RQ2, RQ3, bridge) in our own words.
- **Commit:** `analysis/figures/stage2_*` and `analysis/stage2.ipynb`.

---

## 11. B5: error analysis (Nov 30–Dec 6, both of us)

After step 9 (B3). No GPU needed. Open `analysis/errors.ipynb` and run all cells, or:

```python
!python analysis/errors.py
```

- By default it uses the detector with the best val AUROC and a random 10 + 10 examples. To look at the most confident errors too, set `PICK = "extreme"` in the notebook (or `--pick extreme`).
- Read each example in `analysis/figures/b5_examples.md` together: the marked number, the labels of the other numbers in that answer, and the report lines with that value.
- Write the three **"Our reading"** cells (wrong_calc, false positives, false negatives) in our own words.
- **Commit:** `analysis/figures/b5_*` and `analysis/errors.ipynb`.

---

## 12. R1 (optional stretch): FinQA → TAT-QA transfer (Nov 30–Dec 6)

Only if time allows, after the FinQA runs of steps 7–9. TAT-QA outputs go into `tatqa/` so nothing of FinQA is overwritten. Use the same `--row` for features as for FinQA.

```python
!python data/build_tatqa.py
```

```python
!python gen/generate.py --setting greedy --subset data/tatqa_subset.jsonl --out-dir tatqa/gen
!python gen/generate.py --setting temp12 --subset data/tatqa_subset.jsonl --out-dir tatqa/gen
```

(greedy alone may have too few fabricated numbers; add more settings if time allows.)

```python
!python labels/label.py --setting greedy --subset data/tatqa_subset.jsonl --answers-dir tatqa/gen --out-dir tatqa/labels --handcheck 30
!python labels/label.py --setting temp12 --subset data/tatqa_subset.jsonl --answers-dir tatqa/gen --out-dir tatqa/labels
!python baseline/confidence.py --answers-dir tatqa/gen --labels-dir tatqa/labels --out-dir tatqa/baseline
!python features/extract.py --answers-dir tatqa/gen --labels-dir tatqa/labels --out-dir tatqa/features
!python detector/transfer.py
```

- `build_tatqa.py` prints how many derivations don't reproduce their answer (expect about 0.3%). Record its counts in `LOG.md`.
- Hand-check `tatqa/labels/handcheck.csv` the same way as step 5 (compare with `labels/agreement.py --sheet-a ... --sheet-b ... --key tatqa/labels/handcheck_key.csv --out tatqa/labels/handcheck_disagreements.csv`), because TAT-QA's answers and derivations differ from FinQA's.
- `transfer.py` refuses features made with a different `--row` or different HF cutoffs than FinQA's.
- **Commit:** `data/tatqa_subset.jsonl`, `tatqa/gen/`, `tatqa/labels/`, `tatqa/baseline/`, `detector/transfer_results.csv`, `detector/tatqa_scores.csv` (`tatqa/features/` stays in Drive).

---

## Any time: reviews and explanations

- **`ANALYSIS_TODO.md` section 1:** read each choice Claude made in code. Tick it if we keep it, or tell Claude what to change.
- **Section 2:** each of us must be able to explain every AI-written file line by line, and check the expected values in its tests by hand (README §3). For A2, also add a few test strings of our own to `tests/test_numbers.py`, in the styles the model's answers actually use.
