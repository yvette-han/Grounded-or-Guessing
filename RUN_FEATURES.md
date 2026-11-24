# Running the features (Eduardo)

_Last updated: Nov 24._

Yvette runs the features for `temp07`, `temp12` and `topp09` on her RTX 4060. This file is what Eduardo runs:

- **Part 1, now:** finish `greedy` and `beam5` and share the files.
- **Part 2, only if Yvette asks:** her 20-answer test ran out of memory or was too slow, so her three settings run on your server instead.

Every run uses the code on `main` from commit `b6b197e` on (float32, one layer at a time), the default `--dtype float32`, and the **same `--row` for all five settings**: `number`, unless we chose `predicting`.

---

## Part 1 · greedy and beam5 (now)

1. Let the current run finish. For each setting, check the printed lines:
   - `rows NaN/inf or off by >1%: 0`
   - lookback within 0–1
   - no `skipped ... answers` line
   - the number count equals the line count of `labels/numbers_{setting}.jsonl`
2. Copy `greedy.npz` and `beam5.npz` into the shared Drive folder `GoG-shared/features`, not git (about 50 MB each). From the server, e.g. `scp` them to your laptop, then upload them in the browser.
3. In `LOG.md`, under your run notes: time, GPU, dtype, row and the check lines. Push.
4. Tell Yvette.

---

## Part 2 · temp07, temp12, topp09 (only if Yvette asks)

Yvette first pushes `gen/answers_{temp07,temp12,topp09}.jsonl` and `labels/numbers_{temp07,temp12,topp09}.jsonl` to `main`.

1. Pull on the server:

```bash
git pull origin main
```

2. Optional quick check on the first 20 answers, into a scratch folder:

```bash
python features/extract.py --setting temp07 --limit 20 --out-dir features/check
```

3. Full run. Add `--row predicting` only if that's our row:

```bash
python features/extract.py --setting temp07 temp12 topp09 --out-dir <server features folder>
```

   - Time: about 3 s per answer on the UNLV GPU (from the greedy and beam5 run), so about 50 min per setting and 2.5 h for all three.
   - Checks: the same as Part 1. The script stops by itself if an answers file was made with a different model.

4. Copy `temp07.npz`, `temp12.npz` and `topp09.npz` into `GoG-shared/features`, add the run note to `LOG.md`, push, and tell Yvette.

**Confidence:** Yvette runs it for her settings on her own GPU (it fits in 8 GB). If she asks you to run it instead:

```bash
python baseline/confidence.py --setting temp07 temp12 topp09
```

It takes a few minutes. Push the three `baseline/confidence_*.npz` files; they go in git.

---

## If a label rule changes after the hand-check

The feature files store each number's label, and the detectors check that it matches the confidence file. So after relabeling, confidence and features are redone for every setting that was relabeled, unless we add a script that rewrites only the labels (ask Claude).

## What goes back to Yvette

| What | Where |
|---|---|
| `greedy.npz`, `beam5.npz` | `GoG-shared/features` (Drive) |
| `temp07.npz`, `temp12.npz`, `topp09.npz` (Part 2 only) | `GoG-shared/features` (Drive) |
| Run notes | `LOG.md` (git) |
