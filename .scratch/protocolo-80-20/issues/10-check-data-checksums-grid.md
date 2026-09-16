# 10 – `check_data.py`: one-off checksum verification and per-class crop grid

Status: done
Blocked by: 03

## Description

`uv run python check_data.py` verifies once that the 1 124 linked NPZ files match the `artifact_checksum` column of the cropped-hypercube manifest, and saves `sanity-check/data_sanity_check.png` with two crops per class rendered through the same model-input preparation the network sees, with the band triplet in nm in the title. Per-load checksum verification is deliberately not done anywhere else (spec: cheap checks on every load, checksums once here).

Rewrite of the existing `check_data.py`, about 60 lines, no GPU.

## Done when

- [x] `uv run python check_data.py` prints `1124 files, 0 checksum mismatches` and writes `sanity-check/data_sanity_check.png` (8 panels, class labels C0..C3, nm in the title).
- [x] Corrupting a copy of one NPZ in a temporary manifest makes it report one mismatch and exit non-zero (record the command).
- [x] `SKIP_SYNC=1 ./init.sh` is green (compileall includes the script).

## Evidence

**The 1124 linked artifacts are the ones the manifest describes** (16 Sep, 3 min
12 s, no GPU):

```
$ uv run python check_data.py
1124 files, 0 checksum mismatches
loading 868 train crops
normalization on train foreground: mean (0.65546626556172, 0.6584057421588269, 0.22467686763531047), std (0.11717051462839731, 0.125102268224151, 0.07519302019349307)
wrote /home/pablosetra/Documents/tfg/select-best-bands/sanity-check/data_sanity_check.png
```

Those six numbers are **digit for digit** the `normalization` block of
`out/tables/exp_21_final_metrics_366_262_225.json`, so the figure is drawn
through the very normalization the reported model was trained under, and the
crops behind it are the same crops.

**A changed artifact is caught and stops the run.** A temporary manifest beside
1124 symlinks to the real NPZ files, one of them replaced by its first 1000
bytes:

```
$ S=$SCRATCH/corrupt
$ mkdir -p "$S/cropped_hypercubes" && cp data/cropped_hypercubes.csv "$S/"
$ for f in data/cropped_hypercubes/*.npz; do ln -s "$(readlink -f "$f")" "$S/cropped_hypercubes/$(basename "$f")"; done
$ VICTIM=$(ls "$S/cropped_hypercubes" | head -1)
$ rm "$S/cropped_hypercubes/$VICTIM" && head -c 1000 "data/cropped_hypercubes/$VICTIM" > "$S/cropped_hypercubes/$VICTIM"
$ uv run python check_data.py --manifest "$S/cropped_hypercubes.csv"
  mismatched: 006236fce64a824c12f4bc9c80a5ec488646ca9e5ed7f24369f3e8c30a43f533
1124 files, 1 checksum mismatch
$ echo $?
1
```

The figure is not written in that run: a grid of crops that are not the crops
the manifest describes would be a sanity check that lies.

**The figure**: `sanity-check/data_sanity_check.png`, 8 panels (two train crops
of each of C0, C1, C2, C3), titled `bands 366/262/225 = 891.02 nm / 747.5 nm /
697.05 nm`. Held-out crops are dropped before a single NPZ is opened, so the
grid cannot show one.

**The gate**: `SKIP_SYNC=1 ./init.sh` green with **97 tests** (88 before), and
`init.sh` now *imports* `check_data` instead of only compiling it — the broken
import ticket 03 left behind is gone. `git status --short out/` is empty:
nothing here writes to the experiment record.

**After `/code-review`** (both axes), six changes:

- `cnn/data_setup.artifact_path` is now the one place a manifest row becomes a
  path; `check_data` had grown a second copy of those checks that had already
  dropped the backslash one.
- `train_final.load_crops` moved to `cnn/data_setup.load_partition_crops` with
  the manifest as a parameter, so `check_data --manifest` reuses it instead of
  keeping a third copy of the SpectralAxis guard.
- **The caption was wrong**: it said "one grey scale for the whole grid" of a
  three-channel false-colour figure. It now says the bands are drawn as R/G/B.
- **The grid hid the failure it exists to show**: `prepare_model_input` zeroes
  the background in *normalized* space, so a global min/max put it mid-range —
  where an all-zero channel would have looked exactly like no fig at all. The
  scale is now fitted to the fig pixels and the background painted flat grey
  outside it.
- One class per column (it was filled class-major across a 4-wide grid, so a
  row read C0, C0, C1, C1).
- `--per-class` dropped: the ticket fixes the count at two, and the number
  lives in `config.py` like every other constant.

The figure was redrawn after all six. Moving `load_crops` touches the
training path, so the smoke chain ran end to end:
`./run.sh 99 --population 4 --generations 1 --epochs 1` (~7,5 min) produced the
ten `exp_99_*` tables, the three figures and the `.pt`, with the same winner
`(391, 201, 105)` ticket 05 recorded for those settings; deleted afterwards,
`git status --short out/` empty, and `data/evaluation_partitions.csv` still
`diff`-identical to `tests/data/reference_evaluation_partitions.csv`.
`SKIP_SYNC=1 ./init.sh` green, 97 tests.

## Comments

