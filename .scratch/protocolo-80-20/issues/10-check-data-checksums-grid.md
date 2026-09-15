# 10 – `check_data.py`: one-off checksum verification and per-class crop grid

Status: ready-for-agent
Blocked by: 03

## Description

`uv run python check_data.py` verifies once that the 1 124 linked NPZ files match the `artifact_checksum` column of the cropped-hypercube manifest, and saves `sanity-check/data_sanity_check.png` with two crops per class rendered through the same model-input preparation the network sees, with the band triplet in nm in the title. Per-load checksum verification is deliberately not done anywhere else (spec: cheap checks on every load, checksums once here).

Rewrite of the existing `check_data.py`, about 60 lines, no GPU.

## Done when

- [ ] `uv run python check_data.py` prints `1124 files, 0 checksum mismatches` and writes `sanity-check/data_sanity_check.png` (8 panels, class labels C0..C3, nm in the title).
- [ ] Corrupting a copy of one NPZ in a temporary manifest makes it report one mismatch and exit non-zero (record the command).
- [ ] `SKIP_SYNC=1 ./init.sh` is green (compileall includes the script).

## Evidence

_(none yet)_

## Comments

