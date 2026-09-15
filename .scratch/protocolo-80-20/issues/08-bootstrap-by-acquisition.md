# 08 – Bootstrap confidence interval by acquisition

Status: ready-for-agent
Blocked by: 07

## Description

`uv run python bootstrap.py N` reads the per-crop test predictions of experiment N, resamples whole acquisitions 5000 times with `default_rng(1729)`, and writes the 95 % percentile interval of the weighted F1 plus per-acquisition weighted F1 and accuracy, so the test result is always quoted with its width. No model is executed.

Copied almost literally from fig-aflatoxin's `scripts/bootstrap-test-interval.py`; defaults from `config.py`; output `out/tables/exp_NN_bootstrap_R_G_B.json`; the bands (indices and nm) come from the predictions/metrics files of experiment N, so no `--bands` flag is needed when only one final model exists for N.

## Done when

- [ ] `uv run python bootstrap.py 21` prints point weighted F1 `0.722` and 95 % interval `[0.645, 0.848]` (3 decimals) and writes `out/tables/exp_21_bootstrap_366_262_225.json` with the interval, the number of resamples, the seed and the per-acquisition table.
- [ ] Running it twice gives identical JSON.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

