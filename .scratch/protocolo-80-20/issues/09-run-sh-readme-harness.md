# 09 – `run.sh` chain, README, fitness plot, `init.sh` and `CLAUDE.md`

Status: ready-for-agent
Blocked by: 08

## Description

One command reproduces the whole chain for an experiment number, and the documentation and harness describe the new protocol instead of the old one.

- `run.sh N` (N required; 21 is the replay, so the first real run is 22) with `set -euo pipefail`, `CUDA_VISIBLE_DEVICES` defaulting to GPU 1, runs split → `ga.py N` → `train_final.py N` → `bootstrap.py N`, logging to `out/logs/experiment_N.log`, `final_N.log`, `bootstrap_N.log`. Extra arguments are passed through to `ga.py` and `train_final.py` so a smoke chain is possible.
- `plot_fitness_evolution.py` takes N as argument and reads the new `ga_stats` columns.
- `create_summary_table.py` stays as the v1 tool (out of scope to adapt beyond not breaking).
- README: the protocol in plain words (acquisition-grouped 80/20 split, validation 80/20 inside train, train-foreground normalization, validation weighted F1 as fitness, final model with one test read, bootstrap by acquisition), how to launch, resume and verify, the v1/v2 boundary at experiment 21, the results table with experiment 21, and the exact command for the real run (`nohup ./run.sh 22 > out/logs/run_22.log 2>&1 &`, `tail -f`, `wc -l` on the candidates CSV). Absorbs repo-health/06.
- `init.sh`: imports `ga`, `cnn.data_setup`, `cnn.engine`, `cnn.model`; checks `data/cropped_hypercubes.csv`, `data/spectral_axes.csv`, `data/evaluation_partitions.csv` (warn, not fail, when `data/` is absent); runs pytest; final hint unchanged.
- `CLAUDE.md` Layout, Invariants and Verification name the four scripts, `config.py`, the smoke flags and the 21 boundary. `progress.md` reflects the new state.

## Done when

- [ ] `CUDA_VISIBLE_DEVICES=1 ./run.sh 99 --population 4 --generations 1 --epochs 1` completes all four steps and leaves `exp_99_ga_stats.csv`, `exp_99_candidates.csv`, `exp_99_final_metrics_*.json`, `exp_99_bootstrap_*.json` and the three figures; then `rm out/*/exp_99_*` and `out/models/exp_99_*`.
- [ ] `uv run python plot_fitness_evolution.py 21` regenerates `out/figures/exp_21_fitness_evolution.png`.
- [ ] `./init.sh` (full, with sync) is green and prints the three manifest checks and the pytest summary.
- [ ] README no longer mentions `train_dataset.csv`, `ELITISM_SIZE`, `engine2` or accuracy as the fitness, and states the 21 boundary.
- [ ] `grep -n "config.py\|train_final\|bootstrap.py\|split_dataset" CLAUDE.md` finds all four.

## Evidence

_(none yet)_

## Comments

