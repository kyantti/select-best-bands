# CLAUDE.md

Genetic algorithm (DEAP) that selects the best 3 of 448 hyperspectral bands for
aflatoxin classification in figs, scoring each band triplet by fine-tuning a
ResNet50 on the resulting RGB images. See `README.md` for the research context.

## Startup Workflow (every session)

1. `pwd` must be the repo root (`select-best-bands`).
2. Read this file, then `progress.md`, then the open tickets in `.scratch/*/issues/`.
3. Run `./init.sh`. If it is red, fixing that is the first task.
4. `git log --oneline -5` and `git status` before editing.

## Layout

- `config.py` – **every constant of the experiment**: paths, class names, device, seeds, partition proportions, GA parameters, CNN hyperparameters, bootstrap. Imported by everything else.
- `split_dataset.py` – writes `data/evaluation_partitions.csv`: the Acquisition-grouped, class-stratified train/validation/test split.
- `ga.py <experiment_number>` – the GA search with CNN fitness: genetic operators, the on-disk candidate cache that makes a relaunch a replay, and the outputs. `ga.py --evaluate R G B` trains and scores one triplet on its own; `--no-evaluate` replays a search from a warm cache without training anything.
- `train_final.py <experiment_number> [--bands R G B]` – the final model: trains on all 28 train Acquisitions, saves the state dict, and only then opens the held-out test crops for a single inference. Without `--bands` it reads the winner from `exp_NN_ga_summary.json`. It refuses a number an earlier protocol owns, and refuses to rewrite its own result under settings that would change it.
- `bootstrap.py <experiment_number> [--bands R G B]` – the width of the test result: resamples whole test Acquisitions 5000 times over the predictions `train_final.py` wrote and records the 95 % percentile interval of the weighted F1 plus the per-acquisition table. Runs no model and reads no crop.
- `cnn/engine.py`, `cnn/data_setup.py`, `cnn/model.py` – training loops, the data path (manifest, partition, NPZ, normalization, dataset), and the ResNet50 contract (data identity, per-candidate seed, metrics, `evaluate_candidate`).
- `run.sh <experiment_number> [flags]` – the whole chain for one number: `split_dataset.py` → `ga.py N` → `train_final.py N` → `bootstrap.py N`, logging to `out/logs/experiment_N.log`, `final_N.log` and `bootstrap_N.log`, with `CUDA_VISIBLE_DEVICES` defaulting to GPU 1. Flags after the number are routed to the steps that take them (`--population`/`--generations`/`--no-evaluate` to `ga.py`, `--epochs` to `ga.py` and `train_final.py`, `--bands` to `train_final.py` and `bootstrap.py`, `--resamples`/`--seed` to `bootstrap.py`), so a smoke chain is one command.
- `plot_fitness_evolution.py <experiment_number>` – redraws `exp_NN_fitness_evolution.png` from `exp_NN_ga_stats.csv` and `exp_NN_ga_summary.json` alone, byte for byte what the search drew. No GPU, no crops.
- `check_data.py`, `create_summary_table.py` – post-hoc utilities of the earlier protocol (`create_summary_table.py` reads experiments 1–10 only).
- `data/` – gitignored: `cropped_hypercubes/` (symlink to the 1124 NPZ crops), `cropped_hypercubes.csv`, `spectral_axes.csv`, `evaluation_partitions.csv`.
- `out/tables`, `out/figures`, `out/logs` – experiment results. **Tracked in git and part of the thesis record.**
- `tests/test_protocol.py` – the protocol tests (no GPU, no `data/`); `tests/data/*.csv` are the versioned reference manifests they check against.
- `.scratch/<feature>/` – issue tracker: `spec.md` plus one markdown ticket per `issues/NN-*.md`. Tracked in git.

## Invariants (do not break)

- **Never overwrite or delete existing `out/` results.** Each experiment number `NN` owns `out/tables/exp_NN_*` and `out/figures/exp_NN_*`. New runs use a new number.
- **Experiments 1–20 are the earlier protocol; 21 is the replay of the 10 Sep 2026 search, so the first new real run is 22.** `ga.py` and `train_final.py` refuse a number an earlier protocol owns. Throwaway smoke runs take 90–99 and are deleted afterwards (`out/tables`, `out/figures`, `out/models` and `out/logs`).
- **Never commit `data/`.** It is gitignored on purpose.
- **Full runs are hours on an A100.** Do not start `ga.py` or `run.sh` with the default `GENERATIONS`/`POPULATION_SIZE`/`NUM_EPOCHS` of `config.py` unless the user asked for a real experiment. To verify code paths use the smoke flags on a throwaway number: `./run.sh 99 --population 4 --generations 1 --epochs 1`.
- **Do not change the GA or CNN hyperparameters** in `config.py` unless that is the feature. Past results depend on them.
- Use `uv run ...` for every Python invocation. Never `pip install` into `.venv`.

## Verification and Definition of Done

```bash
./init.sh                        # sync deps, compile, import config + cnn, check data files
uv run python -c "import config" # fastest single check: catches broken imports
SKIP_SYNC=1 ./init.sh            # when deps are already installed
```

A change to `config.py`, `split_dataset.py`, `ga.py`, `train_final.py`,
`bootstrap.py` or `cnn/` is not done until `./init.sh` is green. A change that
touches the training/evaluation path is not done until a smoke chain on a
throwaway number has run end to end:

```bash
./run.sh 99 --population 4 --generations 1 --epochs 1
rm -f out/tables/exp_99_* out/figures/exp_99_* out/models/exp_99_* out/logs/*_99.log
```

It must leave `exp_99_ga_stats.csv`, `exp_99_candidates.csv`,
`exp_99_final_metrics_*.json`, `exp_99_bootstrap_*.json` and the three figures
(`fitness_evolution`, `confusion_matrix`, `training_history`). Delete them
afterwards — `rm -f`, because a glob that matches nothing must not stop the
line: `out/` is the thesis record, not a scratch directory.

## Working rules

- **One feature at a time.** Pick one `Status: ready-for-agent` ticket in `.scratch/*/issues/` whose `Blocked by:` tickets are all `done`, set it `in-progress`, and name it in `progress.md`. Never work on `needs-triage`, `needs-info` or `ready-for-human` items.
- Do not widen scope: no refactors, renames, or "cleanups" outside the active feature.
- Record evidence (command + result) under the ticket's `## Evidence` when marking `done`. If you need the author's decision, set the ticket to `needs-info` with the question under `## Comments` and stop.
- Status vocabulary and transitions: `docs/agents/triage-labels.md`.
- Long-running experiments: launch through `run.sh` or `nohup`, log to `out/logs/`, and record the PID and expected finish in `progress.md`.

## End of session

1. Update `progress.md` (state, blockers, next step).
2. Update the ticket's `Status:` and `## Evidence` in `.scratch/`.
3. Leave `./init.sh` green. Commit only when asked.

## Agent skills

### Issue tracker

Local markdown: one ticket per file under `.scratch/<feature>/issues/`, tracked in git; there is no external tracker. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default roles plus `in-progress` and `done`, as the `Status:` line of each ticket. One status per ticket. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root, created lazily. See `docs/agents/domain.md`.
