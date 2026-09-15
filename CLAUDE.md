# CLAUDE.md

Genetic algorithm (DEAP) that selects the best 3 of 448 hyperspectral bands for
aflatoxin classification in figs, scoring each band triplet by fine-tuning a
ResNet50 on the resulting RGB images. See `README.md` for the research context.

## Startup Workflow (every session)

1. `pwd` must be the repo root (`select-best-bands`).
2. Read this file, then `feature_list.json`, then `progress.md`.
3. Run `./init.sh`. If it is red, fixing that is the first task.
4. `git log --oneline -5` and `git status` before editing.

## Layout

- `ga.py <experiment_number>` – the whole pipeline: GA loop + CNN fitness. Single entry point.
- `cnn/engine.py`, `cnn/data_setup.py`, `cnn/util/helper_functions.py` – training loop, dataset, plots. Imported by `ga.py`.
- `cnn/train.py`, `cnn/model_info.py` – standalone scripts run from inside `cnn/` (bare imports). Not used by the GA.
- `run.sh` – runs a range of experiments sequentially with `nohup`-style logs.
- `check_data.py`, `create_summary_table.py`, `plot_fitness_evolution.py` – post-hoc utilities.
- `train_dataset.csv`, `test_dataset.csv` – manifests (`filepath,label`) pointing into `data/processed/` (gitignored, ~GBs of `.npy`).
- `out/tables`, `out/figures`, `out/logs` – experiment results. **Tracked in git and part of the thesis record.**
- `tests/data/*.csv` – reference manifests only. There is no automated test suite yet.

## Invariants (do not break)

- **Never overwrite or delete existing `out/` results.** Each experiment number `NN` owns `out/tables/exp_NN_*` and `out/figures/exp_NN_*`. New runs use a new number.
- **Never commit `data/`.** It is gitignored on purpose.
- **Full runs are hours on an A100.** Do not start `ga.py` with default `GENERATIONS`/`POPULATION_SIZE`/`NUM_EPOCHS` unless the user asked for a real experiment. Use a smoke configuration (see feat-003) to verify code paths.
- **Do not change the GA or CNN hyperparameters** at the top of `ga.py` unless that is the feature. Past results depend on them.
- Use `uv run ...` for every Python invocation. Never `pip install` into `.venv`.

## Verification and Definition of Done

```bash
./init.sh                      # sync deps, compile, import ga.py, check manifests
uv run python -c "import ga"   # fastest single check: catches broken imports
SKIP_SYNC=1 ./init.sh          # when deps are already installed
```

A change to `ga.py` or `cnn/` is not done until `./init.sh` is green. A change that
touches the training/evaluation path is not done until a smoke run of `ga.py`
produces `out/tables/exp_NN_ga_stats.csv` and `exp_NN_cnn_results_*.csv` for a
throwaway experiment number (use 90–99 and delete the outputs afterwards).

## Working rules

- **One feature at a time.** Pick a `ready-for-agent` feature from `feature_list.json` whose `dependencies` are all `done`, set it `in-progress`, and name it in `progress.md`. Never work on `needs-triage`, `needs-info` or `ready-for-human` items.
- Do not widen scope: no refactors, renames, or "cleanups" outside the active feature.
- Record evidence (command + result) in `feature_list.json` when marking `done`. If you need the author's decision, set the feature to `needs-info` with the question in `evidence` and stop.
- Status vocabulary and transitions: `docs/agents/triage-labels.md`.
- Long-running experiments: launch through `run.sh` or `nohup`, log to `out/logs/`, and record the PID and expected finish in `progress.md`.

## End of session

1. Update `progress.md` (state, blockers, next step).
2. Update `feature_list.json` statuses and evidence.
3. Leave `./init.sh` green. Commit only when asked.

## Agent skills

### Issue tracker

Work is tracked as entries in `feature_list.json` at the repo root; there is no external tracker. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default roles plus `in-progress` and `done`, all as values of each feature's `status` field. One status per feature. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root, created lazily. See `docs/agents/domain.md`.
