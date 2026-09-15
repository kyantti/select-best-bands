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
- `ga.py <experiment_number>` – the GA search with CNN fitness. Being rewritten by the `protocolo-80-20` feature: `ga.py --evaluate R G B` (train and score one triplet) works; the search itself arrives with ticket 05.
- `cnn/engine.py`, `cnn/data_setup.py`, `cnn/model.py` – training loops, the data path (manifest, partition, NPZ, normalization, dataset), and the ResNet50 contract (data identity, per-candidate seed, metrics, `evaluate_candidate`).
- `run.sh` – runs a range of experiments sequentially with `nohup`-style logs.
- `check_data.py`, `create_summary_table.py`, `plot_fitness_evolution.py` – post-hoc utilities.
- `data/` – gitignored: `cropped_hypercubes/` (symlink to the 1124 NPZ crops), `cropped_hypercubes.csv`, `spectral_axes.csv`, `evaluation_partitions.csv`.
- `out/tables`, `out/figures`, `out/logs` – experiment results. **Tracked in git and part of the thesis record.**
- `tests/test_protocol.py` – the protocol tests (no GPU, no `data/`); `tests/data/*.csv` are the versioned reference manifests they check against.
- `.scratch/<feature>/` – issue tracker: `spec.md` plus one markdown ticket per `issues/NN-*.md`. Tracked in git.

## Invariants (do not break)

- **Never overwrite or delete existing `out/` results.** Each experiment number `NN` owns `out/tables/exp_NN_*` and `out/figures/exp_NN_*`. New runs use a new number.
- **Never commit `data/`.** It is gitignored on purpose.
- **Full runs are hours on an A100.** Do not start `ga.py` with the default `GENERATIONS`/`POPULATION_SIZE`/`NUM_EPOCHS` of `config.py` unless the user asked for a real experiment. Use a smoke configuration (see `.scratch/repo-health/issues/03-smoke-run-config.md`) to verify code paths.
- **Do not change the GA or CNN hyperparameters** in `config.py` unless that is the feature. Past results depend on them.
- Use `uv run ...` for every Python invocation. Never `pip install` into `.venv`.

## Verification and Definition of Done

```bash
./init.sh                        # sync deps, compile, import config + cnn, check data files
uv run python -c "import config" # fastest single check: catches broken imports
SKIP_SYNC=1 ./init.sh            # when deps are already installed
```

A change to `ga.py` or `cnn/` is not done until `./init.sh` is green. A change that
touches the training/evaluation path is not done until a smoke run of `ga.py`
produces `out/tables/exp_NN_ga_stats.csv` and `exp_NN_cnn_results_*.csv` for a
throwaway experiment number (use 90–99 and delete the outputs afterwards).

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
