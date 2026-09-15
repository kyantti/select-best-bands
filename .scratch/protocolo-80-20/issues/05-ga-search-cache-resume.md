# 05 – Genetic search with on-disk candidate cache and resume

Status: ready-for-agent
Blocked by: 04

## Description

`uv run python ga.py N` runs the DEAP search on validation weighted F1, appends every evaluated candidate to a CSV as soon as it finishes, and continues from that CSV when relaunched with the same experiment number. `import ga` returns to `init.sh`.

Behaviour (copied from fig-aflatoxin's genetic module and GA stage):

- Individuals are plain lists (no `deap.creator`); tournament of 3; blend crossover α 0.5 with integer truncation; gaussian mutation μ 0 σ 20 per-gene probability 0.3; both followed by a repair that guarantees three distinct bands **without reordering** them; clamp to the spectral-axis range. Population 20, 25 generations, cxpb 0.8, mutpb 0.15, hall of fame of one that only observes.
- Fitness key `(weighted_f1, -ordinal_mae, -indices)`. The reported winner is re-tie-broken over every evaluated candidate that ties the best, independent of DEAP's internal order.
- `random.seed(STUDY_SEED)` immediately before the population is created; the evolution's random state is saved and restored around every training call so the trajectory does not depend on how many candidates came from cache.
- Per-candidate seed from ticket 04 (`candidate_seed(bands, identity)`), so a fitness does not depend on when or by whom the candidate is found.
- Search only ever loads train-fit and validation rows.
- Cache: `out/tables/exp_NN_candidates.csv` (bands, candidate seed, four metrics, first generation, seconds, wavelengths nm), appended with flush + fsync after each evaluation, floats written with `repr`; sidecar `out/tables/exp_NN_ga_config.json` with the fitness contract (candidate seed, training hyperparameters, device type, partition seed and proportions, band count, data identity, library versions, ResNet weights) and the GA parameters. On load it aborts if the fitness contract differs, checks each row's candidate seed, and warns if only the GA parameters differ. Resume is a replay: same study seed, same trajectory, cached candidates are free hits. No DEAP state is serialized.
- `--no-evaluate`: any cache miss is an error (used by ticket 06).
- `run_search(band_count, evaluator, cache, params)` accepts an injected evaluator so tests need no torch.
- Outputs: `exp_NN_ga_stats.csv` (gen, nevals, population_size, unique_population_candidates, avg, std, min, max, best, best_wavelengths_nm), `exp_NN_candidates.csv`, `exp_NN_ga_config.json`, `exp_NN_ga_summary.json` (winner in indices and nm, validation metrics, elapsed, unique/hits/restored, versions), `exp_NN_cnn_results_R_G_B.csv` (winner's validation metrics with nm), `out/figures/exp_NN_fitness_evolution.png` with nm in the title.
- CLI overrides for smoke: `--population`, `--generations`, `--epochs` (flags, not environment variables).

Tests (fake evaluator, no torch): operators keep three distinct in-range bands after init/crossover/mutation and preserve order on repair; permutations are evaluated separately and repeats are cache hits; a fake evaluator that reseeds `random` does not change the candidate sequence; a second run with the warm CSV makes zero evaluator calls and reports the same winner; a cache written under another candidate seed is refused.

## Done when

- [ ] `uv run pytest -q tests/test_protocol.py -k ga` passes.
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python ga.py 99 --population 4 --generations 2 --epochs 1` writes `out/tables/exp_99_ga_stats.csv`, `exp_99_candidates.csv`, `exp_99_ga_config.json`, `exp_99_ga_summary.json`, `exp_99_cnn_results_R_G_B.csv` and `out/figures/exp_99_fitness_evolution.png`, every triplet with its nm.
- [ ] Killing that run mid-way and relaunching the same command prints `restored > 0`, evaluates no candidate already in the CSV, and ends with the same winner as an uninterrupted run.
- [ ] Relaunching with `--epochs 2` aborts with a message naming the fitness-contract mismatch.
- [ ] `rm out/*/exp_99_*` afterwards.
- [ ] `uv run python -c "import ga"` succeeds and `SKIP_SYNC=1 ./init.sh` imports `ga` again and is green.

## Evidence

_(none yet)_

## Comments

