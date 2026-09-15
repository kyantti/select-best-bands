# 05 – Genetic search with on-disk candidate cache and resume

Status: done
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

- [x] `uv run pytest -q tests/test_protocol.py -k ga` passes.
- [x] `CUDA_VISIBLE_DEVICES=1 uv run python ga.py 99 --population 4 --generations 2 --epochs 1` writes `out/tables/exp_99_ga_stats.csv`, `exp_99_candidates.csv`, `exp_99_ga_config.json`, `exp_99_ga_summary.json`, `exp_99_cnn_results_R_G_B.csv` and `out/figures/exp_99_fitness_evolution.png`, every triplet with its nm.
- [x] Killing that run mid-way and relaunching the same command prints `restored > 0`, evaluates no candidate already in the CSV, and ends with the same winner as an uninterrupted run.
- [x] Relaunching with `--epochs 2` aborts with a message naming the fitness-contract mismatch.
- [x] `rm out/*/exp_99_*` afterwards.
- [x] `uv run python -c "import ga"` succeeds and `SKIP_SYNC=1 ./init.sh` imports `ga` again and is green.

## Evidence

**The smoke experiment writes all six outputs** (15 Sep,
`CUDA_VISIBLE_DEVICES=1 uv run python ga.py 99 --population 4 --generations 2 --epochs 1`,
32.9 s): 8 unique candidates, 3 cache hits, winner `(391, 201, 105)` =
925.89 / 664.49 / 535.58 nm, weighted F1 `0.4696528369802027`.  Every triplet
carries its nm in `exp_99_candidates.csv`, `exp_99_ga_stats.csv`
(`best_wavelengths_nm`), `exp_99_cnn_results_391_201_105.csv` and the figure
title.

```
gen,nevals,population_size,unique_population_candidates,avg,std,min,max,best,best_wavelengths_nm
0,4,4,4,0.29201609251674787,0.11106185668681742,0.12759834871296744,0.4209702797202798,"[399, 148, 424]","[937.08, 593.06, 972.14]"
1,4,4,3,0.40108184008075826,0.06563601136335576,0.29273396390227074,0.4696528369802027,"[391, 201, 105]","[925.89, 664.49, 535.58]"
2,3,4,3,0.3559835747294876,0.0747171178779497,0.23885521885521882,0.4209702797202798,"[399, 148, 424]","[937.08, 593.06, 972.14]"
```

**Resume is a replay, not a different search.**  The candidates CSV was truncated
to its first 4 rows (exactly what a kill leaves: each row is flushed and fsynced
before the next candidate starts) and the run relaunched.  It printed
`restored 4`, trained only the 4 candidates that were missing, re-evaluated none
of the 4 it had, and ended on the same winner.  Against the uninterrupted run:

- `exp_99_ga_stats.csv` — **identical**, generation by generation.
- `exp_99_candidates.csv` — **identical** on every column but `seconds`: same
  candidates, same order, same fitness.
- `exp_99_cnn_results_391_201_105.csv` — **identical**.

`--no-evaluate` then replayed the whole search from the warm CSV: `restored 8`,
`evaluated 0`, same winner.  That is the path ticket 06 needs.

**The fitness contract is enforced.**  Relaunching with `--epochs 2` exits 1 with

```
ga.py: out/tables/exp_99_candidates.csv was written under another fitness contract
(epochs differ); its numbers are not comparable. Use a new experiment number, or
delete out/tables/exp_99_ga_config.json and the CSV.
```

**Tests:** `uv run pytest -q tests/test_protocol.py -k ga` -> 12 passed; the whole
file -> 61 passed (51 before).  The GA tests use an injected evaluator and need
no GPU and no `data/`.

**Cleanup:** `rm out/tables/exp_99_* out/figures/exp_99_*` done; `git status` on
`out/` is clean.

**Gate:** `uv run python -c "import ga"` succeeds and `SKIP_SYNC=1 ./init.sh`
imports `ga` again and is green end to end.

**Files:** `ga.py` (operators, `CandidateFitness`, `CandidateCache`,
`run_search`, `fitness_contract`, `make_evaluator`, the outputs, the figure and
the search CLI), `tests/test_protocol.py` (+12 tests), `init.sh` (`import ga`
restored; its stale comment now names `check_data.py`, which ticket 10 owns).

## Comments

**`out/` is protected by an ownership rule.**  A search refuses to start when
`exp_NN_ga_stats.csv` exists but `exp_NN_candidates.csv` does not: that number
belongs to a run this code cannot resume (experiments 1–20), so continuing would
overwrite a result rather than extend one.  A number with both files is a resume
and rewrites its own end-of-run outputs on purpose.

**Review.** `/code-review` ran both axes; both confirmed the operators, the
evolution loop, `candidate_fitness_key` and the seeding point are literal against
fig-aflatoxin.  Acted on: the winner and `unique_candidates` now range over **the
candidates this run proposed**, not every row in the file — the cache may hold
rows from a run with other GA parameters, or ticket 06's preloaded replay, and a
winner the search never met is not a search result (regression test
`test_ga_winner_is_never_a_candidate_the_search_did_not_propose`); the sidecar is
rewritten after the GA-parameter warning, which otherwise left it describing the
previous run; `CandidateCache.restored` is snapshotted at load instead of being a
property over the growing entry dict; the operator test now checks the bounds of
the initialized candidates too; `experiment_paths` no longer mixes a `str` into a
`dict[str, Path]` and the winner's own path comes from `winner_results_path`
rather than being assembled inline.  Kept deliberately: the extra `candidate_seed`
column in `exp_NN_cnn_results_R_G_B.csv`, which lets any winner be re-checked
with `--evaluate`.

**Statistics column names keep v1's spelling** (`gen, nevals, avg, std, min,
max, best`) so `plot_fitness_evolution.py` and `create_summary_table.py` still
read them; `population_size`, `unique_population_candidates` and
`best_wavelengths_nm` are added beside them, and `best` stays the JSON triplet
v1 wrote.
