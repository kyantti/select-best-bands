# 06 – Replay the 10 Sep search without GPU as experiment 21

Status: done
Blocked by: 05

## Description

Experiment 21 is the port of the real run of 10 Sep 2026. Its candidate cache is built from the 383 candidates that run evaluated, and `ga.py 21 --no-evaluate` walks the exact same trajectory: same per-generation history, same winner, zero training. This proves the operators, seeding, random-state isolation, cache and tie-breaking are ported unchanged, and it leaves `exp_21_*` as the record that tickets 07 and 08 complete.

Source: `runs/run_ga_study/8b6e000fa03e3b1cb91dc72d3f11614933fb752e88ba45dfe4d4ff9639524cd0/{candidate_diagnostics.csv,generation_history.csv,ga_study_report.json}` in `~/Documents/fig-aflatoxin`. A short conversion script (kept in the scratchpad, not the repo) writes `out/tables/exp_21_candidates.csv` and its `exp_21_ga_config.json` sidecar in the ticket-05 format, with the candidate seed of each row recomputed from the real data identity so the cache loader accepts it.

## Done when

- [x] `out/tables/exp_21_candidates.csv` has 383 rows and `uv run python ga.py 21 --no-evaluate` loads it without a contract mismatch.
- [x] The run ends with 383 restored, 0 evaluations, winner 366/262/225 (891.0 / 747.5 / 697.1 nm), validation weighted F1 `0.8700979843225085`.
- [x] `out/tables/exp_21_ga_stats.csv` matches `generation_history.csv` of the real run generation by generation on avg / std / min / max / best candidate (compare with a one-liner; record the command).
- [x] `out/figures/exp_21_fitness_evolution.png` exists.
- [x] No file under `out/` from experiments 1–20 changed (`git status --short out/` shows only `exp_21_*`).

## Evidence

**Done (16 Sep).** The port walks the 10 Sep trajectory generation by generation with zero training.

1. Warm cache built from the real run by a throwaway script in the scratchpad
   (`build_exp21_cache.py`: `ga.load_selection_data()` for the data identity, then one row per
   candidate of `candidate_diagnostics.csv` in ticket-05's `CANDIDATE_FIELDS` order, with
   `candidate_seed(candidate, identity)` recomputed here and `wavelengths_nm` taken from this
   repo's `spectral_axes.csv`; the sidecar is `ga.fitness_contract(...)` + `SearchParameters()`).
   `383 candidates in candidate_diagnostics.csv` → `out/tables/exp_21_candidates.csv` (383 rows)
   and `out/tables/exp_21_ga_config.json`.

   The winner's row carries the numbers ticket 04 verified against the real run:
   `366,262,225,3104252108,0.8700979843225085,0.8763610130071496,0.1875,0.8711063372717508,21,...`
   — the seed the live data derives is the seed the 10 Sep run trained under, so the cache loads
   with no contract mismatch.

2. `uv run python ga.py 21 --no-evaluate` (3 min 17 s, all of it loading the 868 crops):

   ```
   exp_21: 448 bands, population 20, 25 generations, 50 epochs, restored 383
   winner         (366, 262, 225)
   wavelengths_nm 891.02, 747.5, 697.05
   weighted_f1    0.8700979843225085
   unique 383 | hits 451 | restored 383 | evaluated 0
   ```

   `unique 383 | hits 451` is the real run's `ga_study_report.json` (`"unique_evaluations": 383`,
   `"hits": 68`) read the other way round: 451 proposals, 383 of them distinct.

3. Stats compared generation by generation with the real `generation_history.csv`:

   ```
   uv run python -c "
   import csv, json
   ours = list(csv.DictReader(open('out/tables/exp_21_ga_stats.csv')))
   theirs = list(csv.DictReader(open('/home/pablosetra/Documents/fig-aflatoxin/runs/run_ga_study/8b6e000fa03e3b1cb91dc72d3f11614933fb752e88ba45dfe4d4ff9639524cd0/generation_history.csv')))
   assert len(ours) == len(theirs), (len(ours), len(theirs))
   pairs = [('avg','weighted_f1_avg'),('std','weighted_f1_std'),('min','weighted_f1_min'),('max','weighted_f1_max')]
   bad = [(int(a['gen']), k) for a, b in zip(ours, theirs) for k, j in pairs if float(a[k]) != float(b[j])]
   bad += [(int(a['gen']),'best') for a, b in zip(ours, theirs) if json.loads(a['best']) != json.loads(b['best_candidate_indices'])]
   bad += [(int(a['gen']),'popsize') for a, b in zip(ours, theirs) if (a['population_size'], a['unique_population_candidates']) != (b['population_size'], b['unique_population_candidates'])]
   print(f'{len(ours)} generations compared, {len(bad)} mismatches', bad)
   "
   → 26 generations compared, 0 mismatches []
   ```

   Exact float equality, not a tolerance, on avg / std / min / max, and the same best candidate
   and the same population sizes in all 26 generations (0–25).

4. `out/figures/exp_21_fitness_evolution.png` written (104 kB): the best-in-population curve
   steps up to 0.8701 at generation 21, which is the `first_generation` the cache records for
   the winner.

5. `git status --short out/` lists only the six new `exp_21_*` files (candidates, ga_config,
   ga_stats, ga_summary, cnn_results_366_262_225, the figure) as untracked; nothing from
   experiments 1–20 is modified.

6. `SKIP_SYNC=1 ./init.sh` green, 62 tests passed. No source file changed: this ticket is a
   record, produced by code tickets 01–05 already verified.

Note for ticket 07: `out/tables/exp_21_ga_summary.json` names the winner, so
`train_final.py 21` without `--bands` has the file it reads.

## Comments

Author decision (15 Sep): the replay is experiment 21, not 20 (the plan's `exp_20` would overwrite the original experiment 20). The first new search Pablo launches is 22.
