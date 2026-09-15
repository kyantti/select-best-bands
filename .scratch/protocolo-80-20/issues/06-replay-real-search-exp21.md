# 06 – Replay the 10 Sep search without GPU as experiment 21

Status: ready-for-agent
Blocked by: 05

## Description

Experiment 21 is the port of the real run of 10 Sep 2026. Its candidate cache is built from the 383 candidates that run evaluated, and `ga.py 21 --no-evaluate` walks the exact same trajectory: same per-generation history, same winner, zero training. This proves the operators, seeding, random-state isolation, cache and tie-breaking are ported unchanged, and it leaves `exp_21_*` as the record that tickets 07 and 08 complete.

Source: `runs/run_ga_study/8b6e000fa03e3b1cb91dc72d3f11614933fb752e88ba45dfe4d4ff9639524cd0/{candidate_diagnostics.csv,generation_history.csv,ga_study_report.json}` in `~/Documents/fig-aflatoxin`. A short conversion script (kept in the scratchpad, not the repo) writes `out/tables/exp_21_candidates.csv` and its `exp_21_ga_config.json` sidecar in the ticket-05 format, with the candidate seed of each row recomputed from the real data identity so the cache loader accepts it.

## Done when

- [ ] `out/tables/exp_21_candidates.csv` has 383 rows and `uv run python ga.py 21 --no-evaluate` loads it without a contract mismatch.
- [ ] The run ends with 383 restored, 0 evaluations, winner 366/262/225 (891.0 / 747.5 / 697.1 nm), validation weighted F1 `0.8700979843225085`.
- [ ] `out/tables/exp_21_ga_stats.csv` matches `generation_history.csv` of the real run generation by generation on avg / std / min / max / best candidate (compare with a one-liner; record the command).
- [ ] `out/figures/exp_21_fitness_evolution.png` exists.
- [ ] No file under `out/` from experiments 1–20 changed (`git status --short out/` shows only `exp_21_*`).

## Evidence

_(none yet)_

## Comments

Author decision (15 Sep): the replay is experiment 21, not 20 (the plan's `exp_20` would overwrite the original experiment 20). The first new search Pablo launches is 22.
