# 03 – Smoke-run configuration

Status: ready-for-agent
Blocked by: 02

## Description

Allow overriding GENERATIONS, POPULATION_SIZE and NUM_EPOCHS from the environment (e.g. GA_GENERATIONS=1 GA_POP=3 GA_EPOCHS=1) without changing the defaults, so an end-to-end run finishes in minutes.

## Done when

`GA_GENERATIONS=1 GA_POP=3 GA_EPOCHS=1 uv run python ga.py 99` writes out/tables/exp_99_ga_stats.csv and exp_99_cnn_results_*.csv, and those files are then deleted.

## Evidence

_(none yet)_

## Comments

