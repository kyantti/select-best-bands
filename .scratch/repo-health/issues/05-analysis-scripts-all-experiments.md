# 05 – Analysis scripts cover all experiments

Status: ready-for-agent
Blocked by: 01

## Description

create_summary_table.py hard-codes experiments 1..10 and plot_fitness_evolution.py hard-codes exp_10 while out/tables now holds experiments 1..20. Discover experiment numbers from the files in out/tables instead, and take the experiment for the fitness plot as a CLI argument.

## Done when

`uv run python create_summary_table.py` lists 20 rows in out/tables/summary_results.csv.

## Evidence

_(none yet)_

## Comments

