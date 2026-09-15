# 06 – README matches the code

Status: ready-for-agent
Blocked by: 02

## Description

README shows POPULATION_SIZE=20, MUTATION_PROB=0.15, ELITISM_SIZE=1 but ga.py uses 25, 0.05, 2; it says `uv run ga.py` runs 5 experiments but ga.py requires an experiment number; it links sanity-check/data_sanity_check.png which was deleted in 7357da7. Fix those, and document run.sh and the experiment-number ownership of out/ files. Also: README says 898 training samples, the manifest has 897 rows.

## Done when

_(see Description)_

## Evidence

_(none yet)_

## Comments

