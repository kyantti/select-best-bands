# repo-health

**Goal:** make `select-best-bands` runnable and restartable for coding agents: `./init.sh` green,
`ga.py` importable, a minutes-long smoke configuration, GPU-free tests, and analysis scripts and
README that match the code and the 20 completed experiments.

**Out of scope:** new GA experiments, hyperparameter changes, anything under `out/` (immutable thesis record).

## Tickets

| # | Ticket | Blocked by |
|---|---|---|
| 01 | Environment bootstrap | – |
| 02 | Restore ga.py imports after engine2 removal | 01 |
| 03 | Smoke-run configuration | 02 |
| 04 | Pytest smoke tests | 01 |
| 05 | Analysis scripts cover all experiments | 01 |
| 06 | README matches the code | 02 |

Status of each ticket lives in its own file under `issues/`. Closed tickets stay in place.
