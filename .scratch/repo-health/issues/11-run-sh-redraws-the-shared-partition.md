# 11 – `run.sh` redraws the shared partition, and nothing records which policy drew it

Status: needs-triage
Blocked by:

## Description

`run.sh` starts every chain with a bare `uv run python -u split_dataset.py`,
which rewrites `data/evaluation_partitions.csv` under whatever
`config.VALIDATION_BALANCE` says at that moment. `--validation-balance` is not
among the routed flags.

The comment above it is true as far as it goes — *"The split is deterministic
(config.PARTITION_SEED), so rewriting it costs nothing and guarantees every step
below reads the same partition"* — but it is only deterministic **given the
policy**, and the policy is now a variable.

Two ways that bites:

1. A run in flight under one policy, and a smoke chain under another. The
   documented smoke command `./run.sh 99 --population 4 --generations 1
   --epochs 1` silently redraws the shared partition. Experiment 22 then cannot
   resume: `CandidateCache.restore` refuses every row, because the data identity
   no longer derives the recorded candidate seeds. The refusal is loud; the
   clobbering that caused it was silent.
2. Nothing on disk says which policy drew the file that is there.
   `PARTITION_FIELDS` has no policy column, so `data/evaluation_partitions.csv`
   is the same shape either way and the only way to tell is to count crops per
   class.

README's "🚀 The phase-2 run" warns the reader not to run `run.sh` while a run
is in flight, and tells them to set the constant rather than pass the flag. Both
are stopgaps.

## Done when

- [ ] The partition file records the policy it was drawn under, so a step that
      reads it can tell.
- [ ] A chain, a search resume or a final model refuses a partition drawn under
      a policy other than the one it expects, instead of deriving other seeds
      from it.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

Encontrado por `/code-review high` al cerrar `protocolo-80-20/17`. Añadir una
columna a `evaluation_partitions.csv` cambia `PARTITION_FIELDS`, que los tests
de protocolo comparan contra `tests/data/reference_evaluation_partitions.csv`:
o se actualiza la referencia versionada — y entonces la comprobación «la
partición reproduce la de referencia» del ticket 17 compara otra cosa — o el
dato va a un sidecar aparte. Por eso va a triaje.
