# 09 – The final-model guard can be walked past with NUM_WORKERS

Status: needs-triage
Blocked by:

## Description

`train_final.RECORDED_SETTINGS` — what
`ga.refuse_a_rerun_that_would_not_reproduce` compares to decide whether a rerun
is a replay — lists the bands, the seed, the epochs, the batch size, the
learning rate, the image size, the device type, the backbone checkpoint and the
test-read count. It does **not** list `num_workers`, and nothing in
`exp_NN_final_metrics_*.json` records the partition either.

`config.py` says the worker count is part of the result, in as many words:

> 8 persistent workers are part of the reproducible result: the DataLoader
> generator is split across them, so another count gives another augmentation
> stream and another fitness.

`ga.fitness_contract` records `num_workers`, `partition_seed`,
`test_proportion` and `validation_proportion`; `pretrain.py`'s sidecar records
`num_workers` and `partition_seed`. `train_final.py` records none of them.

So: set `NUM_WORKERS = 4` (or move `PARTITION_SEED`, or `TEST_PROPORTION`) and
re-run `train_final.py 21 --bands 366 262 225`. The guard sees identical
settings, says nothing, and **overwrites** `exp_21_final_metrics_366_262_225.json`,
the confusion matrix, the predictions, the history and both figures with a
different number. That is exactly what the `out/` invariant exists to prevent,
and it is the one result the thesis quotes.

`bootstrap.py` closes the same hole from the other side by putting
`weighted_f1` itself into its `RECORDED_SETTINGS`, so a predictions file that
changed underneath a recorded interval shows up as a refusal.
`train_final.py` has no equivalent.

## Done when

- [ ] Changing `NUM_WORKERS` (or any partition constant) and re-running
      `train_final.py 21 --bands 366 262 225` is refused, not silently written.
- [ ] A replay under unchanged settings still rewrites the same bytes, so
      `train_final.py 21 --bands 366 262 225` stays the re-verification command
      of `protocolo-80-20/17`.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

**Va a triaje porque la salida tiene un coste.** Añadir `num_workers` y la
identidad de la partición al fichero de métricas cambia el esquema de un
artefacto que ya está en el registro: `exp_21_final_metrics_366_262_225.json`
commiteado no tiene esos campos, así que la primera escritura después del cambio
saldría distinta del fichero que hay. Lo mismo que `repo-health/08`. Salidas
posibles:

1. Añadirlos y reescribir `exp_21_final_metrics_366_262_225.json` una vez, en el
   mismo commit, explicando por qué.
2. Compararlos sin escribirlos: el guard puede leer `ga.library_versions()` y
   los constantes en vivo y tratar la **ausencia** del campo como «no
   comprobable», que es como `backbone_checkpoint` ya se comporta.

Encontrado por `/code-review high` al cerrar `protocolo-80-20/17`, que es
documentación y no podía tocar `train_final.py`.
