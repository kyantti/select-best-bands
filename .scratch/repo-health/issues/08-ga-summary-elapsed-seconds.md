# 08 – The GA summary carries a stopwatch, so a replay is not byte-for-byte

Status: needs-triage
Blocked by:

## Description

`out/tables/exp_NN_ga_summary.json` records `elapsed_seconds`, which is wall
clock. Every other field of every artifact this protocol writes is a function of
the inputs, so re-running a step rewrites the same bytes — that is how a number
quoted in the thesis gets re-verified, and `train_final.write_metrics` says so
in as many words:

> It carries no timestamp and no elapsed time on purpose: two runs of the same
> experiment number and triplet write the same bytes, so a rerun that changed
> something would show up as a diff.

`ga.py` does not follow that rule. Replaying the finished search of experiment
21 from its warm cache (`uv run python ga.py 21 --no-evaluate`, ticket
`protocolo-80-20/17`, check 5 of 5) left `git status` dirty on one field and
nothing else:

```
-  "elapsed_seconds": 0.01089058630168438,
+  "elapsed_seconds": 0.017415492795407772,
```

Every other file the replay touched — `exp_21_ga_stats.csv`,
`exp_21_candidates.csv`, `exp_21_cnn_results_366_262_225.csv` and
`exp_21_fitness_evolution.png` — was byte-identical, and the winner and the
per-generation history were unchanged. So the search *is* a replay; the summary
just can't prove it.

The cost is small but real: anyone who re-verifies the search gets a dirty
working tree over a stopwatch reading, and may commit it, which edits the thesis
record with a number that means nothing.

## Done when

- [ ] Two runs of `ga.py NN --no-evaluate` over the same warm cache write
      byte-identical `exp_NN_ga_summary.json`.
- [ ] Whatever the elapsed time is worth keeping for is still available — the
      search already prints it per candidate and logs to `out/logs/`.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

**Va a triaje porque la decisión no es del agente.** Quitar `elapsed_seconds`
del resumen cambia el esquema de un artefacto que ya está en el registro: el
`exp_21_ga_summary.json` commiteado tiene el campo, así que la primera escritura
después del cambio saldría distinta del fichero que hay, que es exactamente lo
que el invariante de `out/` prohíbe. Las salidas que se me ocurren:

1. Dejarlo como está y documentar que el resumen del GA es el único artefacto
   que no replica byte a byte. Coste: cero. El que re-verifique tiene que saber
   revertirlo a mano.
2. Quitar el campo y reescribir `exp_21_ga_summary.json` una vez, como parte del
   mismo commit que lo quita, dejando dicho en el mensaje por qué.
3. Mantenerlo pero fuera del artefacto — a `out/logs/` o a un sidecar que no
   forme parte del registro.

Encontrado al cerrar `protocolo-80-20/17`, que no podía tocar `ga.py`: ese
ticket re-verifica los valores por defecto, no los cambia.
