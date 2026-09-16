# 08 – Bootstrap confidence interval by acquisition

Status: done
Blocked by: 07

## Description

`uv run python bootstrap.py N` reads the per-crop test predictions of experiment N, resamples whole acquisitions 5000 times with `default_rng(1729)`, and writes the 95 % percentile interval of the weighted F1 plus per-acquisition weighted F1 and accuracy, so the test result is always quoted with its width. No model is executed.

Copied almost literally from fig-aflatoxin's `scripts/bootstrap-test-interval.py`; defaults from `config.py`; output `out/tables/exp_NN_bootstrap_R_G_B.json`; the bands (indices and nm) come from the predictions/metrics files of experiment N, so no `--bands` flag is needed when only one final model exists for N.

## Done when

- [x] `uv run python bootstrap.py 21` prints point weighted F1 `0.722` and 95 % interval `[0.645, 0.848]` (3 decimals) and writes `out/tables/exp_21_bootstrap_366_262_225.json` with the interval, the number of resamples, the seed and the per-acquisition table.
- [x] Running it twice gives identical JSON.
- [x] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

**El intervalo del 10 de septiembre sale exacto, sin ejecutar ningún modelo.**
`uv run python bootstrap.py 21` (sin `--bands`: encuentra el único modelo final
del experimento 21 por sus `exp_21_final_metrics_*.json`) imprime

```
candidate      (366, 262, 225)
wavelengths_nm 891.02, 747.5, 697.05
test crops     256 in 8 acquisitions
weighted F1 0.722  95% CI [0.645, 0.848]  (5000 resamples of 8 acquisitions, seed 1729)
```

y escribe `out/tables/exp_21_bootstrap_366_262_225.json` con
`weighted_f1 0.722142952443074`, `weighted_f1_ci95 [0.6446439130060168,
0.847819167724381]`, `resamples 5000`, `seed 1729`, `resample_unit
"acquisition"`, `interval_method "percentile"`, `confidence_level 0.95` y la
tabla de las 8 capturas. El `macro_f1` que calcula sobre las predicciones,
`0.7223819971537002`, coincide dígito a dígito con el del
`exp_21_final_metrics_366_262_225.json` que escribió el ticket 07: las
predicciones dan la vuelta sin perder nada.

**Repetirlo es un replay.** Segunda corrida: `diff` vacío contra la primera
(el JSON no lleva marca de tiempo). Cambiar los ajustes se rechaza en vez de
sobrescribir: `bootstrap.py 21 --seed 99` → `exp_21_bootstrap_366_262_225.json
was recorded with a different seed; rewriting it would replace one result under
out/ with another` (salida 1). `bootstrap.py 22` → `exp_22 has no final model
under out/tables ... Run train_final.py 22 first` (salida 1).

**Las dos capturas flojas quedan a la vista** para el ticket 11: la tabla sale
peor-primero y las dos primeras son `042ef12d…` (C0, 40 recortes, weighted F1
0.689, exactitud 0.53) y `f7ab3a6e…` (C1, 40 recortes, 0.689, 0.53), frente a
0.86–0.95 en las otras seis. De ahí viene la anchura del intervalo.

**No abre ni un recorte ni un checkpoint.** El fixture `bootstrap_dataset`
apunta `HYPERCUBES_MANIFEST`, `SPECTRAL_AXES`, `PARTITIONS` y `MODELS_DIR` a
rutas que no existen, así que cualquier lectura de datos o de pesos haría
fallar los tests del comando completo.

**83 tests verdes** (71 antes), 12 nuevos:
`resamples_whole_acquisitions_and_never_a_single_crop`,
`repeats_exactly_under_the_same_seed_and_moves_under_another`,
`interval_is_the_percentile_pair_of_the_resampled_scores`,
`per_acquisition_table_names_every_acquisition_worst_first`,
`reads_the_only_final_model_of_the_experiment`,
`says_which_experiment_has_no_final_model`,
`asks_for_bands_when_the_experiment_has_two_final_models`,
`of_a_triplet_with_no_predictions_names_the_missing_file`,
`run_twice_writes_the_same_json`,
`refuses_a_rerun_that_would_not_reproduce_the_recorded_interval`,
`refuses_an_interval_whose_predictions_have_changed_underneath` y
`reproduces_the_recorded_interval_of_experiment_21` (este último remuestrea las
5 000 veces reales sobre el CSV versionado y exige `0.722142952443074` y
`(0.645, 0.848)` a tres decimales; cuesta ~8 s de los 17 s del suite).

`SKIP_SYNC=1 ./init.sh` verde de principio a fin, con `bootstrap.py` ya
compilado e importado en la puerta.

**Tras `/code-review`** (dos ejes, estándares y spec), cuatro cambios, y la
salida quedó **byte a byte idéntica** (`diff` vacío contra el JSON escrito
antes de tocar nada):

1. La guarda de `out/` vivía duplicada en `train_final.py` y en `bootstrap.py`
   con el mismo mensaje palabra por palabra. Se mueve a `ga.py`
   (`refuse_a_rerun_that_would_not_reproduce(path, settings, names)`, con
   `band_suffix` al lado) y los dos pasos la comparten: cada uno decide qué
   cuenta como «la misma corrida», la regla vive en un sitio.
2. La guarda no miraba las predicciones: con las mismas bandas, semilla y
   remuestreos, un `test_predictions.csv` distinto habría sobrescrito el
   intervalo registrado en silencio. Ahora `weighted_f1` forma parte de
   `RECORDED_SETTINGS`, y el test nuevo lo fija.
3. `(actual, predicted, acquisition)` viajaban juntos por cinco funciones y
   `write_bootstrap` tenía diez parámetros. Nace `TestPredictions` (con
   `within()` para un remuestreo o una captura) y `FinalResult`; el escritor
   baja a cinco parámetros y ya no calcula métricas.
4. El comentario que justificaba `labels=` era falso: en weighted F1 una clase
   ausente pesa cero, así que `labels=` no cambia el número (comprobado:
   `0.6666666666666666` con y sin). Sí cambia macro (`0.667` → `0.333` en un
   caso de dos clases). El comentario dice ahora la razón real —consistencia
   con `cnn.model.classification_metrics` y silenciar zero-division— y
   `BOOTSTRAP_CONFIDENCE_LEVEL` / `BOOTSTRAP_PERCENTILES` pasan a `config.py`,
   donde CLAUDE.md dice que viven las constantes del experimento.

## Comments

