# 09 – `run.sh` chain, README, fitness plot, `init.sh` and `CLAUDE.md`

Status: done
Blocked by: 08

## Description

One command reproduces the whole chain for an experiment number, and the documentation and harness describe the new protocol instead of the old one.

- `run.sh N` (N required; 21 is the replay, so the first real run is 22) with `set -euo pipefail`, `CUDA_VISIBLE_DEVICES` defaulting to GPU 1, runs split → `ga.py N` → `train_final.py N` → `bootstrap.py N`, logging to `out/logs/experiment_N.log`, `final_N.log`, `bootstrap_N.log`. Extra arguments are passed through to `ga.py` and `train_final.py` so a smoke chain is possible.
- `plot_fitness_evolution.py` takes N as argument and reads the new `ga_stats` columns.
- `create_summary_table.py` stays as the v1 tool (out of scope to adapt beyond not breaking).
- README: the protocol in plain words (acquisition-grouped 80/20 split, validation 80/20 inside train, train-foreground normalization, validation weighted F1 as fitness, final model with one test read, bootstrap by acquisition), how to launch, resume and verify, the v1/v2 boundary at experiment 21, the results table with experiment 21, and the exact command for the real run (`nohup ./run.sh 22 > out/logs/run_22.log 2>&1 &`, `tail -f`, `wc -l` on the candidates CSV). Absorbs repo-health/06.
- `init.sh`: imports `ga`, `cnn.data_setup`, `cnn.engine`, `cnn.model`; checks `data/cropped_hypercubes.csv`, `data/spectral_axes.csv`, `data/evaluation_partitions.csv` (warn, not fail, when `data/` is absent); runs pytest; final hint unchanged.
- `CLAUDE.md` Layout, Invariants and Verification name the four scripts, `config.py`, the smoke flags and the 21 boundary. `progress.md` reflects the new state.

## Done when

- [x] `CUDA_VISIBLE_DEVICES=1 ./run.sh 99 --population 4 --generations 1 --epochs 1` completes all four steps and leaves `exp_99_ga_stats.csv`, `exp_99_candidates.csv`, `exp_99_final_metrics_*.json`, `exp_99_bootstrap_*.json` and the three figures; then `rm out/*/exp_99_*` and `out/models/exp_99_*`.
- [x] `uv run python plot_fitness_evolution.py 21` regenerates `out/figures/exp_21_fitness_evolution.png`.
- [x] `./init.sh` (full, with sync) is green and prints the three manifest checks and the pytest summary.
- [x] README no longer mentions `train_dataset.csv`, `ELITISM_SIZE`, `engine2` or accuracy as the fitness, and states the 21 boundary.
- [x] `grep -n "config.py\|train_final\|bootstrap.py\|split_dataset" CLAUDE.md` finds all four.

## Evidence

**16 Sep 2026. La cadena entera corre con un solo comando y la figura se puede
redibujar sin GPU.**

- `CUDA_VISIBLE_DEVICES=1 ./run.sh 99 --population 4 --generations 1 --epochs 1`
  completó los cuatro pasos en ~8 min (11:22:21 split → 11:22:27 `ga.py` →
  11:26:16 `train_final.py` → 11:30:17 `bootstrap.py`, salida 0). Dejó los diez
  `out/tables/exp_99_*` (incluidos `ga_stats.csv`, `candidates.csv`,
  `final_metrics_391_201_105.json` y `bootstrap_391_201_105.json`) y las **tres**
  figuras (`fitness_evolution`, `confusion_matrix`, `training_history`), más
  `out/models/exp_99_model_391_201_105.pt` y los tres logs
  (`experiment_99.log`, `final_99.log`, `bootstrap_99.log`). Ganador del smoke
  `(391, 201, 105)`, test weighted F1 0.416 IC [0.351, 0.571]. Todo borrado
  después: `git status --short out/` vacío.
- El paso `split_dataset.py` de la cadena reescribe `data/evaluation_partitions.csv`
  y el fichero sigue siendo **idéntico** a `tests/data/reference_evaluation_partitions.csv`
  (`diff` vacío): reejecutar la cadena no mueve la partición.
- `uv run python plot_fitness_evolution.py 21` regenera
  `out/figures/exp_21_fitness_evolution.png` **byte a byte** igual a la que
  escribió `ga.py` en el ticket 06 (`cmp` vacío, `git status --short out/`
  limpio). Lee `exp_21_ga_stats.csv` (26 generaciones) y toma el título del
  ganador de `exp_21_ga_summary.json`; rechaza en una línea las columnas de los
  experimentos 1–20 y nombra el fichero que falta.
- `./init.sh` completo (con `uv sync`) verde: imports OK, los tres manifiestos
  `ok`, torch 2.13.0+cu126 con 4 GPUs y **88 tests** (83 antes; 5 nuevos para el
  redibujo).
- `grep -n "train_dataset.csv\|ELITISM_SIZE\|engine2\|sanity-check" README.md`
  no encuentra nada; el README dice que el fitness es el weighted F1 de
  validación y marca la frontera v1/v2 en el 21.
- `grep -n "config.py\|train_final\|bootstrap.py\|split_dataset" CLAUDE.md`
  encuentra los cuatro.
- `run.sh` se niega en una línea (salida 1) sin número, con `022` (cero a la
  izquierda: `printf` lo leería en octal), con un flag que ningún paso acepta y
  con `--bands` incompleto.

Tras `/code-review` (dos ejes, en paralelo): corregidos dos errores de hecho del
README (las dos capturas flojas son weighted F1 0.689 / exactitud 0.53 frente a
0.857–0.951 / 0.75–0.91 en las otras seis — antes comparaba F1 contra exactitud;
y «26 generaciones» es la población inicial más las 25 de `config.GENERATIONS`),
el `rm` de la limpieza pasa a `rm -f`, `read_generations` deriva sus columnas de
`ga.STATS_FIELDS` en vez de repetirlas, y `--bands` incompleto ya no revienta con
un «unbound variable» de bash.

Fuera del alcance literal del ticket, con motivo: `ga.py` expone `STATS_FIELDS`
como constante de módulo (el esquema de la curva vivía dentro de `write_stats`,
y ahora lo leen los dos lados) e `init.sh` importa también
`plot_fitness_evolution`.

## Comments

