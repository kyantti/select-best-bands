# Session Progress Log

## Current State

**Last Updated:** 2026-09-16 (sesión 6)
**Branch:** feature/protocolo-80-20 (creada desde `feature/experiments`@9075d9b, que contiene b7f1fc9)
**Active Feature:** `protocolo-80-20` – 17 tickets (`.scratch/protocolo-80-20/issues/01..17`): fase 1 (01–10, el porte verificable bit a bit) y fase 2 (11–17, las mejoras de la corrida siguiente, apagadas por defecto). **01–07 `done`**; ninguno `in-progress`. Next pick: protocolo-80-20/08 – Intervalo bootstrap por captura (`ready-for-agent`, su bloqueo 07 está `done`); lee `out/tables/exp_21_test_predictions_366_262_225.csv`, que ya está escrito.

**No empieces por repo-health/02.** La feature `protocolo-80-20` reescribe `ga.py` entero, así que 02 (restaurar el import de `engine2`), 03 (overrides por entorno) y 04 (tests de los operadores viejos) quedan sin objeto. Están pendientes de triaje por el autor. `repo-health/05` sigue siendo válido e independiente; `06` lo absorbe el README nuevo.

## Status

### What's Done

- [x] Experiments 1–20 completed; results tracked in `out/tables`, `out/figures`, `out/logs` (see `run.log`).
- [x] Harness created: `CLAUDE.md`, `init.sh`, `progress.md`, `session-handoff.md`.
- [x] repo-health/01: `uv sync` populated `.venv`; compile, manifest (897 train / 226 test rows, 0 missing) and CUDA (4 GPUs visible) checks pass.
- [x] Matt Pocock skills configured: `docs/agents/{issue-tracker,triage-labels,domain}.md` and an `## Agent skills` section in `CLAUDE.md`. Issue tracker is local markdown under `.scratch/`.

- [x] `protocolo-80-20`: spec publicado en `.scratch/protocolo-80-20/spec.md` (`ready-for-agent`). Plan de implementación detallado, con los valores de referencia para la verificación bit a bit, en `~/.claude/plans/quiero-reorganizar-el-proyecto-parsed-llama.md`.
- [x] **protocolo-80-20/01** (15 Sep): rama `feature/protocolo-80-20`; pines exactos en `pyproject.toml` con índice uv cu126 (torch 2.13.0+cu126, torchvision 0.28.0, deap 1.4.4, numpy 2.5.1, sklearn 1.9.0, pandas 2.3.3, matplotlib 3.11.1, seaborn 0.13.2, pytest 9.1.1) y `.venv` resincronizado a python 3.13.5; `.python-version` versionado; `/data/` + `out/models/` en `.gitignore` (así `tests/data/*.csv` quedan versionados); borrados `cnn/train.py`, `cnn/model_info.py`, `cnn/util/`, `train_dataset.csv`, `test_dataset.csv`, `run.log`; `config.py` con todas las constantes de la corrida del 10 Sep; `init.sh` y `CLAUDE.md` actualizados. `SKIP_SYNC=1 ./init.sh` verde.

### What's In Progress

- [ ] `protocolo-80-20` fase 1: 01–07 `done`; tickets 08–10 en `ready-for-agent`. Cadena: 08 (bootstrap) → 09; 10 (`check_data.py`) solo depende de 03.
- [ ] `protocolo-80-20` fase 2: tickets 11–17 en `ready-for-agent`, todos detrás de la fase 1. 11 (capturas flojas del test, 0 h) depende de 08 y va primero; 12 (validación equilibrada por recortes), 13 (punto de inyección de checkpoint, delta 0.00e+00) y 16 (`FINAL_SEEDS`) dependen de 09; 14 (`pretrain.py` SimCLR) de 13; 15 (puerta pareada, ~4,5 h) de 12 y 14; 17 (re-verificar los valores por defecto y entregar el comando de la corrida) de 12, 13, 14 y 16. Cada constante de fase 2 tiene por defecto el comportamiento del 10 Sep.

### What's Next

1. protocolo-80-20/08 (intervalo bootstrap por captura, `bootstrap.py`). No ejecuta ningún modelo: remuestrea 5 000 veces las 8 capturas de `out/tables/exp_21_test_predictions_366_262_225.csv` con `default_rng(1729)`. Objetivo: F1 ponderado 0,722 con intervalo 95 % `[0.645, 0.848]`.
2. Triar `repo-health/02,03,04,06` (probablemente `wontfix`: la feature nueva los deja sin objeto; 06 lo absorbe protocolo-80-20/09).
3. Seguir la cadena 08 → 09; el replay (06) y el modelo final (07) ya ocupan el número 21, la primera corrida real es la 22.

## Blockers / Risks

- [ ] `check_data.py` sigue sin importar tras el ticket 03 (`HypercubeDataset` desapareció); lo reescribe el ticket 10. `init.sh` lo compila pero no lo importa, así que la puerta sigue verde. `ga.py` ya está entero y `init.sh` vuelve a importarlo.
- [ ] `data/` is gitignored; `init.sh` warns instead of failing when it is absent so code-only work is possible on machines without the dataset.
- [ ] A default `ga.py` run is 50 generations × up to 25 individuals × up to 50 epochs on an A100. Never launch one as "verification".

## Decisions Made

- **Instruction file is `CLAUDE.md`, not `AGENTS.md`**: the author uses Claude Code.
- **Manifest check warns rather than fails when `data/` is missing**: agents may work on the code on machines without the dataset.
- **Issue tracker is Matt Pocock's local markdown, not `feature_list.json`** (2026-09-15): tickets live in `.scratch/repo-health/issues/NN-*.md` with a `Status:` line and a `Blocked by:` line. `feature_list.json` was migrated (feat-00N → repo-health/0N) and deleted.
- **One status vocabulary for harness and skills**: `Status:` uses the five Matt Pocock triage roles plus `in-progress` and `done`. There is no `blocked` status (use `Blocked by:` or `needs-info`). See `docs/agents/triage-labels.md`.
- **`out/` is immutable history**: every experiment number owns its files; new runs take a new number, throwaway smoke runs use 90–99 and are deleted afterwards.
- **Port del protocolo fiable a este repo** (2026-09-15): la estructura plana de este repo se queda; el comportamiento del protocolo 80/20 de `~/Documents/fig-aflatoxin` se copia literal (reparto por captura, validación dentro del train, weighted F1 como fitness, normalización train-foreground, test leído una vez). Sin MLflow, huellas, pydantic ni DAG. Entrada: los 1.124 NPZ ya recortados, enlazados en `data/`. Ver `.scratch/protocolo-80-20/spec.md`.
- **Se conserva la derivación exacta de la semilla por candidato**: permite verificar el port reproduciendo bit a bit el resultado del 10 Sep y replicar la búsqueda sin GPU desde los 383 candidatos ya evaluados.

## Files Modified This Session

- `CLAUDE.md`, `init.sh`, `progress.md`, `session-handoff.md` – created (harness only, no source changes).
- `docs/agents/issue-tracker.md`, `docs/agents/triage-labels.md`, `docs/agents/domain.md` – created by `/setup-matt-pocock-skills`.
- `.scratch/repo-health/{spec.md,issues/01..06-*.md}` – created from the former `feature_list.json`, which was then deleted.
- `CLAUDE.md` (Startup, Layout, Working rules, End of session, Agent skills), `init.sh` (final hint) – point at `.scratch/`.
- protocolo-80-20/07: `train_final.py` (nuevo: `FinalTraining`/`FinalEvaluation`/`FinalModel`, `train_final_model`, `predict_test`, `load_crops`, `winner_bands`, `final_paths`, las dos guardas de `out/`, las salidas y las dos figuras), `tests/test_protocol.py` (+9 tests), `ga.py` (`write_predictions` acepta `exclusive=`, y es el único sitio donde vive el esquema de predicciones), `init.sh` (compila e importa `train_final`), `CLAUDE.md` (Layout). Nuevos en `out/`: `tables/exp_21_{final_metrics,confusion_matrix,test_predictions,training_history}_366_262_225.*` y `figures/exp_21_{confusion_matrix,training_history}_366_262_225.png`; `out/models/exp_21_model_366_262_225.pt` (91 MB, gitignored).
- protocolo-80-20/06: sin cambios de código. Nuevos en `out/`: `tables/exp_21_{candidates.csv,ga_config.json,ga_stats.csv,ga_summary.json,cnn_results_366_262_225.csv}` y `figures/exp_21_fitness_evolution.png`; `.scratch/protocolo-80-20/issues/06-*.md` (`done` + evidencia).
- protocolo-80-20/05: `ga.py` (operadores genéticos, `CandidateFitness`, `CandidateCache`, `run_search`, `fitness_contract`, `make_evaluator`, salidas, figura y CLI de búsqueda), `tests/test_protocol.py` (+12 tests), `init.sh` (vuelve `import ga`).
- protocolo-80-20/04: `cnn/model.py` (nuevo: identidad de datos, semilla por candidato, `build_resnet50`, métricas, `evaluate_candidate`, `predict`), `ga.py` (reescrito: CLI `--evaluate R G B` + `load_selection_data()`), `tests/test_protocol.py` (+9 tests), `init.sh` (importa `cnn.model`), `out/tables/evaluate_366_262_225_validation_predictions.csv` (nuevo, artefacto de verificación fuera del espacio `exp_NN_`).
- protocolo-80-20/03: `cnn/data_setup.py` (reescrito entero: eje espectral, `Hypercube`, `load_hypercubes`, normalización, `prepare_model_input`, `SelectedBandDataset`; conserva partición y esquemas del 02, con `load_cropped_hypercubes` renombrado a `load_manifest`), `cnn/engine.py` (copia literal), `tests/test_protocol.py`, `tests/data/reference_spectral_axes.csv` (nuevo, versionado).
- protocolo-80-20/02: `split_dataset.py` y `tests/test_protocol.py` (nuevos), `cnn/data_setup.py` (esquemas + `load_partitions`), `pyproject.toml` (`[tool.pytest.ini_options]`), `init.sh` (compila e importa `split_dataset`), `CLAUDE.md` (Layout).
- protocolo-80-20/01: `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `config.py` (nuevo), `init.sh`, `CLAUDE.md` (Layout, Invariants, Verification); borrados `cnn/train.py`, `cnn/model_info.py`, `cnn/util/helper_functions.py`, `train_dataset.csv`, `test_dataset.csv`, `run.log`; `tests/data/*.csv` pasan a estar versionados.

## Evidence of Completion

- [x] **protocolo-80-20/07** (16 Sep): el modelo final reproduce **bit a bit** el test del 10 Sep. `train_final.py 21 --bands 366 262 225`: weighted F1 `0.722142952443074`, macro `0.7223819971537002`, MAE `0.5`, QWK `0.5815011372251706`, matriz `[[50,3,6,13],[6,43,9,10],[4,7,46,3],[4,3,3,46]]`, mean y std congelados exactos. `diff` vacío contra `runs/train_final_model/bac0f8e3.../` en **tres** ficheros: predicciones de test (256 filas, las cuatro columnas), historia de entrenamiento (las 50 épocas) y matriz de confusión. El `.pt` está en disco **69,7 ms antes** de la primera línea que menciona el test, y el test `test_final_saves_the_model_before_the_test_set_is_opened` falla si se invierte el orden. Sin `--bands` saca el ganador del `exp_21_ga_summary.json` y escribe los mismos bytes (cuatro corridas, checksums idénticos). Tras `/code-review`: se añadió `refuse_a_rerun_that_would_not_reproduce`, que rechaza en menos de un segundo un `--epochs 1` sobre el resultado de 50 épocas. 71 tests verdes.

- [x] **protocolo-80-20/06** (16 Sep): **el replay de la corrida del 10 Sep es exacto**. Cache caliente de 383 filas construida desde `candidate_diagnostics.csv` con la semilla por candidato recalculada aquí (la fila ganadora sale `366,262,225,3104252108,0.8700979843225085,...`, la misma que verificó el ticket 04). `ga.py 21 --no-evaluate`: `restored 383`, `evaluated 0`, `unique 383 | hits 451` (los `383 unique_evaluations` + `68 hits` del `ga_study_report.json`), ganador `(366, 262, 225)` a 891.02 / 747.5 / 697.05 nm con weighted F1 `0.8700979843225085`. `exp_21_ga_stats.csv` vs `generation_history.csv`: **26 generaciones, 0 discrepancias** con igualdad exacta de floats en avg / std / min / max y el mismo mejor candidato. Figura escrita; `git status --short out/` solo lista los seis `exp_21_*` nuevos. 62 tests verdes.

- [x] **protocolo-80-20/05** (16 Sep): la búsqueda corre y **la reanudación es un replay**. Smoke `ga.py 99 --population 4 --generations 2 --epochs 1`: seis salidas, 8 candidatos únicos, ganador `(391, 201, 105)`. Truncando el CSV a 4 filas y relanzando: `restored 4`, entrena solo los 4 que faltaban y `exp_99_ga_stats.csv` sale **idéntico** al de la corrida sin interrumpir (y `exp_99_candidates.csv` idéntico salvo `seconds`). `--no-evaluate` replica la búsqueda entera con 0 evaluaciones. `--epochs 2` aborta nombrando el contrato de fitness. 61 tests verdes (51 antes); `exp_99_*` borrado.

- [x] **protocolo-80-20/04** (15 Sep): el port reproduce **bit a bit** la corrida del 10 Sep. `ga.py --evaluate 366 262 225` imprime semilla de candidato `3104252108`, mean y std congelados y weighted F1 `0.8700979843225085`, macro `0.8763610130071496`, MAE `0.1875`, QWK `0.8711063372717508`; las 160 predicciones de validación son **byte a byte** el `validation_predictions.csv` de la corrida `d49fc1f9...` (las cuatro columnas, no solo las tres pedidas). Los cuatro checksums de identidad de datos coinciden con su `success.json`. 51 tests verdes (42 antes). 39,2 s de entrenamiento en la A100.

- [x] **protocolo-80-20/03** (15 Sep): `cnn/engine.py` es byte a byte el de fig-aflatoxin (sin early stopping) y el camino de datos reproduce la normalización de la corrida real: con los 708 recortes de ajuste y las bandas 366/262/225, mean `[0.6560380893489065, 0.6596170752860457, 0.2233070946048868]` y std `[0.11694627746639438, 0.12472051938804612, 0.07551858819936108]`, idénticos a los registrados. 42 tests verdes.

- [x] **protocolo-80-20/02** (15 Sep): `uv run python split_dataset.py` escribe `data/evaluation_partitions.csv` **byte a byte idéntico** a `tests/data/reference_evaluation_partitions.csv` (22 / 6 / 8 capturas, 708 / 160 / 256 recortes). 18 tests en `tests/test_protocol.py`, todos verdes; `init.sh` ya corre pytest.

- [x] `SKIP_SYNC=1 ./init.sh` verde de principio a fin (2026-09-15, tras protocolo-80-20/01). Avisa (no falla) de que falta `data/evaluation_partitions.csv`, que escribe el ticket 02, y salta pytest porque aún no hay `tests/test_*.py`.

## Notes for Next Session

- **`out/` tiene dos guardas, no una** (ticket 07): `train_final.py` rechaza un número que pertenece a un protocolo anterior (los experimentos 1–20, que tienen `classification_report` y `ga_stats` pero ni cache ni summary) **y** rechaza reescribir su propio resultado con otros ajustes (bandas, semilla, épocas, batch, lr, tamaño de imagen, dispositivo) comparándolos con el `exp_NN_final_metrics_*.json` ya escrito. Repetir la misma corrida sí está permitido: escribe los mismos bytes y es como se vuelve a verificar el registro.

- **`out/` tiene una regla de propiedad nueva**: `ga.py N` se niega a arrancar si existe `exp_NN_ga_stats.csv` pero no `exp_NN_candidates.csv`, porque ese número es de una corrida que este código no puede reanudar (los experimentos 1–20). Con los dos ficheros es una reanudación y reescribe sus propias salidas finales a propósito.

- **Cargar los recortes cuesta ~2,5 min** (708 en 123 s, 868 en ~150 s): es la descompresión zlib de los NPZ en `load_hypercubes`, es CPU, y fig-aflatoxin paga lo mismo. Por eso el smoke de una época del ticket 04 tarda ~3 min y no «menos de un minuto». Para el ticket 05 da igual: la búsqueda carga una vez y deja los recortes en RAM.
- **`check_data.py` sigue sin importar** tras el ticket 03 (`HypercubeDataset` desapareció); lo reescribe el ticket 10. `init.sh` lo compila pero no lo importa, así que la puerta sigue verde.

- `cnn/train.py`, `cnn/model_info.py` y `cnn/util/` ya están borrados (protocolo-80-20/01); su contenido sigue en la historia, en `b7f1fc9`.
- `create_summary_table.py` only reads experiments 1–10 (repo-health/05).
- **El nm redondeado del ticket 03 era la prosa de la memoria**: los artefactos de la corrida guardan `891.02` y `697.05`. `wavelengths_of` devuelve el valor crudo; redondear, si acaso, al imprimir.
- **Cabos sueltos que dejan los borrados del ticket 01** (cada uno tiene ya su ticket, no se tocan ahora): `check_data.py:116` sigue teniendo `default="train_dataset.csv"`, que ya no existe → ticket 10 lo reescribe; el README describe `helper_functions.py`, `train_dataset.csv` y `test_dataset.csv` → ticket 09; `cnn/data_setup.py:11` define su propio `NUM_WORKERS = os.cpu_count()`, que compite con `config.NUM_WORKERS = 8` (que sí forma parte del resultado reproducible) → ticket 03 sustituye el módulo entero. `init.sh` compila `check_data.py` y `ga.py` pero no los importa, así que la puerta sigue verde con ellos rotos.
- **Las tres trampas del spec están resueltas por el ticket 01**: `/data/` en el `.gitignore`, `.venv` con torch 2.13.0+cu126 y python 3.13.5, e `init.sh` sin la comprobación de `train_dataset.csv`/`test_dataset.csv`.
- `data/` (symlink `cropped_hypercubes` a los 1.124 NPZ, `cropped_hypercubes.csv`, `spectral_axes.csv`) y `out/models/` están en el árbol y siguen ignorados a propósito; `tests/data/*.csv` ya no lo están.
- Hubo una rama `feature/protocolo-80-20` creada la madrugada del 15 Sep y borrada esa misma mañana; sus dos commits (`f1e4759`, `3a19e48`, solo borrados de ficheros obsoletos) cuelgan en el reflog y no hacen falta. La rama se vuelve a crear desde `feature/experiments`.
