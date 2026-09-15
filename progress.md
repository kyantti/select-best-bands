# Session Progress Log

## Current State

**Last Updated:** 2026-09-15 (noche, 2)
**Branch:** feature/protocolo-80-20 (creada desde `feature/experiments`@9075d9b, que contiene b7f1fc9)
**Active Feature:** `protocolo-80-20` – 17 tickets (`.scratch/protocolo-80-20/issues/01..17`): fase 1 (01–10, el porte verificable bit a bit) y fase 2 (11–17, las mejoras de la corrida siguiente, apagadas por defecto). **01 y 02 `done`**; ninguno `in-progress`. Next pick: protocolo-80-20/03 – Hypercube loading, train-only normalization, augmentation and training engine (`ready-for-agent`, su bloqueo 02 está `done`).

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

- [ ] `protocolo-80-20` fase 1: 01 y 02 `done`; tickets 03–10 en `ready-for-agent`. Cadena: 03 → 04 → 05 → 06 (replay de la corrida del 10 Sep como exp_21) → 07 → 08 → 09; 10 (`check_data.py`) solo depende de 03.
- [ ] `protocolo-80-20` fase 2: tickets 11–17 en `ready-for-agent`, todos detrás de la fase 1. 11 (capturas flojas del test, 0 h) depende de 08 y va primero; 12 (validación equilibrada por recortes), 13 (punto de inyección de checkpoint, delta 0.00e+00) y 16 (`FINAL_SEEDS`) dependen de 09; 14 (`pretrain.py` SimCLR) de 13; 15 (puerta pareada, ~4,5 h) de 12 y 14; 17 (re-verificar los valores por defecto y entregar el comando de la corrida) de 12, 13, 14 y 16. Cada constante de fase 2 tiene por defecto el comportamiento del 10 Sep.

### What's Next

1. protocolo-80-20/03 (`load_hypercubes`, normalización train-foreground, `prepare_model_input`, `cnn/engine.py` de fig-aflatoxin). Es el módulo que sustituye entero el `cnn/data_setup.py` actual, así que hay que conservar allí `load_partitions` y los esquemas del ticket 02.
2. Triar `repo-health/02,03,04,06` (probablemente `wontfix`: la feature nueva los deja sin objeto; 06 lo absorbe protocolo-80-20/09).
3. Seguir la cadena 03 → 09; el replay (06) usa el número 21, la primera corrida real es la 22.

## Blockers / Risks

- [ ] `ga.py` sigue roto al importar (`cnn.engine2`) y por eso `init.sh` ya no lo importa; protocolo-80-20/05 lo reescribe y devuelve `import ga` a `init.sh`.
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
- protocolo-80-20/02: `split_dataset.py` y `tests/test_protocol.py` (nuevos), `cnn/data_setup.py` (esquemas + `load_partitions`), `pyproject.toml` (`[tool.pytest.ini_options]`), `init.sh` (compila e importa `split_dataset`), `CLAUDE.md` (Layout).
- protocolo-80-20/01: `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `config.py` (nuevo), `init.sh`, `CLAUDE.md` (Layout, Invariants, Verification); borrados `cnn/train.py`, `cnn/model_info.py`, `cnn/util/helper_functions.py`, `train_dataset.csv`, `test_dataset.csv`, `run.log`; `tests/data/*.csv` pasan a estar versionados.

## Evidence of Completion

- [x] **protocolo-80-20/02** (15 Sep): `uv run python split_dataset.py` escribe `data/evaluation_partitions.csv` **byte a byte idéntico** a `tests/data/reference_evaluation_partitions.csv` (22 / 6 / 8 capturas, 708 / 160 / 256 recortes). 18 tests en `tests/test_protocol.py`, todos verdes; `init.sh` ya corre pytest.

- [x] `SKIP_SYNC=1 ./init.sh` verde de principio a fin (2026-09-15, tras protocolo-80-20/01). Avisa (no falla) de que falta `data/evaluation_partitions.csv`, que escribe el ticket 02, y salta pytest porque aún no hay `tests/test_*.py`.

## Notes for Next Session

- `cnn/train.py`, `cnn/model_info.py` y `cnn/util/` ya están borrados (protocolo-80-20/01); su contenido sigue en la historia, en `b7f1fc9`.
- `create_summary_table.py` only reads experiments 1–10 (repo-health/05).
- **Cabos sueltos que dejan los borrados del ticket 01** (cada uno tiene ya su ticket, no se tocan ahora): `check_data.py:116` sigue teniendo `default="train_dataset.csv"`, que ya no existe → ticket 10 lo reescribe; el README describe `helper_functions.py`, `train_dataset.csv` y `test_dataset.csv` → ticket 09; `cnn/data_setup.py:11` define su propio `NUM_WORKERS = os.cpu_count()`, que compite con `config.NUM_WORKERS = 8` (que sí forma parte del resultado reproducible) → ticket 03 sustituye el módulo entero. `init.sh` compila `check_data.py` y `ga.py` pero no los importa, así que la puerta sigue verde con ellos rotos.
- **Las tres trampas del spec están resueltas por el ticket 01**: `/data/` en el `.gitignore`, `.venv` con torch 2.13.0+cu126 y python 3.13.5, e `init.sh` sin la comprobación de `train_dataset.csv`/`test_dataset.csv`.
- `data/` (symlink `cropped_hypercubes` a los 1.124 NPZ, `cropped_hypercubes.csv`, `spectral_axes.csv`) y `out/models/` están en el árbol y siguen ignorados a propósito; `tests/data/*.csv` ya no lo están.
- Hubo una rama `feature/protocolo-80-20` creada la madrugada del 15 Sep y borrada esa misma mañana; sus dos commits (`f1e4759`, `3a19e48`, solo borrados de ficheros obsoletos) cuelgan en el reflog y no hacen falta. La rama se vuelve a crear desde `feature/experiments`.
