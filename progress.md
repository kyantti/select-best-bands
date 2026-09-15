# Session Progress Log

## Current State

**Last Updated:** 2026-09-15 (tarde)
**Branch:** feature/experiments at b7f1fc9
**Active Feature:** `protocolo-80-20` – spec publicado, sin tickets todavía. Next step: `/to-tickets` sobre `.scratch/protocolo-80-20/spec.md`, luego implementar.

**No empieces por repo-health/02.** La feature `protocolo-80-20` reescribe `ga.py` entero, así que 02 (restaurar el import de `engine2`), 03 (overrides por entorno) y 04 (tests de los operadores viejos) quedan sin objeto. Están pendientes de triaje por el autor. `repo-health/05` sigue siendo válido e independiente; `06` lo absorbe el README nuevo.

## Status

### What's Done

- [x] Experiments 1–20 completed; results tracked in `out/tables`, `out/figures`, `out/logs` (see `run.log`).
- [x] Harness created: `CLAUDE.md`, `init.sh`, `progress.md`, `session-handoff.md`.
- [x] repo-health/01: `uv sync` populated `.venv`; compile, manifest (897 train / 226 test rows, 0 missing) and CUDA (4 GPUs visible) checks pass.
- [x] Matt Pocock skills configured: `docs/agents/{issue-tracker,triage-labels,domain}.md` and an `## Agent skills` section in `CLAUDE.md`. Issue tracker is local markdown under `.scratch/`.

- [x] `protocolo-80-20`: spec publicado en `.scratch/protocolo-80-20/spec.md` (`ready-for-agent`). Plan de implementación detallado, con los valores de referencia para la verificación bit a bit, en `~/.claude/plans/quiero-reorganizar-el-proyecto-parsed-llama.md`.

### What's In Progress

- [ ] `protocolo-80-20`: sin tickets. Siguiente acción: `/to-tickets` sobre el spec.

### What's Next

1. `/to-tickets` sobre `.scratch/protocolo-80-20/spec.md`.
2. Triar `repo-health/02,03,04` (probablemente `wontfix`: la feature nueva los deja sin objeto).
3. Implementar `protocolo-80-20` empezando por la rama y el entorno (paso 1 del plan).

## Blockers / Risks

- [ ] `ga.py` is broken at import (repo-health/02). The last 10 experiments were run before that deletion, so results are valid, but no new experiment can start.
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

## Evidence of Completion

- [x] `./init.sh` up to the import step: green (2026-09-15).
- [ ] `./init.sh` fully green: blocked on repo-health/02.

## Notes for Next Session

- `cnn/train.py` and `cnn/model_info.py` use bare `import engine` and only work when run from inside `cnn/`; they are not on the GA path. La feature `protocolo-80-20` los borra.
- `create_summary_table.py` only reads experiments 1–10 (repo-health/05).
- **Trampas conocidas antes de implementar** (detalladas en el spec): `.gitignore` tiene `data/` sin barra inicial, así que ignora también `tests/data/` y los CSV de referencia ya copiados allí (cambiar a `/data/`). El `.venv` tiene torch 2.7.1 por el `uv.lock` viejo; el protocolo necesita 2.13.0+cu126 y los pines del plan. `init.sh` comprueba `train_dataset.csv`/`test_dataset.csv`, que la feature borra: hay que actualizarlo en la misma feature.
- En el árbol de trabajo ya están, sin commitear: el symlink `data/cropped_hypercubes` a los 1.124 NPZ, `data/cropped_hypercubes.csv`, `data/spectral_axes.csv`, `tests/data/*.csv` y `out/models/`.
- Hubo una rama `feature/protocolo-80-20` creada la madrugada del 15 Sep y borrada esa misma mañana; sus dos commits (`f1e4759`, `3a19e48`, solo borrados de ficheros obsoletos) cuelgan en el reflog y no hacen falta. La rama se vuelve a crear desde `feature/experiments`.
