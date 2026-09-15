# Session Handoff

## Current Objective

- Goal: llevar el protocolo 80/20 fiable de `~/Documents/fig-aflatoxin` a la estructura plana de este repo, en la rama `feature/protocolo-80-20`.
- Current status: spec publicado y `ready-for-agent`; sin tickets y sin código escrito.
- Branch / commit: feature/experiments @ b7f1fc9

## Completed This Session

- [x] Analizados los dos proyectos y localizado qué código hace fiable el resultado del 10 Sep 2026 (weighted F1 0,722 en test, IC [0,645, 0,848], bandas 366/262/225).
- [x] Plan de implementación escrito en `~/.claude/plans/quiero-reorganizar-el-proyecto-parsed-llama.md`: pasos, origen de cada bloque de código a copiar, diseño de la caché, tests, verificación y riesgos de divergencia.
- [x] Spec publicado en `.scratch/protocolo-80-20/spec.md` (`Status: ready-for-agent`), con 43 historias de usuario y las decisiones de implementación y de prueba.
- [x] Preparado el dataset de entrada en el árbol de trabajo: symlink `data/cropped_hypercubes` a los 1.124 NPZ, más los manifiestos y los CSV de referencia de `tests/data/`.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Dataset enlazado | `ls data/cropped_hypercubes \| wc -l` | 1124 | symlink a `fig-aflatoxin/runs/crop_hypercubes/225599af…` |
| Partición de referencia | recuento del CSV copiado | 22/6/8 capturas, 708/160/256 recortes | `tests/data/reference_evaluation_partitions.csv` |
| Entorno objetivo | `.venv/bin/python -c "import torch"` en fig-aflatoxin | 2.13.0+cu126, CUDA True, 4 A100 | aquí el `.venv` tiene 2.7.1 (lock viejo) |

## Files Changed

- New: `.scratch/protocolo-80-20/spec.md`.
- Modified: `progress.md`, `session-handoff.md`.
- Untracked en el árbol: `data/cropped_hypercubes` (symlink), `data/cropped_hypercubes.csv`, `data/spectral_axes.csv`, `tests/data/*.csv`, `out/models/`.

## Decisions Made

- See `progress.md` → Decisions Made.

## Blockers / Risks

- `repo-health/02,03,04` quedan sin objeto con esta feature (reescribe `ga.py`). Pendientes de triaje del autor; no empezar por ellos.
- La verificación bit a bit exige las versiones fijadas, GPU A100 y `num_workers=8`. Cualquier desviación explica ±0,01–0,02 en F1 sin ser un fallo del port.
- La corrida real son ~22 h de GPU y la lanza el autor, no un agente.

## Next Session Startup

1. Leer `CLAUDE.md`, este fichero y `progress.md`.
2. Leer `.scratch/protocolo-80-20/spec.md` y el plan en `~/.claude/plans/quiero-reorganizar-el-proyecto-parsed-llama.md`.
3. `/to-tickets` sobre el spec para crear `.scratch/protocolo-80-20/issues/NN-*.md`.
4. `SKIP_SYNC=1 ./init.sh` falla hoy en el import (`cnn.engine2`); se arregla solo cuando la feature reescriba `ga.py` e `init.sh`.

## Recommended Next Step

- `/to-tickets`, y luego el paso 1 del plan: recrear la rama `feature/protocolo-80-20` desde `feature/experiments`, fijar `pyproject.toml` a las versiones de la corrida real y resincronizar el `.venv`.
