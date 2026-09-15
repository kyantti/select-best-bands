# Session Progress Log

## Current State

**Last Updated:** 2026-09-15
**Branch:** feature/experiments (clean at 7357da7)
**Active Feature:** none `in-progress`. Next pick: feat-002 – Restore ga.py imports after engine2 removal (`ready-for-agent`)

## Status

### What's Done

- [x] Experiments 1–20 completed; results tracked in `out/tables`, `out/figures`, `out/logs` (see `run.log`).
- [x] Harness created: `CLAUDE.md`, `feature_list.json`, `init.sh`, `progress.md`, `session-handoff.md`.
- [x] feat-001: `uv sync` populated `.venv`; compile, manifest (897 train / 226 test rows, 0 missing) and CUDA (4 GPUs visible) checks pass.
- [x] Matt Pocock skills configured: `docs/agents/{issue-tracker,triage-labels,domain}.md` and an `## Agent skills` section in `CLAUDE.md`. `feature_list.json` is the issue tracker.

### What's In Progress

- [ ] feat-002: `ready-for-agent`, not yet claimed. `./init.sh` currently stops at the import smoke test with `ModuleNotFoundError: No module named 'cnn.engine2'`.

### What's Next

1. feat-002 (see In Progress).
2. feat-003: smoke-run env overrides so verification does not need an hours-long GA run.

## Blockers / Risks

- [ ] `ga.py` is broken at import (feat-002). The last 10 experiments were run before that deletion, so results are valid, but no new experiment can start.
- [ ] `data/` is gitignored; `init.sh` warns instead of failing when it is absent so code-only work is possible on machines without the dataset.
- [ ] A default `ga.py` run is 50 generations × up to 25 individuals × up to 50 epochs on an A100. Never launch one as "verification".

## Decisions Made

- **Instruction file is `CLAUDE.md`, not `AGENTS.md`**: the author uses Claude Code.
- **Manifest check warns rather than fails when `data/` is missing**: agents may work on the code on machines without the dataset.
- **One status vocabulary for harness and skills**: `feature_list.json.status` uses the five Matt Pocock triage roles plus `in-progress` and `done`. `not-started` was migrated to `ready-for-agent` on 2026-09-15; there is no `blocked` status (use `dependencies` or `needs-info`). See `docs/agents/triage-labels.md`.
- **`out/` is immutable history**: every experiment number owns its files; new runs take a new number, throwaway smoke runs use 90–99 and are deleted afterwards.

## Files Modified This Session

- `CLAUDE.md`, `feature_list.json`, `init.sh`, `progress.md`, `session-handoff.md` – created (harness only, no source changes).
- `docs/agents/issue-tracker.md`, `docs/agents/triage-labels.md`, `docs/agents/domain.md` – created by `/setup-matt-pocock-skills`.
- `CLAUDE.md` (Working rules, Agent skills), `feature_list.json` (statuses), `init.sh` (final hint) – aligned to the unified status vocabulary.

## Evidence of Completion

- [x] `./init.sh` up to the import step: green (2026-09-15).
- [ ] `./init.sh` fully green: blocked on feat-002.

## Notes for Next Session

- `cnn/train.py` and `cnn/model_info.py` use bare `import engine` and only work when run from inside `cnn/`; they are not on the GA path.
- `create_summary_table.py` only reads experiments 1–10 (feat-005).
