# Session Handoff

## Current Objective

- Goal: make the repo restartable for agents; get `./init.sh` green.
- Current status: harness files written and validated; `.venv` synced; `ga.py` import broken (missing `cnn.engine2`).
- Branch / commit: feature/experiments @ 7357da7

## Completed This Session

- [x] Harness scaffolded and tailored (CLAUDE.md, init.sh, progress.md, this file).
- [x] Diagnosed why `ga.py` cannot run: `cnn/engine2.py` deleted in 7357da7 while still imported.
- [x] Configured Matt Pocock skills (`docs/agents/`) with local markdown under `.scratch/` as the tracker; migrated the six features from `feature_list.json` (deleted) into `.scratch/repo-health/issues/`.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Import | `uv run python -c "import ga"` | FAIL | `ModuleNotFoundError: No module named 'cnn.engine2'` |
| Deps | `./init.sh` (uv sync) | PASS | torch 2.7.1 installed |
| Compile | `uv run python -m compileall ...` | PASS | |
| Manifests | init.sh manifest step | PASS | 897 train / 226 test rows, 0 missing |
| CUDA | `torch.cuda.is_available()` | True | 4 devices |

## Files Changed

- New: `CLAUDE.md`, `init.sh`, `progress.md`, `session-handoff.md`, `docs/agents/{issue-tracker,triage-labels,domain}.md`, `.scratch/repo-health/`

## Decisions Made

- See `progress.md` → Decisions Made.

## Blockers / Risks

- repo-health/02 blocks every feature that runs the GA.

## Next Session Startup

1. Read `CLAUDE.md`.
2. Read `progress.md` and `.scratch/repo-health/issues/`. Claim one `ready-for-agent` ticket by setting its `Status:` to `in-progress`.
3. Run `SKIP_SYNC=1 ./init.sh` (deps already installed). Expect it to fail at the import step until repo-health/02 is done.

## Recommended Next Step

- repo-health/02. Quickest safe fix: `git show 7357da7^:cnn/engine2.py > cnn/engine2.py`; cleaner fix: port its `y_preds/y_true` collection and the sklearn report/confusion matrix into `cnn/engine.py` and switch `ga.py` to `cnn.engine`.
