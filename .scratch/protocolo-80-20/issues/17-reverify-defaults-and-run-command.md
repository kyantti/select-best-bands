# 17 – Re-verify the defaults and hand over the command for the real run

Status: ready-for-agent
Blocked by: 12, 13, 14, 16

## Description

Closing ticket of phase 2: with every improvement in the tree and every phase-2 constant at its default, the phase-1 system test is run again and has to give the same numbers as before. Then the documentation and the launch command for the ~22 h run are handed over. Launching that run is Pablo's, not the agent's.

- Re-run the phase-1 system test with `VALIDATION_BALANCE = "acquisition_stratified"`, `BACKBONE_CHECKPOINT = None`, `FINAL_SEEDS = [2718]`: the split reproduces the reference partition; 366/262/225 gives seed 3104252108 and validation weighted F1 `0.8700979843225085`; the final model gives `0.722142952443074` with the same confusion matrix and predictions; the bootstrap gives `[0.645, 0.848]`; the cache-only search over the 383 real candidates ends with the same winner and the same per-generation history.
- README and `CLAUDE.md` gain the phase-2 section: what each constant does, what its default is, why the default is the 10 Sep behaviour, and how to turn each improvement on.
- The launch command for the full phase-2 run is written down with the constants it needs set, the expected wall-clock (~22 h), the log to `tail -f`, and how to resume it if it dies. `progress.md` records that it is Pablo's to launch.
- Tolerance, unchanged from phase 1: exact with the same versions, the same A100, 8 persistent workers and manifest order; any deviation from those conditions explains ±0.01–0.02 without being a port failure.

## Done when

- [ ] Every check of the phase-1 system test passes again with the phase-2 defaults; `## Evidence` lists command and output for each of the five (split, candidate, final model, bootstrap, cache-only search).
- [ ] `uv run pytest -q` passes whole, phase-1 and phase-2 tests together.
- [ ] `grep -n "VALIDATION_BALANCE\|BACKBONE_CHECKPOINT\|FINAL_SEEDS" README.md` finds all three with their defaults explained.
- [ ] README states the launch command for the phase-2 run and that phase 1 and phase 2 are never mixed in one run without this ticket having passed first.
- [ ] `./init.sh` (full, with sync) is green.
- [ ] `git status --short out/` shows no change to any file of experiments 1–20.

## Evidence

_(none yet)_

## Comments

The rule this ticket enforces (spec, 15 Sep): if an improvement changes the 10 Sep number, the verification disappears. Running the system test with the defaults after the improvements land is what proves each one is off unless asked for — and therefore that a difference in the long run comes from the improvement and not from the port.
