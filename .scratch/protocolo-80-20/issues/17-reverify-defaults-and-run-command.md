# 17 – Re-verify the defaults and hand over the command for the real run

Status: done
Blocked by: 12, 13, 14, 16

## Description

Closing ticket of phase 2: with every improvement in the tree and every phase-2 constant at its default, the phase-1 system test is run again and has to give the same numbers as before. Then the documentation and the launch command for the ~22 h run are handed over. Launching that run is Pablo's, not the agent's.

- Re-run the phase-1 system test with `VALIDATION_BALANCE = "acquisition_stratified"`, `BACKBONE_CHECKPOINT = None`, `FINAL_SEEDS = [2718]`: the split reproduces the reference partition; 366/262/225 gives seed 3104252108 and validation weighted F1 `0.8700979843225085`; the final model gives `0.722142952443074` with the same confusion matrix and predictions; the bootstrap gives `[0.645, 0.848]`; the cache-only search over the 383 real candidates ends with the same winner and the same per-generation history.
- README and `CLAUDE.md` gain the phase-2 section: what each constant does, what its default is, why the default is the 10 Sep behaviour, and how to turn each improvement on.
- The launch command for the full phase-2 run is written down with the constants it needs set, the expected wall-clock (~22 h), the log to `tail -f`, and how to resume it if it dies. `progress.md` records that it is Pablo's to launch.
- Tolerance, unchanged from phase 1: exact with the same versions, the same A100, 8 persistent workers and manifest order; any deviation from those conditions explains ±0.01–0.02 without being a port failure.

## Done when

- [x] Every check of the phase-1 system test passes again with the phase-2 defaults; `## Evidence` lists command and output for each of the five (split, candidate, final model, bootstrap, cache-only search).
- [x] `uv run pytest -q` passes whole, phase-1 and phase-2 tests together.
- [x] `grep -n "VALIDATION_BALANCE\|BACKBONE_CHECKPOINT\|FINAL_SEEDS" README.md` finds all three with their defaults explained.
- [x] README states the launch command for the phase-2 run and that phase 1 and phase 2 are never mixed in one run without this ticket having passed first.
- [x] `./init.sh` (full, with sync) is green.
- [x] `git status --short out/` shows no change to any file of experiments 1–20.

## Evidence

**Done (23 Sep).** With every phase-2 improvement in the tree and all three
constants at their defaults, the phase-1 system test gives the same numbers as
before. Checked first:

```
$ uv run python -c "import config; print(...)"
VALIDATION_BALANCE   'acquisition_stratified'
BACKBONE_CHECKPOINT  None
FINAL_SEEDS          [2718]
```

### 1 / 5 — the split reproduces the reference partition

```
$ uv run python split_dataset.py
train-fit    22 acquisitions   708 crops  C0=152 C1=189 C2=164 C3=203
validation    6 acquisitions   160 crops  C0=52 C1=24 C2=60 C3=24
test          8 acquisitions   256 crops  C0=72 C1=68 C2=60 C3=56
acquisition_stratified: max/min crops per class 2.50 (target 1.50, not reached)
$ diff tests/data/reference_evaluation_partitions.csv data/evaluation_partitions.csv
(empty)
```

The 2.50 is the 10 Sep boundary, unchanged by ticket 12 being in the tree.

### 2 / 5 — one candidate: the derived seed and the fitness

`ga.py --evaluate` first **refused**, correctly, because its predictions file
from an earlier run is already under `out/`: *"is already there; results under
out/ are not overwritten. Move it aside, or pass --predictions with another
path."* Re-run with the predictions diverted off the record:

```
$ CUDA_VISIBLE_DEVICES=1 uv run python ga.py --evaluate 366 262 225 \
    --predictions <scratch>/t17_validation_predictions.csv
candidate      (366, 262, 225)
wavelengths_nm 891.02, 747.5, 697.05
candidate seed 3104252108
mean           [0.6560380893489065, 0.6596170752860457, 0.2233070946048868]
std            [0.11694627746639438, 0.12472051938804612, 0.07551858819936108]
weighted_f1    0.8700979843225085
macro_f1       0.8763610130071496
ordinal_mae    0.1875
quadratic_weighted_kappa 0.8711063372717508
elapsed_s      52.1
```

Seed `3104252108` and weighted F1 `0.8700979843225085`, both as the ticket
quotes. The 160 validation predictions also `diff` empty against the recorded
`out/tables/evaluate_366_262_225_validation_predictions.csv`.

### 3 / 5 — the final model

```
$ CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 21 --bands 366 262 225
mean           [0.65546626556172, 0.6584057421588269, 0.22467686763531047]
std            [0.11717051462839731, 0.125102268224151, 0.07519302019349307]
weighted_f1    0.722142952443074
macro_f1       0.7223819971537002
ordinal_mae    0.5
quadratic_weighted_kappa 0.5815011372251706
confusion      [[50, 3, 6, 13], [6, 43, 9, 10], [4, 7, 46, 3], [4, 3, 3, 46]]
```

`diff` empty on all four tables against copies taken before the run — so the
`FINAL_SEEDS` list of ticket 16 does not move the single-seed result.

### 4 / 5 — the interval

```
$ uv run python bootstrap.py 21
weighted F1 0.722  95% CI [0.645, 0.848]  (5000 resamples of 8 acquisitions, seed 1729)
```

`exp_21_bootstrap_366_262_225.json` byte-identical.

### 5 / 5 — the cache-only search

```
$ uv run python ga.py 21 --no-evaluate
exp_21: 448 bands, population 20, 25 generations, 50 epochs, restored 383
winner         (366, 262, 225)
weighted_f1    0.8700979843225085
unique 383 | hits 451 | restored 383 | evaluated 0
```

Same winner, and `exp_21_ga_stats.csv` (the per-generation history),
`exp_21_candidates.csv`, `exp_21_cnn_results_366_262_225.csv` and
`exp_21_fitness_evolution.png` all byte-identical.

### The whole record, and the one field that is not reproducible

All 15 `exp_21_*` artifacts were copied aside first and diffed after the five
checks. **14 of 15 byte-identical.** The exception is
`exp_21_ga_summary.json`, and only in `elapsed_seconds` — wall clock, 0.0109 s
against 0.0174 s for the same zero-training replay. Reverted, filed for triage
as `repo-health/08`; `ga.py` is the one writer that puts a stopwatch reading in
an artifact, which `train_final.write_metrics` deliberately does not.

```
$ git status --short out/
(empty)
```

### Tests, docs and the handover

- `uv run pytest -q` → **193 passed**; full `./init.sh` (with sync) green.
- README gains **"The three phase-2 switches"** (a table: default, what the
  default means, and how to turn each one on) and **"🚀 The phase-2 run"** (the
  five re-verification commands, the by-hand launch, ~22 h, the log to
  `tail -f`, and how to resume). `grep -n "VALIDATION_BALANCE\|BACKBONE_CHECKPOINT\|FINAL_SEEDS" README.md`
  finds all three.
- `CLAUDE.md` gains the same section, compressed.
- Both say **never mix phase 1 and phase 2 in one run without re-verifying the
  defaults first**, that such a run takes a new experiment number, and that
  **launching it is Pablo's, not an agent's**. `progress.md` records the same.

### What `/code-review high` found

It reviewed the whole branch, not just this ticket. **One finding was this
ticket's own work and was fixed here**; the rest are pre-existing code and were
filed for triage, because ticket 17 is documentation and does not touch code.

**Fixed here — the launch recipe I first wrote was not runnable, and its
failure mode was a leak.** Step 1 redrew the partition under
`crop_count_balanced`; step 2's `pretrain.py --cohort fit` then refuses, because
`out/models/pretrain_fit.pt` has existed since 16 Sep and the recipe never said
`--force`. The natural workaround — keep the existing checkpoint — is a
contaminated search, and I verified the size of it:

```
pretrain_fit.pt saw          : 22 acquisitions
balanced validation          : 6
balanced validation ALREADY in pretrain_fit.pt: 5
```

Five of the six acquisitions the balanced policy scores the fitness on are
inside the 22 the backbone was adapted to, and **nothing in `ga.py` would have
caught it**. The recipe now: sets the constant instead of passing the flag (so
`pretrain.py`'s sidecar and `run.sh`'s split agree with it), rebuilds both
checkpoints with `--force` as a required step when the boundary moves, and adds
a leakage check the reader runs *before* spending the 22 hours. That check is in
the README as a runnable snippet and passes on the current tree (0 validation
and 0 test acquisitions seen). Two more warnings went in: steps 1 and 2 are one
decision, and `run.sh` must not be run while a search is in flight.

**Filed for triage, not fixed** (all outside this ticket; none affects the five
checks above, because every one of them runs with the improvements switched off):

| | |
|---|---|
| `repo-health/09` | `train_final.RECORDED_SETTINGS` omits `num_workers` and the partition identity, so changing the worker count silently overwrites `exp_21`'s result — the one number the thesis quotes. `config.py` says the worker count *is* part of the result; `ga.fitness_contract` records it, `train_final.py` does not. |
| `repo-health/10` | **"Two checkpoints, never one" is documentation only.** Nothing reads the `pretrain_*.json` sidecar; `checkpoint_path` only checks `is_file()`. `ga.py 22 --checkpoint out/models/pretrain_train.pt` — one character off — is accepted silently and contaminates every cached fitness of a 22-hour run. The most expensive of the open findings. |
| `repo-health/11` | `run.sh` redraws the shared partition unconditionally, and nothing on disk records which policy drew it. |
| `repo-health/12` | A search killed early leaves `exp_NN_ga_config.json`, which `train_final.py` then reads as an earlier protocol's, falsely accusing a number this protocol just created. |
| `repo-health/13` | `analyze_test_errors.py` reads a CSV without closing it. |

Also cleared by the review and worth recording: `torch.use_deterministic_algorithms(True)`
raises no cuBLAS alert on this box; a NaN kappa cannot reach `CandidateFitness`
because both partitions structurally carry four classes; and `ga.py`'s refusal
of experiments 1–20 does cover all twenty.

### Two things the launch still waits on

1. **Ticket 15 has not run.** Whether the contrastive backbone belongs in the
   run at all is what its paired gate decides, against a rule frozen before the
   numbers exist. Until it has run, the ImageNet arm — step 3 without
   `--checkpoint` — is the run this protocol is verified for, and the README
   says so.
2. **The test-read decision of ticket 16 is Pablo's** and is still open. It
   binds the launch, not the code: `FINAL_SEEDS = [2718]` keeps today's
   behaviour until he answers.

## Comments

The rule this ticket enforces (spec, 15 Sep): if an improvement changes the 10 Sep number, the verification disappears. Running the system test with the defaults after the improvements land is what proves each one is off unless asked for — and therefore that a difference in the long run comes from the improvement and not from the port.
