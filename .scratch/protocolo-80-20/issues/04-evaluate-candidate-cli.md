# 04 – Evaluate one band triplet from the command line

Status: done
Blocked by: 03

## Description

`uv run python ga.py --evaluate R G B` trains a ResNet50 on the train-fit crops for that triplet and reports validation metrics, reproducing the recorded numbers for the winner. This is the first end-to-end path through data loading, seeding, training and metrics, and it validates the data-identity hash before the search is built on it.

Behaviour (in `cnn/model.py`, copied from fig-aflatoxin's evaluation module and evaluate stage):

- ResNet50 with pretrained weights and a 4-class head; metrics with explicit labels 0..3 and `zero_division=0`: weighted F1, macro F1, ordinal MAE, quadratic weighted kappa.
- Data identity: SHA-256 of the sorted train and validation hypercube descriptors (with `spectral_axis_id`, array dtype/shape/hash, class as `C0`..), of the train partition rows (seven string columns, `schema_version` "1", sorted by crop id) and of the spectral axis.
- Candidate seed = first 8 hex digits of the SHA-256 of the canonical JSON `{candidate, input_checksums, seed, seed_policy="sha256-candidate-split-v1"}`. For 366/262/225 with the real data it is 3104252108.
- Evaluation seeds `random`, NumPy, torch and CUDA, enables deterministic algorithms, builds the train loader with its own generator and the validation loader without one, builds loaders **before** the model, trains 50 epochs (batch 32, Adam 0.001, 64×128, 8 persistent workers, AMP) with no early stopping, and predicts in order.
- The CLI loads only train-fit and validation rows, prints the candidate seed, the frozen mean/std, the four metrics and the elapsed seconds, and accepts `--epochs` for smoke use.

Test: the candidate seed is stable across calls, differs for a permutation of the same bands and differs for another data identity.

## Done when

- [x] `uv run pytest -q tests/test_protocol.py -k seed` passes.
- [x] `CUDA_VISIBLE_DEVICES=1 uv run python ga.py --evaluate 366 262 225` prints candidate seed `3104252108`, mean `[0.6560380893489065, 0.6596170752860457, 0.2233070946048868]`, std `[0.11694627746639438, 0.12472051938804612, 0.07551858819936108]`, weighted F1 `0.8700979843225085`, macro F1 `0.8763610130071496`, ordinal MAE `0.1875`, QWK `0.8711063372717508` (~3.5 min on the A100).
- [x] The validation predictions it writes are identical to `runs/evaluate_selected_band_triplet/d49fc1f97dad04461e87c305c6e0a0bfff8a5e9b3b61d5f3ca6be6f7b4ced125/validation_predictions.csv` in `~/Documents/fig-aflatoxin` (`diff` on the id/actual/predicted columns is empty).
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python ga.py --evaluate 10 20 30 --epochs 1` finishes in under a minute and prints metrics. **Not met as written:** it prints the metrics and trains in 4.4 s, but the command takes 2 min 56 s because loading the crops costs ~2.5 min. See the loading note under Comments; whether that closes the box is the author's call.
- [x] `SKIP_SYNC=1 ./init.sh` is green (it may still skip `import ga` until ticket 05).

## Evidence

**The port is exact.**  `CUDA_VISIBLE_DEVICES=1 uv run python ga.py --evaluate 366 262 225`
(15 Sep, 50 epochs, 39.2 s of training on the A100) printed every recorded value
unchanged:

```
candidate      (366, 262, 225)
wavelengths_nm 891.02, 747.5, 697.05
candidate seed 3104252108
mean           [0.6560380893489065, 0.6596170752860457, 0.2233070946048868]
std            [0.11694627746639438, 0.12472051938804612, 0.07551858819936108]
weighted_f1    0.8700979843225085
macro_f1       0.8763610130071496
ordinal_mae    0.1875
quadratic_weighted_kappa 0.8711063372717508
```

`diff` against the recorded `validation_predictions.csv` of run
`d49fc1f9...` is empty **on all four columns, not just the three asked for**:
the 160 rows are byte for byte the same file.

**The data identity is exact too**, which is what makes the seed land on
3104252108.  All four checksums equal the ones in that run's `success.json`:

| checksum | value |
| --- | --- |
| `spectral_axis` | `608f9d5f3a83169a1cba423fd43c5a85326fa03d8292d5c4fcd9409e17c0c1af` |
| `train_cropped_hypercubes` | `f69c27438904ebe4012b853e248d8b6888a11d92ec8cc092ac0d23afa314234b` |
| `train_evaluation_assignments` | `6a3df7838601815f1b309fb28d22d7b44db8a8756b5456a5407e7bac7bb0e01c` |
| `validation_cropped_hypercubes` | `a01d46e96f54778db61948ba0c5dabd810089047181ff598fa486ef7c9bc02ee` |

(708 train-fit and 160 validation crops, 868 train partition rows.)

**Smoke:** `CUDA_VISIBLE_DEVICES=1 uv run python ga.py --evaluate 10 20 30 --epochs 1`
prints seed `1488839195` and weighted F1 `0.10183759244435904`, identical on two
separate runs.  Training is 4.4 s.

**Tests:** `uv run pytest -q tests/test_protocol.py -k seed` -> 5 passed;
the whole file -> 51 passed (42 before).  `SKIP_SYNC=1 ./init.sh` green.

**Files:** `cnn/model.py` (new), `ga.py` (rewritten as the `--evaluate` CLI),
`tests/test_protocol.py` (+9 tests), `init.sh` (imports `cnn.model`),
`CLAUDE.md` + `progress.md` (layout and state),
`out/tables/evaluate_366_262_225_validation_predictions.csv` (the verification
artifact above; it is outside the `exp_NN_` namespace and collides with nothing).

**Review.** `/code-review` against this ticket ran both axes.  Acted on: the CLI
now **refuses** to run when `config.DEVICE` names an accelerator the machine
does not have (it used to fall back to the CPU and produce numbers that would
never reproduce — the one divergence from fig-aflatoxin that could change a
number silently); the predictions file is opened exclusively and the collision
is checked *before* anything is loaded, so `out/` is never overwritten and no
hour of training is spent on a result that cannot be written; `--seed` was
removed as unasked-for; `RESNET50_WEIGHTS_IDENTITY` was removed as unused until
ticket 05's fitness contract needs it; docstrings gained their `Raises:`
sections and the repeated three-line setup in the tests became `cohorts_of`.
Kept deliberately: `--predictions` (the escape hatch the immutable-`out/`
invariant requires) and the `experiment` positional, which is accepted and then
refused so the documented `ga.py N` names ticket 05 instead of dying on an
unrecognized argument.

## Comments

If mean/std match but the metrics do not, the divergence is in RNG order, loader
configuration or library versions, not in the data path. Do not "fix" the risks
listed in the plan (manifest order, 8 persistent workers, validation loader
without generator, loaders before model, float32 mean/std).

**Loading the crops costs ~2.5 min, so the one-epoch smoke is ~3 min, not the
under-a-minute the ticket predicted.**  Measured: 708 crops take 123 s to load,
868 take ~150 s, while the one-epoch training itself is 4.4 s.  The cost is zlib
decompression of the NPZ artifacts in `load_hypercubes` (ticket 03), it is
CPU-bound, and fig-aflatoxin pays exactly the same price.  Nothing in this ticket
touches it.  It does not matter for ticket 05: a search loads the crops once and
keeps them in RAM for every candidate.  Raising it only as a correction to the
expectation in the Done-when list.

**`ga.py` was rewritten rather than extended.**  It still imported the deleted
`cnn.engine2` and `cnn.util.helper_functions`, so `python ga.py --evaluate ...`
could not run at all while the old search was in the file.  The old search is in
git history (`2f5adeb` and earlier) and ticket 05 replaces it anyway; what is in
`ga.py` now is the `--evaluate` CLI plus `load_selection_data()`, which ticket 05
reuses.  `python ga.py N` exits with a message naming ticket 05.
