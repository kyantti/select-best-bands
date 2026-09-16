# 12 – Crop-count-balanced validation split behind `VALIDATION_BALANCE`

Status: done
Blocked by: 09

## Description

`split_dataset.py` gains a second validation policy that balances the validation partition by **number of crops** instead of only by acquisitions, and records the ratio it achieved. The default stays the 10 Sep behaviour, so the phase-1 verification survives.

Why: stratifying the train/validation boundary by acquisition left validation with 52 C0 and 60 C2 crops against 24 C1 and 24 C3 — a max/min ratio of 2.5. The fitness is weighted F1, which weights by support, so the search was scored on a validation set that cared 2.5× more about two classes than the other two.

Behaviour:

- `config.VALIDATION_BALANCE` defaults to `"acquisition_stratified"` (today's `train_test_split` inside train). With `"crop_count_balanced"`, `assign_partitions` takes a policy parameter and the train/validation boundary becomes a deterministic choice: enumerate the admissible subsets of train acquisitions of the requested size (at least one acquisition per class on **both** sides), score each by the max/min ratio of crops per class, keep the minimum, break ties by the sorted acquisition identities.
- The train/test boundary is untouched under either policy: it stays the 10 Sep one.
- Frozen criterion: target ratio ≤ 1.5. If no admissible subset reaches it, the best one is published and the number recorded — grouping by acquisition and one-acquisition-per-class are never relaxed, and the split fails before relaxing them.
- The achieved ratio, the policy name and the per-class crop counts are written to the split summary so the report can quote the improvement over 2.5.

Tests (no GPU): on the real manifest, `crop_count_balanced` reaches a ratio ≤ 1.5 and leaves the train/test boundary identical to the reference partition; the default policy still reproduces the 10 Sep reference partition byte for byte; on synthetic data where no admissible subset reaches the target, the best subset is published, the ratio is recorded, and neither grouping nor stratification is relaxed.

## Done when

- [x] `uv run python -c "import config; print(config.VALIDATION_BALANCE)"` prints `acquisition_stratified`.
- [x] With the default, `uv run python split_dataset.py` still reproduces `tests/data/reference_evaluation_partitions.csv` exactly (same `diff` as ticket 02) and still prints 22 / 6 / 8 acquisitions and 708 / 160 / 256 crops.
- [x] `uv run python split_dataset.py --validation-balance crop_count_balanced` prints the achieved max/min crop ratio (≤ 1.5), the per-class validation crop counts, and 8 test acquisitions with the same identities as the reference partition.
- [x] `uv run pytest -q tests/test_protocol.py -k balance` passes (real-manifest ratio, default reproduction, unreachable-target synthetic case).
- [x] Running the balanced split twice writes an identical manifest.
- [x] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

**The validation the search is scored on went from 2.5 to 1.2, and the default did not move.**
16 Sep, no GPU.

- `uv run python -c "import config; print(config.VALIDATION_BALANCE)"` → `acquisition_stratified`.
- `uv run python split_dataset.py` (bare) still prints `22 / 6 / 8` acquisitions and
  `708 / 160 / 256` crops, and `diff data/evaluation_partitions.csv
  tests/data/reference_evaluation_partitions.csv` is **empty**: the 10 Sep boundary
  byte for byte. Its own summary line now names the weakness it leaves behind —
  `acquisition_stratified: max/min crops per class 2.50 (target 1.50, not reached)`.
- `uv run python split_dataset.py --validation-balance crop_count_balanced` (6.8 s, the
  376 740 subsets of 28 taken 6 at a time) prints
  `validation 6 acquisitions 176 crops  C0=48 C1=48 C2=40 C3=40` and
  `crop_count_balanced: max/min crops per class 1.20 (target 1.50, reached)`.
  **1.20 against 2.50.** The test line is unchanged — `8 acquisitions 256 crops
  C0=72 C1=68 C2=60 C3=56` — and the eight test acquisition identities `diff` empty
  against the reference partition, so only the boundary inside train moved.
- Two balanced runs in a row: `diff` empty.
- `uv run pytest -q tests/test_protocol.py -k balance` → **18 passed**. They cover the
  real-manifest ratio (1.2, and the 48/48/40/40 an independent enumeration found), the
  default reproducing the reference, the untouched train/test boundary, determinism, the
  unknown-policy refusal, the two summary lines, and the synthetic case where C0 carries
  ten crops per acquisition against one for the rest: the best subset is published at
  ratio 10.0 with `reaches_target` false, no acquisition crosses a boundary and no group
  has lost a class.
- `SKIP_SYNC=1 ./init.sh` green: **129 tests** (111 before, 18 new).
- `git status --short out/` empty: this ticket writes nothing under `out/`.

After `/code-review` (Standards + Spec): the refusal both policies raise moved into one
`unsplittable()` (it was written out three times); `validation_balance()` became
`measure_balance()` because it read like `config.VALIDATION_BALANCE`, which is the policy,
not the measurement, and the module-level `crops_per_class()` became
`count_crops_per_class()` because it collided with the `ValidationBalance` field of that
name; `main` grew the `ValueError → one line on stderr, exit 1` handler the other scripts
have, and now weighs the partition **before** writing it, so one that cannot be weighed
never reaches disk; the untouched-boundary test compares against the recorded reference
instead of a recomputation; and `held_out_size` is now pinned against scikit-learn itself
over six sizes rather than against the single number 6.

## Comments

Phase 2 rule (spec, 15 Sep): the default value of every phase-2 constant reproduces the 10 Sep run, and the phase-1 system test is re-run with the defaults after this ticket lands (ticket 17). Changing the split changes the fitness of every candidate, so the search has to be repeated whole — that is why this, ticket 14 and ticket 16 go into the same run rather than one at a time.

Closed 16 Sep 2026. Two notes for ticket 17, which owns the docs and the launch command:

- **The switch for a real run is `config.VALIDATION_BALANCE`, not a flag.** `run.sh` calls
  `split_dataset.py` bare, so it picks up whatever `config.py` says — the chain needs no new
  routing. `--validation-balance` is for inspecting a policy without editing `config.py`; a
  balanced partition written by hand *is* overwritten by the next `run.sh`, which is why the
  constant is the switch. Worth one line in the README.
- **The balanced validation is 176 crops, not 160**, and train-fit drops to 692. Every
  candidate's fitness changes, so nothing under `out/` from experiment 21 or earlier is
  comparable with a run that turns this on.
