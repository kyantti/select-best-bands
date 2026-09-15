# 12 – Crop-count-balanced validation split behind `VALIDATION_BALANCE`

Status: ready-for-agent
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

- [ ] `uv run python -c "import config; print(config.VALIDATION_BALANCE)"` prints `acquisition_stratified`.
- [ ] With the default, `uv run python split_dataset.py` still reproduces `tests/data/reference_evaluation_partitions.csv` exactly (same `diff` as ticket 02) and still prints 22 / 6 / 8 acquisitions and 708 / 160 / 256 crops.
- [ ] `uv run python split_dataset.py --validation-balance crop_count_balanced` prints the achieved max/min crop ratio (≤ 1.5), the per-class validation crop counts, and 8 test acquisitions with the same identities as the reference partition.
- [ ] `uv run pytest -q tests/test_protocol.py -k balance` passes (real-manifest ratio, default reproduction, unreachable-target synthetic case).
- [ ] Running the balanced split twice writes an identical manifest.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

Phase 2 rule (spec, 15 Sep): the default value of every phase-2 constant reproduces the 10 Sep run, and the phase-1 system test is re-run with the defaults after this ticket lands (ticket 17). Changing the split changes the fitness of every candidate, so the search has to be repeated whole — that is why this, ticket 14 and ticket 16 go into the same run rather than one at a time.
