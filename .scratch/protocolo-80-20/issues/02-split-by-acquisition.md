# 02 – Acquisition-grouped, stratified 80/20 split

Status: ready-for-agent
Blocked by: 01

## Description

`split_dataset.py` reads the cropped-hypercube manifest and writes `data/evaluation_partitions.csv`, and a partition loader in `cnn/data_setup.py` reads that file back with leakage checks. The split reproduces the 10 Sep 2026 partition exactly, so every later result is comparable with the recorded one.

Behaviour (copied from fig-aflatoxin's partition stage, not reinvented):

- The manifest is deduplicated to acquisitions with their SeverityClass; identities are sorted; a stratified `train_test_split` with the partition seed separates train/test, and a second one with the same seed separates train/validation inside train. Every crop inherits its acquisition's partition. Output columns include `partition` (train/test) and `selection_partition` (train/validation/empty for test).
- The split fails with a clear message if any class has no acquisition on either side of either boundary, instead of silently degrading stratification.
- The loader revalidates that no acquisition and no crop crosses a boundary, that test rows have no selection partition, and that the three groups (train-fit, validation, test) are all present. A hand-edited manifest that leaks is rejected.
- The script prints a count per (partition, selection_partition, class).

Tests (no GPU, no `data/`): grouping and stratification on a synthetic manifest of 24 acquisitions × 3 crops; exact reproduction of the reference partition in `tests/data/` from the reference manifest; failure when a class has no acquisitions on one side; loader rejection of a leaking or incomplete manifest (parametrized).

## Done when

- [ ] `uv run python split_dataset.py` writes `data/evaluation_partitions.csv` and prints 22 / 6 / 8 acquisitions and 708 / 160 / 256 crops for train-fit / validation / test.
- [ ] `diff <(cut -d, -f2- data/evaluation_partitions.csv) <(cut -d, -f2- tests/data/reference_evaluation_partitions.csv)` is empty.
- [ ] `uv run pytest -q tests/test_protocol.py -k split` passes (stratification, reproduction, class-without-acquisitions, leakage rejection).
- [ ] `SKIP_SYNC=1 ./init.sh` is green and runs pytest.

## Evidence

_(none yet)_

## Comments

