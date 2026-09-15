# 02 – Acquisition-grouped, stratified 80/20 split

Status: done
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

- [x] `uv run python split_dataset.py` writes `data/evaluation_partitions.csv` and prints 22 / 6 / 8 acquisitions and 708 / 160 / 256 crops for train-fit / validation / test.
- [x] `diff <(cut -d, -f2- data/evaluation_partitions.csv) <(cut -d, -f2- tests/data/reference_evaluation_partitions.csv)` is empty.
- [x] `uv run pytest -q tests/test_protocol.py -k split` passes (stratification, reproduction, class-without-acquisitions, leakage rejection).
- [x] `SKIP_SYNC=1 ./init.sh` is green and runs pytest.

## Evidence

```
$ uv run python split_dataset.py
/home/pablosetra/Documents/tfg/select-best-bands/data/cropped_hypercubes.csv: 1124 crops
train-fit    22 acquisitions   708 crops  C0=152 C1=189 C2=164 C3=203
validation    6 acquisitions   160 crops  C0=52 C1=24 C2=60 C3=24
test          8 acquisitions   256 crops  C0=72 C1=68 C2=60 C3=56
wrote /home/pablosetra/Documents/tfg/select-best-bands/data/evaluation_partitions.csv

$ diff <(cut -d, -f2- data/evaluation_partitions.csv) <(cut -d, -f2- tests/data/reference_evaluation_partitions.csv)
                       # empty; the two files are in fact byte-identical
$ uv run pytest -q tests/test_protocol.py -k split
18 passed

$ SKIP_SYNC=1 ./init.sh
... imports OK / four data files present / 18 passed
=== init OK ===
```

The 18 tests are: acquisitions kept whole, every class on both sides of both
boundaries, exact reproduction of the reference partition, the three recorded
group sizes (22/708, 6/160, 8/256), the input manifest still being the recorded
one, both class-starvation failures (sklearn's refusal and our own count guard),
and seven rejected corruptions of the partition manifest — each verified to fail
at its own level (Acquisition crossing, crop crossing, test row with a selection
partition, missing validation group, missing test group, invented partition
name, duplicate CroppedHypercube, duplicate crop).

Notes for the next ticket:

- `assign_partitions` lives in `split_dataset.py`, `load_partitions` and the
  manifest schemas in `cnn/data_setup.py` (which ticket 03 rewrites around them).
- `pyproject.toml` gained `[tool.pytest.ini_options]` with `pythonpath = ["."]`,
  so tests import the root scripts.
- Known limit, inherited from fig-aflatoxin: the loader refuses leaks and
  duplicates but does not check that the partition covers the whole manifest, so
  a hand-edit that *deletes* rows is not detected.
- `check_data.py` still verifies no checksums; ticket 10 gives it that job. The
  loader's docstring says so rather than claiming a guarantee that is not there.

## Comments

