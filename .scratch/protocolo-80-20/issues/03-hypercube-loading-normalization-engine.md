# 03 – Hypercube loading, train-only normalization, augmentation and training engine

Status: ready-for-agent
Blocked by: 02

## Description

`cnn/data_setup.py` can turn partition rows into in-memory hypercubes and model inputs, and `cnn/engine.py` trains without early stopping. This is the data path every later step (candidate evaluation, final model, prediction) uses, so it must be copied literally from fig-aflatoxin's evaluation module.

Behaviour:

- The spectral axis is read from `data/spectral_axes.csv` (one row, JSON wavelengths); the band range comes from its length, never from a constant. A single function translates three distinct in-range band indices to nm and is the only source of nm for every table, summary and figure.
- Hypercubes load from NPZ in **manifest order**, filtered by the requested identities (never sorted, never iterated from a set), with `allow_pickle=False`, key/shape/dtype checks, and a lineage check (crop, acquisition and class agree between manifest and partition rows). A missing identity is an error. Progress is printed every 100 files. Each hypercube keeps its `spectral_axis_id`.
- Loading with train rows only can never return a test identity.
- Per-channel normalization is fitted on train foreground pixels only (population std, float64), and a zero-variance channel raises. Applying it casts mean/std to float32 before subtracting/dividing and keeps background at zero.
- Model input preparation: normalize → resize → flips → rotation, applied to image and mask together, then background re-zeroed.
- The dataset draws augmentation decisions with `torch.rand` so DataLoader workers seeded from the generator reproduce them.
- `cnn/engine.py` is replaced by fig-aflatoxin's engine (AMP on CUDA, train_step/test_step/train, no early stopping).

Tests (synthetic 1×1×5 NPZ in `tmp_path`, no GPU): loader returns no test identities and rejects mismatched lineage; normalization uses only train foreground and zeroes background and fails on zero variance; augmentation keeps mask aligned and background zero.

## Done when

- [ ] `uv run pytest -q tests/test_protocol.py -k "load or normalization or augmentation"` passes.
- [ ] `uv run python -c "from cnn.data_setup import load_spectral_axis, wavelengths_of; import config; w = load_spectral_axis(config.SPECTRAL_AXES); print(len(w), wavelengths_of((366, 262, 225), w))"` prints `448 (891.0, 747.5, 697.1)` (rounded as the manifest states them).
- [ ] `uv run python -c "import cnn.engine; assert not hasattr(cnn.engine, 'EarlyStopping')"` succeeds.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

