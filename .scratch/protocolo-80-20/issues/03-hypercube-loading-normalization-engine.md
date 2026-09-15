# 03 – Hypercube loading, train-only normalization, augmentation and training engine

Status: done
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

- [x] `uv run pytest -q tests/test_protocol.py -k "load or normalization or augmentation"` passes.
- [x] `uv run python -c "from cnn.data_setup import load_spectral_axis, wavelengths_of; import config; w = load_spectral_axis(config.SPECTRAL_AXES); print(len(w), wavelengths_of((366, 262, 225), w))"` prints `448 (891.02, 747.5, 697.05)` — the unrounded values, see the note below about this line.
- [x] `uv run python -c "import cnn.engine; assert not hasattr(cnn.engine, 'EarlyStopping')"` succeeds.
- [x] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

```
$ uv run pytest -q tests/test_protocol.py -k "load or normalization or augmentation"
25 passed, 17 deselected
$ uv run python -c "... load_spectral_axis(config.SPECTRAL_AXES) ... wavelengths_of((366, 262, 225), w)"
448 (891.02, 747.5, 697.05)
$ uv run python -c "import cnn.engine; assert not hasattr(cnn.engine, 'EarlyStopping')"
$ diff cnn/engine.py ~/Documents/fig-aflatoxin/src/fig_aflatoxin/modeling/engine.py
                       # empty: the engine is byte-identical to the original
$ SKIP_SYNC=1 ./init.sh
... imports OK / four data files present / 42 passed
=== init OK ===
```

**The data path was checked against the real run, not only against synthetic
crops.** Loading the 708 train-fit crops (2 min, ~15 GB of RAM) and fitting the
normalization for bands 366/262/225 gives, to the last digit, the values ticket
04 records for the 10 Sep run:

```
mean [0.6560380893489065, 0.6596170752860457, 0.2233070946048868]
std  [0.11694627746639438, 0.12472051938804612, 0.07551858819936108]
```

and `prepare_model_input` on a real crop returns a (3, 64, 128) float32 tensor
with the background exactly zero and augmentation reproducible from the seed.

**The `(891.0, 747.5, 697.1)` in the done-when line above is the thesis prose,
not the recorded output.** fig-aflatoxin's own artifacts under `runs/` contain
`891.02` (28 times) and `697.05` (119 times), and `891.0`/`697.1` zero times;
`round(697.05, 1)` is `697.0` in Python anyway, so no rounding rule produces the
line as written. `wavelengths_of` therefore returns the raw axis values, which is
what fig-aflatoxin's `SelectedBandTriplet.from_indices` does. Round at print
sites if the thesis wants one decimal.

Notes for the next tickets:

- Ticket 02's `load_cropped_hypercubes` (manifest rows) is now `load_manifest`,
  because the array loader arriving here is `load_hypercubes` and the two names
  sat one letter apart. `split_dataset.py` was updated with it.
- `check_data.py` no longer imports (`HypercubeDataset` is gone) and `ga.py`
  still calls the old dataloader helpers. Both are rewritten by tickets 10 and
  05; `init.sh` compiles them but does not import them.
- `spectral_axis_id(wavelengths)` gives the content address of the axis, so
  ticket 04 can check that the crops it loaded were cut against the axis whose
  wavelengths it is reporting, the way fig-aflatoxin's evaluate stage did.
- `mean`/`std` travel separately from `indices` (the original bundled them in a
  `NormalizationStatistics`), so nothing stops statistics fitted for one triplet
  being applied to another. Ticket 04 owns that pairing.
- `wavelengths_of` keeps the original's `type(index) is not int` check, so GA
  genes that are `numpy` integers must be cast before they reach it.

## Comments

