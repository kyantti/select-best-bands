# 04 – Evaluate one band triplet from the command line

Status: ready-for-agent
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

- [ ] `uv run pytest -q tests/test_protocol.py -k seed` passes.
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python ga.py --evaluate 366 262 225` prints candidate seed `3104252108`, mean `[0.6560380893489065, 0.6596170752860457, 0.2233070946048868]`, std `[0.11694627746639438, 0.12472051938804612, 0.07551858819936108]`, weighted F1 `0.8700979843225085`, macro F1 `0.8763610130071496`, ordinal MAE `0.1875`, QWK `0.8711063372717508` (~3.5 min on the A100).
- [ ] The validation predictions it writes are identical to `runs/evaluate_selected_band_triplet/d49fc1f97dad04461e87c305c6e0a0bfff8a5e9b3b61d5f3ca6be6f7b4ced125/validation_predictions.csv` in `~/Documents/fig-aflatoxin` (`diff` on the id/actual/predicted columns is empty).
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python ga.py --evaluate 10 20 30 --epochs 1` finishes in under a minute and prints metrics.
- [ ] `SKIP_SYNC=1 ./init.sh` is green (it may still skip `import ga` until ticket 05).

## Evidence

_(none yet)_

## Comments

If mean/std match but the metrics do not, the divergence is in RNG order, loader configuration or library versions, not in the data path. Do not "fix" the risks listed in the plan (manifest order, 8 persistent workers, validation loader without generator, loaders before model, float32 mean/std).
