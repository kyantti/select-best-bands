# 14 – `pretrain.py`: SimCLR over band-triplet views, two checkpoints

Status: ready-for-agent
Blocked by: 13

## Description

`uv run python pretrain.py` pretrains a ResNet50 backbone with a SimCLR loop whose two views of a crop are two random band triplets of that same crop, and writes a checkpoint that ticket 13's injection point can load. Contrastive initialization is the only intervention in the fig-aflatoxin record that beat its own proposed criterion (+0.023 pooled, 8 of 8 candidates), rewritten here in this repo's shape — the original experiment-13 scripts are not ported and its checkpoints were deleted on 14 Sep.

Behaviour:

- Two views of a crop are two random band triplets drawn from the same crop, put through the same model-input preparation the network sees (normalize → resize → flips → rotation, image and mask together, background re-zeroed). NT-Xent loss, backbone only, no classification head.
- **Two checkpoints, never one**: `--cohort fit` pretrains on the 22 train-fit acquisitions and is the one the search uses; `--cohort train` pretrains on the 28 train acquisitions and is the one the final model uses. Neither may see a test acquisition.
- Each checkpoint is saved to `out/models/` (gitignored) next to a sidecar recording the acquisition identities it saw, the cohort name, the seed, the epochs and the library versions.
- Only train rows are ever loaded, the same way the search loads them, so a test crop cannot enter through the unsupervised path.

Tests (no GPU where possible): the saved cohort of a `fit` checkpoint contains no test and no validation acquisition; the cohort of a `train` checkpoint contains no test acquisition; the two view-drawing calls on one crop return two distinct in-range triplets of three distinct bands.

## Done when

- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python pretrain.py --cohort fit` writes `out/models/pretrain_fit.pt` and its sidecar listing exactly 22 acquisition identities (~3 min).
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python pretrain.py --cohort train` writes `out/models/pretrain_train.pt` and its sidecar listing exactly 28 acquisition identities (~3 min).
- [ ] `uv run pytest -q tests/test_protocol.py -k pretrain` passes (cohort exclusion for both checkpoints, distinct band-triplet views).
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python ga.py --evaluate 366 262 225 --checkpoint out/models/pretrain_fit.pt --epochs 1` runs to completion and prints metrics (correctness of the number is ticket 15's job, not this one's).
- [ ] Re-running the same cohort with the same seed produces a checkpoint with identical parameters.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

Two checkpoints is a protocol requirement, not an optimization: the search must never be initialized from a backbone that saw validation crops, or the fitness it optimizes is contaminated by the set it is scored on.
