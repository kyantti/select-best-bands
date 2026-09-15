# 13 – Optional backbone checkpoint, inert when unset

Status: ready-for-agent
Blocked by: 09

## Description

`cnn/model.py` learns to load an optional pretrained backbone before the usual fine-tuning, and proves that with no checkpoint the code path is exactly today's. This ticket is the licence to touch the evaluator at all: the delta-zero test is what guarantees the injection point cannot move the published number by itself.

Behaviour:

- `config.BACKBONE_CHECKPOINT` defaults to `None`. With `None`, the model is built exactly as in ticket 04 (ResNet50 with ImageNet weights, 4-class head) and no new call is made — the same ops in the same order, so the RNG stream is untouched.
- With a path, the checkpoint's `state_dict` is loaded into the backbone after the ImageNet weights and before fine-tuning; the head is never loaded from it. A checkpoint whose keys or shapes do not match the backbone is an error, not a silent partial load. `--checkpoint` overrides the constant on `ga.py --evaluate` and `train_final.py`.
- The checkpoint identity (path plus the SHA-256 of its bytes, or `null`) joins the fitness contract of the candidate cache, so candidates evaluated with and without a checkpoint can never be mixed in one CSV.
- The candidate seed derivation does **not** change: a checkpoint changes the fitness contract, not the seed, so 366/262/225 still derives 3104252108.

Tests: without a checkpoint, the built model's parameters are identical to the one built before this ticket (compare against a freshly constructed reference); a mismatched checkpoint raises; a cache written with one checkpoint identity is refused when loaded under another.

## Done when

- [ ] `uv run python -c "import config; print(config.BACKBONE_CHECKPOINT)"` prints `None`.
- [ ] `uv run pytest -q tests/test_protocol.py -k checkpoint` passes.
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python ga.py --evaluate 366 262 225` still prints candidate seed `3104252108` and weighted F1 `0.8700979843225085`, and the run reports the delta against the recorded value as exactly `0.00e+00`.
- [ ] A cache written with `BACKBONE_CHECKPOINT=None` aborts with a fitness-contract mismatch naming the checkpoint when reloaded with `--checkpoint` set (record the command).
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

This is the only phase-2 ticket that edits the evaluator. The `0.00e+00` check is not a formality: if it is not exactly zero, the injection point has perturbed the RNG order or the construction sequence and the port stops being verifiable. Do not proceed to ticket 14 until it is.
