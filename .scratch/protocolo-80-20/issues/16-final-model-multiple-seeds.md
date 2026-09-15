# 16 – Final model over a list of seeds, with the test-read count declared

Status: ready-for-agent
Blocked by: 09

## Description

`train_final.py` accepts a list of training seeds instead of one, so the noise of training can be told apart from the noise of the split. With a single-element list its output is identical to phase 1's, and the artifact always states how many times the test set was read.

Behaviour:

- `config.FINAL_SEEDS` defaults to `[2718]`. With one element, every file `train_final.py` writes is byte-for-byte the phase-1 output of ticket 07 — same names, same contents.
- With several seeds, each seed trains its own final model on all of train and writes its own predictions and metrics, suffixed by seed; a summary reports the mean and the spread of the metrics across seeds.
- Every output that reports a test result also reports the number of test reads, one per seed. The artifact can never claim a single read when there were ten.
- The order of operations of ticket 07 holds per seed: the model is saved before any test row is loaded.
- `--seeds` overrides the constant on the command line.

Test: with `FINAL_SEEDS = [2718]` the output of `train_final.py` is identical to the phase-1 output; the declared test-read count equals the number of seeds for any list.

## Done when

- [ ] `uv run python -c "import config; print(config.FINAL_SEEDS)"` prints `[2718]`.
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 21 --bands 366 262 225` still prints weighted F1 `0.722142952443074` and the confusion matrix of ticket 07, and its five tables and two figures `diff` empty against the phase-1 ones.
- [ ] `uv run pytest -q tests/test_protocol.py -k seeds` passes (single-seed identity, declared read count equals seed count).
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 99 --bands 10 20 30 --seeds 1 2 3 --epochs 1` writes per-seed predictions and metrics plus a summary with the mean and spread, and declares 3 test reads; then `rm out/*/exp_99_*` and `out/models/exp_99_*`.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

Open protocol decision (spec, 15 Sep — **Pablo's, not the agent's**): ten seeds read the test ten times, and that is only honest if nothing is selected on it — the seeds fixed in `config.py` before launching, the headline being the mean of the ten with its spread, and none discarded afterwards. The alternative is to run the ten on validation and not touch the test at all. Pablo decides which, and tells the professor **before** launching, not after seeing the number.

This ticket is `ready-for-agent` anyway because the decision binds the launch, not the code: the mechanics (seed list, per-seed outputs, declared read count) are the same either way, and `FINAL_SEEDS = [2718]` keeps today's behaviour until Pablo says otherwise. Do not launch a multi-seed run without his answer.
