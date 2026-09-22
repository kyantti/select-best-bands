# 16 – Final model over a list of seeds, with the test-read count declared

Status: done
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

- [x] `uv run python -c "import config; print(config.FINAL_SEEDS)"` prints `[2718]`.
- [x] `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 21 --bands 366 262 225` still prints weighted F1 `0.722142952443074` and the confusion matrix of ticket 07, and its five tables and two figures `diff` empty against the phase-1 ones.
- [x] `uv run pytest -q tests/test_protocol.py -k seeds` passes (single-seed identity, declared read count equals seed count).
- [x] `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 99 --bands 10 20 30 --seeds 1 2 3 --epochs 1` writes per-seed predictions and metrics plus a summary with the mean and spread, and declares 3 test reads; then `rm out/*/exp_99_*` and `out/models/exp_99_*`.
- [x] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

**Done (23 Sep).** `config.FINAL_SEED` became `config.FINAL_SEEDS = [2718]`, and
`train_final.py` trains one final model per seed. Nothing launched with several
seeds: that decision is Pablo's and is still open (see `## Comments`).

1. **The default is the 10 Sep run, unchanged.**
   `uv run python -c "import config; print(config.FINAL_SEEDS)"` → `[2718]`.
   `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 21 --bands 366 262 225`
   (49.7 s of training) printed the ticket-07 values exactly:

   ```
   weighted_f1    0.722142952443074
   macro_f1       0.7223819971537002
   ordinal_mae    0.5
   quadratic_weighted_kappa 0.5815011372251706
   confusion      [[50, 3, 6, 13], [6, 43, 9, 10], [4, 7, 46, 3], [4, 3, 3, 46]]
   ```

   `diff` against copies of the phase-1 artifacts taken before the run is empty on
   all four tables (`final_metrics`, `confusion_matrix`, `test_predictions`,
   `training_history`), and `git status --short out/` is empty afterwards — the two
   PNGs are unchanged too. The single-seed path writes the same names and the same
   bytes it wrote before this ticket.

2. **Several seeds, and the count is declared.**
   `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 99 --bands 10 20 30 --seeds 1 2 3 --epochs 1`
   wrote 3 models, 3×4 tables, 3×2 figures and one summary. Every one of the four
   metrics files says `"test_evaluation_count": 3` and `"final_seeds": [1, 2, 3]` —
   including each per-seed file, so no artifact of the run can be quoted as a single
   held-out read. The summary gave (1 epoch, so the numbers are noise by design):

   ```
   weighted_f1    mean 0.24981646251451953  std 0.005737513706506216
                  min 0.24476855880801524   max 0.25605633480014667
   ```

   Deleted afterwards: `rm -f out/tables/exp_99_* out/figures/exp_99_* out/models/exp_99_* out/logs/*_99.log`,
   `git status --short out/` empty.

3. **The order of ticket 07 holds for the last seed as strictly as for the first.**
   All three models are trained and saved *before* the test partition is opened —
   the log shows the three `model …_seed{1,2,3}.pt` lines, then
   `--- the test set is opened for the first time, after the save ---`, then the
   256 test crops. `test_final_seeds_save_every_model_before_the_test_set_is_opened`
   watches this: at the `test` load, all three model files already exist.

4. **The refusals fire before anything is trained**, against the real files:

   ```
   $ … train_final.py 99 --bands 10 20 30 --seeds 1 2 --epochs 1
   train_final.py: exp_99_final_metrics_10_20_30_seed1.json was recorded with a
   different test_evaluation_count; rewriting it would replace one result under
   out/ with another. Use a new experiment number.
   $ … train_final.py 99 --bands 10 20 30 --seeds 1 1 --epochs 1
   train_final.py: the final seeds [1, 1] repeat a seed; each one trains one model
   ```

   `test_evaluation_count` joined `RECORDED_SETTINGS` for this: rewriting seed 1's
   file under a shorter list would turn three reads of the held-out set into two.

5. **One experiment number is one claim.** A multi-seed run beside an unsuffixed
   result would not overwrite it — the names differ, so the settings guard never
   sees them — but the number would then carry two answers about the same crops,
   one saying it read the test once and the others three times. Refused in both
   directions (`refuse_a_number_that_already_holds_another_seeding`).

6. `uv run pytest -q tests/test_protocol.py -k seeds` → **23 passed**;
   `uv run pytest -q` → **193 passed** (was 169); full `./init.sh` green.

### What `/code-review high` found, and what was done

- **The chain could have died at the last step of a 22-hour run.** `bootstrap.py`
  and `analyze_test_errors.py` resolve their inputs through
  `final_paths(experiment, bands)` with the new `seed` at its `None` default, so a
  multi-seed run left them reading a file nobody wrote. Setting
  `config.FINAL_SEEDS = [a, b, c]` and launching `./run.sh N` would have run the
  search and every final model and *then* failed. `run.sh` now reads the seed count
  before the split and refuses in a second:

  ```
  run.sh: config.FINAL_SEEDS names 2 seeds, and this chain ends in
          bootstrap.py, which resamples one set of test predictions.
          Run train_final.py by hand for a multi-seed final model.
  ```

  `bootstrap.py` refuses with the names of the per-seed files that are actually
  there instead of "run train_final.py first", and `published_triplets` no longer
  lists one triplet three times because three seeds named it.
- **`min`/`max` answered by where a NaN sat in the list.** `spread_over_seeds` went
  out of its way to carry an undefined kappa through `mean`/`std`, but `min`/`max`
  do not propagate NaN, so a run whose second seed collapsed to one class would
  have quoted the other two as if all three had scored — and the same seeds in
  another order would have written other bytes. One undefined seed now makes that
  metric's whole summary undefined.
- **`--seeds` was undocumented as unrouted.** `run.sh`'s usage comment and the
  `CLAUDE.md` `run.sh` bullet now say so, beside `--checkpoint`.

**Still open, deliberately:** whether a multi-seed run has one interval per seed or
one over the pooled predictions is a protocol decision, not a rename, so
`bootstrap.py` refuses rather than guesses. A multi-seed final model is
`train_final.py` by hand today.

## Comments

Open protocol decision (spec, 15 Sep — **Pablo's, not the agent's**): ten seeds read the test ten times, and that is only honest if nothing is selected on it — the seeds fixed in `config.py` before launching, the headline being the mean of the ten with its spread, and none discarded afterwards. The alternative is to run the ten on validation and not touch the test at all. Pablo decides which, and tells the professor **before** launching, not after seeing the number.

This ticket is `ready-for-agent` anyway because the decision binds the launch, not the code: the mechanics (seed list, per-seed outputs, declared read count) are the same either way, and `FINAL_SEEDS = [2718]` keeps today's behaviour until Pablo says otherwise. Do not launch a multi-seed run without his answer.
