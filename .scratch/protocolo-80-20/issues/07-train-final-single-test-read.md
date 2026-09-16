# 07 – Final model on all of train, one read of the test set

Status: done
Blocked by: 05

## Description

`uv run python train_final.py N [--bands R G B]` trains the final ResNet50 on every train crop (fit and validation, 28 acquisitions), saves the model, and only then loads the test crops and runs a single inference. Without `--bands` it reads the winner from `exp_NN_ga_summary.json`. For 366/262/225 it reproduces the 10 Sep test result.

Behaviour (copied from fig-aflatoxin's final stage):

- Load train rows only; fit normalization on them; seed everything with the final seed 2718; train 50 epochs calling only the training step (no evaluation loader exists during training).
- Save the state dict to `out/models/exp_NN_model_R_G_B.pt` (gitignored) **before** any test row is loaded.
- Then load test rows, predict once in manifest order, compute the metrics of ticket 04 plus per-class recall.
- Outputs: `exp_NN_final_metrics_R_G_B.json` (bands in indices and nm, test metrics, per-class recall, frozen mean/std, seed), `exp_NN_confusion_matrix_R_G_B.csv` and `.png` (seaborn, nm in title), `exp_NN_test_predictions_R_G_B.csv` (crop id, acquisition id, actual, predicted), `exp_NN_training_history_R_G_B.csv` and `.png` (loss and accuracy curves).
- `--epochs` override for smoke.

## Done when

- [x] `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 21 --bands 366 262 225` prints weighted F1 `0.722142952443074`, macro F1 `0.7223819971537002`, ordinal MAE `0.5`, QWK `0.5815011372251706`, confusion matrix `[[50,3,6,13],[6,43,9,10],[4,7,46,3],[4,3,3,46]]`, mean `[0.65546626556172, 0.6584057421588269, 0.22467686763531047]`, std `[0.11717051462839731, 0.125102268224151, 0.07519302019349307]` (~3.5 min).
- [x] `diff` of `out/tables/exp_21_test_predictions_366_262_225.csv` against `runs/train_final_model/bac0f8e3ee76f941d1cf7ee55ebd0fb859ef8cb28a9ad7e0c8c73a8af8abe4e2/test_predictions.csv` in `~/Documents/fig-aflatoxin` is empty on the id/actual/predicted columns.
- [x] `out/models/exp_21_model_366_262_225.pt` has a modification time earlier than the first test-set log line, and the script contains no evaluation call before the save (state this in Evidence).
- [x] `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 21` (no `--bands`) picks 366/262/225 from the summary written by ticket 06.
- [x] `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 99 --bands 10 20 30 --epochs 1` produces the five tables and two figures for experiment 99; then `rm out/*/exp_99_*`.
- [x] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

**Done (16 Sep).** `train_final.py` reproduces the 10 Sep test result bit for bit, and the
artifacts it writes are byte-identical to that run's.

1. `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 21 --bands 366 262 225` (5 min 19 s
   total, 43 s of it training; the rest is decompressing 1 124 NPZ crops) printed every value
   the ticket asked for:

   ```
   mean           [0.65546626556172, 0.6584057421588269, 0.22467686763531047]
   std            [0.11717051462839731, 0.125102268224151, 0.07519302019349307]
   weighted_f1    0.722142952443074
   macro_f1       0.7223819971537002
   ordinal_mae    0.5
   quadratic_weighted_kappa 0.5815011372251706
   confusion      [[50, 3, 6, 13], [6, 43, 9, 10], [4, 7, 46, 3], [4, 3, 3, 46]]
   ```

2. Against `runs/train_final_model/bac0f8e3.../` in `~/Documents/fig-aflatoxin`, `diff` is empty
   on **three** files, not just the id/actual/predicted columns asked for:

   - `exp_21_test_predictions_366_262_225.csv` — byte for byte, `acquisition_id` included
     (256 rows);
   - `exp_21_training_history_366_262_225.csv` — all 50 epochs of loss and accuracy, which
     means every gradient step of the run matched;
   - `exp_21_confusion_matrix_366_262_225.csv`.

3. **Nothing is evaluated before the save.** `train_final_model` calls `engine.train_step` and
   never `engine.train` (which takes a second loader and evaluates on it every epoch) — there is
   no evaluation call, and no test loader exists, until after `torch.save`. In a run logged with
   millisecond stamps and `PYTHONUNBUFFERED=1`:

   ```
   out/models/exp_21_model_366_262_225.pt   mtime 10:29:47.226731
   [10:29:47.288573] model          .../exp_21_model_366_262_225.pt (42.3 s)
   [10:29:47.288691] --- the test set is opened for the first time, after the save ---
   [10:29:47.296382] loading 256 test crops
   ```

   The state dict is on disk 69.7 ms before the first test-set line. The protocol test
   `test_final_saves_the_model_before_the_test_set_is_opened` watches `load_crops` and asserts
   `[("train", False), ("test", True)]` — the model file does not exist when the train crops are
   read and does exist when the test crops are; moving the test load before the save makes it
   fail (checked by mutating the source).

4. `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 21` with no `--bands` read
   `(366, 262, 225)` out of `exp_21_ga_summary.json` and wrote the same four tables with
   **identical checksums** (`md5sum -c` clean). Four runs in total, always the same bytes.

5. Smoke: `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 99 --bands 10 20 30 --epochs 1`
   wrote `exp_99_final_metrics_10_20_30.json`, `exp_99_confusion_matrix_10_20_30.csv`,
   `exp_99_test_predictions_10_20_30.csv` (257 lines), `exp_99_training_history_10_20_30.csv`,
   `out/models/exp_99_model_10_20_30.pt` and both figures, then `rm out/*/exp_99_*`. Note: that
   is **four** tables under `out/tables` plus the model, not the five the checkbox says; the
   Outputs list in the Description is what was built.

6. `SKIP_SYNC=1 ./init.sh` green, 71 tests (70 before the review fix, 62 before the ticket).
   `init.sh` now compiles and imports `train_final`, and `CLAUDE.md`'s Layout lists it.

**After `/code-review` (both axes agreed on one real defect).** The first version let
`train_final.py 21 --epochs 1` silently replace the 50-epoch `exp_21` tables, against
*"Never overwrite or delete existing `out/` results"*. Fixed with
`refuse_a_rerun_that_would_not_reproduce`, which compares the bands, seed, epochs, batch size,
learning rate, image size and device against the recorded `final_metrics` JSON before anything
is loaded: a faithful replay is allowed (that is how the record is re-verified), anything else
is refused in under a second.

```
$ uv run python train_final.py 21 --epochs 1
train_final.py: exp_21_final_metrics_366_262_225.json was recorded with a different epochs;
rewriting it would replace one result under out/ with another. Use a new experiment number.
$ uv run python train_final.py 20 --bands 124 155 322
train_final.py: .../exp_20_classification_report_124_155_322.csv belongs to an earlier run that
this script cannot reproduce, so experiment exp_20 is not free. Use a new experiment number.
```

Both checked against the real `out/`; the four `exp_21` checksums were unchanged afterwards.
Also from the review: the duplicated predictions writer is gone (`ga.write_predictions` grew an
`exclusive` keyword and owns the one schema, which ticket 08 reads), the evaluator's output is
length- and range-checked as the reference stage does, and `winner_bands` now checks the triplet
length its docstring promises.

## Comments

For triage, two things the review raised that are deliberately **not** fixed here, because they
are refactors outside this ticket:

- `train_final.py` imports `ga` for `select_device`, `library_versions`, `experiment_prefix` and
  `experiment_paths`. That is experiment infrastructure, not search: it would sit better in
  `config.py` or a small shared module, but moving it means touching `ga.py`, which tickets 05
  and 06 just verified bit for bit.
- The `exp_NN` namespace now has two "is this number free" rules with different semantics
  (`ga.refuse_to_reuse_an_old_experiment_number` and `refuse_a_number_an_earlier_protocol_owns`).
  Worth one rule when someone owns that decision.

The metrics JSON carries more fields than the Description listed (`schema_version`,
`evaluation_scope`, `test_feedback_used`, the hyperparameters, the crop and Acquisition counts,
library versions). That is on purpose: it mirrors `final_evaluation.json` of the reference run,
which the Description says to copy, and ticket 08 reads the band indices from it.

