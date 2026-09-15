# 07 – Final model on all of train, one read of the test set

Status: ready-for-agent
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

- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 21 --bands 366 262 225` prints weighted F1 `0.722142952443074`, macro F1 `0.7223819971537002`, ordinal MAE `0.5`, QWK `0.5815011372251706`, confusion matrix `[[50,3,6,13],[6,43,9,10],[4,7,46,3],[4,3,3,46]]`, mean `[0.65546626556172, 0.6584057421588269, 0.22467686763531047]`, std `[0.11717051462839731, 0.125102268224151, 0.07519302019349307]` (~3.5 min).
- [ ] `diff` of `out/tables/exp_21_test_predictions_366_262_225.csv` against `runs/train_final_model/bac0f8e3ee76f941d1cf7ee55ebd0fb859ef8cb28a9ad7e0c8c73a8af8abe4e2/test_predictions.csv` in `~/Documents/fig-aflatoxin` is empty on the id/actual/predicted columns.
- [ ] `out/models/exp_21_model_366_262_225.pt` has a modification time earlier than the first test-set log line, and the script contains no evaluation call before the save (state this in Evidence).
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 21` (no `--bands`) picks 366/262/225 from the summary written by ticket 06.
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python train_final.py 99 --bands 10 20 30 --epochs 1` produces the five tables and two figures for experiment 99; then `rm out/*/exp_99_*`.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

