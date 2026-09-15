# 02 – Restore ga.py imports after engine2 removal

Status: ready-for-agent
Blocked by: 01

## Description

Commit 7357da7 deleted cnn/engine2.py as unused, but ga.py still does `import cnn.engine2` and reads `classification_report`/`confusion_matrix` from its results. Either restore engine2 from `git show 7357da7^:cnn/engine2.py` or port its per-sample prediction collection (test_step returning y_preds/y_true, sklearn report + confusion matrix in train) into cnn/engine.py and change the import.

## Done when

`uv run python -c "import ga"` succeeds and `./init.sh` is green.

## Evidence

_(none yet)_

## Comments

