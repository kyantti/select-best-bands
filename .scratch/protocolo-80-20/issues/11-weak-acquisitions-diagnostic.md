# 11 – Diagnose the two weak test acquisitions before touching the model

Status: ready-for-agent
Blocked by: 08

## Description

First ticket of **phase 2**, and the one that goes first of all of them: a short script that answers whether the two weakest test acquisitions are a data problem before any improvement tunes the model against one. Zero GPU.

Of the eight test acquisitions, two sit at 0.689 weighted F1 against 0.857–0.951 for the other six: one is entirely C0, the other entirely C1. Seventeen of the errors are C0↔C3 confusions, which are the extreme classes and should be the easy ones to tell apart.

The script crosses the per-crop test predictions of an experiment with the partition manifest and the cropped-hypercube manifest, and reports:

- Errors per acquisition and per (actual, predicted) class pair, so the C0↔C3 block is visible and it is clear whether those 17 confusions are concentrated in the two weak acquisitions or spread across all eight.
- Whether the two weak acquisitions differ from the other six in anything measurable from the crops themselves: mean and spread of foreground reflectance per band range (exposure/illumination), foreground pixel fraction and crop area (segmentation quality), crop count.
- A crop grid of the misclassified crops from the two weak acquisitions so their images can be looked at.

The output is a table plus a figure; the conclusion (data problem or not) goes in this ticket's `## Evidence` in prose. Nothing in `config.py` changes and no other module is touched.

## Done when

- [ ] `uv run python analyze_test_errors.py 21` prints the per-acquisition weighted F1 (matching the per-acquisition table written by `bootstrap.py`), the confusion breakdown per acquisition, and the exposure / foreground-fraction comparison of the two weak acquisitions against the other six.
- [ ] It writes `out/tables/exp_21_test_error_analysis.csv` and `out/figures/exp_21_test_errors.png` (misclassified crops of the two weak acquisitions, actual and predicted class in each panel title, bands in nm).
- [ ] Running it twice produces identical CSV.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.
- [ ] `## Evidence` states, in one paragraph, whether the two acquisitions look broken (exposure, illumination, segmentation or labels) or simply hard, and therefore whether phase 2 proceeds as planned.

## Evidence

_(none yet)_

## Comments

Phase-2 ordering (spec, 15 Sep): this ticket is 0 h of GPU and gates nothing technically, but it goes first deliberately — if the labels or the crops of those two acquisitions are wrong, every later improvement is tuning a model against a data problem.
