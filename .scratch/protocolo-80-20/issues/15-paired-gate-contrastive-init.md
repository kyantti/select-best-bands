# 15 – The paid gate: paired control/treatment trial before the 22-hour run

Status: ready-for-agent
Blocked by: 12, 14

## Description

A short paired trial that decides whether contrastive initialization survives the protocol change, paid before the expensive run rather than after it. ~4.5 h of GPU, about 80 trainings.

Behaviour:

- A fixed panel of 8 candidates from the 10 Sep study: the winner plus 7 spread across the observed fitness range. The panel is written into `config.py` (or a versioned table) **before** the trial runs and is never revised afterwards.
- Two arms: control (ImageNet weights) and treatment (`out/models/pretrain_fit.pt`), 5 seeds each, all 80 trainings scored on the **crop-count-balanced** validation partition of ticket 12.
- The control has to be paid for: the experiment-9 CSVs were produced under the nested protocol and are not comparable.
- Decision rule, frozen before launch: continue if the pooled delta reaches **+0.010** and is positive for at least **6 of 8** candidates. Otherwise contrastive initialization is dropped and it is recorded that the experiment-13 gain does not survive the protocol change.
- No intermediate reads and no second panel: the trial is run once, the rule is applied once.
- Outputs: a per-(candidate, arm, seed) table, a per-candidate summary with the paired delta, and the pooled delta with the decision, under `out/tables/exp_NN_gate_*`.

## Done when

- [ ] The 8-candidate panel is committed before any training runs, and `## Evidence` names the commit or the file that fixed it.
- [ ] `CUDA_VISIBLE_DEVICES=1 uv run python gate.py NN` runs all 80 trainings, appending each result as it finishes so a crash does not lose the run, and is resumable the way the candidate cache is.
- [ ] It writes `out/tables/exp_NN_gate_runs.csv` (candidate, arm, seed, four metrics, seconds), `exp_NN_gate_summary.json` (per-candidate paired delta, pooled delta, how many of 8 are positive, the decision) and `out/figures/exp_NN_gate_deltas.png`.
- [ ] The summary states the decision by applying the frozen rule (+0.010 pooled and ≥ 6 of 8 positive), not by interpretation.
- [ ] A smoke run with `--epochs 1` and a 2-candidate panel completes end to end under experiment 99; then `rm out/*/exp_99_*`.
- [ ] `## Evidence` records the pooled delta, the per-candidate count, and whether contrastive initialization proceeds into the long run or is dropped.
- [ ] `SKIP_SYNC=1 ./init.sh` is green.

## Evidence

_(none yet)_

## Comments

The rule is frozen before the numbers exist on purpose: a gate that is reinterpreted after seeing its own result is not a gate. If the delta lands just under +0.010, the answer is "dropped", and that is a publishable finding — the experiment-13 gain did not survive the change of protocol.
