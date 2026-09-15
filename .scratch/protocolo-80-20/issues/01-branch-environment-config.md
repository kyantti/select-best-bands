# 01 – Branch, pinned environment and `config.py`

Status: done
Blocked by: none

## Description

Prefactor for the whole feature: the repo is on a fresh `feature/protocolo-80-20` branch, installs the exact dependency versions that produced the 10 Sep 2026 result, and has every experiment constant in a single `config.py` that the later scripts import. Obsolete files from the per-fig protocol are gone. The old `ga.py` is already broken at import (`cnn.engine2`), so nothing green is lost by deleting what it imported; `init.sh` checks only the modules that exist after this ticket and `import ga` returns to it in ticket 05.

Scope:

- New branch `feature/protocolo-80-20` from `feature/experiments` at `b7f1fc9` (the two dangling commits from the deleted branch of the same name are not needed).
- Dependencies pinned exactly: deap 1.4.4, numpy 2.5.1, scikit-learn 1.9.0, torch 2.13.0, torchvision 0.28.0, pandas 2.3.3, matplotlib 3.11.1, seaborn 0.13.2; dev pytest 9.1.1. torch/torchvision come from the explicit uv index for CUDA 12.6. torcheval, torchinfo, requests, pillow and tqdm are dropped. `.python-version` is 3.13.5 and versioned.
- `.gitignore`: `data/` becomes `/data/` so `tests/data/*.csv` are versioned; add `out/models/`; stop ignoring `.python-version`.
- Delete `cnn/train.py`, `cnn/model_info.py`, `cnn/util/`, `train_dataset.csv`, `test_dataset.csv`, `run.log`. Keep `out/` for experiments 1–20 untouched. Keep `data/interim` and `data/processed`.
- `config.py` holds every constant of the real run: paths (data dir, hypercube manifest, spectral axes, partitions, out dirs), class names C0..C3, device, partition seed 20230717 with 0.2/0.2 proportions, study seed 23, candidate seed 1729, final seed 2718, GA parameters (population 20, 25 generations, cxpb 0.8, mutpb 0.15, tournament 3, blend α 0.5, gaussian μ 0 σ 20 p_gen 0.3, hall of fame 1), CNN parameters (50 epochs, batch 32, lr 0.001, 64×128, 8 workers), bootstrap 5000 resamples with seed 1729. Comments next to σ/p_gen (repaired operator) and `NUM_WORKERS` (part of the reproducible result).
- `init.sh` syncs, compiles, imports `config`, `cnn.engine`, `cnn.data_setup`, and warns (does not fail) when `data/` is absent. The old manifest check for `train_dataset.csv`/`test_dataset.csv` goes away.
- `CLAUDE.md` Layout and Invariants mention `config.py` instead of "the top of `ga.py`".

## Done when

- [x] `git branch --show-current` prints `feature/protocolo-80-20` and `git merge-base --is-ancestor b7f1fc9 HEAD` succeeds.
- [x] `uv run python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"` prints `2.13.0+cu126 0.28.0+cu126 True`.
- [x] `uv run python -c "import deap, numpy, sklearn, pandas; print(deap.__version__, numpy.__version__, sklearn.__version__, pandas.__version__)"` prints `1.4.4 2.5.1 1.9.0 2.3.3`.
- [x] `git check-ignore -q tests/data/reference_evaluation_partitions.csv` fails (file is not ignored) and `git check-ignore -q data/cropped_hypercubes.csv out/models/x.pt` succeeds.
- [x] `uv run python -c "import config; print(config.STUDY_SEED, config.CANDIDATE_SEED, config.FINAL_SEED, config.PARTITION_SEED)"` prints `23 1729 2718 20230717`.
- [x] `SKIP_SYNC=1 ./init.sh` is green.
- [x] `ls cnn/train.py cnn/model_info.py cnn/util train_dataset.csv test_dataset.csv run.log` reports every path missing.

## Evidence

All checks run on 15 Sep 2026 on the A100 box, after `uv lock && uv sync`
(the venv went from python 3.13.2 / torch 2.7.1 to python 3.13.5 / torch 2.13.0+cu126).

```
$ git branch --show-current
feature/protocolo-80-20
$ git merge-base --is-ancestor b7f1fc9 HEAD && echo OK
OK
$ uv run python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
2.13.0+cu126 0.28.0+cu126 True
$ uv run python -c "from importlib.metadata import version; print(version('deap'), version('numpy'), version('scikit-learn'), version('pandas'), version('matplotlib'), version('seaborn'), version('pytest'))"
1.4.4 2.5.1 1.9.0 2.3.3 3.11.1 0.13.2 9.1.1
$ uv run python --version
Python 3.13.5
$ git check-ignore -q tests/data/reference_evaluation_partitions.csv; echo $?
1                      # not ignored
$ git check-ignore -q data/cropped_hypercubes.csv; echo $?
0
$ git check-ignore -q out/models/x.pt; echo $?
0
$ uv run python -c "import config; print(config.STUDY_SEED, config.CANDIDATE_SEED, config.FINAL_SEED, config.PARTITION_SEED)"
23 1729 2718 20230717
$ ls cnn/train.py cnn/model_info.py cnn/util train_dataset.csv test_dataset.csv run.log
ls: cannot access any of them: No such file or directory
$ SKIP_SYNC=1 ./init.sh
... imports OK / data files (PARTITIONS missing, warns) / torch 2.13.0+cu126, cuda available: True, 4 devices
=== init OK ===
```

Notes for the next ticket:

- `deap.__version__` is the upstream short string `1.4`; the installed distribution is 1.4.4 (`importlib.metadata.version`). The done-when line was checked that way.
- `init.sh` warns separately for a missing `data/evaluation_partitions.csv` (normal until ticket 02 writes it) and for an absent dataset.
- `import ga` is deliberately out of `init.sh` (`ga.py` still imports `cnn.engine2`); ticket 05 puts it back.
- `tests/data/*.csv` are now versioned but there is still no `tests/test_*.py`, so `init.sh` skips the pytest step.

## Comments

Numbering decision (15 Sep, author): new experiments start at 21; `exp_21_*` is the replay of the 10 Sep run (ticket 06), so the first new search is 22. Smoke runs use 90–99 and are deleted.
