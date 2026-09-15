# 01 – Environment bootstrap

Status: done
Blocked by: none

## Description

`uv sync` populates .venv (it is currently empty) and `./init.sh` passes compile + manifest checks on this machine.

## Done when

_(see Description)_

## Evidence

2026-09-15: `./init.sh` ran `uv sync` (installed torch 2.7.1 + deps into .venv), compileall passed, then stopped at the ga.py import with ModuleNotFoundError: cnn.engine2 (that is issue 02). Run individually: cnn.engine/data_setup/helper_functions import OK; train 897 rows / test 226 rows, 0 missing files, labels {0,1,2,3}; torch.cuda.is_available()=True, 4 devices.

## Comments

