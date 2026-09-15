# 04 – Pytest smoke tests

Status: ready-for-agent
Blocked by: 01

## Description

Add pytest as a dev dependency and tests that need no GPU and no data/: cx_blend_clamped and mut_gaussian_clamped keep bands in [START_BAND, END_BAND]; HypercubeDataset maps a tiny synthetic (h,w,448) .npy to a 3-channel image; train/test manifest rows parse and have labels in {0,1,2,3}. Wire `uv run pytest` into init.sh.

## Done when

init.sh runs the suite green.

## Evidence

_(none yet)_

## Comments

