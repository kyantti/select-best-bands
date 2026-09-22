#!/bin/bash
# The whole chain of one experiment number, in order and with its logs:
#
#   split_dataset.py -> ga.py N -> train_final.py N -> bootstrap.py N
#
# Experiments 1-20 belong to the earlier protocol and this chain refuses them;
# 21 is the replay of the 10 Sep 2026 search, so the first real run is 22.
# Numbers 90-99 are throwaway smoke runs: delete their out/ files afterwards.
#
# Usage:
#   ./run.sh 22                                                   a real run
#   ./run.sh 99 --population 4 --generations 1 --epochs 1          a smoke chain
#   nohup ./run.sh 22 > out/logs/run_22.log 2>&1 &                 hours, unattended
#
# Any flag after the number is routed to the steps that understand it:
#   --population, --generations, --no-evaluate   ga.py
#   --epochs                                     ga.py and train_final.py
#   --bands R G B                                train_final.py and bootstrap.py
#   --resamples, --seed                          bootstrap.py
#
# `--checkpoint` and `--seeds` are deliberately not routed; see the refusals
# below and the note in CLAUDE.md.
set -euo pipefail
cd "$(dirname "$0")"

# GPU 0 is usually busy with someone else's job; the result does not depend on
# which card runs it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [ $# -lt 1 ]; then
  echo "usage: ./run.sh <experiment_number> [flags for ga.py / train_final.py / bootstrap.py]" >&2
  exit 1
fi

EXPERIMENT="$1"
shift
# No leading zeros: the steps read the number as decimal and printf would read
# it as octal, so `022` and `18` would name the same run two different ways.
if ! [[ "$EXPERIMENT" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "run.sh: '$EXPERIMENT' is not an experiment number" >&2
  exit 1
fi

ga_args=()
final_args=()
bootstrap_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --population|--generations)
      ga_args+=("$1" "$2"); shift 2 ;;
    --epochs)
      ga_args+=("$1" "$2"); final_args+=("$1" "$2"); shift 2 ;;
    --no-evaluate)
      ga_args+=("$1"); shift ;;
    --bands)
      if [ $# -lt 4 ]; then
        echo "run.sh: --bands takes three band indices, as in --bands 366 262 225" >&2
        exit 1
      fi
      final_args+=("$1" "$2" "$3" "$4"); bootstrap_args+=("$1" "$2" "$3" "$4"); shift 4 ;;
    --resamples|--seed)
      bootstrap_args+=("$1" "$2"); shift 2 ;;
    *)
      echo "run.sh: no step takes '$1'; see the usage comment at the top of this file" >&2
      exit 1 ;;
  esac
done

# `--seeds` is not routed, and a multi-seed `config.FINAL_SEEDS` cannot go
# through this chain either: it ends in bootstrap.py, which resamples one set of
# test predictions, and a run of several seeds writes one set per seed.  Whether
# such a run has one interval per seed or one over the pooled predictions is an
# open protocol decision.  Refusing in a second beats dying at the last step of a
# 22-hour chain, so the count is read before the split.
FINAL_SEED_COUNT="$(uv run python -c 'import config; print(len(config.FINAL_SEEDS))')"
if [ "$FINAL_SEED_COUNT" -ne 1 ]; then
  echo "run.sh: config.FINAL_SEEDS names ${FINAL_SEED_COUNT} seeds, and this chain ends in" >&2
  echo "        bootstrap.py, which resamples one set of test predictions." >&2
  echo "        Run train_final.py by hand for a multi-seed final model." >&2
  exit 1
fi

mkdir -p out/logs

SEARCH_LOG="out/logs/experiment_${EXPERIMENT}.log"
FINAL_LOG="out/logs/final_${EXPERIMENT}.log"
BOOTSTRAP_LOG="out/logs/bootstrap_${EXPERIMENT}.log"

# `set -e` already stops the chain on a refusal, but the step that refused is
# only named in its own log, so say it here too.
step() {
  local name="$1" log="$2"
  shift 2
  echo "[$(date '+%F %T')] $name -> $log"
  if ! uv run python -u "$@" > "$log" 2>&1; then
    echo "run.sh: $name failed; the last lines of $log are:" >&2
    tail -n 20 "$log" >&2
    exit 1
  fi
}

echo "=== experiment ${EXPERIMENT} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="

# The split is deterministic (config.PARTITION_SEED), so rewriting it costs
# nothing and guarantees every step below reads the same partition.
echo "[$(date '+%F %T')] split_dataset.py"
uv run python -u split_dataset.py

step "ga.py ${EXPERIMENT}" "$SEARCH_LOG" ga.py "$EXPERIMENT" ${ga_args[@]+"${ga_args[@]}"}
step "train_final.py ${EXPERIMENT}" "$FINAL_LOG" train_final.py "$EXPERIMENT" \
  ${final_args[@]+"${final_args[@]}"}
step "bootstrap.py ${EXPERIMENT}" "$BOOTSTRAP_LOG" bootstrap.py "$EXPERIMENT" \
  ${bootstrap_args[@]+"${bootstrap_args[@]}"}

echo "=== experiment ${EXPERIMENT} done ==="
tail -n 4 "$BOOTSTRAP_LOG"
ls -1 out/tables/exp_"$(printf '%02d' "$EXPERIMENT")"_* out/figures/exp_"$(printf '%02d' "$EXPERIMENT")"_*
