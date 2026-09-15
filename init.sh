#!/bin/bash
# Standard startup + verification for select-best-bands.
# Usage: ./init.sh            (full: sync deps, compile, import, check manifests)
#        SKIP_SYNC=1 ./init.sh (skip `uv sync` when deps are already installed)
set -e
cd "$(dirname "$0")"

echo "=== select-best-bands: init ==="

if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [ "${SKIP_SYNC:-0}" != "1" ]; then
  echo "=== uv sync (first run downloads torch + CUDA, several GB) ==="
  uv sync
fi

echo "=== syntax check ==="
uv run python -m compileall -q ga.py check_data.py create_summary_table.py plot_fitness_evolution.py cnn

echo "=== import smoke test (ga.py + cnn package) ==="
uv run python -c "import ga, cnn.engine, cnn.data_setup, cnn.util.helper_functions; print('imports OK')"

echo "=== dataset manifests ==="
uv run python - <<'PY'
import csv, os, sys
bad = 0
for name in ("train_dataset.csv", "test_dataset.csv"):
    rows = list(csv.DictReader(open(name)))
    missing = [r["filepath"] for r in rows if not os.path.exists(r["filepath"])]
    labels = {r["label"] for r in rows}
    ok_labels = labels <= {"0", "1", "2", "3"}
    print(f"{name}: {len(rows)} rows, {len(missing)} missing files, labels={sorted(labels)}")
    if not ok_labels:
        print(f"FAIL: unexpected labels in {name}: {sorted(labels - {'0','1','2','3'})}")
        bad = 1
    if missing:
        print(f"WARN: data/ not present for {name} (gitignored). Code work is fine; ga.py cannot run here.")
if bad:
    sys.exit(1)
PY

echo "=== GPU ==="
uv run python -c "import torch; print('cuda available:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

if [ -d tests ] && ls tests/test_*.py >/dev/null 2>&1; then
  echo "=== pytest ==="
  uv run pytest -q
fi

echo ""
echo "=== init OK ==="
echo "Next: read feature_list.json, pick ONE ready-for-agent feature whose dependencies are done, set it in-progress, work only on it."
