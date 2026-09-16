#!/bin/bash
# Standard startup + verification for select-best-bands.
# Usage: ./init.sh            (full: sync deps, compile, import, check data files)
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
# check_data.py is compiled but not imported: it still uses the loaders that
# protocolo-80-20/03 replaced, until ticket 10 rewrites it.
uv run python -m compileall -q config.py split_dataset.py ga.py train_final.py bootstrap.py check_data.py create_summary_table.py plot_fitness_evolution.py cnn

echo "=== import smoke test (config + scripts + cnn package) ==="
uv run python -c "import config, split_dataset, ga, train_final, bootstrap, plot_fitness_evolution, cnn.engine, cnn.data_setup, cnn.model; print('imports OK')"

echo "=== data files ==="
uv run python - <<'PY'
import config

expected = [
    ("dataset", config.HYPERCUBES_DIR),
    ("dataset", config.HYPERCUBES_MANIFEST),
    ("dataset", config.SPECTRAL_AXES),
    ("partition", config.PARTITIONS),
]
missing = set()
for kind, path in expected:
    print(f"{'ok  ' if path.exists() else 'MISS'} {path}")
    if not path.exists():
        missing.add(kind)
if "dataset" in missing:
    print("WARN: data/ is gitignored and incomplete here. Code work is fine, but no")
    print("      step that reads hypercubes can run on this machine.")
if "partition" in missing:
    print("WARN: no partition manifest yet. `split_dataset.py` writes it (ticket 02).")
PY

echo "=== GPU ==="
uv run python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

if [ -d tests ] && ls tests/test_*.py >/dev/null 2>&1; then
  echo "=== pytest ==="
  uv run pytest -q
fi

echo ""
echo "=== init OK ==="
echo "Next: read .scratch/*/issues/, pick ONE ready-for-agent ticket whose blockers are done, set it in-progress, work only on it."
