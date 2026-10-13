#!/bin/bash
# Create (or refresh) the Python environment at $GROWN_VENV: the vendored freerun package and this repository, both
# editable. Run once, on a login node or locally:
#   [GROWN_SHARE=/big/fs/dir] [UV=/path/to/uv] bash hpc/grown/setup_env.sh
# With uv (found on PATH or given as $UV) the venv gets a uv-managed Python 3.12; otherwise $PYTHON (default python3,
# 3.11 or newer) is used. GROWN_EXTRAS (default "logs") picks the optional dependencies: add "cuda" for jax on GPU,
# which only the learned editor's training uses.
set -euo pipefail
GROWN_NO_MODULES=1 source "$(dirname "$0")/env.sh"
UV=${UV:-$(command -v uv || true)}
EXTRAS=${GROWN_EXTRAS:-logs}
if [ -n "$UV" ]; then
  export UV_PYTHON_INSTALL_DIR=$GROWN_SHARE/.tools/pythons UV_CACHE_DIR=$GROWN_SHARE/.cache/uv
  "$UV" venv --clear --python 3.12 --python-preference only-managed "$GROWN_VENV"
  "$UV" pip install --python "$PY" -e "$GROWN_ROOT/freerun" -e "$GROWN_ROOT[$EXTRAS]"
else
  "${PYTHON:-python3}" -m venv --clear "$GROWN_VENV"
  "$PY" -m pip install -q -e "$GROWN_ROOT/freerun" -e "$GROWN_ROOT[$EXTRAS]"
fi
"$PY" -c "import jax, numpy, freerun, grown; print('jax', jax.__version__, 'numpy', numpy.__version__, 'freerun', freerun.__version__)"
du -sh "$GROWN_VENV" 2>/dev/null || true
