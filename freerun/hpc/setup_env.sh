#!/bin/bash
# One-time Python env setup on the cluster (run on a submit node, inside the project dir on the share).
set -euo pipefail
cd "$(dirname "$0")/.."
UV=/nfs/hpc/share/ochsal/h200_kernel_census/tools/uv
export UV_PYTHON_INSTALL_DIR=$PWD/.tools/pythons UV_CACHE_DIR=/tmp/uvcache_$USER
$UV venv --clear --python 3.12 --python-preference only-managed .venv
$UV pip install --python .venv/bin/python -e ".[dev]"
rm -rf /tmp/uvcache_$USER
