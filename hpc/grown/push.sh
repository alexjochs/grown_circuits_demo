#!/bin/bash
# Copy this checkout's code to a cluster login node and record its commit there (in .grown_commit).
#   [GROWN_REMOTE=submit-a] [GROWN_DEST=grown_circuits_demo] hpc/grown/push.sh
# Runs, logs, venvs, builds and caches are never copied. The remote tree is overwritten.
set -euo pipefail
cd "$(dirname "$0")/../.."
REMOTE=${GROWN_REMOTE:-submit-a}
DEST=${GROWN_DEST:-grown_circuits_demo}
rsync -a --delete --exclude /.venv --exclude /.work --exclude .git --exclude /runs --exclude "/logs/*.out" \
  --exclude "*.so" --exclude /freerun/build --exclude __pycache__ --exclude "*.egg-info" --exclude /.grown_commit \
  ./ "$REMOTE:$DEST/"
COMMIT=$(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo "+dirty")
ssh "$REMOTE" "echo $COMMIT > $DEST/.grown_commit"
echo "pushed $COMMIT to $REMOTE:$DEST"
