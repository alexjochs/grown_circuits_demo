#!/bin/bash
# Sync the working tree (code only) to the cluster share. Usage: [FREERUN_REMOTE=host] [FREERUN_DEST=dir] hpc/push.sh
# Patterns starting with / are anchored at the project root, so scripts/build.sh is not mistaken for a build dir.
# Also records the current commit (with "+dirty" for uncommitted changes) in COMMIT there, for the job logs.
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=${FREERUN_REMOTE:-osu-hpc}
DEST=${FREERUN_DEST:-/nfs/hpc/share/ochsal/freerun}
rev=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
git diff --quiet HEAD 2>/dev/null || rev="$rev+dirty"
rsync -a --delete --exclude /.venv --exclude /.tools --exclude /tools --exclude .git --exclude "/build*" \
  --exclude "/lib_*" --exclude /logs --exclude /runs --exclude "*.so" --exclude .pytest_cache --exclude __pycache__ \
  --exclude "*.egg-info" --exclude /COMMIT ./ "$REMOTE:$DEST/"
ssh "$REMOTE" "echo $rev > $DEST/COMMIT"
