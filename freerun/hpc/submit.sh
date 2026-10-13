#!/bin/bash
# Submit a GPU job for the checkout this script lives in (run it on a submit node).
# Usage: hpc/submit.sh [sbatch options] -- <command...>
# Logs go to <checkout>/logs/<job-name>_<job-id>.out, so several checkouts on the share do not collide.
set -euo pipefail
DIR=$(cd "$(dirname "$0")/.." && pwd)
opts=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done
if [ "${1:-}" = "--" ]; then shift; fi
if [ $# -eq 0 ]; then echo "usage: hpc/submit.sh [sbatch options] -- <command...>" >&2; exit 2; fi
mkdir -p "$DIR/logs"
cd "$DIR"
exec sbatch --export=ALL,FREERUN_DIR="$DIR" --output="$DIR/logs/%x_%j.out" --error="$DIR/logs/%x_%j.out" \
  "${opts[@]}" hpc/job.sbatch "$@"
