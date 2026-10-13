#!/bin/bash
# Submit a grid of configs x seeds, one job.sbatch each (one GPU; runs under $GROWN_RUNS, logs in logs/). Run on a
# login node:
#
#   hpc/grown/sweep.sh [-n] [-d JOBID] [-t TIME] "configs/grown/exp/e0_*.yaml" "0 1 2" [grown_search.py args]
#
#   -n         dry run: print the sbatch commands
#   -d JOBID   start only after JOBID succeeded
#   -t TIME    time limit per job (default: job.sbatch's 01:55:00)
#   -s         skip config/seed pairs whose run directory already has eval.json
#   -c N       at most N of these jobs at once: job k waits for job k-N to end (afterany), N chains in submission
#              order (default: no cap; slurm's GPU count is the only limit)
# Base configs (*_base.yaml) are skipped. Prints one "name seed jobid" line per submission.
set -euo pipefail
cd "$(dirname "$0")/../.."
GROWN_NO_MODULES=1 source hpc/grown/env.sh
DRY= DEP= TIME= SKIP= CAP=0
while getopts "nd:t:sc:" o; do
  case $o in n) DRY=1 ;; d) DEP="afterok:$OPTARG" ;; t) TIME="--time=$OPTARG" ;; s) SKIP=1 ;; c) CAP=$OPTARG ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
CONFIGS=$1 SEEDS=$2
shift 2
IDS=()
for cfg in $CONFIGS; do
  case $(basename "$cfg") in *_base.yaml) continue ;; esac
  name=$(basename "$cfg" .yaml)
  for seed in $SEEDS; do
    if [ -n "$SKIP" ] && [ -f "$GROWN_RUNS/${name}_s${seed}/eval.json" ]; then echo "$name $seed done"; continue; fi
    dep=$DEP
    n=${#IDS[@]}
    if [ "$CAP" -gt 0 ] && [ "$n" -ge "$CAP" ]; then dep=${dep:+$dep,}afterany:${IDS[$((n - CAP))]}; fi
    cmd=(sbatch --parsable --job-name "g_${name}_s${seed}" ${dep:+--dependency=$dep} $TIME hpc/grown/job.sbatch "$cfg" "$seed" "$@")
    if [ -n "$DRY" ]; then echo "${cmd[*]}"; IDS+=("DRY$n"); else id=$("${cmd[@]}"); IDS+=("$id"); echo "$name $seed $id"; fi
  done
done
