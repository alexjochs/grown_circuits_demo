#!/bin/bash
# Full compute-sanitizer racecheck reports (shared-memory hazards) over several small runs, run inside hpc/job.sbatch.
set -uo pipefail
for n in 3 6 8; do
  for mode in "" jitter; do
    echo "=== racecheck small_cuda_run $n $mode"
    compute-sanitizer --tool racecheck --racecheck-report all --error-exitcode 9 \
      python bench/small_cuda_run.py $n $mode 2>&1 | grep -v "^========= *$" | head -60
  done
done
