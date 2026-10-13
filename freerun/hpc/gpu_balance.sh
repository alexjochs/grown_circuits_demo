#!/bin/bash
# Experiment: load balance with a few much busier circuits (closed_loop.py --workload mixed), launch queue in size
# order versus reordered by activity. Run inside hpc/job.sbatch.
set -uo pipefail
python bench/small_cuda_run.py 3 jitter || echo "CORRECTNESS FAILURE"
CL="python bench/closed_loop.py --backend cuda --steps 200"
for r in 0 16 1; do
  $CL --workload mixed --circuits 4096 --gates 1024 --reorder $r
  $CL --workload mixed --circuits 16384 --gates 256 --reorder $r
  $CL --workload random --circuits 4096 --gates 1024 --reorder $r
done
