#!/bin/bash
# Performance record, run inside hpc/job.sbatch: the closed-loop benchmark (the first-class access pattern) and the
# bench.py throughput grid, without jitter and with 1 % jitter (the production configuration). Extra arguments go to
# every closed-loop run (e.g. --steps 200).
set -uo pipefail
python bench/small_cuda_run.py 3 jitter || echo "CORRECTNESS FAILURE"
for j in 0 0.01; do
  echo "=== closed loop, jitter $j: steps of 1 us, fresh inputs every step"
  CL="python bench/closed_loop.py --backend cuda --jitter $j"
  $CL --circuits 4096 --gates 1024 "$@"
  $CL --circuits 16384 --gates 256 "$@"
  $CL --circuits 1024 --gates 4096 "$@"
  $CL --circuits 4096 --gates 1024 --workload rings "$@"
done
for j in 0 0.01; do
  echo "=== bench.py grid, jitter $j"
  B="python bench/bench.py --backend cuda --jitter $j"
  $B --workload random --circuits 256 --gates 4096 --T 100000
  $B --workload rings --circuits 256 --gates 4096 --T 100000
  $B --workload random --circuits 1024 --gates 1024 --T 100000
  $B --workload random --circuits 8192 --gates 1024 --T 100000
  $B --workload random --circuits 64 --gates 65536 --T 50000
  $B --workload random --circuits 8 --gates 1048576 --T 10000
done
