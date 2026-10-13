#!/bin/bash
# Throughput with and without per-transition jitter (SPEC §10), run inside hpc/job.sbatch:
#   mkdir -p logs && sbatch --job-name fr_jbench --time 00:30:00 hpc/job.sbatch hpc/jitter_bench.sh
set -uo pipefail
python bench/small_cuda_run.py 6 jitter || echo "CORRECTNESS FAILURE"
B="python bench/bench.py --backend cuda"
for j in 0 0.01; do
  $B --jitter $j --workload random --circuits 1024 --gates 1024 --T 100000
  $B --jitter $j --workload random --circuits 256 --gates 4096 --T 100000
  $B --jitter $j --workload rings --circuits 256 --gates 4096 --T 100000
  $B --jitter $j --workload random --circuits 64 --gates 65536 --T 50000
done
for j in 0 0.01; do
  python bench/bench.py --backend cpu --threads 16 --jitter $j --workload random --circuits 256 --gates 4096 --T 100000
done
