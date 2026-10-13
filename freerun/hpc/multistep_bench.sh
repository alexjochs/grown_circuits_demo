#!/bin/bash
# Multi-step runs (SPEC §13) against the step loop, the grown search's access pattern: S = 256 steps of K = 10^3,
# 10^5, 10^7 ticks for 1034 and 4096 small circuits (bench/multistep.py). CUDA, then the CPU backend on 16 threads.
# Run inside hpc/job.sbatch (backends built):  hpc/submit.sh --job-name fr_msbench --time 01:30:00 -- bash hpc/multistep_bench.sh
set -uo pipefail
export PYTHONPATH=$PWD/python${PYTHONPATH:+:$PYTHONPATH}
MS="python bench/multistep.py --steps 256 --dt 1000 100000 10000000"
for B in 1034 4096; do
  echo "=== $B circuits, feed-forward (quiescent between input changes)"
  $MS --backend cuda --circuits $B --workload ff
  echo "=== $B circuits, cyclic with a budget of 1000 gate events per step"
  $MS --backend cuda --circuits $B --workload cyclic --budget 1000
  echo "=== $B circuits, cyclic, budget 1000, 1 % jitter"
  $MS --backend cuda --circuits $B --workload cyclic --budget 1000 --jitter 0.01 --repeat 2
done
for B in 1034 4096; do
  echo "=== CPU (16 threads), $B circuits"
  $MS --backend cpu --threads 16 --circuits $B --workload ff --repeat 2
  $MS --backend cpu --threads 16 --circuits $B --workload cyclic --budget 1000 --repeat 2
done
