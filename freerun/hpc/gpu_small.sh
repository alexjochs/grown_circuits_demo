#!/bin/bash
# The grown-search regime: hundreds to a few thousand circuits of a few hundred gates, short control steps
# (10^3-10^4 ticks with gate delays of 30-120 ticks), fresh inputs every step. Per-call overhead dominates here.
# CUDA and the CPU backend (16 threads), stepping API and the pre-stepping API. Run inside hpc/job.sbatch.
set -uo pipefail
CL="python bench/closed_loop.py --gates 256 --delay 30 120 --steps 1000"
for dt in 1000 10000; do
  for B in 256 1024 4096; do
    echo "=== $B circuits x 256 gates, steps of $dt ticks"
    for j in 0 0.01; do
      $CL --backend cuda --circuits $B --dt $dt --jitter $j
      $CL --backend cpu --threads 16 --circuits $B --dt $dt --jitter $j
    done
    $CL --backend cuda --circuits $B --dt $dt --legacy --steps 200
    $CL --backend cpu --threads 16 --circuits $B --dt $dt --legacy --steps 200
  done
done
