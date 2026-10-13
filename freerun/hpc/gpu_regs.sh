#!/bin/bash
# Experiment: register caps of the warp-per-circuit kernel (more resident warps against spills). Run inside
# hpc/job.sbatch. The block kernels are capped in the source (FR_BLOCK_MAXNREG).
set -uo pipefail
for rr in 128 112; do
  cmake -S . -B build_wr$rr -DCMAKE_BUILD_TYPE=Release -DFREERUN_CUDA=ON -DFREERUN_LIBDIR=$PWD/lib_wr$rr \
    -DCMAKE_CUDA_FLAGS="-DFR_WARP_MAXNREG=$rr" > /dev/null && cmake --build build_wr$rr -j 16 > /dev/null
done
CL="python bench/closed_loop.py --backend cuda --steps 100"
for v in default wr128 wr112; do
  if [ $v = default ]; then unset FREERUN_LIBDIR; else export FREERUN_LIBDIR=$PWD/lib_$v; fi
  echo "=== variant $v"
  python bench/small_cuda_run.py 4 jitter || echo "CORRECTNESS FAILURE in $v"
  for j in 0 0.01; do
    $CL --jitter $j --circuits 16384 --gates 256
    $CL --jitter $j --circuits 4096 --gates 1024 --block-threshold 100000
  done
done
