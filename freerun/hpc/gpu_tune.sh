#!/bin/bash
# Tuning sweep for the windowed kernel, run inside hpc/job.sbatch: group size per circuit size, and the value of each
# lever (shared-memory state, packed layout). Extra arguments go to every closed-loop run.
set -uo pipefail
CL="python bench/closed_loop.py --backend cuda --steps 100"
echo "=== 4096 x 1024: group size and levers"
for v in "" "--block-threshold 100000" "--block-threshold 0 --block-threads 64" \
         "--block-threshold 0 --block-threads 128" "--no-smem" "--no-pack" "--no-smem --no-pack"; do
  $CL --circuits 4096 --gates 1024 $v "$@"
done
echo "=== 1024 x 4096"
for v in "" "--block-threshold 100000" "--block-threshold 0 --block-threads 64" \
         "--block-threshold 0 --block-threads 128" "--block-threshold 0 --block-threads 256" \
         "--block-threshold 0 --block-threads 512"; do
  $CL --circuits 1024 --gates 4096 $v "$@"
done
echo "=== 16384 x 256"
for v in "" "--block-threshold 0 --block-threads 64"; do
  $CL --circuits 16384 --gates 256 $v "$@"
done
echo "=== 512 x 16384 (state in global memory)"
for v in "" "--block-threshold 0 --block-threads 128" "--block-threshold 0 --block-threads 256" \
         "--block-threshold 0 --block-threads 512"; do
  $CL --circuits 512 --gates 16384 $v "$@"
done
echo "=== large circuits"
B="python bench/bench.py --backend cuda"
$B --workload random --circuits 64 --gates 65536 --T 50000
$B --workload random --circuits 8 --gates 1048576 --T 10000
