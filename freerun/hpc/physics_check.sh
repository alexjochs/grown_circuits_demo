#!/bin/bash
# CUDA conformance for the physics extensions (jitter), run inside hpc/job.sbatch:
#   mkdir -p logs && sbatch --job-name fr_phys hpc/job.sbatch hpc/physics_check.sh
set -uo pipefail
CS=compute-sanitizer
for mode in "" jitter; do
  echo "=== compute-sanitizer ${mode:-plain}: memcheck, racecheck, synccheck"
  $CS --tool memcheck --error-exitcode 9 python bench/small_cuda_run.py 4 $mode 2>&1 | tail -3
  $CS --tool racecheck --error-exitcode 9 python bench/small_cuda_run.py 2 $mode 2>&1 | tail -3
  $CS --tool synccheck --error-exitcode 9 python bench/small_cuda_run.py 2 $mode 2>&1 | tail -3
done
echo "=== differential tests: cpu and every cuda kernel vs the reference (with and without jitter)"
TESTS=$(ls tests/test_backends.py tests/test_edge_cases.py tests/test_jitter.py tests/test_energy.py \
  tests/test_variation.py 2>/dev/null)
timeout 2700 python -m pytest -q -n 8 $TESTS 2>&1 | tail -15
