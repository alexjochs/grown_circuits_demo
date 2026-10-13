#!/bin/bash
# GPU validation, run inside hpc/job.sbatch (backends already built): sanitizers on every CUDA kernel and the stepping
# API (including multi-step runs), without and with per-transition jitter, then the differential suites against the
# reference. This checkout's package comes first on the path (a shared venv may have another checkout installed).
set -uo pipefail
export PYTHONPATH=$PWD/python${PYTHONPATH:+:$PYTHONPATH}
for mode in "" jitter; do
  echo "=== sanitizers on all CUDA kernels and the stepping API ${mode:-(no jitter)}"
  for tool in memcheck racecheck synccheck; do
    echo "--- $tool"
    compute-sanitizer --tool $tool --error-exitcode 9 python bench/small_cuda_run.py 3 $mode 2>&1 | tail -3
    echo "exit code ${PIPESTATUS[0]}"
  done
done
echo "=== differential tests: cpu and every cuda variant vs the reference (with and without jitter)"
timeout 3300 python -m pytest -q -n 8 tests/test_backends.py tests/test_edge_cases.py tests/test_stepping.py tests/test_multistep.py \
  tests/test_jitter.py tests/test_energy.py tests/test_variation.py 2>&1 | tail -15
