#!/bin/bash
# The multi-step merge gate on the GPU, inside hpc/job.sbatch (backends built): physical-K agreement (CUDA variants ==
# CPU == reference), the sanitizers and differential suites (gpu_check.sh), then the throughput bench.
#   hpc/submit.sh --job-name fr_ms_validate --time 02:30:00 -- bash hpc/ms_validate.sh
set -uo pipefail
cd "$(dirname "$0")/.."
bash hpc/physK_check.sh; echo "PHYSK_EXIT $?"
bash hpc/gpu_check.sh; echo "GPUCHECK_DONE"
bash hpc/multistep_bench.sh; echo "BENCH_DONE"
