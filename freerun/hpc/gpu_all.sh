#!/bin/bash
# Validation, then the performance record. Arguments go to hpc/gpu_bench.sh.
cd "$(dirname "$0")/.."
hpc/gpu_check.sh
hpc/gpu_bench.sh "$@"
