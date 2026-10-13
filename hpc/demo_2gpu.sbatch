#!/bin/bash
#SBATCH --job-name=grown_demo
#SBATCH --account=sy-grp
#SBATCH --partition=sy-grp
#SBATCH --nodelist=cn-x-1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# Multi-GPU demo: the E0 search (7 signal tasks, freerun CUDA backend) split over the GPUs of this job, one process
# per GPU, then a fresh-episode evaluation of every shard and one summary table. The GPUs do not communicate.
# Submit from the repository root after hpc/grown/setup_env.sh:
#
#   sbatch hpc/demo_2gpu.sbatch                                         # 2 GPUs
#   sbatch --gres=gpu:8 --cpus-per-task=8 hpc/demo_2gpu.sbatch          # N GPUs: the same script, N from --gres
#
# Variables (export before sbatch): CONFIG (default configs/grown/exp/e0_fr_unit_cuda.yaml), GENERATIONS (default:
# the config's 1000), SEEDS (default auto: more seeds when there are more GPUs than tasks).
# The account, partition and node are those of the OSU HPC cluster; override them on the command line
# (sbatch --account A --partition P --nodelist N) or with SBATCH_ACCOUNT / SBATCH_PARTITION.
set -euo pipefail
source "${GROWN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/hpc/grown/env.sh"
cd "$GROWN_ROOT"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv
echo "job ${SLURM_JOB_ID:-local} on $(hostname), GPUs ${CUDA_VISIBLE_DEVICES:-?}, $SLURM_CPUS_PER_TASK CPUs, code $(cat .grown_commit 2>/dev/null)"
T0=$(date +%s)
build_freerun
echo "build: $(( $(date +%s) - T0 )) s"
"$PY" scripts/grown_multi_gpu.py --config "${CONFIG:-configs/grown/exp/e0_fr_unit_cuda.yaml}" --backend cuda \
  --seeds ${SEEDS:-auto} ${GENERATIONS:+--generations $GENERATIONS} --eval \
  --out "$GROWN_RUNS/demo_${SLURM_JOB_ID:-local}"
echo "job wall: $(( $(date +%s) - T0 )) s"
