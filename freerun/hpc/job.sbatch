#!/bin/bash
#SBATCH --job-name=freerun
#SBATCH --account=sy-grp
#SBATCH --partition=sy-grp
#SBATCH --nodelist=cn-x-1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# Usage: hpc/submit.sh [sbatch options] -- <command...>  (sets FREERUN_DIR and the log path for this checkout), or
# from the project directory on the share (logs/ must exist; Slurm resolves the log path against the submission dir):
#   mkdir -p logs && sbatch [--job-name X --time T] hpc/job.sbatch <command...>
# Builds the CPU and CUDA backends on the compute node, then runs the command from the project dir, which is
# $FREERUN_DIR if set, else the directory sbatch was called from.
set -euo pipefail
cd "${FREERUN_DIR:-$SLURM_SUBMIT_DIR}"
source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module load cuda/12.8 cmake/3.27.9 gcc/12.4 >/dev/null 2>&1 || module load cuda/12.8 cmake/3.27.9
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv
echo "project dir: $PWD  commit: $(cat COMMIT 2>/dev/null || echo unknown)"
BUILD_DIR=build_node scripts/build.sh -DFREERUN_CUDA=ON > /dev/null
export PATH=$PWD/.venv/bin:$PATH
"$@"
