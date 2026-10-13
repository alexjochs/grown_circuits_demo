#!/bin/bash
#SBATCH --job-name=g_cpu
#SBATCH --account=sy-grp
#SBATCH --partition=sy-grp
#SBATCH --nodelist=cn-x-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# A shell script on CPUs only (no GPU), with the grown environment and $GROWN_ROOT/src first on PYTHONPATH; the
# script sees $PY (the venv's python):
#
#   sbatch hpc/grown/cpu.sbatch my_script.sh [args]
# Submit from the repository root (logs/ is resolved against it). The account, partition and node are those of
# the OSU HPC cluster; override them on the command line (sbatch --account A --partition P --nodelist N) or
# with SBATCH_ACCOUNT / SBATCH_PARTITION.
set -uo pipefail
source "${GROWN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/hpc/grown/env.sh"
cd "$GROWN_ROOT"
[ -e "$FREERUN_LIBDIR/libfreerun_cpu.so" ] || build_freerun
export PYTHONPATH=$GROWN_ROOT/src PY
echo "job ${SLURM_JOB_ID:-local} on $(hostname), code $(cat .grown_commit 2>/dev/null): $*"
bash "$@"
