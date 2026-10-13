#!/bin/bash
#SBATCH --job-name=grown_long
#SBATCH --account=sy-grp
#SBATCH --partition=sy-grp
#SBATCH --nodelist=cn-x-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# A multi-day grown_search run that survives its time limit, on CPUs only (no GPU is requested: E3-phys places every
# lane on the CPU threads; pass --set substrate.backend=cpu so nothing touches a GPU, or add --gres=gpu:1 for a config
# that needs one). The search checkpoints (archive.save_every /
# save_seconds) and runs with --checkpoint-resume, so it continues from its newest checkpoint whenever it starts;
# 15 minutes before the limit the job stops the search and requeues itself. The limit is 3 days, not the partition's 14:
# sy-grp's QOS caps the group's running CPU-minutes (2,580,480: CPUs x remaining time limit, all users together), and
# three 32-CPU 14-day jobs used 75 % of it, blocking further jobs; a 3-day limit reserves 138,240 CPU-minutes per
# job. When the search reaches search.generations it evaluates the run on fresh episodes (eval.json) and writes
# job.json. Do not push changed src/ to the checkout while its runs are live (a requeued job would run new code).
# Submit from the repository root (logs/ is resolved against it). The account, partition and node are those of
# the OSU HPC cluster; override them on the command line (sbatch --account A --partition P --nodelist N) or
# with SBATCH_ACCOUNT / SBATCH_PARTITION.
#
#   sbatch --job-name g_e3p_s0 hpc/grown/longrun.sbatch \
#       configs/grown/exp/e3_phys_b1e5.yaml 0 --set substrate.backend=cpu [grown_search.py args]
# Mid-run summary without stopping it (CPU, from the newest checkpoint):
#   $PY scripts/grown_eval.py --run <out> --checkpoint latest --backend cpu
set -uo pipefail
source "${GROWN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/hpc/grown/env.sh"
cd "$GROWN_ROOT"
[ -e "$FREERUN_LIBDIR/libfreerun_cpu.so" ] || build_freerun
CONFIG=$1; SEED=$2; shift 2
NAME=$("$PY" -c "import sys; sys.path.insert(0, 'src'); from grown import config; print(config.load(sys.argv[1])['name'])" "$CONFIG")
OUT=${GROWN_OUT:-$GROWN_RUNS/${NAME}_s${SEED}}
mkdir -p "$OUT"
echo "=== job ${SLURM_JOB_ID:-local} (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname) $(date), code $(cat .grown_commit 2>/dev/null), $SLURM_CPUS_PER_TASK CPUs, out $OUT" | tee -a "$OUT/jobs.log"
SMI=
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,nounits \
    -l 10 >> "$OUT/gpu_util.csv" 2>/dev/null &
  SMI=$!
fi
"$PY" scripts/grown_search.py --config "$CONFIG" --seed "$SEED" --out "$OUT" --checkpoint-resume "$@" \
  >> "$OUT/search.log" 2>&1 &
PID=$!
requeue() {
  echo "=== $(date): time limit near, stopping the search (newest checkpoint kept) and requeueing" | tee -a "$OUT/jobs.log"
  kill -TERM "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  [ -n "$SMI" ] && kill "$SMI" 2>/dev/null
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue USR1
wait "$PID"
RC=$?
[ -n "$SMI" ] && kill "$SMI" 2>/dev/null
echo "=== $(date): search exit $RC" | tee -a "$OUT/jobs.log"
if [ "$RC" = 0 ]; then
  "$PY" scripts/grown_eval.py --run "$OUT" --quiet 2>&1 | tail -n 30
  "$PY" - "$OUT" <<'PYEOF'
import json, os, sys
out = sys.argv[1]
s = json.load(open(os.path.join(out, "summary.json")))
rec = dict(job_id=os.environ.get("SLURM_JOB_ID"), restarts=os.environ.get("SLURM_RESTART_COUNT"),
           code_commit=open(".grown_commit").read().strip() if os.path.exists(".grown_commit") else None, edits=s["edits"], generations=s["generations"],
           loop_seconds=s["loop_seconds"], solved=s["solved"], n_tasks=s["n_tasks"],
           resumed_checkpoint=s.get("resumed_checkpoint"))
json.dump(rec, open(os.path.join(out, "job.json"), "w"), indent=1)
print("job.json:", json.dumps(rec))
PYEOF
fi
exit "$RC"
