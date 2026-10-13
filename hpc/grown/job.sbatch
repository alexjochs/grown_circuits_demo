#!/bin/bash
#SBATCH --job-name=grown
#SBATCH --account=sy-grp
#SBATCH --partition=sy-grp
#SBATCH --nodelist=cn-x-1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# One grown_search config on one GPU, then grown_eval on fresh episodes. Usually submitted by sweep.sh.
# Submit from the repository root (logs/ is resolved against it). The account, partition and node are those of
# the OSU HPC cluster; override them on the command line (sbatch --account A --partition P --nodelist N) or
# with SBATCH_ACCOUNT / SBATCH_PARTITION.
#
#   sbatch [--job-name N] hpc/grown/job.sbatch <config.yaml> <seed> [grown_search.py args, e.g. --set k=v]
#   sbatch hpc/grown/job.sbatch --run <out dir> <script.py | script.sh> [args]    any script, same bookkeeping
#
# Search runs go to $GROWN_RUNS/<config name>_s<seed> (default $GROWN_SHARE/runs, see env.sh; set GROWN_OUT for
# an explicit directory). Next to the search outputs the run directory gets:
#   job.json       job id, node, GPU, code commit, wall seconds, GPU-hours (1 GPU x wall), edits/s,
#                  GPU utilization summary
#   gpu_util.csv   nvidia-smi samples every 500 ms (utilization.gpu, memory, power) for the whole job
#   eval.json      scripts/grown_eval.py on fresh episodes
#   search.log     a copy of the search's stdout
set -uo pipefail
source "${GROWN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/hpc/grown/env.sh"
cd "$GROWN_ROOT"
[ -e "$FREERUN_LIBDIR/libfreerun_cuda.so" ] || build_freerun
if [ "${1:-}" = "--run" ]; then
  OUT=$2; shift 2; MODE=script
else
  CONFIG=$1; SEED=$2; shift 2; MODE=search
  NAME=$("$PY" -c "import sys; sys.path.insert(0, 'src'); from grown import config; print(config.load(sys.argv[1])['name'])" "$CONFIG")
  OUT=${GROWN_OUT:-$GROWN_RUNS/${NAME}_s${SEED}}
fi
mkdir -p "$OUT"
T0=$(date +%s)
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,nounits \
  -lms 500 > "$OUT/gpu_util.csv" 2>/dev/null &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
echo "job ${SLURM_JOB_ID:-local} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}, code $(cat .grown_commit 2>/dev/null), out $OUT"
if [ "$MODE" = search ]; then
  "$PY" scripts/grown_search.py --config "$CONFIG" --seed "$SEED" --out "$OUT" "$@" 2>&1 | tee "$OUT/search.log"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
  if [ "$RC" = 0 ]; then
    "$PY" scripts/grown_eval.py --run "$OUT" --quiet 2>&1 | tail -n 30
    RC=${PIPESTATUS[0]}
  fi
else
  if [[ $1 == *.sh ]]; then bash "$@"; else "$PY" "$@"; fi 2>&1 | tee "$OUT/job_stdout.log"
  RC=${PIPESTATUS[0]}
  T1=$(date +%s)
fi
T2=$(date +%s)
kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
"$PY" - "$OUT" "$T0" "$T1" "$T2" "$RC" "$MODE" <<'EOF'
import csv, json, os, socket, subprocess, sys
out, t0, t1, t2, rc, mode = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6]
util = []
try:
    with open(os.path.join(out, "gpu_util.csv")) as f:
        for row in csv.reader(f):
            try:
                util.append(float(row[1]))
            except (ValueError, IndexError):
                pass
except OSError:
    pass
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
rec = dict(job_id=os.environ.get("SLURM_JOB_ID"), job_name=os.environ.get("SLURM_JOB_NAME"), node=socket.gethostname(),
           gpu=gpu.strip(), gpus=1, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
           cpus=os.environ.get("SLURM_CPUS_PER_TASK"), mode=mode, exit_code=rc,
           code_commit=open(".grown_commit").read().strip() if os.path.exists(".grown_commit") else None,
           start=t0, search_seconds=t1 - t0, wall_seconds=t2 - t0, gpu_hours=(t2 - t0) / 3600.0,
           gpu_util_mean=sum(util) / len(util) if util else None, gpu_util_max=max(util) if util else None,
           gpu_util_samples=len(util), gpu_util_nonzero_frac=sum(u > 0 for u in util) / len(util) if util else None)
try:
    s = json.load(open(os.path.join(out, "summary.json")))
    rec.update(edits=s["edits"], loop_seconds=s["loop_seconds"], edits_per_s=s["edits_per_s"], workers=s["workers"],
               solved=s["solved"], n_tasks=s["n_tasks"])
except (OSError, KeyError, ValueError):
    pass
json.dump(rec, open(os.path.join(out, "job.json"), "w"), indent=1)
print("job.json:", json.dumps(rec))
EOF
echo "exit $RC"
exit "$RC"
