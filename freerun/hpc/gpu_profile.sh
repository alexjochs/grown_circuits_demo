#!/bin/bash
# Profiling of the windowed kernel on the first-class workload, run inside hpc/job.sbatch.
# Usage: hpc/gpu_profile.sh [closed_loop.py args], default 4096 circuits x 1024 gates.
# Prints the summary sections and the stall breakdown; writes per-SASS-instruction metrics with their CUDA source
# lines to logs/prof_sass_<job>.csv.
set -uo pipefail
ARGS=${*:---circuits 4096 --gates 1024}
REP=logs/prof_${SLURM_JOB_ID:-local}
echo "=== ncu: closed loop $ARGS (one step after warm-up)"
ncu --kernel-name regex:k_run_windowed --launch-skip 2 --launch-count 1 --import-source yes --set full -o "$REP" -f \
  python bench/closed_loop.py --backend cuda $ARGS --steps 1 --warmup 2 2>&1 | grep -v "^==PROF==" | tail -3
ncu -i "$REP.ncu-rep" --section SpeedOfLight --section Occupancy --section WarpStateStats --section SchedulerStats \
  --section InstructionStats 2>&1 | grep -v "^ *---\|^$" | grep -v "^          " | tail -60
echo "=== stall reasons (cycles per issued instruction)"
ncu -i "$REP.ncu-rep" --page raw --csv --metrics regex:smsp__average_warps_issue_stalled_.*_per_issue_active.ratio 2>&1 |
  python -c '
import csv, sys
rows = list(csv.reader(sys.stdin))
hdr, val = rows[0], rows[-1]
pairs = sorted(((float(v), h) for h, v in zip(hdr, val) if "stalled" in h and v.replace(".", "", 1).isdigit()), reverse=True)
for v, h in pairs[:14]: print(f"  {v:8.3f}  {h}")'
ncu -i "$REP.ncu-rep" --page source --csv --print-source cuda,sass > "logs/prof_sass_${SLURM_JOB_ID:-local}.csv" 2>&1
echo "per-instruction metrics: logs/prof_sass_${SLURM_JOB_ID:-local}.csv"
head -2 "logs/prof_sass_${SLURM_JOB_ID:-local}.csv" | cut -c1-600
