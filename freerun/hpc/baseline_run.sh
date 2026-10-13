#!/bin/bash
# Baseline measurements in an older checkout (one without the stepping API), for docs/performance.md: copy this
# script and bench/closed_loop.py into that tree, then run it inside hpc/job.sbatch. Legacy-API closed loop and the
# bench grid with nsys kernel times, without and with 1 % jitter (drop --jitter for trees older than the jitter work).
set -uo pipefail
OUT=${TMPDIR:-/tmp}/fr_nsys_$$
summary() {
  python - "$1" <<'PY'
import csv, sys, subprocess
out = subprocess.run(["nsys", "stats", "-q", "-r", "cuda_gpu_kern_sum", "-f", "csv", sys.argv[1]],
                     capture_output=True, text=True).stdout
rows = list(csv.DictReader(l for l in out.splitlines() if l.strip() and not l.startswith("Processing")))
tot = sum(float(r["Total Time (ns)"]) for r in rows if "run" in r["Name"])
print(f"  run-kernel total {tot / 1e6:.1f} ms")
PY
}
prof() {
  echo "=== $*"
  nsys profile -t cuda -o "$OUT" -f true "$@" 2>/dev/null | grep '^{'
  summary "$OUT.nsys-rep"
  rm -f "$OUT.nsys-rep" "$OUT.sqlite"
}
for j in 0 0.01; do
  CL="python bench/closed_loop.py --backend cuda --legacy --steps 100 --jitter $j"
  prof $CL --circuits 4096 --gates 1024
  prof $CL --circuits 16384 --gates 256
  prof $CL --circuits 1024 --gates 4096
  prof $CL --circuits 4096 --gates 1024 --workload rings
  B="python bench/bench.py --backend cuda --jitter $j"
  prof $B --workload random --circuits 256 --gates 4096 --T 100000
  prof $B --workload rings --circuits 256 --gates 4096 --T 100000
  prof $B --workload random --circuits 1024 --gates 1024 --T 100000
  prof $B --workload random --circuits 8192 --gates 1024 --T 100000
  prof $B --workload random --circuits 64 --gates 65536 --T 50000
  prof $B --workload random --circuits 8 --gates 1048576 --T 10000
done
