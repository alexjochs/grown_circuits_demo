#!/bin/bash
# Kernel-only GPU time for bench.py configurations (nsys), next to bench.py's wall-clock numbers.
# Usage (inside hpc/job.sbatch): hpc/kernel_times.sh [extra bench.py args]
set -uo pipefail
OUT=${TMPDIR:-/tmp}/fr_nsys_$$
summary() {
  python - "$1" <<'PY'
import csv, sys, subprocess
out = subprocess.run(["nsys", "stats", "-q", "-r", "cuda_gpu_kern_sum", "-f", "csv", sys.argv[1]],
                     capture_output=True, text=True).stdout
rows = list(csv.DictReader(l for l in out.splitlines() if l.strip() and not l.startswith("Processing")))
tot = sum(float(r["Total Time (ns)"]) for r in rows)
for r in rows:
    print(f"  kernel {r['Name'][:40]:40s} {float(r['Total Time (ns)']) / 1e6:10.1f} ms  x{r['Instances']}")
print(f"  kernel total {tot / 1e6:.1f} ms")
PY
}
run() {
  echo "=== bench $*"
  nsys profile -t cuda -o "$OUT" -f true python bench/bench.py --backend cuda "$@" 2>/dev/null | grep '^{'
  summary "$OUT.nsys-rep"
  rm -f "$OUT.nsys-rep" "$OUT.sqlite"
}
run --workload random --circuits 256 --gates 4096 --T 100000 "$@"
run --workload rings --circuits 256 --gates 4096 --T 100000 "$@"
run --workload random --circuits 1024 --gates 1024 --T 100000 "$@"
run --workload random --circuits 8192 --gates 1024 --T 100000 "$@"
run --workload random --circuits 64 --gates 65536 --T 50000 "$@"
run --workload random --circuits 8 --gates 1048576 --T 10000 "$@"
