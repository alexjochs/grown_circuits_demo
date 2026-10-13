#!/bin/bash
# CPU backend thread scaling (bench/cpu_threads.py), inside hpc/job.sbatch (backends built), with as many CPUs as the
# job has: 3-gate rings, then a dumped search generation (LANES_NPZ, from evolved_cartpole's grown_lane_probe --dump).
#   hpc/submit.sh --job-name fr_cpuscale --cpus-per-task=64 --time 02:00:00 -- bash hpc/cpu_scaling.sh [lanes.npz]
set -uo pipefail
export PYTHONPATH=$PWD/python${PYTHONPATH:+:$PYTHONPATH}
N=${SLURM_CPUS_PER_TASK:-$(nproc)}
TH="1 2 8 32 64"
[[ " $TH " == *" $N "* ]] || TH="$TH $N"
echo "=== $N CPUs: $(lscpu | grep 'Model name' | sed 's/ \+/ /g')"
echo "=== 3-gate rings, 1024 lanes, 4 steps of 10^6 ticks"
python bench/cpu_threads.py rings --lanes 1024 --K 1000000 --steps 4 --threads $TH
if [ -n "${1:-}" ]; then
  echo "=== search generation lanes $1, 4 steps each"
  python bench/cpu_threads.py npz --file "$1" --steps 4 --threads $TH
fi
