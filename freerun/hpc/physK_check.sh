#!/bin/bash
# run_steps at physical step lengths (K = 10^5 .. 10^8 ticks, budgets 10^3 .. 10^6 events per step, early stop,
# jitter and delay mismatch): every CUDA kernel variant == CPU == reference (bench/multistep_physK.py), two seeds.
# Run inside hpc/job.sbatch (backends built):  hpc/submit.sh --job-name fr_physK --time 01:00:00 -- bash hpc/physK_check.sh
set -uo pipefail
export PYTHONPATH=$PWD/python${PYTHONPATH:+:$PYTHONPATH}
rc=0
for seed in 0 1; do
  echo "=== seed $seed"
  python bench/multistep_physK.py --seed $seed --circuits 32 --cases 2 || rc=1
done
echo "=== no jitter, no mismatch"
python bench/multistep_physK.py --seed 2 --circuits 32 --cases 1 --jitter 0 --mismatch 0 || rc=1
exit $rc
