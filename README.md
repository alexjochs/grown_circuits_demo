# Grown Circuits: test repository

Grown Circuits searches for control circuits using a gate-level simulator with real
cell delays, process variation and energy, running thousands of different circuits at once on GPUs and CPUs. The
search is guided by a learned editor that proposes circuit edits and by a library of reusable subcircuits that the
system grows from its own history. 

## Directory

| path | contents |
|---|---|
| `freerun/` | the simulator: batched event-driven simulation of clockless gate-level circuits, C++ CPU and CUDA engines (`csrc/`), Python package (`python/`) |
| `src/grown/tasks/` | task families and their teachers (signal tasks, generated bodies, sensorimotor tasks) |
| `src/grown/circuit/` | the circuit representation (graphs of primitive and library tokens) and the edit moves |
| `src/grown/archive/` | the quality-diversity archive (MAP-Elites) and its descriptors |
| `src/grown/search.py` | the search loop: per-task archives, teacher-matched fitness, many tasks in parallel |
| `src/grown/sim/` | substrates: the freerun adapter, device populations with SKY130 timing and energy, a slow delay element, a fast synchronous stand-in |
| `src/grown/editor/` | the learned editor (a transformer that proposes edits), its training and evaluation |
| `src/grown/library/` | library learning: compressing the search's history into reusable subcircuits |
| `src/grown/log/` | the edit log (every edit and its fate under selection) |
| `scripts/` | entry points: `grown_search.py`, `grown_eval.py`, `grown_library.py`, `grown_editor.py`, `grown_loop.py`, `grown_tasks.py`, and the multi-GPU launcher `grown_multi_gpu.py` |
| `configs/grown/` | run configurations; `exp/` holds the experiment configs |
| `hpc/` | Slurm scripts: environment setup, single-GPU and multi-day jobs, sweeps, and the multi-GPU demo `demo_2gpu.sbatch` |

## Running the demo

Needs Linux, Python 3.11 or newer, a C++ compiler and CMake; the GPU steps need the CUDA toolkit and NVIDIA GPUs.

1. Install (a fresh virtual environment, the simulator and this package):
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e ./freerun -e .
   freerun/scripts/build.sh                      # compiles the simulator's CPU engine
   ```
2. Run a short search on the CPU (7 signal tasks, 200 generations, about half a minute):
   ```bash
   .venv/bin/python scripts/grown_search.py --config configs/grown/exp/e0_fr_unit_cpu.yaml --seed 0 \
       --generations 200 --workers 4 --out runs/e0_cpu_s0
   ```
3. Evaluate the result on fresh episodes:
   ```bash
   .venv/bin/python scripts/grown_eval.py --run runs/e0_cpu_s0
   ```
4. On the cluster, set up the environment once (from the repository root on a login node), then submit the 2-GPU
   job, which compiles the CUDA engine on the GPU node, runs the search split over 2 GPUs and prints one table:
   ```bash
   export GROWN_SHARE=/path/to/large/disk/grown_circuits_demo   # venv, build and runs go here
   bash hpc/grown/setup_env.sh
   sbatch hpc/demo_2gpu.sbatch                                  # output: logs/grown_demo_<jobid>.out
   ```
5. To use N GPUs, request them:
   ```bash
   sbatch --gres=gpu:8 --cpus-per-task=8 hpc/demo_2gpu.sbatch
   ```
   On any machine with GPUs the launcher also runs directly:
   `.venv/bin/python scripts/grown_multi_gpu.py --config configs/grown/exp/e0_fr_unit_cuda.yaml --gpus 2`
   (after `freerun/scripts/build.sh -DFREERUN_CUDA=ON`).

The configurations whose names contain `sky130` or `phys` use the SKY130 cell library and need the PDK, which is not
included here (2 GB); the demo configurations do not.

The Slurm scripts default to the OSU HPC account, partition and node; override them with `sbatch --account`,
`--partition` and `--nodelist`.

The multi-GPU launcher starts one process per GPU, each running an ordinary search on its share of the tasks (and of
the seeds, when there are more GPUs than tasks). The GPUs do not communicate with each other: every task's search is
independent, so a shard gives the same search it would on one GPU.


## Scaling

The search shards across tasks, seeds and lanes with no communication between GPUs, so a rack is more shards.

