#!/bin/bash
# Environment for grown jobs on a Slurm cluster; sourced by the sbatch scripts, setup_env.sh and sweep.sh.
# Every path is relative to the repository unless overridden from the environment. On a cluster with a small home
# directory, point GROWN_SHARE at a large filesystem (the venv, the freerun build, runs and the kernel cache go there):
#   export GROWN_SHARE=/nfs/hpc/share/$USER/grown_circuits_demo
GROWN_ROOT=${GROWN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
GROWN_SHARE=${GROWN_SHARE:-$GROWN_ROOT/.work}
GROWN_VENV=${GROWN_VENV:-$GROWN_SHARE/venv}
GROWN_RUNS=${GROWN_RUNS:-$GROWN_SHARE/runs}
export GROWN_CACHE=${GROWN_CACHE:-$GROWN_SHARE/cache}          # compiled stand-in kernel (grown.sim._csync)
export FREERUN_HOME=${FREERUN_HOME:-$GROWN_ROOT/freerun}       # the vendored simulator
export FREERUN_LIBDIR=${FREERUN_LIBDIR:-$GROWN_SHARE/freerun_lib}
FREERUN_BUILD=${FREERUN_BUILD:-$GROWN_SHARE/freerun_build}
FREERUN_CUDA_ARCH=${FREERUN_CUDA_ARCH:-90}                     # 90: H100/H200; set for other GPUs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
case ":${PYTHONPATH:-}:" in                                    # this checkout's code first
  *":$GROWN_ROOT/src:"*) ;;
  *) export PYTHONPATH=$GROWN_ROOT/src${PYTHONPATH:+:$PYTHONPATH} ;;
esac
export GROWN_ROOT GROWN_SHARE GROWN_RUNS
PY=$GROWN_VENV/bin/python
mkdir -p "$GROWN_RUNS" "$GROWN_CACHE" "$FREERUN_LIBDIR"
if [ -z "${GROWN_NO_MODULES:-}" ]; then
  # Environment modules of the OSU HPC cluster (override with GROWN_MODULES, or GROWN_NO_MODULES=1 to skip):
  # cuda for nvcc and the runtime of libfreerun_cuda.so, cmake for the build, gcc for the stand-in kernel.
  source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
  module load ${GROWN_MODULES:-cuda/12.8 cmake/3.27.9 gcc/12.4} >/dev/null 2>&1 \
    || module load cuda/12.8 cmake/3.27.9 >/dev/null 2>&1 || true
fi

# Build freerun's CPU backend, and its CUDA backend when nvcc is available, into $FREERUN_LIBDIR (run it on a node of
# the target GPU type; FREERUN_CUDA_ARCH sets the architecture).
build_freerun() {
  local cuda=OFF
  command -v nvcc >/dev/null && cuda=ON
  cmake -S "$FREERUN_HOME" -B "$FREERUN_BUILD" -DCMAKE_BUILD_TYPE=Release -DFREERUN_CUDA=$cuda \
    -DCMAKE_CUDA_ARCHITECTURES="$FREERUN_CUDA_ARCH" -DFREERUN_LIBDIR="$FREERUN_LIBDIR" >/dev/null
  cmake --build "$FREERUN_BUILD" -j "${SLURM_CPUS_PER_TASK:-4}" >/dev/null
  echo "freerun built (CUDA $cuda): $(ls "$FREERUN_LIBDIR" | tr '\n' ' ')"
}
