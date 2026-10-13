#!/bin/bash
# Build the compiled backends into python/freerun/_lib. Usage: scripts/build.sh [extra cmake args, e.g. -DFREERUN_CUDA=ON]
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD_DIR:-build}
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "$@" > /dev/null
cmake --build "$BUILD" -j "$(nproc)"
