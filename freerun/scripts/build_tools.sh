#!/bin/bash
# Build or fetch the CPU reference tools into tools/ (gitignored). No root needed. Idempotent: each tool is skipped
# if its install prefix already exists. Usage: scripts/build_tools.sh [tool ...]   (default: all)
#
# Versions are pinned so cross-check results are reproducible.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=$ROOT/tools
SRC=$T/src
JOBS=${JOBS:-$(nproc)}
mkdir -p "$T" "$SRC"

GPERF_VER=3.1
IVERILOG_TAG=v13_0
NGSPICE_VER=47
NVC_VER=1.23.0
GHDL_VER=6.0.0
SKY130_VER=1e3554706ddbb2529193bdabbc7128296f3cb44a  # ciel (open_pdks) build of sky130A/B
CUDA_NVCC_VER=12.8.93     # local compile checks only (no GPU needed); cluster builds use the cuda module
CUDA_RT_VER=12.8.90

fetch() {  # fetch <url> <dest>
  if [ ! -f "$2" ]; then
    curl -fL --retry 3 -o "$2.part" "$1" || { echo "download failed: $1" >&2; exit 1; }
    mv -f "$2.part" "$2"
  fi
}

build_gperf() {
  [ -x "$T/gperf/bin/gperf" ] && return
  fetch https://ftp.gnu.org/pub/gnu/gperf/gperf-$GPERF_VER.tar.gz "$SRC/gperf-$GPERF_VER.tar.gz"
  rm -rf "$SRC/gperf-$GPERF_VER" && tar -C "$SRC" -xzf "$SRC/gperf-$GPERF_VER.tar.gz"
  (cd "$SRC/gperf-$GPERF_VER" && ./configure --prefix="$T/gperf" && make -j"$JOBS" && make install)
}

build_iverilog() {
  [ -x "$T/iverilog/bin/iverilog" ] && return
  build_gperf
  fetch https://github.com/steveicarus/iverilog/archive/refs/tags/$IVERILOG_TAG.tar.gz "$SRC/iverilog-$IVERILOG_TAG.tar.gz"
  rm -rf "$SRC/iverilog-$IVERILOG_TAG" && mkdir -p "$SRC/iverilog-$IVERILOG_TAG"
  tar -C "$SRC/iverilog-$IVERILOG_TAG" --strip-components=1 -xzf "$SRC/iverilog-$IVERILOG_TAG.tar.gz"
  (cd "$SRC/iverilog-$IVERILOG_TAG" && export PATH="$T/gperf/bin:$PATH" && sh autoconf.sh \
    && ./configure --prefix="$T/iverilog" && make -j"$JOBS" && make install)
}

build_ngspice() {
  [ -x "$T/ngspice/bin/ngspice" ] && return
  fetch https://downloads.sourceforge.net/project/ngspice/ng-spice-rework/$NGSPICE_VER/ngspice-$NGSPICE_VER.tar.gz \
    "$SRC/ngspice-$NGSPICE_VER.tar.gz"
  rm -rf "$SRC/ngspice-$NGSPICE_VER" && tar -C "$SRC" -xzf "$SRC/ngspice-$NGSPICE_VER.tar.gz"
  (cd "$SRC/ngspice-$NGSPICE_VER" && mkdir -p build && cd build \
    && ../configure --prefix="$T/ngspice" --with-readline=yes --without-x --enable-openmp --enable-osdi --disable-debug \
    && make -j"$JOBS" && make install)
}

fetch_nvc() {  # prebuilt Ubuntu 24.04 package, unpacked without dpkg -i
  [ -x "$T/nvc/usr/bin/nvc" ] && return
  fetch https://github.com/nickg/nvc/releases/download/r$NVC_VER/nvc_$NVC_VER-1_amd64_ubuntu-24.04.deb \
    "$SRC/nvc_$NVC_VER.deb"
  rm -rf "$T/nvc" && dpkg-deb -x "$SRC/nvc_$NVC_VER.deb" "$T/nvc"
}

fetch_ghdl() {  # prebuilt mcode backend (no LLVM/GCC needed)
  [ -x "$T/ghdl/bin/ghdl" ] && return
  fetch https://github.com/ghdl/ghdl/releases/download/v$GHDL_VER/ghdl-mcode-$GHDL_VER-ubuntu24.04-x86_64.tar.gz \
    "$SRC/ghdl-$GHDL_VER.tar.gz"
  rm -rf "$T/ghdl" && mkdir -p "$T/ghdl" && tar -C "$T/ghdl" --strip-components=1 -xzf "$SRC/ghdl-$GHDL_VER.tar.gz"
}

fetch_cuda() {  # NVIDIA redistributable archives: nvcc, cudart, cccl
  [ -x "$T/cuda/bin/nvcc" ] && return
  local base=https://developer.download.nvidia.com/compute/cuda/redist
  mkdir -p "$T/cuda"
  for p in cuda_nvcc:$CUDA_NVCC_VER cuda_cudart:$CUDA_RT_VER cuda_cccl:$CUDA_RT_VER; do
    local name=${p%%:*} ver=${p##*:}
    fetch $base/$name/linux-x86_64/$name-linux-x86_64-$ver-archive.tar.xz "$SRC/$name-$ver.tar.xz"
    tar -C "$T/cuda" --strip-components=1 -xf "$SRC/$name-$ver.tar.xz"
  done
  [ -e "$T/cuda/lib64" ] || ln -s lib "$T/cuda/lib64"
}

fetch_pdk() {  # SKY130 with the sky130_fd_sc_hd standard cells (Liberty, Verilog models, SPICE models)
  [ -d "$T/pdk/sky130A" ] && return
  local py=$ROOT/.venv/bin/python
  "$py" -m pip --version >/dev/null 2>&1 || true
  "$ROOT/tools/uv/uv" pip install -q --python "$py" ciel
  "$ROOT/.venv/bin/ciel" enable --pdk-family sky130 --pdk-root "$T/pdk" -l sky130_fd_sc_hd -l sky130_fd_pr $SKY130_VER
  ln -sfn ciel/sky130/versions/$SKY130_VER/sky130A "$T/pdk/sky130A"
}

ALL=(iverilog nvc ghdl ngspice cuda pdk)
for t in "${@:-${ALL[@]}}"; do
  case $t in
    iverilog) build_iverilog ;; ngspice) build_ngspice ;; nvc) fetch_nvc ;; ghdl) fetch_ghdl ;; cuda) fetch_cuda ;; pdk) fetch_pdk ;;
    *) echo "unknown tool $t" >&2; exit 1 ;;
  esac
  echo "== $t ok"
done
