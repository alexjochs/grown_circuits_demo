"""Compiled kernel for the synchronous stand-in (same semantics as the numpy kernel in sync.py, bit for bit).

The C source below is compiled on first use with the system C compiler into a cache directory ($GROWN_CACHE or
~/.cache/grown/csync) keyed by the source hash, and loaded with ctypes. Lanes are simulated one after another, each
over its own gate count (no padding to the batch's largest circuit). If no compiler is available, `load()` returns
None and SyncSubstrate falls back to numpy.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile

import numpy as np

SRC = r"""
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* One lane at a time; lane b runs circuit c = lane_circ[b] (or c = b when lane_circ is NULL) of the netlist batch.
   Nets of the lane: 0..I-1 inputs, I..I+G_c-1 gates. Each tick: inputs of tick t (if t < Kx) are applied, every gate
   computes from the current net values, all gates commit together, outputs are accumulated. */
int grown_sync_run(int B, int S, int K, int Kx, int I, int n_out,
                   const int64_t* gate_off, const int64_t* pin_off, const int32_t* fanin, const uint64_t* tt,
                   const uint8_t* init, const int32_t* outs, const uint8_t* x,
                   uint8_t* out, float* high, double* toggles, const int64_t* lane_circ, int64_t* step_tog) {
    int64_t gmax = 0;
    for (int b = 0; b < B; b++) {
        const int64_t c = lane_circ ? lane_circ[b] : b;
        int64_t g = gate_off[c + 1] - gate_off[c]; if (g > gmax) gmax = g;
    }
    uint8_t* vals = (uint8_t*)malloc((size_t)(I + gmax + 1));
    uint8_t* nxt = (uint8_t*)malloc((size_t)(gmax + 1));
    int64_t* h = (int64_t*)malloc(sizeof(int64_t) * (size_t)(n_out + 1));
    if (!vals || !nxt || !h) { free(vals); free(nxt); free(h); return 1; }
    for (int b = 0; b < B; b++) {
        const int64_t c = lane_circ ? lane_circ[b] : b;
        const int64_t g0 = gate_off[c], G = gate_off[c + 1] - g0;
        const int64_t* po = pin_off + g0;
        const uint64_t* tb = tt + g0;
        const int32_t* ob = outs + c * n_out;
        const uint8_t* xb = x + (int64_t)b * S * Kx * I;
        uint8_t* outb = out + (int64_t)b * S * n_out;
        float* highb = high + (int64_t)b * S * n_out;
        memset(vals, 0, (size_t)I);
        memcpy(vals + I, init + g0, (size_t)G);
        int64_t tog = 0;
        for (int s = 0; s < S; s++) {
            const int64_t tog0 = tog;
            for (int j = 0; j < n_out; j++) h[j] = 0;
            for (int t = 0; t < K; t++) {
                if (t < Kx) memcpy(vals, xb + ((int64_t)s * Kx + t) * I, (size_t)I);
                for (int64_t g = 0; g < G; g++) {
                    unsigned idx = 0;
                    const int64_t p0 = po[g], p1 = po[g + 1];
                    for (int64_t p = p0; p < p1; p++) idx |= (unsigned)vals[fanin[p]] << (p - p0);
                    nxt[g] = (uint8_t)((tb[g] >> idx) & 1u);
                }
                uint8_t* gv = vals + I;
                for (int64_t g = 0; g < G; g++) { tog += gv[g] != nxt[g]; gv[g] = nxt[g]; }
                for (int j = 0; j < n_out; j++) h[j] += vals[ob[j]];
            }
            for (int j = 0; j < n_out; j++) {
                outb[(int64_t)s * n_out + j] = vals[ob[j]];
                highb[(int64_t)s * n_out + j] = (float)((double)h[j] / K);
            }
            if (step_tog) step_tog[(int64_t)b * S + s] = tog - tog0;
        }
        toggles[b] = (double)tog;
    }
    free(vals); free(nxt); free(h);
    return 0;
}
"""

_LIB = None
_TRIED = False


def _cache_dir() -> str:
    d = os.environ.get("GROWN_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "grown")
    d = os.path.join(d, "csync")
    os.makedirs(d, exist_ok=True)
    return d


def load():
    """The compiled library, or None when it cannot be built (no compiler) or GROWN_NO_CSYNC is set."""
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    if os.environ.get("GROWN_NO_CSYNC"):
        return None
    try:
        h = hashlib.sha1(SRC.encode()).hexdigest()[:16]
        path = os.path.join(_cache_dir(), f"grown_sync_{h}.so")
        if not os.path.exists(path):
            cc = os.environ.get("CC", "cc")
            with tempfile.TemporaryDirectory() as td:
                src = os.path.join(td, "k.c")
                with open(src, "w") as f:
                    f.write(SRC)
                tmp = path + f".tmp{os.getpid()}"
                subprocess.run([cc, "-O3", "-shared", "-fPIC", "-o", tmp, src], check=True, capture_output=True)
                os.replace(tmp, path)                    # atomic: concurrent workers cannot load a torn file
        lib = ctypes.CDLL(path)
        vp = ctypes.c_void_p
        lib.grown_sync_run.argtypes = [ctypes.c_int] * 6 + [vp] * 12
        lib.grown_sync_run.restype = ctypes.c_int
        _LIB = lib
    except Exception:          # no compiler, read-only cache, ...: numpy fallback
        _LIB = None
    return _LIB


def run(nets, x, K: int):
    """nets: flattened netlists (same I, n_out); x uint8 [B, S, Kx, I]. -> (out, high, toggles, events)."""
    from ..circuit.netbatch import NetBatch
    return run_batch(NetBatch.from_netlists(nets), None, x, K)


def run_batch(nb, lane_circ, x, K: int):
    """nb: a NetBatch; lane_circ int64 [L] (circuit of each lane, None = lane b runs circuit b); x uint8
    [L, S, Kx, I]. -> (out [L, S, n_out], high, toggles [L], events [L, S]: gate transitions committed in each
    step's window (sK, (s+1)K])."""
    lib = load()
    L, S, Kx, I = x.shape
    n_out = nb.n_out
    x = np.ascontiguousarray(x, np.uint8)
    out = np.zeros((L, S, n_out), np.uint8)
    high = np.zeros((L, S, n_out), np.float32)
    toggles = np.zeros(L, np.float64)
    events = np.zeros((L, S), np.int64)
    c = lambda a, dt: np.ascontiguousarray(a, dt)  # noqa: E731
    arrs = [c(nb.goff, np.int64), c(nb.poff, np.int64), c(nb.fanin, np.int32), c(nb.tt, np.uint64),
            c(nb.init, np.uint8), c(nb.outs, np.int32), x, out, high, toggles]
    lc = None if lane_circ is None else c(lane_circ, np.int64)
    p = lambda a: a.__array_interface__["data"][0]  # noqa: E731
    rc = lib.grown_sync_run(L, S, K, Kx, I, n_out, *[p(a) for a in arrs], None if lc is None else p(lc),
                            p(events))
    if rc:
        raise MemoryError("grown_sync_run: allocation failed")
    return out, high, toggles, events
