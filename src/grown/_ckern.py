"""Compiled helpers for the search loop's graph work (built on first use, see _cbuild): colour refinement for
canonical hashing (canon.py), strongly connected components and depth over flattened netlists, and the live-node
cone (archive descriptors). Each has a pure numpy / Python twin with identical results; `lib()` is None when no C
compiler is available and callers use the twin.
"""
from __future__ import annotations

import ctypes

import numpy as np

from ._cbuild import load, ptr

SRC = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t P[9] = {0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
                              0xD6E8FEB86659FD93ULL, 0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL,
                              0x8EBC6AF09C88C6E3ULL, 0x589965CC75374CC3ULL, 0x1D8E4E27C47D124FULL};

static inline uint64_t mix(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 27; x *= 0x94D049BB133111EBULL; return x ^ (x >> 31);
}

/* number of distinct values in a[0..n) (open addressing over a power-of-two table of cap >= 2n slots) */
static int64_t distinct(const uint64_t* a, int64_t n, uint64_t* tab, uint8_t* occ, int64_t cap) {
    memset(occ, 0, (size_t)cap);
    const uint64_t mask = (uint64_t)cap - 1;
    int64_t cnt = 0;
    for (int64_t i = 0; i < n; i++) {
        const uint64_t x = a[i];
        uint64_t h = mix(x ^ 0x2545F4914F6CDD1DULL) & mask;
        int found = 0;
        while (occ[h]) { if (tab[h] == x) { found = 1; break; } h = (h + 1) & mask; }
        if (!found) { occ[h] = 1; tab[h] = x; cnt++; }
    }
    return cnt;
}

/* Colour refinement, the same arithmetic as canon._refine_numpy (uint64 wrap-around; the per-vertex sums are
   order independent). lab [V] is refined in place to the last labelling that increased the number of classes.
   Returns the number of classes, or -1 on allocation failure. */
int64_t grown_refine(int64_t V, int64_t E, uint64_t* lab, const int64_t* src, const int64_t* dst, const int64_t* pin,
                     const int64_t* port) {
    if (V == 0) return 0;
    int64_t cap = 1;
    while (cap < 2 * V) cap <<= 1;
    uint64_t* buf = (uint64_t*)malloc(sizeof(uint64_t) * (size_t)(3 * V + cap));
    uint8_t* occ = (uint8_t*)malloc((size_t)cap);
    if (!buf || !occ) { free(buf); free(occ); return -1; }
    uint64_t *acc_in = buf, *acc_out = buf + V, *nw = buf + 2 * V, *tab = buf + 3 * V;
    int64_t ncls = distinct(lab, V, tab, occ, cap);
    for (;;) {
        memset(acc_in, 0, sizeof(uint64_t) * (size_t)V);
        memset(acc_out, 0, sizeof(uint64_t) * (size_t)V);
        for (int64_t e = 0; e < E; e++) {
            const uint64_t pi = (uint64_t)pin[e], po = (uint64_t)port[e];
            acc_in[dst[e]] += mix(lab[src[e]] * P[0] + po * P[1] + pi * P[2] + P[3]);
            acc_out[src[e]] += mix(lab[dst[e]] * P[4] + pi * P[5] + po * P[6] + P[7]);
        }
        for (int64_t v = 0; v < V; v++)
            nw[v] = mix(lab[v] * P[8] + mix(acc_in[v] + P[0]) * P[1] + mix(acc_out[v] + P[2]));
        const int64_t n = distinct(nw, V, tab, occ, cap);
        if (n <= ncls) break;
        memcpy(lab, nw, sizeof(uint64_t) * (size_t)V);
        ncls = n;
    }
    free(buf); free(occ);
    return ncls;
}

/* Strongly connected components of each circuit's gate graph (gate a -> gate g when a's output net feeds g),
   Tarjan's algorithm, iterative. Circuit b: gates [goff[b], goff[b+1]) (global ids), gate g's pins
   [poff[g], poff[g+1]) index `fanin` (local net ids: 0..I-1 primary inputs, I+j gate j of the circuit).
   out[4b..4b+3] = gates on a cycle, cyclic components (size > 1 or self-looped), largest cyclic component,
   longest path in gates through the condensation from any input-driven gate (0 unless want_depth). */
int grown_scc_batch(int64_t B, int64_t I, const int64_t* goff, const int64_t* poff, const int32_t* fanin,
                    int want_depth, int64_t* out, const uint8_t* which) {
    int64_t gmax = 0, emax = 0;
    for (int64_t b = 0; b < B; b++) {
        int64_t G = goff[b + 1] - goff[b], Ep = poff[goff[b + 1]] - poff[goff[b]];
        if (G > gmax) gmax = G;
        if (Ep > emax) emax = Ep;
    }
    int64_t* w64 = (int64_t*)malloc(sizeof(int64_t) * (size_t)(10 * gmax + 3 + emax));
    uint8_t* w8 = (uint8_t*)malloc((size_t)(3 * gmax + 1));
    if (!w64 || !w8) { free(w64); free(w8); return 1; }
    /* soff, index, low: gmax + 1 entries (index and low double as component offsets and cursors); succ: emax */
    int64_t *soff = w64, *index = soff + gmax + 1, *low = index + gmax + 1, *succ = low + gmax + 1,
            *stk = succ + emax, *comp = stk + gmax, *csv = comp + gmax, *cse = csv + gmax, *csize = cse + gmax,
            *best = csize + gmax, *members = best + gmax;
    uint8_t *onst = w8, *selfl = w8 + gmax, *driven = w8 + 2 * gmax;
    for (int64_t b = 0; b < B; b++) {
        const int64_t g0 = goff[b], G = goff[b + 1] - g0;
        int64_t* o = out + 4 * b;
        o[0] = o[1] = o[2] = o[3] = 0;
        if (G == 0 || (which && !which[b])) continue;
        /* successor lists in pin order */
        memset(soff, 0, sizeof(int64_t) * (size_t)(G + 1));
        memset(selfl, 0, (size_t)G);
        memset(driven, 0, (size_t)G);
        for (int64_t g = 0; g < G; g++)
            for (int64_t p = poff[g0 + g]; p < poff[g0 + g + 1]; p++) {
                const int64_t s = fanin[p];
                if (s >= I) { soff[s - I + 1]++; if (s - I == g) selfl[g] = 1; }
                else driven[g] = 1;
            }
        for (int64_t g = 0; g < G; g++) soff[g + 1] += soff[g];
        for (int64_t g = 0; g < G; g++) csize[g] = soff[g];             /* fill cursor */
        for (int64_t g = 0; g < G; g++)
            for (int64_t p = poff[g0 + g]; p < poff[g0 + g + 1]; p++) {
                const int64_t s = fanin[p];
                if (s >= I) succ[csize[s - I]++] = g;
            }
        for (int64_t g = 0; g < G; g++) { index[g] = -1; onst[g] = 0; }
        int64_t idx = 0, sp = 0, ncomp = 0;
        for (int64_t r = 0; r < G; r++) {
            if (index[r] != -1) continue;
            int64_t csp = 0;
            index[r] = low[r] = idx++; stk[sp++] = r; onst[r] = 1; csv[csp] = r; cse[csp] = soff[r]; csp++;
            while (csp > 0) {
                const int64_t v = csv[csp - 1];
                if (cse[csp - 1] < soff[v + 1]) {
                    const int64_t w = succ[cse[csp - 1]++];
                    if (index[w] == -1) {
                        index[w] = low[w] = idx++; stk[sp++] = w; onst[w] = 1;
                        csv[csp] = w; cse[csp] = soff[w]; csp++;
                    } else if (onst[w] && index[w] < low[v]) low[v] = index[w];
                } else {
                    if (low[v] == index[v]) {
                        int64_t size = 0, w;
                        do { w = stk[--sp]; onst[w] = 0; comp[w] = ncomp; size++; } while (w != v);
                        csize[ncomp] = size;
                        const int cyc = size > 1 || selfl[v];
                        if (cyc) { o[0] += size; o[1]++; if (size > o[2]) o[2] = size; }
                        ncomp++;
                    }
                    csp--;
                    if (csp > 0) { const int64_t u = csv[csp - 1]; if (low[v] < low[u]) low[u] = low[v]; }
                }
            }
        }
        if (!want_depth) continue;
        /* members grouped by component (counting sort); components in reverse emission order are topological */
        for (int64_t c = 0; c <= ncomp; c++) index[c] = 0;                /* reuse: component offsets */
        for (int64_t g = 0; g < G; g++) index[comp[g] + 1]++;
        for (int64_t c = 0; c < ncomp; c++) index[c + 1] += index[c];
        for (int64_t c = 0; c < ncomp; c++) low[c] = index[c];           /* reuse: cursors */
        for (int64_t g = 0; g < G; g++) members[low[comp[g]]++] = g;
        int64_t depth = 0;
        for (int64_t c = 0; c < ncomp; c++) best[c] = 0;
        for (int64_t c = ncomp - 1; c >= 0; c--) {
            if (best[c] == 0)
                for (int64_t m = index[c]; m < index[c + 1]; m++) if (driven[members[m]]) { best[c] = 1; break; }
            if (best[c] == 0) continue;
            if (best[c] > depth) depth = best[c];
            for (int64_t m = index[c]; m < index[c + 1]; m++) {
                const int64_t v = members[m];
                for (int64_t e = soff[v]; e < soff[v + 1]; e++) {
                    const int64_t cw = comp[succ[e]];
                    if (cw != c && best[cw] < best[c] + 1) best[cw] = best[c] + 1;
                }
            }
        }
        o[3] = depth;
    }
    free(w64); free(w8);
    return 0;
}

/* Flatten B packed module-free circuits (circuit.packed) into one netlist batch (circuit.netbatch), the same
   arrays circuit.flatten produces per circuit: gates are the GATE and DELAY nodes in node order (DELAY: truth
   table BUF), wires are aliases resolved to the net they carry. T holds 10 addresses per circuit: op int8 [n],
   ptr int64 [n+1], src int32 [P], port int32 [P], tt uint64 [n], init uint8 [n], rise int64 [P], fall int64 [P],
   out_src int32 [n_out], out_port int32 [n_out]. Capacities gcap / pcap bound the output.
   Returns 0; 1 + b for a zero-delay loop (or a malformed wire) in circuit b; -1 on allocation failure; -2 when the
   capacities are exceeded. */
int64_t grown_flatten_batch(int64_t B, int64_t I, int64_t n_out, const int64_t* nn, const uint64_t* T, int64_t gcap,
                            int64_t pcap, int64_t* goff, int64_t* poff, int32_t* fanin, uint64_t* tt, int32_t* dr,
                            int32_t* df, uint8_t* init, int32_t* outs, int64_t* gnode) {
    int64_t nmax = 1;
    for (int64_t b = 0; b < B; b++) if (nn[b] > nmax) nmax = nn[b];
    int64_t* net = (int64_t*)malloc(sizeof(int64_t) * (size_t)nmax);
    int64_t* path = (int64_t*)malloc(sizeof(int64_t) * (size_t)nmax);
    uint8_t* onp = (uint8_t*)calloc((size_t)nmax, 1);
    if (!net || !path || !onp) { free(net); free(path); free(onp); return -1; }
    int64_t G = 0, Pg = 0;
    goff[0] = 0; poff[0] = 0;
    for (int64_t b = 0; b < B; b++) {
        const int8_t* op = (const int8_t*)T[10 * b + 0];
        const int64_t* ptr = (const int64_t*)T[10 * b + 1];
        const int32_t* src = (const int32_t*)T[10 * b + 2];
        const int32_t* port = (const int32_t*)T[10 * b + 3];
        const uint64_t* tb = (const uint64_t*)T[10 * b + 4];
        const uint8_t* ib = (const uint8_t*)T[10 * b + 5];
        const int64_t* rise = (const int64_t*)T[10 * b + 6];
        const int64_t* fall = (const int64_t*)T[10 * b + 7];
        const int32_t* osrc = (const int32_t*)T[10 * b + 8];
        const int32_t* oport = (const int32_t*)T[10 * b + 9];
        const int64_t n = nn[b];
        int64_t g = 0;
        for (int64_t i = 0; i < n; i++) net[i] = op[i] != 2 ? I + g++ : -1;
        for (int64_t i = 0; i < n; i++) {
            if (net[i] >= 0) continue;
            int64_t cur = i, len = 0, val = 0;
            for (;;) {
                if (op[cur] != 2 || net[cur] >= 0) { val = net[cur]; break; }
                if (onp[cur] || ptr[cur + 1] - ptr[cur] != 1) {
                    for (int64_t q = 0; q < len; q++) onp[path[q]] = 0;
                    free(net); free(path); free(onp); return 1 + b;
                }
                onp[cur] = 1; path[len++] = cur;
                const int64_t s = src[ptr[cur]];
                if (s < 0) { val = port[ptr[cur]]; break; }
                cur = s;
            }
            for (int64_t q = 0; q < len; q++) { net[path[q]] = val; onp[path[q]] = 0; }
        }
        if (G + g > gcap) { free(net); free(path); free(onp); return -2; }
        for (int64_t i = 0; i < n; i++) {
            if (op[i] == 2) continue;
            const int64_t gi = net[i] - I + G;
            gnode[gi] = i;
            tt[gi] = op[i] == 1 ? 2u : tb[i];
            init[gi] = ib[i];
            if (Pg + ptr[i + 1] - ptr[i] > pcap) { free(net); free(path); free(onp); return -2; }
            for (int64_t p = ptr[i]; p < ptr[i + 1]; p++) {
                const int64_t s = src[p];
                fanin[Pg] = (int32_t)(s < 0 ? port[p] : net[s]);
                dr[Pg] = (int32_t)rise[p];
                df[Pg] = (int32_t)fall[p];
                Pg++;
            }
            poff[gi + 1] = Pg;
        }
        for (int64_t j = 0; j < n_out; j++) {
            const int64_t s = osrc[j];
            outs[b * n_out + j] = (int32_t)(s < 0 ? oport[j] : net[s]);
        }
        G += g;
        goff[b + 1] = G;
    }
    free(net); free(path); free(onp);
    return 0;
}

/* Nodes in the backward cone of the outputs. Node i's pins [ptr[i], ptr[i+1]) read node src[p] (-1: input). */
int grown_live(int64_t n, const int64_t* ptr, const int32_t* src, int64_t n_out, const int32_t* out_src,
               uint8_t* live) {
    int64_t* stack = (int64_t*)malloc(sizeof(int64_t) * (size_t)(n + n_out + 1));
    if (!stack) return 1;
    int64_t sp = 0;
    memset(live, 0, (size_t)n);
    for (int64_t j = 0; j < n_out; j++) if (out_src[j] >= 0) stack[sp++] = out_src[j];
    while (sp > 0) {
        const int64_t i = stack[--sp];
        if (live[i]) continue;
        live[i] = 1;
        for (int64_t p = ptr[i]; p < ptr[i + 1]; p++) if (src[p] >= 0 && !live[src[p]]) stack[sp++] = src[p];
        if (sp > n) {                       /* compact: drop entries already live (bounded by n distinct nodes) */
            int64_t k = 0;
            for (int64_t q = 0; q < sp; q++) if (!live[stack[q]]) stack[k++] = stack[q];
            sp = k;
        }
    }
    free(stack);
    return 0;
}
"""

_vp, _i64, _int = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
_SIGS = {
    "grown_refine": ([_i64, _i64, _vp, _vp, _vp, _vp, _vp], _i64),
    "grown_scc_batch": ([_i64, _i64, _vp, _vp, _vp, _int, _vp, _vp], _int),
    "grown_live": ([_i64, _vp, _vp, _i64, _vp, _vp], _int),
    "grown_flatten_batch": ([_i64, _i64, _i64, _vp, _vp, _i64, _i64] + [_vp] * 9, _i64),
}


def lib():
    return load("grown_kern", SRC, _SIGS)


def refine(lab: np.ndarray, src: np.ndarray, dst: np.ndarray, pin: np.ndarray, port: np.ndarray, count=False):
    """Refined copy of lab (uint64 [V]); edge arrays int64 [E] (contiguous). count: also return the number of
    classes."""
    out = np.array(lab, np.uint64, copy=True)
    rc = lib().grown_refine(len(out), len(src), ptr(out), ptr(src), ptr(dst), ptr(pin), ptr(port))
    if rc < 0:
        raise MemoryError("grown_refine: allocation failed")
    return (out, int(rc)) if count else out


def scc_batch(n_inputs: int, goff: np.ndarray, poff: np.ndarray, fanin: np.ndarray, depth: bool,
              which: np.ndarray | None = None) -> np.ndarray:
    """-> int64 [B, 4]: gates on cycles, cyclic SCCs, largest cyclic SCC, depth (see SRC); rows with which[b]
    false are skipped (zeros)."""
    B = len(goff) - 1
    out = np.zeros((B, 4), np.int64)
    goff = np.ascontiguousarray(goff, np.int64)
    poff = np.ascontiguousarray(poff, np.int64)
    fanin = np.ascontiguousarray(fanin, np.int32)
    w = None if which is None else np.ascontiguousarray(which, np.uint8)
    if lib().grown_scc_batch(B, n_inputs, ptr(goff), ptr(poff), ptr(fanin), int(depth), ptr(out),
                             None if w is None else ptr(w)):
        raise MemoryError("grown_scc_batch: allocation failed")
    return out


def live(n: int, ptr_: np.ndarray, src: np.ndarray, out_src: np.ndarray) -> np.ndarray:
    m = np.zeros(n, np.uint8)
    ptr_ = np.ascontiguousarray(ptr_, np.int64)
    src = np.ascontiguousarray(src, np.int32)
    out_src = np.ascontiguousarray(out_src, np.int32)
    if lib().grown_live(n, ptr(ptr_), ptr(src), len(out_src), ptr(out_src), ptr(m)):
        raise MemoryError("grown_live: allocation failed")
    return m.astype(bool)
