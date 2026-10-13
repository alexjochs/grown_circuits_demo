"""Compiled kernels over packed circuits (circuit.packed) stored in "regions": one contiguous buffer per circuit.

Region layout for a circuit with n nodes, P pins and m outputs (byte offsets, in this order, 8-byte aligned blocks
first): ptr int64 [n+1], tt uint64 [n], drive int64 [n], lab uint64 [n], rise int64 [P], fall int64 [P],
src int32 [P], port int32 [P], out_src int32 [m], out_port int32 [m], op int8 [n], init uint8 [n]; total rounded
up to a multiple of 8. `lab` holds the token labels canonical hashing starts from (8-byte BLAKE2b of each node's
length-prefixed Node.key, as canon._circuit_graph computes them).

Kernels (C source COMMON + KERNELS, built on first use like the csync kernel, see grown._cbuild):
  grown_labels      token labels of every node of a region
  grown_hash        canonical hash of a region when colour refinement is discrete (returns 1 when ties remain:
                    the caller falls back to canon's individualization search); byte-identical serialization
                    to canon._serialize and the same BLAKE2b-128
  grown_flatten_regions   NetBatch arrays for a batch of regions (same as circuit.netbatch.flatten_packed)
The breed kernel (moves on regions with numpy's random streams) lives in _cbreed.py and shares COMMON.
"""
from __future__ import annotations

import ctypes

import numpy as np

COMMON = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- BLAKE2b (RFC 7693), unkeyed, digest length 1..64 ---------------------------------------------------------- */
static const uint64_t B2_IV[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
                                  0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
                                  0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
static const uint8_t B2_SIGMA[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};
#define B2_ROTR(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#define B2_G(a, b, c, d, x, y) { a = a + b + x; d = B2_ROTR(d ^ a, 32); c = c + d; b = B2_ROTR(b ^ c, 24); \
    a = a + b + y; d = B2_ROTR(d ^ a, 16); c = c + d; b = B2_ROTR(b ^ c, 63); }

static inline uint64_t b2_load64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }  /* little endian */

static void b2_compress(uint64_t h[8], const uint8_t* block, uint64_t t, int last) {
    uint64_t v[16], m[16];
    for (int i = 0; i < 16; i++) m[i] = b2_load64(block + 8 * i);
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = B2_IV[i]; }
    v[12] ^= t;
    if (last) v[14] = ~v[14];
    for (int r = 0; r < 12; r++) {
        const uint8_t* s = B2_SIGMA[r];
        B2_G(v[0], v[4], v[8], v[12], m[s[0]], m[s[1]]);
        B2_G(v[1], v[5], v[9], v[13], m[s[2]], m[s[3]]);
        B2_G(v[2], v[6], v[10], v[14], m[s[4]], m[s[5]]);
        B2_G(v[3], v[7], v[11], v[15], m[s[6]], m[s[7]]);
        B2_G(v[0], v[5], v[10], v[15], m[s[8]], m[s[9]]);
        B2_G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        B2_G(v[2], v[7], v[8], v[13], m[s[12]], m[s[13]]);
        B2_G(v[3], v[4], v[9], v[14], m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

/* BLAKE2b of data[0..len) with digest length outlen (<= 64) into out. */
static void blake2b(uint8_t* out, int outlen, const uint8_t* data, int64_t len) {
    uint64_t h[8];
    for (int i = 0; i < 8; i++) h[i] = B2_IV[i];
    h[0] ^= 0x01010000ULL ^ (uint64_t)outlen;
    uint8_t block[128];
    int64_t off = 0;
    while (len - off > 128) { b2_compress(h, data + off, (uint64_t)(off + 128), 0); off += 128; }
    int64_t rest = len - off;
    memset(block, 0, 128);
    if (rest > 0) memcpy(block, data + off, (size_t)rest);
    b2_compress(h, block, (uint64_t)len, 1);
    uint8_t full[64];
    memcpy(full, h, 64);
    memcpy(out, full, (size_t)outlen);
}

static inline uint64_t h8(const uint8_t* data, int64_t len) {
    uint8_t d[8]; blake2b(d, 8, data, len); uint64_t v; memcpy(&v, d, 8); return v;
}

/* ---- regions ------------------------------------------------------------------------------------------------------ */
typedef struct {
    int64_t n, P, m;
    int64_t* ptr; uint64_t* tt; int64_t* drive; uint64_t* lab; int64_t* rise; int64_t* fall;
    int32_t* src; int32_t* port; int32_t* out_src; int32_t* out_port; int8_t* op; uint8_t* init;
} region_t;

static int64_t region_size(int64_t n, int64_t P, int64_t m) {
    int64_t s = 8 * (n + 1) + 8 * n * 3 + 8 * P * 2 + 4 * P * 2 + 4 * m * 2 + 2 * n;
    return (s + 7) & ~(int64_t)7;
}

static void region_at(region_t* r, uint8_t* base, int64_t n, int64_t P, int64_t m) {
    r->n = n; r->P = P; r->m = m;
    uint8_t* q = base;
    r->ptr = (int64_t*)q; q += 8 * (n + 1);
    r->tt = (uint64_t*)q; q += 8 * n;
    r->drive = (int64_t*)q; q += 8 * n;
    r->lab = (uint64_t*)q; q += 8 * n;
    r->rise = (int64_t*)q; q += 8 * P;
    r->fall = (int64_t*)q; q += 8 * P;
    r->src = (int32_t*)q; q += 4 * P;
    r->port = (int32_t*)q; q += 4 * P;
    r->out_src = (int32_t*)q; q += 4 * m;
    r->out_port = (int32_t*)q; q += 4 * m;
    r->op = (int8_t*)q; q += n;
    r->init = (uint8_t*)q;
}

/* length-prefixed Node.key bytes of node i (struct "<H" length, then the key); returns the byte count */
static int node_vb(const region_t* r, int64_t i, uint8_t* out) {
    const int op = r->op[i];
    const int64_t a = r->ptr[i], k = r->ptr[i + 1] - a;
    uint8_t* q = out + 2;
    if (op == 0) {
        q[0] = 0; q[1] = (uint8_t)k; memcpy(q + 2, &r->tt[i], 8); q[10] = r->init[i];
        int32_t d = (int32_t)r->drive[i]; memcpy(q + 11, &d, 4);
        for (int64_t j = 0; j < k; j++) { int32_t x = (int32_t)r->rise[a + j]; memcpy(q + 15 + 4 * j, &x, 4); }
        for (int64_t j = 0; j < k; j++) { int32_t x = (int32_t)r->fall[a + j]; memcpy(q + 15 + 4 * (k + j), &x, 4); }
        const uint16_t L = (uint16_t)(15 + 8 * k);
        memcpy(out, &L, 2);
        return 2 + L;
    }
    if (op == 1) {
        q[0] = 1; q[1] = r->init[i];
        int32_t d = (int32_t)r->drive[i], x = (int32_t)r->rise[a], y = (int32_t)r->fall[a];
        memcpy(q + 2, &d, 4); memcpy(q + 6, &x, 4); memcpy(q + 10, &y, 4);
        const uint16_t L = 14; memcpy(out, &L, 2);
        return 16;
    }
    q[0] = 2;
    const uint16_t L = 1; memcpy(out, &L, 2);
    return 3;
}

static inline uint64_t node_lab(const region_t* r, int64_t i) {
    uint8_t buf[2 + 15 + 8 * 64];
    const int len = node_vb(r, i, buf);
    return h8(buf, len);
}
"""

KERNELS = r"""
static const uint64_t P_[9] = {0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
                               0xD6E8FEB86659FD93ULL, 0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL,
                               0x8EBC6AF09C88C6E3ULL, 0x589965CC75374CC3ULL, 0x1D8E4E27C47D124FULL};

static inline uint64_t mix(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 27; x *= 0x94D049BB133111EBULL; return x ^ (x >> 31);
}

void grown_labels(uint8_t* base, int64_t n, int64_t P, int64_t m) {
    region_t r; region_at(&r, base, n, P, m);
    for (int64_t i = 0; i < n; i++) r.lab[i] = node_lab(&r, i);
}

static void blake2b_digest(uint8_t* out, int outlen, const uint8_t* data, int64_t len) { blake2b(out, outlen, data, len); }

void grown_blake2b(uint8_t* out, int64_t outlen, const uint8_t* data, int64_t len) { blake2b_digest(out, (int)outlen, data, len); }

typedef struct { uint64_t lab; int64_t v; } lv_t;
static int lv_cmp(const void* a, const void* b) {
    const uint64_t x = ((const lv_t*)a)->lab, y = ((const lv_t*)b)->lab;
    if (x != y) return x < y ? -1 : 1;
    const int64_t u = ((const lv_t*)a)->v, w = ((const lv_t*)b)->v;
    return u < w ? -1 : (u > w);
}

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

/* Canonical hash of the circuit in a region (n_in inputs): canon.circuit_hash when colour refinement ends
   discrete. Returns 0 and writes 16 bytes to out; 1 when ties remain (no output); -1 on allocation failure. */
int64_t grown_hash(uint8_t* base, int64_t n, int64_t P, int64_t m, int64_t n_in, uint8_t* out) {
    region_t r; region_at(&r, base, n, P, m);
    const int64_t I = n_in, V = I + n, E = P;
    int64_t cap = 1;
    while (cap < 2 * V + 2) cap <<= 1;
    uint64_t* lab = (uint64_t*)malloc(sizeof(uint64_t) * (size_t)(4 * V + 4 + cap));
    uint8_t* occ = (uint8_t*)malloc((size_t)cap);
    int64_t* e = (int64_t*)malloc(sizeof(int64_t) * (size_t)(4 * E + 1));
    lv_t* lv = (lv_t*)malloc(sizeof(lv_t) * (size_t)(V + 1));
    int64_t* rank = (int64_t*)malloc(sizeof(int64_t) * (size_t)(V + 1));
    if (!lab || !occ || !e || !lv || !rank) { free(lab); free(occ); free(e); free(lv); free(rank); return -1; }
    uint64_t *acc_in = lab + V, *acc_out = lab + 2 * V, *nw = lab + 3 * V, *tab = lab + 4 * V + 4;
    int64_t *esrc = e, *edst = e + E, *epin = e + 2 * E, *eport = e + 3 * E;
    uint8_t buf[2 + 15 + 8 * 64];
    for (int64_t i = 0; i < I; i++) {
        const uint16_t L = 6; uint32_t ii = (uint32_t)i;
        memcpy(buf, &L, 2); buf[2] = 0xff; buf[3] = 'I'; memcpy(buf + 4, &ii, 4);
        lab[i] = h8(buf, 8);
    }
    for (int64_t i = 0; i < n; i++) lab[I + i] = r.lab[i];
    for (int64_t j = 0; j < m; j++) {
        const int64_t s = r.out_src[j];
        const int64_t v = s < 0 ? r.out_port[j] : I + s;
        const uint32_t q = s < 0 ? 0u : (uint32_t)r.out_port[j], jj = (uint32_t)j;
        buf[0] = 'O'; memcpy(buf + 1, &jj, 4); memcpy(buf + 5, &q, 4);
        lab[v] += h8(buf, 9);
    }
    for (int64_t i = 0; i < n; i++)
        for (int64_t p = r.ptr[i]; p < r.ptr[i + 1]; p++) {
            const int64_t s = r.src[p];
            edst[p] = I + i; epin[p] = p - r.ptr[i];
            esrc[p] = s < 0 ? r.port[p] : I + s; eport[p] = s < 0 ? 0 : r.port[p];
        }
    /* refinement: canon._refine */
    int64_t ncls = distinct(lab, V, tab, occ, cap);
    for (;;) {
        memset(acc_in, 0, sizeof(uint64_t) * (size_t)V);
        memset(acc_out, 0, sizeof(uint64_t) * (size_t)V);
        for (int64_t k = 0; k < E; k++) {
            const uint64_t pi = (uint64_t)epin[k], po = (uint64_t)eport[k];
            acc_in[edst[k]] += mix(lab[esrc[k]] * P_[0] + po * P_[1] + pi * P_[2] + P_[3]);
            acc_out[esrc[k]] += mix(lab[edst[k]] * P_[4] + pi * P_[5] + po * P_[6] + P_[7]);
        }
        for (int64_t v = 0; v < V; v++)
            nw[v] = mix(lab[v] * P_[8] + mix(acc_in[v] + P_[0]) * P_[1] + mix(acc_out[v] + P_[2]));
        const int64_t c = distinct(nw, V, tab, occ, cap);
        if (c <= ncls) break;
        memcpy(lab, nw, sizeof(uint64_t) * (size_t)V);
        ncls = c;
    }
    if (ncls < V) { free(lab); free(occ); free(e); free(lv); free(rank); return 1; }
    for (int64_t v = 0; v < V; v++) { lv[v].lab = lab[v]; lv[v].v = v; }
    qsort(lv, (size_t)V, sizeof(lv_t), lv_cmp);
    for (int64_t k = 0; k < V; k++) rank[lv[k].v] = k;
    /* serialization: canon._serialize */
    int64_t size = 9 + 8 + 8 * I + 16 * E + 8 * m;
    for (int64_t i = 0; i < n; i++) size += r.op[i] == 0 ? 17 + 8 * (r.ptr[i + 1] - r.ptr[i]) : (r.op[i] == 1 ? 16 : 3);
    uint8_t* ser = (uint8_t*)malloc((size_t)size);
    if (!ser) { free(lab); free(occ); free(e); free(lv); free(rank); return -1; }
    uint8_t* q = ser;
    { uint32_t a = (uint32_t)I, b = (uint32_t)m; q[0] = 'C'; memcpy(q + 1, &a, 4); memcpy(q + 5, &b, 4); q += 9; }
    { uint32_t a = (uint32_t)V, b = (uint32_t)E; memcpy(q, &a, 4); memcpy(q + 4, &b, 4); q += 8; }
    for (int64_t k = 0; k < V; k++) {
        const int64_t v = lv[k].v;
        if (v < I) { const uint16_t L = 6; uint32_t ii = (uint32_t)v; memcpy(q, &L, 2); q[2] = 0xff; q[3] = 'I';
                     memcpy(q + 4, &ii, 4); q += 8; }
        else q += node_vb(&r, v - I, q);
    }
    for (int64_t k = 0; k < V; k++) {
        const int64_t v = lv[k].v;
        if (v < I) continue;
        const int64_t i = v - I;
        for (int64_t p = r.ptr[i]; p < r.ptr[i + 1]; p++) {
            uint32_t w[4] = {(uint32_t)k, (uint32_t)(p - r.ptr[i]), (uint32_t)rank[esrc[p]], (uint32_t)eport[p]};
            memcpy(q, w, 16); q += 16;
        }
    }
    for (int64_t j = 0; j < m; j++) {
        const int64_t s = r.out_src[j];
        uint32_t w[2] = {(uint32_t)rank[s < 0 ? r.out_port[j] : I + s], s < 0 ? 0u : (uint32_t)r.out_port[j]};
        memcpy(q, w, 8); q += 8;
    }
    blake2b(out, 16, ser, (int64_t)(q - ser));
    free(ser); free(lab); free(occ); free(e); free(lv); free(rank);
    return 0;
}

/* NetBatch arrays for B regions (addresses addr[b], sizes nn[b], pp[b]; common n_in and m outputs); the same
   output as grown_flatten_batch in grown._ckern. Returns 0, 1 + b for a zero-delay loop in circuit b, -2 when a
   capacity is exceeded, -1 on allocation failure. */
int64_t grown_flatten_regions(int64_t B, int64_t I, int64_t m, const uint64_t* addr, const int64_t* nn,
                              const int64_t* pp, int64_t gcap, int64_t pcap, int64_t* goff, int64_t* poff,
                              int32_t* fanin, uint64_t* tt, int32_t* dr, int32_t* df, uint8_t* init, int32_t* outs,
                              int64_t* gnode) {
    int64_t nmax = 1;
    for (int64_t b = 0; b < B; b++) if (nn[b] > nmax) nmax = nn[b];
    int64_t* net = (int64_t*)malloc(sizeof(int64_t) * (size_t)nmax);
    int64_t* path = (int64_t*)malloc(sizeof(int64_t) * (size_t)nmax);
    uint8_t* onp = (uint8_t*)calloc((size_t)nmax, 1);
    if (!net || !path || !onp) { free(net); free(path); free(onp); return -1; }
    int64_t G = 0, Pg = 0;
    goff[0] = 0; poff[0] = 0;
    for (int64_t b = 0; b < B; b++) {
        region_t r; region_at(&r, (uint8_t*)(uintptr_t)addr[b], nn[b], pp[b], m);
        const int8_t* op = r.op; const int64_t* ptr = r.ptr; const int32_t* src = r.src; const int32_t* port = r.port;
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
            tt[gi] = op[i] == 1 ? 2u : r.tt[i];
            init[gi] = r.init[i];
            if (Pg + ptr[i + 1] - ptr[i] > pcap) { free(net); free(path); free(onp); return -2; }
            for (int64_t p = ptr[i]; p < ptr[i + 1]; p++) {
                const int64_t s = src[p];
                fanin[Pg] = (int32_t)(s < 0 ? port[p] : net[s]);
                dr[Pg] = (int32_t)r.rise[p];
                df[Pg] = (int32_t)r.fall[p];
                Pg++;
            }
            poff[gi + 1] = Pg;
        }
        for (int64_t j = 0; j < m; j++) {
            const int64_t s = r.out_src[j];
            outs[b * m + j] = (int32_t)(s < 0 ? r.out_port[j] : net[s]);
        }
        G += g;
        goff[b + 1] = G;
    }
    free(net); free(path); free(onp);
    return 0;
}

/* live nodes (backward cone of the outputs) of each region: counts[b] */
int64_t grown_live_regions(int64_t B, int64_t m, const uint64_t* addr, const int64_t* nn, const int64_t* pp,
                           int64_t* counts) {
    int64_t nmax = 1;
    for (int64_t b = 0; b < B; b++) if (nn[b] > nmax) nmax = nn[b];
    uint8_t* live = (uint8_t*)malloc((size_t)nmax);
    int64_t* stack = (int64_t*)malloc(sizeof(int64_t) * (size_t)(2 * nmax + m + 1));
    if (!live || !stack) { free(live); free(stack); return -1; }
    for (int64_t b = 0; b < B; b++) {
        region_t r; region_at(&r, (uint8_t*)(uintptr_t)addr[b], nn[b], pp[b], m);
        const int64_t n = nn[b];
        memset(live, 0, (size_t)(n ? n : 1));
        int64_t sp = 0, cnt = 0;
        for (int64_t j = 0; j < m; j++) if (r.out_src[j] >= 0) stack[sp++] = r.out_src[j];
        while (sp > 0) {
            const int64_t i = stack[--sp];
            if (live[i]) continue;
            live[i] = 1; cnt++;
            for (int64_t p = r.ptr[i]; p < r.ptr[i + 1]; p++)
                if (r.src[p] >= 0 && !live[r.src[p]]) stack[sp++] = r.src[p];
            if (sp > n) { int64_t k = 0; for (int64_t q = 0; q < sp; q++) if (!live[stack[q]]) stack[k++] = stack[q];
                          sp = k; }
        }
        counts[b] = cnt;
    }
    free(live); free(stack);
    return 0;
}
"""

_vp, _i64 = ctypes.c_void_p, ctypes.c_int64
_SIGS = {
    "grown_labels": ([_vp, _i64, _i64, _i64], None),
    "grown_blake2b": ([_vp, _i64, _vp, _i64], None),
    "grown_hash": ([_vp, _i64, _i64, _i64, _i64, _vp], _i64),
    "grown_flatten_regions": ([_i64, _i64, _i64, _vp, _vp, _vp, _i64, _i64] + [_vp] * 9, _i64),
    "grown_live_regions": ([_i64, _i64, _vp, _vp, _vp, _vp], _i64),
}


def lib():
    from .._cbuild import load
    return load("grown_packed", COMMON + KERNELS, _SIGS)


# --- layout (Python mirror of region_at) -----------------------------------------------------------------------------

_FIELDS = (("ptr", np.int64, 1), ("tt", np.uint64, 0), ("drive", np.int64, 0), ("lab", np.uint64, 0),
           ("rise", np.int64, 2), ("fall", np.int64, 2), ("src", np.int32, 2), ("port", np.int32, 2),
           ("out_src", np.int32, 3), ("out_port", np.int32, 3), ("op", np.int8, 0), ("init", np.uint8, 0))


def region_size(n: int, P: int, m: int) -> int:
    s = 8 * (n + 1) + 24 * n + 16 * P + 8 * P + 8 * m + 2 * n
    return (s + 7) & ~7


def region_views(buf: np.ndarray, offset: int, n: int, P: int, m: int) -> dict:
    """Numpy views of a region's arrays (buf: uint8 array)."""
    out = {}
    o = offset
    for name, dt, kind in _FIELDS:
        cnt = (n, n + 1, P, m)[kind]
        nb = np.dtype(dt).itemsize * cnt
        out[name] = buf[o:o + nb].view(dt)
        o += nb
    return out
