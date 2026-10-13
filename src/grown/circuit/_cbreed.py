"""Compiled breeding: UniformOperator's packed path (moves.py `sample_packed` + `apply_packed_move`) for a whole
batch of parents in one call, on regions (see _cpacked), drawing from the caller's numpy Generator through
numpy's own C random library (libnpyrandom: the functions Generator.integers / random / normal call), so it consumes
exactly the same draws and produces the same moves and children as the Python paths. Tested against both in
tests/grown/test_cbreed.py.

Restrictions (the caller uses the Python path otherwise): module-free valid parents, no active library modules
(INSERT_MODULE never applicable), no regenerator, no learned proposal, arity_max >= 1.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

from ._cpacked import COMMON

SRC = r"""
#include <stdbool.h>
#include <math.h>

typedef struct bitgen { void *state; uint64_t (*next_uint64)(void*); uint32_t (*next_uint32)(void*);
                        double (*next_double)(void*); uint64_t (*next_raw)(void*); } bitgen_t;
extern void random_bounded_uint64_fill(bitgen_t*, uint64_t, uint64_t, intptr_t, bool, uint64_t*);
extern double random_normal(bitgen_t*, double, double);

enum { GATE = 0, DELAY = 1, WIRE = 2 };
enum { TT_REPLACE = 0, TT_FLIP, REWIRE, ADD_NODE, DELETE_NODE, DUPLICATE, INSERT_MODULE, ADJUST_PARAM, REGENERATE,
       INLINE_MODULE };
enum { F_RISE = 0, F_FALL, F_INIT, F_DRIVE };
#define INP (-1)
#define OUTC (-2)
#define NONE (-3)
#define EXCL_NONE INT64_MIN
#define MAXDELAY 2147483647LL

/* rng.integers(lo, hi) (hi exclusive) */
static inline int64_t rint2(bitgen_t* g, int64_t lo, int64_t hi) {
    uint64_t out;
    random_bounded_uint64_fill(g, (uint64_t)lo, (uint64_t)(hi - 1 - lo), 1, false, &out);
    return (int64_t)out;
}
static inline int64_t rint1(bitgen_t* g, int64_t n) { return rint2(g, 0, n); }
static inline double rnd(bitgen_t* g) { return g->next_double(g->state); }

static uint64_t random_tt(bitgen_t* g, int64_t k) {
    uint64_t bits[64];
    const int64_t nb = (int64_t)1 << k;
    random_bounded_uint64_fill(g, 0, 1, nb, false, bits);
    uint64_t tt = 0;
    for (int64_t j = 0; j < nb; j++) tt |= bits[j] << j;
    return tt;
}

static inline int bisect_right(const double* cdf, int64_t n, double u) {
    int lo = 0, hi = (int)n;
    while (lo < hi) { int mid = (lo + hi) / 2; if (u < cdf[mid]) hi = mid; else lo = mid + 1; }
    return lo;
}

typedef struct {
    int64_t max_nodes, arity_max, dlo, dhi, drive_max, dup_max, regen_noop, attempts, delay_clamp;
    double p_const, p_splice, sigma;
    const double* kind_cdf;
    const int64_t* tab_n; const int64_t* tab_types; const double* tab_cdf;
} cfg_t;

typedef struct { int64_t type, node, pin, a, b, nd; int64_t d[64]; } move_t;

typedef struct {               /* per-parent sampling context */
    const region_t* p; int64_t n_in, n_wire;
    int64_t* gates; int64_t ng; int gates_ok;
    int64_t* params; int64_t np_; int params_ok;
    int64_t* nop; int nop_ok;
    uint8_t* inS; uint8_t* inF; int64_t* fr; int64_t* S;
} ctx_t;

static void need_gates(ctx_t* c) {
    if (c->gates_ok) return;
    c->ng = 0;
    for (int64_t i = 0; i < c->p->n; i++) if (c->p->op[i] == GATE) c->gates[c->ng++] = i;
    c->gates_ok = 1;
}
static void need_params(ctx_t* c) {
    if (c->params_ok) return;
    c->np_ = 0;
    for (int64_t i = 0; i < c->p->n; i++) if (c->p->op[i] != WIRE) c->params[c->np_++] = i;
    c->params_ok = 1;
}
static void need_nop(ctx_t* c) {
    if (c->nop_ok) return;
    for (int64_t i = 0; i < c->p->n; i++) for (int64_t q = c->p->ptr[i]; q < c->p->ptr[i + 1]; q++) c->nop[q] = i;
    c->nop_ok = 1;
}

static void pin_at(const region_t* p, int64_t idx, int64_t* consumer, int64_t* pin) {
    if (idx >= p->P) { *consumer = OUTC; *pin = idx - p->P; return; }
    int64_t lo = 0, hi = p->n;                      /* largest i with ptr[i] <= idx */
    while (hi - lo > 1) { int64_t mid = (lo + hi) / 2; if (p->ptr[mid] <= idx) lo = mid; else hi = mid; }
    while (lo + 1 <= p->n && p->ptr[lo + 1] <= idx) lo++;
    *consumer = lo; *pin = idx - p->ptr[lo];
}
static void ref_at(const region_t* p, int64_t consumer, int64_t pin, int64_t* s, int64_t* q) {
    if (consumer == OUTC) { *s = p->out_src[pin]; *q = p->out_port[pin]; return; }
    const int64_t a = p->ptr[consumer];
    *s = p->src[a + pin]; *q = p->port[a + pin];
}

/* _f_port: returns 0 for None */
static int f_port(ctx_t* c, bitgen_t* g, int64_t n_extra, int64_t exclude, int64_t* s, int64_t* q) {
    const int64_t n = c->p->n;
    const int64_t N = c->n_in + n + n_extra - ((exclude != EXCL_NONE && exclude >= 0 && exclude < n) ? 1 : 0);
    if (N <= 0) return 0;
    int64_t j = rint1(g, N);
    if (j < c->n_in) { *s = INP; *q = j; return 1; }
    j -= c->n_in;
    if (exclude != EXCL_NONE && exclude >= 0 && exclude <= j) j++;
    *s = j; *q = 0;
    return 1;
}

static int f_gate(ctx_t* c, bitgen_t* g, int64_t* i) {
    need_gates(c);
    if (!c->ng) return 0;
    *i = c->gates[rint1(g, c->ng)];
    return 1;
}

static int f_splice(ctx_t* c, bitgen_t* g, const cfg_t* cf, int64_t* consumer, int64_t* pin) {
    if (rnd(g) >= cf->p_splice) return 0;
    const int64_t npins = c->p->P + c->p->m;
    if (!npins) return 0;
    pin_at(c->p, rint1(g, npins), consumer, pin);
    return 1;
}

static int64_t f_delay(bitgen_t* g, const cfg_t* cf) {
    if (cf->dlo == cf->dhi) return cf->dlo;
    return rint2(g, cf->dlo, cf->dhi + 1);
}

static void add_nbrs(ctx_t* c, int64_t j, int64_t* nf) {
    const region_t* p = c->p;
    for (int64_t q = p->ptr[j]; q < p->ptr[j + 1]; q++) {
        const int64_t s = p->src[q];
        if (s >= 0 && s != j && !c->inS[s] && !c->inF[s]) { c->inF[s] = 1; c->fr[(*nf)++] = s; }
    }
    need_nop(c);
    for (int64_t q = 0; q < p->P; q++)
        if (p->src[q] == j) {
            const int64_t u = c->nop[q];
            if (u != j && !c->inS[u] && !c->inF[u]) { c->inF[u] = 1; c->fr[(*nf)++] = u; }
        }
}

static int cmp_i64(const void* a, const void* b) {
    const int64_t x = *(const int64_t*)a, y = *(const int64_t*)b; return (x > y) - (x < y);
}

/* _f_region: S (sorted) into c->S, returns |S|; marks are cleared on return except inS (cleared by caller) */
static int64_t f_region(ctx_t* c, bitgen_t* g, const cfg_t* cf, int64_t seed) {
    const int64_t size = rint2(g, 1, cf->dup_max + 1);
    int64_t ns = 0, nf = 0;
    c->S[ns++] = seed; c->inS[seed] = 1;
    if (size > 1) add_nbrs(c, seed, &nf);
    while (ns < size && nf > 0) {
        qsort(c->fr, (size_t)nf, sizeof(int64_t), cmp_i64);
        const int64_t k = rint1(g, nf);
        const int64_t j = c->fr[k];
        c->S[ns++] = j; c->inS[j] = 1; c->inF[j] = 0;
        memmove(c->fr + k, c->fr + k + 1, sizeof(int64_t) * (size_t)(nf - k - 1)); nf--;
        add_nbrs(c, j, &nf);
    }
    for (int64_t t = 0; t < nf; t++) c->inF[c->fr[t]] = 0;
    qsort(c->S, (size_t)ns, sizeof(int64_t), cmp_i64);
    return ns;
}

static int64_t new_value(bitgen_t* g, const cfg_t* cf, int f, int64_t old) {
    if (f == F_INIT) return 1 - old;
    if (f == F_DRIVE) {
        int64_t v = old + (rnd(g) < 0.5 ? 1 : -1);
        if (v < 1) v = 1;
        if (v > cf->drive_max) v = cf->drive_max;
        return v != old ? v : (old % cf->drive_max) + 1;
    }
    const double z = random_normal(g, 0.0, cf->sigma);
    const double x = (double)old * pow(2.0, z);
    int64_t v = x >= 4.0e18 ? (int64_t)4e18 : (int64_t)nearbyint(x);
    if (v == old) { const double u = rnd(g); v = old + ((u < 0.5 || old == 1) ? 1 : -1); }
    if (cf->delay_clamp) { if (v < cf->dlo) v = cf->dlo; if (v > cf->dhi) v = cf->dhi; return v; }
    if (v < 1) v = 1;
    if (v > MAXDELAY) v = MAXDELAY;
    return v;
}

/* sample_packed: 1 with a move, 0 when sample() raises (no applicable move / could not sample) */
static int sample(ctx_t* c, bitgen_t* g, const cfg_t* cf, int key, move_t* mv) {
    const region_t* p = c->p;
    const int64_t nt = cf->tab_n[key];
    if (nt == 0) return 0;
    const int64_t* types = cf->tab_types + 10 * key;
    const double* cdf = cf->tab_cdf + 10 * key;
    const int64_t n = p->n, I = c->n_in;
    for (int64_t att = 0; att < cf->attempts; att++) {
        const int64_t t = types[bisect_right(cdf, nt, rnd(g))];
        mv->type = t; mv->node = -1; mv->pin = -1; mv->a = 0; mv->b = 0; mv->nd = 0;
        if (t == TT_REPLACE) {
            int64_t i; if (!f_gate(c, g, &i)) continue;
            const int64_t k = p->ptr[i + 1] - p->ptr[i];
            uint64_t tt = random_tt(g, k);
            if (tt == p->tt[i]) tt ^= (uint64_t)1 << rint1(g, (int64_t)1 << k);
            mv->node = i; mv->a = (int64_t)tt;
            return 1;
        }
        if (t == TT_FLIP) {
            int64_t i; if (!f_gate(c, g, &i)) continue;
            mv->node = i; mv->a = rint1(g, (int64_t)1 << (p->ptr[i + 1] - p->ptr[i]));
            return 1;
        }
        if (t == REWIRE) {
            int64_t consumer, pin, os, oq;
            pin_at(p, rint1(g, p->P + p->m), &consumer, &pin);
            ref_at(p, consumer, pin, &os, &oq);
            const int64_t pos_old = os == INP ? oq : I + os;
            const int64_t mm = I + n - 1;
            if (mm <= 0) continue;
            int64_t j = rint1(g, mm);
            if (j >= pos_old) j++;
            mv->node = consumer; mv->pin = pin;
            if (j < I) { mv->a = INP; mv->b = j; } else { mv->a = j - I; mv->b = 0; }
            return 1;
        }
        if (t == ADD_NODE) {
            const int kind = bisect_right(cf->kind_cdf, 3, rnd(g));
            int64_t k;
            if (kind == 0) k = rnd(g) < cf->p_const ? 0 : rint2(g, 1, cf->arity_max + 1);
            else k = 1;
            int64_t ins[2 * 64];
            for (int64_t j = 0; j < k; j++) f_port(c, g, 1, EXCL_NONE, &ins[2 * j], &ins[2 * j + 1]);
            int64_t sc, sp;
            const int spl = f_splice(c, g, cf, &sc, &sp);
            if (spl && k) ref_at(p, sc, sp, &ins[0], &ins[1]);
            int64_t* d = mv->d; int64_t nd = 0;
            if (kind == 0) {
                const uint64_t tt = random_tt(g, k);
                int64_t rise[64], fall[64];
                for (int64_t j = 0; j < k; j++) rise[j] = f_delay(g, cf);
                for (int64_t j = 0; j < k; j++) fall[j] = f_delay(g, cf);
                const int64_t init = k == 0 ? (int64_t)(tt & 1) : rint1(g, 2);
                d[nd++] = GATE; d[nd++] = k;
                for (int64_t j = 0; j < 2 * k; j++) d[nd++] = ins[j];
                d[nd++] = (int64_t)tt; d[nd++] = init; d[nd++] = 1;
                for (int64_t j = 0; j < k; j++) d[nd++] = rise[j];
                for (int64_t j = 0; j < k; j++) d[nd++] = fall[j];
            } else if (kind == 1) {
                const int64_t r = f_delay(g, cf), f = f_delay(g, cf), init = rint1(g, 2);
                d[nd++] = DELAY; d[nd++] = 1; d[nd++] = ins[0]; d[nd++] = ins[1];
                d[nd++] = init; d[nd++] = 1; d[nd++] = r; d[nd++] = f;
            } else {
                d[nd++] = WIRE; d[nd++] = 1; d[nd++] = ins[0]; d[nd++] = ins[1];
            }
            mv->nd = nd; mv->node = n;
            if (spl) { mv->a = sc; mv->b = sp; } else { mv->a = NONE; mv->b = 0; }
            return 1;
        }
        if (t == DELETE_NODE) {
            const int64_t i = rint1(g, n);
            int64_t s, q;
            if (!f_port(c, g, 0, i, &s, &q)) continue;
            mv->node = i; mv->a = s; mv->b = q;
            return 1;
        }
        if (t == DUPLICATE || t == REGENERATE) {
            const int64_t seed = rint1(g, n);
            const int64_t ns = f_region(c, g, cf, seed);
            for (int64_t j = 0; j < ns; j++) mv->d[j] = c->S[j];
            mv->nd = ns; mv->node = c->S[0];
            if (t == REGENERATE) { for (int64_t j = 0; j < ns; j++) c->inS[c->S[j]] = 0; return 1; }
            if (n + ns > cf->max_nodes) { for (int64_t j = 0; j < ns; j++) c->inS[c->S[j]] = 0; continue; }
            need_nop(c);
            int64_t next = 0, pick_c = 0, pick_p = 0;
            for (int64_t q = 0; q < p->P; q++)
                if (p->src[q] >= 0 && c->inS[p->src[q]] && !c->inS[c->nop[q]]) next++;
            for (int64_t j = 0; j < p->m; j++) if (p->out_src[j] >= 0 && c->inS[p->out_src[j]]) next++;
            int moved = 0;
            if (next && rnd(g) < 0.5) {
                int64_t idx = rint1(g, next), seen = 0;
                for (int64_t q = 0; q < p->P && !moved; q++)
                    if (p->src[q] >= 0 && c->inS[p->src[q]] && !c->inS[c->nop[q]]) {
                        if (seen++ == idx) { pick_c = c->nop[q]; pick_p = q - p->ptr[pick_c]; moved = 1; }
                    }
                for (int64_t j = 0; j < p->m && !moved; j++)
                    if (p->out_src[j] >= 0 && c->inS[p->out_src[j]]) {
                        if (seen++ == idx) { pick_c = OUTC; pick_p = j; moved = 1; }
                    }
            }
            for (int64_t j = 0; j < ns; j++) c->inS[c->S[j]] = 0;
            if (moved) { mv->a = pick_c; mv->b = pick_p; } else { mv->a = NONE; mv->b = 0; }
            return 1;
        }
        if (t == ADJUST_PARAM) {
            int64_t i;
            if (c->n_wire == 0) i = rint1(g, n);
            else { need_params(c); if (!c->np_) continue; i = c->params[rint1(g, c->np_)]; }
            const int64_t a = p->ptr[i], k = p->ptr[i + 1] - a;
            int f; int64_t pin;
            if (p->op[i] == GATE) {
                const int64_t o = rint1(g, 2 * k + 1 + (k ? 1 : 0));
                if (o < k) { f = F_RISE; pin = o; } else if (o < 2 * k) { f = F_FALL; pin = o - k; }
                else if (o == 2 * k) { f = F_DRIVE; pin = 0; } else { f = F_INIT; pin = 0; }
            } else {
                static const int FS[4] = {F_RISE, F_FALL, F_INIT, F_DRIVE};
                f = FS[rint1(g, 4)]; pin = 0;
            }
            const int64_t old = f == F_RISE ? p->rise[a + pin] : f == F_FALL ? p->fall[a + pin] :
                                f == F_INIT ? (int64_t)p->init[i] : p->drive[i];
            mv->node = i; mv->pin = pin; mv->a = f; mv->b = new_value(g, cf, f, old);
            return 1;
        }
        /* INSERT_MODULE / INLINE_MODULE: never applicable here */
    }
    return 0;
}

/* ---- validation on the child (graph.validate's checks for what an edit can touch) ---------------------------- */
static int ref_ok(const region_t* r, int64_t n_in, int64_t s, int64_t q) {
    if (s == INP) return q >= 0 && q < n_in;
    return s >= 0 && s < r->n && q == 0;
}
static int node_ok(const region_t* r, int64_t n_in, int64_t i) {
    const int64_t a = r->ptr[i], b = r->ptr[i + 1], k = b - a;
    for (int64_t p = a; p < b; p++) if (!ref_ok(r, n_in, r->src[p], r->port[p])) return 0;
    const int op = r->op[i];
    if (op == GATE) {
        if (k > 6) return 0;
        const uint64_t mask = k == 6 ? ~(uint64_t)0 : (((uint64_t)1 << ((uint64_t)1 << k)) - 1);
        if (r->tt[i] & ~mask) return 0;
        if (k == 0 && r->init[i] != (r->tt[i] & 1)) return 0;
    } else if (k != 1) return 0;
    if (op != WIRE) {
        if (r->init[i] > 1) return 0;
        for (int64_t p = a; p < b; p++)
            if (r->rise[p] < 1 || r->fall[p] < 1 || r->rise[p] > MAXDELAY || r->fall[p] > MAXDELAY) return 0;
    }
    return 1;
}
static int wires_ok(const region_t* r, int64_t n_wire, const int64_t* w, int64_t nw) {
    if (!n_wire) return 1;
    for (int64_t t = 0; t < nw; t++) {
        const int64_t s0 = w[t];
        if (r->op[s0] != WIRE) continue;
        int64_t cur = s0, steps = 0;
        for (;;) {
            const int64_t s = r->src[r->ptr[cur]];
            if (s < 0 || r->op[s] != WIRE) break;
            if (s == s0 || steps > n_wire) return 0;
            cur = s; steps++;
        }
    }
    return 1;
}

static void copy_node_arrays(region_t* d, int64_t di, const region_t* s, int64_t si, int64_t cnt) {
    memcpy(d->tt + di, s->tt + si, 8 * (size_t)cnt); memcpy(d->drive + di, s->drive + si, 8 * (size_t)cnt);
    memcpy(d->lab + di, s->lab + si, 8 * (size_t)cnt); memcpy(d->op + di, s->op + si, (size_t)cnt);
    memcpy(d->init + di, s->init + si, (size_t)cnt);
}
static void copy_pin_arrays(region_t* d, int64_t di, const region_t* s, int64_t si, int64_t cnt) {
    memcpy(d->rise + di, s->rise + si, 8 * (size_t)cnt); memcpy(d->fall + di, s->fall + si, 8 * (size_t)cnt);
    memcpy(d->src + di, s->src + si, 4 * (size_t)cnt); memcpy(d->port + di, s->port + si, 4 * (size_t)cnt);
}

/* apply + validate. Writes the child region at dst (capacity cap bytes). Returns 1 valid, 0 invalid, 2 the child
   is the parent itself (REGENERATE without a regenerator), -2 capacity. *cn, *cP, *cnw: the child's sizes. */
static int apply(const ctx_t* c, const move_t* mv, uint8_t* dst, int64_t cap, int64_t* cn, int64_t* cP,
                 int64_t* cnw) {
    const region_t* p = c->p;
    const int64_t n = p->n, P = p->P, m = p->m, I = c->n_in;
    region_t r;
    int64_t w[70]; int64_t nw = 0;
    const int t = (int)mv->type;
    if (t == REGENERATE) return 2;
    if (t == TT_REPLACE || t == TT_FLIP || t == REWIRE || t == ADJUST_PARAM) {
        const int64_t sz = region_size(n, P, m);
        if (sz > cap) return -2;
        memcpy(dst, (const void*)p->ptr, (size_t)sz);
        region_at(&r, dst, n, P, m);
        *cn = n; *cP = P; *cnw = c->n_wire;
        if (t == TT_REPLACE || t == TT_FLIP) {
            const int64_t i = mv->node;
            if (p->op[i] != GATE) return 0;
            const int64_t k = p->ptr[i + 1] - p->ptr[i];
            const uint64_t tt = t == TT_REPLACE ? (uint64_t)mv->a : p->tt[i] ^ ((uint64_t)1 << mv->a);
            const uint64_t mask = k == 6 ? ~(uint64_t)0 : (((uint64_t)1 << ((uint64_t)1 << k)) - 1);
            if (tt & ~mask) return 0;
            r.tt[i] = tt;
            if (k == 0) r.init[i] = (uint8_t)(tt & 1);
            r.lab[i] = node_lab(&r, i);
            return 1;
        }
        if (t == REWIRE) {
            if (mv->node == OUTC) { r.out_src[mv->pin] = (int32_t)mv->a; r.out_port[mv->pin] = (int32_t)mv->b; }
            else { const int64_t q = p->ptr[mv->node] + mv->pin; r.src[q] = (int32_t)mv->a; r.port[q] = (int32_t)mv->b; }
            if (!ref_ok(&r, I, mv->a, mv->b)) return 0;
            if (mv->node != OUTC) { w[0] = mv->node; if (!wires_ok(&r, c->n_wire, w, 1)) return 0; }
            return 1;
        }
        /* ADJUST_PARAM */
        const int64_t i = mv->node, a = p->ptr[i], k = p->ptr[i + 1] - a;
        if (p->op[i] == WIRE) return 0;
        if (mv->a == F_RISE || mv->a == F_FALL) {
            if (!(mv->pin >= -k && mv->pin < k)) return 0;
            const int64_t j = a + ((mv->pin % k) + k) % k;
            if (mv->a == F_RISE) r.rise[j] = mv->b; else r.fall[j] = mv->b;
        } else if (mv->a == F_INIT) {
            if (p->op[i] == GATE && k == 0) return 0;
            if (mv->b < 0 || mv->b > 255) return 0;
            r.init[i] = (uint8_t)mv->b;
        } else {
            if (mv->b < -2147483648LL || mv->b > 2147483647LL) return 0;
            r.drive[i] = mv->b;
        }
        if (!node_ok(&r, I, i)) return 0;
        r.lab[i] = node_lab(&r, i);
        return 1;
    }
    if (t == ADD_NODE) {
        const int64_t* d = mv->d;
        const int64_t op = d[0], k = d[1];
        if (k > 6) return 0;
        const int64_t n2 = n + 1, P2 = P + k;
        const int64_t sz = region_size(n2, P2, m);
        if (sz > cap) return -2;
        region_at(&r, dst, n2, P2, m);
        memcpy(r.ptr, p->ptr, 8 * (size_t)(n + 1)); r.ptr[n + 1] = P + k;
        copy_node_arrays(&r, 0, p, 0, n);
        copy_pin_arrays(&r, 0, p, 0, P);
        memcpy(r.out_src, p->out_src, 4 * (size_t)m); memcpy(r.out_port, p->out_port, 4 * (size_t)m);
        for (int64_t j = 0; j < k; j++) { r.src[P + j] = (int32_t)d[2 + 2 * j]; r.port[P + j] = (int32_t)d[3 + 2 * j]; }
        const int64_t x = 2 + 2 * k;
        r.op[n] = (int8_t)op;
        if (op == GATE) {
            r.tt[n] = (uint64_t)d[x]; r.init[n] = (uint8_t)d[x + 1]; r.drive[n] = d[x + 2];
            for (int64_t j = 0; j < k; j++) { r.rise[P + j] = d[x + 3 + j]; r.fall[P + j] = d[x + 3 + k + j]; }
        } else if (op == DELAY) {
            r.tt[n] = 0; r.init[n] = (uint8_t)d[x]; r.drive[n] = d[x + 1]; r.rise[P] = d[x + 2]; r.fall[P] = d[x + 3];
        } else {
            r.tt[n] = 0; r.init[n] = 0; r.drive[n] = 1; r.rise[P] = 0; r.fall[P] = 0;
        }
        *cn = n2; *cP = P2; *cnw = c->n_wire + (op == WIRE);
        w[nw++] = n;
        if (mv->a != NONE) {
            if (mv->a == OUTC) { r.out_src[mv->b] = (int32_t)n; r.out_port[mv->b] = 0; }
            else { const int64_t q = r.ptr[mv->a] + mv->b; r.src[q] = (int32_t)n; r.port[q] = 0; w[nw++] = mv->a; }
        }
        if (!node_ok(&r, I, n)) return 0;
        r.lab[n] = node_lab(&r, n);
        if (!wires_ok(&r, *cnw, w, nw)) return 0;
        return 1;
    }
    if (t == DELETE_NODE) {
        const int64_t i = mv->node;
        if (mv->a == i) return 0;
        const int64_t a = p->ptr[i], b = p->ptr[i + 1], k = b - a;
        int64_t rs = mv->a, rq = mv->b;
        if (k > 0 && p->src[a] != i) { rs = p->src[a]; rq = p->port[a]; }
        const int64_t n2 = n - 1, P2 = P - k;
        const int64_t sz = region_size(n2, P2, m);
        if (sz > cap) return -2;
        region_at(&r, dst, n2, P2, m);
        for (int64_t j = 0; j <= i; j++) r.ptr[j] = p->ptr[j];
        for (int64_t j = i + 1; j <= n2; j++) r.ptr[j] = p->ptr[j + 1] - k;
        copy_node_arrays(&r, 0, p, 0, i); copy_node_arrays(&r, i, p, i + 1, n - i - 1);
        copy_pin_arrays(&r, 0, p, 0, a); copy_pin_arrays(&r, a, p, b, P - b);
        const int64_t rs2 = rs > i ? rs - 1 : rs;
        int hit = 0;
        int64_t node = 0;
        for (int64_t q = 0; q < P2; q++) {
            while (node < n2 && r.ptr[node + 1] <= q) node++;
            const int64_t s = r.src[q];
            if (s == i) {
                r.src[q] = (int32_t)rs2; r.port[q] = (int32_t)rq; hit = 1;
                if (c->n_wire && r.op[node] == WIRE && nw < 70) w[nw++] = node;
                else if (c->n_wire && r.op[node] == WIRE) return -3;
            } else if (s > i) r.src[q] = s - 1;
        }
        for (int64_t j = 0; j < m; j++) {
            const int64_t s = p->out_src[j];
            r.out_port[j] = p->out_port[j];
            if (s == i) { r.out_src[j] = (int32_t)rs2; r.out_port[j] = (int32_t)rq; hit = 1; }
            else r.out_src[j] = s > i ? s - 1 : s;
        }
        *cn = n2; *cP = P2; *cnw = c->n_wire - (p->op[i] == WIRE);
        if (hit) {                 /* the redirect target, checked in the parent's numbering */
            if (rs == INP) { if (!(rq >= 0 && rq < I)) return 0; }
            else if (!(rs >= 0 && rs < n && rq == 0)) return 0;
        }
        if (!wires_ok(&r, *cnw, w, nw)) return 0;
        return 1;
    }
    if (t == DUPLICATE) {
        const int64_t ns = mv->nd;
        const int64_t* S = mv->d;
        int64_t kk = 0;
        for (int64_t j = 0; j < ns; j++) kk += p->ptr[S[j] + 1] - p->ptr[S[j]];
        const int64_t n2 = n + ns, P2 = P + kk;
        const int64_t sz = region_size(n2, P2, m);
        if (sz > cap) return -2;
        region_at(&r, dst, n2, P2, m);
        memcpy(r.ptr, p->ptr, 8 * (size_t)(n + 1));
        copy_node_arrays(&r, 0, p, 0, n);
        copy_pin_arrays(&r, 0, p, 0, P);
        memcpy(r.out_src, p->out_src, 4 * (size_t)m); memcpy(r.out_port, p->out_port, 4 * (size_t)m);
        int64_t q2 = P, nwire = c->n_wire;
        for (int64_t j = 0; j < ns; j++) {
            const int64_t s = S[j], a = p->ptr[s], b = p->ptr[s + 1];
            copy_node_arrays(&r, n + j, p, s, 1);
            copy_pin_arrays(&r, q2, p, a, b - a);
            for (int64_t q = a; q < b; q++) {
                const int64_t src = p->src[q];
                if (src >= 0) for (int64_t u = 0; u < ns; u++) if (S[u] == src) { r.src[q2 + q - a] = (int32_t)(n + u); break; }
            }
            q2 += b - a;
            r.ptr[n + j + 1] = q2;
            if (p->op[s] == WIRE) { nwire++; if (nw < 70) w[nw++] = n + j; }
        }
        *cn = n2; *cP = P2; *cnw = nwire;
        if (mv->a != NONE) {
            int64_t rs, rq;
            ref_at(p, mv->a, mv->b, &rs, &rq);
            int64_t pos = -1, apos = -1;
            for (int64_t u = 0; u < ns; u++) { if (S[u] == rs) pos = n + u; if (S[u] == mv->a) apos = u; }
            if (rs < 0 || pos < 0 || (mv->a >= 0 && apos >= 0)) return 0;
            if (mv->a == OUTC) r.out_src[mv->b] = (int32_t)pos;
            else { r.src[p->ptr[mv->a] + mv->b] = (int32_t)pos; w[nw++] = mv->a; }
        }
        if (!wires_ok(&r, nwire, w, nw)) return 0;
        return 1;
    }
    return 0;
}

/* Breed B children. Parent b: region at paddr[b] with pn[b] nodes, pP[b] pins, pnw[b] wires (m outputs, n_in
   inputs, valid). Children are written one after another into out (capacity out_cap bytes). Per child: status
   (0 invalid: mutate raised; 1 a new child; 2 the parent itself), offset, n, P, n_wire, move (type, node, pin, a,
   b) and payload (move.data). Returns 0, -1 allocation failure, -2 out of capacity (status filled up to the
   failing child, whose status is -2). */
int64_t grown_breed(int64_t B, int64_t n_in, int64_t m, const uint64_t* paddr, const int64_t* pn, const int64_t* pP,
                    const int64_t* pnw, const int64_t* ci, const double* cd, const double* kind_cdf,
                    const int64_t* tab_n, const int64_t* tab_types, const double* tab_cdf, void* bitgen,
                    uint8_t* out, int64_t out_cap, int64_t* status, int64_t* coff, int64_t* cn, int64_t* cP,
                    int64_t* cnw, int64_t* mvc, int64_t* pay, int64_t paycap, int64_t* pay_n) {
    bitgen_t* g = (bitgen_t*)bitgen;
    cfg_t cf;
    cf.max_nodes = ci[0]; cf.arity_max = ci[1]; cf.dlo = ci[2]; cf.dhi = ci[3]; cf.drive_max = ci[4];
    cf.dup_max = ci[5]; cf.regen_noop = ci[6]; cf.attempts = ci[7]; cf.delay_clamp = ci[8];
    cf.p_const = cd[0]; cf.p_splice = cd[1]; cf.sigma = cd[2];
    cf.kind_cdf = kind_cdf; cf.tab_n = tab_n; cf.tab_types = tab_types; cf.tab_cdf = tab_cdf;
    int64_t nmax = 1, Pmax = 1;
    for (int64_t b = 0; b < B; b++) { if (pn[b] > nmax) nmax = pn[b]; if (pP[b] > Pmax) Pmax = pP[b]; }
    int64_t* wk = (int64_t*)malloc(sizeof(int64_t) * (size_t)(4 * nmax + Pmax + 8));
    uint8_t* marks = (uint8_t*)calloc((size_t)(2 * nmax + 2), 1);
    if (!wk || !marks) { free(wk); free(marks); return -1; }
    int64_t off = 0;
    move_t mv;
    for (int64_t b = 0; b < B; b++) {
        region_t p; region_at(&p, (uint8_t*)(uintptr_t)paddr[b], pn[b], pP[b], m);
        ctx_t c;
        c.p = &p; c.n_in = n_in; c.n_wire = pnw[b];
        c.gates = wk; c.params = wk + nmax; c.fr = wk + 2 * nmax; c.S = wk + 3 * nmax; c.nop = wk + 4 * nmax;
        c.gates_ok = c.params_ok = c.nop_ok = 0;
        c.inS = marks; c.inF = marks + nmax + 1;
        const int64_t n = p.n;
        int has_gate = 0;
        for (int64_t i = 0; i < n; i++) if (p.op[i] == GATE) { has_gate = 1; break; }
        const int grow = n < cf.max_nodes;
        const int key = has_gate | ((p.P + m > 0 && n_in + n > 1) << 1) | (grow << 2) | ((n > 0) << 3) |
                        ((n > 0 && grow) << 4) | (0 << 5) | ((n > c.n_wire) << 6) | ((n > 0 && cf.regen_noop) << 7);
        status[b] = 0; coff[b] = off; cn[b] = cP[b] = cnw[b] = 0; pay_n[b] = 0;
        for (int64_t j = 0; j < 5; j++) mvc[5 * b + j] = 0;
        mvc[5 * b + 0] = -1;
        for (int64_t att = 0; att < cf.attempts; att++) {
            if (!sample(&c, g, &cf, key, &mv)) break;            /* sample() raised: mutate raises */
            const int rc = apply(&c, &mv, out + off, out_cap - off, &cn[b], &cP[b], &cnw[b]);
            if (rc == -2 || rc == -3) { status[b] = -2; free(wk); free(marks); return -2; }
            if (rc == 0) continue;
            status[b] = rc == 2 ? 2 : 1;
            mvc[5 * b + 0] = mv.type; mvc[5 * b + 1] = mv.node; mvc[5 * b + 2] = mv.pin; mvc[5 * b + 3] = mv.a;
            mvc[5 * b + 4] = mv.b;
            if (mv.nd > paycap) { status[b] = -2; free(wk); free(marks); return -2; }
            for (int64_t j = 0; j < mv.nd; j++) pay[paycap * b + j] = mv.d[j];
            pay_n[b] = mv.nd;
            if (rc == 1) off += region_size(cn[b], cP[b], m);
            else { cn[b] = pn[b]; cP[b] = pP[b]; cnw[b] = pnw[b]; }
            break;
        }
    }
    free(wk); free(marks);
    return 0;
}
"""

_vp, _i64 = ctypes.c_void_p, ctypes.c_int64
_SIGS = {"grown_breed": ([_i64, _i64, _i64] + [_vp] * 11 + [_vp, _i64] + [_vp] * 7 + [_i64, _vp], _i64)}


def lib():
    from .._cbuild import load
    try:
        import numpy as _np
        a = os.path.join(os.path.dirname(_np.__file__), "random", "lib", "libnpyrandom.a")
    except Exception:            # pragma: no cover
        return None
    if not os.path.exists(a):
        return None
    # the static library is part of the build: its path and numpy's version key the cache entry
    tag = f"/* numpy {np.__version__} {a} */\n"
    return load("grown_breed", tag + COMMON + SRC, _SIGS, link=[a, "-lm"])
