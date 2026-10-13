// CUDA backend. Two kernels, both one warp per circuit:
//   k_run           exact per-timestamp stepping (SPEC §5), selected with FR_FLAG_STEPPED
//   k_run_windowed  lookahead windows (cuda_window.cuh), the default
//
// Each circuit keeps an "active list" of its pending transitions (act_t[i], act_g[i]) with entry_of[g] pointing
// back into it; a cancelled or committed entry becomes a hole (act_t = FR_NEVER) and holes are compacted away when
// the list fills up or becomes sparse. A step scans the list once: entries due at t commit (phase A) and the minimum
// over the survivors gives the next time together with anything scheduled in phase B. Persistent warps pull
// circuits from a queue ordered by size (largest first) and run each one to the end of the window.
//
// Multi-step runs (fr_cuda_run_steps, docs/SPEC.md §13) use the same kernels instantiated with Steps = true: every
// step of a whole episode in one launch, per-step outputs written on the device at the step boundaries.
//
// Must match python/freerun/ref.py bit-exactly; tests/test_backends.py checks this.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "fanout.h"
#include "freerun.h"
#include "jitter.h"
#include "steps.h"

namespace {

constexpr unsigned FULL = 0xffffffffu;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int32_t SLACK = 32;  // extra active-list capacity per circuit

struct DevBatch {
  int64_t B;
  const int64_t *net_off, *gate_off, *edge_off;
  const int32_t *n_inputs, *fanin_ptr, *fanin, *dr, *df, *fo_ptr, *fo_edge, *edge_gate;
  const uint64_t* tt;
  // jitter (SPEC §10); all null when no circuit has jitter
  const uint64_t* jit_seed;
  const uint32_t* jit_cid;
  const int32_t* jit_dmin;
  const int64_t* jit_amp;
};

struct DevState {
  uint8_t* val;
  int64_t* pend;
  int32_t* entry_of;
  int64_t* act_t;
  int32_t* act_g;
  int32_t* act_count;
  int32_t* act_valid;
  int32_t *dmin_r, *dmin_f;
  int32_t* stamp;
  int32_t* epoch;
  int32_t* n_causes;
  uint8_t *own_hit, *was_committed;
  int32_t* changed;
  int32_t* touched;
  int64_t *toggles, *high, *seg;
  const uint8_t* obs;  // [N] nets whose toggles / high time are maintained
  const uint8_t* watch;
  int64_t* tr_t;
  int32_t* tr_net;
  uint8_t* tr_v;
  int64_t tr_cap;
  unsigned long long* tr_count;
  int64_t* stats;
  int64_t* n_windows;  // [B] windows processed in the last run (windowed kernels)
  uint64_t* jn;  // [G] jitter transition counters n_g
};

struct DevStim {
  const int64_t* off;
  const int64_t* t;
  const int32_t* net;
  const uint8_t* v;
};

// A multi-step run (freerun.h fr_steps / fr_steps_out) on the device; unused (zero) for plain runs.
struct DevSteps {
  const int64_t *K, *S, *budget;  // [B]; budget null = none
  int64_t offset, s_max;
  int32_t n_out;
  int hold;                  // FR_STEPS_HOLD_AT_END
  const int32_t* out_net;    // [B * n_out] (set_io)
  uint8_t* val;              // [B * s_max * n_out] or null
  int64_t *high, *tog;       // [B * s_max * n_out] or null
  int64_t* events;           // [B * s_max] or null
  int64_t* stop;             // [B] or null
  int64_t* prev;             // [B * n_out * 2] scratch: cumulative high time and toggles at the last boundary
};

struct View {  // one circuit's slices
  int32_t I, G, N, cap;
  const int32_t *fptr, *fanin, *dr, *df, *fo_ptr, *fo_edge, *edge_gate;
  const uint64_t* tt;
  uint8_t* val;
  int64_t* pend;
  int32_t* entry_of;
  int64_t* act_t;
  int32_t* act_g;
  int32_t *dmin_r, *dmin_f, *stamp, *n_causes;
  uint8_t *own_hit, *was_committed;
  int32_t *changed, *touched;
  int64_t *toggles, *high, *seg;
  const uint8_t* obs;
  const uint8_t* watch;
  fr_jitter_view jit;  // jit.dmin == 0: no jitter
  uint64_t* jn;
};

__device__ View view(const DevBatch& d, const DevState& s, int64_t b) {
  View v;
  const int64_t no = d.net_off[b], go = d.gate_off[b], eo = d.edge_off[b];
  v.I = d.n_inputs[b];
  v.G = int32_t(d.gate_off[b + 1] - go);
  v.N = int32_t(d.net_off[b + 1] - no);
  v.cap = v.G + SLACK;
  v.fptr = d.fanin_ptr + go + b;
  v.fanin = d.fanin + eo;
  v.dr = d.dr + eo;
  v.df = d.df + eo;
  v.fo_ptr = d.fo_ptr + no + b;
  v.fo_edge = d.fo_edge + eo;
  v.edge_gate = d.edge_gate + eo;
  v.tt = d.tt + go;
  v.val = s.val + no;
  v.pend = s.pend + go;
  v.entry_of = s.entry_of + go;
  v.act_t = s.act_t + go + SLACK * b;
  v.act_g = s.act_g + go + SLACK * b;
  v.dmin_r = s.dmin_r + go;
  v.dmin_f = s.dmin_f + go;
  v.stamp = s.stamp + go;
  v.n_causes = s.n_causes + go;
  v.own_hit = s.own_hit + go;
  v.was_committed = s.was_committed + go;
  v.changed = s.changed + no;
  v.touched = s.touched + go;
  v.toggles = s.toggles + no;
  v.high = s.high + no;
  v.seg = s.seg + no;
  v.obs = s.obs + no;
  v.watch = s.watch ? s.watch + no : nullptr;
  v.jit = fr_jitter_view{0, 0, 0, nullptr};
  if (d.jit_dmin && d.jit_dmin[b] > 0)
    v.jit = fr_jitter_view{d.jit_seed[b], d.jit_cid[b], d.jit_dmin[b], d.jit_amp + go};
  v.jn = s.jn + go;
  return v;
}

__device__ __forceinline__ unsigned lanemask_lt() {
  unsigned m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}

__device__ __forceinline__ int64_t warp_min(int64_t x) {
  for (int o = 16; o > 0; o >>= 1) x = min(x, (int64_t)__shfl_xor_sync(FULL, (long long)x, o));
  return x;
}

__device__ __forceinline__ int64_t warp_sum(int64_t x) {
  for (int o = 16; o > 0; o >>= 1) x += (int64_t)__shfl_xor_sync(FULL, (long long)x, o);
  return x;
}

__device__ __forceinline__ int eval_gate(const View& c, int32_t g) {
  uint32_t idx = 0;
  for (int32_t e = c.fptr[g], i = 0; e < c.fptr[g + 1]; ++e, ++i) idx |= uint32_t(c.val[c.fanin[e]]) << i;
  return int((c.tt[g] >> idx) & 1u);
}

// Remove holes from the active list, preserving order. Returns the new count (= number of valid entries).
__device__ int32_t compact(View& c, int32_t count, int lane) {
  int32_t w = 0;
  for (int32_t base = 0; base < count; base += 32) {
    const int32_t i = base + lane;
    bool alive = false;
    int64_t at = 0;
    int32_t g = 0;
    if (i < count) {
      at = c.act_t[i];
      g = c.act_g[i];
      alive = at != FR_NEVER;
    }
    __syncwarp();
    const unsigned m = __ballot_sync(FULL, alive);
    if (alive) {
      const int32_t pos = w + __popc(m & lanemask_lt());
      c.act_t[pos] = at;
      c.act_g[pos] = g;
      c.entry_of[g] = pos;
    }
    w += __popc(m);
    __syncwarp();
  }
  return w;
}

__global__ void k_init(DevState s, int64_t G) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < G; i += int64_t(gridDim.x) * blockDim.x) {
    s.pend[i] = FR_NEVER;
    s.entry_of[i] = -1;
    s.dmin_r[i] = INT_MAX;
    s.dmin_f[i] = INT_MAX;
    s.stamp[i] = 0;
    s.n_causes[i] = 0;
    s.own_hit[i] = 0;
    s.was_committed[i] = 0;
    s.jn[i] = 0;
  }
}

// Power-up (SPEC §5): evaluate every gate with all pins.
__global__ void k_powerup(DevBatch d, DevState s) {
  const int lane = threadIdx.x & 31;
  const int64_t wid = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) >> 5;
  const int64_t nw = (int64_t(gridDim.x) * blockDim.x) >> 5;
  for (int64_t b = wid; b < d.B; b += nw) {
    View c = view(d, s, b);
    int32_t count = 0;
    for (int32_t base = 0; base < c.G; base += 32) {
      const int32_t g = base + lane;
      bool app = false;
      int64_t tn = 0;
      if (g < c.G && c.fptr[g] < c.fptr[g + 1]) {
        const int v = eval_gate(c, g);
        if (v != c.val[c.I + g]) {
          int32_t m = INT_MAX;
          for (int32_t e = c.fptr[g]; e < c.fptr[g + 1]; ++e) m = min(m, v ? c.dr[e] : c.df[e]);
          tn = fr_arc_delay(c.jit, g, m, true, c.jn[g]);
          app = true;
        }
      }
      const unsigned msk = __ballot_sync(FULL, app);
      if (app) {
        const int32_t pos = count + __popc(msk & lanemask_lt());
        c.act_t[pos] = tn;
        c.act_g[pos] = g;
        c.entry_of[g] = pos;
        c.pend[g] = tn;
      }
      count += __popc(msk);
    }
    if (lane == 0) {
      s.act_count[b] = count;
      s.act_valid[b] = count;
      s.epoch[b] = 0;
    }
  }
}

template <bool Steps>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
    k_run(DevBatch d, DevState s, DevStim st, const int64_t* order, unsigned long long* next, int64_t t0, int64_t T,
          int stats, DevSteps sp) {
  __shared__ int32_t s_cnt[WARPS_PER_BLOCK];
  const int lane = threadIdx.x & 31;
  int32_t* cnt = &s_cnt[threadIdx.x >> 5];
  const unsigned lt = lanemask_lt();
  for (;;) {
    unsigned long long bi = 0;
    if (lane == 0) bi = atomicAdd(next, 1ull);
    bi = __shfl_sync(FULL, bi, 0);
    if (bi >= (unsigned long long)d.B) return;
    const int64_t b = order[bi];
    View c = view(d, s, b);

    for (int32_t n = lane; n < c.N; n += 32)
      if (c.obs[n]) {
        c.toggles[n] = 0;
        c.high[n] = 0;
        c.seg[n] = t0;
      }
    if (lane == 0) s.tr_count[b] = 0;
    int32_t count = s.act_count[b], valid = s.act_valid[b], epoch = s.epoch[b];
    int64_t si = 0, se = 0;
    if (st.off) {
      si = st.off[b];
      se = st.off[b + 1];
    }
    int64_t n_ev = 0, n_ties = 0, n_cancel = 0, n_adv = 0, first_tie = FR_NEVER;  // per lane

    int64_t tnext = FR_NEVER;
    for (int32_t i = lane; i < count; i += 32) tnext = min(tnext, c.act_t[i]);
    tnext = warp_min(tnext);
    __syncwarp();

    // Multi-step runs: boundary k at w0 + k K closes step k - 1 (warp-uniform state)
    int64_t Kb = 0, Sb = 0, bud = -1, w0 = 0, kb = 0, stop = -1, gev = 0;
    bool frozen = false;
    if constexpr (Steps) {
      Kb = sp.K[b];
      Sb = sp.S[b];
      bud = sp.budget ? sp.budget[b] : -1;
      w0 = t0 + sp.offset;
    }
    auto emit = [&](int64_t k, bool zero) {
      const int64_t tb = w0 + k * Kb;
      for (int32_t j = lane; j < sp.n_out; j += 32) {
        const int32_t n = sp.out_net[b * sp.n_out + j];
        int64_t* pv = sp.prev + 2 * (b * sp.n_out + j);
        uint8_t v = 0;
        int64_t H = 0, Tg = 0;
        if (n >= 0) {
          v = c.val[n];
          H = c.high[n] + (v ? tb - c.seg[n] : 0);
          Tg = c.toggles[n];
        }
        if (k >= 1) {
          const int64_t o = (b * sp.s_max + k - 1) * sp.n_out + j;
          if (sp.val) sp.val[o] = zero ? 0 : v;
          if (sp.high) sp.high[o] = zero ? 0 : H - pv[0];
          if (sp.tog) sp.tog[o] = zero ? 0 : Tg - pv[1];
        }
        pv[0] = H;
        pv[1] = Tg;
      }
      if (k >= 1 && lane == 0 && sp.events) sp.events[b * sp.s_max + k - 1] = zero ? 0 : gev;
      gev = 0;
    };

    for (;;) {
      int64_t t = tnext;
      if (si < se) t = min(t, st.t[si]);
      if constexpr (Steps) {
        __syncwarp();
        for (const int64_t lim = min(t, T); kb <= Sb && w0 + kb * Kb <= lim;) {
          emit(kb, false);
          if (++kb > Sb && sp.hold) frozen = true;
        }
        if (frozen) break;
      }
      if (t >= T) break;

      // ---- Phase A: commit transitions due at t, apply stimulus at t ----
      int32_t nchg = 0;
      int64_t surv = FR_NEVER;
      for (int32_t base = 0; base < count; base += 32) {
        const int32_t i = base + lane;
        bool commit = false;
        int32_t g = 0;
        if (i < count) {
          const int64_t at = c.act_t[i];
          if (at == t) {
            g = c.act_g[i];
            commit = true;
            c.act_t[i] = FR_NEVER;
            c.entry_of[g] = -1;
            c.pend[g] = FR_NEVER;
          } else {
            surv = min(surv, at);
          }
        }
        const unsigned m = __ballot_sync(FULL, commit);
        if (commit) {
          c.changed[nchg + __popc(m & lt)] = c.I + g;
          if (stats) c.was_committed[g] = 1;
        }
        nchg += __popc(m);
      }
      const int32_t ncommit = nchg;
      valid -= ncommit;
      while (si < se && st.t[si] == t) {
        const int64_t j = si + lane;
        const bool take = j < se && st.t[j] == t;
        bool ch = false;
        int32_t n = 0;
        if (take) {
          n = st.net[j];
          ch = c.val[n] != st.v[j];
        }
        const unsigned mt = __ballot_sync(FULL, take), mc = __ballot_sync(FULL, ch);
        if (ch) c.changed[nchg + __popc(mc & lt)] = n;
        nchg += __popc(mc);
        si += __popc(mt);
      }
      __syncwarp();
      for (int32_t k = lane; k < nchg; k += 32) {
        const int32_t n = c.changed[k];
        const uint8_t old = c.val[n];
        if (c.obs[n]) {
          if (old) c.high[n] += t - c.seg[n];
          c.seg[n] = t;
          c.toggles[n] += 1;
        }
        c.val[n] = old ^ 1;
        if (c.watch && c.watch[n]) {
          const unsigned long long pos = atomicAdd(&s.tr_count[b], 1ull);
          if ((int64_t)pos < s.tr_cap) {
            const int64_t o = b * s.tr_cap + (int64_t)pos;
            s.tr_t[o] = t;
            s.tr_net[o] = n;
            s.tr_v[o] = old ^ 1;
          }
        }
      }
      if (lane == 0) n_ev += nchg;
      if (nchg == 0) {  // nothing happened at t (a cancelled transition left a stale minimum)
        tnext = warp_min(surv);
        continue;
      }

      // ---- Phase B: gather touched gates and their min arc delays, then evaluate ----
      if (++epoch == INT_MAX) {
        for (int32_t g = lane; g < c.G; g += 32) c.stamp[g] = 0;
        epoch = 1;
      }
      if (lane == 0) *cnt = 0;
      __syncwarp();
      for (int32_t k = lane; k < nchg; k += 32) {
        const int32_t n = c.changed[k];
        const int32_t q0 = c.fo_ptr[n], q1 = c.fo_ptr[n + 1];
        for (int32_t q = q0; q < q1; ++q) {
          const int32_t e = c.fo_edge[q];
          const int32_t g = c.edge_gate[e];
          atomicMin(&c.dmin_r[g], c.dr[e]);
          atomicMin(&c.dmin_f[g], c.df[e]);
          if (atomicExch(&c.stamp[g], epoch) != epoch) c.touched[atomicAdd(cnt, 1)] = g;
          if (stats) {
            if (q == q0 || c.edge_gate[c.fo_edge[q - 1]] != g) atomicAdd(&c.n_causes[g], 1);
            if (n == c.I + g) c.own_hit[g] = 1;
          }
        }
      }
      __syncwarp();
      const int32_t ntouch = *cnt;
      if (count + ntouch > c.cap || count > 2 * valid + 64) {
        count = compact(c, count, lane);
        valid = count;
      }
      int64_t fresh = FR_NEVER;
      for (int32_t base = 0; base < ntouch; base += 32) {
        const int32_t k = base + lane;
        bool app = false, cancel = false;
        int32_t g = 0;
        int64_t tn = 0;
        if (k < ntouch) {
          g = c.touched[k];
          const int32_t rmin = c.dmin_r[g], fmin = c.dmin_f[g];
          c.dmin_r[g] = INT_MAX;
          c.dmin_f[g] = INT_MAX;
          if (stats) {
            const int causes = c.n_causes[g] + ((c.was_committed[g] && !c.own_hit[g]) ? 1 : 0);
            if (causes >= 2) {
              n_ties += 1;
              first_tie = min(first_tie, t);
            }
            c.n_causes[g] = 0;
            c.own_hit[g] = 0;
          }
          const int y = c.val[c.I + g];
          const int v = eval_gate(c, g);
          const int64_t p = c.pend[g];
          if (v == y) {
            if (p != FR_NEVER) {
              c.act_t[c.entry_of[g]] = FR_NEVER;
              c.entry_of[g] = -1;
              c.pend[g] = FR_NEVER;
              cancel = true;
            }
          } else {
            tn = t + fr_arc_delay(c.jit, g, v ? rmin : fmin, p == FR_NEVER, c.jn[g]);
            if (p == FR_NEVER) {
              app = true;
            } else if (tn < p) {
              c.act_t[c.entry_of[g]] = tn;
              c.pend[g] = tn;
              fresh = min(fresh, tn);
              n_adv += 1;
            }
          }
        }
        const unsigned ma = __ballot_sync(FULL, app), mcn = __ballot_sync(FULL, cancel);
        if (app) {
          const int32_t pos = count + __popc(ma & lt);
          c.act_t[pos] = tn;
          c.act_g[pos] = g;
          c.entry_of[g] = pos;
          c.pend[g] = tn;
          fresh = min(fresh, tn);
        }
        count += __popc(ma);
        valid += __popc(ma) - __popc(mcn);
        n_cancel += cancel;
      }
      if (stats) {
        __syncwarp();
        for (int32_t k = lane; k < ncommit; k += 32) c.was_committed[c.changed[k] - c.I] = 0;
      }
      __syncwarp();
      tnext = warp_min(min(surv, fresh));
      if constexpr (Steps) {
        if (kb >= 1 && kb <= Sb) {  // t lies in step kb - 1
          gev += ncommit;
          if (bud >= 0 && gev > bud) {
            stop = kb - 1;
            frozen = true;
            break;
          }
        }
      }
    }

    if constexpr (Steps) {
      if (frozen) {  // nothing pending and no more stimulus: the circuit holds its state
        for (int32_t g = lane; g < c.G; g += 32) {
          c.pend[g] = FR_NEVER;
          c.entry_of[g] = -1;
        }
        count = valid = 0;
      }
      __syncwarp();
      for (; kb <= Sb; ++kb) emit(kb, stop >= 0 && kb > stop + 1);
      if (lane == 0 && sp.stop) sp.stop[b] = stop;
      __syncwarp();
    }
    for (int32_t n = lane; n < c.N; n += 32)
      if (c.obs[n] && c.val[n]) c.high[n] += T - c.seg[n];
    n_ev = warp_sum(n_ev);
    n_ties = warp_sum(n_ties);
    n_cancel = warp_sum(n_cancel);
    n_adv = warp_sum(n_adv);
    first_tie = warp_min(first_tie);
    if (lane == 0) {
      s.act_count[b] = count;
      s.act_valid[b] = valid;
      s.epoch[b] = epoch;
      int64_t* o = s.stats + b * FR_NSTATS;
      o[FR_STAT_EVENTS] = n_ev;
      o[FR_STAT_TIES] = n_ties;
      o[FR_STAT_FIRST_TIE] = n_ties ? first_tie : -1;
      o[FR_STAT_CANCELS] = n_cancel;
      o[FR_STAT_ADVANCES] = n_adv;
    }
    __syncwarp();
  }
}

#include "cuda_window.cuh"

// ---- kernels behind the host API ---------------------------------------------------------------------------------

// The stimulus of one run call, merged per circuit: first the dense closed-loop inputs (d_off[b + 1] - d_off[b]
// entries at t_now; d_off is the prefix sum of min(n_in, n_inputs[b])), then the host triples. The run kernels
// process stimulus one timestamp group at a time, so the order within a timestamp does not matter.
__global__ void k_merge_stim(int64_t B, int32_t n_in, const uint8_t* in, const int64_t* d_off, int64_t t_now,
                             const int64_t* h_off, const int64_t* h_t, const int32_t* h_net, const uint8_t* h_v,
                             int64_t* m_off, int64_t* m_t, int32_t* m_net, uint8_t* m_v) {
  for (int64_t b = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; b < B; b += int64_t(gridDim.x) * blockDim.x) {
    const int64_t h0 = h_off ? h_off[0] : 0;
    const int64_t hb = h_off ? h_off[b] - h0 : 0, he = h_off ? h_off[b + 1] - h0 : 0;
    const int64_t db = in ? d_off[b] : 0, k = in ? d_off[b + 1] - db : 0;
    const int64_t base = db + hb;
    m_off[b] = base;
    if (b == B - 1) m_off[B] = (in ? d_off[B] : 0) + (h_off ? h_off[B] - h0 : 0);
    for (int64_t i = 0; i < k; ++i) {
      m_t[base + i] = t_now;
      m_net[base + i] = int32_t(i);
      m_v[base + i] = in[b * n_in + i];
    }
    for (int64_t j = hb; j < he; ++j) {
      m_t[base + k + j - hb] = h_t[j];
      m_net[base + k + j - hb] = h_net[j];
      m_v[base + k + j - hb] = h_v[j];
    }
  }
}

// Dense outputs: value at T-, and the high time and toggles over the run (kept by the run kernels for output nets).
__global__ void k_gather_out(int64_t B, int32_t n_out, const int64_t* net_off, const int32_t* out_net, DevState s,
                             uint8_t* ov, int64_t* oh, int64_t* ot) {
  const int64_t K = B * n_out;
  for (int64_t k = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; k < K; k += int64_t(gridDim.x) * blockDim.x) {
    const int32_t n = out_net[k];
    const int64_t gn = n >= 0 ? net_off[k / n_out] + n : 0;
    if (ov) ov[k] = n >= 0 ? s.val[gn] : 0;
    if (oh) oh[k] = n >= 0 ? s.high[gn] : 0;
    if (ot) ot[k] = n >= 0 ? s.toggles[gn] : 0;
  }
}

// Load a state (values, pending times at t_ref) into the selected circuits, translated by `shift` = t_now - t_ref, and
// rebuild their persistent active lists (used by the per-timestamp kernel; the windowed kernel rebuilds its own from
// the pending times at every run). One warp per circuit.
__global__ void k_load_state(DevBatch d, DevState s, const uint8_t* sval, const int64_t* spend, const uint64_t* sjn,
                             int64_t shift, const uint8_t* mask) {
  const int lane = threadIdx.x & 31;
  const int64_t wid = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) >> 5;
  const int64_t nw = (int64_t(gridDim.x) * blockDim.x) >> 5;
  for (int64_t b = wid; b < d.B; b += nw) {
    if (mask && !mask[b]) continue;
    View c = view(d, s, b);
    const int64_t no = d.net_off[b], go = d.gate_off[b];
    for (int32_t n = lane; n < c.N; n += 32) c.val[n] = sval[no + n];
    int32_t count = 0;
    for (int32_t base = 0; base < c.G; base += 32) {
      const int32_t g = base + lane;
      int64_t p = FR_NEVER;
      if (g < c.G) {
        p = spend[go + g];
        if (p != FR_NEVER) p += shift;
        c.jn[g] = sjn ? sjn[go + g] : 0;
        c.pend[g] = p;
        c.entry_of[g] = -1;
      }
      const bool app = p != FR_NEVER;
      const unsigned msk = __ballot_sync(FULL, app);
      if (app) {
        const int32_t pos = count + __popc(msk & lanemask_lt());
        c.act_t[pos] = p;
        c.act_g[pos] = g;
        c.entry_of[g] = pos;
      }
      count += __popc(msk);
    }
    if (lane == 0) {
      s.act_count[b] = count;
      s.act_valid[b] = count;
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------

template <class T>
struct DevBuf {
  T* p = nullptr;
  size_t n = 0;
  void alloc(size_t count) {
    free();
    n = count;
    if (count) cudaMalloc(&p, count * sizeof(T));
  }
  void reserve(size_t count) {  // grow-only, contents not kept
    if (count > n) alloc(count);
  }
  void upload(const T* src, size_t count) {
    alloc(count);
    if (count) cudaMemcpy(p, src, count * sizeof(T), cudaMemcpyHostToDevice);
  }
  void free() {
    if (p) cudaFree(p);
    p = nullptr;
    n = 0;
  }
  DevBuf() = default;
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  ~DevBuf() { free(); }
};

struct Snapshot {
  int64_t t = 0;
  DevBuf<uint8_t> val;
  DevBuf<int64_t> pend;
  DevBuf<uint64_t> jn;  // jitter transition counters
};

struct Engine {
  int64_t B = 0, N = 0, G = 0, E = 0;
  int flags = 0;
  int64_t t_now = 0;
  int64_t tr_cap = 0;
  bool watching = false;
  bool net_obs = true;
  std::string error;
  int grid = 0, n_sm = 0, max_smem = 0;
  DevBuf<int64_t> net_off, gate_off, edge_off, order, order_stepped;
  DevBuf<int32_t> n_inputs, fanin_ptr, fanin, dr, df, fo_ptr, fo_edge, edge_gate;
  DevBuf<uint64_t> tt;
  DevBuf<uint8_t> val, watch, obs, own_hit, was_committed, tr_v;
  DevBuf<int64_t> pend, act_t, toggles, high, seg, tr_t, stats, n_windows;
  DevBuf<int32_t> entry_of, act_g, act_count, act_valid, dmin_r, dmin_f, stamp, epoch, n_causes, changed, touched,
      tr_net;
  DevBuf<unsigned long long> tr_count, next;
  DevBuf<uint64_t> jit_seed, jn;
  DevBuf<uint32_t> jit_cid;
  DevBuf<int32_t> jit_dmin;
  DevBuf<int64_t> jit_amp;
  std::vector<int64_t> h_net_off, h_gate_off, h_edge_off, h_order;
  std::vector<int32_t> h_n_inputs, h_fo_ptr;
  int64_t block_threshold = -1;  // circuits with more gates than this run one per thread block; -1 = automatic
  int block_threads = 256;
  cudaStream_t stream = nullptr;  // all engine work is ordered on this stream ...
  static constexpr int NKS = 4;
  cudaStream_t kstreams[NKS] = {};  // ... except the run kernels, which may run side by side
  cudaEvent_t kt_start = nullptr, kt_stop = nullptr, ev_ext = nullptr, ev_k[NKS] = {};
  double prof[FR_NPROF] = {0, 0, 0};
  bool prof_windows_valid = false;
  // closed-loop I/O
  int32_t n_in = 0, n_out = 0;
  int io_flags = 0;
  std::vector<int32_t> h_out_net;
  std::vector<int64_t> h_dense_k;  // per circuit: number of dense inputs
  int64_t dense_total = 0;
  DevBuf<int32_t> out_net;
  DevBuf<int64_t> dense_off;
  DevBuf<uint8_t> d_in, d_out_val;
  DevBuf<int64_t> d_out_high, d_out_tog;
  // multi-step runs: per-circuit parameters, per-step outputs (when the caller's arrays are on the host), scratch
  DevBuf<int64_t> sp_K, sp_S, sp_bud, sp_prev, sp_high, sp_tog, sp_events, sp_stop;
  DevBuf<uint8_t> sp_val;
  // stimulus of the current run: host triples as uploaded, and merged with the dense inputs
  DevBuf<int64_t> hs_off, hs_t, ms_off, ms_t;
  DevBuf<int32_t> hs_net, ms_net;
  DevBuf<uint8_t> hs_v, ms_v;
  std::vector<std::unique_ptr<Snapshot>> snaps;  // by id; null = freed; 0 = power-up
  // windowed kernel: lookahead per circuit, netlist layouts, launch plan and global scratch for big circuits
  DevBuf<int64_t> L;
  DevBuf<uint4> gin16;
  DevBuf<uint16_t> fo_ptr16, fo_gate16;
  DevBuf<int32_t> fo_gate;
  std::vector<uint8_t> h_packed;  // per circuit: uses the 16-bit packed layout
  struct Segment {                // circuits launched together: same group size, layout and state placement
    int threads;
    bool packed, smem;
    int64_t begin, end;  // range of the launch queue `order`
    int64_t smem_bytes;  // dynamic shared memory per block
    int launch_threads = 0, grid = 0;  // launch configuration, computed once with the plan
    int launch_threads_steps = 0, grid_steps = 0;  // ... and of the multi-step kernels
  };
  std::vector<Segment> plan;
  bool plan_valid = false;
  std::vector<int64_t> h_queue;  // host copy of the launch queue `order`
  int reorder_every = 16, runs_since_reorder = 0;
  void reorder_by_cost();
  DevBuf<unsigned char> scratch;
  DevBuf<int64_t> scratch_off;

  NetArrays nets_arrays() const { return NetArrays{gin16.p, fo_ptr16.p, fo_gate16.p, fo_gate.p}; }
  void make_plan();
  void launch_windowed(const DevStim& st, int64_t T, int stats_flag, const DevSteps* sp = nullptr);

  DevBatch batch() const {
    return DevBatch{B,    net_off.p, gate_off.p, edge_off.p, n_inputs.p,  fanin_ptr.p, fanin.p,
                    dr.p, df.p,      fo_ptr.p,   fo_edge.p,  edge_gate.p, tt.p,
                    jit_seed.p, jit_cid.p, jit_dmin.p, jit_amp.p};
  }
  DevState state() {
    DevState s;
    s.val = val.p;
    s.pend = pend.p;
    s.entry_of = entry_of.p;
    s.act_t = act_t.p;
    s.act_g = act_g.p;
    s.act_count = act_count.p;
    s.act_valid = act_valid.p;
    s.dmin_r = dmin_r.p;
    s.dmin_f = dmin_f.p;
    s.stamp = stamp.p;
    s.epoch = epoch.p;
    s.n_causes = n_causes.p;
    s.own_hit = own_hit.p;
    s.was_committed = was_committed.p;
    s.changed = changed.p;
    s.touched = touched.p;
    s.toggles = toggles.p;
    s.high = high.p;
    s.seg = seg.p;
    s.obs = obs.p;
    s.watch = watching ? watch.p : nullptr;
    s.tr_t = tr_t.p;
    s.tr_net = tr_net.p;
    s.tr_v = tr_v.p;
    s.tr_cap = tr_cap;
    s.tr_count = tr_count.p;
    s.stats = stats.p;
    s.n_windows = n_windows.p;
    s.jn = jn.p;
    return s;
  }
  bool check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      error = std::string(what) + ": " + cudaGetErrorString(e);
      return false;
    }
    return true;
  }
  int fail(const std::string& msg) {
    error = msg;
    return 1;
  }
  int64_t gates(int64_t b) const { return h_gate_off[b + 1] - h_gate_off[b]; }
  int64_t nets(int64_t b) const { return h_net_off[b + 1] - h_net_off[b]; }
  int warps_grid() const { return int(std::max<int64_t>(1, std::min<int64_t>(int64_t(n_sm) * 16, (B + 3) / 4))); }

  // Nets whose toggles and high time the run kernels maintain: all of them, or only the closed-loop outputs.
  void update_obs() {
    std::vector<uint8_t> m(size_t(N), net_obs ? 1 : 0);
    if (!net_obs)
      for (int64_t b = 0; b < B; ++b)
        for (int32_t j = 0; j < n_out; ++j) {
          const int32_t n = h_out_net[b * n_out + j];
          if (n >= 0) m[h_net_off[b] + n] = 1;
        }
    obs.upload(m.data(), m.size());
  }

  int load_state(const Snapshot& S, const uint8_t* mask) {
    DevBuf<uint8_t> d_mask;
    if (mask) d_mask.upload(mask, B);
    k_load_state<<<warps_grid(), 128, 0, stream>>>(batch(), state(), S.val.p, S.pend.p, S.jn.p, t_now - S.t,
                                                   d_mask.p);
    cudaStreamSynchronize(stream);
    return check("load state") ? 0 : 1;
  }

  int run(int64_t T, const fr_stimulus* s, const uint8_t* in, bool dev, uint8_t* ov, int64_t* oh, int64_t* ot,
          cudaStream_t ext);
  int run_steps(const fr_stimulus* s, const fr_steps* p, const fr_steps_out* o, bool dev, cudaStream_t ext);
};

// Bytes of windowed-kernel working state of circuit b in its layout.
static int64_t state_bytes_of(const Engine& E, int64_t b) {
  const int64_t G = E.gates(b), N = E.nets(b), I = E.h_n_inputs[b];
  return E.h_packed[b] ? win_state_bytes<uint16_t>(G, N, I) : win_state_bytes<int32_t>(G, N, I);
}

template <class Grp, class Net, bool Steps = false>
static auto windowed_kernel() {
  return std::is_same<Grp, WarpGroup>::value ? k_run_windowed_warp<Net, Steps> : k_run_windowed_block<Net, Steps>;
}

// Launch configuration of a segment: block size (register use can limit it; a block group works with any multiple of
// 32) and a grid of as many blocks as fit at once. Shared memory above 48 KB (static plus dynamic) needs the opt-in.
template <class Grp, class Net, bool Steps = false>
static void configure(Engine& E, Engine::Segment& sg) {
  auto kern = windowed_kernel<Grp, Net, Steps>();
  const size_t dyn = sg.smem ? size_t(sg.smem_bytes) : 0;
  if (dyn > 0) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, int(dyn));
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, kern);
  int threads = sg.threads;
  while (threads > 32 && threads > fa.maxThreadsPerBlock) threads /= 2;
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, threads, dyn);
  const int grid = int(std::max<int64_t>(1, std::min<int64_t>(int64_t(E.n_sm) * std::max(per_sm, 1), sg.end - sg.begin)));
  (Steps ? sg.launch_threads_steps : sg.launch_threads) = threads;
  (Steps ? sg.grid_steps : sg.grid) = grid;
}

template <class Grp, class Net, bool Steps = false>
static void launch_one(Engine& E, const Engine::Segment& sg, int si, cudaStream_t st_, const DevStim& st, int64_t T,
                       int stats_flag, const DevSteps& sp) {
  auto kern = windowed_kernel<Grp, Net, Steps>();
  const size_t dyn = sg.smem ? size_t(sg.smem_bytes) : 0;
  if (dyn > 0) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, int(dyn));  // per kernel
  kern<<<Steps ? sg.grid_steps : sg.grid, Steps ? sg.launch_threads_steps : sg.launch_threads, dyn, st_>>>(
      E.batch(), E.nets_arrays(), E.state(), st, E.L.p, E.order.p + sg.begin, sg.end - sg.begin, E.next.p + si,
      E.t_now, T, stats_flag, sg.smem ? nullptr : E.scratch.p, E.scratch_off.p, sp);
}

// Route every circuit to a group size, netlist layout and state placement, and order the launch queue by segment
// (largest circuits first within a segment, so the longest-running ones start early).
void Engine::make_plan() {
  const int64_t budget = (flags & FR_FLAG_NO_SMEM) ? -1 : int64_t(max_smem) - int64_t(sizeof(GroupShared)) - 64;
  int64_t smem_per_sm = 0;
  {
    int dev = 0, v = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    smem_per_sm = v;
  }
  struct Route {
    int threads;
    bool packed, smem;
  };
  std::vector<Route> route(B);
  for (int64_t b = 0; b < B; ++b) {
    const int64_t G = gates(b), bytes = state_bytes_of(*this, b);
    int threads;
    if (block_threshold >= 0) {
      threads = G > block_threshold ? block_threads : 32;
    } else {
      // Automatic. The kernel is latency-bound: aim for ~16 resident warps per SM. Shared-memory state limits how many
      // circuits an SM holds, so circuits that fill an SM get more threads; so do the circuits of small batches.
      const int64_t per_sm = bytes <= budget ? std::max<int64_t>(1, smem_per_sm / (bytes + 1024)) : 32;
      int64_t want = 32 * ((16 + std::min<int64_t>(per_sm, 32) - 1) / std::min<int64_t>(per_sm, 32));
      while (want < 1024 && B * want < int64_t(n_sm) * 1024) want *= 2;
      int64_t t = 32;
      while (t < want && t * 2 * 8 <= G) t *= 2;  // at least 8 gates per thread
      threads = int(std::min<int64_t>(t, 1024));
    }
    route[b] = Route{threads, h_packed[b] != 0, bytes <= budget};
  }
  plan.clear();
  std::vector<int64_t> q;
  std::vector<int64_t> soff(B, 0);
  int64_t scratch_total = 0;
  for (int threads = 32; threads <= 1024; threads *= 2)
    for (int packed = 1; packed >= 0; --packed)
      for (int smem = 1; smem >= 0; --smem) {
        Segment sg{threads, packed != 0, smem != 0, int64_t(q.size()), 0, 0};
        for (int64_t i = 0; i < B; ++i) {
          const int64_t b = h_order[i];
          const Route& r = route[b];
          if (r.threads != threads || r.packed != (packed != 0) || r.smem != (smem != 0)) continue;
          q.push_back(b);
          const int64_t bytes = state_bytes_of(*this, b);
          if (r.smem) {
            sg.smem_bytes = std::max(sg.smem_bytes, bytes);
          } else {
            soff[b] = scratch_total;
            scratch_total += win_align(bytes);
          }
        }
        sg.end = int64_t(q.size());
        if (sg.end > sg.begin) plan.push_back(sg);
      }
  for (Segment& sg : plan) {
    const bool warp = sg.threads == 32;
    if (warp && sg.packed) configure<WarpGroup, NetPacked>(*this, sg), configure<WarpGroup, NetPacked, true>(*this, sg);
    else if (warp) configure<WarpGroup, NetGeneric>(*this, sg), configure<WarpGroup, NetGeneric, true>(*this, sg);
    else if (sg.packed) configure<BlockGroup, NetPacked>(*this, sg), configure<BlockGroup, NetPacked, true>(*this, sg);
    else configure<BlockGroup, NetGeneric>(*this, sg), configure<BlockGroup, NetGeneric, true>(*this, sg);
  }
  order.upload(q.data(), q.size());
  h_queue = q;
  runs_since_reorder = 0;
  scratch.reserve(size_t(std::max<int64_t>(scratch_total, 16)));
  scratch_off.upload(soff.data(), soff.size());
  next.reserve(plan.size() + 1);
  plan_valid = true;
}

// Longest first within each segment, by the cost of the last run: windows (fixed per-window work) and events.
void Engine::reorder_by_cost() {
  std::vector<int64_t> win(B), st(B * FR_NSTATS), cost(B);
  cudaMemcpy(win.data(), n_windows.p, B * sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(st.data(), stats.p, B * FR_NSTATS * sizeof(int64_t), cudaMemcpyDeviceToHost);
  for (int64_t b = 0; b < B; ++b) cost[b] = 16 * win[b] + st[b * FR_NSTATS + FR_STAT_EVENTS];
  for (const Segment& sg : plan)
    std::stable_sort(h_queue.begin() + sg.begin, h_queue.begin() + sg.end,
                     [&](int64_t a, int64_t b) { return cost[a] > cost[b]; });
  cudaMemcpyAsync(order.p, h_queue.data(), B * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
}

void Engine::launch_windowed(const DevStim& st, int64_t T, int stats_flag, const DevSteps* sp) {
  if (!plan_valid) make_plan();
  if (reorder_every > 0 && ++runs_since_reorder > reorder_every) {
    reorder_by_cost();
    runs_since_reorder = 1;
  }
  cudaMemsetAsync(next.p, 0, plan.size() * sizeof(unsigned long long), stream);
  cudaEventRecord(kt_start, stream);
  const bool fork = plan.size() > 1;  // several segments run side by side on the kernel streams
  if (fork)
    for (int i = 0; i < NKS; ++i) cudaStreamWaitEvent(kstreams[i], kt_start, 0);
  for (size_t i = 0; i < plan.size(); ++i) {
    const Segment& sg = plan[i];
    const bool warp = sg.threads == 32;
    cudaStream_t s_ = fork ? kstreams[i % NKS] : stream;
    const DevSteps none{};
    if (sp) {
      if (warp && sg.packed) launch_one<WarpGroup, NetPacked, true>(*this, sg, int(i), s_, st, T, stats_flag, *sp);
      else if (warp) launch_one<WarpGroup, NetGeneric, true>(*this, sg, int(i), s_, st, T, stats_flag, *sp);
      else if (sg.packed) launch_one<BlockGroup, NetPacked, true>(*this, sg, int(i), s_, st, T, stats_flag, *sp);
      else launch_one<BlockGroup, NetGeneric, true>(*this, sg, int(i), s_, st, T, stats_flag, *sp);
    } else if (warp && sg.packed) launch_one<WarpGroup, NetPacked>(*this, sg, int(i), s_, st, T, stats_flag, none);
    else if (warp) launch_one<WarpGroup, NetGeneric>(*this, sg, int(i), s_, st, T, stats_flag, none);
    else if (sg.packed) launch_one<BlockGroup, NetPacked>(*this, sg, int(i), s_, st, T, stats_flag, none);
    else launch_one<BlockGroup, NetGeneric>(*this, sg, int(i), s_, st, T, stats_flag, none);
  }
  if (fork)
    for (int i = 0; i < NKS; ++i) {
      cudaEventRecord(ev_k[i], kstreams[i]);
      cudaStreamWaitEvent(stream, ev_k[i], 0);
    }
}

int Engine::run(int64_t T, const fr_stimulus* s, const uint8_t* in, bool dev, uint8_t* ov, int64_t* oh, int64_t* ot,
                cudaStream_t ext) {
  const auto c0 = std::chrono::steady_clock::now();
  if (!error.empty()) return 1;
  if (T < t_now) return fail("cannot run backwards");
  if ((oh && !(io_flags & FR_IO_HIGH)) || (ot && !(io_flags & FR_IO_TOGGLES)))
    return fail("step: output high time / toggles were not requested in set_io");
  if (watching && tr_cap == 0) fr_cuda_set_trace_capacity(this, 1 << 16);
  if (dev) {  // order the caller's stream before our reads of its arrays
    cudaEventRecord(ev_ext, ext);
    cudaStreamWaitEvent(stream, ev_ext, 0);
  }
  // ---- stimulus: dense inputs and host triples, merged on the device ----
  const uint8_t* d_inp = nullptr;
  if (in && n_in > 0) {
    if (dev) {
      d_inp = in;
    } else {
      d_in.reserve(size_t(B * n_in));
      cudaMemcpyAsync(d_in.p, in, size_t(B * n_in), cudaMemcpyHostToDevice, stream);
      d_inp = d_in.p;
    }
  }
  const bool triples = s && s->stim_off && s->stim_off[B] > s->stim_off[0];
  if (triples) {
    const int64_t h0 = s->stim_off[0], M = s->stim_off[B] - h0;
    hs_off.reserve(B + 1);
    hs_t.reserve(M);
    hs_net.reserve(M);
    hs_v.reserve(M);
    cudaMemcpyAsync(hs_off.p, s->stim_off, (B + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(hs_t.p, s->t + h0, M * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(hs_net.p, s->net + h0, M * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(hs_v.p, s->v + h0, M, cudaMemcpyHostToDevice, stream);
  }
  DevStim st{nullptr, nullptr, nullptr, nullptr};
  if (d_inp || triples) {
    const int64_t M = (d_inp ? dense_total : 0) +
                      (triples ? s->stim_off[B] - s->stim_off[0] : 0);
    ms_off.reserve(B + 1);
    ms_t.reserve(std::max<int64_t>(M, 1));
    ms_net.reserve(std::max<int64_t>(M, 1));
    ms_v.reserve(std::max<int64_t>(M, 1));
    k_merge_stim<<<int(std::min<int64_t>((B + 127) / 128, 4096)), 128, 0, stream>>>(
        B, n_in, d_inp, dense_off.p, t_now, triples ? hs_off.p : nullptr, hs_t.p, hs_net.p, hs_v.p, ms_off.p, ms_t.p,
        ms_net.p, ms_v.p);
    st = DevStim{ms_off.p, ms_t.p, ms_net.p, ms_v.p};
  }
  // ---- simulation kernels, timed between kt_start and kt_stop ----
  const int stats_flag = (flags & FR_FLAG_STATS) ? 1 : 0;
  if (flags & FR_FLAG_STEPPED) {
    cudaMemsetAsync(next.p, 0, sizeof(unsigned long long), stream);
    cudaEventRecord(kt_start, stream);
    k_run<false><<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(batch(), state(), st, order_stepped.p, next.p, t_now, T,
                                                            stats_flag, DevSteps{});
  } else {
    launch_windowed(st, T, stats_flag);
  }
  cudaEventRecord(kt_stop, stream);
  // ---- dense outputs ----
  if (n_out > 0 && (ov || oh || ot)) {
    uint8_t* gv = ov;
    int64_t *gh = oh, *gt = ot;
    if (!dev) {
      const size_t K = size_t(B * n_out);
      if (ov) d_out_val.reserve(K), gv = d_out_val.p;
      if (oh) d_out_high.reserve(K), gh = d_out_high.p;
      if (ot) d_out_tog.reserve(K), gt = d_out_tog.p;
    }
    const int64_t K = B * n_out;
    k_gather_out<<<int(std::min<int64_t>((K + 255) / 256, 4096)), 256, 0, stream>>>(B, n_out, net_off.p, out_net.p,
                                                                                     state(), gv, gh, gt);
    if (!dev) {
      if (ov) cudaMemcpyAsync(ov, gv, size_t(K), cudaMemcpyDeviceToHost, stream);
      if (oh) cudaMemcpyAsync(oh, gh, size_t(K) * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
      if (ot) cudaMemcpyAsync(ot, gt, size_t(K) * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    }
  }
  cudaStreamSynchronize(stream);
  if (!check("run")) return 1;
  t_now = T;
  float ms = 0;
  cudaEventElapsedTime(&ms, kt_start, kt_stop);
  prof[FR_PROF_KERNEL_S] = ms * 1e-3;
  prof_windows_valid = false;
  if (watching) {
    std::vector<unsigned long long> cnt(B);
    cudaMemcpy(cnt.data(), tr_count.p, B * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    for (int64_t b = 0; b < B; ++b)
      if (int64_t(cnt[b]) > tr_cap)
        return fail("trace capacity exceeded (circuit " + std::to_string(b) + " recorded " + std::to_string(cnt[b]) +
                    " changes, capacity " + std::to_string(tr_cap) + "); increase trace_capacity");
  }
  prof[FR_PROF_CALL_S] = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
  return 0;
}

int Engine::run_steps(const fr_stimulus* s, const fr_steps* p, const fr_steps_out* o_in, bool dev,
                      cudaStream_t ext) {
  const auto c0 = std::chrono::steady_clock::now();
  if (!error.empty()) return 1;
  const fr_steps_out none{nullptr, nullptr, nullptr, nullptr, nullptr};
  const fr_steps_out* o = o_in ? o_in : &none;
  const std::string err = check_steps(B, t_now, p, o, io_flags);
  if (!err.empty()) return fail("run_steps: " + err);
  const int64_t T = steps_end(B, t_now, p);
  if (watching && tr_cap == 0) fr_cuda_set_trace_capacity(this, 1 << 16);
  if (dev) {  // order the caller's stream before our writes to its arrays
    cudaEventRecord(ev_ext, ext);
    cudaStreamWaitEvent(stream, ev_ext, 0);
  }
  // ---- per-circuit step parameters ----
  const size_t nb = size_t(std::max<int64_t>(B, 1));
  sp_K.reserve(nb);
  sp_S.reserve(nb);
  if (B) {
    cudaMemcpyAsync(sp_K.p, p->K, B * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(sp_S.p, p->S, B * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  }
  if (p->budget) {
    sp_bud.reserve(nb);
    if (B) cudaMemcpyAsync(sp_bud.p, p->budget, B * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  }
  sp_prev.reserve(size_t(std::max<int64_t>(2 * B * n_out, 1)));
  // ---- per-step outputs: the caller's device arrays, or ours (copied back below); steps >= S_b stay 0 ----
  const int64_t nO = B * p->s_max * n_out, nE = B * p->s_max;
  uint8_t* gv = o->val;
  int64_t *gh = o->high, *gt = o->tog, *ge = o->events, *gs = o->stop;
  if (!dev) {
    if (gv) sp_val.reserve(size_t(std::max<int64_t>(nO, 1))), gv = sp_val.p;
    if (gh) sp_high.reserve(size_t(std::max<int64_t>(nO, 1))), gh = sp_high.p;
    if (gt) sp_tog.reserve(size_t(std::max<int64_t>(nO, 1))), gt = sp_tog.p;
    if (ge) sp_events.reserve(size_t(std::max<int64_t>(nE, 1))), ge = sp_events.p;
    if (gs) sp_stop.reserve(nb), gs = sp_stop.p;
  }
  if (gv && nO) cudaMemsetAsync(gv, 0, size_t(nO), stream);
  if (gh && nO) cudaMemsetAsync(gh, 0, size_t(nO) * sizeof(int64_t), stream);
  if (gt && nO) cudaMemsetAsync(gt, 0, size_t(nO) * sizeof(int64_t), stream);
  if (ge && nE) cudaMemsetAsync(ge, 0, size_t(nE) * sizeof(int64_t), stream);
  // ---- the whole run's stimulus triples (offsets made relative to the uploaded arrays) ----
  DevStim st{nullptr, nullptr, nullptr, nullptr};
  if (s && s->stim_off && B && s->stim_off[B] > s->stim_off[0]) {
    const int64_t h0 = s->stim_off[0], M = s->stim_off[B] - h0;
    std::vector<int64_t> off(s->stim_off, s->stim_off + B + 1);
    for (auto& x : off) x -= h0;
    hs_off.reserve(B + 1);
    hs_t.reserve(M);
    hs_net.reserve(M);
    hs_v.reserve(M);
    cudaMemcpyAsync(hs_off.p, off.data(), (B + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(hs_t.p, s->t + h0, M * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(hs_net.p, s->net + h0, M * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(hs_v.p, s->v + h0, M, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);  // `off` is a local; pageable copies are staged, but do not rely on it
    st = DevStim{hs_off.p, hs_t.p, hs_net.p, hs_v.p};
  }
  DevSteps sp{sp_K.p, sp_S.p, p->budget ? sp_bud.p : nullptr, p->offset, p->s_max, n_out,
              (p->flags & FR_STEPS_HOLD_AT_END) ? 1 : 0, out_net.p, gv, gh, gt, ge, gs, sp_prev.p};
  // ---- simulation: one launch per kernel segment for the whole episode ----
  const int stats_flag = (flags & FR_FLAG_STATS) ? 1 : 0;
  if (flags & FR_FLAG_STEPPED) {
    cudaMemsetAsync(next.p, 0, sizeof(unsigned long long), stream);
    cudaEventRecord(kt_start, stream);
    k_run<true><<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(batch(), state(), st, order_stepped.p, next.p, t_now, T,
                                                           stats_flag, sp);
  } else {
    launch_windowed(st, T, stats_flag, &sp);
  }
  cudaEventRecord(kt_stop, stream);
  if (!dev) {
    if (o->val && nO) cudaMemcpyAsync(o->val, gv, size_t(nO), cudaMemcpyDeviceToHost, stream);
    if (o->high && nO) cudaMemcpyAsync(o->high, gh, size_t(nO) * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    if (o->tog && nO) cudaMemcpyAsync(o->tog, gt, size_t(nO) * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    if (o->events && nE) cudaMemcpyAsync(o->events, ge, size_t(nE) * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    if (o->stop && B) cudaMemcpyAsync(o->stop, gs, size_t(B) * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  }
  cudaStreamSynchronize(stream);
  if (!check("run_steps")) return 1;
  t_now = T;
  float ms = 0;
  cudaEventElapsedTime(&ms, kt_start, kt_stop);
  prof[FR_PROF_KERNEL_S] = ms * 1e-3;
  prof_windows_valid = false;
  if (watching) {
    std::vector<unsigned long long> cnt(B);
    cudaMemcpy(cnt.data(), tr_count.p, B * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    for (int64_t b = 0; b < B; ++b)
      if (int64_t(cnt[b]) > tr_cap)
        return fail("trace capacity exceeded (circuit " + std::to_string(b) + " recorded " + std::to_string(cnt[b]) +
                    " changes, capacity " + std::to_string(tr_cap) + "); increase trace_capacity");
  }
  prof[FR_PROF_CALL_S] = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
  return 0;
}

}  // namespace

extern "C" {

void* fr_cuda_create(const fr_batch_desc* d, int /*n_threads*/, int flags) {
  auto* E = new Engine();
  const int64_t B = d->n_circuits;
  E->B = B;
  E->flags = flags;
  E->N = d->net_off[B];
  E->G = d->gate_off[B];
  E->E = d->edge_off[B];
  E->h_net_off.assign(d->net_off, d->net_off + B + 1);
  E->h_gate_off.assign(d->gate_off, d->gate_off + B + 1);
  E->h_edge_off.assign(d->edge_off, d->edge_off + B + 1);
  E->h_n_inputs.assign(d->n_inputs, d->n_inputs + B);
  const Fanout fo = batch_fanout(d);
  E->h_fo_ptr = fo.fo_ptr;
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&E->n_sm, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&E->max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaStreamCreateWithFlags(&E->stream, cudaStreamNonBlocking);
  for (auto& st : E->kstreams) cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
  cudaEventCreate(&E->kt_start);
  cudaEventCreate(&E->kt_stop);
  cudaEventCreateWithFlags(&E->ev_ext, cudaEventDisableTiming);
  for (auto& ev : E->ev_k) cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  E->net_off.upload(d->net_off, B + 1);
  E->gate_off.upload(d->gate_off, B + 1);
  E->edge_off.upload(d->edge_off, B + 1);
  E->n_inputs.upload(d->n_inputs, B);
  E->fanin_ptr.upload(d->fanin_ptr, E->G + B);
  E->fanin.upload(d->fanin, E->E);
  E->dr.upload(d->dr, E->E);
  E->df.upload(d->df, E->E);
  E->tt.upload(d->tt, E->G);
  E->val.upload(d->init, E->N);
  E->fo_ptr.upload(fo.fo_ptr.data(), E->N + B);
  E->fo_edge.upload(fo.fo_edge.data(), E->E);
  E->edge_gate.upload(fo.edge_gate.data(), E->E);
  if (d->jit_dmin) {  // jitter (SPEC §10)
    E->jit_seed.upload(d->jit_seed, B);
    E->jit_cid.upload(d->jit_cid, B);
    E->jit_dmin.upload(d->jit_dmin, B);
    E->jit_amp.upload(d->jit_amp, E->G);
  }
  E->jn.alloc(E->G);
  // largest circuits first, so the longest-running ones start early
  std::vector<int64_t> order(B);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return E->gates(a) > E->gates(b); });
  E->order_stepped.upload(order.data(), B);
  E->h_order = order;
  E->pend.alloc(E->G);
  E->entry_of.alloc(E->G);
  E->act_t.alloc(E->G + SLACK * B);
  E->act_g.alloc(E->G + SLACK * B);
  E->act_count.alloc(B);
  E->act_valid.alloc(B);
  E->dmin_r.alloc(E->G);
  E->dmin_f.alloc(E->G);
  E->stamp.alloc(E->G);
  E->epoch.alloc(B);
  E->n_causes.alloc(E->G);
  E->own_hit.alloc(E->G);
  E->was_committed.alloc(E->G);
  E->changed.alloc(E->N);
  E->touched.alloc(E->G);
  E->toggles.alloc(E->N);
  E->high.alloc(E->N);
  E->seg.alloc(E->N);
  E->stats.alloc(B * FR_NSTATS);
  E->n_windows.alloc(B);
  E->tr_count.alloc(B);
  E->next.alloc(1);
  E->update_obs();
  cudaMemset(E->tr_count.p, 0, B * sizeof(unsigned long long));
  cudaMemset(E->n_windows.p, 0, B * sizeof(int64_t));
  if (E->N) {
    cudaMemset(E->toggles.p, 0, E->N * sizeof(int64_t));
    cudaMemset(E->high.p, 0, E->N * sizeof(int64_t));
  }
  {  // windowed kernel: lookahead, netlist layouts
    std::vector<int64_t> L(B);
    for (int64_t b = 0; b < B; ++b) {
      // L = min effective delay (SPEC §10.6): the min arc delay without jitter
      int64_t m = int64_t(1) << 40;  // no arcs: any lookahead is exact
      const int32_t dmin = d->jit_dmin ? d->jit_dmin[b] : 0;
      for (int64_t e = d->edge_off[b]; e < d->edge_off[b + 1]; ++e)
        m = std::min<int64_t>(m, fr_min_effective_delay(std::min(d->dr[e], d->df[e]),
                                                        dmin ? d->jit_amp[d->gate_off[b] + fo.edge_gate[e]] : 0, dmin));
      L[b] = m;
    }
    E->L.upload(L.data(), B);
    std::vector<int32_t> fo_gate(E->E);
    for (int64_t b = 0; b < B; ++b)
      for (int64_t q = d->edge_off[b]; q < d->edge_off[b + 1]; ++q)
        fo_gate[q] = fo.edge_gate[d->edge_off[b] + fo.fo_edge[q]];
    E->fo_gate.upload(fo_gate.data(), fo_gate.size());
    // 16-bit packed layout for circuits with < 2^16 nets (with room for the active-list slack) and < 2^16 edges
    E->h_packed.assign(B, 0);
    std::vector<uint4> gin(E->G, make_uint4(0, 0, 0, 0));
    std::vector<uint16_t> fp16(E->N + B, 0), fg16(E->E, 0);
    for (int64_t b = 0; b < B; ++b) {
      const int64_t N = E->nets(b), G = E->gates(b), Eb = d->edge_off[b + 1] - d->edge_off[b];
      if (N + SLACK >= 0xffff || Eb >= 0xffff || (flags & FR_FLAG_NO_PACK)) continue;
      E->h_packed[b] = 1;
      const int64_t go = d->gate_off[b], eo = d->edge_off[b], no = d->net_off[b];
      const int32_t* fptr = d->fanin_ptr + go + b;
      for (int64_t g = 0; g < G; ++g) {
        uint32_t in[MAX_ARITY] = {0, 0, 0, 0, 0, 0};
        const int32_t k = fptr[g + 1] - fptr[g];
        for (int32_t i = 0; i < k; ++i) in[i] = uint32_t(d->fanin[eo + fptr[g] + i]);
        gin[go + g] = make_uint4(in[0] | in[1] << 16, in[2] | in[3] << 16, in[4] | in[5] << 16,
                                 uint32_t(fptr[g]) | uint32_t(k) << 16);
      }
      for (int64_t n = 0; n <= N; ++n) fp16[no + b + n] = uint16_t(fo.fo_ptr[no + b + n]);
      for (int64_t q = 0; q < Eb; ++q) fg16[eo + q] = uint16_t(fo_gate[eo + q]);
    }
    E->gin16.upload(gin.data(), gin.size());
    E->fo_ptr16.upload(fp16.data(), fp16.size());
    E->fo_gate16.upload(fg16.data(), fg16.size());
  }
  cudaMemset(E->stats.p, 0, B * FR_NSTATS * sizeof(int64_t));
  if (!E->check("allocation")) return E;

  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_run<false>, WARPS_PER_BLOCK * 32, 0);
  E->grid = int(std::max<int64_t>(1, std::min<int64_t>(int64_t(E->n_sm) * std::max(per_sm, 1),
                                                       (B + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK)));
  if (E->G) k_init<<<int(std::min<int64_t>((E->G + 255) / 256, 65535)), 256>>>(E->state(), E->G);
  k_powerup<<<E->grid, WARPS_PER_BLOCK * 32>>>(E->batch(), E->state());
  auto* P = new Snapshot();  // FR_SNAPSHOT_POWERUP
  P->val.alloc(E->N);
  P->pend.alloc(E->G);
  if (E->N) cudaMemcpy(P->val.p, E->val.p, E->N, cudaMemcpyDeviceToDevice);
  if (E->G) cudaMemcpy(P->pend.p, E->pend.p, E->G * sizeof(int64_t), cudaMemcpyDeviceToDevice);
  P->jn.alloc(E->G);
  if (E->G) cudaMemcpy(P->jn.p, E->jn.p, E->G * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
  E->snaps.emplace_back(P);
  cudaDeviceSynchronize();
  E->check("power-up");
  return E;
}

void fr_cuda_destroy(void* h) {
  auto* E = static_cast<Engine*>(h);
  cudaDeviceSynchronize();
  for (auto& st : E->kstreams)
    if (st) cudaStreamDestroy(st);
  if (E->stream) cudaStreamDestroy(E->stream);
  for (cudaEvent_t ev : {E->kt_start, E->kt_stop, E->ev_ext})
    if (ev) cudaEventDestroy(ev);
  for (cudaEvent_t ev : E->ev_k)
    if (ev) cudaEventDestroy(ev);
  delete E;
}

void fr_cuda_set_block_threshold(void* h, int64_t gates, int threads) {
  auto* E = static_cast<Engine*>(h);
  E->block_threshold = gates;  // -1 = automatic
  if (threads > 0) {
    int t = 32;  // a power of two in [32, 1024]
    while (t < 1024 && t * 2 <= threads) t *= 2;
    E->block_threads = t;
  }
  E->plan_valid = false;
}

void fr_cuda_set_reorder(void* h, int every) { static_cast<Engine*>(h)->reorder_every = every; }

void fr_cuda_set_trace_capacity(void* h, int64_t cap) {
  auto* E = static_cast<Engine*>(h);
  E->tr_cap = cap;
  E->tr_t.alloc(size_t(cap * E->B));
  E->tr_net.alloc(size_t(cap * E->B));
  E->tr_v.alloc(size_t(cap * E->B));
}

void fr_cuda_set_watch(void* h, const uint8_t* mask) {
  auto* E = static_cast<Engine*>(h);
  E->watching = mask != nullptr;
  if (mask) E->watch.upload(mask, E->N);
}

void fr_cuda_set_net_observables(void* h, int on) {
  auto* E = static_cast<Engine*>(h);
  E->net_obs = on != 0;
  E->update_obs();
}

int fr_cuda_set_io(void* h, int32_t n_in, int32_t n_out, const int32_t* out_net, int io_flags) {
  auto* E = static_cast<Engine*>(h);
  if (n_in < 0 || n_out < 0 || (n_out > 0 && !out_net)) return E->fail("set_io: bad arguments");
  for (int64_t b = 0; b < E->B; ++b)
    for (int32_t j = 0; j < n_out; ++j) {
      const int32_t n = out_net[b * n_out + j];
      if (n < -1 || n >= E->nets(b))
        return E->fail("set_io: output " + std::to_string(j) + " of circuit " + std::to_string(b) + " is not a net");
    }
  E->n_in = n_in;
  E->n_out = n_out;
  E->io_flags = io_flags;
  E->h_out_net.assign(out_net, out_net + E->B * n_out);
  E->out_net.upload(E->h_out_net.data(), E->h_out_net.size());
  E->h_dense_k.assign(E->B, 0);
  std::vector<int64_t> off(E->B + 1, 0);
  for (int64_t b = 0; b < E->B; ++b) {
    const int32_t k = std::min(n_in, E->h_n_inputs[b]);
    E->h_dense_k[b] = k;
    off[b + 1] = off[b] + k;
  }
  E->dense_off.upload(off.data(), off.size());
  E->dense_total = off[E->B];
  E->update_obs();
  return E->check("set_io") ? 0 : 1;
}

int fr_cuda_step(void* h, int64_t T, const fr_stimulus* s, const uint8_t* in, uint8_t* out_val, int64_t* out_high,
                 int64_t* out_tog) {
  return static_cast<Engine*>(h)->run(T, s, in, false, out_val, out_high, out_tog, nullptr);
}

int fr_cuda_step_device(void* h, int64_t T, const fr_stimulus* s, const uint8_t* in, uint8_t* out_val,
                        int64_t* out_high, int64_t* out_tog, uintptr_t stream) {
  return static_cast<Engine*>(h)->run(T, s, in, true, out_val, out_high, out_tog,
                                      reinterpret_cast<cudaStream_t>(stream));
}

int fr_cuda_run_steps(void* h, const fr_stimulus* s, const fr_steps* p, const fr_steps_out* o) {
  return static_cast<Engine*>(h)->run_steps(s, p, o, false, nullptr);
}

int fr_cuda_run_steps_device(void* h, const fr_stimulus* s, const fr_steps* p, const fr_steps_out* o,
                             uintptr_t stream) {
  return static_cast<Engine*>(h)->run_steps(s, p, o, true, reinterpret_cast<cudaStream_t>(stream));
}

int fr_cuda_run(void* h, int64_t T, const fr_stimulus* s) {
  return static_cast<Engine*>(h)->run(T, s, nullptr, false, nullptr, nullptr, nullptr, nullptr);
}

int64_t fr_cuda_snapshot(void* h) {
  auto* E = static_cast<Engine*>(h);
  auto* S = new Snapshot();
  S->t = E->t_now;
  S->val.alloc(E->N);
  S->pend.alloc(E->G);
  if (E->N) cudaMemcpyAsync(S->val.p, E->val.p, E->N, cudaMemcpyDeviceToDevice, E->stream);
  if (E->G) cudaMemcpyAsync(S->pend.p, E->pend.p, E->G * sizeof(int64_t), cudaMemcpyDeviceToDevice, E->stream);
  S->jn.alloc(E->G);
  if (E->G) cudaMemcpyAsync(S->jn.p, E->jn.p, E->G * sizeof(uint64_t), cudaMemcpyDeviceToDevice, E->stream);
  cudaStreamSynchronize(E->stream);
  E->snaps.emplace_back(S);
  if (!E->check("snapshot")) return -1;
  return int64_t(E->snaps.size()) - 1;
}

int fr_cuda_restore(void* h, int64_t id, const uint8_t* mask) {
  auto* E = static_cast<Engine*>(h);
  if (id < 0 || id >= int64_t(E->snaps.size()) || !E->snaps[id])
    return E->fail("restore: no snapshot " + std::to_string(id));
  return E->load_state(*E->snaps[id], mask);
}

void fr_cuda_snapshot_free(void* h, int64_t id) {
  auto* E = static_cast<Engine*>(h);
  if (id > FR_SNAPSHOT_POWERUP && id < int64_t(E->snaps.size())) E->snaps[id].reset();
}

int fr_cuda_set_state(void* h, int64_t t_ref, const uint8_t* val, const int64_t* pend, const uint64_t* counters,
                      const uint8_t* mask) {
  auto* E = static_cast<Engine*>(h);
  std::vector<int32_t> fptr(E->G + E->B);
  std::vector<uint64_t> tt(E->G);
  cudaMemcpy(fptr.data(), E->fanin_ptr.p, fptr.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(tt.data(), E->tt.p, tt.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  for (int64_t b = 0; b < E->B; ++b) {  // SPEC §2 / §6 constraints on a loadable state
    const int64_t no = E->h_net_off[b], go = E->h_gate_off[b], I = E->h_n_inputs[b];
    for (int64_t n = 0; n < E->nets(b); ++n)
      if (val[no + n] > 1) return E->fail("set_state: circuit " + std::to_string(b) + ": values must be 0 or 1");
    for (int64_t g = 0; g < E->gates(b); ++g) {
      const int64_t p = pend[go + g];
      if (p != FR_NEVER && p < t_ref)
        return E->fail("set_state: circuit " + std::to_string(b) + ": pending time before the state's time");
      if (fptr[go + b + g] == fptr[go + b + g + 1] && (p != FR_NEVER || val[no + I + g] != (tt[go + g] & 1)))
        return E->fail("set_state: circuit " + std::to_string(b) +
                       ": a constant gate must hold its value and have nothing pending");
    }
  }
  Snapshot S;
  S.t = t_ref;
  S.val.upload(val, E->N);
  S.pend.upload(pend, E->G);
  if (counters) S.jn.upload(counters, E->G);  // else null: counters 0
  return E->load_state(S, mask);
}

int64_t fr_cuda_time(void* h) { return static_cast<Engine*>(h)->t_now; }

void fr_cuda_values(void* h, uint8_t* out) {
  auto* E = static_cast<Engine*>(h);
  cudaMemcpy(out, E->val.p, E->N, cudaMemcpyDeviceToHost);
}

void fr_cuda_pending(void* h, int64_t* out) {
  auto* E = static_cast<Engine*>(h);
  cudaMemcpy(out, E->pend.p, E->G * sizeof(int64_t), cudaMemcpyDeviceToHost);
}

void fr_cuda_toggles(void* h, int64_t* out) {
  auto* E = static_cast<Engine*>(h);
  if (E->net_obs) cudaMemcpy(out, E->toggles.p, E->N * sizeof(int64_t), cudaMemcpyDeviceToHost);
  else std::fill(out, out + E->N, 0);
}

void fr_cuda_high(void* h, int64_t* out) {
  auto* E = static_cast<Engine*>(h);
  if (E->net_obs) cudaMemcpy(out, E->high.p, E->N * sizeof(int64_t), cudaMemcpyDeviceToHost);
  else std::fill(out, out + E->N, 0);
}

void fr_cuda_stats(void* h, int64_t* out) {
  auto* E = static_cast<Engine*>(h);
  cudaMemcpy(out, E->stats.p, E->B * FR_NSTATS * sizeof(int64_t), cudaMemcpyDeviceToHost);
}

void fr_cuda_trace_counts(void* h, int64_t* out) {
  auto* E = static_cast<Engine*>(h);
  if (!E->watching) {
    std::fill(out, out + E->B, 0);
    return;
  }
  std::vector<unsigned long long> cnt(E->B);
  cudaMemcpy(cnt.data(), E->tr_count.p, E->B * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  for (int64_t b = 0; b < E->B; ++b) out[b] = std::min<int64_t>(int64_t(cnt[b]), E->tr_cap);
}

void fr_cuda_trace(void* h, int64_t* t, int32_t* net, uint8_t* v) {
  auto* E = static_cast<Engine*>(h);
  if (!E->watching) return;
  std::vector<int64_t> cnt(E->B);
  fr_cuda_trace_counts(h, cnt.data());
  std::vector<int64_t> ht(E->tr_cap);
  std::vector<int32_t> hn(E->tr_cap);
  std::vector<uint8_t> hv(E->tr_cap);
  std::vector<int64_t> idx;
  int64_t k = 0;
  for (int64_t b = 0; b < E->B; ++b) {
    const int64_t m = cnt[b];
    if (!m) continue;
    cudaMemcpy(ht.data(), E->tr_t.p + b * E->tr_cap, m * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(hn.data(), E->tr_net.p + b * E->tr_cap, m * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(hv.data(), E->tr_v.p + b * E->tr_cap, m, cudaMemcpyDeviceToHost);
    idx.resize(m);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int64_t a, int64_t c) { return ht[a] != ht[c] ? ht[a] < ht[c] : hn[a] < hn[c]; });
    for (int64_t i = 0; i < m; ++i, ++k) {
      t[k] = ht[idx[i]];
      net[k] = hn[idx[i]];
      v[k] = hv[idx[i]];
    }
  }
}

void fr_cuda_profile(void* h, double* out) {
  auto* E = static_cast<Engine*>(h);
  if (!E->prof_windows_valid) {
    std::vector<int64_t> w(E->B);
    cudaMemcpy(w.data(), E->n_windows.p, E->B * sizeof(int64_t), cudaMemcpyDeviceToHost);
    E->prof[FR_PROF_WINDOWS] = double(std::accumulate(w.begin(), w.end(), int64_t(0)));
    E->prof_windows_valid = true;
  }
  std::copy(E->prof, E->prof + FR_NPROF, out);
}

void fr_cuda_counters(void* h, uint64_t* out) {
  auto* E = static_cast<Engine*>(h);
  cudaMemcpy(out, E->jn.p, E->G * sizeof(uint64_t), cudaMemcpyDeviceToHost);
}

const char* fr_cuda_error(void* h) { return static_cast<Engine*>(h)->error.c_str(); }

}  // extern "C"
