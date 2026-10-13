// Windowed CUDA kernel (included by cuda_engine.cu). One cooperating group (a warp, or a thread block) per circuit
// processes lookahead windows [t0, t1), t1 <= t0 + L with L = the circuit's minimum arc delay. The algorithm and its
// exactness argument are in docs/ALGORITHM.md; python/freerun/windowed.py is the reference form. Passes per window:
//   P1 stimulus changes in the window, then candidates (transitions pending in the window) from the active list
//   P2 touched gates: candidates plus the fanout of every net that may change in the window
//   P3 commit decisions for the candidates, iterated to a fixed point
//   P4 replay of every touched gate in time order: its pending state at the end of the window, statistics, and the
//      active list
//   P5 apply the net changes (values, toggles, high time, trace)
//
// Multi-step runs (Steps = true; docs/SPEC.md §13): windows are also cut at the circuit's step boundaries, where the
// per-step outputs are written (values, high time and toggles of the output slots, gate-net transitions), and with an
// event budget the window in which the step's count crosses it is cut right after the crossing time (the decisions of
// earlier candidates do not depend on later ones), after which the circuit is frozen.
//
// Every net changes at most once per window: a gate commits at most once (F2), and a window is cut short before a
// stimulus timestamp that would change an input a second time. So a net's change in the current window is fully
// described by one time per net, wt[n] (earlier windows left smaller values behind), plus, for a gate output, the
// gate's commit decision dec[g]. A gate's input events in the window are found by reading its own fanin; there are
// no per-gate event lists to build or sort.
//
// The per-circuit working state of a run (values, change times, active list, ...) lives in shared memory when it fits
// and in a global scratch slice otherwise; only the base pointer differs. The netlist is read from global memory in
// one of two layouts: 16-bit packed (one 16-byte record per gate, for circuits with < 2^16 nets and edges) or the
// generic 32-bit CSR arrays of the batch.
#pragma once

#include <type_traits>

constexpr int MAX_ARITY = 6;

// ---- groups ---------------------------------------------------------------------------------------------------

struct GroupShared {  // per group (= per block) scratch of the group primitives
  int32_t warp_part[2][32];  // double-buffered: one barrier per count / reduce (see BlockGroup)
  long long red[2][32];
  // Append counters of the window passes. The order of appended items has no effect on any result, so threads take
  // slots with shared-memory atomics; a count is read after the next barrier and reset two barriers later.
  int32_t n_cand, n_touch, n_app, n_hole;
  long long item;
};

// The same for multi-step runs (Steps kernels only): committed gate-net transitions of the current circuit so far, and
// its step parameters (step length, count, budget, start of step 0), kept here rather than in registers (read a few
// times per window).
struct GroupSharedSteps : GroupShared {
  unsigned long long gev;
  int64_t st_K, st_S, st_bud, st_w0;
};

// A whole warp as the group (blockDim.x == 32). Every primitive is also a barrier with memory ordering among the lanes.
struct WarpGroup {
  int rank;
  GroupShared* sh;
  static constexpr int size = 32;
  __device__ void sync() const { __syncwarp(); }
  __device__ bool any(bool p) const {
    __syncwarp();
    return __any_sync(FULL, p);
  }
  __device__ int64_t min(int64_t x) const { return warp_min(x); }
  __device__ int64_t sum(int64_t x) const { return warp_sum(x); }
  // Exclusive prefix count of p over the group, and the total.
  __device__ int32_t count(bool p, int32_t& total) const {
    const unsigned m = __ballot_sync(FULL, p);
    total = __popc(m);
    return __popc(m & lanemask_lt());
  }
  // The same for two predicates at once.
  __device__ void count2(bool p, bool q, int32_t& off_p, int32_t& tot_p, int32_t& off_q, int32_t& tot_q) const {
    off_p = count(p, tot_p);
    off_q = count(q, tot_q);
  }
};

// A thread block of a multiple of 32 threads as the group. Every primitive is a block barrier. count, count2 and
// reduce exchange per-warp partials through shared buffers used alternately, so each needs a single barrier: a buffer
// is written again only two primitives later, after every thread has passed the barrier in between.
struct BlockGroup {
  int rank, size;
  GroupShared* sh;
  mutable int phase = 0;
  __device__ void sync() const { __syncthreads(); }
  __device__ bool any(bool p) const { return __syncthreads_or(p) != 0; }
  template <class Op>
  __device__ long long reduce(long long x, Op op) const {
    const int lane = rank & 31, w = rank >> 5, nw = (size + 31) >> 5;
    long long* red = sh->red[phase ^= 1];
    for (int o = 16; o > 0; o >>= 1) x = op(x, (long long)__shfl_xor_sync(FULL, x, o));
    if (lane == 0) red[w] = x;
    __syncthreads();
    long long r = red[0];
    for (int i = 1; i < nw; ++i) r = op(r, red[i]);
    return r;
  }
  __device__ int64_t min(int64_t x) const {
    return reduce(x, [](long long a, long long b) { return a < b ? a : b; });
  }
  __device__ int64_t sum(int64_t x) const {
    return reduce(x, [](long long a, long long b) { return a + b; });
  }
  // Prefix counts of two predicates; per-warp counts travel packed as (q << 16 | p) (at most 1024 threads).
  __device__ void count2(bool p, bool q, int32_t& off_p, int32_t& tot_p, int32_t& off_q, int32_t& tot_q) const {
    const int lane = rank & 31, w = rank >> 5, nw = (size + 31) >> 5;
    int32_t* part = sh->warp_part[phase ^= 1];
    const unsigned mp = __ballot_sync(FULL, p), mq = __ballot_sync(FULL, q);
    if (lane == 0) part[w] = __popc(mp) | __popc(mq) << 16;
    __syncthreads();
    int32_t before = 0, tot = 0;
    for (int i = 0; i < nw; ++i) {
      const int32_t c = part[i];
      before += i < w ? c : 0;
      tot += c;
    }
    const unsigned lt = lanemask_lt();
    off_p = (before & 0xffff) + __popc(mp & lt);
    off_q = (before >> 16) + __popc(mq & lt);
    tot_p = tot & 0xffff;
    tot_q = tot >> 16;
  }
  __device__ int32_t count(bool p, int32_t& total) const {
    int32_t oq, tq;
    int32_t off;
    count2(p, false, off, total, oq, tq);
    return off;
  }
};

// ---- netlist views --------------------------------------------------------------------------------------------

struct GateIn {  // a gate's function and fanin
  uint64_t tt;
  int32_t e0;  // local index of its first edge (pin 0)
  int32_t k;   // arity
  int32_t net[MAX_ARITY];
};

// Netlist arrays beyond the batch description, built at create time.
struct NetArrays {
  const uint4* gin16;        // [G_total] packed gate records (packed circuits only)
  const uint16_t* fo_ptr16;  // [N_total + B] fanout offsets (packed circuits only)
  const uint16_t* fo_gate16; // [E_total] fanout gates (packed circuits only)
  const int32_t* fo_gate;    // [E_total] fanout gates, 32-bit (all circuits)
};

// Circuits with fewer than 2^16 nets and edges. A gate's record holds its 6 fanin nets, first edge and arity:
// x = in0 | in1 << 16, y = in2 | in3 << 16, z = in4 | in5 << 16, w = e0 | arity << 16.
struct NetPacked {
  using Idx = uint16_t;
  int32_t I, G, N;
  const uint4* gin;
  const uint64_t* tt;
  const int32_t *dr, *df;
  const uint16_t *fo_ptr, *fo_gate;
  __device__ GateIn gate(int32_t g) const {
    const uint4 r = gin[g];
    GateIn x;
    x.tt = tt[g];
    x.net[0] = int32_t(r.x & 0xffffu);
    x.net[1] = int32_t(r.x >> 16);
    x.net[2] = int32_t(r.y & 0xffffu);
    x.net[3] = int32_t(r.y >> 16);
    x.net[4] = int32_t(r.z & 0xffffu);
    x.net[5] = int32_t(r.z >> 16);
    x.e0 = int32_t(r.w & 0xffffu);
    x.k = int32_t((r.w >> 16) & 0xffu);
    return x;
  }
  __device__ int32_t fo_begin(int32_t n) const { return fo_ptr[n]; }
  __device__ int32_t fo_end(int32_t n) const { return fo_ptr[n + 1]; }
  __device__ int32_t fo(int32_t q) const { return fo_gate[q]; }
};

// Any circuit: the 32-bit CSR arrays of the batch description.
struct NetGeneric {
  using Idx = int32_t;
  int32_t I, G, N;
  const int32_t *fptr, *fanin;
  const uint64_t* tt;
  const int32_t *dr, *df;
  const int32_t *fo_ptr, *fo_gate;
  __device__ GateIn gate(int32_t g) const {
    GateIn x;
    x.tt = tt[g];
    x.e0 = fptr[g];
    x.k = fptr[g + 1] - x.e0;
#pragma unroll
    for (int i = 0; i < MAX_ARITY; ++i) x.net[i] = i < x.k ? fanin[x.e0 + i] : 0;
    return x;
  }
  __device__ int32_t fo_begin(int32_t n) const { return fo_ptr[n]; }
  __device__ int32_t fo_end(int32_t n) const { return fo_ptr[n + 1]; }
  __device__ int32_t fo(int32_t q) const { return fo_gate[q]; }
};

template <class Net>
__device__ Net net_view(const DevBatch& d, const NetArrays& a, int64_t b) {
  const int64_t no = d.net_off[b], go = d.gate_off[b], eo = d.edge_off[b];
  Net c;
  c.I = d.n_inputs[b];
  c.G = int32_t(d.gate_off[b + 1] - go);
  c.N = int32_t(d.net_off[b + 1] - no);
  c.tt = d.tt + go;
  c.dr = d.dr + eo;
  c.df = d.df + eo;
  if constexpr (std::is_same<Net, NetPacked>::value) {
    c.gin = a.gin16 + go;
    c.fo_ptr = a.fo_ptr16 + no + b;
    c.fo_gate = a.fo_gate16 + eo;
  } else {
    c.fptr = d.fanin_ptr + go + b;
    c.fanin = d.fanin + eo;
    c.fo_ptr = d.fo_ptr + no + b;
    c.fo_gate = a.fo_gate + eo;
  }
  return c;
}

// The netlist (base) delay of the arc from edge e (a pin of some gate) to that gate's output, for an output transition
// to `rise`. The delay an evaluation applies is fr_arc_delay (jitter.h) of the smallest base delay over the changed
// pins; the lookahead L is a lower bound of every value it can return (docs/ALGORITHM.md, F1; SPEC §10.6).
template <class Net>
__device__ __forceinline__ int32_t arc_base_delay(const Net& c, int32_t e, int rise) {
  return rise ? c.dr[e] : c.df[e];
}

// ---- per-circuit working state ----------------------------------------------------------------------------------

template <class Idx>
struct WinState {
  int64_t* wt;      // [N] time of the net's change in the current window; smaller than t0 if none
  int64_t* act_t;   // [cap] active list: pending times (FR_NEVER = hole) ...
  unsigned* tflag;  // [ceil(G/32)] touched-in-this-window bits
  unsigned* obsw;   // [ceil(N/32)] nets whose toggles / high time are maintained
  unsigned* cdep;   // [ceil(G/32)] per candidate slot: another candidate reads this one's decision in P3
  Idx* act_g;       // [cap] ... and their gates
  Idx* entry_of;    // [G] position of the gate's entry in the active list, Idx(-1) if none
  Idx* touched;     // [G] touched gates of the window, candidates first
  Idx* sc;          // [max(I, 1)] input nets changed by stimulus in the window
  uint8_t* val;     // [N] net values at the start of the window
  uint8_t* dec;     // [G] commit decision of a candidate ...
  uint8_t* dec2;    // [G] ... and the next one, during the passes of P3
};

__host__ __device__ inline int64_t win_align(int64_t x) { return (x + 15) & ~int64_t(15); }

// Bytes of working state for a circuit; win_state() carves them in this order.
template <class Idx>
__host__ __device__ inline int64_t win_state_bytes(int64_t G, int64_t N, int64_t I) {
  const int64_t cap = G + SLACK, wG = (G + 31) / 32, wN = (N + 31) / 32;
  return win_align(8 * N) + win_align(8 * cap) + win_align(8 * wG) + win_align(4 * wN) +
         win_align(int64_t(sizeof(Idx)) * (cap + 2 * G + (I > 1 ? I : 1))) + win_align(N + 2 * G);
}

template <class Idx>
__device__ WinState<Idx> win_state(unsigned char* base, int64_t G, int64_t N, int64_t I) {
  const int64_t cap = G + SLACK, wG = (G + 31) / 32, wN = (N + 31) / 32;
  WinState<Idx> w;
  unsigned char* p = base;
  w.wt = reinterpret_cast<int64_t*>(p);
  p += win_align(8 * N);
  w.act_t = reinterpret_cast<int64_t*>(p);
  p += win_align(8 * cap);
  w.tflag = reinterpret_cast<unsigned*>(p);
  w.cdep = w.tflag + wG;
  p += win_align(8 * wG);
  w.obsw = reinterpret_cast<unsigned*>(p);
  p += win_align(4 * wN);
  w.act_g = reinterpret_cast<Idx*>(p);
  w.entry_of = w.act_g + cap;
  w.touched = w.entry_of + G;
  w.sc = w.touched + G;
  p += win_align(int64_t(sizeof(Idx)) * (cap + 2 * G + (I > 1 ? I : 1)));
  w.val = p;
  w.dec = p + N;
  w.dec2 = p + N + G;
  return w;
}

// ---- the windowed run of one circuit ----------------------------------------------------------------------------

constexpr uint32_t NONE32 = 0xffffffffu;

// Offset of time t in the window [t0, t0 + width), or NONE32 if t is outside it. Window widths are below 2^31, so
// offsets fit 32 bits; the subtraction is unsigned so that earlier (or INT64_MIN) times wrap to large values.
__device__ __forceinline__ uint32_t win_offset(int64_t t, int64_t t0, uint64_t width) {
  const uint64_t d = uint64_t(t) - uint64_t(t0);
  return d < width ? uint32_t(d) : NONE32;
}

// Commit decision of candidate g (pending at p in the window): it commits unless an evaluation strictly before p sees
// its function equal to its output, given the current decisions w.dec of the other candidates (see P3).
template <class Net, class Idx>
__device__ __forceinline__ uint8_t decide(const Net& c, const WinState<Idx>& w, int32_t g, int64_t t0) {
  const GateIn gi = c.gate(g);
  const int64_t p = w.wt[c.I + g];
  const int y = w.val[c.I + g];
  const uint8_t* dec = w.dec;
  const uint64_t before = uint64_t(p - t0);  // events strictly before p
  uint32_t x = 0;
  uint32_t tw[MAX_ARITY];
#pragma unroll
  for (int i = 0; i < MAX_ARITY; ++i) {
    tw[i] = NONE32;
    if (i < gi.k) {
      const int32_t n = gi.net[i];
      x |= uint32_t(w.val[n]) << i;
      const uint32_t o = win_offset(w.wt[n], t0, before);
      if (o != NONE32 && (n < c.I || dec[n - c.I])) tw[i] = o;
    }
  }
  for (;;) {  // distinct event times in increasing order; a changing net flips once
    uint32_t tau = NONE32;
#pragma unroll
    for (int i = 0; i < MAX_ARITY; ++i) tau = min(tau, tw[i]);
    if (tau == NONE32) return 1;
#pragma unroll
    for (int i = 0; i < MAX_ARITY; ++i)
      if (tw[i] == tau) {
        x ^= 1u << i;
        tw[i] = NONE32;
      }
    if (int((gi.tt >> x) & 1u) == y) return 0;
  }
}

struct ReplayStats {
  int64_t ties = 0, first_tie = FR_NEVER, cancels = 0, advances = 0;
};

// Replay gate g's events in the window [t0, t1) in time order, exactly as SPEC §5: its own commit (phase A) and its
// valid input changes (phase B). Returns its pending time at the end of the window.
template <class Net, class Idx>
__device__ __forceinline__ int64_t replay(const Net& c, const WinState<Idx>& w, int32_t g, int64_t t0, int64_t t1,
                                          const fr_jitter_view& jit, uint64_t* jn, int stats, ReplayStats& rs) {
  const GateIn gi = c.gate(g);
  const int32_t own = c.I + g;
  const uint64_t width = uint64_t(t1 - t0);
  int64_t amp = 0;  // jitter amplitude and transition counter, loaded up front (SPEC §10)
  uint64_t n_jit = 0;
  if (jit.dmin) {
    amp = jit.amp[g];
    n_jit = jn[g];
  }
  const uint64_t n_jit0 = n_jit;
  int y = w.val[own];
  const uint32_t oc = win_offset(w.wt[own], t0, width);  // own candidate time, if any
  uint32_t own_t = oc != NONE32 && w.dec[g] ? oc : NONE32;  // own commit, still to apply
  int64_t pend;
  if (oc != NONE32) {
    pend = t0 + oc;
  } else {
    const Idx ent = w.entry_of[g];
    pend = ent != Idx(-1) ? w.act_t[ent] : FR_NEVER;
  }
  uint32_t x = 0;
  uint32_t tw[MAX_ARITY];
  int32_t d_rise[MAX_ARITY], d_fall[MAX_ARITY];
#pragma unroll
  for (int i = 0; i < MAX_ARITY; ++i) {
    tw[i] = NONE32;
    if (i < gi.k) {
      const int32_t n = gi.net[i];
      x |= uint32_t(w.val[n]) << i;
      const uint32_t o = win_offset(w.wt[n], t0, width);
      if (o != NONE32 && (n < c.I || w.dec[n - c.I])) tw[i] = o;
    }
  }
#pragma unroll
  for (int i = 0; i < MAX_ARITY; ++i)  // the delays of the arcs that will be used, loaded up front
    if (tw[i] != NONE32) {
      d_rise[i] = arc_base_delay(c, gi.e0 + i, 1);
      d_fall[i] = arc_base_delay(c, gi.e0 + i, 0);
    }
  for (;;) {
    uint32_t te = NONE32;
#pragma unroll
    for (int i = 0; i < MAX_ARITY; ++i) te = min(te, tw[i]);
    const uint32_t tau = min(te, own_t);
    if (tau == NONE32) break;
    bool committed_now = false;
    if (own_t == tau) {  // phase A: the gate's own transition commits first
      y ^= 1;
      pend = FR_NEVER;
      own_t = NONE32;
      committed_now = true;
    }
    if (te != tau) continue;
    unsigned m = 0;  // pins whose nets change at tau
#pragma unroll
    for (int i = 0; i < MAX_ARITY; ++i)
      if (tw[i] == tau) {
        m |= 1u << i;
        tw[i] = NONE32;
      }
    x ^= m;
    if (stats) {  // ties (SPEC §8): two or more causes, a cause being a distinct changed net or the own commit
      int nn = 0;
      bool own_in = false;
#pragma unroll
      for (int i = 0; i < MAX_ARITY; ++i) {
        if (!((m >> i) & 1u)) continue;
        bool dup = false;
#pragma unroll
        for (int j = 0; j < i; ++j) dup |= ((m >> j) & 1u) && gi.net[j] == gi.net[i];
        if (!dup) {
          ++nn;
          own_in |= gi.net[i] == own;
        }
      }
      if (nn + ((committed_now && !own_in) ? 1 : 0) >= 2) {
        rs.ties += 1;
        rs.first_tie = min(rs.first_tie, t0 + tau);
      }
    }
    const int v = int((gi.tt >> x) & 1u);
    if (v == y) {
      if (pend != FR_NEVER) rs.cancels += 1;
      pend = FR_NEVER;
    } else {
      int32_t dmin = INT_MAX;
#pragma unroll
      for (int i = 0; i < MAX_ARITY; ++i)
        if ((m >> i) & 1u) dmin = min(dmin, v ? d_rise[i] : d_fall[i]);
      const int64_t tn = t0 + tau + fr_arc_delay_amp(jit, amp, g, dmin, pend == FR_NEVER, n_jit);
      if (pend == FR_NEVER) {
        pend = tn;
      } else if (tn < pend) {
        pend = tn;
        rs.advances += 1;
      }
    }
  }
  if (n_jit != n_jit0) jn[g] = n_jit;
  return pend;
}

// Remove holes from the active list, preserving order. Returns the new count.
template <class Grp, class Idx>
__device__ int32_t win_compact(const Grp& G, const WinState<Idx>& w, int32_t count) {
  int32_t out = 0;
  for (int32_t base = 0; base < count; base += G.size) {
    const int32_t i = base + G.rank;
    bool alive = false;
    int64_t at = 0;
    Idx g = 0;
    if (i < count) {
      at = w.act_t[i];
      g = w.act_g[i];
      alive = at != FR_NEVER;
    }
    G.sync();  // all reads of this chunk before any write (writes go to positions <= reads)
    int32_t tot;
    const int32_t off = G.count(alive, tot);
    if (alive) {
      w.act_t[out + off] = at;
      w.act_g[out + off] = g;
      w.entry_of[g] = Idx(out + off);
    }
    out += tot;
    G.sync();
  }
  return out;
}

// Per-circuit persistent state and observables in global memory.
struct WinGlobal {
  uint8_t* val;
  int64_t* pend;
  int64_t *toggles, *high, *seg;
  const uint8_t* obs;
  const uint8_t* watch;
  fr_jitter_view jit;  // jit.dmin == 0: no jitter
  uint64_t* jn;        // [G] jitter transition counters
};

// Multi-step runs: the outputs at boundary k (time tb) of circuit b, which closes step k - 1 (k = 0 only records the
// running totals): each output slot's value, high time and toggles since the previous boundary, and the step's gate
// events. zero: the step comes after a budget stop and reports 0. Not inlined: it runs once per step, and keeping it out
// of the window loop keeps that loop's registers.
template <class Grp>
__device__ __noinline__ void steps_emit(const Grp& G, const DevSteps& sp, int64_t b, int64_t k, int64_t tb, bool zero,
                                        const uint8_t* val, const WinGlobal& gl, unsigned long long gev,
                                        unsigned long long gev_base) {
  for (int32_t j = G.rank; j < sp.n_out; j += G.size) {
    const int32_t n = sp.out_net[b * sp.n_out + j];
    int64_t* pv = sp.prev + 2 * (b * sp.n_out + j);
    uint8_t v = 0;
    int64_t H = 0, Tg = 0;
    if (n >= 0) {
      v = val[n];
      H = gl.high[n] + (v ? tb - gl.seg[n] : 0);
      Tg = gl.toggles[n];
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
  if (k >= 1 && G.rank == 0 && sp.events) sp.events[b * sp.s_max + k - 1] = zero ? 0 : int64_t(gev - gev_base);
}

template <class Grp, class Net, bool Steps>
__device__ void run_windowed(const Grp& G, const Net& c, unsigned char* wbase, const WinGlobal& gl, const DevState& s,
                             const DevStim& st, int64_t b, int64_t L, int64_t t_start, int64_t T, int stats,
                             const DevSteps& sp) {
  using Idx = typename Net::Idx;
  const int r = G.rank, S = G.size;
  const int32_t I = c.I, NG = c.G, NN = c.N, cap = NG + SLACK;
  WinState<Idx> w = win_state<Idx>(wbase, NG, NN, I);
  GroupShared& sh = *G.sh;
  GroupSharedSteps* ss = Steps ? static_cast<GroupSharedSteps*>(G.sh) : nullptr;  // windowed_blocks allocates it
  if (r == 0) {
    sh.n_cand = sh.n_touch = sh.n_app = sh.n_hole = 0;
    if constexpr (Steps) {  // visible to the group after the barriers of the stage-in
      ss->gev = 0;
      ss->st_K = sp.K[b];
      ss->st_S = sp.S[b];
      ss->st_bud = sp.budget ? sp.budget[b] : -1;
      ss->st_w0 = t_start + sp.offset;
    }
  }

  // ---- stage in: values, observed-net bits, and the active list rebuilt from the pending times ----
  for (int32_t n = r; n < NN; n += S) {
    w.val[n] = gl.val[n];
    w.wt[n] = INT64_MIN;
  }
  for (int32_t i = r; i < (NG + 31) / 32; i += S) w.tflag[i] = w.cdep[i] = 0;
  for (int32_t base = 0; base < NN; base += S) {
    const int32_t n = base + r;
    const unsigned m = __ballot_sync(FULL, n < NN && gl.obs[n]);
    if ((r & 31) == 0 && n < NN) w.obsw[n >> 5] = m;
  }
  int32_t count = 0;
  for (int32_t base = 0; base < NG; base += S) {
    const int32_t g = base + r;
    const int64_t p = g < NG ? gl.pend[g] : FR_NEVER;
    const bool app = p != FR_NEVER;
    int32_t tot;
    const int32_t off = G.count(app, tot);
    if (g < NG) w.entry_of[g] = app ? Idx(count + off) : Idx(-1);
    if (app) {
      w.act_t[count + off] = p;
      w.act_g[count + off] = Idx(g);
    }
    count += tot;
  }
  int32_t valid = count;
  if (r == 0) s.tr_count[b] = 0;
  G.sync();
  for (int32_t i = r; i < (NN + 31) / 32; i += S)  // observed nets start a new observation window
    for (unsigned m = w.obsw[i]; m; m &= m - 1) {
      const int32_t n = i * 32 + __ffs(m) - 1;
      gl.toggles[n] = 0;
      gl.high[n] = 0;
      gl.seg[n] = t_start;
    }

  auto observe = [&](int32_t n, int64_t tau, uint8_t old) {
    if ((w.obsw[n >> 5] >> (n & 31)) & 1u) {
      if (old) gl.high[n] += tau - gl.seg[n];
      gl.seg[n] = tau;
      gl.toggles[n] += 1;
    }
    if (gl.watch && gl.watch[n]) {
      const unsigned long long pos = atomicAdd(&s.tr_count[b], 1ull);
      if ((int64_t)pos < s.tr_cap) {
        const int64_t o = b * s.tr_cap + (int64_t)pos;
        s.tr_t[o] = tau;
        s.tr_net[o] = n;
        s.tr_v[o] = old ^ 1;
      }
    }
  };

  int64_t si = 0, se = 0;
  if (st.off) {
    si = st.off[b];
    se = st.off[b + 1];
  }
  int64_t snext = si < se ? st.t[si] : FR_NEVER;  // next stimulus time (group-uniform)
  int64_t n_ev = 0, n_win = 0;

  // Multi-step runs: boundary k at w0 + k K closes step k - 1 (the group-uniform state below, the step parameters in
  // shared memory). The gate-net transition count ss->gev is read only where no thread can be adding to it (after the
  // barrier that follows P5).
  int64_t kb = 0;
  int32_t stop = -1;
  unsigned long long gev_base = 0;
  bool frozen = false;
  auto emit = [&](int64_t k, bool zero) {
    const unsigned long long gev = ss->gev;
    steps_emit(G, sp, b, k, ss->st_w0 + k * ss->st_K, zero, w.val, gl, gev, gev_base);
    gev_base = gev;
  };
  ReplayStats rs;
  int64_t tnext = FR_NEVER;
  for (int32_t i = r; i < count; i += S) tnext = min(tnext, w.act_t[i]);
  tnext = G.min(tnext);

  for (;;) {
    const int64_t t0 = min(tnext, snext);
    if constexpr (Steps) {  // the boundaries up to t0 (state at the boundary: every time before it is processed)
      for (const int64_t lim = min(t0, T); kb <= ss->st_S && ss->st_w0 + kb * ss->st_K <= lim;) {
        emit(kb, false);
        if (++kb > ss->st_S && sp.hold) frozen = true;
      }
      if (frozen) break;
    }
    if (t0 >= T) break;
    int64_t t1 = min(t0 + L, T);
    if constexpr (Steps)
      if (kb <= ss->st_S) t1 = min(t1, ss->st_w0 + kb * ss->st_K);
    ++n_win;

    // ---- P1a: stimulus, one timestamp at a time. The window is cut before a timestamp that drives an input changed
    // earlier in this window, so that every net changes at most once per window. ----
    int32_t nS = 0;
    while (si < se && st.t[si] < t1) {
      const int64_t tau = st.t[si];
      int64_t j = si;
      bool conflict = false;
      for (;;) {  // extent [si, j) of the timestamp, and whether it drives an input changed earlier in the window
        const int64_t q = j + r;
        const bool take = q < se && st.t[q] == tau;
        if (take) conflict |= w.wt[st.net[q]] >= t0;
        int32_t n_take;
        G.count(take, n_take);
        j += n_take;
        if (n_take < S) break;
      }
      if (G.any(conflict)) {
        t1 = tau;
        break;
      }
      for (int64_t q0 = si; q0 < j; q0 += S) {
        const int64_t q = q0 + r;
        int32_t n = 0;
        bool ch = false;
        if (q < j) {
          n = st.net[q];
          ch = st.v[q] != w.val[n];
        }
        int32_t tot;
        const int32_t off = G.count(ch, tot);
        if (ch) {
          w.wt[n] = tau;
          w.sc[nS + off] = Idx(n);
        }
        nS += tot;
      }
      si = j;
      G.sync();
    }
    snext = si < se ? st.t[si] : FR_NEVER;

    // ---- P1b: candidates, the transitions pending in [t0, t1); they take the first nK touched slots ----
    int64_t surv = FR_NEVER;
    for (int32_t i = r; i < count; i += S) {
      const int64_t at = w.act_t[i];
      if (at < t1) {
        const int32_t g = w.act_g[i];
        w.touched[atomicAdd(&sh.n_cand, 1)] = Idx(g);
        w.wt[I + g] = at;
        w.dec[g] = 1;
        atomicOr(&w.tflag[g >> 5], 1u << (g & 31));
      } else {
        surv = min(surv, at);
      }
    }
    G.sync();
    const int32_t nK = sh.n_cand;
    if (nK == 0 && nS == 0) {  // a stale minimum, or stimulus that changes nothing (n_cand is still 0)
      G.sync();                // every thread has read n_cand before the next window appends to it
      tnext = G.min(surv);
      continue;
    }

    // ---- P2: touched gates (the fanout of every net that may change), and which candidates have dependents ----
    for (int32_t z = r; z < nK + nS; z += S) {
      const bool is_cand = z < nK;
      const int32_t n = is_cand ? I + int32_t(w.touched[z]) : int32_t(w.sc[z - nK]);
      const int64_t tz = w.wt[n];
      bool dep = false;
      for (int32_t q = c.fo_begin(n), qe = c.fo_end(n); q < qe; q += 4) {
        int32_t gg[4];  // four fanout entries at a time, so that their loads overlap
#pragma unroll
        for (int u = 0; u < 4; ++u) gg[u] = q + u < qe ? c.fo(q + u) : -1;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int32_t g2 = gg[u];
          if (g2 < 0) continue;
          if (is_cand) {
            const int64_t t2 = w.wt[I + g2];
            dep |= t2 > tz && t2 < t1;  // a later candidate, which reads this one's decision in P3
          }
          const unsigned bit = 1u << (g2 & 31);
          if (!(atomicOr(&w.tflag[g2 >> 5], bit) & bit)) w.touched[nK + atomicAdd(&sh.n_touch, 1)] = Idx(g2);
        }
      }
      if (is_cand && dep) atomicOr(&w.cdep[z >> 5], 1u << (z & 31));
    }
    G.sync();
    const int32_t nT = nK + sh.n_touch;

    // ---- P3: commit decisions, iterated to a fixed point. Each pass computes every candidate's decision from the
    // previous pass's decisions (w.dec) into w.dec2, then the buffers swap. Only candidates with dependents are ever
    // read, so only their changes count: a pass in which none of them changed computed every decision from final
    // values, which is the (unique) fixed point. ----
    for (;;) {
      bool changed = false;
      for (int32_t k = r; k < nK; k += S) {
        const int32_t g = w.touched[k];
        const uint8_t d = decide(c, w, g, t0);
        w.dec2[g] = d;
        changed |= d != w.dec[g] && ((w.cdep[k >> 5] >> (k & 31)) & 1u);
      }
      const bool again = G.any(changed);  // also the barrier between this pass's reads and the next one's writes
      uint8_t* tmp = w.dec;
      w.dec = w.dec2;
      w.dec2 = tmp;
      if (!again) break;
    }

    // ---- multi-step budget: if this window's commits take the step's gate-net transitions over the budget, the
    // window ends right after the time of the crossing commit (its (need+1)-th in time order), and the circuit
    // freezes after it. Only cheap counts unless the candidates alone could cross. ----
    if constexpr (Steps) {
      const int64_t bud = ss->st_bud;
      if (bud >= 0 && kb >= 1 && kb <= ss->st_S) {
        const int64_t used = int64_t(ss->gev - gev_base);
        if (used + nK > bud) {
          int64_t n_c = 0;
          for (int32_t base = 0; base < nK; base += S) {
            const int32_t k = base + r;
            int32_t tot;
            G.count(k < nK && w.dec[w.touched[k]], tot);
            n_c += tot;
          }
          if (used + n_c > bud) {
            const int64_t need = bud - used;
            int64_t cand = FR_NEVER;
            for (int32_t k = r; k < nK; k += S) {
              const int32_t g = w.touched[k];
              if (!w.dec[g]) continue;
              const int64_t tk = w.wt[I + g];
              int64_t lt = 0, le = 0;
              for (int32_t j = 0; j < nK; ++j) {
                const int32_t h = w.touched[j];
                if (!w.dec[h]) continue;
                const int64_t tj = w.wt[I + h];
                lt += tj < tk;
                le += tj <= tk;
              }
              if (lt <= need && need < le) cand = min(cand, tk);
            }
            t1 = G.min(cand) + 1;
            stop = int32_t(kb - 1);
            frozen = true;
          }
        }
      }
    }

    // ---- P4: replay every touched gate; update its pending time in the active list ----
    if (count + nT > cap || count > 2 * valid + 64) {
      count = win_compact(G, w, count);
      valid = count;
    }
    int64_t fresh = FR_NEVER;
    for (int32_t j = r; j < nT; j += S) {
      const int32_t g = w.touched[j];
      const int64_t P = replay(c, w, g, t0, t1, gl.jit, gl.jn, stats, rs);
      const Idx ent = w.entry_of[g];
      if (P == FR_NEVER) {
        if (ent != Idx(-1)) {
          w.act_t[ent] = FR_NEVER;
          w.entry_of[g] = Idx(-1);
          atomicAdd(&sh.n_hole, 1);
        }
      } else if (ent != Idx(-1)) {
        w.act_t[ent] = P;
      } else {
        const int32_t pos = count + atomicAdd(&sh.n_app, 1);
        w.act_t[pos] = P;
        w.act_g[pos] = Idx(g);
        w.entry_of[g] = Idx(pos);
      }
      if (P != FR_NEVER) fresh = min(fresh, P);
      atomicAnd(&w.tflag[g >> 5], ~(1u << (g & 31)));
    }
    G.sync();
    count += sh.n_app;
    valid += sh.n_app - sh.n_hole;

    // ---- P5: apply the changes of committed candidates and of stimulus (multi-step: up to a budget cut) ----
    int64_t n_gate = 0;
    for (int32_t k = r; k < nK + nS; k += S) {
      int32_t n;
      if (k < nK) {
        const int32_t g = w.touched[k];
        if ((k & 31) == 0) w.cdep[k >> 5] = 0;  // P3 has read them; one thread per word clears for the next window
        if (!w.dec[g]) continue;
        n = I + g;
        if constexpr (Steps) {
          if (w.wt[n] >= t1) continue;
          ++n_gate;
        }
      } else {
        n = w.sc[k - nK];
        if constexpr (Steps)
          if (w.wt[n] >= t1) continue;
      }
      const uint8_t old = w.val[n];
      observe(n, w.wt[n], old);
      w.val[n] = old ^ 1;
      n_ev += 1;
    }
    if constexpr (Steps)
      if (n_gate) atomicAdd(&ss->gev, (unsigned long long)n_gate);
    G.sync();
    if (r == 0) sh.n_cand = sh.n_touch = sh.n_app = sh.n_hole = 0;  // every thread has read them by now
    G.sync();
    if (frozen) break;  // multi-step budget (group-uniform)
    tnext = G.min(min(surv, fresh));
  }

  // ---- multi-step: the remaining boundaries; a frozen circuit holds its state, and steps after a budget stop
  // report 0 ----
  if constexpr (Steps) {
    for (; kb <= ss->st_S; ++kb) emit(kb, stop >= 0 && kb > stop + 1);
    if (r == 0 && sp.stop) sp.stop[b] = stop;
  }

  // ---- stage out: values and pending times; close the observation window ----
  for (int32_t n = r; n < NN; n += S) gl.val[n] = w.val[n];
  for (int32_t g = r; g < NG; g += S) {
    const Idx ent = w.entry_of[g];
    gl.pend[g] = ent != Idx(-1) && !frozen ? w.act_t[ent] : FR_NEVER;  // a frozen circuit has nothing pending
  }
  for (int32_t i = r; i < (NN + 31) / 32; i += S)
    for (unsigned m = w.obsw[i]; m; m &= m - 1) {
      const int32_t n = i * 32 + __ffs(m) - 1;
      if (w.val[n]) gl.high[n] += T - gl.seg[n];
    }
  n_ev = G.sum(n_ev);
  const int64_t ties = G.sum(rs.ties), cancels = G.sum(rs.cancels), advances = G.sum(rs.advances);
  const int64_t first_tie = G.min(rs.first_tie);
  if (r == 0) {
    s.n_windows[b] = n_win;
    int64_t* o = s.stats + b * FR_NSTATS;
    o[FR_STAT_EVENTS] = n_ev;
    o[FR_STAT_TIES] = ties;
    o[FR_STAT_FIRST_TIE] = ties ? first_tie : -1;
    o[FR_STAT_CANCELS] = cancels;
    o[FR_STAT_ADVANCES] = advances;
  }
  G.sync();
}

// One circuit per group, one group per block; persistent blocks pull circuits from queue[0 .. n). The working state is
// in dynamic shared memory or, if `scratch` is given, at scratch + scratch_off[b] in global memory.
template <class Grp, class Net, bool Steps>
__device__ __forceinline__ void windowed_blocks(DevBatch d, NetArrays na, DevState s, DevStim st, const int64_t* L,
                                                const int64_t* queue, int64_t n, unsigned long long* next,
                                                int64_t t_start, int64_t T, int stats, unsigned char* scratch,
                                                const int64_t* scratch_off, const DevSteps& sp) {
  __shared__ std::conditional_t<Steps, GroupSharedSteps, GroupShared> sh;
  extern __shared__ __align__(16) unsigned char dyn[];
  Grp G;
  G.rank = int(threadIdx.x);
  G.sh = &sh;
  if constexpr (!std::is_same<Grp, WarpGroup>::value) G.size = int(blockDim.x);
  for (;;) {
    if (G.rank == 0) sh.item = (long long)atomicAdd(next, 1ull);
    G.sync();
    const int64_t bi = sh.item;
    G.sync();
    if (bi >= n) return;
    const int64_t b = queue[bi];
    const Net c = net_view<Net>(d, na, b);
    const int64_t no = d.net_off[b], go = d.gate_off[b];
    WinGlobal gl{s.val + no, s.pend + go, s.toggles + no, s.high + no, s.seg + no, s.obs + no,
                 s.watch ? s.watch + no : nullptr, fr_jitter_view{0, 0, 0, nullptr}, s.jn + go};
    if (d.jit_dmin && d.jit_dmin[b] > 0)
      gl.jit = fr_jitter_view{d.jit_seed[b], d.jit_cid[b], d.jit_dmin[b], d.jit_amp + go};
    run_windowed<Grp, Net, Steps>(G, c, scratch ? scratch + scratch_off[b] : dyn, gl, s, st, b, L[b], t_start, T,
                                  stats, sp);
  }
}

#define FR_WINDOWED_PARAMS                                                                                         \
  DevBatch d, NetArrays na, DevState s, DevStim st, const int64_t *L, const int64_t *queue, int64_t n,             \
      unsigned long long *next, int64_t t_start, int64_t T, int stats, unsigned char *scratch,                     \
      const int64_t *scratch_off, DevSteps sp
#define FR_WINDOWED_ARGS d, na, s, st, L, queue, n, next, t_start, T, stats, scratch, scratch_off, sp

// Register caps (measured on H200, hpc/gpu_regs.sh). Resident blocks per SM matter more than the last registers:
// - block kernels at 96 (uncapped ~140): 8 instead of 7 blocks of 64 threads (1024 gates) and 2 instead of 1 block
//   of 256 threads (4096 gates) per SM, 20-40 % faster;
// - the warp kernel at 128: 16 instead of 14 warps per SM for small circuits (256 gates: 10 % faster); at 112 the
//   spills cost more than the occupancy gains.
#ifndef FR_WARP_MAXNREG
#define FR_WARP_MAXNREG 128
#endif
#ifndef FR_BLOCK_MAXNREG
#define FR_BLOCK_MAXNREG 96
#endif

// One warp per circuit. Steps: a multi-step run (separate instantiations, so plain runs are compiled as before).
template <class Net, bool Steps>
__global__ void __maxnreg__(FR_WARP_MAXNREG) k_run_windowed_warp(FR_WINDOWED_PARAMS) {
  windowed_blocks<WarpGroup, Net, Steps>(FR_WINDOWED_ARGS);
}

// One thread block per circuit.
template <class Net, bool Steps>
__global__ void __maxnreg__(FR_BLOCK_MAXNREG) k_run_windowed_block(FR_WINDOWED_PARAMS) {
  windowed_blocks<BlockGroup, Net, Steps>(FR_WINDOWED_ARGS);
}
