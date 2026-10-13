// CPU backend: each circuit is simulated sequentially with a lazy-deletion binary heap; circuits are distributed
// over threads. Semantics: docs/SPEC.md (must match python/freerun/ref.py bit-exactly).
//
// Memory layout: every circuit's mutable state (values, pending times, toggles, high time, segment starts, jitter
// counters, statistics) and its scratch (Worker, Trace) start on their own kLine-byte block, so circuits simulated on
// different threads never write the same cache lines (no false sharing; with packed state and a few-gate circuit per
// lane, 16 threads were slower than one). Cost: a few KB per circuit. The public arrays (values(), pending(), ...)
// keep the packed layout of the batch description: they are gathered from the padded internal arrays.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "fanout.h"
#include "freerun.h"
#include "jitter.h"
#include "steps.h"

namespace {

constexpr size_t kLine = 256;  // padding granule: 64-byte lines alone were not enough on Xeon (the L2 spatial
                                // prefetcher works on line pairs and streams); 256 bytes made 16 threads ~14x one

// std::vector storage starting on a kLine boundary.
template <class T>
struct LineAlloc {
  using value_type = T;
  LineAlloc() = default;
  template <class U>
  LineAlloc(const LineAlloc<U>&) {}
  T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(kLine))); }
  void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t(kLine)); }
  template <class U>
  bool operator==(const LineAlloc<U>&) const { return true; }
  template <class U>
  bool operator!=(const LineAlloc<U>&) const { return false; }
};
template <class T>
using LVec = std::vector<T, LineAlloc<T>>;

// Offsets [B+1] of per-circuit slices of n[b] elements of `bytes` bytes each, every slice starting on a kLine boundary.
std::vector<int64_t> padded_offsets(int64_t B, const int64_t* off, size_t bytes) {
  const int64_t per = int64_t(kLine / bytes);
  std::vector<int64_t> o(size_t(B + 1), 0);
  for (int64_t b = 0; b < B; ++b) o[b + 1] = o[b] + (off[b + 1] - off[b] + per - 1) / per * per;
  return o;
}
constexpr int64_t kStatStride = int64_t(kLine / sizeof(int64_t));  // statistics per circuit, own lines
static_assert(FR_NSTATS <= kStatStride, "statistics must fit one cache line");

struct Circuit {
  // views into the engine-owned global arrays
  int32_t I, G, N, E;
  const int32_t* fanin_ptr;  // [G+1] local edge offsets
  const int32_t* fanin;      // [E]
  const int32_t* dr;
  const int32_t* df;
  const uint64_t* tt;        // [G]
  const int32_t* fo_ptr;     // [N+1]
  const int32_t* fo_edge;    // [E]
  const int32_t* edge_gate;  // [E]
  // state
  uint8_t* val;     // [N]
  int64_t* pend;    // [G]
  int64_t* toggles; // [N]
  int64_t* high;    // [N]
  int64_t* seg;     // [N]
  const uint8_t* watch;  // [N] or null
  const uint8_t* obs;    // [N] 1 = maintain toggles / high time for this net
  bool obs_all;          // every net is observed ...
  const int32_t* obs_nets;  // ... or only these (n_obs distinct nets)
  int32_t n_obs;
  fr_jitter_view jit;    // SPEC §10; jit.dmin == 0: no jitter
  uint64_t* jn;          // [G] transition counters n_g
};

using HeapEntry = std::pair<int64_t, int32_t>;  // (time, gate)
struct Later {
  bool operator()(const HeapEntry& a, const HeapEntry& b) const { return a > b; }
};
using Heap = std::priority_queue<HeapEntry, LVec<HeapEntry>, Later>;

struct alignas(kLine) Trace {
  LVec<int64_t> t;
  LVec<int32_t> net;
  LVec<uint8_t> v;
};

struct alignas(kLine) Worker {  // per-circuit persistent scratch (its own kLine blocks, file comment)
  Heap heap;
  LVec<int32_t> changed, committed, touched;
  LVec<int32_t> stamp, last_net;  // per gate: epoch of last touch, last changed net that reached it
  LVec<int32_t> dmin_r, dmin_f, n_causes;
  LVec<uint8_t> was_committed, own_hit;
  int32_t epoch = 0;
};

// A persistent pool of worker threads: a closed loop calls run() thousands of times, and starting threads on every
// call would cost more than a short step. The calling thread works too. Items are claimed dynamically.
class Pool {
 public:
  explicit Pool(int n_threads) {
    for (int k = 1; k < n_threads; ++k) workers_.emplace_back([this] { work_loop(); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }
  int size() const { return int(workers_.size()) + 1; }

  // Calls f(i) for i in [0, n), spread over the pool; returns when all calls have returned.
  void for_each(int64_t n, const std::function<void(int64_t)>& f) {
    if (workers_.empty() || n <= 1) {
      for (int64_t i = 0; i < n; ++i) f(i);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(m_);
      job_ = &f;
      n_ = n;
      next_.store(0);
      busy_ = int(workers_.size());
      ++generation_;
    }
    cv_.notify_all();
    drain();
    std::unique_lock<std::mutex> lk(m_);
    done_cv_.wait(lk, [this] { return busy_ == 0; });
    job_ = nullptr;
  }

 private:
  void drain() {
    for (int64_t i; (i = next_.fetch_add(1)) < n_;) (*job_)(i);
  }
  void work_loop() {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
      }
      drain();
      std::lock_guard<std::mutex> lk(m_);
      if (--busy_ == 0) done_cv_.notify_one();
    }
  }
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int64_t)>* job_ = nullptr;
  int64_t n_ = 0;
  std::atomic<int64_t> next_{0};
  int busy_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

template <class F>
void parallel_for(Pool& pool, int64_t n, F f) {
  const std::function<void(int64_t)> fn = f;
  pool.for_each(n, fn);
}

struct Snapshot {  // packed layout (the batch description's)
  int64_t t;
  std::vector<uint8_t> val;
  std::vector<int64_t> pend;
  std::vector<uint64_t> jn;  // jitter transition counters
};

struct Engine {
  int64_t B;
  std::vector<int64_t> net_off, gate_off, edge_off;
  std::vector<int32_t> n_inputs, fanin_ptr, fanin, dr, df, fo_ptr, fo_edge, edge_gate;
  std::vector<uint64_t> tt;
  std::vector<uint8_t> watch;
  // mutable state, padded per circuit (file comment): slices at val_off (bytes), net_off64 (per-net int64) and
  // gate_off64 (per-gate 64-bit) offsets, statistics at b * kStatStride
  std::vector<int64_t> val_off, net_off64, gate_off64;
  LVec<uint8_t> val;
  LVec<int64_t> pend, toggles, high, seg;
  LVec<uint64_t> jn;             // transition counters
  LVec<int64_t> stats;
  std::vector<int64_t> jit_amp;  // [G] packed (empty without jitter)
  std::vector<Circuit> circ;
  std::vector<Worker> work;
  std::vector<Trace> trace;
  int64_t t_now = 0;
  int n_threads;
  std::unique_ptr<Pool> pool;
  int flags;
  bool net_obs = true;        // per-net toggles / high time for every net
  std::vector<uint8_t> obs;   // [N] nets whose toggles / high time are maintained
  std::vector<int32_t> obs_nets;  // the same as local net lists per circuit (without net_obs), at obs_off[b]
  std::vector<int64_t> obs_off;
  int32_t n_in = 0, n_out = 0;
  int io_flags = 0;
  std::vector<int32_t> out_net;  // [B * n_out]
  std::vector<std::unique_ptr<Snapshot>> snaps;  // by id; null = freed
  double prof[FR_NPROF] = {0, 0, 0};
  std::string error;
};

// Nets whose toggles and high time are maintained: all of them, or only the closed-loop outputs.
void update_obs(Engine& E) {
  E.obs.assign(size_t(E.net_off[E.B]), E.net_obs ? 1 : 0);
  E.obs_nets.clear();
  E.obs_off.assign(size_t(E.B + 1), 0);
  for (int64_t b = 0; b < E.B; ++b) {
    if (!E.net_obs)
      for (int32_t j = 0; j < E.n_out; ++j) {
        const int32_t n = E.out_net[b * E.n_out + j];
        if (n >= 0 && !E.obs[E.net_off[b] + n]) {
          E.obs[E.net_off[b] + n] = 1;
          E.obs_nets.push_back(n);
        }
      }
    E.obs_off[b + 1] = int64_t(E.obs_nets.size());
  }
}

// Point every circuit at its slices of the observation mask and list.
void bind_obs(Engine& E) {
  for (int64_t b = 0; b < E.B; ++b) {
    Circuit& c = E.circ[b];
    c.obs = E.obs.data() + E.net_off[b];
    c.obs_all = E.net_obs;
    c.obs_nets = E.obs_nets.data() + E.obs_off[b];
    c.n_obs = int32_t(E.obs_off[b + 1] - E.obs_off[b]);
  }
}

// Calls f(n) for every observed net of the circuit.
template <class F>
inline void for_observed(const Circuit& c, F f) {
  if (c.obs_all)
    for (int32_t n = 0; n < c.N; ++n) f(n);
  else
    for (int32_t i = 0; i < c.n_obs; ++i) f(c.obs_nets[i]);
}

inline int eval_gate(const Circuit& c, int32_t g) {
  uint32_t idx = 0;
  for (int32_t e = c.fanin_ptr[g], i = 0; e < c.fanin_ptr[g + 1]; ++e, ++i) idx |= uint32_t(c.val[c.fanin[e]]) << i;
  return int((c.tt[g] >> idx) & 1u);
}

// SPEC §3. dmin = min arc delay over the evaluated pin set, for the direction of the would-be transition.
// Returns 0 none, 1 cancel, 2 schedule, 3 keep, 4 advance.
inline int evaluate(Circuit& c, Heap& heap, int32_t g, int64_t t, int32_t dmin_rise, int32_t dmin_fall) {
  const int y = c.val[c.I + g];
  const int v = eval_gate(c, g);
  if (v == y) {
    const bool had = c.pend[g] != FR_NEVER;
    c.pend[g] = FR_NEVER;
    return had ? 1 : 0;
  }
  const int64_t tn = t + fr_arc_delay(c.jit, g, v ? dmin_rise : dmin_fall, c.pend[g] == FR_NEVER, c.jn[g]);
  if (c.pend[g] == FR_NEVER || tn < c.pend[g]) {
    const int what = c.pend[g] == FR_NEVER ? 2 : 4;
    c.pend[g] = tn;
    heap.push({tn, g});
    return what;
  }
  return 3;
}

void powerup(Circuit& c, Worker& w) {
  for (int32_t g = 0; g < c.G; ++g) {
    if (c.fanin_ptr[g] == c.fanin_ptr[g + 1]) continue;
    int32_t r = INT32_MAX, f = INT32_MAX;
    for (int32_t e = c.fanin_ptr[g]; e < c.fanin_ptr[g + 1]; ++e) {
      r = std::min(r, c.dr[e]);
      f = std::min(f, c.df[e]);
    }
    evaluate(c, w.heap, g, 0, r, f);
  }
}

// Process time t (SPEC §5). Phase A commits the transitions due at t and applies the stimulus at t: the dense row
// `dense` (n_dense inputs) when given, then the triples s[si..] at t (si advances past them). Phase B evaluates every
// gate with a pin on a changed net. Returns the number of committed gate transitions.
int64_t process_time(Engine& E, Circuit& c, Worker& w, Trace& tr, int64_t* st, int64_t t, const fr_stimulus* s,
                     int64_t& si, int64_t se, const uint8_t* dense, int32_t n_dense) {
  const bool stats = E.flags & FR_FLAG_STATS;
  // Phase A: commit
  w.changed.clear();
  w.committed.clear();
  while (!w.heap.empty() && w.heap.top().first == t) {
    const int32_t g = w.heap.top().second;
    w.heap.pop();
    if (c.pend[g] == t) {
      c.pend[g] = FR_NEVER;
      w.committed.push_back(g);
      w.changed.push_back(c.I + g);
    }
  }
  if (dense)  // dense closed-loop inputs: stimulus at t for inputs 0 .. n_dense - 1
    for (int32_t n = 0; n < n_dense; ++n)
      if (c.val[n] != dense[n]) w.changed.push_back(n);
  for (; si < se && s->t[si] == t; ++si) {
    const int32_t n = s->net[si];
    if (c.val[n] != s->v[si]) w.changed.push_back(n);
  }
  for (const int32_t n : w.changed) {
    const uint8_t old = c.val[n];
    if (c.obs[n]) {
      if (old) c.high[n] += t - c.seg[n];
      c.seg[n] = t;
      c.toggles[n] += 1;
    }
    c.val[n] = old ^ 1;
    if (c.watch && c.watch[n]) { tr.t.push_back(t); tr.net.push_back(n); tr.v.push_back(old ^ 1); }
  }
  st[FR_STAT_EVENTS] += int64_t(w.changed.size());

  // Phase B: gather touched gates with their min arc delays over the changed pins, then evaluate.
  if (++w.epoch == INT32_MAX) { std::fill(w.stamp.begin(), w.stamp.end(), 0); w.epoch = 1; }
  if (stats) for (const int32_t g : w.committed) w.was_committed[g] = 1;
  w.touched.clear();
  for (const int32_t n : w.changed) {
    for (int32_t k = c.fo_ptr[n]; k < c.fo_ptr[n + 1]; ++k) {
      const int32_t e = c.fo_edge[k];
      const int32_t g = c.edge_gate[e];
      if (w.stamp[g] != w.epoch) {
        w.stamp[g] = w.epoch;
        w.touched.push_back(g);
        w.dmin_r[g] = c.dr[e];
        w.dmin_f[g] = c.df[e];
        if (stats) { w.n_causes[g] = 1; w.last_net[g] = n; w.own_hit[g] = (n == c.I + g); }
      } else {
        w.dmin_r[g] = std::min(w.dmin_r[g], c.dr[e]);
        w.dmin_f[g] = std::min(w.dmin_f[g], c.df[e]);
        if (stats && w.last_net[g] != n) { w.n_causes[g] += 1; w.last_net[g] = n; w.own_hit[g] |= (n == c.I + g); }
      }
    }
  }
  for (const int32_t g : w.touched) {
    if (stats) {
      const int causes = w.n_causes[g] + (w.was_committed[g] && !w.own_hit[g] ? 1 : 0);
      if (causes >= 2) {
        if (st[FR_STAT_TIES] == 0) st[FR_STAT_FIRST_TIE] = t;
        st[FR_STAT_TIES] += 1;
      }
    }
    const int what = evaluate(c, w.heap, g, t, w.dmin_r[g], w.dmin_f[g]);
    if (what == 1) st[FR_STAT_CANCELS] += 1;
    else if (what == 4) st[FR_STAT_ADVANCES] += 1;
  }
  if (stats) for (const int32_t g : w.committed) w.was_committed[g] = 0;
  return int64_t(w.committed.size());
}

// Start of a run window [t0, ...) of circuit b: statistics, trace and the observed nets' toggles and high time.
void begin_window(Engine& E, int64_t b, int64_t t0) {
  Circuit& c = E.circ[b];
  int64_t* st = &E.stats[b * kStatStride];
  st[FR_STAT_EVENTS] = st[FR_STAT_TIES] = st[FR_STAT_CANCELS] = st[FR_STAT_ADVANCES] = 0;
  st[FR_STAT_FIRST_TIE] = -1;
  Trace& tr = E.trace[b];
  tr.t.clear(); tr.net.clear(); tr.v.clear();
  for_observed(c, [&](int32_t n) {
    c.toggles[n] = 0;
    c.high[n] = 0;
    c.seg[n] = t0;
  });
}

// End of a run window at T: close the high time of the observed nets; within one time step, trace rows were appended in
// commit order, and the canonical order is (t, net).
void end_window(Engine& E, int64_t b, int64_t T) {
  Circuit& c = E.circ[b];
  Trace& tr = E.trace[b];
  for_observed(c, [&](int32_t n) {
    if (c.val[n]) c.high[n] += T - c.seg[n];
  });
  const size_t m = tr.t.size();
  for (size_t i = 0, j; i < m; i = j) {
    for (j = i + 1; j < m && tr.t[j] == tr.t[i]; ++j) {}
    if (j - i > 1) {
      std::vector<std::pair<int32_t, uint8_t>> tmp;
      for (size_t k = i; k < j; ++k) tmp.push_back({tr.net[k], tr.v[k]});
      std::sort(tmp.begin(), tmp.end());
      for (size_t k = i; k < j; ++k) { tr.net[k] = tmp[k - i].first; tr.v[k] = tmp[k - i].second; }
    }
  }
}

// The next time to process: the earliest pending transition or stimulus (FR_NEVER if none). Drops stale heap entries.
inline int64_t next_time(Circuit& c, Worker& w, const fr_stimulus* s, int64_t si, int64_t se) {
  while (!w.heap.empty() && c.pend[w.heap.top().second] != w.heap.top().first) w.heap.pop();  // stale
  int64_t t = FR_NEVER;
  if (!w.heap.empty()) t = w.heap.top().first;
  if (si < se && s->t[si] < t) t = s->t[si];
  return t;
}

// in: this circuit's dense input row (n_dense entries, applied at t0), or null.
void run_circuit(Engine& E, int64_t b, int64_t t0, int64_t T, const fr_stimulus* s, const uint8_t* in,
                 int32_t n_dense) {
  Circuit& c = E.circ[b];
  Worker& w = E.work[b];
  begin_window(E, b, t0);
  int64_t si = 0, se = 0;
  if (s && s->stim_off) { si = s->stim_off[b]; se = s->stim_off[b + 1]; }
  bool dense = in != nullptr && n_dense > 0;
  for (;;) {
    int64_t t = next_time(c, w, s, si, se);
    if (dense) t = std::min(t, t0);
    if (t >= T) break;
    process_time(E, c, w, E.trace[b], &E.stats[b * kStatStride], t, s, si, se, dense && t == t0 ? in : nullptr,
                 n_dense);
    if (t == t0) dense = false;
  }
  end_window(E, b, T);
}

// A multi-step run of circuit b over [t0, T) (SPEC §13; freerun.h fr_steps): run_circuit with per-step observables at
// the step boundaries w0 + k K (k = 0 .. S; boundary k closes step k - 1) and the event budget.
void run_circuit_steps(Engine& E, int64_t b, int64_t t0, int64_t T, const fr_stimulus* s, const fr_steps* p,
                       const fr_steps_out* o) {
  Circuit& c = E.circ[b];
  Worker& w = E.work[b];
  begin_window(E, b, t0);
  int64_t si = 0, se = 0;
  if (s && s->stim_off) { si = s->stim_off[b]; se = s->stim_off[b + 1]; }
  const int64_t K = p->K[b], S = p->S[b], bud = p->budget ? p->budget[b] : -1, w0 = t0 + p->offset;
  const bool hold = p->flags & FR_STEPS_HOLD_AT_END;
  const int32_t n_out = E.n_out;
  std::vector<int64_t> prev(size_t(2 * n_out), 0);  // cumulative high time and toggles at the last boundary
  int64_t gev = 0;                                   // gate-net transitions in the current step
  int64_t kb = 0, stop = -1;
  bool frozen = false;
  auto emit = [&](int64_t k, bool zero) {  // boundary k at w0 + k K: closes step k - 1, opens step k
    const int64_t tb = w0 + k * K;
    for (int32_t j = 0; j < n_out; ++j) {
      const int32_t n = E.out_net[b * n_out + j];
      uint8_t v = 0;
      int64_t H = 0, Tg = 0;
      if (n >= 0) {
        v = c.val[n];
        H = c.high[n] + (v ? tb - c.seg[n] : 0);
        Tg = c.toggles[n];
      }
      if (k >= 1) {
        const int64_t i = (b * p->s_max + k - 1) * n_out + j;
        if (o->val) o->val[i] = zero ? 0 : v;
        if (o->high) o->high[i] = zero ? 0 : H - prev[2 * j];
        if (o->tog) o->tog[i] = zero ? 0 : Tg - prev[2 * j + 1];
      }
      prev[2 * j] = H;
      prev[2 * j + 1] = Tg;
    }
    if (k >= 1 && o->events) o->events[b * p->s_max + k - 1] = zero ? 0 : gev;
    gev = 0;
  };
  for (;;) {
    const int64_t t = next_time(c, w, s, si, se);
    for (const int64_t lim = std::min(t, T); kb <= S && w0 + kb * K <= lim;) {
      emit(kb, false);
      if (++kb > S && hold) frozen = true;
    }
    if (frozen || t >= T) break;
    const int64_t ng = process_time(E, c, w, E.trace[b], &E.stats[b * kStatStride], t, s, si, se, nullptr, 0);
    if (kb >= 1 && kb <= S) {  // t lies in step kb - 1
      gev += ng;
      if (bud >= 0 && gev > bud) {
        stop = kb - 1;
        frozen = true;
        break;
      }
    }
  }
  if (frozen) {  // no pending transitions and no more stimulus: the circuit holds its state from here on
    for (int32_t g = 0; g < c.G; ++g) c.pend[g] = FR_NEVER;
    w.heap = Heap();
  }
  for (; kb <= S; ++kb) emit(kb, stop >= 0 && kb > stop + 1);
  if (o->stop) o->stop[b] = stop;
  end_window(E, b, T);
}

// Gather a padded per-circuit array into the packed layout (slices at `packed` offsets, `pad` offsets inside `src`).
template <class T, class A>
void gather(const Engine& E, const std::vector<T, A>& src, const std::vector<int64_t>& pad, const std::vector<int64_t>& packed,
            T* out) {
  for (int64_t b = 0; b < E.B; ++b)
    std::memcpy(out + packed[b], src.data() + pad[b], size_t(packed[b + 1] - packed[b]) * sizeof(T));
}

Snapshot* snapshot(const Engine& E) {
  auto* S = new Snapshot{E.t_now, std::vector<uint8_t>(size_t(E.net_off[E.B])),
                         std::vector<int64_t>(size_t(E.gate_off[E.B])), std::vector<uint64_t>(size_t(E.gate_off[E.B]))};
  gather(E, E.val, E.val_off, E.net_off, S->val.data());
  gather(E, E.pend, E.gate_off64, E.gate_off, S->pend.data());
  gather(E, E.jn, E.gate_off64, E.gate_off, S->jn.data());
  return S;
}

// Load a state (values, pending times at t_ref, transition counters or null = 0) into the selected circuits,
// translated to the current time.
void load_state(Engine& E, int64_t t_ref, const uint8_t* val, const int64_t* pend, const uint64_t* jn,
                const uint8_t* mask) {
  const int64_t shift = E.t_now - t_ref;
  parallel_for(*E.pool, E.B, [&](int64_t b) {
    if (mask && !mask[b]) return;
    Circuit& c = E.circ[b];
    Worker& w = E.work[b];
    const int64_t no = E.net_off[b], go = E.gate_off[b];
    std::memcpy(c.val, val + no, size_t(c.N));
    for (int32_t g = 0; g < c.G; ++g) c.jn[g] = jn ? jn[go + g] : 0;
    LVec<HeapEntry> items;
    for (int32_t g = 0; g < c.G; ++g) {
      const int64_t p = pend[go + g];
      c.pend[g] = p == FR_NEVER ? FR_NEVER : p + shift;
      if (p != FR_NEVER) items.push_back({c.pend[g], g});
    }
    w.heap = Heap(Later(), std::move(items));
  });
}

// SPEC §2 / §6 constraints on a loadable state. Returns an error message, or empty.
std::string check_state(const Engine& E, int64_t t_ref, const uint8_t* val, const int64_t* pend) {
  for (int64_t b = 0; b < E.B; ++b) {
    const Circuit& c = E.circ[b];
    const int64_t no = E.net_off[b], go = E.gate_off[b];
    for (int32_t n = 0; n < c.N; ++n)
      if (val[no + n] > 1) return "circuit " + std::to_string(b) + ": values must be 0 or 1";
    for (int32_t g = 0; g < c.G; ++g) {
      const int64_t p = pend[go + g];
      if (p != FR_NEVER && p < t_ref)
        return "circuit " + std::to_string(b) + ": pending time before the state's time";
      if (c.fanin_ptr[g] == c.fanin_ptr[g + 1] && (p != FR_NEVER || val[no + c.I + g] != (c.tt[g] & 1)))
        return "circuit " + std::to_string(b) + ": a constant gate must hold its value and have nothing pending";
    }
  }
  return {};
}

}  // namespace

extern "C" {

void* fr_cpu_create(const fr_batch_desc* d, int n_threads, int flags) {
  auto* E = new Engine();
  const int64_t B = d->n_circuits;
  E->B = B;
  E->flags = flags;
  E->n_threads = n_threads > 0 ? n_threads : int(std::max(1u, std::thread::hardware_concurrency()));
  E->n_threads = int(std::max<int64_t>(1, std::min<int64_t>(E->n_threads, B)));
  E->pool.reset(new Pool(E->n_threads));
  const int64_t G = d->gate_off[B], Etot = d->edge_off[B];
  E->net_off.assign(d->net_off, d->net_off + B + 1);
  E->gate_off.assign(d->gate_off, d->gate_off + B + 1);
  E->edge_off.assign(d->edge_off, d->edge_off + B + 1);
  E->n_inputs.assign(d->n_inputs, d->n_inputs + B);
  E->fanin_ptr.assign(d->fanin_ptr, d->fanin_ptr + G + B);
  E->fanin.assign(d->fanin, d->fanin + Etot);
  E->dr.assign(d->dr, d->dr + Etot);
  E->df.assign(d->df, d->df + Etot);
  E->tt.assign(d->tt, d->tt + G);
  E->val_off = padded_offsets(B, d->net_off, sizeof(uint8_t));
  E->net_off64 = padded_offsets(B, d->net_off, sizeof(int64_t));
  E->gate_off64 = padded_offsets(B, d->gate_off, sizeof(int64_t));
  E->val.assign(size_t(E->val_off[B]), 0);
  for (int64_t b = 0; b < B; ++b)
    std::memcpy(E->val.data() + E->val_off[b], d->init + d->net_off[b], size_t(d->net_off[b + 1] - d->net_off[b]));
  Fanout fo = batch_fanout(d);
  E->fo_ptr = std::move(fo.fo_ptr);
  E->fo_edge = std::move(fo.fo_edge);
  E->edge_gate = std::move(fo.edge_gate);
  E->pend.assign(size_t(E->gate_off64[B]), FR_NEVER);
  E->toggles.assign(size_t(E->net_off64[B]), 0);
  E->high.assign(size_t(E->net_off64[B]), 0);
  E->seg.assign(size_t(E->net_off64[B]), 0);
  E->stats.assign(size_t(B * kStatStride), 0);
  E->jn.assign(size_t(E->gate_off64[B]), 0);
  if (d->jit_amp) E->jit_amp.assign(d->jit_amp, d->jit_amp + G);
  E->circ.resize(B);
  E->work.resize(B);
  E->trace.resize(B);
  update_obs(*E);
  for (int64_t b = 0; b < B; ++b) {
    Circuit& c = E->circ[b];
    const int64_t no = E->net_off[b], go = E->gate_off[b], eo = E->edge_off[b];
    c.I = E->n_inputs[b];
    c.G = int32_t(E->gate_off[b + 1] - go);
    c.N = int32_t(E->net_off[b + 1] - no);
    c.E = int32_t(E->edge_off[b + 1] - eo);
    c.fanin_ptr = &E->fanin_ptr[go + b];
    c.fanin = E->fanin.data() + eo;
    c.dr = E->dr.data() + eo;
    c.df = E->df.data() + eo;
    c.tt = E->tt.data() + go;
    c.fo_ptr = &E->fo_ptr[no + b];
    c.fo_edge = E->fo_edge.data() + eo;
    c.edge_gate = E->edge_gate.data() + eo;
    c.val = E->val.data() + E->val_off[b];
    c.pend = E->pend.data() + E->gate_off64[b];
    c.toggles = E->toggles.data() + E->net_off64[b];
    c.high = E->high.data() + E->net_off64[b];
    c.seg = E->seg.data() + E->net_off64[b];
    c.watch = nullptr;
    c.jit = fr_jitter_view{0, 0, 0, nullptr};
    if (d->jit_dmin && d->jit_dmin[b] > 0)
      c.jit = fr_jitter_view{d->jit_seed[b], d->jit_cid[b], d->jit_dmin[b], E->jit_amp.data() + go};
    c.jn = E->jn.data() + E->gate_off64[b];
    Worker& w = E->work[b];
    w.stamp.assign(c.G, 0);
    w.last_net.assign(c.G, -1);
    w.dmin_r.assign(c.G, 0);
    w.dmin_f.assign(c.G, 0);
    w.n_causes.assign(c.G, 0);
    w.was_committed.assign(c.G, 0);
    w.own_hit.assign(c.G, 0);
  }
  bind_obs(*E);
  parallel_for(*E->pool, B, [&](int64_t b) { powerup(E->circ[b], E->work[b]); });
  E->snaps.emplace_back(snapshot(*E));  // FR_SNAPSHOT_POWERUP (t_now = 0)
  return E;
}

void fr_cpu_destroy(void* h) { delete static_cast<Engine*>(h); }

void fr_cpu_set_watch(void* h, const uint8_t* mask) {
  auto* E = static_cast<Engine*>(h);
  if (!mask) {
    E->watch.clear();
    for (auto& c : E->circ) c.watch = nullptr;
    return;
  }
  E->watch.assign(mask, mask + E->net_off[E->B]);
  for (int64_t b = 0; b < E->B; ++b) E->circ[b].watch = E->watch.data() + E->net_off[b];
}

void fr_cpu_set_trace_capacity(void*, int64_t) {}

void fr_cpu_set_net_observables(void* h, int on) {
  auto* E = static_cast<Engine*>(h);
  E->net_obs = on != 0;
  update_obs(*E);
  bind_obs(*E);
}

int fr_cpu_set_io(void* h, int32_t n_in, int32_t n_out, const int32_t* out_net, int io_flags) {
  auto* E = static_cast<Engine*>(h);
  if (n_in < 0 || n_out < 0 || (n_out > 0 && !out_net)) {
    E->error = "set_io: bad arguments";
    return 1;
  }
  for (int64_t b = 0; b < E->B; ++b)
    for (int32_t j = 0; j < n_out; ++j) {
      const int32_t n = out_net[b * n_out + j];
      if (n < -1 || n >= E->circ[b].N) {
        E->error = "set_io: output " + std::to_string(j) + " of circuit " + std::to_string(b) + " is not a net";
        return 1;
      }
    }
  E->n_in = n_in;
  E->n_out = n_out;
  E->io_flags = io_flags;
  E->out_net.assign(out_net, out_net + E->B * n_out);
  fr_cpu_set_net_observables(h, E->net_obs);
  return 0;
}

int fr_cpu_step(void* h, int64_t T, const fr_stimulus* s, const uint8_t* in, uint8_t* out_val, int64_t* out_high,
                int64_t* out_tog) {
  auto* E = static_cast<Engine*>(h);
  const auto c0 = std::chrono::steady_clock::now();
  if (T < E->t_now) {
    E->error = "cannot run backwards";
    return 1;
  }
  if ((out_high && !(E->io_flags & FR_IO_HIGH)) || (out_tog && !(E->io_flags & FR_IO_TOGGLES))) {
    E->error = "step: output high time / toggles were not requested in set_io";
    return 1;
  }
  const int64_t t0 = E->t_now;
  parallel_for(*E->pool, E->B, [&](int64_t b) {
    const int32_t nd = std::min(E->n_in, E->circ[b].I);
    run_circuit(*E, b, t0, T, s, in ? in + b * E->n_in : nullptr, nd);
    const Circuit& c = E->circ[b];
    for (int32_t j = 0; j < E->n_out; ++j) {
      const int64_t k = b * E->n_out + j;
      const int32_t n = E->out_net[k];
      if (out_val) out_val[k] = n >= 0 ? c.val[n] : 0;
      if (out_high) out_high[k] = n >= 0 ? c.high[n] : 0;
      if (out_tog) out_tog[k] = n >= 0 ? c.toggles[n] : 0;
    }
  });
  E->t_now = T;
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
  E->prof[FR_PROF_CALL_S] = E->prof[FR_PROF_KERNEL_S] = dt;
  E->prof[FR_PROF_WINDOWS] = 0;
  return 0;
}

int fr_cpu_run_steps(void* h, const fr_stimulus* s, const fr_steps* p, const fr_steps_out* o) {
  auto* E = static_cast<Engine*>(h);
  const auto c0 = std::chrono::steady_clock::now();
  std::string err = check_steps(E->B, E->t_now, p, o, E->io_flags);
  if (!err.empty()) {
    E->error = "run_steps: " + err;
    return 1;
  }
  const int64_t T = steps_end(E->B, E->t_now, p);
  const int64_t t0 = E->t_now;
  const fr_steps_out none{nullptr, nullptr, nullptr, nullptr, nullptr};
  parallel_for(*E->pool, E->B, [&](int64_t b) { run_circuit_steps(*E, b, t0, T, s, p, o ? o : &none); });
  E->t_now = T;
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
  E->prof[FR_PROF_CALL_S] = E->prof[FR_PROF_KERNEL_S] = dt;
  E->prof[FR_PROF_WINDOWS] = 0;
  return 0;
}

int fr_cpu_run(void* h, int64_t T, const fr_stimulus* s) {
  return fr_cpu_step(h, T, s, nullptr, nullptr, nullptr, nullptr);
}

int64_t fr_cpu_snapshot(void* h) {
  auto* E = static_cast<Engine*>(h);
  E->snaps.emplace_back(snapshot(*E));
  return int64_t(E->snaps.size()) - 1;
}

int fr_cpu_restore(void* h, int64_t id, const uint8_t* mask) {
  auto* E = static_cast<Engine*>(h);
  if (id < 0 || id >= int64_t(E->snaps.size()) || !E->snaps[id]) {
    E->error = "restore: no snapshot " + std::to_string(id);
    return 1;
  }
  const Snapshot& S = *E->snaps[id];
  load_state(*E, S.t, S.val.data(), S.pend.data(), S.jn.data(), mask);
  return 0;
}

void fr_cpu_snapshot_free(void* h, int64_t id) {
  auto* E = static_cast<Engine*>(h);
  if (id > FR_SNAPSHOT_POWERUP && id < int64_t(E->snaps.size())) E->snaps[id].reset();
}

int fr_cpu_set_state(void* h, int64_t t_ref, const uint8_t* val, const int64_t* pend, const uint64_t* counters,
                     const uint8_t* mask) {
  auto* E = static_cast<Engine*>(h);
  const std::string err = check_state(*E, t_ref, val, pend);
  if (!err.empty()) {
    E->error = "set_state: " + err;
    return 1;
  }
  load_state(*E, t_ref, val, pend, counters, mask);
  return 0;
}

int64_t fr_cpu_time(void* h) { return static_cast<Engine*>(h)->t_now; }

void fr_cpu_values(void* h, uint8_t* out) {
  auto* E = static_cast<Engine*>(h);
  gather(*E, E->val, E->val_off, E->net_off, out);
}

void fr_cpu_pending(void* h, int64_t* out) {
  auto* E = static_cast<Engine*>(h);
  gather(*E, E->pend, E->gate_off64, E->gate_off, out);
}

void fr_cpu_toggles(void* h, int64_t* out) {
  auto* E = static_cast<Engine*>(h);
  if (E->net_obs) gather(*E, E->toggles, E->net_off64, E->net_off, out);
  else std::fill(out, out + E->net_off[E->B], 0);
}

void fr_cpu_high(void* h, int64_t* out) {
  auto* E = static_cast<Engine*>(h);
  if (E->net_obs) gather(*E, E->high, E->net_off64, E->net_off, out);
  else std::fill(out, out + E->net_off[E->B], 0);
}

void fr_cpu_stats(void* h, int64_t* out) {
  auto* E = static_cast<Engine*>(h);
  for (int64_t b = 0; b < E->B; ++b)
    std::memcpy(out + b * FR_NSTATS, E->stats.data() + b * kStatStride, FR_NSTATS * sizeof(int64_t));
}

void fr_cpu_trace_counts(void* h, int64_t* out) {
  auto* E = static_cast<Engine*>(h);
  for (int64_t b = 0; b < E->B; ++b) out[b] = int64_t(E->trace[b].t.size());
}

void fr_cpu_trace(void* h, int64_t* t, int32_t* net, uint8_t* v) {
  auto* E = static_cast<Engine*>(h);
  size_t k = 0;
  for (const auto& tr : E->trace) {
    std::memcpy(t + k, tr.t.data(), tr.t.size() * sizeof(int64_t));
    std::memcpy(net + k, tr.net.data(), tr.net.size() * sizeof(int32_t));
    std::memcpy(v + k, tr.v.data(), tr.v.size());
    k += tr.t.size();
  }
}

void fr_cpu_profile(void* h, double* out) {
  auto* E = static_cast<Engine*>(h);
  std::copy(E->prof, E->prof + FR_NPROF, out);
}

void fr_cpu_counters(void* h, uint64_t* out) {
  auto* E = static_cast<Engine*>(h);
  gather(*E, E->jn, E->gate_off64, E->gate_off, out);
}

const char* fr_cpu_error(void* h) { return static_cast<Engine*>(h)->error.c_str(); }

}  // extern "C"
