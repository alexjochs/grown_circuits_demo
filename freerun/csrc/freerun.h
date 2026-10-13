/* freerun C ABI, shared by all compiled backends (CPU, CUDA). Semantics: docs/SPEC.md.
 *
 * A batch is a concatenation of circuits. Global arrays are indexed by global gate / net / edge / fanout-entry index;
 * the values stored in them that refer to nets, gates or edges are LOCAL to the circuit (int32), so each circuit can
 * be processed with 32-bit indices. Offsets (int64) map circuit b to its slice of each global array.
 *
 * Every function that can fail returns 0 on success and nonzero on error; fr_<backend>_error() then describes it.
 */
#ifndef FREERUN_H
#define FREERUN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FR_NEVER INT64_MAX

typedef struct fr_batch_desc {
  int64_t n_circuits;
  const int64_t* net_off;    /* [B+1] */
  const int64_t* gate_off;   /* [B+1] */
  const int64_t* edge_off;   /* [B+1] */
  const int32_t* n_inputs;   /* [B] */
  const int32_t* fanin_ptr;  /* [G_total + B]: per circuit, G_b + 1 LOCAL edge offsets (circuit b's slice starts
                                at gate_off[b] + b) */
  const int32_t* fanin;      /* [E_total] local net index of each pin */
  const int32_t* dr;         /* [E_total] rise delay of arc pin -> output */
  const int32_t* df;         /* [E_total] fall delay */
  const uint64_t* tt;        /* [G_total] truth tables */
  const uint8_t* init;       /* [N_total] initial values */
  /* Fanout (optional: all three NULL lets the engine derive them from the fanin) */
  const int32_t* fo_ptr;     /* [N_total + B]: per circuit, N_b + 1 LOCAL fanout offsets (slice at net_off[b] + b) */
  const int32_t* fo_edge;    /* [F_total] local edge index of each fanout entry; F_total = E_total, and circuit b's
                                fanout entries live at edge_off[b] + fo_ptr[...]. Entries of a net are sorted by
                                (gate, pin). */
  const int32_t* edge_gate;  /* [E_total] local gate index owning each edge */
  /* Per-transition jitter (docs/SPEC.md §10). All four NULL: no circuit in the batch has jitter. */
  const uint64_t* jit_seed;  /* [B] RNG key */
  const uint32_t* jit_cid;   /* [B] circuit id */
  const int32_t* jit_dmin;   /* [B] delay floor; 0 = no jitter for this circuit */
  const int64_t* jit_amp;    /* [G_total] fixed-point amplitude a_g (0 = none) */
} fr_batch_desc;

/* Stimulus for one run call: events sorted by (circuit, t, net); circuit b's events are [stim_off[b], stim_off[b+1]).
 * All times must be in [current time, T). Nets are local primary-input indices. */
typedef struct fr_stimulus {
  const int64_t* stim_off;  /* [B+1] or NULL for no stimulus */
  const int64_t* t;
  const int32_t* net;
  const uint8_t* v;
} fr_stimulus;

/* Per-circuit statistics of the last run window. */
enum { FR_STAT_EVENTS = 0, FR_STAT_TIES, FR_STAT_FIRST_TIE, FR_STAT_CANCELS, FR_STAT_ADVANCES, FR_NSTATS };

/* Flags */
enum {
  FR_FLAG_STATS = 1,   /* count ties / cancels / advances (diagnostics; small overhead) */
  FR_FLAG_STEPPED = 2, /* CUDA: per-timestamp kernel instead of lookahead windows (for testing and comparison) */
  FR_FLAG_NO_SMEM = 4, /* CUDA: never stage circuit state in shared memory (for testing and comparison) */
  FR_FLAG_NO_PACK = 8  /* CUDA: always use the 32-bit netlist layout (for testing and comparison) */
};

/* Closed-loop I/O (fr_*_set_io): which per-output observables step() reports besides the values. */
enum { FR_IO_HIGH = 1, FR_IO_TOGGLES = 2 };

/* Multi-step runs (docs/SPEC.md §13): a whole open-loop episode of S control steps in one call. Step s of circuit b
 * observes the window [t0 + offset + s K_b, t0 + offset + (s+1) K_b), t0 = the current time; the call processes every
 * time < T_end = t0 + offset + max_b S_b K_b, exactly like run(T_end) with the given stimulus, and reports per step and
 * output slot (set_io) the value at the window's end, the ticks at 1 and the changes in the window, and per step the
 * committed gate-net transitions (primary inputs excluded). With a budget, a circuit whose gate-net transitions in one
 * step exceed it is frozen right after the time at which that happened: its pending transitions are dropped and its
 * remaining stimulus of the call is ignored (inputs hold), so it never changes again; that step reports the state and
 * counts up to the freeze (the values then held to the window's end), later steps report 0, and stop[b] is the step.
 * FR_STEPS_HOLD_AT_END freezes each circuit the same way at the end of its own last step (no flag). */
enum { FR_STEPS_HOLD_AT_END = 1 };

typedef struct fr_steps {
  const int64_t* K;      /* [B] ticks per step, >= 1 */
  const int64_t* S;      /* [B] steps, >= 0 */
  const int64_t* budget; /* [B] or NULL: max gate-net transitions per step, -1 = none */
  int64_t offset;        /* >= 0: start of step 0 relative to the current time */
  int64_t s_max;         /* step stride of the per-step output arrays, >= max S */
  int flags;             /* FR_STEPS_* */
} fr_steps;

/* Per-step outputs; any pointer may be NULL. Entries of steps >= S_b are 0. */
typedef struct fr_steps_out {
  uint8_t* val;    /* [B * s_max * n_out] value of each output slot at the end of the step's window */
  int64_t* high;   /* [B * s_max * n_out] ticks at 1 in the window (needs FR_IO_HIGH) */
  int64_t* tog;    /* [B * s_max * n_out] changes in the window (needs FR_IO_TOGGLES) */
  int64_t* events; /* [B * s_max] committed gate-net transitions in the window */
  int64_t* stop;   /* [B] step at which the circuit was frozen over its budget, -1 = none */
} fr_steps_out;

/* Snapshot id of the power-up state (time 0); restoring it resets circuits. */
#define FR_SNAPSHOT_POWERUP 0

/* Profile of the last run/step call (fr_*_profile). */
enum { FR_PROF_CALL_S = 0, /* wall time inside the call */
       FR_PROF_KERNEL_S,   /* time spent simulating (CUDA: device time of the simulation kernels) */
       FR_PROF_WINDOWS,    /* CUDA windowed kernels: lookahead windows processed, summed over circuits */
       FR_NPROF };

/* ---- CPU backend ---------------------------------------------------------------------------------------------
 *
 * Batch and observation set-up */
void* fr_cpu_create(const fr_batch_desc* d, int n_threads, int flags);
void fr_cpu_destroy(void* h);
void fr_cpu_set_watch(void* h, const uint8_t* mask /* [N_total] or NULL = record nothing */);
void fr_cpu_set_trace_capacity(void* h, int64_t cap); /* no-op on CPU (traces grow as needed) */
/* Per-net toggle counts and high time (fr_*_toggles / fr_*_high) are maintained by default. A closed-loop client that
 * only reads its outputs turns them off (on = 0); the two getters then return zeros. */
void fr_cpu_set_net_observables(void* h, int on);

/* Running. run(T) processes every time < T (SPEC §6). step() is run() plus dense closed-loop I/O:
 *   in        uint8 [B * n_in] or NULL. Row b drives primary inputs 0 .. min(n_in, n_inputs[b]) - 1 of circuit b at
 *             the current time, exactly as stimulus triples (current time, i, in[b * n_in + i]) would. Triples in s
 *             must not drive the same input at the same time.
 *   out_val   uint8 [B * n_out] or NULL: value of each output net at T- (0 for an unused slot).
 *   out_high  int64 [B * n_out] or NULL: ticks each output spent at 1 in [previous time, T) (needs FR_IO_HIGH).
 *   out_tog   int64 [B * n_out] or NULL: number of changes of each output in [previous time, T) (needs FR_IO_TOGGLES).
 * Both record the same per-run observables (trace, per-net toggles and high time, stats). */
int fr_cpu_run(void* h, int64_t T, const fr_stimulus* s);
/* n_in: width of the dense input rows. out_net: [B * n_out] local net of each output slot, -1 = unused. */
int fr_cpu_set_io(void* h, int32_t n_in, int32_t n_out, const int32_t* out_net, int io_flags);
int fr_cpu_step(void* h, int64_t T, const fr_stimulus* s, const uint8_t* in, uint8_t* out_val, int64_t* out_high,
                int64_t* out_tog);
/* A multi-step run (fr_steps above): stimulus s for the whole run, sorted by (circuit, t, net), all times in
 * [current time, T_end); outputs as declared in set_io. The current time becomes T_end. */
int fr_cpu_run_steps(void* h, const fr_stimulus* s, const fr_steps* p, const fr_steps_out* o);

/* State. A snapshot holds every circuit's net values, pending transitions and jitter transition counters (SPEC §10.2)
 * at the time it was taken. Restoring it
 * (for the circuits selected by mask [B], NULL = all) translates that state to the current time: values are copied,
 * and a pending time p becomes p - t_snapshot + t_now. Engine time never moves backwards. Snapshot
 * FR_SNAPSHOT_POWERUP always exists and resets circuits to their power-up state. Queued stimulus is not affected.
 * set_state loads a state given as host arrays val [N_total], pend [G_total] (FR_NEVER = none, otherwise >= t_ref) and
 * counters [G_total] (NULL = all 0), taken at time t_ref, the same way. Counters are copied, not translated. */
int64_t fr_cpu_snapshot(void* h); /* returns the new snapshot's id (>= 1), or -1 on error */
int fr_cpu_restore(void* h, int64_t id, const uint8_t* mask);
void fr_cpu_snapshot_free(void* h, int64_t id);
int fr_cpu_set_state(void* h, int64_t t_ref, const uint8_t* val, const int64_t* pend, const uint64_t* counters,
                     const uint8_t* mask);

/* Readout (of the last run/step and the current state) */
int64_t fr_cpu_time(void* h);
void fr_cpu_values(void* h, uint8_t* out /* [N_total] */);
void fr_cpu_pending(void* h, int64_t* out /* [G_total], FR_NEVER = none */);
void fr_cpu_toggles(void* h, int64_t* out /* [N_total] */);
void fr_cpu_high(void* h, int64_t* out /* [N_total] */);
void fr_cpu_stats(void* h, int64_t* out /* [B * FR_NSTATS] */);
void fr_cpu_trace_counts(void* h, int64_t* out /* [B] */);
void fr_cpu_trace(void* h, int64_t* t, int32_t* net, uint8_t* v); /* concatenated by circuit, each sorted (t, net) */
void fr_cpu_profile(void* h, double* out /* [FR_NPROF] */);
void fr_cpu_counters(void* h, uint64_t* out /* [G_total] jitter transition counters n_g, 0 without jitter */);
const char* fr_cpu_error(void* h);

/* ---- CUDA backend: the same functions with the fr_cuda_ prefix. n_threads is ignored (the current device is used).
 * Traces go to a fixed per-circuit buffer (fr_cuda_set_trace_capacity, default 65536 rows); run() returns 2 if a
 * circuit overflows it. Snapshots live in device memory. ---- */
void* fr_cuda_create(const fr_batch_desc* d, int n_threads, int flags);
void fr_cuda_destroy(void* h);
void fr_cuda_set_watch(void* h, const uint8_t* mask);
void fr_cuda_set_trace_capacity(void* h, int64_t cap);
void fr_cuda_set_net_observables(void* h, int on);
/* Circuits with more than `gates` gates run one per thread block of `threads` threads; the others one per warp.
 * 0 sends every circuit to the block kernel; -1 (default) chooses both from the batch size and the GPU. */
void fr_cuda_set_block_threshold(void* h, int64_t gates, int threads);
/* Every `every` runs (default 16; 0 = never), reorder the circuits of each launch by the cost of their last run
 * (windows and events), most expensive first, so that busy circuits do not start last. Results are unaffected. */
void fr_cuda_set_reorder(void* h, int every);
int fr_cuda_run(void* h, int64_t T, const fr_stimulus* s);
int fr_cuda_set_io(void* h, int32_t n_in, int32_t n_out, const int32_t* out_net, int io_flags);
int fr_cuda_step(void* h, int64_t T, const fr_stimulus* s, const uint8_t* in, uint8_t* out_val, int64_t* out_high,
                 int64_t* out_tog);
/* step() with device pointers for the dense I/O arrays (zero-copy for CUDA arrays of other libraries). Work already
 * queued on `stream` (a cudaStream_t, 0 = legacy default stream) is ordered before the step reads `in`; the outputs
 * are complete when the call returns. */
int fr_cuda_step_device(void* h, int64_t T, const fr_stimulus* s, const uint8_t* in, uint8_t* out_val,
                        int64_t* out_high, int64_t* out_tog, uintptr_t stream);
int fr_cuda_run_steps(void* h, const fr_stimulus* s, const fr_steps* p, const fr_steps_out* o);
/* run_steps() with device pointers for the outputs in o (K, S and budget stay host arrays). Work queued on `stream`
 * is ordered before the call writes them; they are complete when it returns. */
int fr_cuda_run_steps_device(void* h, const fr_stimulus* s, const fr_steps* p, const fr_steps_out* o,
                             uintptr_t stream);
int64_t fr_cuda_snapshot(void* h);
int fr_cuda_restore(void* h, int64_t id, const uint8_t* mask);
void fr_cuda_snapshot_free(void* h, int64_t id);
int fr_cuda_set_state(void* h, int64_t t_ref, const uint8_t* val, const int64_t* pend, const uint64_t* counters,
                      const uint8_t* mask);
int64_t fr_cuda_time(void* h);
void fr_cuda_values(void* h, uint8_t* out);
void fr_cuda_pending(void* h, int64_t* out);
void fr_cuda_toggles(void* h, int64_t* out);
void fr_cuda_high(void* h, int64_t* out);
void fr_cuda_stats(void* h, int64_t* out);
void fr_cuda_trace_counts(void* h, int64_t* out);
void fr_cuda_trace(void* h, int64_t* t, int32_t* net, uint8_t* v);
void fr_cuda_profile(void* h, double* out);
void fr_cuda_counters(void* h, uint64_t* out);
const char* fr_cuda_error(void* h);

#ifdef __cplusplus
}
#endif
#endif
