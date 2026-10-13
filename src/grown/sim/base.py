"""The Substrate protocol: run a batch of circuits under per-step inputs.

Timing convention (shared by every backend). A control step lasts K substrate ticks. Step s's inputs are applied
at time s*K and held until (s+1)*K. The reported output of step s is the value of each output net at time
(s+1)*K, after that time's transitions commit (in freerun terms: after phase A at t = (s+1)K). It therefore
includes every transition caused by inputs of steps <= s and none caused by step s+1. An output wired straight
to a primary input reports that step's input. `high` is the fraction of the step's K ticks (s*K, (s+1)*K] during
which the output was 1, and `toggles` counts transitions of gate nets (primary inputs excluded) over the episode.

Per-tick inputs (`per_tick=True`): inputs are [B, S, K, n_in] (or [S, K, n_in], shared) and tick k of step s is
applied at time s*K + k and held for one tick. This is how sub-step input codes reach the circuit (a sigma-delta
pulse train is K bits per control step, see grown.tasks.IOSpec.encode_ticks / encode_episode). Held inputs are the
special case where all K ticks of a step are equal. An output wired straight to a primary input then reports the
step's last tick, and its `high` is the mean over the step's K ticks.

With every gate delay equal to 1 tick, freerun's semantics (inertial delay, SPEC §3-§5) reduce exactly to the
synchronous unit-delay network of the stand-in backend: y_g(t+1) = f_g(net values at t). That is the shared
semantics the adapter tests check.

Event streams (physical K; docs/grown_time.md has the full contract). At a physical control step K is 10^5 to 10^8
ticks and per-tick inputs [B, S, K, n_in] cannot exist. `grown.sim.stepper.EventStreams` carries each input stream
as sparse changes (t, wire, v) instead, `EventLaneBatch` is the request that references them, and
`EvalResult.events` [L, S] reports the committed gate-net transitions of every step (the event budget is applied to
it). Outputs keep the convention above: a gate-driven output of step s is its value at (s+1)K after that time's
commits and `high` the fraction of ticks (sK, (s+1)K] at 1; an output wired to primary input i reports the input's
value in effect at tick (s+1)K - 1 and the fraction of ticks [sK, (s+1)K) at 1 (`wired_outputs`), which is what the
stand-in has always done. The helpers at the end of this module (events_from_dense, events_to_dense, select_streams,
concat_streams, window_counts, wired_outputs, flag_budget) work on any EventStreams. The stand-in refuses
K > STANDIN_MAX_K (4096).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


@dataclass
class EvalResult:
    outputs: np.ndarray     # uint8 [B, S, n_out]
    high: np.ndarray        # float32 [B, S, n_out]
    toggles: np.ndarray     # float64 [B]
    n_gates: np.ndarray     # int64 [B]
    ticks: int              # K
    extra: dict = field(default_factory=dict)   # name -> [B]: optional per-lane behaviour values (EXTRA_BEHAVIOR)
    events: np.ndarray | None = None            # int64 [B, S]: gate-net transitions per step (freerun step path)
    over_budget: np.ndarray | None = None       # bool [B]: lane stopped after a step over its event budget

    @property
    def toggle_rate(self) -> np.ndarray:
        """Transitions per gate per tick, in [0, 1]."""
        S = self.outputs.shape[1]
        return self.toggles / (np.maximum(self.n_gates, 1) * max(S * self.ticks, 1))


# Per-lane behaviour values a substrate may add in EvalResult.extra (the search averages them over a circuit's
# episodes into its behaviour dict, so archive axes can use them): grown.sim.devices (FreerunSubstrate with
# substrate.physical) fills them.
#   gap_duty      mean |duty(device) - duty(idealized twin)| per step and output wire
#   gap_disagree  fraction of (step, output) samples where device and twin differ
#   energy        joules per control step (per decision) of the lane the fitness reads
#   log10_energy  log10 of that
#   events_per_step  mean gate transitions per control step of the lane (the search adds it, grown.search)
#   over_budget      1 if the lane exceeded the event budget in some step (the search adds it)
#   capped           1 if the lane hit the compute guard (TimeModel.sim_event_cap) and was not simulated to the end
#   slow_static_w    static power of the circuit's slow delay elements (grown.sim.slow), W (freerun substrate)
#   slow_elements    number of slow delay elements in the circuit
EXTRA_BEHAVIOR = ("gap_duty", "gap_disagree", "energy", "log10_energy", "energy_static", "events_per_step", "over_budget",
                  "capped", "slow_static_w", "slow_elements")
STANDIN_MAX_K = 4096       # largest K the synchronous stand-in accepts (it steps every tick of every lane)
DENSE_MAX_K = 1 << 16      # largest K events_to_dense expands to per-tick arrays


class Substrate(Protocol):
    name: str

    def evaluate(self, circuits: Sequence, inputs: np.ndarray, K: int, library=None,
                 per_tick: bool = False) -> EvalResult:
        """circuits: B circuits with the same n_in and n_out; inputs: uint8 [B, S, n_in] (or [S, n_in], shared),
        or with per_tick [B, S, K, n_in] (or [S, K, n_in])."""
        ...


def broadcast_inputs(inputs, B: int) -> np.ndarray:
    x = np.asarray(inputs, np.uint8)
    if x.ndim == 2:
        x = np.broadcast_to(x, (B,) + x.shape)
    if x.ndim != 3 or x.shape[0] != B:
        raise ValueError(f"inputs must be [B, S, n_in] or [S, n_in], got {x.shape}")
    return x


def tick_inputs(inputs, B: int, K: int, per_tick: bool = False) -> np.ndarray:
    """Normalize inputs to uint8 [B, S, Kx, n_in] with Kx = K (per tick) or 1 (held for the whole step)."""
    x = np.asarray(inputs, np.uint8)
    if not per_tick:
        return broadcast_inputs(x, B)[:, :, None, :]
    if x.ndim == 3:
        x = np.broadcast_to(x, (B,) + x.shape)
    if x.ndim != 4 or x.shape[0] != B or x.shape[2] != K:
        raise ValueError(f"per-tick inputs must be [B, S, K, n_in] or [S, K, n_in] with K={K}, got {x.shape}")
    return x


def evaluate_lanes(substrate, circuits, inputs, K: int, library=None, per_tick: bool = False,
                   nets=None) -> EvalResult:
    """Evaluate every circuit on every one of E input streams (inputs [E, S, n_in] or [E, S, K, n_in] per tick).
    Lane b * E + e of the result is circuit b on stream e, each starting from the circuit's initial state. Circuits
    are flattened once (`nets`: their NetBatch, when the caller already has it) and shared across their E lanes:
    in place when the backend exposes `evaluate_batch`, as per-lane netlists for `evaluate_netlists`."""
    x = np.asarray(inputs, np.uint8)
    E, B = x.shape[0], len(circuits)
    xb = np.broadcast_to(x[None], (B,) + x.shape).reshape((B * E,) + x.shape[1:])
    if hasattr(substrate, "evaluate_batch"):
        from ..circuit.netbatch import flatten_batch
        nb = nets if nets is not None else flatten_batch(circuits, library)
        return substrate.evaluate_batch(nb, np.repeat(np.arange(B, dtype=np.int64), E), xb, K, per_tick=per_tick)
    if hasattr(substrate, "evaluate_netlists"):
        if nets is not None:
            ns = nets.netlists()
        else:
            from ..circuit.flatten import flatten
            ns = [flatten(c, library) for c in circuits]
        return substrate.evaluate_netlists([n for n in ns for _ in range(E)], xb, K, per_tick=per_tick)
    return substrate.evaluate([c for c in circuits for _ in range(E)], xb, K, library, per_tick=per_tick)



def evaluate_many(substrate, jobs) -> list:
    """Several batches through one substrate: jobs are (nets, lane_circ, inputs, K, per_tick) (grown.sim.stepper.Job)
    with nets a NetBatch-like object (goff, poff, ...) or a list of flattened Netlists. Substrates that implement
    `evaluate_many` (FreerunSubstrate: one engine batch for all jobs) are called once; otherwise every job is one
    `evaluate_batch` / `evaluate_netlists` call. -> one EvalResult per job."""
    if hasattr(substrate, "evaluate_many"):
        return substrate.evaluate_many(jobs)
    from .stepper import Job, NetArrays
    out = []
    for j in jobs:
        j = j if isinstance(j, Job) else Job(*j)
        if hasattr(substrate, "evaluate_batch"):
            out.append(substrate.evaluate_batch(j.nets, j.lane_circ, j.x, j.K, per_tick=j.per_tick))
            continue
        from .freerun_backend import _netlists
        nets = j.nets if isinstance(j.nets, list) else _netlists(NetArrays.of(j.nets))
        lc = range(len(nets)) if j.lane_circ is None else np.asarray(j.lane_circ)
        out.append(substrate.evaluate_netlists([nets[c] for c in lc], j.x, j.K, per_tick=j.per_tick))
    return out


# ---- one call per generation: several lane batches at once (additive protocol) -------------------------------------

@dataclass(eq=False)
class LaneBatch:
    """One request of `evaluate_requests`: lanes of flattened circuits with shared widths, K and episode length.

    nets         circuit.netbatch.NetBatch: the request's circuits (same n_inputs and n_out)
    lane_circ    int64 [L]: circuit of each lane (an index into nets)
    streams      uint8 [E, S, Kx, n_in]: distinct input streams (Kx = K per tick, 1 = held for the step)
    lane_stream  int64 [L]: stream of each lane
    K            ticks per control step

    Lane l runs circuit lane_circ[l] on streams[lane_stream[l]] from the circuit's initial state. Inputs are kept
    as streams plus an index so a backend can upload E streams instead of L copies; `inputs()` materializes them."""
    nets: object
    lane_circ: np.ndarray
    streams: np.ndarray
    lane_stream: np.ndarray
    K: int

    def inputs(self) -> np.ndarray:
        """uint8 [L, S, Kx, n_in]."""
        return self.streams[self.lane_stream]


@dataclass(eq=False)
class EventLaneBatch:
    """A request with sparse inputs (the physical-dt regime, K = dt / tick up to 10^8): like LaneBatch, but the
    distinct input streams are grown.sim.stepper.EventStreams (per stream, (t_tick, wire, value) input changes sorted
    by time; the t = 0 events are the power-up values) and the episode length S (control steps) is explicit. Step s
    samples ticks (sK, (s+1)K] (module docstring). FreerunSubstrate evaluates these natively (one engine batch,
    no per-tick arrays); `inputs()` materializes per-tick arrays for other backends (small S * K only)."""
    nets: object
    lane_circ: np.ndarray
    streams: object
    lane_stream: np.ndarray
    K: int
    S: int
    budget: float | None = None     # events per step: a substrate MAY stop a lane after the first step over it
    #                                 (EvalResult.over_budget; later steps report outputs 0, high 0, events 0)

    MAX_DENSE_TICKS = 1 << 22

    def inputs(self) -> np.ndarray:
        """uint8 [L, S, K, n_in] per tick (for backends without event streams)."""
        T = int(self.S) * int(self.K)
        if T > self.MAX_DENSE_TICKS:
            raise ValueError(f"{T} ticks per episode is too long for per-tick inputs; use the freerun step path")
        st = self.streams
        I = int(self.nets.n_inputs)
        x = np.zeros((len(st), T, I), np.uint8)
        e = st.stream_of()
        for k in np.argsort(st.t, kind="stable"):               # in time order: later events overwrite the tail
            if st.t[k] < T:
                x[e[k], st.t[k]:, st.wire[k]] = st.v[k]
        return x.reshape(len(st), int(self.S), int(self.K), I)[np.asarray(self.lane_stream)]


def evaluate_requests(substrate, requests) -> list:
    """Evaluate several LaneBatches -> one EvalResult per request (lanes in request order).

    A backend that can run everything in one launch (per-lane K, widths and lengths) implements either
    `substrate.evaluate_requests(requests)` (LaneBatches) or `substrate.evaluate_many(jobs)` with
    job = (nets, lane_circ, inputs uint8 [L, S, Kx, n_in], K, per_tick); the search calls this once per generation
    with every task's lanes. Otherwise this loops over the per-request protocol: `evaluate_batch(nets, lane_circ,
    inputs, K, per_tick)` when the backend has it, else `evaluate_netlists` with one netlist per lane. In that loop an
    EventLaneBatch goes to `evaluate_events(nets, lane_circ, streams, lane_stream, K, S, budget)` when the backend has
    it (the stand-in, the reference EventSubstrate), else through `inputs()` (per-tick arrays, short episodes only)."""
    native = getattr(substrate, "evaluate_requests", None)
    if native is not None:
        return native(requests)
    many = getattr(substrate, "evaluate_many", None)
    if many is not None:
        jobs = []
        for r in requests:
            x = r.inputs()
            jobs.append((r.nets, r.lane_circ, x, r.K, x.shape[2] > 1))
        return many(jobs)
    out = []
    for r in requests:
        if isinstance(r, EventLaneBatch) and hasattr(substrate, "evaluate_events"):
            out.append(substrate.evaluate_events(r.nets, r.lane_circ, r.streams, r.lane_stream, r.K, r.S, r.budget))
            continue
        x = r.inputs()
        per_tick = x.shape[2] > 1
        xin = x if per_tick else x[:, :, 0]
        if hasattr(substrate, "evaluate_batch"):
            out.append(substrate.evaluate_batch(r.nets, r.lane_circ, xin, r.K, per_tick=per_tick))
        else:
            nets = r.nets.netlists()
            out.append(substrate.evaluate_netlists([nets[c] for c in r.lane_circ.tolist()], xin, r.K,
                                                   per_tick=per_tick))
    return out


# ---- event-stream helpers (docs/grown_time.md); streams are grown.sim.stepper.EventStreams -------------------------

def _streams_cls():
    from .stepper import EventStreams          # stepper imports this module: import lazily
    return EventStreams


def stream_counts(st) -> np.ndarray:
    """Events per stream [E]."""
    return np.diff(np.asarray(st.off))


def stream_rows(st, e: int):
    """(t, wire, v) of stream e."""
    a, b = int(st.off[e]), int(st.off[e + 1])
    return st.t[a:b], st.wire[a:b], st.v[a:b]


def events_from_dense(x, K: int, keep: bool = True):
    """EventStreams of dense inputs x uint8 [E, S, Kx, n_in] (Kx = K: per tick; Kx = 1: held for the step). Row
    (s, k) is applied at tick s*K + k*(K // Kx); only changes are emitted (power-up 1s at t = 0). keep: cache x on the
    streams as `.dense` so dense consumers (the stand-in at legacy K) never re-expand."""
    x = np.asarray(x, np.uint8)
    if x.ndim == 3:
        x = x[:, :, None, :]
    E, S, Kx, I = x.shape
    if Kx not in (1, K):
        raise ValueError(f"dense inputs need Kx = 1 or K = {K}, got {Kx}")
    flat = x.reshape(E, S * Kx, I)
    prev = np.concatenate([np.zeros_like(flat[:, :1]), flat[:, :-1]], 1)
    e, r, i = np.nonzero(flat != prev)
    off = np.zeros(E + 1, np.int64)
    off[1:] = np.cumsum(np.bincount(e, minlength=E))
    t = (r // Kx).astype(np.int64) * K + (r % Kx).astype(np.int64) * (K // Kx)
    st = _streams_cls()(off, t, i.astype(np.int32), flat[e, r, i].astype(np.uint8))
    st.dense = x if keep else None
    return st


def events_to_dense(st, S: int, K: int, n_in: int) -> np.ndarray:
    """uint8 [E, S, Kx, n_in]: the cached dense array when the streams carry one (Kx = K or 1), else the wire values
    at every tick (Kx = K; K <= DENSE_MAX_K: the stand-in and tests). Later events overwrite earlier ones."""
    d = getattr(st, "dense", None)
    if d is not None and d.shape[1] == S and d.shape[3] == n_in and d.shape[2] in (1, K):
        return d
    if K > DENSE_MAX_K:
        raise ValueError(f"K = {K} ticks per step is above DENSE_MAX_K = {DENSE_MAX_K}: per-tick inputs would be "
                         f"{S * K} ticks per stream; this needs an event-driven substrate (docs/grown_time.md)")
    T = S * K
    E = len(st.off) - 1
    out = np.zeros((E, T, n_in), np.uint8)
    for e in range(E):
        t, w, v = stream_rows(st, e)
        for i in np.unique(w):
            m = (w == i) & (t < T)
            ti, vi = t[m], v[m]
            if len(ti):
                out[e, ti[0]:, i] = np.repeat(vi, np.diff(np.concatenate([ti, [T]])))
    return out.reshape(E, S, K, n_in)


def select_streams(st, idx):
    """The streams idx (in that order), keeping a dense cache."""
    idx = np.asarray(idx, np.int64)
    off = np.zeros(len(idx) + 1, np.int64)
    off[1:] = np.cumsum(stream_counts(st)[idx])
    rows = np.concatenate([np.arange(st.off[e], st.off[e + 1]) for e in idx]) if len(idx) else np.zeros(0, np.int64)
    out = _streams_cls()(off, st.t[rows], st.wire[rows], st.v[rows])
    d = getattr(st, "dense", None)
    out.dense = None if d is None else d[idx]
    return out


def concat_streams(parts):
    """One EventStreams with the streams of every part in order (dense caches concatenated when all have one)."""
    parts = list(parts)
    if len(parts) == 1:
        return parts[0]
    cnt = np.concatenate([stream_counts(p) for p in parts])
    off = np.zeros(len(cnt) + 1, np.int64)
    off[1:] = np.cumsum(cnt)
    out = _streams_cls()(off, np.concatenate([p.t for p in parts]), np.concatenate([p.wire for p in parts]),
                         np.concatenate([p.v for p in parts]))
    ds = [getattr(p, "dense", None) for p in parts]
    out.dense = np.concatenate(ds) if all(d is not None for d in ds) and len({d.shape[1:] for d in ds}) == 1 \
        else None
    return out


def window_counts(st, S: int, K: int, lo: int = 1) -> np.ndarray:
    """int64 [E, S]: events with t in sK + lo .. (s+1)K + lo - 1 (lo = 1: the gate window (sK, (s+1)K])."""
    sid = st.stream_of()
    s = (st.t - lo) // K
    ok = (st.t >= lo) & (s < S)
    out = np.zeros((len(st), S), np.int64)
    np.add.at(out, (sid[ok], s[ok]), 1)
    return out


def wire_step_stats(t, v, S: int, K: int):
    """One input wire's events (t sorted, v) -> (value in effect at tick (s+1)K - 1, uint8 [S]; ticks at 1 in
    [sK, (s+1)K), int64 [S]). The wire is 0 before its first event; events that do not change it are harmless."""
    t = np.asarray(t, np.int64)
    v = np.asarray(v, np.int64)
    ends = (np.arange(S, dtype=np.int64) + 1) * K
    if not len(t):
        return np.zeros(S, np.uint8), np.zeros(S, np.int64)
    seg = np.diff(np.concatenate([t, [max(int(t[-1]), int(ends[-1]))]]))
    C = np.concatenate([[0], np.cumsum(v * seg)])            # C[k]: ticks at 1 in [0, t[k])

    def H(tau):                                               # ticks at 1 in [0, tau)
        k = np.searchsorted(t, tau, side="left") - 1          # last event strictly before tau
        kk = np.maximum(k, 0)
        return np.where(k >= 0, C[kk] + v[kk] * (tau - t[kk]), 0)
    k = np.searchsorted(t, ends - 1, side="right") - 1
    val = np.where(k >= 0, v[np.maximum(k, 0)], 0).astype(np.uint8)
    return val, (H(ends) - H(ends - K)).astype(np.int64)


def wired_outputs(streams, lane_stream, out_nets, n_in: int, K: int, S: int):
    """Outputs wired straight to a primary input (net < n_in), from the event streams without simulation.

    out_nets int [L, n_out] (or [n_out], shared): output net of every lane. Returns (outputs uint8 [L, S, n_out],
    high float32 [L, S, n_out], mask bool [L, n_out] of the wired slots); other slots are 0. Convention (the
    stand-in's): the value in effect at tick (s+1)K - 1, high over ticks [sK, (s+1)K)."""
    lane_stream = np.asarray(lane_stream, np.int64)
    L = len(lane_stream)
    o = np.asarray(out_nets, np.int64)
    if o.ndim == 1:
        o = np.broadcast_to(o, (L, len(o)))
    n_out = o.shape[1]
    outv = np.zeros((L, S, n_out), np.uint8)
    high = np.zeros((L, S, n_out), np.float32)
    mask = (o >= 0) & (o < n_in)
    cache: dict = {}
    for l, j in zip(*np.nonzero(mask)):
        key = (int(lane_stream[l]), int(o[l, j]))
        if key not in cache:
            t, w, v = stream_rows(streams, key[0])
            m = w == key[1]
            cache[key] = wire_step_stats(t[m], v[m], S, K)
        val, h = cache[key]
        outv[l, :, j] = val
        high[l, :, j] = h / K
    return outv, high, mask


def step_events_from_toggles(res: EvalResult) -> np.ndarray:
    """events [L, S] when a backend reports only episode totals: the total spread evenly (approximate)."""
    S = res.outputs.shape[1]
    return np.repeat((np.asarray(res.toggles, np.float64) / max(S, 1))[:, None], S, 1)


def flag_budget(res: EvalResult, budget: float | None) -> EvalResult:
    """Set res.over_budget (bool [L]). A flag the substrate set is authoritative (it may stop a lane mid-step, so the
    stop step's events can be below that step's full count); lanes it did not flag (or all lanes, when it set none,
    e.g. no early stop was asked for) are over budget when some step's events exceed `budget`. budget None: nobody
    is over unless the substrate said so."""
    L = res.outputs.shape[0]
    sub = None if res.over_budget is None else np.asarray(res.over_budget, bool)
    if budget is None:
        res.over_budget = sub if sub is not None else np.zeros(L, bool)
        return res
    ev = np.asarray(res.events if res.events is not None else step_events_from_toggles(res))
    over = ev.max(1) > budget if ev.ndim == 2 and ev.shape[1] else np.zeros(L, bool)
    res.over_budget = over if sub is None else (sub | over)
    return res
