"""Many circuits through freerun's closed-loop stepping API (freerun SPEC §6: set_io / step / reset), in one batch.

A *lane* is one circuit on one input stream (one episode). Lanes of a batch may differ in everything: netlist,
number of inputs and outputs, ticks per control step K, number of steps S, input stream. `LaneSet.build(jobs)`
concatenates them straight into freerun's batch arrays (no per-circuit `freerun.Circuit` objects); `run_lanes`
creates one Simulator (state on the device for the CUDA backend), steps the whole batch through every episode and
returns per-lane outputs in the shared timing convention of grown.sim.base.

Timing. base.py's convention samples step s over the ticks (sK, (s+1)K]: the output of step s is the value of the net
after time (s+1)K's commits, and its duty is the mean over ticks sK+1 .. (s+1)K. freerun's step(T, x) applies a dense
input row at the current time t and reports values at T- and high time over [t, T), i.e. one tick earlier. So the
stepper cuts time at every sK (where step s's inputs arrive, as a dense row) and at every sK + 1 (where step s's
sampling window starts): two calls per control step, [sK, sK+1) and [sK+1, (s+1)K). The duty of step s is the high
time of the calls in [sK+1, (s+1)K+1) and its output the value at the end of the call that ends at (s+1)K+1. With
K = 1 the two cuts coincide (one call per step). Lanes with different K are cut at the union of their boundaries.
Input changes that do not fall on a call boundary (per-tick sigma-delta codes, or another lane's K) are queued as
stimulus triples (`add_stimulus_arrays`), which freerun defines as the same thing as a dense input (SPEC §6).

Multi-step runs. Since every lane's inputs are known for the whole episode (open-loop streams), `run_lanes` by default
makes ONE engine call per batch: freerun's `Simulator.run_steps` (its SPEC §13) with offset 1, which processes exactly
the same times and reports the same per-step outputs, high time and gate events at the same boundaries, with the input
changes as stimulus triples and the event budget applied inside the engine. Results are bit-identical to the two-call
step loop below for every lane under budget (tests/grown/test_stepper.py); the loop stays (`multistep=False`,
substrate.stimulus: step) for differential tests and for freerun libraries without run_steps.

Inputs are event streams. Internally every job's inputs are `EventStreams`: per distinct stream, (t, wire, value)
input changes sorted by time, exactly freerun stimulus triples; dense per-tick arrays (the K = 8 regime, base.py's
`inputs [L, S, Kx, n_in]`) are converted to their changes first, so nothing in the stepper scales with K except the
engine's own event count. Changes at a call boundary go in as that call's dense row (held codes change only at step
boundaries); the others as triples. Step lengths of 10^5-10^8 ticks (physical dt over a 10 ps tick) cost the same
two engine calls per step as K = 8; a quiescent circuit costs only those calls, an oscillator its real events.

Power-up. Every lane's primary inputs start at the stream's t = 0 values (the netlist's input init; wires without a
t = 0 event start at 0), exactly as the stimulus path did, so the power-up evaluation already sees step 0's inputs
(this matters with non-unit delays).

Toggles of gate nets are freerun's per-step event counts (`Simulator.events`, net changes including primary inputs)
minus the input changes the stepper applied, per step (`LaneResult.events` [L, S]: gate-net transitions in step s's
sampling window, the per-step event budget's measure) and per episode (`toggles`); outputs wired straight to a primary
input are read from the stream (base.py: value at the step's last tick, mean over ticks sK .. (s+1)K - 1), not from
the engine. No traces and no per-net observables are kept (`watch=None,
net_observables=False`) unless per-net toggle slots are requested (`net_toggles=True`, for exact energy): then every
gate net is an extra output slot with toggles.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .base import tick_inputs


BUDGET_CHUNK = 1024      # ticks: the first chunk of a long call when an event budget is set (then x4 each)
# Budget-following chunks (2026-10-10, physical-K E1 agent; flagged change to the adapter): after each chunk, the
# next cut is pulled in to where the busiest lane still under budget would cross it at its rate so far in the step
# (+6 %), but never closer than max(BUDGET_CHUNK, half the ticks already run in this call). A steady oscillator then
# stops at ~1.06x its budget instead of up to 4x; lanes that stay under budget are unaffected (cuts never change
# results, only where the budget is checked). False restores the plain geometric chunks.
BUDGET_FOLLOW = True
# Profiling: when a list, every multi-step engine call appends a dict (lanes, ticks of the longest lane, engine events,
# the largest lane's events, wall and engine-kernel seconds), so a profile can tell a call bound by its busiest lane
# (events are processed in time order within a lane) from one bound by the total. None: off.
CALL_LOG: list | None = None


def _budget_cut(used, budget, active, elapsed, T: int, st0: int):
    """The next budget check time (BUDGET_FOLLOW): where the first active lane would cross its budget at its event
    rate so far in the current step, plus 6 %, floored at T + max(BUDGET_CHUNK, (T - st0) // 2). None: no active
    lane has events."""
    busy = active & (used > 0) & np.isfinite(budget)
    if not busy.any():
        return None
    u = used[busy].astype(np.float64)
    rate = u / np.maximum(elapsed[busy], 1)
    ttb = float(np.min((budget[busy] - u) / rate))
    lo = T + max(BUDGET_CHUNK, (T - st0) // 2)
    return max(T + int(math.ceil(max(ttb, 0.0) * 1.0625)) + 1, lo)


def ranges(starts, lens) -> np.ndarray:
    """Concatenated aranges [starts[i], starts[i] + lens[i])."""
    starts = np.asarray(starts, np.int64)
    lens = np.asarray(lens, np.int64)
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    off = np.repeat(np.cumsum(lens) - lens, lens)
    return np.repeat(starts, lens) + (np.arange(tot, dtype=np.int64) - off)


def stream_hash(x: np.ndarray) -> np.ndarray:
    """uint64 per row of uint8 [n, ...]: a polynomial hash of the row's bytes (with its length), vectorized."""
    n = x.shape[0]
    b = np.ascontiguousarray(x).reshape(n, -1)
    pad = (-b.shape[1]) % 8
    if pad:
        b = np.concatenate([b, np.zeros((n, pad), np.uint8)], 1)
    w = b.view(np.uint64)
    with np.errstate(over="ignore"):
        if w.shape[1]:
            pw = np.cumprod(np.full(w.shape[1], 0x9E3779B97F4A7C15, np.uint64))
            h = (_mix(w) * pw[None, :]).sum(1, dtype=np.uint64)
        else:
            h = np.zeros(n, np.uint64)
    return _mix(h ^ np.uint64(b.shape[1]))


def _mix(x):
    """splitmix64 finalizer on uint64 arrays."""
    with np.errstate(over="ignore"):
        x = np.asarray(x, np.uint64).copy()
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


@dataclass(eq=False)
class EventStreams:
    """Input event streams (freerun stimulus triples): stream e's events are off[e] .. off[e+1]-1, sorted by (t, wire),
    at most one per (t, wire). t is in ticks from the episode start; the events at t = 0 are the power-up input
    values (wires without one start at 0). An event that does not change its wire is allowed and has no effect."""
    off: np.ndarray     # int64 [E+1]
    t: np.ndarray       # int64 [M]
    wire: np.ndarray    # int32 [M]
    v: np.ndarray       # uint8 [M]

    def __len__(self):
        return len(self.off) - 1

    @staticmethod
    def from_ticks(x) -> "EventStreams":
        """From per-tick inputs uint8 [E, T, I]: every wire's value at t = 0, then its changes."""
        x = np.asarray(x, np.uint8)
        E, T, I = x.shape
        e0 = np.repeat(np.arange(E, dtype=np.int64), I)
        w0 = np.tile(np.arange(I, dtype=np.int64), E)
        e1, u, w1 = np.nonzero(x[:, 1:] != x[:, :-1]) if T > 1 else (np.zeros(0, np.int64),) * 3
        e = np.concatenate([e0, e1])
        t = np.concatenate([np.zeros(E * I, np.int64), u.astype(np.int64) + 1])
        w = np.concatenate([w0, w1])
        o = np.lexsort((w, t, e))
        e, t, w = e[o], t[o], w[o]
        v = x[e, t, w] if len(e) else np.zeros(0, np.uint8)
        off = np.searchsorted(e, np.arange(E + 1)).astype(np.int64)
        return EventStreams(off, t.astype(np.int64), w.astype(np.int32), v.astype(np.uint8))

    @staticmethod
    def from_lists(streams) -> "EventStreams":
        """From one list of (t, wire, value) per stream (sorted here)."""
        rows = [np.asarray(list(s), np.int64).reshape(-1, 3) for s in streams]
        e = np.concatenate([np.full(len(r), i, np.int64) for i, r in enumerate(rows)]) if rows else np.zeros(0, np.int64)
        a = np.concatenate(rows) if rows else np.zeros((0, 3), np.int64)
        o = np.lexsort((a[:, 1], a[:, 0], e))
        e, a = e[o], a[o]
        return EventStreams(np.searchsorted(e, np.arange(len(rows) + 1)).astype(np.int64), a[:, 0].copy(),
                            a[:, 1].astype(np.int32), a[:, 2].astype(np.uint8))

    def stream_of(self) -> np.ndarray:
        return np.repeat(np.arange(len(self), dtype=np.int64), np.diff(self.off))

    def validate(self, n_in: int) -> None:
        e = self.stream_of()
        if len(self.t):
            if self.t.min() < 0 or self.wire.min() < 0 or self.wire.max() >= n_in or self.v.max() > 1:
                raise ValueError("events need t >= 0, 0 <= wire < n_in and values 0/1")
            key_bad = (e[1:] == e[:-1]) & ((self.t[1:] < self.t[:-1]) |
                                           ((self.t[1:] == self.t[:-1]) & (self.wire[1:] <= self.wire[:-1])))
            if key_bad.any():
                raise ValueError("events must be sorted by (t, wire) within each stream, one per (t, wire)")

    def initial(self, n_in: int) -> np.ndarray:
        """uint8 [E, n_in]: the power-up input values (the t = 0 events)."""
        out = np.zeros((len(self), n_in), np.uint8)
        m = self.t == 0
        out[self.stream_of()[m], self.wire[m]] = self.v[m]
        return out

    def effective(self, n_in: int, t_max: int | None = None):
        """(stream, t, wire, value) of the events at t > 0 (and <= t_max) that change their wire, sorted by
        (stream, t, wire)."""
        e = self.stream_of()
        m = self.t > 0
        if t_max is not None:
            m &= self.t <= t_max
        e, t, w, v = e[m], self.t[m], self.wire[m].astype(np.int64), self.v[m]
        o = np.lexsort((t, w, e))                         # per (stream, wire) in time order
        e2, w2, v2 = e[o], w[o], v[o]
        prev = np.empty(len(o), np.uint8)
        if len(o):
            prev[1:] = v2[:-1]
            first = np.ones(len(o), bool)
            first[1:] = (e2[1:] != e2[:-1]) | (w2[1:] != w2[:-1])
            init = self.initial(n_in)
            prev[first] = init[e2[first], w2[first]]
            keep = np.zeros(len(o), bool)
            keep[o] = v2 != prev
        else:
            keep = np.zeros(0, bool)
        return e[keep], t[keep], w[keep], v[keep]


@dataclass(eq=False)
class NetArrays:
    """Flattened netlists in concatenated arrays (the layout of circuit.netbatch.NetBatch, which is accepted as is).
    goff [B+1] gate offsets, poff [G+1] global pin offsets per gate, fanin / dr / df [P] (local nets), tt [G],
    init [G] (gate nets only), outs [B, n_out] (local nets)."""
    n_inputs: int
    n_out: int
    goff: np.ndarray
    poff: np.ndarray
    fanin: np.ndarray
    tt: np.ndarray
    dr: np.ndarray
    df: np.ndarray
    init: np.ndarray
    outs: np.ndarray

    def __len__(self):
        return len(self.goff) - 1

    @staticmethod
    def of(nets) -> "NetArrays":
        """A NetBatch-like object (has goff) is returned as is; a list of flatten.Netlist is concatenated."""
        if hasattr(nets, "goff"):
            return nets
        nets = list(nets)
        if not nets:
            raise ValueError("empty batch")
        I, n_out = nets[0].n_inputs, len(nets[0].outputs)
        if any(n.n_inputs != I or len(n.outputs) != n_out for n in nets):
            raise ValueError("all circuits in a batch need the same number of inputs and outputs")
        G = np.array([n.n_gates for n in nets], np.int64)
        goff = np.zeros(len(nets) + 1, np.int64)
        np.cumsum(G, out=goff[1:])
        npins = np.concatenate([np.diff(n.fanin_ptr) for n in nets]).astype(np.int64)
        poff = np.zeros(len(npins) + 1, np.int64)
        np.cumsum(npins, out=poff[1:])
        cat = np.concatenate
        return NetArrays(I, n_out, goff, poff, cat([n.fanin for n in nets]).astype(np.int32),
                         cat([n.tt for n in nets]).astype(np.uint64), cat([n.dr for n in nets]).astype(np.int32),
                         cat([n.df for n in nets]).astype(np.int32), cat([n.init[I:] for n in nets]).astype(np.uint8),
                         np.stack([np.asarray(n.outputs) for n in nets]).astype(np.int32).reshape(len(nets), n_out))


@dataclass
class Job:
    """Circuits `nets` (NetArrays-like or a list of Netlists), lane b running circuit lane_circ[b] (None: lane b runs
    circuit b) on inputs x[b] (uint8 [L, S, n_in], or per tick [L, S, K, n_in]; [L, S, 1, n_in] is held), K ticks
    per control step. With `rows` (int64 [L]), x holds distinct streams instead and lane b reads x[rows[b]]
    (grown.sim.base.LaneBatch: streams + lane_stream). x may also be an EventStreams (sparse input changes, the
    physical-dt regime): then `S` (steps per episode) is required and `rows` maps lanes to its streams (default:
    lane b reads stream b)."""
    nets: object
    lane_circ: object
    x: object
    K: int
    per_tick: bool = False
    rows: object = None
    S: int | None = None
    budget: float | None = None     # events per step; a lane over it stops (run_lanes)


@dataclass(eq=False)
class LaneSet:
    """Every lane of a batch, in freerun's batch layout (csrc/freerun.h), plus what the stepper needs."""
    n_inputs: np.ndarray        # int32 [L]
    net_off: np.ndarray         # int64 [L+1]
    gate_off: np.ndarray
    edge_off: np.ndarray
    fanin_ptr: np.ndarray       # int32 [G + L]: per lane, G_l + 1 local pin offsets
    fanin: np.ndarray           # int32 [E] local nets
    dr: np.ndarray              # int32 [E]
    df: np.ndarray
    tt: np.ndarray              # uint64 [G]
    init: np.ndarray            # uint8 [N]: inputs = the stream's first tick, then the gates' init
    outs: np.ndarray            # int32 [L, n_out_max]: local output nets, -1 = no slot
    n_out: np.ndarray           # int64 [L]
    K: np.ndarray               # int64 [L]
    S: np.ndarray               # int64 [L]
    job: np.ndarray             # int64 [L] job index
    circ: np.ndarray            # int64 [L] circuit index within its job
    row: np.ndarray             # int64 [L] the lane's stream in its job's EventStreams
    ev: list                    # per job: EventStreams of its distinct input streams
    jitter: dict | None = None  # {seed uint64 [L], cid uint32 [L], dmin int32 [L], amp int64 [G]}
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.n_inputs)

    @property
    def n_gates(self) -> np.ndarray:
        return np.diff(self.gate_off)

    @property
    def arity(self) -> np.ndarray:
        """int64 [G]: arity of every gate."""
        L = len(self)
        fp = self.fanin_ptr.astype(np.int64)
        G = self.n_gates
        lo = ranges(self.gate_off[:-1] + np.arange(L), G)       # positions of each gate's first pointer
        return fp[lo + 1] - fp[lo]

    def edge_lane(self) -> np.ndarray:
        return np.repeat(np.arange(len(self), dtype=np.int64), np.diff(self.edge_off))

    def gate_lane(self) -> np.ndarray:
        return np.repeat(np.arange(len(self), dtype=np.int64), self.n_gates)

    def edge_gate(self) -> np.ndarray:
        """int64 [E]: global gate index of every arc."""
        return np.repeat(np.arange(int(self.gate_off[-1]), dtype=np.int64), self.arity)

    @staticmethod
    def build(jobs) -> "LaneSet":
        parts = []
        evs = []
        for j, jb in enumerate(jobs):
            na = NetArrays.of(jb.nets)
            lc = np.arange(len(na), dtype=np.int64) if jb.lane_circ is None else np.asarray(jb.lane_circ, np.int64)
            L = len(lc)
            K = int(jb.K)
            if isinstance(jb.x, EventStreams):
                ev = jb.x
                if jb.S is None:
                    raise ValueError("an event-stream job needs S (steps per episode)")
                S, I = int(jb.S), int(na.n_inputs)
                row = np.arange(L, dtype=np.int64) if jb.rows is None else np.asarray(jb.rows, np.int64)
                if len(row) != L or (L and (row.min() < 0 or row.max() >= len(ev))):
                    raise ValueError("rows needs one valid stream index per lane")
                ev.validate(I)
                evs.append(ev)
                init_in = ev.initial(I)[row] if L else np.zeros((0, I), np.uint8)
                parts.append(_circuit_part(na, lc, L, I, K, S, init_in, row))
                continue
            xin = np.asarray(jb.x, np.uint8)
            per_tick = jb.per_tick
            if xin.ndim == 4 and xin.shape[2] == 1 and K != 1:            # [., S, 1, n_in]: held for the step
                xin, per_tick = xin[:, :, 0], False
            elif xin.ndim == 4 and K == 1:
                per_tick = True
            if jb.rows is not None:                                        # distinct streams + an index per lane
                row = np.asarray(jb.rows, np.int64)
                if len(row) != L:
                    raise ValueError("rows needs one stream index per lane")
                x = tick_inputs(xin, len(xin), K, per_tick)
            else:
                row = np.arange(L, dtype=np.int64)
                x = tick_inputs(xin, L, K, per_tick)                       # [L, S, Kx, I]
            n_x, S, Kx, I = x.shape
            if I != na.n_inputs:
                raise ValueError(f"inputs have {I} bits, circuits take {na.n_inputs}")
            xj = np.ascontiguousarray(np.repeat(x, K, axis=2) if Kx == 1 and K > 1 else x).reshape(n_x, S * K, I)
            if jb.rows is None and L > 1:               # lanes usually share a few episode streams: keep each once
                h = stream_hash(xj)
                _, first, inv = np.unique(h, return_index=True, return_inverse=True)
                if np.array_equal(xj[first][inv], xj):
                    xj, row = xj[first], inv.astype(np.int64)
            evs.append(EventStreams.from_ticks(xj))
            init_in = xj[row, 0, :] if S * K else np.zeros((L, I), np.uint8)
            parts.append(_circuit_part(na, lc, L, I, K, S, init_in, row))
        Lt = sum(p["L"] for p in parts)
        n_in = np.concatenate([np.full(p["L"], p["I"], np.int32) for p in parts])
        G = np.concatenate([p["G"] for p in parts]).astype(np.int64)
        P = np.concatenate([p["P"] for p in parts]).astype(np.int64)
        N = n_in.astype(np.int64) + G
        off = lambda a: np.concatenate([[0], np.cumsum(a)]).astype(np.int64)  # noqa: E731
        net_off, gate_off, edge_off = off(N), off(G), off(P)
        init = np.zeros(int(net_off[-1]), np.uint8)
        init[ranges(net_off[:-1], n_in)] = np.concatenate([p["init_in"].reshape(-1) for p in parts])
        init[ranges(net_off[:-1] + n_in, G)] = np.concatenate([p["ginit"] for p in parts])
        W = max(p["n_out"] for p in parts)
        outs = np.full((Lt, W), -1, np.int32)
        r = 0
        for p in parts:
            outs[r:r + p["L"], :p["n_out"]] = p["outs"]
            r += p["L"]
        lanes = LaneSet(
            n_in, net_off, gate_off, edge_off, np.concatenate([p["fp"] for p in parts]).astype(np.int32),
            np.concatenate([p["fanin"] for p in parts]).astype(np.int32),
            np.concatenate([p["dr"] for p in parts]).astype(np.int32),
            np.concatenate([p["df"] for p in parts]).astype(np.int32),
            np.concatenate([p["tt"] for p in parts]).astype(np.uint64), init, outs,
            np.concatenate([np.full(p["L"], p["n_out"], np.int64) for p in parts]),
            np.concatenate([np.full(p["L"], p["K"], np.int64) for p in parts]),
            np.concatenate([np.full(p["L"], p["S"], np.int64) for p in parts]),
            np.concatenate([np.full(p["L"], j, np.int64) for j, p in enumerate(parts)]),
            np.concatenate([p["lc"] for p in parts]),
            np.concatenate([p["row"] for p in parts]), evs)
        if any(jb.budget is not None for jb in jobs):
            lanes.extra["budget"] = np.concatenate([np.full(p["L"], np.inf if jb.budget is None else float(jb.budget))
                                                    for p, jb in zip(parts, jobs)])
        lanes.validate()
        return lanes

    def validate(self) -> None:
        """freerun.Circuit's checks for every lane at once (the batch is built without Circuit objects)."""
        k = self.arity
        if len(k) and (k.min() < 0 or k.max() > 6):
            raise ValueError("gate arity must be in [0, 6]")
        if len(self.fanin):
            if min(int(self.dr.min()), int(self.df.min())) < 1:
                raise ValueError("delays must be >= 1 tick")
            nn = np.diff(self.net_off)[self.edge_lane()]
            if self.fanin.min() < 0 or (self.fanin >= nn).any():
                raise ValueError("fanin references a net out of range")
        width = np.left_shift(np.uint64(1), k.astype(np.uint64))
        mask = np.where(k == 6, np.uint64(0), ~((np.uint64(1) << width) - np.uint64(1)))
        if (self.tt & mask).any():
            raise ValueError("truth table has bits set at or above 2^arity")
        const = np.flatnonzero(k == 0)
        if len(const):
            gl = self.gate_lane()[const]
            net = self.net_off[gl] + self.n_inputs[gl] + const - self.gate_off[gl]
            if (self.init[net] != (self.tt[const] & np.uint64(1))).any():
                raise ValueError("constant (arity-0) gate init must equal its truth table bit 0")
        if self.init.max(initial=0) > 1:
            raise ValueError("init must be 0/1")

    def fr_batch(self):
        """A freerun.batch.Batch over these arrays (no Circuit objects; the engines derive the fanout)."""
        from .freerun_backend import import_freerun
        import_freerun()
        from freerun.batch import Batch
        b = Batch.__new__(Batch)
        b.circuits = None
        b.B = len(self)
        b.net_off, b.gate_off, b.edge_off = (np.ascontiguousarray(a, np.int64)
                                             for a in (self.net_off, self.gate_off, self.edge_off))
        b.n_inputs = np.ascontiguousarray(self.n_inputs, np.int32)
        b.fanin_ptr = np.ascontiguousarray(self.fanin_ptr, np.int32)
        b.fanin = np.ascontiguousarray(self.fanin, np.int32)
        b.dr = np.ascontiguousarray(self.dr, np.int32)
        b.df = np.ascontiguousarray(self.df, np.int32)
        b.tt = np.ascontiguousarray(self.tt, np.uint64)
        b.init = np.ascontiguousarray(self.init, np.uint8)
        j = self.jitter
        b.has_jitter = j is not None
        L = len(self)
        b.jit_seed = np.ascontiguousarray(j["seed"], np.uint64) if j else np.zeros(L, np.uint64)
        b.jit_cid = np.ascontiguousarray(j["cid"], np.uint32) if j else np.zeros(L, np.uint32)
        b.jit_dmin = np.ascontiguousarray(j["dmin"], np.int32) if j else np.zeros(L, np.int32)
        b.jit_amp = np.ascontiguousarray(j["amp"], np.int64) if j else np.zeros(int(self.gate_off[-1]), np.int64)
        return b

    def circuits(self) -> list:
        """freerun.Circuit per lane (the reference backend needs them; slow)."""
        from .freerun_backend import import_freerun
        import_freerun()
        from freerun.netlist import Circuit, Jitter
        out = []
        for b in range(len(self)):
            g0, g1 = int(self.gate_off[b]), int(self.gate_off[b + 1])
            e0, e1 = int(self.edge_off[b]), int(self.edge_off[b + 1])
            fp = self.fanin_ptr[g0 + b:g1 + b + 1].astype(np.int64)
            jit = None
            if self.jitter is not None and int(self.jitter["dmin"][b]) > 0:
                jit = Jitter(self.jitter["amp"][g0:g1], int(self.jitter["seed"][b]), int(self.jitter["cid"][b]),
                             int(self.jitter["dmin"][b]))
            out.append(Circuit(int(self.n_inputs[b]), fp, self.fanin[e0:e1], self.tt[g0:g1], self.dr[e0:e1],
                               self.df[e0:e1], self.init[self.net_off[b]:self.net_off[b + 1]], jit))
        return out

    def select(self, idx) -> "LaneSet":
        """The lanes idx (any order, repeats allowed) as a new LaneSet sharing the jobs' input streams."""
        idx = np.asarray(idx, np.int64)
        G, P = self.n_gates[idx], np.diff(self.edge_off)[idx]
        N = np.diff(self.net_off)[idx]
        off = lambda a: np.concatenate([[0], np.cumsum(a)]).astype(np.int64)  # noqa: E731
        gidx = ranges(self.gate_off[idx], G)
        pidx = ranges(self.edge_off[idx], P)
        fidx = ranges(self.gate_off[idx] + idx, G + 1)
        jit = None
        if self.jitter is not None:
            jit = dict(seed=self.jitter["seed"][idx], cid=self.jitter["cid"][idx], dmin=self.jitter["dmin"][idx],
                       amp=self.jitter["amp"][gidx])
        return LaneSet(self.n_inputs[idx], off(N), off(G), off(P), self.fanin_ptr[fidx], self.fanin[pidx],
                       self.dr[pidx], self.df[pidx], self.tt[gidx], self.init[ranges(self.net_off[idx], N)],
                       self.outs[idx], self.n_out[idx], self.K[idx], self.S[idx], self.job[idx], self.circ[idx],
                       self.row[idx], self.ev, jit, {k: v[idx] for k, v in self.extra.items()})

    def lane_events(self, b: int):
        """(t, wire, value) arrays of lane b's input stream."""
        ev = self.ev[int(self.job[b])]
        r = int(self.row[b])
        s = slice(int(ev.off[r]), int(ev.off[r + 1]))
        return ev.t[s], ev.wire[s], ev.v[s]

    def lane_x(self, b: int) -> np.ndarray:
        """uint8 [S*K, n_in]: lane b's input at every tick (reconstructed from its events; small K only)."""
        T, I = int(self.S[b] * self.K[b]), int(self.n_inputs[b])
        x = np.zeros((T, I), np.uint8)
        t, w, v = self.lane_events(b)
        for tt, ww, vv in zip(t.tolist(), w.tolist(), v.tolist()):
            if tt < T:
                x[tt:, ww] = vv
        return x


def _circuit_part(na, lc, L, I, K, S, init_in, row) -> dict:
    """The netlist arrays of a job's lanes (circuit lc[b] for lane b)."""
    goff = np.asarray(na.goff, np.int64)
    poff = np.asarray(na.poff, np.int64)
    G = goff[lc + 1] - goff[lc]
    gidx = ranges(goff[lc], G)
    p0, p1 = poff[goff[lc]], poff[goff[lc + 1]]
    pidx = ranges(p0, p1 - p0)
    fp = poff[ranges(goff[lc], G + 1)] - np.repeat(p0, G + 1)
    outs = np.asarray(na.outs, np.int64)[lc].reshape(L, na.n_out)
    return dict(L=L, I=I, G=G, P=p1 - p0, tt=np.asarray(na.tt, np.uint64)[gidx],
                ginit=np.asarray(na.init, np.uint8)[gidx], fanin=np.asarray(na.fanin, np.int32)[pidx],
                dr=np.asarray(na.dr, np.int32)[pidx], df=np.asarray(na.df, np.int32)[pidx], fp=fp,
                outs=outs, n_out=na.n_out, K=K, S=S, init_in=init_in, lc=lc, row=row)


def _integral(times, vals, v0, points):
    """Integral over [0, p) of the piecewise-constant signal v0, then vals[k] from times[k] on; and its value at p."""
    bt = np.concatenate([[0], times]).astype(np.int64)
    bv = np.concatenate([[v0], vals]).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(np.diff(bt) * bv[:-1])])
    i = np.searchsorted(bt, points, side="right") - 1
    return cum[i] + bv[i] * (points - bt[i]), bv[i]


@dataclass
class LaneResult:
    outputs: np.ndarray     # uint8 [L, S_max, n_out_max] (per lane only [:S_l, :n_out_l] is meaningful)
    high: np.ndarray        # float32 [L, S_max, n_out_max]
    toggles: np.ndarray     # float64 [L] gate-net transitions over the lane's episode
    net_toggles: np.ndarray | None = None   # int64 [G] per gate net over the episode (net_toggles=True)
    calls: int = 0
    events: np.ndarray | None = None        # int64 [L, S_max]: gate-net transitions in each step's sampling window
    over_budget: np.ndarray | None = None   # bool [L]: stopped after a step over its event budget (None: no budget)


class Schedule:
    """Call boundaries for a LaneSet, input changes split into dense rows (at boundaries) and triples (between)."""

    def __init__(self, lanes: LaneSet):
        K, S = lanes.K, lanes.S
        pts = [np.zeros(1, np.int64)]
        for k, s in set(zip(K.tolist(), S.tolist())):
            t = np.arange(s + 1, dtype=np.int64) * k
            pts += [t, t + 1]
        self.bnd = np.unique(np.concatenate(pts))
        self.n_calls = len(self.bnd) - 1
        self.end = S * K + 1                                    # each lane's episode is [0, S K + 1)
        L = len(lanes)
        n_in = int(lanes.n_inputs.max(initial=0))
        self.n_in = n_in
        starts = self.bnd[:-1]
        n = self.n_calls
        has = np.zeros(n, bool)
        tl, tu, tn, tv = [], [], [], []
        self.in_calls = np.zeros((n, L), np.int64)               # effective input changes per call and lane
        self.in_changes = np.zeros(L, np.int64)
        self.in_wire_changes = []                                 # per job: int64 [streams, n_in_j] over the episode
        dense_parts = []
        for j, ev in enumerate(lanes.ev):                        # per distinct stream, then fanned out to its lanes
            lanes_j = np.flatnonzero(lanes.job == j)
            if not len(lanes_j):
                self.in_wire_changes.append(None)
                continue
            I = int(lanes.n_inputs[lanes_j[0]])
            T_end = int((lanes.S[lanes_j] * lanes.K[lanes_j]).max())
            e, t, w, v = ev.effective(I, T_end)
            U = len(ev)
            row = lanes.row[lanes_j]
            self.in_wire_changes.append(np.bincount(e * I + w, minlength=U * I).reshape(U, I))
            call = np.searchsorted(self.bnd, t, side="right") - 1
            at = self.bnd[call] == t
            pc = np.bincount(e * n + call, minlength=U * n).reshape(U, n)
            self.in_calls[:, lanes_j] = pc[row].T
            self.in_changes[lanes_j] = pc.sum(1)[row]
            if at.any():                                         # dense rows: every wire's value at the call start
                has[call[at]] = True
                dense_parts.append((lanes_j, row, I, ev, e, t, w, v))
            m = np.flatnonzero(~at)
            if len(m):                                           # triples: every lane of the event's stream
                order = np.argsort(row, kind="stable")
                cnt = np.bincount(row, minlength=U)
                first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
                eb = e[m]
                rep = cnt[eb]
                k = np.repeat(np.arange(len(m)), rep)
                r = np.arange(len(k)) - np.repeat(np.cumsum(rep) - rep, rep)
                tl.append(lanes_j[order[first[eb][k] + r]])
                tu.append(t[m][k])
                tn.append(w[m][k])
                tv.append(v[m][k].astype(np.int64))
        self.has_dense = has
        dcalls = np.flatnonzero(has)
        self.dense_of = np.full(n, -1, np.int64)
        self.dense_of[dcalls] = np.arange(len(dcalls))
        self.dense = np.zeros((len(dcalls), L, n_in), np.uint8)
        a = starts[dcalls]
        for lanes_j, row, I, ev, e, t, w, v in dense_parts:
            U = len(ev)
            init = ev.initial(I)
            g = e * I + w                                        # (stream, wire) groups, events in time order
            o = np.lexsort((t, g))
            g, tt, vv = g[o], t[o], v[o]
            big = int(self.bnd[-1]) + 2
            keys = g * big + tt
            q = (np.arange(U * I, dtype=np.int64)[:, None] * big + a[None, :]).ravel()
            k = np.searchsorted(keys, q, side="right") - 1
            gq = np.repeat(np.arange(U * I, dtype=np.int64), len(a))
            ok = (k >= 0) & (g[np.maximum(k, 0)] == gq) if len(g) else np.zeros(len(q), bool)
            val = np.where(ok, vv[np.maximum(k, 0)] if len(vv) else 0, init.reshape(-1)[gq]).reshape(U, I, len(a))
            self.dense[:, lanes_j, :I] = np.transpose(val[row], (2, 0, 1))
        if tl:
            tl, tu, tn, tv = (np.concatenate(a) for a in (tl, tu, tn, tv))
        else:
            tl = tu = tn = tv = np.zeros(0, np.int64)
        call = np.searchsorted(self.bnd, tu, side="right") - 1
        o = np.lexsort((tn, tu, tl, call))
        self.trip = (tl[o].astype(np.int64), tu[o].astype(np.int64), tn[o].astype(np.int64), tv[o].astype(np.int64))
        self.trip_off = np.searchsorted(call[o], np.arange(self.n_calls + 1))


def run_lanes(lanes: LaneSet, backend: str = "cpu", threads: int = 0, net_toggles: bool = False, timing=None,
              sim_kw: dict | None = None, repeat: int = 1, multistep: bool | None = None) -> LaneResult:
    """Run every lane through its episode (module docstring). repeat > 1 runs the episodes again on the same
    Simulator (device-resident state), resetting every lane to power-up at the current time before each repetition
    (freerun Simulator.reset, SPEC §6), and returns the last repetition. timing: a dict accumulating seconds per phase
    (create, stimulus, kernel, extract, post, destroy) plus calls / steps (control-step cuts: the step loop's engine
    calls) / launches (engine calls actually made) / lanes / ticks / events / engine_kernel (the engine's own kernel
    time, freerun Simulator.profile) / engine_windows.

    multistep (default: whenever the freerun library has it): the whole episode of every lane in one engine call,
    freerun's Simulator.run_steps (its SPEC §13), instead of the step loop's two calls per control step. Every lane's
    input events are known up front (open-loop streams), and run_steps samples exactly this module's convention
    (offset 1: step s is (sK, (s+1)K]), so outputs, high fractions, per-step events and toggles are bit-identical to the
    step loop for every lane that stays under its event budget. A lane over budget is frozen right after the tick at
    which its step count crossed the budget (the loop freezes it at the end of the call or chunk in which it sees the
    crossing): same flag and stop step, the stop step's counts are smaller, later steps are 0 in both."""
    if multistep is None or multistep:
        ok = _multistep_ok(lanes, backend)
        if ok:
            return _run_lanes_multistep(lanes, backend, threads, net_toggles, timing, sim_kw, repeat)
        if multistep:
            raise RuntimeError("multistep: this freerun library has no run_steps (or a budget is negative)")
    return _run_lanes_loop(lanes, backend, threads, net_toggles, timing, sim_kw, repeat)


def _multistep_ok(lanes: LaneSet, backend: str) -> bool:
    from .freerun_backend import import_freerun
    frs = import_freerun().sim
    if not hasattr(frs.Simulator, "run_steps") or not getattr(frs, "supports_run_steps", lambda b: False)(backend):
        return False
    budget = lanes.extra.get("budget")
    return budget is None or not (np.isfinite(budget) & (budget < 0)).any()


def _out_slots(lanes: LaneSet, net_toggles: bool):
    """Engine output slots: (gate_slot [L, W] outputs read from the engine, out_net [L, Wt] local nets (-1 unused);
    with net_toggles every gate net gets an extra slot after the W outputs)."""
    L = len(lanes)
    I_l = lanes.n_inputs.astype(np.int64)
    gate_slot = lanes.outs >= I_l[:, None]                       # outputs read from the engine
    out_net = np.where(gate_slot, lanes.outs, -1).astype(np.int32)
    if net_toggles:
        Gm = int(lanes.n_gates.max(initial=0))
        gn = np.full((L, Gm), -1, np.int32)
        gl = lanes.gate_lane()
        gi = np.arange(int(lanes.gate_off[-1])) - lanes.gate_off[gl]
        gn[gl, gi] = (I_l[gl] + gi).astype(np.int32)
        out_net = np.concatenate([out_net, gn], 1)
    if out_net.shape[1] == 0:
        out_net = np.full((L, 1), -1, np.int32)
    return gate_slot, out_net


def _fill_wired(lanes: LaneSet, gate_slot, out, high) -> None:
    """Outputs wired to a primary input: the stream itself (base.py), from its events."""
    wired = (~gate_slot) & (lanes.outs >= 0)
    cache: dict = {}
    for b, w in zip(*np.nonzero(wired)):
        Sb, Kb = int(lanes.S[b]), int(lanes.K[b])
        wire = int(lanes.outs[b, w])
        key = (int(lanes.job[b]), int(lanes.row[b]), wire, Sb, Kb)
        if key not in cache:
            t, ww, v = lanes.lane_events(b)
            m = ww == wire
            t, v = t[m], v[m]
            v0 = int(v[0]) if len(t) and t[0] == 0 else 0
            pts = np.arange(Sb + 1, dtype=np.int64) * Kb
            F, _ = _integral(t, v, v0, pts)
            _, last = _integral(t, v, v0, pts[1:] - 1)
            cache[key] = (last, (np.diff(F) / Kb).astype(np.float32))
        out[b, :Sb, w], high[b, :Sb, w] = cache[key]


def lane_triples(lanes: LaneSet):
    """Every lane's effective input changes at 0 < t <= S K as freerun stimulus arrays (off [L+1], t, wire, value),
    sorted by (lane, t, wire): each lane's slice is its stream's changes, which are already in (t, wire) order, so
    nothing is sorted."""
    L = len(lanes)
    cnt = np.zeros(L, np.int64)
    parts = []
    for j, ev in enumerate(lanes.ev):
        lanes_j = np.flatnonzero(lanes.job == j)
        if not len(lanes_j):
            continue
        I = int(lanes.n_inputs[lanes_j[0]])
        e, t, w, v = ev.effective(I, int((lanes.S[lanes_j] * lanes.K[lanes_j]).max()))   # sorted by (stream, t, wire)
        soff = np.searchsorted(e, np.arange(len(ev) + 1))
        row = lanes.row[lanes_j]
        cnt[lanes_j] = soff[row + 1] - soff[row]
        parts.append((lanes_j, soff[row], t, w, v))
    off = np.zeros(L + 1, np.int64)
    np.cumsum(cnt, out=off[1:])
    M = int(off[-1])
    T, Wr, V = np.empty(M, np.int64), np.empty(M, np.int32), np.empty(M, np.uint8)
    for lanes_j, src, t, w, v in parts:
        dst = ranges(off[lanes_j], cnt[lanes_j])
        k = ranges(src, cnt[lanes_j])
        T[dst], Wr[dst], V[dst] = t[k], w[k], v[k]
    return off, T, Wr, V


def _n_cuts(lanes: LaneSet) -> int:
    """The step loop's number of engine calls (Schedule.n_calls) without building its schedule."""
    pts = [np.zeros(1, np.int64)]
    for k, s in set(zip(lanes.K.tolist(), lanes.S.tolist())):
        t = np.arange(s + 1, dtype=np.int64) * k
        pts += [t, t + 1]
    return len(np.unique(np.concatenate(pts))) - 1


def _run_lanes_multistep(lanes: LaneSet, backend: str, threads: int, net_toggles: bool, timing, sim_kw,
                         repeat: int) -> LaneResult:
    """run_lanes through freerun's Simulator.run_steps: one engine call per repetition."""
    from .freerun_backend import import_freerun
    frs = import_freerun().sim
    tm = timing if timing is not None else {}

    def add(k, v):
        tm[k] = tm.get(k, 0.0) + v
    t0 = time.perf_counter()
    L = len(lanes)
    W = lanes.outs.shape[1]
    gate_slot, out_net = _out_slots(lanes, net_toggles)
    off, st_t, st_w, st_v = lane_triples(lanes)
    n_in = int(lanes.n_inputs.max(initial=0))
    budget = lanes.extra.get("budget")
    budgeted = budget is not None and np.isfinite(budget).any()
    t1 = time.perf_counter()
    add("stimulus", t1 - t0)
    kw = dict(sim_kw or {})
    if backend == "ref":
        sim = frs.Simulator(lanes.circuits(), backend="ref", watch=None, **kw)
    else:
        sim = frs.Simulator(lanes.fr_batch(), backend=backend, watch=None, threads=threads, net_observables=False,
                            **kw)
    sim.set_io(n_in, out_net, high=True, toggles=net_toggles)
    t2 = time.perf_counter()
    add("create", t2 - t1)
    ek = ew = ev_all = 0.0
    try:
        for rep in range(max(int(repeat), 1)):
            t_0 = 0
            if rep:
                t_0 = sim.t
                sim.reset()                                      # power-up state, shifted to the current time
            r = sim.run_steps(lanes.K, lanes.S, (off, st_t + t_0, st_w, st_v), budget=budget if budgeted else None,
                              offset=1, hold_at_end=True)
            if backend != "ref":
                p = sim.profile()
                ek += p["kernel_s"]
                ew += p["windows"]
            ev_all += float(sim.events().sum())
        t3 = time.perf_counter()
    finally:
        t4 = time.perf_counter()
        sim.close()
        add("destroy", time.perf_counter() - t4)
    add("kernel", t3 - t2)
    add("engine_kernel", ek)
    add("engine_windows", ew)
    t5 = time.perf_counter()
    Smax = int(lanes.S.max(initial=0))
    out = np.zeros((L, Smax, W), np.uint8)
    high = np.zeros((L, Smax, W), np.float32)
    if W:
        out[:] = np.where(gate_slot[:, None, :], r.values[:, :, :W], 0)
        high[:] = np.where(gate_slot[:, None, :], r.high[:, :, :W] / lanes.K[:, None, None], 0).astype(np.float32)
        _fill_wired(lanes, gate_slot, out, high)
    events = np.ascontiguousarray(r.events, dtype=np.int64)
    net_tog = None
    if net_toggles:
        Gm = out_net.shape[1] - W
        per = r.toggles[:, :, W:W + Gm].sum(1)                     # [L, Gm] over the episode
        gl = lanes.gate_lane()
        gi = np.arange(int(lanes.gate_off[-1])) - lanes.gate_off[gl]
        net_tog = per[gl, gi]
    over_budget = None
    if budgeted:
        over_budget = r.stop >= 0
        for lb in np.flatnonzero(over_budget):                   # later steps: outputs 0, high 0 (wired ones too)
            s0 = int(r.stop[lb]) + 1
            out[lb, s0:] = 0
            high[lb, s0:] = 0
    tog = events.sum(1)
    add("post", time.perf_counter() - t5)
    n = _n_cuts(lanes)
    add("calls", 1)
    add("steps", n)
    add("launches", max(int(repeat), 1))
    add("lanes", L)
    add("ticks", float((lanes.S * lanes.K + 1).sum()))
    add("events", ev_all)
    if CALL_LOG is not None:
        per_lane = events.sum(1)
        CALL_LOG.append(dict(lanes=L, max_ticks=int((lanes.S * lanes.K).max(initial=0)),
                             events=float(per_lane.sum()), max_lane_events=float(per_lane.max(initial=0)),
                             lanes_over_1e6=int((per_lane > 1e6).sum()), wall_s=t3 - t2, kernel_s=ek,
                             K=sorted(set(lanes.K.tolist()))))
    return LaneResult(out, high, tog.astype(np.float64), net_tog, n, events, over_budget)


def _run_lanes_loop(lanes: LaneSet, backend: str = "cpu", threads: int = 0, net_toggles: bool = False, timing=None,
                    sim_kw: dict | None = None, repeat: int = 1) -> LaneResult:
    """run_lanes through the step loop: two engine calls per control step (module docstring)."""
    from .freerun_backend import import_freerun
    frs = import_freerun().sim
    tm = timing if timing is not None else {}

    def add(k, v):
        tm[k] = tm.get(k, 0.0) + v
    t0 = time.perf_counter()
    sch = Schedule(lanes)
    L = len(lanes)
    W = lanes.outs.shape[1]
    gate_slot, out_net = _out_slots(lanes, net_toggles)
    Wt = out_net.shape[1]
    t1 = time.perf_counter()
    add("stimulus", t1 - t0)
    kw = dict(sim_kw or {})
    if backend == "ref":
        sim = frs.Simulator(lanes.circuits(), backend="ref", watch=None, **kw)
    else:
        sim = frs.Simulator(lanes.fr_batch(), backend=backend, watch=None, threads=threads, net_observables=False,
                            **kw)
    sim.set_io(sch.n_in, out_net, high=True, toggles=net_toggles)
    n = sch.n_calls
    VALS = np.empty((n, L, Wt), np.uint8)
    HIGH = np.empty((n, L, Wt), np.int64)
    TOG = np.empty((n, L, Wt), np.int64) if net_toggles else None
    EV = np.empty((n, L), np.int64)
    t2 = time.perf_counter()
    add("create", t2 - t1)
    tl, tu, tn, tv = sch.trip
    ek = ew = 0.0
    launches = 0
    budget = lanes.extra.get("budget")
    budgeted = budget is not None and np.isfinite(budget).any()
    tmp_out = frs.StepResult(np.empty((L, Wt), np.uint8), np.empty((L, Wt), np.int64),
                             np.empty((L, Wt), np.int64) if net_toggles else None)
    stop_step = np.full(L, -1, np.int64)                         # the step after which a lane was stopped
    stop_call = np.full(L, n, np.int64)                          # the call during which it was stopped
    try:
        for rep in range(max(int(repeat), 1)):
            t_0 = 0
            if rep:
                t_0 = sim.t
                sim.reset()                                      # power-up state, shifted to the current time
            frozen = np.zeros(L, bool)
            frozen_in = np.zeros((L, sch.n_in), np.uint8)
            run_cum = np.zeros(L, np.int64)
            win_cum = np.zeros(L, np.int64)
            for j in range(n):
                a, b = sch.trip_off[j], sch.trip_off[j + 1]
                dense = sch.dense[sch.dense_of[j]] if sch.has_dense[j] else None
                if dense is not None and frozen.any():           # stopped lanes keep their inputs
                    dense = dense.copy()
                    dense[frozen] = frozen_in[frozen]
                st0, st1 = int(sch.bnd[j]), int(sch.bnd[j + 1])
                in_win = None
                if budgeted:
                    sidx = (st0 - 1) // lanes.K
                    in_win = (st0 >= 1) & (sidx < lanes.S)       # the call lies in step sidx's sampling window
                    opening = in_win & ((st0 - 1) % lanes.K == 0)
                    win_cum[opening] = run_cum[opening]
                cuts = [st1]
                if budgeted and st1 - st0 > 4 * BUDGET_CHUNK and not frozen.all():
                    # long calls in geometric chunks, so a lane is stopped soon after it crosses its budget
                    cuts = [st0 + BUDGET_CHUNK * 4**k for k in range(64) if st0 + BUDGET_CHUNK * 4**k < st1] + [st1]
                last_trip = int(tu[b - 1]) if b > a else -1
                ci = 0
                while ci < len(cuts):
                    T = cuts[ci]
                    first = ci == 0
                    o = frs.StepResult(VALS[j], HIGH[j], TOG[j] if TOG is not None else None) if first else tmp_out
                    if b > a:
                        # this chunk's triples only (2026-10-10 fix, physical-K E1 agent): queued stimulus survives
                        # load_state, so a lane frozen mid-call must not have later input changes queued; one would
                        # re-excite the stopped lane (e.g. a ring held in an inconsistent state) and run it unchecked
                        uu = tu[a:b]
                        m = (uu >= (st0 if first else cuts[ci - 1])) & (uu < T)
                        if frozen.any():
                            m &= ~frozen[tl[a:b]]
                        if m.all():
                            sim.add_stimulus_arrays(tl[a:b], uu + t_0, tn[a:b], tv[a:b])
                        elif m.any():
                            sim.add_stimulus_arrays(tl[a:b][m], uu[m] + t_0, tn[a:b][m], tv[a:b][m])
                    sim.step(T + t_0, dense if first else None, out=o)
                    launches += 1
                    e = sim.events()
                    if first:
                        EV[j] = e
                    else:
                        EV[j] += e
                        VALS[j] = tmp_out.values
                        HIGH[j] += tmp_out.high
                        if TOG is not None:
                            TOG[j] += tmp_out.toggles
                    if backend != "ref":
                        p = sim.profile()
                        ek += p["kernel_s"]
                        ew += p["windows"]
                    final = ci == len(cuts) - 1
                    if budgeted:
                        cur = run_cum + EV[j] - sch.in_calls[j]           # gate events so far (lower bound)
                        over = in_win & ~frozen & (cur - win_cum > budget)
                        if over.any():                           # freeze: no pending transitions, no input changes
                            stop_step[over] = sidx[over]
                            stop_call[over] = j
                            t_now, val, pend, cnt = sim.state()
                            gl = lanes.gate_lane()
                            pend = pend.copy()
                            pend[over[gl]] = -1
                            sim.load_state((t_now, val, pend, cnt), over)
                            for lb in np.flatnonzero(over):
                                frozen_in[lb, :int(lanes.n_inputs[lb])] = \
                                    val[lanes.net_off[lb]:lanes.net_off[lb] + lanes.n_inputs[lb]]
                            frozen |= over
                        if not final and not (e[~frozen] > 0).any() and last_trip < T:
                            cuts = cuts[:ci + 1] + [st1]         # quiet and no input ahead: the rest in one call
                        elif not final and BUDGET_FOLLOW:
                            nxt = _budget_cut(cur - win_cum, budget, in_win & ~frozen,
                                              T - (sidx * lanes.K + 1), T, st0)
                            if nxt is not None and nxt < cuts[ci + 1]:
                                cuts.insert(ci + 1, nxt)
                    ci += 1
                if budgeted:
                    run_cum += EV[j] - sch.in_calls[j]
        t3 = time.perf_counter()
    finally:
        t4 = time.perf_counter()
        sim.close()
        add("destroy", time.perf_counter() - t4)
    add("kernel", t3 - t2)
    add("engine_kernel", ek)
    add("engine_windows", ew)
    t5 = time.perf_counter()
    # per lane and step: calls [sK+1, (s+1)K+1)
    idx = lambda t: np.searchsorted(sch.bnd, t)  # noqa: E731
    Smax = int(lanes.S.max(initial=0))
    out = np.zeros((L, Smax, W), np.uint8)
    high = np.zeros((L, Smax, W), np.float32)
    if W:
        C = np.zeros((n + 1, L, W), np.int64)
        np.cumsum(HIGH[:, :, :W], axis=0, out=C[1:])
        for Kg, Sg in set(zip(lanes.K.tolist(), lanes.S.tolist())):   # lanes of one (K, S) share their calls
            g = np.flatnonzero((lanes.K == Kg) & (lanes.S == Sg))
            st = np.arange(Sg, dtype=np.int64)
            j0, j1 = idx(st * Kg + 1), idx((st + 1) * Kg + 1)
            high[g, :Sg] = np.transpose((C[j1][:, g] - C[j0][:, g]) / Kg, (1, 0, 2)).astype(np.float32)
            out[g, :Sg] = np.transpose(VALS[j1 - 1][:, g, :W], (1, 0, 2))
        _fill_wired(lanes, gate_slot, out, high)
    endc = idx(sch.end)
    in_calls = sch.in_calls
    if budgeted and (stop_call < n).any():                          # stopped lanes get no inputs after the stop
        in_calls = np.where(np.arange(n)[:, None] > stop_call[None, :], 0, in_calls)
    EVg = EV - in_calls                                             # gate-net transitions per call and lane
    cum = np.concatenate([np.zeros((1, L), np.int64), np.cumsum(EVg, 0)])
    tog = cum[endc, np.arange(L)]
    events = np.zeros((L, Smax), np.int64)
    for Kg, Sg in set(zip(lanes.K.tolist(), lanes.S.tolist())):
        g = np.flatnonzero((lanes.K == Kg) & (lanes.S == Sg))
        st = np.arange(Sg, dtype=np.int64)
        j0, j1 = idx(st * Kg + 1), idx((st + 1) * Kg + 1)
        events[g, :Sg] = (cum[j1][:, g] - cum[j0][:, g]).T
    net_tog = None
    if net_toggles:
        Gm = out_net.shape[1] - W
        T = np.concatenate([np.zeros((1, L, Gm), np.int64), np.cumsum(TOG[:, :, W:W + Gm], 0)])
        per = T[endc, np.arange(L)]                                  # [L, Gm]
        gl = lanes.gate_lane()
        gi = np.arange(int(lanes.gate_off[-1])) - lanes.gate_off[gl]
        net_tog = per[gl, gi]
    over_budget = None
    if budgeted:
        over_budget = stop_step >= 0
        for lb in np.flatnonzero(over_budget):                   # later steps: outputs 0, high 0, events 0
            s0 = int(stop_step[lb]) + 1
            out[lb, s0:] = 0
            high[lb, s0:] = 0
            events[lb, s0:] = 0
        tog = events.sum(1)
    add("post", time.perf_counter() - t5)
    add("calls", 1)
    add("steps", n)
    add("launches", launches)
    add("lanes", L)
    add("ticks", float((lanes.S * lanes.K + 1).sum()))
    add("events", float(EV.sum()))
    return LaneResult(out, high, tog.astype(np.float64), net_tog, n, events, over_budget)
