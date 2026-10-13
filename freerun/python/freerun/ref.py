"""Reference simulator: the executable form of docs/SPEC.md.

Written for clarity, not speed. Every other backend is tested for bit-exact agreement with this one, so any change
here is a change to the semantics and must be reflected in SPEC.md first.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from . import jitter as _jit
from .netlist import Circuit

NEVER = None  # "no pending transition" (SPEC: p_g = infinity)


@dataclass
class Window:
    """Observables over a run window [t0, t1) (SPEC §6)."""
    t0: int
    t1: int
    trace: np.ndarray    # int64[M, 3] rows (t, net, value), ordered by (t, net); only watched nets
    toggles: np.ndarray  # int64[N] number of changes per net
    high: np.ndarray     # int64[N] ticks spent at 1 per net
    ties: int            # number of (gate, time) pairs with >= 2 causes (SPEC §8)
    first_tie: int       # time of the first tie in the window, -1 if none
    events: int          # number of net changes (all nets)
    cancels: int         # evaluations that cancelled a pending transition (rejected pulses)
    advances: int        # evaluations that moved a pending transition earlier (earliest arc wins)
    values: np.ndarray | None = None  # uint8[N] net values at t1- (with toggles, they give the values at t0)

    @property
    def initial(self) -> np.ndarray:
        """Net values at the start of the window: final values xor toggle parity."""
        return (self.values ^ (self.toggles & 1)).astype(np.uint8)


class RefSim:
    """Simulates one circuit. Construct, add stimulus, call run(T) repeatedly."""

    def __init__(self, circuit: Circuit, watch=None):
        self.c = circuit
        I, G, N = circuit.n_inputs, circuit.n_gates, circuit.n_nets
        self.val = [int(x) for x in circuit.init]
        self.pend = [NEVER] * G
        self.heap: list[tuple[int, int]] = []  # (time, gate); stale entries are skipped
        self.stim: dict[int, dict[int, int]] = {}  # time -> {input net: value}
        self.stim_times: list[int] = []  # heap of times present in self.stim
        self.t = 0  # every time < self.t has been processed
        self.watch = None if watch is None else set(int(n) for n in watch)

        self.fanin = [[int(n) for n in circuit.fanin[circuit.pins(g)]] for g in range(G)]
        self.dr = [[int(d) for d in circuit.dr[circuit.pins(g)]] for g in range(G)]
        self.df = [[int(d) for d in circuit.df[circuit.pins(g)]] for g in range(G)]
        self.tt = [int(x) for x in circuit.tt]
        self.fanout: list[list[tuple[int, int]]] = [[] for _ in range(N)]
        for g in range(G):
            for pin, n in enumerate(self.fanin[g]):
                self.fanout[n].append((g, pin))
        # Jitter (SPEC §10): parameters and the per-gate transition counters n_g (kept only with jitter).
        self.jit = circuit.jitter
        self.amp = [int(a) for a in self.jit.amp] if self.jit is not None else None
        self.n = [0] * G

        # Power-up (SPEC §5): evaluate every gate of arity >= 1 at time 0 with P = all pins.
        for g in range(G):
            if self.fanin[g]:
                self._evaluate(g, 0, range(len(self.fanin[g])))

    # --- SPEC §3 ---------------------------------------------------------------------------------------------
    def _f(self, g: int) -> int:
        idx = 0
        for i, n in enumerate(self.fanin[g]):
            idx |= self.val[n] << i
        return (self.tt[g] >> idx) & 1

    def _evaluate(self, g: int, t: int, pins) -> str:
        """Returns what happened, for statistics: none, cancel, schedule, keep or advance."""
        y = self.val[self.c.n_inputs + g]
        v = self._f(g)
        if v == y:
            had = self.pend[g] is not NEVER
            self.pend[g] = NEVER
            return "cancel" if had else "none"
        delays = self.dr[g] if v == 1 else self.df[g]
        tn = t + self._delay(g, min(delays[i] for i in pins), self.pend[g] is NEVER)
        if self.pend[g] is NEVER or tn < self.pend[g]:
            what = "schedule" if self.pend[g] is NEVER else "advance"
            self.pend[g] = tn
            heapq.heappush(self.heap, (tn, g))
            return what
        return "keep"

    def _delay(self, g: int, d: int, create: bool) -> int:
        """Effective delay of a transition scheduled by gate g with base delay d (SPEC §10.4). create: p_g was
        infinity, so this evaluation creates transition n_g; otherwise it re-evaluates the pending one, n_g - 1."""
        if self.jit is None:
            return d
        if create:
            k = self.n[g]
            self.n[g] += 1
        else:
            k = self.n[g] - 1
        j = _jit.offset(self.amp[g], self.jit.seed, self.jit.cid, g, k)
        return _jit.effective(d, j, self.jit.dmin)

    # --- SPEC §4 ---------------------------------------------------------------------------------------------
    def add_stimulus(self, events) -> None:
        """events: iterable of (t, input_net, value). Times must be >= the current time."""
        for t, n, v in events:
            t, n, v = int(t), int(n), int(v)
            if not 0 <= n < self.c.n_inputs:
                raise ValueError(f"net {n} is not a primary input")
            if v not in (0, 1):
                raise ValueError("stimulus values must be 0 or 1")
            if t < self.t:
                raise ValueError(f"stimulus at {t} is before the current time {self.t}")
            slot = self.stim.setdefault(t, {})
            if n in slot:
                raise ValueError(f"two stimulus events for net {n} at time {t}")
            if len(slot) == 0:
                heapq.heappush(self.stim_times, t)
            slot[n] = v

    # --- SPEC §5 ---------------------------------------------------------------------------------------------
    def _next_time(self):
        while self.heap and self.pend[self.heap[0][1]] != self.heap[0][0]:
            heapq.heappop(self.heap)  # stale: cancelled or moved earlier
        cands = []
        if self.heap:
            cands.append(self.heap[0][0])
        if self.stim_times:
            cands.append(self.stim_times[0])
        return min(cands) if cands else None

    def _process(self, t: int, obs) -> int:
        """Process time t; returns the number of committed gate transitions."""
        I = self.c.n_inputs
        # Phase A: commit gate transitions due at t and stimulus at t.
        committed = []
        while self.heap and self.heap[0][0] == t:
            _, g = heapq.heappop(self.heap)
            if self.pend[g] == t:
                self.pend[g] = NEVER
                committed.append(g)
        changed = []
        for g in committed:
            changed.append(I + g)
        if self.stim_times and self.stim_times[0] == t:
            heapq.heappop(self.stim_times)
            for n, v in self.stim.pop(t).items():
                if self.val[n] != v:
                    changed.append(n)
        for n in changed:
            obs.change(t, n, self.val[n])
            self.val[n] ^= 1
        # Phase B: evaluate every gate with a pin on a changed net, against post-commit values.
        pins_of: dict[int, list[int]] = {}
        nets_of: dict[int, set[int]] = {}
        for n in changed:
            for g, pin in self.fanout[n]:
                pins_of.setdefault(g, []).append(pin)
                nets_of.setdefault(g, set()).add(n)
        committed_set = set(committed)
        for g, pins in pins_of.items():
            causes = nets_of[g] | ({I + g} if g in committed_set else set())  # a self-loop is one cause
            if len(causes) >= 2:
                if obs.ties == 0:
                    obs.first_tie = t
                obs.ties += 1
            what = self._evaluate(g, t, pins)
            if what == "cancel":
                obs.cancels += 1
            elif what == "advance":
                obs.advances += 1
        return len(committed)

    def run(self, T: int) -> Window:
        """Process every time t < T (SPEC §6) and return the observables of the window [self.t, T)."""
        T = int(T)
        if T < self.t:
            raise ValueError(f"cannot run backwards from {self.t} to {T}")
        obs = _Observer(self.t, self.val, self.watch)
        while True:
            t = self._next_time()
            if t is None or t >= T:
                break
            self._process(t, obs)
        self.t = T
        return obs.finish(T, self.val)

    def run_steps(self, T: int, w0: int, K: int, S: int, outputs, budget: int = -1, hold: bool = False):
        """A multi-step run (SPEC §13): process every time t < T as run(T) does, and observe S steps, step s being the
        window [w0 + s K, w0 + (s+1) K) (inside [self.t, T)). outputs: net per output slot (-1 = unused).
        budget >= 0: once the committed gate-net transitions of a step exceed it after processing some time t, the
        circuit is frozen (pending transitions dropped, its stimulus before T discarded), so it holds its state;
        hold: freeze it the same way at the end of step S - 1.
        Returns (Window over [self.t, T), values uint8 [S, n_out], high int64 [S, n_out], toggles int64 [S, n_out],
        events int64 [S], stop) with stop the step at which the budget froze the circuit, -1 if none; steps after
        it report 0."""
        T, w0, K, S = int(T), int(w0), int(K), int(S)
        if T < self.t or w0 < self.t or K < 1 or S < 0 or w0 + S * K > T:
            raise ValueError("run_steps: need self.t <= w0, K >= 1, S >= 0 and w0 + S K <= T")
        outs = [int(n) for n in outputs]
        n_out = len(outs)
        obs = _Observer(self.t, self.val, self.watch)
        vals = np.zeros((S, n_out), np.uint8)
        high = np.zeros((S, n_out), np.int64)
        tog = np.zeros((S, n_out), np.int64)
        ev = np.zeros(S, np.int64)
        prev = np.zeros((n_out, 2), np.int64)
        state = dict(gev=0)

        def emit(k, zero=False):  # boundary k at w0 + k K: closes step k - 1, opens step k
            tb = w0 + k * K
            for j, n in enumerate(outs):
                v = h = c = 0
                if n >= 0:
                    v = self.val[n]
                    h = int(obs.high[n]) + (tb - int(obs.seg[n]) if v else 0)
                    c = int(obs.toggles[n])
                if k >= 1 and not zero:
                    vals[k - 1, j], high[k - 1, j], tog[k - 1, j] = v, h - prev[j, 0], c - prev[j, 1]
                prev[j] = h, c
            if k >= 1 and not zero:
                ev[k - 1] = state["gev"]
            state["gev"] = 0

        kb, stop, frozen = 0, -1, False
        while True:
            t = self._next_time()
            lim = T if t is None else min(t, T)
            while kb <= S and w0 + kb * K <= lim:
                emit(kb)
                kb += 1
                if kb > S and hold:
                    frozen = True
            if frozen or t is None or t >= T:
                break
            n_gate = self._process(t, obs)
            if 1 <= kb <= S:                                 # t lies in step kb - 1
                state["gev"] += n_gate
                if budget >= 0 and state["gev"] > budget:
                    stop, frozen = kb - 1, True
                    break
        if frozen:
            self.pend = [NEVER] * len(self.pend)
            self.heap = []
            for u in [u for u in self.stim if u < T]:
                del self.stim[u]
            self.stim_times = [u for u in self.stim_times if u >= T]
            heapq.heapify(self.stim_times)
        while kb <= S:
            emit(kb, zero=stop >= 0 and kb > stop + 1)
            kb += 1
        self.t = T
        return obs.finish(T, self.val), vals, high, tog, ev, stop

    # --- state -------------------------------------------------------------------------------------------------
    def values(self) -> np.ndarray:
        return np.array(self.val, dtype=np.uint8)

    def pending(self) -> np.ndarray:
        """Pending transition time per gate, -1 for none."""
        return np.array([-1 if p is NEVER else p for p in self.pend], dtype=np.int64)

    def counters(self) -> np.ndarray:
        """Transition counters n_g (SPEC §10.2); all 0 for a circuit without jitter."""
        return np.array(self.n, dtype=np.uint64)

    def load_state(self, t_ref: int, values, pending, counters=None) -> None:
        """Load a state (net values, pending times with -1 for none, transition counters; None = all 0) taken at time
        t_ref, translated to the current time (SPEC §6): a pending time p becomes p - t_ref + self.t. Queued stimulus
        is kept."""
        shift = self.t - int(t_ref)
        self.val = [int(x) for x in values]
        self.pend = [NEVER if p < 0 else int(p) + shift for p in pending]
        self.heap = [(p, g) for g, p in enumerate(self.pend) if p is not NEVER]
        heapq.heapify(self.heap)
        self.n = [0] * len(self.pend) if counters is None else [int(k) for k in counters]


class _Observer:
    def __init__(self, t0, val, watch):
        self.t0, self.watch = t0, watch
        N = len(val)
        self.trace: list[tuple[int, int, int]] = []
        self.toggles = np.zeros(N, dtype=np.int64)
        self.high = np.zeros(N, dtype=np.int64)
        self.seg = np.full(N, t0, dtype=np.int64)  # start of the current constant segment within the window
        self.ties = 0
        self.first_tie = -1
        self.events = 0
        self.cancels = 0
        self.advances = 0

    def change(self, t, n, old):
        if old == 1:
            self.high[n] += t - self.seg[n]
        self.seg[n] = t
        self.toggles[n] += 1
        self.events += 1
        if self.watch is None or n in self.watch:
            self.trace.append((t, n, 1 - old))

    def finish(self, T, val) -> Window:
        v = np.array(val, dtype=np.int64)
        self.high += v * (T - self.seg)
        tr = np.array(sorted(self.trace), dtype=np.int64).reshape(-1, 3)
        return Window(self.t0, T, tr, self.toggles, self.high, self.ties, self.first_tie, self.events,
                      self.cancels, self.advances, v.astype(np.uint8))


def simulate(circuit: Circuit, T: int, stimulus=(), watch=None) -> Window:
    """Convenience: one window [0, T)."""
    s = RefSim(circuit, watch)
    s.add_stimulus(stimulus)
    return s.run(T)
