"""Stand-in synchronous backend: unit-delay LUT semantics (adapted from evocirc.rollout), in numpy.

Every gate (GATE or DELAY token, after flattening) updates every tick from the previous tick's net values:
y_g(t+1) = tt_g[sum_i x_i(t) << i]. Per-arc delay values are ignored (every arc is one tick). Arbitrary feedback
is allowed. Circuits of different sizes are padded into one dense batch; padding pins read a constant-0 net.

Two kernels with identical results: a compiled C kernel (_csync.py, built on first use with the system compiler;
lanes run one after another over their own gate counts) and a numpy kernel used when no compiler is available or
kernel="numpy". numpy: all lanes' nets live in one flat uint8 array; each tick is one gather of the pin nets, a row
index built from at most A (the batch's largest arity) bit planes, and one gather from the lanes' expanded truth
tables. Rows a gate cannot address (pins >= its arity read the constant-0 net) are never read.
"""
from __future__ import annotations

import numpy as np

from . import _csync
from .base import STANDIN_MAX_K, EvalResult, events_to_dense, tick_inputs

MAX_ARITY = 6


class SyncSubstrate:
    """Event-stream requests (EventLaneBatch, docs/grown_time.md; sim.base.evaluate_requests calls
    evaluate_events) are accepted for K <= max_K (default STANDIN_MAX_K = 4096): they are expanded to per-tick inputs,
    which is all a unit-delay tick simulator can do. Above that it raises: physical K (10^5 and up) needs an
    event-driven substrate. Results carry per-step events."""
    name = "sync"
    threadsafe = True            # no state between calls; the compiled kernel releases the GIL

    def __init__(self, kernel: str = "auto", max_K: int = STANDIN_MAX_K):
        if kernel not in ("auto", "c", "numpy"):
            raise ValueError(f"unknown kernel {kernel!r}")
        self.kernel = kernel
        self.max_K = int(max_K)
        if kernel == "c" and _csync.load() is None:
            raise RuntimeError("compiled sync kernel unavailable (no C compiler?)")

    def _check_K(self, K: int) -> None:
        if K > self.max_K:
            raise ValueError(f"SyncSubstrate: K = {K} ticks per control step is above the stand-in's limit "
                             f"{self.max_K}. The stand-in steps every tick of every lane, so it only runs legacy and "
                             "small-K configs; physical control steps (K = dt / tick, 10^5 and up) need an event-driven "
                             "substrate (freerun; docs/grown_time.md)")

    def evaluate_events(self, nets, lane_circ, streams, lane_stream, K: int, S: int, budget=None) -> EvalResult:
        """An EventLaneBatch's lanes (sim.base.evaluate_requests calls this): the streams expanded to per-tick inputs
        (K <= max_K). `budget` is ignored (whole lanes are simulated); the search flags lanes over budget from the
        per-step events."""
        self._check_K(K)
        x = events_to_dense(streams, S, K, nets.n_inputs)
        lanes = x[np.asarray(lane_stream, np.int64)]
        per_tick = lanes.shape[2] > 1
        return self.evaluate_batch(nets, lane_circ, lanes if per_tick else lanes[:, :, 0], K, per_tick=per_tick)

    def evaluate(self, circuits, inputs, K: int, library=None, per_tick: bool = False) -> EvalResult:
        from ..circuit.netbatch import flatten_batch
        return self.evaluate_batch(flatten_batch(circuits, library), None, inputs, K, per_tick=per_tick)

    def evaluate_batch(self, nb, lane_circ, inputs, K: int, per_tick: bool = False) -> EvalResult:
        """Evaluate a NetBatch (circuit.netbatch): lane b runs circuit lane_circ[b] (None: one lane per circuit)
        on inputs[b]. The compiled kernel reads the batch in place; the numpy kernel gets per-lane netlists."""
        L = len(nb) if lane_circ is None else len(lane_circ)
        if self.kernel == "numpy" or _csync.load() is None:
            nets = nb.netlists()
            return self.evaluate_netlists(nets if lane_circ is None else [nets[c] for c in np.asarray(lane_circ)],
                                          inputs, K, per_tick=per_tick)
        self._check_K(K)
        x = tick_inputs(inputs, L, K, per_tick)
        if x.shape[3] != nb.n_inputs:
            raise ValueError(f"inputs have {x.shape[3]} bits, circuits take {nb.n_inputs}")
        out, high, tog, ev = _csync.run_batch(nb, lane_circ, x, K)
        G = nb.n_gates if lane_circ is None else nb.n_gates[np.asarray(lane_circ)]
        return EvalResult(out, high, tog, G.astype(np.int64), K, events=ev)

    def evaluate_netlists(self, nets, inputs, K: int, per_tick: bool = False) -> EvalResult:
        self._check_K(K)
        B = len(nets)
        x = tick_inputs(inputs, B, K, per_tick)                    # [B, S, Kx, I]
        S, Kx = x.shape[1], x.shape[2]
        I = nets[0].n_inputs
        n_out = len(nets[0].outputs)
        if any(n.n_inputs != I or len(n.outputs) != n_out for n in nets):
            raise ValueError("all circuits in a batch need the same number of inputs and outputs")
        if x.shape[3] != I:
            raise ValueError(f"inputs have {x.shape[3]} bits, circuits take {I}")
        if self.kernel != "numpy" and _csync.load() is not None:
            out, high, tog, ev = _csync.run(nets, x, K)
            return EvalResult(out, high, tog, np.array([n.n_gates for n in nets], np.int64), K, events=ev)
        G = max((n.n_gates for n in nets), default=0)
        A = max([int(n.arity.max()) for n in nets if n.n_gates] + [1])
        if A > MAX_ARITY:
            raise ValueError(f"arity {A} > {MAX_ARITY}")
        N = I + G + 1                                               # nets per lane; the last is constant 0
        ZERO = I + G
        fan = np.full((B, G, A), ZERO, np.intp)
        ttb = np.zeros((B, G, 1 << A), np.uint8)
        vals = np.zeros((B, N), np.uint8)
        outs = np.zeros((B, n_out), np.intp)
        rows = np.arange(1 << A, dtype=np.uint64)
        for b, n in enumerate(nets):
            g = n.n_gates
            if g:
                k = n.arity
                pin = np.arange(len(n.fanin)) - np.repeat(n.fanin_ptr[:-1], k)
                fan[b, np.repeat(np.arange(g), k), pin] = n.fanin
                ttb[b, :g] = (n.tt[:, None] >> rows[None, :]) & np.uint64(1)
                vals[b, I:I + g] = n.init[I:]
            outs[b] = n.outputs
        lane = np.arange(B, dtype=np.intp) * N
        fanf = fan + lane[:, None, None]
        outf = outs + lane[:, None]
        base = (np.arange(B * G, dtype=np.intp) << A).reshape(B, G)
        ttf = ttb.ravel()
        flat = vals.reshape(-1)
        gates = vals[:, I:I + G]                                    # view
        ins = vals[:, :I]                                           # view
        out = np.zeros((B, S, n_out), np.uint8)
        high = np.zeros((B, S, n_out), np.float32)
        tog = np.zeros((B, G), np.int64)
        ev = np.zeros((B, S), np.int64)
        h = np.zeros((B, n_out), np.int64)
        for s in range(S):
            h[:] = 0
            tog0 = tog.sum(1)
            for t in range(K):
                if t < Kx:
                    ins[:] = x[:, s, t]
                xs = flat[fanf]                                     # [B, G, A]
                idx = base + xs[..., 0]
                for a in range(1, A):
                    idx += xs[..., a].astype(np.intp) << a
                new = ttf[idx]
                tog += new ^ gates
                gates[:] = new
                h += flat[outf]
            out[:, s] = flat[outf]
            high[:, s] = h / K
            ev[:, s] = tog.sum(1) - tog0
        return EvalResult(out, high, tog.sum(1).astype(np.float64), np.array([n.n_gates for n in nets], np.int64), K,
                          events=ev)
