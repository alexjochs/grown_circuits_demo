"""Reference event-stream substrate on freerun's public API (docs/grown_time.md): physical K, slowly, on the CPU.

This is the contract of docs/grown_time.md implemented in the plainest way, so the pipeline runs end to end at
physical K (10^5 to 10^8 ticks per control step) today, and so the production adapter (sim/freerun_backend.py,
owned separately) has an executable reference to agree with. It is not fast: one freerun call per control step.

How a request runs (EventLaneBatch: nets, lane_circ, streams, lane_stream, K, S, budget):
  circuits   lane l's flattened netlist as a freerun.Circuit; the primary inputs' initial values are the stream's
             t = 0 events (the power-up input values; wires without one start at 0)
  stimulus   every other event (t, wire, v) of the lane's stream is a freerun stimulus triple, added step by step
             (only the next step's events are queued, so freerun's per-run sort stays small)
  stepping   one `step(T)` per control step with no dense inputs: T = 1 first (time 0, power-up), then
             T = (s+1)K + 1 for step s, so the step's window is ticks sK+1 .. (s+1)K, exactly base.py's (sK, (s+1)K]
             (plus a probe cut inside the step at large K, below)
  outputs    `set_io(outputs=gate output nets, high=True)`: the value at T- (after the commits at (s+1)K) and the
             ticks at 1 in the window; high = ticks / K. No trace is recorded (watch=None, net observables off).
             Outputs wired to a primary input come from the streams (sim.base.wired_outputs, the stand-in's
             convention)
  events     Simulator.events() counts every net change of the call, primary inputs included; the stimulus added
             for the same call (all real changes) is subtracted, so EvalResult.events [L, S] is gate-net transitions
             only
  budget     when the request carries one, a lane whose events in a step exceed it is flagged over_budget and
             stopped: its pending transitions are cleared (load_state with the lane's pending times set to none) and
             its later stimulus is dropped, so it is quiescent and costs nothing for the rest of the episode. Its
             later steps report outputs 0, high 0, events 0. A lane under budget is never touched.
  probe      for K >= probe_min_K (2^20) the budget is also checked after the first K / probe (1/8) of every step
             (two engine calls per step): a lane whose partial count already exceeds the step's budget is over
             whatever the rest of the step does. It is stopped there; that step reports its partial events and
             outputs 0, high 0. At 100 us steps this saves 7/8 of an oscillator's first step (~2 * 10^6 events).

Engine time: freerun ticks are our ticks (one freerun tick = TimeModel.tick seconds); the absolute scale only
matters through K and the delays the circuits carry.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from .base import EvalResult, EventLaneBatch, events_from_dense, wired_outputs
from .freerun_backend import import_freerun, to_freerun


class EventSubstrate:
    name = "freerun_events"
    threadsafe = False

    def __init__(self, backend: str = "cpu", threads: int = 1, home: str | None = None, libdir: str | None = None,
                 early_stop: bool = True, probe: int = 8, probe_min_K: int = 1 << 20):
        """backend: freerun backend ("cpu", "cuda", "ref"); threads: CPU backend threads (0: all cores);
        early_stop: honour the request's budget by stopping lanes over it (see module doc); probe, probe_min_K:
        for K >= probe_min_K, check the budget also after the first K // probe ticks of every step (0: never)."""
        if backend not in ("cpu", "cuda", "ref"):
            raise ValueError(f"unknown freerun backend {backend!r}")
        self.backend, self.threads, self.early_stop = backend, int(threads), bool(early_stop)
        self.probe, self.probe_min_K = int(probe), int(probe_min_K)
        self.timing: dict = defaultdict(float)
        if home or libdir:
            import_freerun(home, libdir)

    @classmethod
    def from_config(cls, d: dict) -> "EventSubstrate":
        return cls(d.get("backend", "cpu"), int(d.get("threads", 1) or 0), d.get("home"), d.get("libdir"),
                   bool(d.get("early_stop", True)), int(d.get("probe", 8)), int(d.get("probe_min_K", 1 << 20)))

    # -- protocol ----------------------------------------------------------------------------------------------
    def evaluate_requests(self, requests) -> list:
        out = []
        for r in requests:
            if isinstance(r, EventLaneBatch):
                out.append(self.evaluate_events(r.nets, r.lane_circ, r.streams, r.lane_stream, r.K, r.S, r.budget))
            else:                                   # a dense LaneBatch: the same waveform as events
                x = r.streams
                out.append(self.evaluate_events(r.nets, r.lane_circ, events_from_dense(x, r.K, keep=False),
                                                r.lane_stream, r.K, x.shape[1]))
        return out

    def evaluate_events(self, nets, lane_circ, streams, lane_stream, K: int, S: int,
                        budget: float | None = None) -> EvalResult:
        fr = import_freerun()
        tm = self.timing
        t0 = time.perf_counter()
        K, S = int(K), int(S)
        lane_circ = np.asarray(lane_circ, np.int64)
        lane_stream = np.asarray(lane_stream, np.int64)
        L = len(lane_circ)
        I, n_out = int(nets.n_inputs), int(nets.n_out)
        netlists = nets.netlists()
        outs = np.asarray(nets.outs, np.int64)[lane_circ]                       # [L, n_out]
        gate_slot = outs >= I
        n_gates = np.asarray(nets.n_gates, np.int64)[lane_circ]

        # power-up input values (the t = 0 events) per stream, and every later real change, per lane
        x0 = streams.initial(I)
        se, st, sw, sv = streams.effective(I)                                  # sorted by (stream, t, wire)
        cnt = np.bincount(se, minlength=len(streams))
        soff = np.concatenate([[0], np.cumsum(cnt)])
        rows = np.concatenate([np.arange(soff[e], soff[e + 1]) for e in lane_stream]) if L else \
            np.zeros(0, np.int64)
        lane_of = np.repeat(np.arange(L, dtype=np.int64), cnt[lane_stream])
        o = np.argsort(st[rows], kind="stable")
        ev_b, ev_t = lane_of[o], st[rows][o]
        ev_n, ev_v = sw[rows][o].astype(np.int64), sv[rows][o].astype(np.int64)

        circuits = [to_freerun(netlists[c], x0[e]) for c, e in zip(lane_circ.tolist(), lane_stream.tolist())]
        tm["convert"] += time.perf_counter() - t0

        outputs = np.zeros((L, S, n_out), np.uint8)
        high = np.zeros((L, S, n_out), np.float32)
        events = np.zeros((L, S), np.int64)
        over = np.zeros(L, bool)
        alive = np.ones(L, bool)
        stop = np.full(L, S, np.int64)                                          # first step not simulated
        if L:
            t1 = time.perf_counter()
            sim = fr.sim.Simulator(circuits, backend=self.backend, watch=None, threads=self.threads,
                                   net_observables=False)
            out_net = np.where(gate_slot, outs, -1).astype(np.int32)
            sim.set_io(n_inputs=0, outputs=out_net, high=True)
            gate_off = np.asarray(sim.batch.gate_off, np.int64)
            tm["create"] += time.perf_counter() - t1
            try:
                t2 = time.perf_counter()
                sim.step(1)                                                     # time 0: power-up
                p = 0
                probe = budget is not None and self.early_stop and self.probe and K >= self.probe_min_K
                for s in range(S):
                    T_end = (s + 1) * K + 1
                    cuts = [s * K + 1 + K // self.probe, T_end] if probe else [T_end]
                    acc = np.zeros(L, np.int64)
                    acc_high = np.zeros((L, n_out), np.int64)
                    for T in cuts:
                        q = int(np.searchsorted(ev_t, T, side="left"))
                        n_in_ev = np.zeros(L, np.int64)
                        if q > p:
                            m = alive[ev_b[p:q]]
                            if m.any():
                                sim.add_stimulus_arrays(ev_b[p:q][m], ev_t[p:q][m], ev_n[p:q][m], ev_v[p:q][m])
                                n_in_ev = np.bincount(ev_b[p:q][m], minlength=L)
                            p = q
                        r = sim.step(T)
                        acc += sim.events() - n_in_ev                           # gate transitions only
                        acc_high += r.high
                        if T == T_end:                                          # the step ran in full
                            a = alive
                            outputs[a, s] = r.values[a]
                            high[a, s] = acc_high[a] / K
                        if budget is not None and self.early_stop:
                            newly = alive & (acc > budget)
                            if newly.any():
                                events[newly, s] = acc[newly]
                                over |= newly
                                stop[newly] = s + 1 if T == T_end else s
                                alive = alive & ~newly
                                self._quiesce(sim, newly, gate_off)
                        if not alive.any():
                            break
                    events[alive, s] = acc[alive]
                    if not alive.any():
                        break
                tm["kernel"] += time.perf_counter() - t2
            finally:
                sim.close()
        wv, wh, wmask = wired_outputs(streams, lane_stream, outs, I, K, S)
        if wmask.any():
            ll, jj = np.nonzero(wmask)
            outputs[ll, :, jj] = wv[ll, :, jj]
            high[ll, :, jj] = wh[ll, :, jj]
            for l in np.flatnonzero(stop < S):                                   # stopped lanes report 0 afterwards
                outputs[l, stop[l]:] = 0
                high[l, stop[l]:] = 0
        tm["calls"] += 1
        tm["lanes"] += L
        tm["steps"] += L * S
        tm["events"] += float(events.sum())
        tm["seconds"] += time.perf_counter() - t0
        return EvalResult(outputs, high, events.sum(1).astype(np.float64), n_gates, K, events=events,
                          over_budget=over if budget is not None and self.early_stop else None)

    @staticmethod
    def _quiesce(sim, lanes, gate_off) -> None:
        """Clear every pending transition of the selected lanes: with no stimulus left they never change again."""
        t, val, pend, cnt = sim.state()
        pend = np.array(pend, np.int64)
        for b in np.flatnonzero(lanes):
            pend[gate_off[b]:gate_off[b + 1]] = -1
        sim.load_state((t, val, pend, cnt), mask=np.asarray(lanes, bool))
