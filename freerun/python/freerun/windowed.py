"""Windowed (lookahead) formulation of the SPEC semantics, the algorithm behind the parallel backends.

Per-timestamp stepping exposes almost no parallelism within one circuit when delays are fine-grained (events rarely
share a timestamp). This module processes all events in a lookahead window W = [t0, t1), t1 <= t0 + L, where L is the
circuit's minimum arc delay, in a few data-parallel passes. It mirrors the CUDA kernel (csrc/cuda_window.cuh) pass by
pass and is tested to be bit-exact with ref.py.

Why it is exact (each fact follows from SPEC §3-§5 and delays >= L):

  F1. An evaluation at tau in W schedules or advances to tau + d >= t0 + L: never into W.
  F2. Hence the only transitions that can commit in W are those pending in W when it starts ("candidates"), and
      each gate commits at most once in W.
  F3. A re-evaluation at tau < p of a gate pending at p in W either cancels (function == output) or keeps p
      (tau + d > p). Once cancelled, a gate cannot commit again within W (by F1).
  So candidate g commits iff no evaluation at a time tau in [t0, p_g) sees f_g == y_g. Those evaluations depend
  only on input changes before p_g, i.e. on decisions about candidates with earlier times: the dependency order
  is strict in time. A fixed-point iteration over all candidates therefore converges: after pass k every decision
  among the k earliest candidate times is final.

  Every window narrower than L is exact too. Windows are cut short before a stimulus timestamp that would change an
  input a second time, so every net changes at most once per window: a net's change is one time per net (plus, for a
  gate output, its candidate's commit decision), and a gate finds its input events by reading its own fanin.

  Once commits are decided, each gate's events in W (input changes and its own commit) are replayed in time order
  to obtain its pending state at the end of W, exactly as the sequential simulator would.

Passes (each is data-parallel over candidates, changed nets or gates):
  1. stimulus changes in W (cutting W if needed), then candidates K (pending in W)
  2. touched gates: K plus the fanout of every net that may change; which candidates have dependents
  3. commit decisions, each pass from the previous pass's, until no decision that another candidate reads changes
  4. replay every touched gate: its pending state at the end of W, statistics
  5. apply the net changes (values, toggles, high time, trace)
"""
from __future__ import annotations

import heapq

from .jitter import lookahead as lookahead_of
from .netlist import Circuit
from .ref import NEVER, RefSim, Window, _Observer


class WindowedSim(RefSim):
    """Same interface and results as RefSim; processes lookahead windows instead of single timestamps."""

    def __init__(self, circuit: Circuit, watch=None, lookahead: int | None = None):
        super().__init__(circuit, watch)
        E = len(circuit.fanin)
        L = lookahead_of(circuit) if E else 1  # min effective delay, including jitter (SPEC §10.6)
        self.L = L if lookahead is None else int(lookahead)
        if self.L < 1 or (E and self.L > L):
            raise ValueError(f"lookahead must be in [1, min effective delay = {L}]")
        self.passes = 0   # statistics: fixed-point passes
        self.windows = 0

    def run(self, T: int) -> Window:
        T = int(T)
        if T < self.t:
            raise ValueError(f"cannot run backwards from {self.t} to {T}")
        obs = _Observer(self.t, self.val, self.watch)
        while True:
            t0 = self._next_time()
            if t0 is None or t0 >= T:
                break
            self._window(t0, min(t0 + self.L, T), obs)
        self.t = T
        return obs.finish(T, self.val)

    def _window(self, t0: int, t1: int, obs) -> None:
        I = self.c.n_inputs
        self.windows += 1
        wt = {}  # net -> time of its (only) change in this window

        # --- pass 1: stimulus changes, cutting the window before a timestamp that drives a net changed earlier -----
        stim = []
        while self.stim_times and self.stim_times[0] < t1:
            t = self.stim_times[0]
            if any(n in wt for n in self.stim[t]):
                t1 = t
                break
            heapq.heappop(self.stim_times)
            for n, v in sorted(self.stim.pop(t).items()):
                if v != self.val[n]:
                    wt[n] = t
                    stim.append(n)
        cand = [g for g, p in enumerate(self.pend) if p is not NEVER and t0 <= p < t1]
        for g in cand:
            wt[I + g] = self.pend[g]
        while self.heap and self.heap[0][0] < t1:  # entries in W are now owned by this window
            heapq.heappop(self.heap)

        # --- pass 2: touched gates, and which candidates have dependents (later candidates in their fanout) --------
        touched = list(cand)
        seen = set(cand)
        dependents = {}
        for n in [I + g for g in cand] + stim:
            for g2, _ in self.fanout[n]:
                if n >= I:
                    dependents[n - I] = dependents.get(n - I, False) or wt.get(I + g2, -1) > wt[n]
                if g2 not in seen:
                    seen.add(g2)
                    touched.append(g2)

        # --- pass 3: commit decisions, each pass from the previous one's, until no read decision changes ---------
        dec = {g: True for g in cand}

        def changes(g, before):
            """The gate's valid input changes at times < before, as sorted (time, pin)."""
            out = []
            for pin, n in enumerate(self.fanin[g]):
                t = wt.get(n)
                if t is not None and t < before and (n < I or dec[n - I]):
                    out.append((t, pin))
            return sorted(out)

        def decide(g):
            p, y = self.pend[g], self.val[I + g]
            x = [self.val[n] for n in self.fanin[g]]
            ev = changes(g, p)
            for t in sorted({t for t, _ in ev}):
                for te, pin in ev:
                    if te == t:
                        x[pin] ^= 1  # a net changes at most once per window: its new value is the opposite
                if self._f_of(g, x) == y:
                    return False
            return True

        while True:
            self.passes += 1
            new = {g: decide(g) for g in cand}
            stale = any(new[g] != dec[g] and dependents.get(g, False) for g in cand)
            dec = new
            if not stale:
                break

        # --- pass 4: replay touched gates ------------------------------------------------------------------------
        new_pend = {g: self._replay(g, dec, changes(g, t1), t0, t1, obs) for g in touched}

        # --- pass 5: apply the changes ---------------------------------------------------------------------------
        real = [(wt[I + g], I + g) for g in cand if dec[g]] + [(wt[n], n) for n in stim]
        for t, n in sorted(real):
            obs.change(t, n, self.val[n])
        for _, n in real:
            self.val[n] ^= 1
        for g, p in new_pend.items():
            self.pend[g] = p
            if p is not NEVER:
                heapq.heappush(self.heap, (p, g))

    def _f_of(self, g, x):
        idx = 0
        for pin in range(len(self.fanin[g])):
            idx |= x[pin] << pin
        return (self.tt[g] >> idx) & 1

    def _replay(self, g, dec, ev, t0, t1, obs):
        """Apply g's events in W (own commit, valid input changes) in time order, per SPEC §5; return its pending
        time at the end of W."""
        I = self.c.n_inputs
        y = self.val[I + g]
        x = [self.val[n] for n in self.fanin[g]]
        pend = self.pend[g]
        own = pend if pend is not NEVER and t0 <= pend < t1 and dec[g] else None
        for t in sorted({t for t, _ in ev} | ({own} if own is not None else set())):
            committed_now = own == t
            if committed_now:
                y ^= 1
                pend = NEVER
            pins = [pin for te, pin in ev if te == t]
            if not pins:
                continue
            for pin in pins:
                x[pin] ^= 1
            nets = {self.fanin[g][pin] for pin in pins}
            if len(nets | ({I + g} if committed_now else set())) >= 2:
                obs.first_tie = t if obs.ties == 0 else min(obs.first_tie, t)
                obs.ties += 1
            v = self._f_of(g, x)
            if v == y:
                if pend is not NEVER:
                    obs.cancels += 1
                pend = NEVER
            else:
                d = min((self.dr[g] if v else self.df[g])[q] for q in pins)
                tn = t + self._delay(g, d, pend is NEVER)
                if pend is NEVER or tn < pend:
                    if pend is not NEVER:
                        obs.advances += 1
                    pend = tn
        assert pend is NEVER or pend >= t1, "a transition was scheduled into the window it was created in"
        return pend
