#!/usr/bin/env python
"""run_steps at physical step lengths: CUDA (every kernel variant) == CPU == reference, bit for bit.

The regime of the grown search's event-stream adapter (evolved_cartpole docs/grown_time.md): ticks of 10 ps, control
steps of 1 us to 1 ms (K = 10^5 .. 10^8 ticks), per-step event budgets of 10^3 .. 10^6 gate transitions with early
stop, SKY130-like delays (arcs of 3-12 ticks), device variation (per-arc delay mismatch) and per-transition jitter.

Per case: B random circuits (mostly cyclic: random feedback netlists, many of which oscillate at ~1/tau transitions per
second, plus gated rings and feed-forward ones), inputs as in the adapter: sample-and-hold codes changing at step
boundaries, a strobe wire high for W ticks from every boundary, and on some circuits sigma-delta-like pulse trains
(pulses of width P = 10^4 ticks). One run_steps call (offset 1, as the adapter samples (sK, (s+1)K]; hold_at_end, so a
circuit whose episode is shorter than the call's stops at its own end, as in the adapter) on every backend;
values, high time, toggles, per-step events and stop steps must be identical, and so must the final state.

    bench/multistep_physK.py [--backends ref cpu cuda ...] [--K 100000 1000000 10000000 100000000] [--cases 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from freerun import gen  # noqa: E402
from freerun.netlist import Circuit, Jitter  # noqa: E402
from freerun.sim import Simulator  # noqa: E402

N_DATA = 3
W = 10_000          # strobe width and pulse width: one 10 MHz modulator period at 10 ps
NAND2 = 0b0111


def gated_ring(stages, rng):
    """NAND(enable, last) followed by stages-1 inverters: oscillates while input 0 is 1. Inputs: data + strobe."""
    n_in = N_DATA + 1
    I = n_in
    ptr, fan, tt = [0], [], []
    en = int(rng.integers(0, n_in))
    fan += [en, I + stages - 1]
    tt.append(NAND2)
    ptr.append(len(fan))
    for g in range(1, stages):
        fan.append(I + g - 1)
        tt.append(0b01)
        ptr.append(len(fan))
    E = len(fan)
    return Circuit(n_in, np.array(ptr), np.array(fan), np.array(tt, np.uint64), rng.integers(3, 13, E),
                   rng.integers(3, 13, E), np.zeros(n_in + stages, np.uint8))


def build(rng, B, jitter_sigma, mismatch):
    circuits = []
    for b in range(B):
        kind = rng.choice(["cyclic", "cyclic", "ring", "ff"])
        if kind == "ring":
            c = gated_ring(int(rng.integers(1, 6)), rng)
        else:
            c = gen.random_circuit(rng, N_DATA + 1, int(rng.integers(8, 33)), delay=(3, 12), max_arity=3,
                                   feedforward=kind == "ff", tt_pool=gen.SKY130_LIKE)
        if mismatch > 0 and b % 2 == 0:       # a device: per-arc delay mismatch (lognormal), rounded to ticks
            f = np.exp(rng.normal(0, mismatch, size=(2, len(c.dr))))
            c = Circuit(c.n_inputs, c.fanin_ptr, c.fanin, c.tt, np.maximum(1, np.rint(c.dr * f[0])),
                        np.maximum(1, np.rint(c.df * f[1])), c.init)
        if jitter_sigma > 0 and b % 3 != 2:
            from freerun.jitter import sigma_relative
            c = c.with_jitter(Jitter.from_sigma(sigma_relative(c, jitter_sigma), seed=int(rng.integers(1 << 30)),
                                                cid=b))
        circuits.append((str(kind), c))
    return circuits


def stimulus(rng, circuits, K, S):
    """Per circuit: data codes held per step (changes at s*K), the strobe (wire N_DATA) high on [s*K, s*K + W) (its
    value at t = 0 is the power-up value, not a triple), and on every third circuit pulses of width W on wire 0 at
    random multiples of W in each step (at most 32 per step)."""
    trip = []
    for b, (_, c) in enumerate(circuits):
        ev = {}
        k = int(K[b])
        cur = c.init[:N_DATA].copy()
        for s in range(int(S[b])):
            t0 = s * k
            if s > 0:
                code = rng.integers(0, 2, N_DATA)
                for i in np.flatnonzero(code != cur):
                    ev[(t0, int(i))] = int(code[i])
                cur = code
                ev[(t0, N_DATA)] = 1
            if W < k:
                ev[(t0 + W, N_DATA)] = 0
            if b % 3 == 0 and k >= 4 * W:
                slots = np.sort(rng.choice(np.arange(1, k // W - 1), size=min(32, k // W - 2), replace=False))
                for j in slots:
                    ev[(t0 + int(j) * W, 0)] = 1
                    ev[(t0 + int(j) * W + W // 2, 0)] = 0
        trip.append(sorted((t, i, v) for (t, i), v in ev.items() if t > 0))
    off = np.concatenate([[0], np.cumsum([len(t) for t in trip])]).astype(np.int64)
    rows = np.array([e for t in trip for e in t], np.int64).reshape(-1, 3)
    return off, rows[:, 0], rows[:, 1], rows[:, 2]


def run(backend, circuits, K, S, stim, budget):
    from test_backends import CUDA_VARIANTS
    cs = [c for _, c in circuits]
    if backend.startswith("cuda"):
        sim = Simulator(cs, "cuda", watch=None, net_observables=False, **CUDA_VARIANTS[backend])
    else:
        sim = Simulator(cs, backend, watch=None, net_observables=False)
    outs = [[c.n_nets - 1, c.n_nets - 2, N_DATA] for c in cs]
    sim.set_io(0, outs, high=True, toggles=True)
    t0 = time.perf_counter()
    r = sim.run_steps(K, S, stim, budget=budget, offset=1, hold_at_end=True)   # as the grown adapter calls it
    wall = time.perf_counter() - t0
    st = sim.state()
    ev = sim.events()
    sim.close()
    return r, st, ev, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["ref", "cpu", "cuda", "cuda-stepped", "cuda-block",
                                                      "cuda-block-global", "cuda-block64", "cuda-generic-global",
                                                      "cuda-generic-block"])
    ap.add_argument("--K", type=int, nargs="+", default=[10**5, 10**6, 10**7, 10**8])
    ap.add_argument("--circuits", type=int, default=32)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--cases", type=int, default=2, help="random cases per K (plus one mixed-K case)")
    ap.add_argument("--jitter", type=float, default=0.01)
    ap.add_argument("--mismatch", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    plan = [(k, c) for k in a.K for c in range(a.cases)] + [(None, 0)]
    fails = 0
    for k, case in plan:
        circuits = build(rng, a.circuits, a.jitter, a.mismatch)
        K = (np.full(a.circuits, k, np.int64) if k is not None else rng.choice(a.K, a.circuits).astype(np.int64))
        S = rng.integers(2, a.steps + 1, a.circuits).astype(np.int64)
        budget = rng.choice([1e3, 1e4, 1e5, 1e6], a.circuits, p=[0.4, 0.3, 0.2, 0.1])
        ff = np.array([kind == "ff" for kind, _ in circuits])
        budget[ff & (rng.random(a.circuits) < 0.5)] = np.inf      # feed-forward: quiescent between edges, no budget
        stim = stimulus(rng, circuits, K, S)
        res = {}
        for be in a.backends:
            res[be] = run(be, circuits, K, S, stim, budget)
        base = a.backends[0]
        r0, st0, ev0, _ = res[base]
        same = {}
        for be in a.backends[1:]:
            r, st, ev, _ = res[be]
            ok = all(np.array_equal(np.asarray(getattr(r0, f)), np.asarray(getattr(r, f)))
                     for f in ("values", "high", "toggles", "events", "stop"))
            ok = ok and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(st0, st))
            ok = ok and np.array_equal(np.asarray(ev0), np.asarray(ev))
            same[be] = bool(ok)
            fails += not ok
        stop = np.asarray(r0.stop)
        print(json.dumps(dict(bench="multistep_physK", K=k if k is not None else "mixed", case=case,
                              circuits=a.circuits, kinds={x: int(sum(kd == x for kd, _ in circuits))
                                                          for x in ("cyclic", "ring", "ff")},
                              steps=int(S.sum()), stimulus=int(len(stim[1])), jitter=a.jitter, mismatch=a.mismatch,
                              budgets={str(int(v)) if np.isfinite(v) else "none": int((budget == v).sum())
                                       for v in np.unique(budget)},
                              stopped=int((stop >= 0).sum()), events=int(np.asarray(r0.events).sum()),
                              max_step_events=int(np.asarray(r0.events).max()),
                              wall_s={be: round(res[be][3], 3) for be in a.backends},
                              equal_to=base, same=same)), flush=True)
    print(json.dumps(dict(bench="multistep_physK", cases=len(plan), backends=a.backends, failures=fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
