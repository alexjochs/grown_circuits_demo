#!/usr/bin/env python
"""Multi-step runs vs the step loop: a whole open-loop episode per launch (Simulator.run_steps, SPEC §13) against the
same episode driven by step() calls, the access pattern of a search that evaluates one generation of circuits.

Every circuit is a small random netlist (the grown search's regime: 8-32 gates, arc delays 3-12 ticks = 30-120 ps at
10 ps per tick) with sample-and-hold inputs that take a new random code at every step boundary. One episode is S steps
of K ticks for the whole batch, the work of one generation. Two ways to run it:

  loop   the step loop of a client that samples step s over (sK, (s+1)K]: per step a call [sK, sK+1) that applies the
         new input row, and a call [sK+1, (s+1)K+1) to the end of the step; with a budget the long call is cut in
         geometric chunks (1024 ticks, then x4) and lanes over budget are frozen with load_state, as the grown adapter
         does. 2 S engine calls (more with chunks).
  steps  one run_steps call (offset 1, inputs as stimulus triples, the budget in the engine).

Workloads: ff (feed-forward: quiescent between input changes; the cost is the calls) and cyclic (feedback allowed:
some circuits oscillate, so a per-step event budget is needed at large K; --budget).
Reports ms per episode (= per generation), engine kernel ms, events/s, and checks that both give the same per-step
outputs (exactly, without a budget; for lanes that stay under it, with one).

Usage: bench/multistep.py --backend cuda --circuits 1034 --steps 256 --dt 1000 100000 10000000 [--workload cyclic --budget 1000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from freerun import gen  # noqa: E402
from freerun.sim import Simulator  # noqa: E402

CHUNK = 1024


def build(a, rng):
    circuits = []
    for _ in range(a.circuits):
        G = int(rng.integers(a.gates[0], a.gates[1] + 1))
        c = gen.random_circuit(rng, a.inputs, G, delay=(3, 12), feedforward=a.workload == "ff", max_arity=3,
                               tt_pool=gen.SKY130_LIKE)
        circuits.append(c)
    if a.jitter > 0:
        from freerun.jitter import sigma_relative
        from freerun.netlist import Jitter
        circuits = [c.with_jitter(Jitter.from_sigma(sigma_relative(c, a.jitter), seed=a.seed, cid=b))
                    for b, c in enumerate(circuits)]
    return circuits


def make_sim(a, circuits):
    kw = {}
    if a.backend == "cuda" and a.no_smem:
        kw["smem"] = False
    return Simulator(circuits, a.backend, watch=None, net_observables=False, threads=a.threads, **kw)


def outputs(circuits, n_out):
    return [list(range(c.n_nets - n_out, c.n_nets)) for c in circuits]


def run_loop(a, circuits, x, K, S, budget):
    """The step loop (module docstring): per step s a call [sK, sK+1) with the step's input row and a call
    [sK+1, (s+1)K) (in chunks with a budget); step s's window (sK, (s+1)K] ends with the next step's first call.
    x [B, S, I]: the code of each step (x[:, 0] = the circuits' power-up inputs)."""
    B = len(circuits)
    sim = make_sim(a, circuits)
    sim.set_io(a.inputs, outputs(circuits, a.outputs), high=True)
    n_out = a.outputs
    val = np.zeros((B, S, n_out), np.uint8)
    high = np.zeros((B, S, n_out), np.int64)
    ev = np.zeros((B, S), np.int64)
    frozen = np.zeros(B, bool)
    cur = x[:, 0].copy()                                   # input values in effect
    t_kernel, calls, events = 0.0, 0, 0

    def call(T, row=None):
        nonlocal t_kernel, calls, events
        r = sim.step(T, row)
        e = sim.events()
        t_kernel += sim.profile()["kernel_s"]
        calls += 1
        events += int(e.sum())
        return r, e

    def freeze(s):
        over = (ev[:, s] > budget) & ~frozen
        if over.any():
            tn, v, p, c = sim.state()
            for b in np.flatnonzero(over):
                p[sim.batch.gates(b)] = -1
            sim.load_state((tn, v, p, c), over)
            frozen[over] = True

    t0 = time.perf_counter()
    for s in range(S + 1):
        row = None
        n_in = np.zeros(B, np.int64)
        if s < S:
            row = np.where(frozen[:, None], cur, x[:, s])
            n_in = (row != cur).sum(1)
            cur = row
        r, e = call(s * K + 1, row)                        # [sK, sK+1): the last tick of step s - 1's window
        if s >= 1:
            val[:, s - 1] = r.values
            high[:, s - 1] += r.high
            ev[:, s - 1] += np.where(frozen, 0, e - n_in)
            if budget is not None:
                freeze(s - 1)
        if s == S:
            break
        lo, end = s * K + 1, (s + 1) * K
        cuts = [end]
        if budget is not None and end - lo > 4 * CHUNK:
            cuts = [lo + CHUNK * 4 ** k for k in range(64) if lo + CHUNK * 4 ** k < end] + [end]
        for T in cuts:
            if T <= sim.t:
                continue
            r, e = call(T)
            high[:, s] += r.high
            ev[:, s] += e
            if budget is not None:
                freeze(s)
    wall = time.perf_counter() - t0
    sim.close()
    return dict(wall=wall, kernel=t_kernel, calls=calls, events=events), (val, high, ev, frozen)


def stimulus(x, K):
    """Sample-and-hold codes as triples: every wire whose value changes at a step boundary sK, s >= 1 (step 0's code
    is the circuits' power-up input values)."""
    B, S, I = x.shape
    b, s, i = np.nonzero(x[:, 1:] != x[:, :-1])
    s = s + 1
    t = s.astype(np.int64) * K
    return b, t, i, x[b, s, i]


def run_steps(a, circuits, x, K, S, budget):
    B = len(circuits)
    sim = make_sim(a, circuits)
    sim.set_io(a.inputs, outputs(circuits, a.outputs), high=True)
    t0 = time.perf_counter()
    b, t, i, v = stimulus(x, K)
    o = np.lexsort((i, t, b))
    b, t, i, v = b[o], t[o], i[o], v[o]
    off = np.searchsorted(b, np.arange(B + 1)).astype(np.int64)
    t_pack = time.perf_counter() - t0
    r = sim.run_steps(K, S, (off, t, i, v), budget=budget, offset=1)
    wall = time.perf_counter() - t0
    events = int(sim.events().sum())
    k = sim.profile()["kernel_s"]
    sim.close()
    return dict(wall=wall, kernel=k, calls=1, events=events, pack=t_pack), (r.values, r.high, r.events, r.stop >= 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--workload", default="ff", choices=["ff", "cyclic"])
    ap.add_argument("--circuits", type=int, default=1034)
    ap.add_argument("--gates", type=int, nargs=2, default=(8, 32))
    ap.add_argument("--inputs", type=int, default=3)
    ap.add_argument("--outputs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--dt", type=int, nargs="+", default=[1000, 100_000, 10_000_000], help="ticks per step (K)")
    ap.add_argument("--budget", type=float, default=None, help="gate events per step (None: no budget)")
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3, help="episodes per mode (the first is a warm-up)")
    ap.add_argument("--modes", nargs="+", default=["loop", "steps"])
    ap.add_argument("--no-smem", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    circuits = build(a, rng)
    for K in a.dt:
        S = a.steps
        x = rng.integers(0, 2, size=(a.circuits, S, a.inputs), dtype=np.uint8)
        x[:, 0] = [c.init[:a.inputs] for c in circuits]   # step 0's code = the power-up inputs
        res = {}
        outs = {}
        for mode in a.modes:
            fn = run_loop if mode == "loop" else run_steps
            runs = [fn(a, circuits, x, K, S, a.budget) for _ in range(max(a.repeat, 1))]
            keep = runs[1:] or runs
            res[mode] = {k: float(np.median([r[0][k] for r in keep])) for k in keep[0][0]}
            outs[mode] = keep[-1][1]
        same = None
        if len(outs) == 2:
            (v1, h1, e1, f1), (v2, h2, e2, f2) = outs["loop"], outs["steps"]
            ok = ~(f1 | f2)
            same = bool(np.array_equal(v1[ok], v2[ok]) and np.array_equal(h1[ok], h2[ok]) and
                        np.array_equal(e1[ok], e2[ok]) and np.array_equal(f1, f2))
        for mode, r in res.items():
            print(json.dumps(dict(bench="multistep", backend=a.backend, mode=mode, workload=a.workload,
                                  circuits=a.circuits, gates=list(a.gates), steps=S, K=K, budget=a.budget,
                                  jitter=a.jitter, ms_per_episode=round(1e3 * r["wall"], 3),
                                  kernel_ms=round(1e3 * r["kernel"], 3), calls=int(r["calls"]),
                                  events=int(r["events"]), events_per_s=r["events"] / max(r["wall"], 1e-12),
                                  stopped=int(outs[mode][3].sum()), same_as_loop=same)), flush=True)
        if len(res) == 2:
            print(json.dumps(dict(bench="multistep", K=K, circuits=a.circuits, workload=a.workload,
                                  speedup=round(res["loop"]["wall"] / res["steps"]["wall"], 2))), flush=True)


if __name__ == "__main__":
    main()
