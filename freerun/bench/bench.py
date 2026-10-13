#!/usr/bin/env python
"""Throughput benchmark: events/s for a batch of circuits on a given backend.

Usage: bench/bench.py --backend cpu --workload random --circuits 256 --gates 4096 --T 200000 [--windows 10]

Workloads:
  random  random SKY130-like gate functions, local wiring (+-16), arc delays 30..120, 16 inputs with random
          toggles every ~2000 ticks
  rings   independent 5-inverter rings filling the circuit (every gate toggles every ~2 stage delays): the
          maximum-activity case
All runs use watch=None (no trace) and stats off, i.e. the production configuration.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from freerun import gen  # noqa: E402
from freerun.netlist import INV, Circuit  # noqa: E402
from freerun.sim import Simulator  # noqa: E402


def rings_circuit(rng, n_gates, stages=5, delay=(30, 120)):
    n_rings = max(1, n_gates // stages)
    G = n_rings * stages
    g = np.arange(G)
    src = np.where(g % stages == 0, g + stages - 1, g - 1)  # stage i reads stage i-1 of the same ring
    dr = rng.integers(delay[0], delay[1] + 1, size=G)
    df = rng.integers(delay[0], delay[1] + 1, size=G)
    init = (g % stages) % 2  # one transition circulating per ring
    return Circuit(0, np.arange(G + 1), src, np.full(G, INV, dtype=np.uint64), dr, df, init)


def build(workload, rng, B, G, T):
    circuits, stims = [], []
    for _ in range(B):
        if workload == "random":
            circuits.append(gen.random_circuit(rng, 16, G, delay=(30, 120), locality=16, tt_pool=gen.SKY130_LIKE))
            stims.append(gen.random_stimulus(rng, 16, T, 2000))
        elif workload == "rings":
            circuits.append(rings_circuit(rng, G))
            stims.append([])
        else:
            raise ValueError(workload)
    return circuits, stims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--workload", default="random")
    ap.add_argument("--circuits", type=int, default=256)
    ap.add_argument("--gates", type=int, default=4096)
    ap.add_argument("--T", type=int, default=200_000)
    ap.add_argument("--windows", type=int, default=1, help="split [0, T) into this many run() calls")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stepped", action="store_true", help="cuda: per-timestamp kernel instead of windows")
    ap.add_argument("--block-threshold", type=int, default=None, help="cuda: gates above which a circuit gets a block")
    ap.add_argument("--block-threads", type=int, default=0)
    ap.add_argument("--no-smem", action="store_true", help="cuda: never stage state in shared memory")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="per-transition jitter sigma relative to each gate's mean delay (SPEC §10); 0 = off")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    circuits, stims = build(a.workload, rng, a.circuits, a.gates, a.T)
    if a.jitter > 0:
        from freerun.jitter import sigma_relative
        from freerun.netlist import Jitter
        circuits = [c.with_jitter(Jitter.from_sigma(sigma_relative(c, a.jitter), seed=a.seed, cid=b))
                    for b, c in enumerate(circuits)]
    t_build = time.time() - t0
    t0 = time.time()
    kw = dict(stepped=True) if a.stepped else {}
    if a.no_smem:
        kw["smem"] = False
    if a.block_threshold is not None or a.block_threads:
        kw.update(block_threshold=a.block_threshold, block_threads=a.block_threads)
    sim = Simulator(circuits, a.backend, watch=None, threads=a.threads, **kw)
    ev = [(b, *e) for b, s in enumerate(stims) for e in s]
    if ev:
        sim.add_stimulus_arrays(*np.array(ev, dtype=np.int64).T)
    t_init = time.time() - t0
    events, t_run, t_kernel, n_win = 0, 0.0, 0.0, 0
    cuts = np.linspace(0, a.T, a.windows + 1)[1:].astype(np.int64)
    for T in cuts:
        t0 = time.time()
        ws = sim.run(int(T))
        t_run += time.time() - t0
        prof = sim.profile()
        t_kernel += prof["kernel_s"]
        n_win += prof["windows"]
        events += sum(w.events for w in ws)
    res = dict(backend=a.backend + ("-stepped" if a.stepped else ""), block_threshold=a.block_threshold,
               block_threads=a.block_threads, smem=not a.no_smem, jitter=a.jitter, workload=a.workload,
               circuits=a.circuits, gates=a.gates,
               total_gates=sum(c.n_gates for c in circuits), T=a.T, windows=a.windows, events=events,
               run_s=round(t_run, 4), kernel_s=round(t_kernel, 4), init_s=round(t_init, 3), build_s=round(t_build, 2),
               events_per_window=round(events / n_win, 1) if n_win else None,
               events_per_s=events / t_run if t_run > 0 else float("nan"),
               events_per_kernel_s=events / t_kernel if t_kernel > 0 else float("nan"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
