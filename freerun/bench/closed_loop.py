#!/usr/bin/env python
"""Closed-loop benchmark: the access pattern of a controller in the loop.

Every step advances all circuits by dt ticks, after feeding each circuit a fresh random input vector, and reads
each circuit's output nets (values and high time). State stays in the engine (on the device for CUDA); only the
dense input and output arrays cross the host boundary.

Usage: bench/closed_loop.py --backend cuda --circuits 4096 --gates 1024 --steps 1000 --dt 1000000

Workloads (1 tick = 1 fs):
  random  random SKY130-like gate functions with local wiring (+-16), arc delays 30-120 ps, 16 primary inputs
  rings   5-inverter rings filling the circuit, delays 30-120 ps (maximum activity, no inputs)
  mixed   random, except every 16th circuit has 4x shorter delays (a few circuits much busier than the rest)
Reports per-step wall time, simulation-kernel time, the fraction of wall time outside the kernels, and events/s.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import rings_circuit  # noqa: E402


def build(workload, rng, B, G, n_in, delay):
    if workload == "random":
        return [gen.random_circuit(rng, n_in, G, delay=delay, locality=16, tt_pool=gen.SKY130_LIKE) for _ in range(B)]
    if workload == "rings":
        return [rings_circuit(rng, G, delay=delay) for _ in range(B)]
    if workload == "mixed":
        return [gen.random_circuit(rng, n_in, G, delay=(delay[0] // 4, delay[1] // 4) if b % 16 == 7 else delay,
                                   locality=16, tt_pool=gen.SKY130_LIKE) for b in range(B)]
    raise ValueError(workload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--workload", default="random")
    ap.add_argument("--circuits", type=int, default=4096)
    ap.add_argument("--gates", type=int, default=1024)
    ap.add_argument("--inputs", type=int, default=16)
    ap.add_argument("--outputs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dt", type=int, default=1_000_000, help="ticks per step (1 tick = 1 fs)")
    ap.add_argument("--delay", type=int, nargs=2, default=(30_000, 120_000), help="arc delay range in ticks")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stepped", action="store_true", help="cuda: per-timestamp kernel")
    ap.add_argument("--block-threshold", type=int, default=None, help="cuda: gates above which a circuit gets a block")
    ap.add_argument("--block-threads", type=int, default=0)
    ap.add_argument("--no-smem", action="store_true", help="cuda: keep working state in global memory")
    ap.add_argument("--no-pack", action="store_true", help="cuda: 32-bit netlist layout only")
    ap.add_argument("--reorder", type=int, default=None, help="cuda: reorder the queue by activity every N runs (0 = off)")
    ap.add_argument("--no-high", action="store_true", help="read output values only")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="per-transition jitter sigma relative to each gate's mean delay (SPEC §10); 0 = off")
    ap.add_argument("--legacy", action="store_true",
                    help="emulate the pre-stepping API: stimulus triples + run() + full value readback (no kernel time)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    t0 = time.perf_counter()
    circuits = build(a.workload, rng, a.circuits, a.gates, a.inputs, tuple(a.delay))
    if a.jitter > 0:
        from freerun.jitter import sigma_relative
        from freerun.netlist import Jitter
        circuits = [c.with_jitter(Jitter.from_sigma(sigma_relative(c, a.jitter), seed=a.seed, cid=b))
                    for b, c in enumerate(circuits)]
    t_build = time.perf_counter() - t0

    if a.legacy:
        return legacy(a, rng, circuits, t_build)
    t0 = time.perf_counter()
    kw = {}
    if a.backend == "cuda":
        kw = dict(stepped=a.stepped, smem=not a.no_smem, packed=not a.no_pack)
        if a.block_threshold is not None or a.block_threads:
            kw.update(block_threshold=a.block_threshold, block_threads=a.block_threads)
        if a.reorder is not None:
            kw.update(reorder_every=a.reorder)
    sim = Simulator(circuits, a.backend, watch=None, net_observables=False, threads=a.threads, **kw)
    t_create = time.perf_counter() - t0
    t0 = time.perf_counter()
    outs = [list(range(c.n_nets - a.outputs, c.n_nets)) for c in circuits]
    sim.set_io(a.inputs, outs, high=not a.no_high)
    t_io = time.perf_counter() - t0

    B = a.circuits
    pool = [rng.integers(0, 2, size=(B, a.inputs), dtype=np.uint8) for _ in range(64)]
    for k in range(a.warmup):
        sim.step(sim.t + a.dt, pool[k % 64])
    wall, kern, events = [], [], 0
    for k in range(a.steps):
        x = pool[(a.warmup + k) % 64]
        t0 = time.perf_counter()
        r = sim.step(sim.t + a.dt, x)
        wall.append(time.perf_counter() - t0)
        kern.append(sim.profile()["kernel_s"])
        events += int(sim.events().sum())
    wall, kern = np.array(wall), np.array(kern)
    assert r.values.shape == (B, a.outputs)

    t0 = time.perf_counter()
    snap = sim.snapshot()
    t_snap = time.perf_counter() - t0
    t0 = time.perf_counter()
    sim.restore(snap)
    t_restore = time.perf_counter() - t0
    t0 = time.perf_counter()
    sim.reset(np.arange(B) % 2 == 0)
    t_reset = time.perf_counter() - t0

    res = dict(bench="closed_loop", backend=a.backend + ("-stepped" if a.stepped else ""), workload=a.workload,
               jitter=a.jitter, block_threshold=a.block_threshold, block_threads=a.block_threads, smem=not a.no_smem,
               packed=not a.no_pack, reorder=a.reorder,
               circuits=B, gates=a.gates, inputs=a.inputs, outputs=a.outputs, dt=a.dt, steps=a.steps,
               events=events, events_per_circuit_step=round(events / (B * a.steps), 1),
               step_ms_mean=round(1e3 * wall.mean(), 3), step_ms_median=round(1e3 * float(np.median(wall)), 3),
               kernel_ms_mean=round(1e3 * kern.mean(), 3), outside_kernel=round(1 - kern.sum() / wall.sum(), 4),
               events_per_s=events / wall.sum(), build_s=round(t_build, 2), create_s=round(t_create, 3),
               set_io_s=round(t_io, 4), snapshot_ms=round(1e3 * t_snap, 2), restore_ms=round(1e3 * t_restore, 2),
               reset_ms=round(1e3 * t_reset, 2))
    print(json.dumps(res), flush=True)


def legacy(a, rng, circuits, t_build):
    """The same loop through the API that existed before step(): per-circuit stimulus triples at the step start,
    run(T) (which also returns per-net windows), and one readback of all net values per step."""
    t0 = time.perf_counter()
    sim = Simulator(circuits, a.backend, watch=None, threads=a.threads)
    t_create = time.perf_counter() - t0
    B = len(circuits)
    out_idx = np.concatenate([sim.batch.net_off[b] + np.arange(c.n_nets - a.outputs, c.n_nets)
                              for b, c in enumerate(circuits)])
    pool = [rng.integers(0, 2, size=(B, a.inputs), dtype=np.uint8) for _ in range(64)]
    wall, events = [], 0
    for k in range(a.warmup + a.steps):
        x = pool[k % 64]
        t0 = time.perf_counter()
        for b, c in enumerate(circuits):
            sim.add_stimulus(b, [(sim.t, i, int(x[b, i])) for i in range(min(a.inputs, c.n_inputs))])
        ws = sim.run(sim.t + a.dt)
        vals = sim._get("values", sim.batch.n_nets, np.uint8)[out_idx]
        if not a.no_high:
            high = np.concatenate([w.high[-a.outputs:] for w in ws])
        if k >= a.warmup:
            wall.append(time.perf_counter() - t0)
            events += sum(w.events for w in ws)
    assert len(vals) == B * a.outputs
    wall = np.array(wall)
    res = dict(bench="closed_loop", backend=a.backend + "-legacy-api", workload=a.workload, jitter=a.jitter,
               circuits=B, gates=a.gates,
               inputs=a.inputs, outputs=a.outputs, dt=a.dt, steps=a.steps, events=events,
               events_per_circuit_step=round(events / (B * a.steps), 1), step_ms_mean=round(1e3 * wall.mean(), 3),
               step_ms_median=round(1e3 * float(np.median(wall)), 3), events_per_s=events / wall.sum(),
               build_s=round(t_build, 2), create_s=round(t_create, 3))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
