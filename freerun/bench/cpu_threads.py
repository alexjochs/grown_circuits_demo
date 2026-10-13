#!/usr/bin/env python
"""CPU backend thread scaling of multi-step runs (Simulator.run_steps, SPEC §13): one circuit per lane, lanes spread
over threads. Workloads:

  rings   B identical 3-gate rings (arcs of 5 ticks): the smallest circuits, the worst case for threads that share
          cache lines; S steps of K ticks, no stimulus, no budget
  npz     a dumped engine call (evolved_cartpole scripts/grown_lane_probe.py --dump): the lanes of a real search
          generation with their netlists, step lengths, stimulus and output nets, run for at most --steps steps each
          (the full episode with --steps 0)

For every thread count: wall seconds, events, events/s, speedup over one thread, and whether the per-step outputs,
events and final state equal the one-thread run's (they must: lanes are independent).

    bench/cpu_threads.py rings --threads 1 2 8 32 64 [--lanes 1024 --K 1000000 --steps 4]
    bench/cpu_threads.py npz --file lanes.npz --threads 1 2 8 32 64 [--steps 4]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from freerun import gen  # noqa: E402
from freerun.batch import Batch  # noqa: E402
from freerun.sim import Simulator  # noqa: E402


def ring_case(a):
    c = gen.ring(3, rise=5, fall=5, init=[1, 0, 0])
    B = a.lanes
    return dict(batch=[c] * B, K=np.full(B, a.K, np.int64), S=np.full(B, a.steps, np.int64), stim=None,
                outs=[[2]] * B, n_in=0)


def npz_case(a):
    z = np.load(a.file)
    b = Batch.__new__(Batch)
    b.circuits = None
    b.B = len(z["K"])
    for k in ("net_off", "gate_off", "edge_off"):
        setattr(b, k, np.ascontiguousarray(z[k], np.int64))
    b.n_inputs = np.ascontiguousarray(z["n_inputs"], np.int32)
    for k in ("fanin_ptr", "fanin", "dr", "df"):
        setattr(b, k, np.ascontiguousarray(z[k], np.int32))
    b.tt = np.ascontiguousarray(z["tt"], np.uint64)
    b.init = np.ascontiguousarray(z["init"], np.uint8)
    b.has_jitter = False
    b.jit_seed = np.zeros(b.B, np.uint64)
    b.jit_cid = np.zeros(b.B, np.uint32)
    b.jit_dmin = np.zeros(b.B, np.int32)
    b.jit_amp = np.zeros(int(b.gate_off[-1]), np.int64)
    K = np.asarray(z["K"], np.int64)
    S = np.asarray(z["S"], np.int64)
    if a.steps > 0:
        S = np.minimum(S, a.steps)
    off, t, net, v = (np.asarray(z[k]) for k in ("stim_off", "stim_t", "stim_net", "stim_v"))
    keep = t <= np.repeat(S * K, np.diff(off))                  # triples within each lane's (shortened) episode
    lane = np.repeat(np.arange(b.B), np.diff(off))[keep]
    off2 = np.zeros(b.B + 1, np.int64)
    np.cumsum(np.bincount(lane, minlength=b.B), out=off2[1:])
    stim = (off2, t[keep].astype(np.int64), net[keep].astype(np.int32), v[keep].astype(np.uint8))
    return dict(batch=b, K=K, S=S, stim=stim, outs=np.asarray(z["outs"], np.int32), n_in=int(z["n_in"]))


def run(case, threads):
    sim = Simulator(case["batch"], "cpu", watch=None, net_observables=False, threads=threads)
    sim.set_io(case["n_in"], case["outs"], high=True)
    t0 = time.perf_counter()
    r = sim.run_steps(case["K"], case["S"], case["stim"], offset=1, hold_at_end=True)
    wall = time.perf_counter() - t0
    _, vals, pend, _ = sim.state()
    out = (np.asarray(r.values), np.asarray(r.high), np.asarray(r.events), vals, pend)
    sim.close()
    return wall, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workload", choices=["rings", "npz"])
    ap.add_argument("--file", default=None)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 8, 32, 64])
    ap.add_argument("--lanes", type=int, default=1024)
    ap.add_argument("--K", type=int, default=1_000_000)
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    case = ring_case(a) if a.workload == "rings" else npz_case(a)
    base = None
    for th in a.threads:
        wall, out = run(case, th)
        ev = float(out[2].sum())
        if base is None:
            base = (wall, out)
        same = all(np.array_equal(x, y) for x, y in zip(out, base[1]))
        print(json.dumps(dict(bench="cpu_threads", workload=a.workload, lanes=int(len(case["K"])), steps=a.steps,
                              threads=th, wall_s=round(wall, 3), events=ev, events_per_s=ev / wall,
                              speedup=round(base[0] / wall, 2), same_as_first=same)), flush=True)


if __name__ == "__main__":
    main()
