#!/usr/bin/env python
"""Idealization gap (docs/SPEC.md §12): a population of realistic SKY130 devices vs their idealized twins.

Two designs, each simulated as a population of devices (corners, die-to-die scale, cell mismatch, per-transition
jitter) together with the nominal twin and the uniform-delay twin, all in one batch:
  * a 5-stage ring oscillator (NAND enable + 4 inverters): oscillation period and energy per cycle;
  * a cross-coupled NAND latch: the narrowest set pulse that flips it.

    .venv/bin/python examples/idealization_gap.py [--devices 64] [--seed 1] [--backend cpu]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from freerun import cells  # noqa: E402
from freerun.energy import energy  # noqa: E402
from freerun.sim import Simulator, load_library  # noqa: E402
from freerun.variation import Box, Design, LibraryFamily, ideal_twin, population  # noqa: E402

LIBDIR = Path(__file__).resolve().parents[1] / "tools/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib"
FS = 1_000  # ticks per ps (1 tick = 1 fs)
CORNERS = [Box("ss", (-40, 100), (1.40, 1.60)), Box("tt", (25, 100), 1.80), Box("ff", (-40, 100), (1.65, 1.95))]


def ring_design(fam):
    inst = [cells.Instance("g0", "sky130_fd_sc_hd__nand2_1", {"A": "en", "B": "n4", "Y": "n0"})]
    inst += [cells.Instance(f"g{i}", "sky130_fd_sc_hd__inv_1", {"A": f"n{i - 1}", "Y": f"n{i}"}) for i in range(1, 5)]
    nl = cells.CellNetlist(inputs=["en"], outputs=["n4"], instances=inst)
    return Design(nl, fam, init={"n0": 1, "n1": 0, "n2": 1, "n3": 0, "n4": 1})


def latch_design(fam):
    nl = cells.CellNetlist(inputs=["s_n", "r_n"], outputs=["q", "qn"], instances=[
        cells.Instance("u0", "sky130_fd_sc_hd__nand2_1", {"A": "s_n", "B": "qn", "Y": "q"}),
        cells.Instance("u1", "sky130_fd_sc_hd__nand2_1", {"A": "r_n", "B": "q", "Y": "qn"})])
    return Design(nl, fam, init={"s_n": 1, "r_n": 1, "q": 0, "qn": 1})


def pick_backend(name):
    if name:
        return name
    try:
        load_library("cpu")
        return "cpu"
    except (RuntimeError, OSError):
        return "ref"


def ring_gap(fam, n, seed, backend, T_ns=20):
    d = ring_design(fam)
    devs = population(d, n, seed, corners=CORNERS, scale_sigma=0.03, mismatch_sigma=0.05, jitter_rel=0.01)
    twin, uni = ideal_twin(devs[0]), ideal_twin(devs[0], uniform=True)
    circuits = [x.circuit for x in devs] + [twin, uni]
    sim = Simulator(circuits, backend, watch=[[d.nominal.net_index["n4"]]] * len(circuits))
    for b in range(len(circuits)):
        sim.add_stimulus(b, [(100 * FS, 0, 1)])
    T = T_ns * 1_000 * FS
    sim.run(T // 4)  # settle
    wins = sim.run(T)
    periods, epc = [], []
    models = [x.energy for x in devs] + [devs[0].nominal_energy] * 2
    for w, m in zip(wins, models):
        rises = w.trace[w.trace[:, 2] == 1, 0]
        p = np.diff(rises)
        periods.append((p.mean() / FS, p.std() / FS))
        epc.append(energy(w, m).total / max(len(rises), 1))
    return devs, periods, epc


def latch_gap(fam, n, seed, backend, widths_ps=range(20, 161, 2), trials=8):
    """Narrowest S_n low pulse that sets the latch, per device (each width tried `trials` times, at different
    times so the jitter draws differ) and for the twins."""
    d = latch_design(fam)
    devs = population(d, n, seed, corners=CORNERS, scale_sigma=0.03, mismatch_sigma=0.05, jitter_rel=0.01)
    twin, uni = ideal_twin(devs[0]), ideal_twin(devs[0], uniform=True)
    base = [x.circuit for x in devs] + [twin, uni]
    q = d.nominal.net_index["q"]
    circuits, keys = [], []
    for i, c in enumerate(base):
        for w in widths_ps:
            for k in range(trials):
                circuits.append(c)
                keys.append((i, w, k))
    sim = Simulator(circuits, backend, watch=None)
    for b, (i, w, k) in enumerate(keys):
        t0 = (1000 + 37 * k) * FS  # copies of a circuit share its jitter stream, so vary the pulse time
        sim.add_stimulus(b, [(t0, 0, 0), (t0 + w * FS, 0, 1)])
    sim.run(4000 * FS)
    ok = np.array([sim.values(b)[q] == 1 for b in range(len(circuits))])
    thr = []
    for i in range(len(base)):
        sets = {w: np.mean([ok[b] for b, (j, ww, _) in enumerate(keys) if j == i and ww == w]) for w in widths_ps}
        lo = min((w for w in widths_ps if sets[w] > 0), default=None)   # sometimes sets
        hi = min((w for w in widths_ps if all(sets[v] == 1 for v in widths_ps if v >= w)), default=None)  # always
        thr.append((lo, hi))
    return devs, thr


def run(n_devices=64, seed=1, backend=None, quick=False):
    fam = LibraryFamily(LIBDIR)
    backend = pick_backend(backend)
    devs, periods, epc = ring_gap(fam, n_devices, seed, backend, T_ns=5 if quick else 20)
    ldevs, thr = latch_gap(fam, n_devices, seed, backend, widths_ps=range(20, 161, 4 if quick else 2),
                           trials=2 if quick else 8)
    return dict(backend=backend, devices=devs, periods=periods, energy_per_cycle=epc, latch_devices=ldevs,
                latch_thresholds=thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    r = run(args.devices, args.seed, args.backend)
    n = args.devices
    per = np.array([p for p, _ in r["periods"]])
    sd = np.array([s for _, s in r["periods"]])
    epc = np.array(r["energy_per_cycle"]) * 1e15
    print(f"backend: {r['backend']}, {n} devices (ss/tt/ff boxes, 3% scale, 5% mismatch, 1% jitter)\n")
    print("Ring oscillator, 5 stages")
    print(f"  nominal twin:      period {per[n]:8.2f} ps (sd {sd[n]:.2f}), {epc[n]:.2f} fJ/cycle")
    print(f"  uniform twin:      period {per[n + 1]:8.2f} ps (sd {sd[n + 1]:.2f}), {epc[n + 1]:.2f} fJ/cycle")
    print(f"  devices:           period {per[:n].mean():8.2f} ps mean, {per[:n].min():.2f} .. {per[:n].max():.2f}; "
          f"cycle-to-cycle sd {sd[:n].mean():.2f} ps; {epc[:n].mean():.2f} fJ/cycle "
          f"({epc[:n].min():.2f} .. {epc[:n].max():.2f})")
    gap = per[:n] / per[n] - 1
    print(f"  gap vs nominal:    {100 * gap.mean():+.1f}% mean, {100 * np.abs(gap).max():.1f}% max |gap|")
    for proc in ("ss", "tt", "ff"):
        idx = [i for i, d in enumerate(r["devices"]) if d.conditions.corner.process == proc]
        if idx:
            print(f"    {proc}: {len(idx):3d} devices, period {per[idx].mean():8.2f} ps, {epc[idx].mean():.2f} fJ/cycle")
    thr = r["latch_thresholds"]
    print("\nNAND latch, narrowest S_n pulse that sets it (ps): (sometimes, always)")
    print(f"  nominal twin:      {thr[n]}")
    print(f"  uniform twin:      {thr[n + 1]}")
    lo = np.array([t[0] for t in thr[:n] if t[0] is not None], dtype=float)
    hi = np.array([t[1] for t in thr[:n] if t[1] is not None], dtype=float)
    print(f"  devices:           sometimes {lo.mean():.1f} ({lo.min():.0f} .. {lo.max():.0f}), always {hi.mean():.1f} "
          f"({hi.min():.0f} .. {hi.max():.0f})")


if __name__ == "__main__":
    main()
