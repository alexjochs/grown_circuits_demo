#!/usr/bin/env python
"""Energy accounting (docs/SPEC.md §11) vs SKY130 transistor level (ngspice): energy per cycle of ring oscillators
and inverter / NAND chains, and static leakage. Prints a Markdown table for docs/validation.md.

    nice -n 10 .venv/bin/python scripts/energy_vs_spice.py [--quick]

Energies are compared per cycle (ring) or per input cycle (chain), so the 1.2-1.45x NLDM-vs-SPICE delay offset
(docs/validation.md §2) does not enter. Leakage is compared on a static chain.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from freerun import cells, liberty, spice  # noqa: E402
from freerun.energy import energy  # noqa: E402
from freerun.ref import simulate  # noqa: E402

LIBDIR = spice.PDK / "libs.ref/sky130_fd_sc_hd/lib"
CORNERS = {"tt": ("tt_025C_1v80", 25.0, 1.80), "ss": ("ss_100C_1v60", 100.0, 1.60), "ff": ("ff_n40C_1v95", -40.0, 1.95)}
_libs = {}


def lib_for(corner):
    if corner not in _libs:
        _libs[corner] = liberty.load(LIBDIR / f"sky130_fd_sc_hd__{CORNERS[corner][0]}.lib")
    return _libs[corner]


def ring(n=5, cell="sky130_fd_sc_hd__inv_1", fanout_cell=None):
    inst = [cells.Instance(f"x{i}", cell, {"A": f"n{(i - 1) % n}", "Y": f"n{i}"}) for i in range(n)]
    if fanout_cell:  # one extra inverter on every ring node, outputs left floating (but switching)
        inst += [cells.Instance(f"f{i}", fanout_cell, {"A": f"n{i}", "Y": f"m{i}"}) for i in range(n)]
    nl = cells.CellNetlist(inputs=[], instances=inst)
    init = {f"n{i}": i % 2 for i in range(n)}
    if fanout_cell:
        init.update({f"m{i}": 1 - i % 2 for i in range(n)})
    return nl, init


def chain(n=4, cell="sky130_fd_sc_hd__inv_1"):
    if cell.startswith("sky130_fd_sc_hd__nand2"):
        inst = [cells.Instance(f"x{i}", cell, {"A": "a" if i == 0 else f"n{i}", "B": "one", "Y": f"n{i + 1}"})
                for i in range(n)]
        return cells.CellNetlist(inputs=["a"], instances=inst, consts={"one": 1})
    inst = [cells.Instance(f"x{i}", cell, {"A": "a" if i == 0 else f"n{i}", "Y": f"n{i + 1}"}) for i in range(n)]
    return cells.CellNetlist(inputs=["a"], instances=inst)


def ring_case(corner, fanout_cell=None, T=6e-9, settle=1.5e-9):
    name, temp, vdd = CORNERS[corner]
    nl, init = ring(fanout_cell=fanout_cell)
    deck, nets = spice.to_spice(nl, init=init, T=T, step=0.5e-12, corner=corner, temp=temp, vdd=vdd,
                                supply_current=True)
    t, V = spice.run(deck, nets)
    rises = [tt for tt, n, v in spice.digitize(t, V, vdd, nets) if n == "n0" and v == 1 and tt > settle]
    K = len(rises) - 1
    e_sp = spice.supply_energy(t, V, nets, vdd, rises[0], rises[-1]) / K
    per_sp = (rises[-1] - rises[0]) / K
    a = cells.to_circuit(nl, lib_for(corner), init=init)
    Tf = 50_000_000
    w = simulate(a.circuit, Tf)
    n0 = a.net_index["n0"]
    fr_rises = w.trace[(w.trace[:, 1] == n0) & (w.trace[:, 2] == 1), 0]
    cyc = len(fr_rises)
    e = energy(w, a.energy)
    per_fr = float(np.diff(fr_rises[1:]).mean()) * 1e-15
    return dict(e_sp=e_sp, e_fr=e.total / cyc, sw=e.switching / cyc, internal=e.internal / cyc,
                leak=e.leakage / cyc, per_sp=per_sp, per_fr=per_fr)


def chain_case(cell, corner="tt", n_cycles=8, period=1.2e-9, input_slew=30e-12):
    name, temp, vdd = CORNERS[corner]
    nl = chain(cell=cell)
    stim = []
    t0 = 0.3e-9
    for k in range(n_cycles):
        stim += [(t0 + k * period, "a", 1), (t0 + k * period + period / 2, "a", 0)]
    T = t0 + n_cycles * period + 0.2e-9
    inv = not cell.startswith("sky130_fd_sc_hd__and") and not cell.startswith("sky130_fd_sc_hd__buf")
    init = {"a": 0}
    v = 0
    for i in range(1, 5):
        v = 1 - v if inv else v
        init[f"n{i}"] = v
    deck, nets = spice.to_spice(nl, stimulus=stim, init=init, T=T, step=0.5e-12, input_slew=input_slew, corner=corner,
                                temp=temp, vdd=vdd, supply_current=True)
    t, V = spice.run(deck, nets)
    # whole input cycles 1 .. n_cycles-1 (the first one includes the start-up)
    a_lo, a_hi = t0 + period - period / 4, t0 + n_cycles * period - period / 4
    e_sp = spice.supply_energy(t, V, nets, vdd, a_lo, a_hi) / (n_cycles - 1)
    a = cells.to_circuit(nl, lib_for(corner), init=init, input_slew=input_slew * 1e9)
    fs = 1e15
    st = [(int(round(tt * fs)), a.net_index[nn], vv) for tt, nn, vv in stim]
    from freerun.ref import RefSim
    s = RefSim(a.circuit)
    s.add_stimulus(st)
    s.run(int(round(a_lo * fs)))
    w = s.run(int(round(a_hi * fs)))
    e = energy(w, a.energy)
    k = n_cycles - 1
    return dict(e_sp=e_sp, e_fr=e.total / k, sw=e.switching / k, internal=e.internal / k, leak=e.leakage / k)


def leakage_case(cell="sky130_fd_sc_hd__inv_1", corner="tt"):
    """Static 4-stage chain with the input at 0 and at 1: SPICE supply current vs Liberty leakage."""
    name, temp, vdd = CORNERS[corner]
    lib = lib_for(corner)
    out = []
    for a_val in (0, 1):
        nl = chain(cell=cell)
        init = {"a": a_val}
        v = a_val
        for i in range(1, 5):
            v = 1 - v
            init[f"n{i}"] = v
        deck, nets = spice.to_spice(nl, init=init, T=2e-9, step=10e-12, corner=corner, temp=temp, vdd=vdd,
                                    supply_current=True)
        t, V = spice.run(deck, nets)
        p_sp = spice.supply_energy(t, V, nets, vdd, 1e-9, 2e-9) / 1e-9
        c = lib[cell]
        states = dict(c.leakage_states)
        p_state = sum(states["A" if init["a" if i == 0 else f"n{i}"] else "!A"] for i in range(4)) * lib.leakage_unit_w
        out.append(dict(a=a_val, p_sp=p_sp, p_state=p_state, p_avg=4 * c.leakage * lib.leakage_unit_w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tt corner only")
    args = ap.parse_args()
    if not spice.available():
        sys.exit("ngspice / SKY130 not installed (scripts/build_tools.sh)")
    fj = 1e15
    print("| case | corner | SPICE fJ/cycle | freerun fJ/cycle (switching + internal + leakage) | freerun / SPICE |")
    print("|---|---|---|---|---|")
    for corner in (["tt"] if args.quick else ["tt", "ss", "ff"]):
        for label, fo in (("ring 5 x inv_1", None), ("ring 5 x inv_1, fanout 2", "sky130_fd_sc_hd__inv_1")):
            r = ring_case(corner, fo)
            print(f"| {label} (period SPICE {r['per_sp'] * 1e12:.1f} ps, freerun {r['per_fr'] * 1e12:.1f} ps) | "
                  f"{corner} | {r['e_sp'] * fj:.2f} | {r['e_fr'] * fj:.2f} ({r['sw'] * fj:.2f} + {r['internal'] * fj:.2f}"
                  f" + {r['leak'] * fj:.4f}) | {r['e_fr'] / r['e_sp']:.3f} |", flush=True)
    for cell in ("sky130_fd_sc_hd__inv_1", "sky130_fd_sc_hd__nand2_1"):
        r = chain_case(cell)
        print(f"| chain 4 x {cell.split('__')[1]}, per input cycle | tt | {r['e_sp'] * fj:.2f} | {r['e_fr'] * fj:.2f} "
              f"({r['sw'] * fj:.2f} + {r['internal'] * fj:.2f} + {r['leak'] * fj:.4f}) | {r['e_fr'] / r['e_sp']:.3f} |",
              flush=True)
    print()
    print("| static chain 4 x inv_1 | SPICE pW | Liberty state-dependent pW | Liberty cell_leakage_power (used) pW |")
    print("|---|---|---|---|")
    for r in leakage_case():
        print(f"| input {r['a']} | {r['p_sp'] * 1e12:.2f} | {r['p_state'] * 1e12:.2f} | {r['p_avg'] * 1e12:.2f} |")


if __name__ == "__main__":
    main()
