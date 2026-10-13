#!/usr/bin/env python
"""Hold-out check of corner interpolation (docs/SPEC.md §12): for each characterized SKY130 library that lies
between two others of the same process and temperature, interpolate it from its neighbours and compare the
annotated per-arc delays, net capacitances, internal energies and leakage with the real library.

    nice -n 10 .venv/bin/python scripts/corner_holdout.py

Prints a Markdown table for docs/validation.md. Also compares geometric (log-linear) with linear interpolation of
the delays, which is the choice SPEC §12 makes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from freerun import cells  # noqa: E402
from freerun.variation import Corner, Design, LibraryFamily, _geo, _lin  # noqa: E402

LIBDIR = Path(__file__).resolve().parents[1] / "tools/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib"


def test_netlist():
    """A mixed netlist: a NAND-enabled ring, a NOR/XOR/MUX/AOI cone with fanout and a buffer."""
    I = cells.Instance
    p = "sky130_fd_sc_hd__"
    inst = [I("r0", p + "nand2_1", {"A": "en", "B": "r4", "Y": "r0"}),
            I("r1", p + "inv_1", {"A": "r0", "Y": "r1"}), I("r2", p + "inv_2", {"A": "r1", "Y": "r2"}),
            I("r3", p + "inv_1", {"A": "r2", "Y": "r3"}), I("r4", p + "inv_4", {"A": "r3", "Y": "r4"}),
            I("c0", p + "nor2_1", {"A": "a", "B": "r2", "Y": "c0"}),
            I("c1", p + "xor2_1", {"A": "c0", "B": "b", "X": "c1"}),
            I("c2", p + "mux2_1", {"A0": "c1", "A1": "r3", "S": "a", "X": "c2"}),
            I("c3", p + "a21oi_1", {"A1": "c2", "A2": "c0", "B1": "b", "Y": "c3"}),
            I("c4", p + "buf_2", {"A": "c3", "X": "c4"}),
            I("c5", p + "nand3_1", {"A": "c4", "B": "c1", "C": "r1", "Y": "c5"})]
    return cells.CellNetlist(inputs=["en", "a", "b"], outputs=["c5"], instances=inst)


def main():
    fam = LibraryFamily(LIBDIR)
    nl = test_netlist()
    design = Design(nl, fam, output_load=0.005)
    rows = []
    for c in fam.corners:
        same = sorted(v for v in (x.voltage for x in fam.corners if x.process == c.process and x.temp == c.temp)
                      if v != c.voltage)
        lo = [v for v in same if v < c.voltage]
        hi = [v for v in same if v > c.voltage]
        if not lo or not hi:
            continue
        a, b = Corner(c.process, c.temp, lo[-1]), Corner(c.process, c.temp, hi[0])
        w = (c.voltage - a.voltage) / (b.voltage - a.voltage)
        A, B, X = design.annotated(a), design.annotated(b), design.annotated(c)
        truth = np.concatenate([X.dr_ticks, X.df_ticks])
        da, db = np.concatenate([A.dr_ticks, A.df_ticks]), np.concatenate([B.dr_ticks, B.df_ticks])
        geo = _geo([da, db], [1 - w, w])
        lin = _lin([da, db], [1 - w, w])
        eg = np.abs(geo / truth - 1)
        el = np.abs(lin / truth - 1)
        cap = np.abs(_lin([A.energy.cap, B.energy.cap], [1 - w, w]) / np.where(X.energy.cap > 0, X.energy.cap, 1) - 1)
        cap = cap[X.energy.cap > 0]
        ei = _lin([A.energy.e_rise + A.energy.e_fall, B.energy.e_rise + B.energy.e_fall], [1 - w, w])
        ti = X.energy.e_rise + X.energy.e_fall
        eint = abs(ei.sum() / ti.sum() - 1)
        la, lb, lx = A.energy.leak.sum(), B.energy.leak.sum(), X.energy.leak.sum()
        lk = abs(np.exp((1 - w) * np.log(la) + w * np.log(lb)) / lx - 1)
        rows.append((str(c), f"{a.voltage:.2f}/{b.voltage:.2f}", eg.mean(), eg.max(), el.mean(), el.max(), cap.max(),
                     eint, float(lk)))
    print("| held-out corner | from V | delay err, geometric: mean / max | linear: mean / max | net cap max err |"
          " internal energy (sum) err | leakage (sum) err |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {100 * r[2]:.2f}% / {100 * r[3]:.2f}% | {100 * r[4]:.2f}% / {100 * r[5]:.2f}% | "
              f"{100 * r[6]:.2f}% | {100 * r[7]:.2f}% | {100 * r[8]:.1f}% |")


if __name__ == "__main__":
    main()
