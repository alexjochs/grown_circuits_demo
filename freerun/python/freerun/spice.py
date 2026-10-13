"""Transistor-level reference: simulate a cell netlist in ngspice with the PDK's SPICE models and digitize the result.

Used to measure how far freerun's delay models are from the physics (they are not expected to agree exactly):
  deck = to_spice(netlist, pdk, ...)      # cell subcircuits, PWL inputs, .ic from the initial state
  t, v = run(deck)                        # ngspice batch transient, all nets
  edges = digitize(t, v, vdd)             # (time_s, net, value) at VDD/2 crossings
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .cells import CellNetlist

TOOLS = Path(os.environ.get("FREERUN_TOOLS", Path(__file__).resolve().parents[2] / "tools"))
NGSPICE = TOOLS / "ngspice/bin/ngspice"
PDK = TOOLS / "pdk/sky130A"


def available() -> bool:
    return NGSPICE.exists() and (PDK / "libs.tech/ngspice/sky130.lib.spice").exists()


_ports_cache: dict = {}


def subckt_ports(spice_file: Path) -> dict[str, list[str]]:
    key = str(spice_file)
    if key not in _ports_cache:
        ports = {}
        for m in re.finditer(r"^\.subckt\s+(\S+)\s+([^\n]*)$", spice_file.read_text(), re.M | re.I):
            ports[m.group(1)] = m.group(2).split()
        _ports_cache[key] = ports
    return _ports_cache[key]


def _node(name: str) -> str:
    return "n_" + re.sub(r"[^A-Za-z0-9_]", lambda m: f"_{ord(m.group(0)):02x}", name)


SUPPLY = "<i(vdd)>"  # pseudo net name of the supply current column (to_spice(..., supply_current=True))


def to_spice(nl: CellNetlist, *, stimulus=(), init=None, T: float, step: float = 1e-12, input_slew: float = 50e-12,
             corner: str = "tt", temp: float = 25.0, vdd: float = 1.8, loads: dict | None = None,
             library: str = "sky130_fd_sc_hd", pdk: Path = PDK, supply_current: bool = False) -> tuple[str, list[str]]:
    """SPICE deck for nl. Times in seconds. stimulus: (t, input net, value); input transitions are linear ramps
    centred on t, so the input crosses VDD/2 at t (the time freerun sees the change). input_slew is the 20-80%
    transition time, the Liberty definition (SKY130: slew_*_threshold_pct 20/80), i.e. the ramp lasts
    input_slew / 0.6.
    init: net name -> 0/1 initial values (applied with .ic). loads: net name -> extra farads.
    supply_current: also write the current of the VDD source, as the last column, named SUPPLY (for
    supply_energy). Input sources and load capacitors are not part of it.
    Returns (deck, list of net names written to the output, in column order)."""
    cells_sp = pdk / f"libs.ref/{library}/spice/{library}.spice"
    ports = subckt_ports(cells_sp)
    nets = []
    lines = [f"* freerun deck: {len(nl.instances)} instances",
             f'.lib "{pdk}/libs.tech/ngspice/sky130.lib.spice" {corner}',
             f'.include "{cells_sp}"', ".option scale=1e-6", f".temp {temp}", f"Vdd vpwr 0 {vdd}"]
    supply = {"VPWR": "vpwr", "VPB": "vpwr", "VGND": "0", "VNB": "0"}
    init = init or {}
    for inst in nl.instances:
        if inst.cell not in ports:
            raise ValueError(f"no SPICE subcircuit for {inst.cell}")
        args = []
        for p in ports[inst.cell]:
            if p in supply:
                args.append(supply[p])
            elif p in inst.conns:
                args.append(_node(inst.conns[p]))
                nets.append(inst.conns[p])
            else:
                args.append(_node(f"$nc_{inst.name}_{p}"))
        lines.append(f"X{inst.name} {' '.join(args)} {inst.cell}")
    for n, v in nl.consts.items():
        lines.append(f"V{_node(n)} {_node(n)} 0 {vdd * v}")
        nets.append(n)
    ev = {}
    for t, n, v in sorted(stimulus):
        ev.setdefault(n, []).append((t, v))
    for n in nl.inputs:
        v0 = int(init.get(n, 0))
        pts = [(0.0, v0 * vdd)]
        level = v0
        for t, v in ev.get(n, []):
            if v == level:
                continue
            ramp = input_slew / 0.6
            t0 = t - ramp / 2
            if t0 <= pts[-1][0]:
                raise ValueError(f"input {n}: transitions closer than the input slew")
            pts += [(t0, level * vdd), (t0 + ramp, v * vdd)]
            level = v
        pwl = " ".join(f"{t:.6e} {x:.4f}" for t, x in pts)
        lines.append(f"V{_node(n)} {_node(n)} 0 PWL({pwl})")
        nets.append(n)
    for n, c in (loads or {}).items():
        lines.append(f"C{_node(n)} {_node(n)} 0 {c:.6e}")
    nets = list(dict.fromkeys(nets))
    ics = " ".join(f"v({_node(n)})={vdd * int(init.get(n, 0))}" for n in nets if n not in nl.consts)
    if ics:
        lines.append(f".ic {ics}")
    # no `uic`: the initial operating point is solved with the .ic values as constraints, so internal cell nodes
    # (e.g. the middle of a NAND's NMOS stack) start consistent
    lines.append(f".tran {step:.3e} {T:.6e}")
    vecs = [f"v({_node(n)})" for n in nets]
    if supply_current:
        vecs.append("i(vdd)")
        nets = nets + [SUPPLY]
    lines += [".control", "set ngbehavior=hsa", "set ng_nomodcheck", "option klu", "run",
              "wrdata out.txt " + " ".join(vecs), "quit", ".endc", ".end"]
    return "\n".join(lines) + "\n", nets


def supply_energy(t: np.ndarray, V: np.ndarray, nets: list[str], vdd: float, t0: float, t1: float) -> float:
    """Energy drawn from the VDD source over [t0, t1] in joules (trapezoidal integration of -vdd * i(vdd) over the
    simulator's time points, with the end points interpolated). Needs to_spice(..., supply_current=True)."""
    i = V[:, nets.index(SUPPLY)]
    p = -vdd * i  # ngspice: positive source current flows from + through the source, so a supply delivers i < 0
    inner = (t > t0) & (t < t1)
    ts = np.concatenate([[t0], t[inner], [t1]])
    ps = np.concatenate([[np.interp(t0, t, p)], p[inner], [np.interp(t1, t, p)]])
    return float(np.sum(0.5 * (ps[1:] + ps[:-1]) * np.diff(ts)))


def run(deck: str, nets: list[str], workdir=None, timeout: float = 3600):
    """Run ngspice in batch mode. Returns (t [s], V [len(t), len(nets)]). Raises if the transient stopped early."""
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        (Path(d) / "deck.sp").write_text(deck)
        r = subprocess.run([str(NGSPICE), "-b", "deck.sp"], cwd=d, capture_output=True, text=True, timeout=timeout)
        out = Path(d) / "out.txt"
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"ngspice failed ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        data = np.loadtxt(out)
    # wrdata writes (t, v) column pairs per vector
    data = data.reshape(len(data), -1)
    t = data[:, 0]
    m = re.search(r"\.tran\s+\S+\s+(\S+)", deck)
    if m and t[-1] < 0.999 * float(m.group(1)):
        raise RuntimeError(f"ngspice transient stopped at {t[-1]:.3e} s:\n{r.stdout[-3000:]}")
    V = data[:, 1::2]
    if V.shape[1] != len(nets):
        raise RuntimeError("unexpected ngspice output layout")
    return t, V


def digitize(t: np.ndarray, V: np.ndarray, vdd: float, nets: list[str], threshold: float = 0.5,
             settle: float = 5e-12):
    """Crossings of threshold * vdd (linear interpolation between samples), as rows (time_s, net name, value),
    ordered by time. Crossings before `settle` (the .ic start-up transient) are ignored. A crossing back and forth
    between two samples is not resolved: use a fine enough time step."""
    thr = threshold * vdd
    rows = []
    t_ok = t >= settle
    for j, n in enumerate(nets):
        if n == SUPPLY:
            continue
        x = np.where(t_ok, V[:, j], V[np.argmax(t_ok), j]) - thr
        s = x > 0
        idx = np.nonzero(s[1:] != s[:-1])[0]
        for i in idx:
            frac = x[i] / (x[i] - x[i + 1])
            rows.append((float(t[i] + frac * (t[i + 1] - t[i])), n, int(s[i + 1])))
    rows.sort(key=lambda r: r[0])
    return rows
