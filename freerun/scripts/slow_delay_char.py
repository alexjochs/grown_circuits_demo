#!/usr/bin/env python
"""Characterize slow delay elements in SKY130 with ngspice (docs/slow_delay.md).

Elements under test, `cs`: a current-starved inverter (the classic starved-inverter delay element: Mahapatra,
Khatri & Mahapatra, "Comparison and analysis of delay elements", MWSCAS 2002; Jovanovic & Stojcev, "Current starved
delay element with symmetric load", Int. J. Electronics 93(3), 2006) followed by a standard inv_1 that restores the
edge, so the element is a non-inverting buffer:

    bias: Iref (ideal current reference, drawn from VDD) -> diode NMOS MNB (W/L 1/2) -> Vbn
          MN2 (1/2, gate Vbn) mirrors it into diode PMOS MPB (2/2, hvt) -> Vbp
    stage: VDD - MPS (2/2, gate Vbp) - MPW (1/0.15, gate in) - x - MNW (1/0.15, gate in) - MNS (1/2, gate Vbn) - GND
           x carries C_L (a MIM capacitor in silicon; an ideal capacitor here) plus the devices' parasitics
    restore: sky130_fd_sc_hd__inv_1 (x -> out), loaded by one more inv_1 (a realistic fanout)

Each element carries its own bias branch (2 x Iref from VDD): the static power is honest per element. Knobs: Iref
(the bias) and C_L. Standard-cell reference points (`cell`): sky130_fd_sc_hd dlygate4sd{1,2,3}_1, dlymetal6s{2,4,6}s_1
and buf_1, each driving one inv_1.

Measured per point: delay in -> out for a rising and a falling input (50 % crossings), the output's 20-80 % slew and
the slow node's, dynamic energy per output rise / fall (supply-current integral over the transition minus the static
power), static power with the input low and high (bias branches + leakage), and, where asked, the minimum input pulse
width that propagates (a full output transition) for positive and negative pulses (bisection).

    nice -n 10 .venv/bin/python scripts/slow_delay_char.py sweep --out runs/slow_delay [--workers 6] [--quick]
    .venv/bin/python scripts/slow_delay_char.py report runs/slow_delay   # Markdown tables + fit
    .venv/bin/python scripts/slow_delay_char.py report docs/data/slow_delay_points.json   # the same from the committed points

Results are cached per point in <out>/points/<key>.json, so an interrupted sweep resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(os.environ.get("FREERUN_TOOLS", ROOT / "tools"))
NGSPICE = TOOLS / "ngspice/bin/ngspice"
PDK = TOOLS / "pdk/sky130A"
CELLS = PDK / "libs.ref/sky130_fd_sc_hd/spice/sky130_fd_sc_hd.spice"

# the E2 corner box extremes (grown.sim.devices PRESETS["corners"]) plus nominal
CONDITIONS = [("tt", 25.0, 1.80), ("ss", 100.0, 1.40), ("ss", -40.0, 1.60), ("tt", 100.0, 1.80),
              ("ff", -40.0, 1.95), ("ff", 100.0, 1.65)]
BIASES = [3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 3e-9, 1e-9, 3e-10, 1e-10]
LOADS = [0.0, 1e-13]
REF_CELLS = ["sky130_fd_sc_hd__buf_1", "sky130_fd_sc_hd__dlygate4sd1_1", "sky130_fd_sc_hd__dlygate4sd2_1",
             "sky130_fd_sc_hd__dlygate4sd3_1", "sky130_fd_sc_hd__dlymetal6s2s_1", "sky130_fd_sc_hd__dlymetal6s4s_1",
             "sky130_fd_sc_hd__dlymetal6s6s_1"]
KINDS = ["cs", "cs2", "cs_st"]
C_PAR = 4e-15      # rough parasitic capacitance of the slow node (for time-window estimates only)
SLEW_IN = 50e-12   # input 20-80 % slew (s)


def est_delay(p: dict) -> float:
    if p["kind"] == "cell":
        return {"buf": 1e-10, "dlygate4sd1": 3e-10, "dlygate4sd2": 6e-10, "dlygate4sd3": 9e-10}.get(
            p["cell"].split("__")[1].rsplit("_", 1)[0], 1e-9)
    return max((p["C"] + C_PAR) * 0.9 / p["Ib"], 5e-11)


def deck(p: dict, wave: list, T: float, tstep: float) -> str:
    """wave: [(t, level)] input PWL points (volts are level * vdd)."""
    corner, temp, vdd = p["corner"], p["temp"], p["vdd"]
    L = [f"* slow delay characterization {json.dumps(p)}",
         f'.lib "{PDK}/libs.tech/ngspice/sky130.lib.spice" {corner}', f'.include "{CELLS}"',
         ".option scale=1e-6", f".temp {temp}", f"Vdd vpwr 0 {vdd}"]
    if p["kind"] in ("cs", "cs_st", "cs2"):
        L += [f"Iref vpwr vbn DC {p['Ib']:.6e}",
              "XMNB vbn vbn 0 0 sky130_fd_pr__nfet_01v8 w=1 l=2",
              "XMN2 vbp vbn 0 0 sky130_fd_pr__nfet_01v8 w=1 l=2",
              "XMPB vbp vbp vpwr vpwr sky130_fd_pr__pfet_01v8_hvt w=2 l=2",
              "XMPS p1 vbp vpwr vpwr sky130_fd_pr__pfet_01v8_hvt w=2 l=2",
              "XMPW x in p1 vpwr sky130_fd_pr__pfet_01v8_hvt w=1 l=0.15",
              "XMNW x in n1 0 sky130_fd_pr__nfet_01v8 w=1 l=0.15",
              "XMNS n1 vbn 0 0 sky130_fd_pr__nfet_01v8 w=1 l=2"]
        if p["kind"] == "cs":
            L.append("XR x 0 0 vpwr vpwr out sky130_fd_sc_hd__inv_1")
        elif p["kind"] == "cs2":        # a second starved stage (same bias, no added load), then inv_1 back to out
            L += ["XMPS2 p2 vbp vpwr vpwr sky130_fd_pr__pfet_01v8_hvt w=2 l=2",
                  "XMPW2 x2 x p2 vpwr sky130_fd_pr__pfet_01v8_hvt w=1 l=0.15",
                  "XMNW2 x2 x n2 0 sky130_fd_pr__nfet_01v8 w=1 l=0.15",
                  "XMNS2 n2 vbn 0 0 sky130_fd_pr__nfet_01v8 w=1 l=2",
                  "XR2 x2 0 0 vpwr vpwr xb sky130_fd_sc_hd__inv_1",
                  "XR3 xb 0 0 vpwr vpwr out sky130_fd_sc_hd__inv_1"]
        else:                       # 6-transistor Schmitt trigger restorer (inverting), then inv_1 back to out
            L += ["XSP1 sa x vpwr vpwr sky130_fd_pr__pfet_01v8_hvt w=0.84 l=0.15",
                  "XSP2 xs x sa vpwr sky130_fd_pr__pfet_01v8_hvt w=0.84 l=0.15",
                  "XSP3 sa xs 0 vpwr sky130_fd_pr__pfet_01v8_hvt w=0.42 l=0.15",
                  "XSN1 xs x sb 0 sky130_fd_pr__nfet_01v8 w=0.42 l=0.15",
                  "XSN2 sb x 0 0 sky130_fd_pr__nfet_01v8 w=0.42 l=0.15",
                  "XSN3 sb xs vpwr 0 sky130_fd_pr__nfet_01v8 w=0.42 l=0.15",
                  "XR2 xs 0 0 vpwr vpwr xb sky130_fd_sc_hd__inv_1",
                  "XR3 xb 0 0 vpwr vpwr out sky130_fd_sc_hd__inv_1"]
        if p["C"] > 0:
            L.append(f"Cl x 0 {p['C']:.6e}")
        if p.get("noise_na"):        # white current noise on the slow node (jitter check): rms na, sample period nt
            L.append(f"In x 0 DC 0 TRNOISE({p['noise_na']:.6e} {p['noise_nt']:.6e} 0 0)")
    else:
        L += [f"XD in 0 0 vpwr vpwr out {p['cell']}", "Vx x 0 0"]
    L.append("XF out 0 0 vpwr vpwr o2 sky130_fd_sc_hd__inv_1")
    pts = " ".join(f"{t:.9e} {lv * vdd:.4f}" for t, lv in wave)
    L.append(f"Vin in 0 PWL({pts})")
    seed = [f"set rndseed={int(p['seed'])}"] if "seed" in p else []
    L += [f".tran {tstep:.3e} {T:.9e}", ".control", "set ngbehavior=hsa", "set ng_nomodcheck", *seed, "option klu",
          "run",
          "wrdata out.txt v(in) v(x) v(out) i(vdd)", "quit", ".endc", ".end"]
    return "\n".join(L) + "\n"


def simulate(p, wave, T, tstep):
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "deck.sp").write_text(deck(p, wave, T, tstep))
        r = subprocess.run([str(NGSPICE), "-b", "deck.sp"], cwd=d, capture_output=True, text=True, timeout=7200)
        out = Path(d) / "out.txt"
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"ngspice failed for {p}: {r.stdout[-2000:]} {r.stderr[-2000:]}")
        data = np.loadtxt(out)
    t = data[:, 0]
    if t[-1] < 0.999 * T:
        raise RuntimeError(f"transient stopped at {t[-1]:.3e} of {T:.3e} for {p}")
    return t, data[:, 1], data[:, 3], data[:, 5], data[:, 7]


def static_power(p: dict, level: int) -> float:
    """Supply power at the DC operating point with the input held at `level` (bias branches + leakage)."""
    d = deck(p, [(0.0, level)], 1e-9, 1e-9)
    d = d.replace(".tran 1.000e-09 1.000000000e-09", "").replace("run\nwrdata out.txt v(in) v(x) v(out) i(vdd)",
                                                                 "op\nprint i(vdd)")
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "deck.sp").write_text(d)
        r = subprocess.run([str(NGSPICE), "-b", "deck.sp"], cwd=td, capture_output=True, text=True, timeout=600)
    import re
    m = re.search(r"i\(vdd\)\s*=\s*([-+0-9.eE]+)", r.stdout)
    if not m:
        raise RuntimeError(f"no operating point for {p}: {r.stdout[-1500:]}")
    return -p["vdd"] * float(m.group(1))


def crossings(t, v, thr, t0, t1=np.inf):
    """Times (interpolated) where v crosses thr in (t0, t1), with direction +1 / -1."""
    m = (t > t0) & (t < t1)
    idx = np.nonzero(m[1:] & m[:-1] & ((v[1:] > thr) != (v[:-1] > thr)))[0]
    out = []
    for i in idx:
        f = (v[i] - thr) / (v[i] - v[i + 1])
        out.append((t[i] + f * (t[i + 1] - t[i]), 1 if v[i + 1] > v[i] else -1))
    return out


def integral(t, y, t0, t1):
    m = (t > t0) & (t < t1)
    ts = np.concatenate([[t0], t[m], [t1]])
    ys = np.concatenate([[np.interp(t0, t, y)], y[m], [np.interp(t1, t, y)]])
    return float(np.sum(0.5 * (ys[1:] + ys[:-1]) * np.diff(ts)))


def ramp(t, level_from, level_to):
    r = SLEW_IN / 0.6
    return [(t - r / 2, level_from), (t + r / 2, level_to)]


def edges_point(p: dict) -> dict:
    """Delays, slews, energies and static power for one rising and one falling input edge."""
    vdd = p["vdd"]
    d = est_delay(p)
    q = max(3 * d, 2e-9)           # settling before the first edge (static power comes from .op, not from here)
    span = max(4 * d, 2e-9)        # time between the edges (the output must switch within it, else retried x10)
    for _ in range(4):
        t1 = q
        t2 = t1 + span
        T = t2 + span
        wave = [(0.0, 0)] + ramp(t1, 0, 1) + ramp(t2, 1, 0)
        tstep = max(d / 400, 1e-12)  # print / maximum step; ngspice's error control refines fast edges
        t, vin, vx, vo, i = simulate(p, wave, T, tstep)
        cr = crossings(t, vo, 0.5 * vdd, t1 - 1e-12)
        up = [c for c, s in cr if s > 0 and c < t2]
        dn = [c for c, s in cr if s < 0 and c > t2]
        if up and dn:
            break
        span *= 10                  # leakage-dominated or slower than estimated: a longer window
    pw = -vdd * i                   # power drawn from the supply (ngspice source current is negative)
    res = dict(p, ok=bool(up and dn), window=span)
    # static power: the last 20 % of each quiet stretch (input low before t1, input high before t2, low at the end)
    P_low = integral(t, pw, 0.8 * t1, t1 - 2 * SLEW_IN) / (0.2 * t1 - 2 * SLEW_IN)
    P_high = integral(t, pw, t2 - 0.2 * span, t2 - 2 * SLEW_IN) / (0.2 * span - 2 * SLEW_IN)
    P_low_end = integral(t, pw, T - 0.2 * span, T) / (0.2 * span)
    res.update(P_static_low=static_power(p, 0), P_static_high=static_power(p, 1), P_tran_low=P_low,
               P_tran_high=P_high)
    P_high, P_low_end = res["P_static_high"], res["P_static_low"]
    if not res["ok"]:
        res.update(n_out_crossings=len(cr), x_final=float(vx[-1]), out_final=float(vo[-1]))
        return res
    tu, td = up[0], dn[0]
    res["d_rise"] = tu - t1          # output rise for an input rise (non-inverting element)
    res["d_fall"] = td - t2
    # output and slow-node 20-80 % slews
    for name, v, a, b in (("out", vo, t1, t2), ("x", vx, t1, t2)):
        for edge, lo, hi in (("rise", a, b), ("fall", b, T)):
            c20 = crossings(t, v, 0.2 * vdd, lo, hi)
            c80 = crossings(t, v, 0.8 * vdd, lo, hi)
            if c20 and c80:
                res[f"slew_{name}_{edge}"] = abs(c80[0][0] - c20[0][0])
    # energy per transition: supply energy over the edge's settling window minus the static power of the end state
    for edge, te, Ps_after, t_end in (("rise", t1, P_high, t2 - 0.25 * span), ("fall", t2, P_low_end, T - 0.25 * span)):
        a = te - 2 * SLEW_IN
        res[f"E_{edge}"] = integral(t, pw, a, t_end) - Ps_after * (t_end - a)
    return res


def pulse_propagates(p: dict, w: float, positive: bool) -> bool:
    vdd = p["vdd"]
    d = est_delay(p)
    q = max(3 * d, 2e-9)
    t1 = q
    T = t1 + w + max(4 * d, 3 * w, 2e-9)
    a, b = (0, 1) if positive else (1, 0)
    wave = [(0.0, a)] + ramp(t1, a, b) + ramp(t1 + w, b, a)
    tstep = max(d / 400, 1e-12)    # the pulse's PWL corners are breakpoints: ngspice steps onto them anyway
    t, vin, vx, vo, i = simulate(p, wave, T, tstep)
    return len(crossings(t, vo, 0.5 * vdd, t1 - 1e-12)) >= 1


def min_pulse(p: dict, d_ref: float, positive: bool, iters: int = 6) -> float:
    """Smallest input pulse width (between 50 % crossings) that makes the output move (bisection in log w)."""
    lo, hi = max(d_ref * 1e-3, 2 * SLEW_IN / 0.6), d_ref * 3
    if not pulse_propagates(p, hi, positive):
        return float("inf")
    if pulse_propagates(p, lo, positive):
        return lo
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        if pulse_propagates(p, mid, positive):
            hi = mid
        else:
            lo = mid
    return hi


def key(p):
    return hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]


def job(args):
    p, out, with_pulse = args
    path = Path(out) / "points" / f"{key(dict(p, pulse=with_pulse))}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        r = edges_point(p)
        if with_pulse and r.get("ok"):
            r["w_min_pos"] = min_pulse(p, r["d_rise"], True)
            r["w_min_neg"] = min_pulse(p, r["d_fall"], False)
    except Exception as e:  # pragma: no cover
        r = dict(p, ok=False, error=repr(e)[:500])
    r["pulse"] = with_pulse
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r))
    return r


def plan(quick: bool):
    biases = BIASES[::3] if quick else BIASES
    jobs = []
    for kind, C, Ib in [(k, C, Ib) for k in KINDS for C in LOADS for Ib in biases]:
        if C * 0.9 / Ib > 2e-4:     # beyond the 100 us target range
            continue
        if True:
            base = dict(kind=kind, Ib=Ib, C=C)
            for corner, temp, vdd in CONDITIONS:
                jobs.append((dict(base, corner=corner, temp=temp, vdd=vdd),
                             corner == "tt" and temp == 25.0 and kind != "cs_st" and Ib in BIASES[1::3]))
            for vdd in (1.71, 1.89):
                jobs.append((dict(base, corner="tt", temp=25.0, vdd=vdd), False))
            for temp in (0.0, 85.0):
                jobs.append((dict(base, corner="tt", temp=temp, vdd=1.80), False))
    for cell in REF_CELLS:
        base = dict(kind="cell", cell=cell)
        for corner, temp, vdd in CONDITIONS:
            jobs.append((dict(base, corner=corner, temp=temp, vdd=vdd), corner == "tt" and temp == 25.0))
        for vdd in (1.71, 1.89):
            jobs.append((dict(base, corner="tt", temp=25.0, vdd=vdd), False))
        for temp in (0.0, 85.0):
            jobs.append((dict(base, corner="tt", temp=temp, vdd=1.80), False))
    # slowest first so the pool's tail is short
    jobs.sort(key=lambda j: -est_delay(j[0]) * (10 if j[1] else 1))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--quick", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("out")
    r.add_argument("--export", default=None, help="write the element table (grown.sim.slow format) here")
    r.add_argument("--variant", default="cs2")
    a = ap.parse_args()
    if a.cmd == "sweep":
        jobs = plan(a.quick)
        print(f"{len(jobs)} points", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            for k, res in enumerate(ex.map(job, [(p, a.out, w) for p, w in jobs])):
                tag = res.get("cell") or f"Ib {res.get('Ib'):.0e} C {res.get('C'):.0e}"
                print(f"[{k + 1}/{len(jobs)}] {tag} {res['corner']} {res['temp']} {res['vdd']}: ok={res.get('ok')} "
                      f"d_rise={res.get('d_rise', float('nan')):.3e} d_fall={res.get('d_fall', float('nan')):.3e} "
                      f"P={res.get('P_static_low', float('nan')):.3e} {res.get('error', '')}", flush=True)
    else:
        from slow_delay_report import report
        report(a.out, a.export, a.variant)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
