"""Tables, fit and the element table for docs/slow_delay.md from slow_delay_char.py's points.

    .venv/bin/python scripts/slow_delay_char.py report runs/slow_delay [--export table.json --variant cs2]
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

NOM = ("tt", 25.0, 1.8)


def load(out: str) -> list:
    """The points of a sweep directory (OUT/points/*.json) or of one JSON list (docs/data/slow_delay_points.json)."""
    if out.endswith(".json"):
        return json.load(open(out))
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "points", "*.json")))]


def _key(r):
    return (r["kind"], r.get("cell"), r.get("Ib"), r.get("C"))


def _cond(r):
    return (r["corner"], float(r["temp"]), float(r["vdd"]))


def index(pts):
    by = {}
    for r in pts:
        by.setdefault(_key(r), {})
        prev = by[_key(r)].get(_cond(r))
        if prev is None or (r.get("pulse") and not prev.get("pulse")):
            by[_key(r)][_cond(r)] = r
    return by


def si(x, unit="", digits=3):
    if x is None or not np.isfinite(x):
        return "n/a"
    if x == 0:
        return f"0 {unit}".strip()
    p = {-15: "f", -12: "p", -9: "n", -6: "µ", -3: "m", 0: ""}
    e = int(np.floor(np.log10(abs(x)) / 3) * 3)
    e = min(max(e, -15), 0)
    return f"{x / 10 ** e:.{digits}g} {p[e]}{unit}".strip()


def fit_delay(rows):
    """d = a * (C + c_par) / I per direction over the ok nominal points of one variant (log least squares on a and
    c_par by a scan over c_par)."""
    out = {}
    for edge in ("d_rise", "d_fall"):
        I = np.array([r["Ib"] for r in rows])
        C = np.array([r["C"] for r in rows])
        d = np.array([r[edge] for r in rows])
        best = None
        for cp in np.geomspace(1e-16, 1e-13, 400):
            a = np.exp(np.mean(np.log(d * I / (C + cp))))
            err = np.sqrt(np.mean(np.log(a * (C + cp) / I / d) ** 2))
            if best is None or err < best[2]:
                best = (a, cp, err)
        out[edge] = dict(a_volt=best[0], c_par=best[1], rms_log=best[2])
    return out


def report(out: str, export: str | None = None, variant: str = "cs2"):
    pts = load(out)
    by = index(pts)
    lines = [f"points: {len(pts)}", ""]
    kinds = sorted({k[0] for k in by if k[0] != "cell"})
    # T1: nominal characterization per variant and load
    for kind in kinds:
        for C in sorted({k[3] for k in by if k[0] == kind}):
            rows = sorted([v[NOM] for k, v in by.items() if k[0] == kind and k[3] == C and NOM in v],
                          key=lambda r: -r["Ib"])
            if not rows:
                continue
            lines += [f"### {kind}, C_L = {si(C, 'F')} (tt, 25 C, 1.80 V)", "",
                      "| bias | d rise | d fall | E rise | E fall | static | out slew r/f | w_min+/d | w_min-/d |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                if not r.get("ok"):
                    lines.append(f"| {si(r['Ib'], 'A')} | no switch | | | | {si(r.get('P_static_low'), 'W')} | | | |")
                    continue
                wp = r.get("w_min_pos")
                wn = r.get("w_min_neg")
                lines.append(
                    f"| {si(r['Ib'], 'A')} | {si(r['d_rise'], 's')} | {si(r['d_fall'], 's')} | {si(r['E_rise'], 'J')} "
                    f"| {si(r['E_fall'], 'J')} | {si(r['P_static_low'], 'W')} | {si(r.get('slew_out_rise'), 's')} / "
                    f"{si(r.get('slew_out_fall'), 's')} | {wp / r['d_rise'] if wp else float('nan'):.2f} | "
                    f"{wn / r['d_fall'] if wn else float('nan'):.2f} |")
            lines.append("")
    # T2: reference cells
    lines += ["### Standard-cell reference points (tt, 25 C, 1.80 V; each drives one inv_1)", "",
              "| cell | d rise | d fall | E rise | E fall | leakage (in low / high) | w_min+/d | w_min-/d |",
              "|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(by.items(), key=lambda kv: (kv[1].get(NOM, {}).get("d_rise") or 0)):
        if k[0] != "cell" or NOM not in v or not v[NOM].get("ok"):
            continue
        r = v[NOM]
        wp, wn = r.get("w_min_pos"), r.get("w_min_neg")
        lines.append(f"| {k[1].split('__')[1]} | {si(r['d_rise'], 's')} | {si(r['d_fall'], 's')} | "
                     f"{si(r['E_rise'], 'J')} | {si(r['E_fall'], 'J')} | {si(r['P_static_low'], 'W')} / "
                     f"{si(r['P_static_high'], 'W')} | {wp / r['d_rise'] if wp else float('nan'):.2f} | "
                     f"{wn / r['d_fall'] if wn else float('nan'):.2f} |")
    lines.append("")
    # T3: sensitivities and corners
    lines += ["### Sensitivity and corners (delay ratio to nominal; energy and static ratio)", "",
              "| element | bias / C | dln d / dV | dln d / dT (1/K) | " +
              " | ".join(f"{c} {t:g} C {v:g} V" for c, t, v in
                         [("ss", 100.0, 1.4), ("ss", -40.0, 1.6), ("tt", 100.0, 1.8), ("ff", -40.0, 1.95),
                          ("ff", 100.0, 1.65)]) + " |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(by.items(), key=lambda kv: (kv[0][0], -(kv[0][2] or 0), kv[0][3] or 0, kv[0][1] or "")):
        if NOM not in v or not v[NOM].get("ok"):
            continue
        n = v[NOM]
        dn = 0.5 * (n["d_rise"] + n["d_fall"])

        def dm(c):
            r = v.get(c)
            return 0.5 * (r["d_rise"] + r["d_fall"]) if r and r.get("ok") else None
        sv = [dm(("tt", 25.0, 1.71)), dm(("tt", 25.0, 1.89))]
        st = [dm(("tt", 0.0, 1.8)), dm(("tt", 85.0, 1.8))]
        dv = (np.log(sv[1] / sv[0]) / 0.18) if all(sv) else float("nan")
        dt = (np.log(st[1] / st[0]) / 85.0) if all(st) else float("nan")
        cells = []
        for c in [("ss", 100.0, 1.4), ("ss", -40.0, 1.6), ("tt", 100.0, 1.8), ("ff", -40.0, 1.95), ("ff", 100.0, 1.65)]:
            r = v.get(c)
            if r and r.get("ok"):
                e = (r["E_rise"] + r["E_fall"]) / (n["E_rise"] + n["E_fall"])
                cells.append(f"{dm(c) / dn:.2f} / {e:.2f} / {r['P_static_low'] / n['P_static_low']:.2f}")
            else:
                cells.append("n/a" if not r else "no switch")
        name = k[1].split("__")[1] if k[0] == "cell" else k[0]
        bc = "" if k[0] == "cell" else f"{si(k[2], 'A')} / {si(k[3], 'F')}"
        lines.append(f"| {name} | {bc} | {dv:+.2f} | {dt:+.4f} | " + " | ".join(cells) + " |")
    lines.append("")
    # fit
    for kind in kinds:
        rows = [v[NOM] for k, v in by.items() if k[0] == kind and NOM in v and v[NOM].get("ok")]
        if len(rows) >= 3:
            f = fit_delay(rows)
            lines.append(f"fit {kind}: " + "; ".join(
                f"{e}: d = {f[e]['a_volt']:.3f} V x (C_L + {si(f[e]['c_par'], 'F')}) / I_bias (rms {100 * (np.exp(f[e]['rms_log']) - 1):.0f} %)"
                for e in f))
    text = "\n".join(lines)
    print(text)
    if export:
        export_table(by, variant, export)
    return text


def export_table(by, variant: str, path: str):
    """The grown element table: the nominal ok points of `variant` (both loads), one level per point, sorted by delay,
    keeping per delay the load with the lower power at a 50 % activity of one transition per delay (static + dynamic);
    corner ratios at the E2 box corners."""
    cand = []
    for k, v in by.items():
        if k[0] != variant or NOM not in v or not v[NOM].get("ok"):
            continue
        r = v[NOM]
        d = 0.5 * (r["d_rise"] + r["d_fall"])
        e = 0.5 * (r["E_rise"] + r["E_fall"])
        cand.append((d, r["P_static_low"] + e / d, k, v))
    conds = [("tt", 25.0, 1.8), ("ss", 100.0, 1.4), ("ss", -40.0, 1.6), ("tt", 100.0, 1.8), ("ff", -40.0, 1.95),
             ("ff", 100.0, 1.65)]

    def usable(c):
        """In the target range (>= 1 ns: below it the standard delay cells are as fast and cost picowatts) and not
        leakage-dominated at any corner (static power within 2x and delay within 1.3x of nominal; the hot fast corner
        is where leakage takes over at the lowest biases)."""
        d, p, k, v = c
        if d < 1e-9:
            return False
        n = v[NOM]
        for cd in conds:
            r = v.get(cd)
            if r is None or not r.get("ok"):
                return False
            if r["P_static_low"] > 2 * n["P_static_low"]:
                return False
            if (r["d_rise"] + r["d_fall"]) > 1.3 * (n["d_rise"] + n["d_fall"]) and r["corner"] == "ff":
                return False
        return True
    cand = [c for c in cand if usable(c)]
    cand.sort(key=lambda c: c[0])
    # per delay the cheaper design: drop a point if another within a factor 2 in delay draws less power at one
    # transition per delay (static + E / d): the two loads (0 and 100 fF) overlap in delay
    keep = [c for c in cand if not any(o is not c and 0.5 * c[0] <= o[0] <= 2.0 * c[0] and o[1] < c[1]
                                       for o in cand)]
    levels, rd, re, rs = [], [], [], []
    for d, p, k, v in keep:
        n = v[NOM]
        levels.append(dict(bias_a=k[2], cap_f=k[3], d_rise_s=n["d_rise"], d_fall_s=n["d_fall"], e_rise_j=n["E_rise"],
                           e_fall_j=n["E_fall"], p_static_w=n["P_static_low"],
                           jitter_rel=float(np.sqrt(1.602176634e-19 / (k[2] * d)))))
    for c in conds:
        row_d, row_e, row_s = [], [], []
        for d, p, k, v in keep:
            n, r = v[NOM], v.get(c)
            ok = r is not None and r.get("ok")
            row_d.append((r["d_rise"] + r["d_fall"]) / (n["d_rise"] + n["d_fall"]) if ok else float("nan"))
            row_e.append((r["E_rise"] + r["E_fall"]) / (n["E_rise"] + n["E_fall"]) if ok else float("nan"))
            row_s.append(r["P_static_low"] / n["P_static_low"] if ok else float("nan"))
        rd.append(row_d), re.append(row_e), rs.append(row_s)
    doc = dict(element=f"freerun docs/slow_delay.md, variant {variant}", source="scripts/slow_delay_char.py",
               nominal=list(NOM), levels=levels, conditions=[list(c) for c in conds],
               ratios=dict(delay=rd, energy=re, static=rs),
               jitter="sigma/d = sqrt(q / (I_bias d)): shot noise of the starving current (slow_delay_jitter.py)")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {path}: {len(levels)} levels, {si(levels[0]['d_rise_s'], 's')} .. {si(levels[-1]['d_rise_s'], 's')}")
