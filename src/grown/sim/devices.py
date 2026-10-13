"""Device populations and the idealized twin for search-time evaluation (E2, project.md question 2).

Our gates are lookup tables, not Liberty cells, so freerun's cell annotation does not apply directly. The
convention (E0, the cluster agent's): a circuit's own per-arc delays are its *nominal* delays, read as SKY130 hd
delays at 10 ps per tick (search.operator.delay_range [3, 12] = 30-120 ps), at the nominal corner (tt, 25 C, 1.80 V).

A *device* of a circuit is drawn the way freerun's `variation.vary()` varies a plain circuit (SPEC §12): a global
delay scale s = exp(scale_sigma z), per-gate rise and fall mismatch factors exp(mismatch_sigma z) (each gate is a
cell instance), delays rounded and clamped to >= 1, and per-transition jitter with sigma = jitter_rel x the gate's
mean (varied) arc delay. vary() ignores the corner; here the corner is drawn too (from condition boxes, as freerun's
`draw_conditions`) and enters as a delay factor folded into the scale: the geometric-mean arc-delay ratio of that
corner to nominal over SKY130 cells, interpolated in (T, V) exactly like LibraryFamily.weights (bilinear over the
smallest characterized rectangle; geometric for delays). The ratios come from sky130_lut.json
(scripts/grown_sky130_lut_calibration.py), so no PDK is needed at run time. Everything is drawn with freerun's
Philox streams (variation._words: corner stream 1, scale stream 2, mismatch stream 3 with the gate index), keyed by a
per-lane (seed, cid) that is a hash of (population seed, the lane's netlist, the lane's input stream): a lane's
device depends only on which circuit runs which episode, never on the batch it is in. With scale folded in, the
delays equal `vary(circuit, Conditions(scale=s * f_corner, mismatch_sigma, seed, cid, jitter_rel))` bit for bit
(tested).

The *idealized twin* of a lane is the circuit itself: nominal delays, no mismatch, no jitter, nominal corner.

`run_physical(lanes, spec)` runs every lane twice in one engine batch, as its twin and as its device. The search
reads the outputs of `spec.select` (twin or device) for its fitness; per lane it also gets (EvalResult.extra):
  gap_duty       mean |duty(device) - duty(twin)| over steps and output wires (the idealization gap)
  gap_disagree   fraction of (step, wire) samples where device and twin differ
  energy         joules per control step (per decision) of the selected lane; energy_twin, energy_device both
  log10_energy   log10 of `energy`
Energy (SPEC §11 with LUT constants): switching 1/2 C V^2 per gate-net transition (C = the fanout's pin
capacitances, plus one 1-input pin per circuit output), internal energy per output transition of the gate (per
arity), input-pin internal energy per transition of every net (inputs included), and leakage per gate over the
episode; capacitance, internal energies and leakage scaled by the corner's ratios, V the corner's supply.
energy: "approx" uses each lane's total gate-net transitions with the lane's mean per-net weights (no extra
engine I/O); "exact" reads every gate net's toggles per step (one output slot per gate) and weights each net by its
own capacitance and gate. `spec.run_devices = False` with select twin skips the device lanes (no gap extras).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from .stepper import LaneResult, LaneSet, _mix, run_lanes

M32 = 0xFFFFFFFF
U64 = np.uint64
VAR_KEY_TWEAK = 0x76617269
STREAM_CORNER, STREAM_SCALE, STREAM_MISMATCH = 1, 2, 3
MAX_DELAY = 2**31 - 1
JITTER_K = 2**31 * (3.0 / (2**32 - 1)) ** 0.5           # freerun.jitter.K: amplitude = rint(sigma * K)

PRESETS = {
    # freerun's idealization-gap example and grown.sim.physical's default
    "corners": (("ss", (-40.0, 100.0), (1.40, 1.60)), ("tt", (25.0, 100.0), 1.80), ("ff", (-40.0, 100.0), (1.65, 1.95))),
    "tt": (("tt", 25.0, 1.80),),                         # local variation only (scale, mismatch, jitter)
    "nominal": (("tt", 25.0, 1.80),),
}


@dataclass(frozen=True)
class PhysicalSpec:
    select: str = "device"          # whose outputs the search's fitness reads: "device" or "twin"
    seed: int = 1                   # population seed (a fresh population = another seed)
    preset: str = "corners"         # condition boxes (PRESETS) or boxes given explicitly
    boxes: tuple | None = None      # ((process, temp | (lo, hi), voltage | (lo, hi)[, weight]), ...)
    scale_sigma: float = 0.03
    mismatch_sigma: float = 0.05
    jitter_rel: float = 0.01
    tick: float = 1e-11             # seconds per substrate tick (10 ps: the SKY130 delay convention)
    energy: str = "approx"          # "approx" | "exact" | "off"
    run_devices: bool = True        # with select twin: also run the devices (gap extras)

    @staticmethod
    def from_config(d: dict) -> "PhysicalSpec":
        d = dict(d)
        if d.get("boxes") is not None:
            d["boxes"] = tuple(tuple(tuple(x) if isinstance(x, list) else x for x in b) for b in d["boxes"])
        known = PhysicalSpec.__dataclass_fields__
        bad = [k for k in d if k not in known]
        if bad:
            raise KeyError(f"unknown substrate.physical keys {bad} (known: {sorted(known)})")
        s = PhysicalSpec(**d)
        if s.select not in ("device", "twin"):
            raise ValueError("substrate.physical.select must be device or twin")
        if s.energy not in ("approx", "exact", "off"):
            raise ValueError("substrate.physical.energy must be approx, exact or off")
        if s.select == "device" and not s.run_devices:
            raise ValueError("select device needs run_devices")
        return s

    def box_list(self) -> list:
        raw = self.boxes if self.boxes is not None else PRESETS[self.preset]
        out = []
        for b in raw:
            p, t, v = b[0], b[1], b[2]
            w = float(b[3]) if len(b) > 3 else 1.0
            rng = lambda x: (float(x[0]), float(x[1])) if isinstance(x, (tuple, list)) else (float(x), float(x))  # noqa
            out.append((str(p), rng(t), rng(v), w, isinstance(t, (tuple, list)), isinstance(v, (tuple, list))))
        return out


# ---- calibration table ------------------------------------------------------------------------------------------------

_CAL = None


def calibration() -> dict:
    global _CAL
    if _CAL is None:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sky130_lut.json")) as f:
            _CAL = json.load(f)
    return _CAL


def arity_table(name: str) -> np.ndarray:
    """float64 [7]: a per-arity constant (index = arity; arity 0 = constant gate: 0)."""
    pa = calibration()["per_arity"]
    return np.array([0.0] + [pa[str(k)][name] if str(k) in pa else pa[max(pa, key=int)][name] for k in range(1, 7)])


def corner_factors(process, temp, volt) -> dict:
    """Ratios to nominal at corners (arrays of process labels, temperatures, voltages), interpolated as
    LibraryFamily.weights does: a characterized point exactly; else the smallest characterized (T, V) rectangle (or
    segment) containing the point, bilinear weights; delays and leakage geometric, capacitance and energies linear."""
    cal = calibration()["corners"]
    process = np.asarray(process)
    temp = np.asarray(temp, np.float64)
    volt = np.asarray(volt, np.float64)
    n = len(temp)
    keys = ("delay", "cap", "e_int", "e_in", "leak")
    out = {k: np.full(n, np.nan) for k in keys}
    for p in np.unique(process):
        pts = [c for c in cal if c["process"] == p]
        val = {(c["temp"], c["voltage"]): c for c in pts}
        have = set(val)
        Ts = sorted({t for t, _ in have})
        Vs = sorted({v for _, v in have})
        cands = []
        for i, t1 in enumerate(Ts):
            for t2 in Ts[i:]:
                for j, v1 in enumerate(Vs):
                    for v2 in Vs[j:]:
                        if {(t1, v1), (t1, v2), (t2, v1), (t2, v2)} <= have:
                            dt, dv = (t2 - t1) / 100.0, (v2 - v1) / 0.1
                            cands.append(((dt > 0 and dv > 0, dt * dv, dt + dv), t1, t2, v1, v2))
        cands.sort(key=lambda c: c[0])           # LibraryFamily.weights keeps the first minimum in this order
        m = np.flatnonzero(process == p)
        T, V = temp[m], volt[m]
        pick = np.full(len(m), -1)
        for ci in range(len(cands) - 1, -1, -1):  # last write wins: the earliest candidate in key order
            _, t1, t2, v1, v2 = cands[ci]
            pick[(T >= t1) & (T <= t2) & (V >= v1) & (V <= v2)] = ci
        for q in np.flatnonzero(pick < 0):         # a characterized point outside every full rectangle
            if (T[q], V[q]) not in have:
                raise ValueError(f"corner ({p}, {T[q]}, {V[q]}) is outside the characterized hull")
        for ci in np.unique(pick):
            r = pick == ci
            if ci < 0:
                for k in keys:
                    out[k][m[r]] = [val[(t, v)][k] for t, v in zip(T[r].tolist(), V[r].tolist())]
                continue
            _, t1, t2, v1, v2 = cands[ci]
            u = np.zeros(r.sum()) if t2 == t1 else (T[r] - t1) / (t2 - t1)
            w = np.zeros(r.sum()) if v2 == v1 else (V[r] - v1) / (v2 - v1)
            wts = ((1 - u) * (1 - w), u * (1 - w), (1 - u) * w, u * w)
            cs = ((t1, v1), (t2, v1), (t1, v2), (t2, v2))
            for k in keys:
                f = [val[c][k] for c in cs]
                if k in ("delay", "leak"):
                    out[k][m[r]] = np.exp(sum(wt * np.log(fv) for wt, fv in zip(wts, f)))
                else:
                    out[k][m[r]] = sum(wt * fv for wt, fv in zip(wts, f))
    out["voltage"] = volt
    return out


# ---- Philox (freerun.jitter.philox_np with one key per row) ----------------------------------------------------------

def philox(ctr: np.ndarray, k0: np.ndarray, k1: np.ndarray) -> np.ndarray:
    """Philox4x32-10 with per-row keys: ctr uint64 [n, 4] (32-bit values), k0 / k1 uint64 [n]. -> uint64 [n, 4]."""
    m32 = U64(M32)
    s32 = U64(32)
    c = np.asarray(ctr, U64) & m32
    c0, c1, c2, c3 = (c[:, i].copy() for i in range(4))
    k0 = np.asarray(k0, U64) & m32
    k1 = np.asarray(k1, U64) & m32
    for r in range(10):
        if r:
            k0 = (k0 + U64(0x9E3779B9)) & m32
            k1 = (k1 + U64(0xBB67AE85)) & m32
        p0 = U64(0xD2511F53) * c0
        p1 = U64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = ((p1 >> s32) ^ c1 ^ k0) & m32, p1 & m32, ((p0 >> s32) ^ c3 ^ k1) & m32, p0 & m32
    return np.stack([c0, c1, c2, c3], axis=-1)


def _words(seed, cid, stream: int, index) -> np.ndarray:
    """freerun.variation._words for many (seed, cid, index) rows at once."""
    seed = np.asarray(seed, U64)
    cid = np.asarray(cid, U64)
    index = np.asarray(index, U64)
    seed, cid, index = np.broadcast_arrays(seed, cid, index)
    ctr = np.stack([cid & U64(M32), np.full(cid.shape, stream, U64), index, np.zeros(cid.shape, U64)], -1)
    return philox(ctr.reshape(-1, 4), (seed & U64(M32)).reshape(-1),
                  ((seed >> U64(32)) & U64(M32)).reshape(-1) ^ U64(VAR_KEY_TWEAK))


def _unit(w):
    return (np.asarray(w, np.float64) + 0.5) / 2.0**32


def _normal_pair(w0, w1):
    r = np.sqrt(-2.0 * np.log(_unit(w0)))
    th = 2.0 * np.pi * _unit(w1)
    return r * np.cos(th), r * np.sin(th)


# ---- lane keys --------------------------------------------------------------------------------------------------------

def _event_stream_hash(ev) -> np.ndarray:
    """uint64 per stream of an EventStreams: blake2b of its (t, wire, value) events."""
    out = np.empty(len(ev), U64)
    for e in range(len(ev)):
        s = slice(int(ev.off[e]), int(ev.off[e + 1]))
        h = hashlib.blake2b(digest_size=8)
        for a in (np.asarray(ev.t[s], np.int64), np.asarray(ev.wire[s], np.int32), np.asarray(ev.v[s], np.uint8)):
            h.update(np.ascontiguousarray(a).tobytes())
        out[e] = np.frombuffer(h.digest(), U64)[0]
    return out


def lane_keys(lanes: LaneSet, pop_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(seed uint64 [L], cid uint32 [L]) of every lane: a hash of (population seed, the lane's netlist arrays with
    its nominal delays, the lane's per-tick input stream). Independent of batch composition and lane order."""
    L = len(lanes)
    hc = np.empty(L, U64)
    for b in range(L):
        g0, g1 = int(lanes.gate_off[b]), int(lanes.gate_off[b + 1])
        e0, e1 = int(lanes.edge_off[b]), int(lanes.edge_off[b + 1])
        n0 = int(lanes.net_off[b]) + int(lanes.n_inputs[b])
        h = hashlib.blake2b(digest_size=8)
        h.update(np.array([lanes.n_inputs[b], g1 - g0, e1 - e0, lanes.n_out[b]], np.int64).tobytes())
        for a in (lanes.fanin_ptr[g0 + b:g1 + b + 1], lanes.fanin[e0:e1], lanes.dr[e0:e1], lanes.df[e0:e1],
                  lanes.tt[g0:g1], lanes.init[n0:n0 + g1 - g0], lanes.outs[b, :int(lanes.n_out[b])]):
            h.update(np.ascontiguousarray(a).tobytes())
        hc[b] = np.frombuffer(h.digest(), U64)[0]
    hs = np.empty(L, U64)
    for j, ev in enumerate(lanes.ev):
        m = np.flatnonzero(lanes.job == j)
        if len(m):
            hs[m] = _event_stream_hash(ev)[lanes.row[m]]
    with np.errstate(over="ignore"):
        k = _mix(hc ^ _mix(hs ^ _mix(np.full(L, U64(int(pop_seed) & 0xFFFFFFFFFFFFFFFF)))))
        cid = (_mix(k ^ U64(0x5DEECE66D)) & U64(M32)).astype(np.uint32)
    return k, cid


# ---- device draws -----------------------------------------------------------------------------------------------------

def draw(lanes: LaneSet, spec: PhysicalSpec, seed=None, cid=None) -> dict:
    """Per-lane conditions (freerun.variation.draw_conditions, vectorized): process, temp, voltage, scale, and the
    corner's ratios to nominal."""
    if seed is None:
        seed, cid = lane_keys(lanes, spec.seed)
    boxes = spec.box_list()
    L = len(seed)
    w = _words(seed, cid, STREAM_CORNER, 0)
    p = np.cumsum([b[3] for b in boxes], dtype=np.float64)
    u0 = _unit(w[:, 0]) * p[-1]
    bi = np.minimum(np.searchsorted(p, u0, side="right"), len(boxes) - 1)
    temp = np.empty(L)
    volt = np.empty(L)
    proc = np.empty(L, dtype=object)
    for i, (pr, (t0, t1), (v0, v1), _, tr, vr) in enumerate(boxes):
        m = bi == i
        proc[m] = pr
        temp[m] = np.round(t0 + (t1 - t0) * _unit(w[m, 1]), 9) if tr else t0
        volt[m] = np.round(v0 + (v1 - v0) * _unit(w[m, 2]), 9) if vr else v0
    z, _ = _normal_pair(*_words(seed, cid, STREAM_SCALE, 0)[:, :2].T)
    scale = np.exp(spec.scale_sigma * z)
    f = corner_factors(proc.astype(str), temp, volt)
    f.update(seed=seed, cid=cid, process=proc.astype(str), temp=temp, voltage=volt, scale=scale)
    return f


def apply_devices(lanes: LaneSet, idx, cond: dict, spec: PhysicalSpec) -> None:
    """Vary lanes idx of `lanes` in place (delays, jitter) with the conditions cond (from `draw`, one row per idx):
    freerun.variation.vary with scale = scale x corner delay ratio. Slow delay elements (grown.sim.slow, recognized
    by their nominal delays) take their own corner delay ratio from the SPICE table instead of the gates' and
    jitter sigma = their table's relative jitter x delay instead of spec.jitter_rel."""
    from .slow import lane_slow_levels, table as slow_table
    idx = np.asarray(idx, np.int64)
    s_eff = cond["scale"] * cond["delay"]
    lev = lane_slow_levels(lanes)                      # before any delay is varied
    gidx = np.arange(int(lanes.gate_off[-1]))
    gl = lanes.gate_lane()
    k = lanes.arity
    eg = lanes.edge_gate()
    el = lanes.edge_lane()
    pos = np.full(len(lanes), -1, np.int64)
    pos[idx] = np.arange(len(idx))
    sel_g = np.flatnonzero(pos[gl] >= 0)
    local = gidx[sel_g] - lanes.gate_off[gl[sel_g]]
    row = pos[gl[sel_g]]
    if spec.mismatch_sigma != 0.0 and len(sel_g):
        w = _words(cond["seed"][row], cond["cid"][row], STREAM_MISMATCH, local)
        zr, zf = _normal_pair(w[:, 0], w[:, 1])
        mr = np.ones(int(lanes.gate_off[-1]))
        mf = np.ones(int(lanes.gate_off[-1]))
        mr[sel_g] = np.exp(spec.mismatch_sigma * zr)
        mf[sel_g] = np.exp(spec.mismatch_sigma * zf)
    else:
        mr = mf = np.ones(int(lanes.gate_off[-1]))
    sel_e = np.flatnonzero(pos[el] >= 0)
    se = s_eff[pos[el[sel_e]]]
    slow_e = lev[eg[sel_e]] >= 0
    if slow_e.any():                                   # slow elements: their own corner delay ratio
        t = slow_table()
        ci = t.condition_index(cond["process"], cond["temp"], cond["voltage"])
        rows = pos[el[sel_e]][slow_e]
        r = (t.r_delay[ci[rows], lev[eg[sel_e]][slow_e]] if t.r_delay is not None and (ci >= 0).all()
             else np.ones(int(slow_e.sum())))
        se = se.copy()
        se[slow_e] = cond["scale"][rows] * r
    rnd = lambda x: np.clip(np.rint(x), 1, MAX_DELAY).astype(np.int64)  # noqa: E731
    lanes.dr[sel_e] = rnd(lanes.dr[sel_e].astype(np.float64) * se * mr[eg[sel_e]]).astype(np.int32)
    lanes.df[sel_e] = rnd(lanes.df[sel_e].astype(np.float64) * se * mf[eg[sel_e]]).astype(np.int32)
    if spec.jitter_rel > 0:
        Gt = int(lanes.gate_off[-1])
        if lanes.jitter is None:
            lanes.jitter = dict(seed=np.zeros(len(lanes), U64), cid=np.zeros(len(lanes), np.uint32),
                                dmin=np.zeros(len(lanes), np.int32), amp=np.zeros(Gt, np.int64))
        tot = np.bincount(eg[sel_e], (lanes.dr[sel_e].astype(np.float64) + lanes.df[sel_e].astype(np.float64)),
                          minlength=Gt)
        kk = k.astype(np.float64)
        mean = np.divide(tot, 2 * kk, out=np.zeros(Gt), where=kk > 0)
        sig = spec.jitter_rel * mean
        if (lev >= 0).any():                           # slow elements: their physical relative jitter
            sig = np.where(lev >= 0, slow_table().jitter_rel[np.maximum(lev, 0)] * mean, sig)
        amp = np.rint(sig * JITTER_K).astype(np.int64)
        lanes.jitter["amp"][sel_g] = amp[sel_g]
        lanes.jitter["seed"][idx] = cond["seed"]
        lanes.jitter["cid"][idx] = cond["cid"]
        lanes.jitter["dmin"][idx] = 1                      # vary(): min(jitter_dmin = 1, the minimum delay)


# ---- energy -----------------------------------------------------------------------------------------------------------

def lane_energy(lanes: LaneSet, res: LaneResult, cond: dict | None, tick: float, exact: bool,
                static_out: np.ndarray | None = None, levels: np.ndarray | None = None) -> np.ndarray:
    """Joules per control step of every lane (module docstring). cond: per-lane corner ratios and voltage (None:
    nominal). Slow delay elements (`levels`, grown.sim.slow.lane_slow_levels of the nominal lanes) count their
    table's energy per output transition (instead of the gate terms) and their static power over the episode; the
    static part per step is also written to `static_out` [L] when given. Lanes with slow elements need exact
    (per-gate) toggles: their transitions cost ~1e4 times a gate's."""
    L = len(lanes)
    nom_v = float(calibration()["nominal"]["voltage"])
    one = np.ones(L)
    V = cond["voltage"] if cond is not None else one * nom_v
    fc, fi, fn, fl = ((cond[k] if cond is not None else one) for k in ("cap", "e_int", "e_in", "leak"))
    cap_k, eint_k, ein_k, leak_k = (arity_table(n) for n in ("pin_cap", "e_int", "e_in", "leak"))
    k = lanes.arity
    eg = lanes.edge_gate()
    el = lanes.edge_lane()
    gl = lanes.gate_lane()
    N = int(lanes.net_off[-1])
    reader_k = k[eg]
    net = lanes.net_off[el] + lanes.fanin                       # global net of each arc's source
    C = np.bincount(net, cap_k[reader_k], minlength=N)
    Ein = np.bincount(net, ein_k[reader_k], minlength=N)
    ob = np.flatnonzero(lanes.outs.reshape(-1) >= 0)            # circuit outputs: one 1-input pin of load each
    ol = ob // lanes.outs.shape[1]
    np.add.at(C, lanes.net_off[ol] + lanes.outs.reshape(-1)[ob], cap_k[1])
    gnet = lanes.net_off[gl] + lanes.n_inputs[gl] + (np.arange(len(gl)) - lanes.gate_off[gl])
    w_g = 0.5 * C[gnet] * (V[gl] ** 2) * fc[gl] + eint_k[k] * fi[gl] + Ein[gnet] * fn[gl]   # J per transition
    static = np.zeros(L)
    if levels is not None and (levels >= 0).any():
        from .slow import table as slow_table
        t = slow_table()
        sg = np.flatnonzero(levels >= 0)
        lv = levels[sg]
        r_e = r_s = np.ones(len(sg))
        if cond is not None and t.r_energy is not None:
            ci = t.condition_index(cond["process"], cond["temp"], cond["voltage"])
            r_e, r_s = t.r_energy[ci[gl[sg]], lv], t.r_static[ci[gl[sg]], lv]
        w_g = w_g.copy()
        w_g[sg] = 0.5 * (t.e_rise_j[lv] + t.e_fall_j[lv]) * r_e
        dur_g = (lanes.S[gl[sg]] * lanes.K[gl[sg]] + 1) * tick
        static = np.bincount(gl[sg], t.p_static_w[lv] * r_s * dur_g, minlength=L)
        if not (exact and res.net_toggles is not None):
            raise ValueError("lanes with slow delay elements need exact energy (per-gate toggles)")
    if exact and res.net_toggles is not None:
        e_tog = np.bincount(gl, w_g * res.net_toggles, minlength=L)
    else:
        G = np.maximum(lanes.n_gates, 1)
        e_tog = res.toggles * np.bincount(gl, w_g, minlength=L) / G
    # input nets: input-pin internal energy per change, from the streams
    e_in = np.zeros(L)
    for j, ev in enumerate(lanes.ev):
        m = np.flatnonzero(lanes.job == j)
        if not len(m):
            continue
        I = int(lanes.n_inputs[m[0]])
        e, _, w, _ = ev.effective(I, int((lanes.S[m] * lanes.K[m]).max()))
        ch = np.bincount(e * I + w, minlength=len(ev) * I).reshape(len(ev), I)[lanes.row[m]]  # [Lj, I]
        inet = lanes.net_off[m][:, None] + np.arange(I)[None, :]
        e_in[m] = (ch * Ein[inet]).sum(1) * fn[m]
    dur = (lanes.S * lanes.K + 1) * tick
    leak_g = leak_k[k] * fl[gl]
    if levels is not None:
        leak_g = np.where(levels >= 0, 0.0, leak_g)      # a slow element's leakage is in its static power
    leak = np.bincount(gl, leak_g, minlength=L) * dur
    if static_out is not None:
        static_out[:] = static / np.maximum(lanes.S, 1)
    return (e_tog + e_in + leak + static) / np.maximum(lanes.S, 1)


# ---- the evaluation -------------------------------------------------------------------------------------------------

def run_physical(lanes: LaneSet, spec: PhysicalSpec, backend: str = "cpu", threads: int = 0, timing=None,
                 sim_kw: dict | None = None, multistep: bool | None = None):
    """Every lane as twin and as device in one batch. -> (LaneResult of the selected lanes, extra dict [L])."""
    tm = timing if timing is not None else {}
    t0 = time.perf_counter()
    L = len(lanes)
    with_dev = spec.run_devices
    both = lanes.select(np.concatenate([np.arange(L), np.arange(L)])) if with_dev else lanes
    cond = None
    if with_dev:
        seed, cid = lane_keys(lanes, spec.seed)
        cond = draw(lanes, spec, seed, cid)
        apply_devices(both, np.arange(L, 2 * L), cond, spec)
    tm["devices"] = tm.get("devices", 0.0) + time.perf_counter() - t0
    from .slow import lane_slow_levels
    levels = lane_slow_levels(lanes)
    exact = spec.energy == "exact" or bool((levels >= 0).any())   # slow elements: per-gate toggles
    r = run_lanes(both, backend, threads, net_toggles=exact, timing=tm, sim_kw=sim_kw, multistep=multistep)
    t1 = time.perf_counter()
    tw = slice(0, L)
    dv = slice(L, 2 * L) if with_dev else None
    sel = dv if spec.select == "device" else tw
    nt = r.net_toggles
    G = int(lanes.gate_off[-1])

    def part(s, half):
        return LaneResult(r.outputs[s], r.high[s], r.toggles[s],
                          None if nt is None else nt[half * G:(half + 1) * G], r.calls,
                          None if r.events is None else r.events[s],
                          None if r.over_budget is None else r.over_budget[s])
    extra = {}
    if with_dev:
        mask = np.zeros(r.outputs[tw].shape, bool)
        for b in range(L):
            mask[b, :int(lanes.S[b]), :int(lanes.n_out[b])] = True
        cnt = np.maximum(mask.sum((1, 2)), 1)
        extra["gap_duty"] = (np.abs(r.high[dv] - r.high[tw]) * mask).sum((1, 2)) / cnt
        extra["gap_disagree"] = ((r.outputs[dv] != r.outputs[tw]) & mask).sum((1, 2)) / cnt
    if spec.energy != "off":
        st_t, st_d = np.zeros(L), np.zeros(L)
        et = lane_energy(lanes, part(tw, 0), None, spec.tick, exact, st_t, levels)
        extra["energy_twin"] = et
        if with_dev:
            extra["energy_device"] = lane_energy(lanes, part(dv, 1), cond, spec.tick, exact, st_d, levels)
        extra["energy_static"] = st_d if (with_dev and spec.select == "device") else st_t
        extra["energy"] = extra["energy_device"] if spec.select == "device" else et
        extra["log10_energy"] = np.log10(np.maximum(extra["energy"], 1e-30))
    if with_dev:
        for k in ("scale", "delay", "voltage", "temp"):
            extra["dev_" + k] = cond[k]
    tm["post"] = tm.get("post", 0.0) + time.perf_counter() - t1
    return part(sel, 1 if sel is dv else 0), {k: np.asarray(v, np.float64) for k, v in extra.items()}
