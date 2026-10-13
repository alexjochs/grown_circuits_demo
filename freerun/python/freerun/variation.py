"""Device variation and the idealized twin (docs/SPEC.md §12).

A *design* is a cell netlist plus annotation options. A *device* is one physical instance of it: a corner
(process, temperature, voltage), a global delay scale, per-cell mismatch and per-transition jitter, all drawn
reproducibly from (seed, circuit id). Its *idealized twin* is the same topology with nominal delays (tt, 25 C,
1.80 V, no mismatch, no jitter), optionally with one uniform delay for every arc.

    fam = LibraryFamily("tools/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib")
    design = Design(netlist, fam, init={...})
    devs = population(design, 1000, seed=1, corners=[Box("ss", (-40, 100), (1.40, 1.60)), Box("tt", (25, 100), 1.8),
                                                     Box("ff", (-40, 100), (1.65, 1.95))],
                      scale_sigma=0.03, mismatch_sigma=0.05, jitter_rel=0.01)
    sim = Simulator([d.circuit for d in devs] + [ideal_twin(devs[0])])

Circuits that do not come from a cell library can be varied too (vary(): scale, per-gate mismatch, jitter).

The output is always plain per-arc integer delays plus jitter parameters: no engine change is involved.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import cells, liberty
from .energy import EnergyModel
from .jitter import philox_np, sigma_relative
from .netlist import MAX_DELAY, Circuit, Jitter

VAR_KEY_TWEAK = 0x76617269  # "vari": variation draws use key (seed_lo, seed_hi ^ tweak), jitter uses (seed_lo, seed_hi)
STREAM_CORNER, STREAM_SCALE, STREAM_MISMATCH = 1, 2, 3


# --- corners ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Corner:
    process: str      # "ss", "tt" or "ff" (any label of a characterized process corner)
    temp: float       # degrees C
    voltage: float    # V

    def __str__(self):
        return f"{self.process}_{self.temp:g}C_{self.voltage:.3g}V"


NOMINAL = Corner("tt", 25.0, 1.80)
_LIBNAME = re.compile(r"^(?P<prefix>.+)__(?P<p>[a-z]+)_(?P<n>n?)(?P<t>\d+)C_(?P<v>\d+)v(?P<vf>\d+)\.lib$")


class LibraryFamily:
    """The characterized corner libraries of one cell library, found by file name (SKY130 style:
    <prefix>__<process>_<n?temp>C_<v>v<frac>.lib). Variants with a suffix (e.g. _ccsnoise) are skipped. Libraries
    are parsed lazily and cached."""

    def __init__(self, libdir, prefix: str | None = None, nominal: Corner = NOMINAL):
        self.libdir = Path(libdir)
        self.paths: dict[Corner, Path] = {}
        for p in sorted(self.libdir.glob("*.lib")):
            m = _LIBNAME.match(p.name)
            if not m or (prefix is not None and m["prefix"] != prefix):
                continue
            temp = -float(m["t"]) if m["n"] else float(m["t"])
            volt = float(f"{m['v']}.{m['vf']}")
            self.paths[Corner(m["p"], temp, volt)] = p
        if not self.paths:
            raise ValueError(f"no corner libraries found in {self.libdir}")
        self.nominal = nominal
        self._libs: dict[Corner, liberty.Library] = {}

    @property
    def corners(self) -> list[Corner]:
        return sorted(self.paths)

    def library(self, corner: Corner) -> liberty.Library:
        if corner not in self.paths:
            raise KeyError(f"{corner} is not a characterized corner (have {[str(c) for c in self.corners]})")
        if corner not in self._libs:
            self._libs[corner] = liberty.load(self.paths[corner])
        return self._libs[corner]

    def weights(self, corner: Corner) -> list[tuple[Corner, float]]:
        """Interpolation weights (SPEC §12): the corner itself if characterized; else linear along a characterized
        segment through it; else bilinear over the smallest characterized (T, V) rectangle containing it."""
        pts = [c for c in self.paths if c.process == corner.process]
        if not pts:
            raise ValueError(f"no libraries for process corner {corner.process!r}")
        have = {(c.temp, c.voltage) for c in pts}
        T, V = corner.temp, corner.voltage
        if (T, V) in have:
            return [(Corner(corner.process, T, V), 1.0)]
        Ts = sorted({c.temp for c in pts})
        Vs = sorted({c.voltage for c in pts})
        best = None
        for i, t1 in enumerate(Ts):
            for t2 in Ts[i:]:
                if not t1 <= T <= t2:
                    continue
                for j, v1 in enumerate(Vs):
                    for v2 in Vs[j:]:
                        if not v1 <= V <= v2:
                            continue
                        quad = {(t1, v1), (t1, v2), (t2, v1), (t2, v2)}
                        if not quad <= have:
                            continue
                        dt, dv = (t2 - t1) / 100.0, (v2 - v1) / 0.1  # normalized extents
                        key = (dt > 0 and dv > 0, dt * dv, dt + dv)  # segments first, then smallest
                        if best is None or key < best[0]:
                            best = (key, t1, t2, v1, v2)
        if best is None:
            raise ValueError(f"{corner} is outside the characterized corners of process {corner.process!r}: "
                             f"{sorted(have)}")
        _, t1, t2, v1, v2 = best
        u = 0.0 if t2 == t1 else (T - t1) / (t2 - t1)
        w = 0.0 if v2 == v1 else (V - v1) / (v2 - v1)
        out: dict[Corner, float] = {}
        for (t, v), wt in (((t1, v1), (1 - u) * (1 - w)), ((t2, v1), u * (1 - w)), ((t1, v2), (1 - u) * w),
                           ((t2, v2), u * w)):
            if wt > 0:
                c = Corner(corner.process, t, v)
                out[c] = out.get(c, 0.0) + wt
        return list(out.items())


def _geo(arrays, weights):
    """Weighted geometric mean (linear in the logarithm); falls back to arithmetic where a value is <= 0."""
    a = np.stack([np.asarray(x, dtype=np.float64) for x in arrays])
    w = np.asarray(weights, dtype=np.float64).reshape((-1,) + (1,) * (a.ndim - 1))
    lin = (w * a).sum(axis=0)
    pos = np.all(a > 0, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        geo = np.exp((w * np.log(np.where(a > 0, a, 1.0))).sum(axis=0))
    return np.where(pos, geo, lin)


def _lin(arrays, weights):
    a = np.stack([np.asarray(x, dtype=np.float64) for x in arrays])
    w = np.asarray(weights, dtype=np.float64).reshape((-1,) + (1,) * (a.ndim - 1))
    return (w * a).sum(axis=0)


# --- designs ---------------------------------------------------------------------------------------------------------

@dataclass
class CornerData:
    """Annotation of a design at one (possibly interpolated) corner, before scale and mismatch."""
    corner: Corner
    dr_ticks: np.ndarray
    df_ticks: np.ndarray
    energy: EnergyModel


class Design:
    """A cell netlist plus annotation options (cells.to_circuit keywords), annotated lazily at each characterized
    corner of a library family."""

    def __init__(self, netlist: cells.CellNetlist, family: LibraryFamily, **annotate):
        self.netlist = netlist
        self.family = family
        self.annotate = annotate
        self._ann: dict[Corner, cells.Annotated] = {}

    def annotated(self, corner: Corner) -> cells.Annotated:
        """Annotation at a characterized corner."""
        if corner not in self._ann:
            self._ann[corner] = cells.to_circuit(self.netlist, self.family.library(corner), **self.annotate)
        return self._ann[corner]

    @property
    def nominal(self) -> cells.Annotated:
        return self.annotated(self.family.nominal)

    def at(self, corner: Corner) -> CornerData:
        """Delays (unrounded ticks) and energy model at any corner inside the characterized hull (SPEC §12)."""
        ws = self.family.weights(corner)
        anns = [self.annotated(c) for c, _ in ws]
        w = [x for _, x in ws]
        e0 = anns[0].energy
        if len(anns) == 1:  # a characterized corner: exactly its own annotation (no log/exp round trip)
            return CornerData(corner, anns[0].dr_ticks, anns[0].df_ticks, e0)
        em = EnergyModel(
            vdd=float(corner.voltage), tick=e0.tick, n_inputs=e0.n_inputs,
            cap=_lin([a.energy.cap for a in anns], w), arc_gate=e0.arc_gate, arc_net=e0.arc_net,
            e_rise=_lin([a.energy.e_rise for a in anns], w), e_fall=_lin([a.energy.e_fall for a in anns], w),
            e_in_rise=_lin([a.energy.e_in_rise for a in anns], w), e_in_fall=_lin([a.energy.e_in_fall for a in anns], w),
            leak=_geo([a.energy.leak for a in anns], w))
        return CornerData(corner, _geo([a.dr_ticks for a in anns], w), _geo([a.df_ticks for a in anns], w), em)


# --- random draws (Philox, SPEC §12) ---------------------------------------------------------------------------------

def _words(seed: int, cid: int, stream: int, index) -> np.ndarray:
    """uint64[..., 4] random 32-bit words for counter (cid, stream, index, 0) under the variation key."""
    index = np.asarray(index, dtype=np.uint64)
    ctr = np.stack(np.broadcast_arrays(np.uint64(cid & 0xFFFFFFFF), np.uint64(stream), index, np.uint64(0)), axis=-1)
    key = (seed & 0xFFFFFFFF, ((seed >> 32) & 0xFFFFFFFF) ^ VAR_KEY_TWEAK)
    return philox_np(ctr, key)


def _unit(w) -> np.ndarray:
    """Uniform in (0, 1) from 32-bit words."""
    return (np.asarray(w, dtype=np.float64) + 0.5) / 2.0**32


def _normal_pair(w0, w1) -> tuple[np.ndarray, np.ndarray]:
    """Two independent standard normals (Box-Muller) from two 32-bit words."""
    r = np.sqrt(-2.0 * np.log(_unit(w0)))
    th = 2.0 * np.pi * _unit(w1)
    return r * np.cos(th), r * np.sin(th)


@dataclass(frozen=True)
class Box:
    """A region of conditions to draw from: one process corner, temperature and voltage ranges (or fixed values),
    sampled uniformly. weight: relative probability among the boxes of a population."""
    process: str
    temp: float | tuple = 25.0
    voltage: float | tuple = 1.80
    weight: float = 1.0

    def sample(self, u_t: float, u_v: float) -> Corner:
        def pick(r, u):
            if isinstance(r, (tuple, list)):
                return float(r[0] + (r[1] - r[0]) * u)
            return float(r)
        return Corner(self.process, round(pick(self.temp, u_t), 9), round(pick(self.voltage, u_v), 9))


@dataclass(frozen=True)
class Conditions:
    """Everything that makes a device differ from the nominal design (SPEC §12)."""
    corner: Corner = NOMINAL
    scale: float = 1.0               # global delay scale s_b
    mismatch_sigma: float = 0.0      # sigma of log mismatch factors per cell instance
    seed: int = 0
    cid: int = 0
    jitter_rel: float = 0.0          # per-gate jitter sigma relative to the gate's mean delay at this device
    jitter_dmin: int = 1


def draw_conditions(seed: int, cid: int, corners=(Box("tt"),), scale_sigma: float = 0.0,
                    mismatch_sigma: float = 0.0, jitter_rel: float = 0.0, jitter_dmin: int = 1) -> Conditions:
    """Conditions of device `cid` (SPEC §12). corners: Boxes (or Corners) chosen with probability proportional to
    their weights. Depends only on (seed, cid) and the parameters, never on other devices."""
    boxes = [Box(c.process, c.temp, c.voltage) if isinstance(c, Corner) else c for c in corners]
    w = _words(seed, cid, STREAM_CORNER, 0)
    p = np.cumsum([b.weight for b in boxes], dtype=np.float64)
    u0 = float(_unit(w[0])) * p[-1]
    box = boxes[min(int(np.searchsorted(p, u0, side="right")), len(boxes) - 1)]
    corner = box.sample(float(_unit(w[1])), float(_unit(w[2])))
    z, _ = _normal_pair(*_words(seed, cid, STREAM_SCALE, 0)[:2])
    return Conditions(corner, float(np.exp(scale_sigma * float(z))), mismatch_sigma, seed, cid, jitter_rel,
                      jitter_dmin)


def mismatch_factors(seed: int, cid: int, n: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Per instance (rise, fall) lognormal factors exp(sigma * z), from stream 3 with the instance index."""
    if sigma == 0.0 or n == 0:
        return np.ones(n), np.ones(n)
    w = _words(seed, cid, STREAM_MISMATCH, np.arange(n, dtype=np.uint64))
    zr, zf = _normal_pair(w[:, 0], w[:, 1])
    return np.exp(sigma * zr), np.exp(sigma * zf)


# --- devices ---------------------------------------------------------------------------------------------------------

@dataclass
class Device:
    circuit: Circuit                 # realized: corner delays x scale x mismatch, plus jitter
    nominal: Circuit                 # idealized twin: nominal delays, no mismatch, no jitter
    conditions: Conditions
    energy: EnergyModel | None = None          # at the device's corner
    nominal_energy: EnergyModel | None = None  # of the twin
    annotated: cells.Annotated | None = None   # nominal annotation (net names etc.)
    extra: dict = field(default_factory=dict)


def _round(x) -> np.ndarray:
    return np.clip(np.rint(x), 1, MAX_DELAY).astype(np.int64)


def _finish(base: Circuit, dr_t, df_t, arc_inst, n_inst, cond: Conditions) -> Circuit:
    """Apply scale and mismatch to unrounded delays, round, attach jitter."""
    mr, mf = mismatch_factors(cond.seed, cond.cid, n_inst, cond.mismatch_sigma)
    fr = np.where(arc_inst >= 0, mr[np.maximum(arc_inst, 0)], 1.0) if n_inst else np.ones(len(dr_t))
    ff = np.where(arc_inst >= 0, mf[np.maximum(arc_inst, 0)], 1.0) if n_inst else np.ones(len(df_t))
    c = base.with_delays(_round(dr_t * cond.scale * fr), _round(df_t * cond.scale * ff)).with_jitter(None)
    if cond.jitter_rel > 0:
        dmin = int(min(cond.jitter_dmin, int(min(c.dr.min(), c.df.min())))) if len(c.fanin) else 1
        c = c.with_jitter(Jitter.from_sigma(sigma_relative(c, cond.jitter_rel), seed=cond.seed, cid=cond.cid,
                                            dmin=max(1, dmin)))
    return c


def realize(design: Design, cond: Conditions) -> Device:
    """The device of a design under the given conditions."""
    nom = design.nominal
    cd = design.at(cond.corner)
    arc_gate = cd.energy.arc_gate
    arc_inst = nom.gate_instance[arc_gate] if len(arc_gate) else np.zeros(0, dtype=np.int64)
    c = _finish(nom.circuit, cd.dr_ticks, cd.df_ticks, arc_inst, len(design.netlist.instances), cond)
    return Device(c, nom.circuit.with_jitter(None), cond, cd.energy, nom.energy, nom)


def population(design: Design, n: int, seed: int, *, first_cid: int = 0, **kw) -> list[Device]:
    """n devices with circuit ids first_cid ... first_cid + n - 1 (keywords: see draw_conditions)."""
    return [realize(design, draw_conditions(seed, first_cid + i, **kw)) for i in range(n)]


def vary(circuit: Circuit, cond: Conditions) -> Device:
    """A device from a plain Circuit (no cell library): the circuit's delays are nominal; the corner is ignored;
    scale, per-gate mismatch (each gate counts as a cell instance) and jitter are applied."""
    nominal = circuit.with_jitter(None)
    arc_gate = np.repeat(np.arange(circuit.n_gates), circuit.arity)
    c = _finish(nominal, nominal.dr.astype(np.float64), nominal.df.astype(np.float64), arc_gate, circuit.n_gates,
                replace(cond, corner=NOMINAL))
    return Device(c, nominal, cond)


def ideal_twin(x, uniform: bool | int = False) -> Circuit:
    """The idealized twin (SPEC §12) of a Device or a Circuit: nominal delays, no mismatch, no jitter. For a plain
    Circuit, its own delays are taken as nominal (only jitter is removed). uniform=True gives every arc the rounded
    mean nominal arc delay; an int gives every arc that delay."""
    c = x.nominal if isinstance(x, Device) else x.with_jitter(None)
    if uniform is False or len(c.fanin) == 0:
        return c
    d = int(round((c.dr.mean() + c.df.mean()) / 2)) if uniform is True else int(uniform)
    d = max(1, min(MAX_DELAY, d))
    E = len(c.fanin)
    return c.with_delays(np.full(E, d), np.full(E, d))
