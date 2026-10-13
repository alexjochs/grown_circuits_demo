"""Slow delay elements (SPEC §14): a current-starved delay stage characterized in SKY130 (docs/slow_delay.md).

The engine needs nothing new: a slow delay element is a one-input gate (a buffer) whose rise and fall delays are
long, with the inertial semantics of §3 (the SPICE data puts its minimum propagating pulse at its delay). This module
is the annotation layer: it turns an element table (scripts/slow_delay_char.py report --export) into delays in
ticks, jitter amplitudes (§10, sigma proportional to the delay with the element's own relative jitter), corner
ratios (§12) and the energy terms (§11: energy per output transition, and the static power of its bias branches,
reported separately as EnergyModel.static).

    t = SlowTable.load("slow_table.json")
    r, f = t.ticks(level, tick=1e-12)               # rise and fall delay in ticks
    amp = t.jitter_amplitude(level, tick=1e-12)     # SPEC §10 a_g for the element
    m = t.energy_model(circuit, levels, vdd, tick)  # an EnergyModel whose slow gates carry the table's terms
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .energy import EnergyModel
from .jitter import K as JITTER_K
from .netlist import MAX_DELAY

BUF = 0b10


@dataclass
class SlowTable:
    """Per level (ascending delay): bias current, nominal delays, energy per output rise / fall, static power,
    relative jitter; corner ratios [condition, level] for delay, energy and static power."""
    bias_a: np.ndarray
    d_rise_s: np.ndarray
    d_fall_s: np.ndarray
    e_rise_j: np.ndarray
    e_fall_j: np.ndarray
    p_static_w: np.ndarray
    jitter_rel: np.ndarray
    conditions: list = field(default_factory=list)
    r_delay: np.ndarray | None = None
    r_energy: np.ndarray | None = None
    r_static: np.ndarray | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "SlowTable":
        lv = d["levels"]
        a = lambda k: np.array([x[k] for x in lv], np.float64)  # noqa: E731
        r = d.get("ratios", {})
        g = lambda k: np.array(r[k], np.float64) if k in r else None  # noqa: E731
        return cls(a("bias_a"), a("d_rise_s"), a("d_fall_s"), a("e_rise_j"), a("e_fall_j"), a("p_static_w"),
                   a("jitter_rel"), [tuple(c) for c in d.get("conditions", [])], g("delay"), g("energy"), g("static"))

    @classmethod
    def load(cls, path: str) -> "SlowTable":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @property
    def n_levels(self) -> int:
        return len(self.bias_a)

    def ticks(self, level, tick: float, ratio: float = 1.0):
        """(rise, fall) delay in ticks at a delay ratio to nominal (§12: corner, scale, mismatch), clamped to
        [1, 2^31 - 1]."""
        f = lambda x: int(min(max(round(float(x) * ratio / tick), 1), MAX_DELAY))  # noqa: E731
        return f(self.d_rise_s[level]), f(self.d_fall_s[level])

    def jitter_amplitude(self, level, tick: float, ratio: float = 1.0) -> int:
        """§10 amplitude a_g for an element at `level`: sigma = jitter_rel x its mean delay (in ticks)."""
        r, f = self.ticks(level, tick, ratio)
        return int(round(float(self.jitter_rel[level]) * 0.5 * (r + f) * JITTER_K))

    def condition(self, process: str, temp: float, volt: float) -> int:
        """Index of the characterized condition of `process` nearest to (temp, volt) (the table holds the corner-box
        extremes; nearest, not interpolated)."""
        best, bd = -1, np.inf
        for i, (p, t, v) in enumerate(self.conditions):
            if p == process:
                d = abs(t - temp) / 140.0 + abs(v - volt) / 0.55
                if d < bd:
                    best, bd = i, d
        if best < 0:
            raise ValueError(f"no characterization at process {process!r}")
        return best

    def energy_model(self, circuit, levels, vdd: float, tick: float, cap=None, condition: int | None = None):
        """An EnergyModel for `circuit` (§11) in which gate g with levels[g] >= 0 is a slow element: its arcs carry
        the table's energy per output rise / fall, its static power the table's (bias branches + leakage); other
        gates and all nets carry only switching energy of `cap` (F per net, default 0). condition: an index into
        `conditions` to apply that corner's ratios."""
        lv = np.asarray(levels, np.int64)
        m = EnergyModel.switching_only(circuit, 0.0 if cap is None else cap, vdd, tick)
        ga = m.arc_gate
        sg = lv >= 0
        li = np.maximum(lv, 0)
        re = rs = np.ones(len(lv))
        if condition is not None and self.r_energy is not None:
            re, rs = self.r_energy[condition][li], self.r_static[condition][li]
        m.e_rise = np.where(sg[ga], (self.e_rise_j[li] * re)[ga], 0.0)
        m.e_fall = np.where(sg[ga], (self.e_fall_j[li] * re)[ga], 0.0)
        m.static = np.where(sg, self.p_static_w[li] * rs, 0.0)
        return m
