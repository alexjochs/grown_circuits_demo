"""The slow delay element: a current-starved delay stage characterized in SKY130 with ngspice (freerun
docs/slow_delay.md, SPEC §14), as a search primitive.

Token. A slow delay is a DELAY node (circuit.graph) whose `drive` is SLOW_DRIVE. Its rise and fall delays (substrate
ticks) are not free parameters: they are the element's delays at one of the table's bias levels (log-spaced bias
currents over the physically usable range), so the token's one parameter is its level. The operator (circuit.moves,
`OperatorConfig.slow_weight`) adds slow delays at a uniformly drawn level and moves a slow delay's level up or down;
it never edits its delays or its drive directly. Flattening needs nothing new: a slow delay is a buffer with long
delays (inertial, SPEC §3; the SPICE data puts its minimum propagating pulse at its delay).

Physics. Per level the table gives, at the nominal corner (tt, 25 C, 1.80 V): rise and fall delays, dynamic energy
per output rise and fall (supply-current integral; dominated at long delays by short-circuit current in the stage that
restores the slow edge), static power (the element's own bias branches plus leakage: it is drawn whether or not the
element switches), relative jitter (sigma / delay, shot noise of the starving current), and per corner condition the
ratios of delays, energies and static power to nominal. The instruments (sim.devices, sim.physical) read slow gates
out of a flattened netlist by their delays: a one-input buffer whose (rise, fall) equal a level's ticks exactly
(`SlowTable.levels_of`); gate delays stay far below the table's shortest delay (operator.delay_range, delay_clamp).

    t = table()                      # the packaged table (slow_table.json), at 10 ps per tick
    r, f = t.ticks(level)            # the element's delays in ticks
    t.p_static[level]                # W
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..circuit.graph import SLOW_DRIVE  # noqa: F401  (DELAY.drive of a slow delay element)
BUF = 0b10
TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slow_table.json")


@dataclass
class SlowTable:
    """Per bias level (ascending delay) the element's nominal parameters; corner ratios per condition."""
    bias_a: np.ndarray          # A, the bias current
    d_rise_s: np.ndarray        # s
    d_fall_s: np.ndarray
    e_rise_j: np.ndarray        # J per output rise
    e_fall_j: np.ndarray
    p_static_w: np.ndarray      # W
    jitter_rel: np.ndarray      # sigma / delay
    tick: float = 1e-11         # s per substrate tick
    conditions: list = field(default_factory=list)   # [(process, T, V)] with ratios below, [n_cond, n_levels]
    r_delay: np.ndarray | None = None
    r_energy: np.ndarray | None = None
    r_static: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict, tick: float = 1e-11) -> "SlowTable":
        lv = d["levels"]
        arr = lambda k: np.array([x[k] for x in lv], np.float64)  # noqa: E731
        conds = [tuple(c) for c in d.get("conditions", [])]
        ratios = d.get("ratios", {})
        rr = lambda k: np.array(ratios[k], np.float64) if k in ratios else None  # noqa: E731
        return cls(arr("bias_a"), arr("d_rise_s"), arr("d_fall_s"), arr("e_rise_j"), arr("e_fall_j"),
                   arr("p_static_w"), arr("jitter_rel"), float(tick), conds, rr("delay"), rr("energy"),
                   rr("static"), {k: v for k, v in d.items() if k not in ("levels", "ratios", "conditions")})

    @property
    def n_levels(self) -> int:
        return len(self.bias_a)

    @property
    def rise_ticks(self) -> np.ndarray:
        return np.maximum(np.rint(self.d_rise_s / self.tick), 1).astype(np.int64)

    @property
    def fall_ticks(self) -> np.ndarray:
        return np.maximum(np.rint(self.d_fall_s / self.tick), 1).astype(np.int64)

    def ticks(self, level: int) -> tuple:
        if not 0 <= int(level) < self.n_levels:
            raise ValueError(f"slow delay level {level} outside 0 .. {self.n_levels - 1}")
        return int(self.rise_ticks[level]), int(self.fall_ticks[level])

    @property
    def min_ticks(self) -> int:
        return int(min(self.rise_ticks.min(), self.fall_ticks.min()))

    def level_of(self, rise: int, fall: int) -> int:
        """The level whose ticks are exactly (rise, fall), else -1."""
        m = np.flatnonzero((self.rise_ticks == int(rise)) & (self.fall_ticks == int(fall)))
        return int(m[0]) if len(m) else -1

    def levels_of(self, tt, arity, dr, df, gate_of_arc=None) -> np.ndarray:
        """Per gate of a flattened netlist (arrays tt [G], arity [G], one rise / fall per gate for one-input gates,
        dr / df per arc with `gate_of_arc`): its slow level, or -1 for an ordinary gate."""
        tt = np.asarray(tt)
        arity = np.asarray(arity)
        G = len(tt)
        out = np.full(G, -1, np.int64)
        cand = np.flatnonzero((tt == BUF) & (arity == 1))
        if not len(cand):
            return out
        if gate_of_arc is None:
            r, f = np.asarray(dr)[cand], np.asarray(df)[cand]
        else:
            first = np.full(G, -1, np.int64)
            ga = np.asarray(gate_of_arc)
            idx = np.arange(len(ga))
            first[ga[::-1]] = idx[::-1]
            r, f = np.asarray(dr)[first[cand]], np.asarray(df)[first[cand]]
        big = np.minimum(r, f) >= self.min_ticks
        if not big.any():
            return out
        key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(self.rise_ticks, self.fall_ticks))}
        for g, a, b in zip(cand[big], r[big], f[big]):
            out[g] = key.get((int(a), int(b)), -1)
        return out

    def condition_ratios(self, process: str, temp: float, volt: float):
        """(delay, energy, static) ratios to nominal at a condition, per level: nearest characterized condition of
        the same process (the table holds the E2 box corners; the instruments use those boxes)."""
        if not self.conditions or self.r_delay is None:
            one = np.ones(self.n_levels)
            return one, one, one
        best, bd = None, np.inf
        for i, (p, t, v) in enumerate(self.conditions):
            if p != process:
                continue
            dist = abs(t - temp) / 140.0 + abs(v - volt) / 0.55
            if dist < bd:
                best, bd = i, dist
        if best is None:
            raise ValueError(f"no slow-delay characterization at process {process!r}")
        return self.r_delay[best], self.r_energy[best], self.r_static[best]


    def condition_index(self, process, temp, volt) -> np.ndarray:
        """Per row (arrays), the index of the nearest characterized condition of the same process (see
        condition_ratios); -1 when the table has no corner data."""
        process = np.asarray(process).astype(str)
        temp = np.asarray(temp, np.float64)
        volt = np.asarray(volt, np.float64)
        out = np.full(len(temp), -1, np.int64)
        if not self.conditions or self.r_delay is None:
            return out
        best = np.full(len(temp), np.inf)
        for i, (p, t, v) in enumerate(self.conditions):
            dist = np.where(process == p, np.abs(temp - t) / 140.0 + np.abs(volt - v) / 0.55, np.inf)
            better = dist < best
            out[better] = i
            best[better] = dist[better]
        if (out < 0).any():
            raise ValueError(f"no slow-delay characterization for process {sorted(set(process[out < 0]))}")
        return out


def lane_slow_levels(lanes) -> np.ndarray:
    """int64 [total gates of a stepper.LaneSet]: each gate's slow level (-1: an ordinary gate), from its nominal
    delays. All -1 without a table."""
    G = int(lanes.gate_off[-1])
    try:
        t = table()
    except FileNotFoundError:
        return np.full(G, -1, np.int64)
    return t.levels_of(np.asarray(lanes.tt), lanes.arity, lanes.dr, lanes.df, lanes.edge_gate())


@lru_cache(maxsize=4)
def _load(path: str, tick: float) -> SlowTable:
    with open(path) as f:
        return SlowTable.from_json(json.load(f), tick)


def table(path: str | None = None, tick: float = 1e-11) -> SlowTable:
    return _load(os.path.abspath(path or os.environ.get("GROWN_SLOW_TABLE", TABLE)), float(tick))
