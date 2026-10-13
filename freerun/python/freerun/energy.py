"""Energy accounting (docs/SPEC.md §11): a host-side reduction of the window observables.

    model = annotated.energy                     # from cells.to_circuit (Liberty caps, internal power, leakage)
    e = energy(window, model)                    # Energy(switching, internal, leakage, inputs, static) in joules

Static power other than leakage (the bias branches of slow delay elements, SPEC §14) is a separate per-gate term,
`EnergyModel.static`, integrated over the window like leakage but reported on its own (`Energy.static`).

Energy is not engine state: it is computed from toggle counts and the final net values of a window, so every
backend gives the same result. For large batches this reduction (dot products of toggles with per-net and per-arc
weights) could run on the device at the end of run_windowed, where high time is finalized, to avoid copying toggle
counts back; today the host does it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnergyModel:
    """Per-circuit energy parameters in SI units (SPEC §11). Arrays are indexed like the circuit's nets, gates and
    arcs (an arc is a gate input pin, aligned with Circuit.fanin)."""
    vdd: float              # V
    tick: float             # s per tick
    n_inputs: int
    cap: np.ndarray         # float64[N]  F, load capacitance of each net (fanout pin caps + wire)
    arc_gate: np.ndarray    # int64[E]    gate of each arc
    arc_net: np.ndarray     # int64[E]    input net of each arc
    e_rise: np.ndarray      # float64[E]  J, internal energy of an output rise of the arc's gate, caused via the arc
    e_fall: np.ndarray      # float64[E]  J, of an output fall
    e_in_rise: np.ndarray   # float64[N]  J, input-pin internal energy of the net's fanout per net rise
    e_in_fall: np.ndarray   # float64[N]  J, per net fall
    leak: np.ndarray        # float64[G]  W, leakage power (each cell's leakage on the gate of its first output)
    static: np.ndarray | None = None  # float64[G] W, designed static power (slow delay bias branches, SPEC §14)

    @property
    def n_gates(self) -> int:
        return len(self.leak)

    @staticmethod
    def switching_only(circuit, cap, vdd: float, tick: float, leak=None) -> "EnergyModel":
        """A model with only C*V^2 switching energy (and optional leakage), e.g. for circuits not built from cells."""
        N, G, E = circuit.n_nets, circuit.n_gates, len(circuit.fanin)
        return EnergyModel(vdd, tick, circuit.n_inputs, np.broadcast_to(np.asarray(cap, float), (N,)).copy(),
                           np.repeat(np.arange(G), circuit.arity), circuit.fanin.astype(np.int64), np.zeros(E),
                           np.zeros(E), np.zeros(N), np.zeros(N),
                           np.zeros(G) if leak is None else np.broadcast_to(np.asarray(leak, float), (G,)).copy())

    def gate_internal(self, toggles) -> tuple[np.ndarray, np.ndarray]:
        """Per-gate internal energy per output rise and fall for a window: the arcs' energies weighted by their
        input nets' share of the gate's input toggles (uniform when no input toggled), SPEC §11."""
        G = self.n_gates
        tin = np.asarray(toggles, dtype=np.float64)[self.arc_net]
        tot = np.bincount(self.arc_gate, tin, minlength=G)
        k = np.bincount(self.arc_gate, minlength=G).astype(np.float64)
        w = np.where(tot[self.arc_gate] > 0, tin / np.where(tot > 0, tot, 1.0)[self.arc_gate],
                     1.0 / np.maximum(k, 1.0)[self.arc_gate])
        return (np.bincount(self.arc_gate, w * self.e_rise, minlength=G),
                np.bincount(self.arc_gate, w * self.e_fall, minlength=G))


@dataclass
class Energy:
    """Energy of one circuit over one window, joules. `total` excludes the primary inputs' switching (`inputs`)."""
    switching: float
    internal: float
    leakage: float
    inputs: float
    duration: float  # s
    static: float = 0.0   # designed static power x duration (slow delay elements, SPEC §14); not leakage

    @property
    def total(self) -> float:
        return self.switching + self.internal + self.leakage + self.static

    @property
    def power(self) -> float:
        return self.total / self.duration if self.duration > 0 else float("nan")


def rises_falls(toggles: np.ndarray, final: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact rises and falls per net from toggle counts and final values (SPEC §11)."""
    toggles = np.asarray(toggles, dtype=np.int64)
    final = np.asarray(final, dtype=np.int64)
    initial = final ^ (toggles & 1)
    diff = final - initial  # rises - falls, in {-1, 0, 1}
    rises = (toggles + diff) // 2
    return rises, toggles - rises


def energy(window, model: EnergyModel, per_net: bool = False):
    """Energy of a Window (ref.Window, from any backend) under `model`. per_net: also return the per-net
    (switching + input-pin) and per-gate (internal + leakage) energy arrays in joules."""
    if window.values is None:
        raise ValueError("the window has no final values (Window.values)")
    I = model.n_inputs
    V2 = model.vdd ** 2
    tog = np.asarray(window.toggles, dtype=np.float64)
    rises, falls = rises_falls(window.toggles, window.values)
    sw_net = 0.5 * model.cap * V2 * tog
    pin_net = rises * model.e_in_rise + falls * model.e_in_fall
    er, ef = model.gate_internal(window.toggles)
    gate_int = rises[I:] * er + falls[I:] * ef
    duration = (window.t1 - window.t0) * model.tick
    gate_leak = model.leak * duration
    gate_static = (model.static * duration if model.static is not None
                   else np.zeros_like(gate_leak))
    e = Energy(float(sw_net[I:].sum()), float(pin_net.sum() + gate_int.sum()), float(gate_leak.sum()),
               float(sw_net[:I].sum()), duration, float(gate_static.sum()))
    if per_net:
        return e, sw_net + pin_net, gate_int + gate_leak + gate_static
    return e
