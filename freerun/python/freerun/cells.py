"""Standard-cell netlists: instances of Liberty cells -> a freerun Circuit with per-instance NLDM delays.

Delay model (static, per instance, computed once):
  load(net)  = sum of the input-pin capacitances on the net + wire_cap + wire-load model(fanout)
               + (output_load if the net is an output)
  slew(net)  = (rise, fall) output transitions of the driving arc(s), found by fixed-point iteration because the
               netlist may be cyclic; primary inputs have input_slew
  delay(arc) = NLDM cell_rise / cell_fall at (input slew chosen by unateness, load), see liberty.arc_delays
Delays are rounded to integer ticks (default tick = 1 fs) and clamped to >= 1 tick.

The same loads and slews give the energy model (docs/SPEC.md §11): net capacitances, internal energy per output and
input-pin transition, and leakage per cell.

Each output pin of a cell becomes one gate whose pins are the inputs its function depends on (Cell.inputs order).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import liberty
from .energy import EnergyModel
from .netlist import MAX_DELAY, Circuit


@dataclass
class Instance:
    name: str
    cell: str
    conns: dict  # pin name -> net name


@dataclass
class CellNetlist:
    inputs: list                                   # primary input net names (in this order)
    outputs: list = field(default_factory=list)    # net names of interest (loaded with output_load)
    instances: list = field(default_factory=list)
    consts: dict = field(default_factory=dict)     # net name -> 0/1 (tie nets)


@dataclass
class Annotated:
    circuit: Circuit
    net_names: list            # net index -> name
    net_index: dict            # name -> net index
    gate_source: list          # gate -> (instance name, output pin)
    load: np.ndarray           # per net, library capacitance units
    slew: np.ndarray           # per net [rise, fall], library time units
    tick: float                # seconds per tick
    dr_ticks: np.ndarray | None = None   # float64[E] unrounded rise delays in ticks (aligned with circuit.fanin)
    df_ticks: np.ndarray | None = None   # float64[E] unrounded fall delays
    gate_instance: np.ndarray | None = None  # int64[G] index into netlist.instances, -1 for constants
    energy: EnergyModel | None = None    # SPEC §11, from the same loads and slews
    netlist: CellNetlist | None = None
    options: dict | None = None          # the to_circuit keyword arguments (for re-annotation at other corners)


def to_circuit(nl: CellNetlist, lib: liberty.Library, *, tick: float = 1e-15, input_slew: float | None = None,
               wire_cap: float = 0.0, wire_load: str | None = None, output_load: float = 0.0, combine: str = "mean",
               init=None, slew_tol: float = 1e-9, max_iter: int = 100) -> Annotated:
    """Build a Circuit from a cell netlist. Units of input_slew / wire_cap / output_load are the library's
    (e.g. ns and pF for SKY130). input_slew defaults to the smallest index_1 point of the library's tables.
    wire_load: name of a Liberty wire_load model ("default" for the library's default_wire_load) adding
    cap(fanout) to every net; None adds nothing (only the constant wire_cap)."""
    options = dict(tick=tick, input_slew=input_slew, wire_cap=wire_cap, wire_load=wire_load, output_load=output_load,
                   combine=combine, init=init, slew_tol=slew_tol, max_iter=max_iter)
    # --- gates: one per (instance, output pin with a function) -------------------------------------------------
    gates = []  # (instance, cell, output pin, support pins, truth table)
    driver = {}
    for inst in nl.instances:
        if inst.cell not in lib.cells:
            why = lib.unsupported.get(inst.cell, "not in library")
            raise ValueError(f"instance {inst.name}: cell {inst.cell} unsupported ({why})")
        cell = lib.cells[inst.cell]
        for op in cell.outputs:
            net = inst.conns.get(op.name)
            if net is None:
                continue  # unconnected output
            if net in driver or net in nl.inputs or net in nl.consts:
                raise ValueError(f"net {net} has multiple drivers")
            _, names = liberty.parse_function(op.function)
            support = [p for p in cell.inputs if p in names]
            tt = liberty.function_truth_table(op.function, support)
            driver[net] = len(gates)
            gates.append((inst, cell, op, support, tt))
    const_nets = list(nl.consts)
    # --- nets ---------------------------------------------------------------------------------------------------
    names = list(nl.inputs) + [g[0].conns[g[2].name] for g in gates] + const_nets
    if len(set(names)) != len(names):
        raise ValueError("duplicate net names among inputs / gate outputs / constants")
    index = {n: i for i, n in enumerate(names)}
    I = len(nl.inputs)
    for inst in nl.instances:
        cell = lib.cells[inst.cell]
        for p in cell.inputs:
            n = inst.conns.get(p)
            if n is None:
                raise ValueError(f"instance {inst.name}: input pin {p} unconnected")
            if n not in index:
                raise ValueError(f"instance {inst.name}: net {n} on pin {p} has no driver")
    # --- loads --------------------------------------------------------------------------------------------------
    N = len(names)
    load = np.full(N, float(wire_cap))
    fanout = np.zeros(N, dtype=np.int64)
    for inst in nl.instances:
        cell = lib.cells[inst.cell]
        for p in cell.inputs:
            load[index[inst.conns[p]]] += cell.cap[p]
            fanout[index[inst.conns[p]]] += 1
    if wire_load is not None:
        wl_name = lib.default_wire_load if wire_load == "default" else wire_load
        if wl_name not in lib.wire_loads:
            raise ValueError(f"wire_load model {wire_load!r} not in the library ({sorted(lib.wire_loads)})")
        wl = lib.wire_loads[wl_name]
        load += np.array([wl.cap(int(f)) for f in fanout])
    for n in nl.outputs:
        load[index[n]] += output_load
    # --- slews: fixed point over the (possibly cyclic) netlist ---------------------------------------------------
    if input_slew is None:
        input_slew = min(float(a.cell_rise.index_1[0]) for c in lib.cells.values() for o in c.outputs
                         for arcs in o.arcs.values() for a in arcs if a.cell_rise is not None)
    slew = np.full((N, 2), float(input_slew))

    def gate_arcs(gi, sl):
        inst, cell, op, support, _ = gates[gi]
        net = index[inst.conns[op.name]]
        out = []
        for p in support:
            src = index[inst.conns[p]]
            out.append(liberty.arc_delays(lib, cell, op, p, sl[src, 0], sl[src, 1], load[net], combine))
        return out

    for _ in range(max_iter):
        new = slew.copy()
        for gi, (inst, cell, op, support, _) in enumerate(gates):
            arcs = gate_arcs(gi, slew)
            if arcs:
                net = index[inst.conns[op.name]]
                new[net, 0] = np.mean([a[2] for a in arcs])
                new[net, 1] = np.mean([a[3] for a in arcs])
        done = np.max(np.abs(new - slew)) <= slew_tol
        slew = new
        if done:
            break
    # --- delays -------------------------------------------------------------------------------------------------
    scale = lib.time_unit_s / tick
    k, fanin, dr, df, tts = [], [], [], [], []
    for gi, (inst, cell, op, support, tt) in enumerate(gates):
        arcs = gate_arcs(gi, slew)
        k.append(len(support))
        tts.append(tt)
        for p, a in zip(support, arcs):
            fanin.append(index[inst.conns[p]])
            dr.append(a[0] * scale)
            df.append(a[1] * scale)
    dr_ticks, df_ticks = np.array(dr, dtype=np.float64), np.array(df, dtype=np.float64)
    for n in const_nets:
        k.append(0)
        tts.append(int(nl.consts[n]))
    G = len(gates) + len(const_nets)
    # --- energy model (SPEC §11) --------------------------------------------------------------------------------
    eu, cu = lib.energy_unit_j, lib.cap_unit_f
    inst_index = {inst.name: i for i, inst in enumerate(nl.instances)}
    e_rise, e_fall, leak = [], [], np.zeros(G)
    e_in_rise, e_in_fall = np.zeros(N), np.zeros(N)
    gate_instance = np.full(G, -1, dtype=np.int64)
    seen = set()
    for gi, (inst, cell, op, support, _) in enumerate(gates):
        gate_instance[gi] = inst_index[inst.name]
        net = index[inst.conns[op.name]]
        for p in support:
            src = index[inst.conns[p]]
            er, ef = liberty.output_internal_energy(cell, op, p, slew[src, 0], slew[src, 1], load[net])
            e_rise.append(er * eu)
            e_fall.append(ef * eu)
        if inst.name not in seen:  # a cell's leakage and input-pin energy are counted once, on its first output
            seen.add(inst.name)
            leak[gi] = cell.leakage * lib.leakage_unit_w
            for p in cell.inputs:
                n = index[inst.conns[p]]
                r, f = liberty.input_internal_energy(cell, p, slew[n, 0], slew[n, 1])
                e_in_rise[n] += r * eu
                e_in_fall[n] += f * eu
    arc_gate = np.repeat(np.arange(G), k)
    emodel = EnergyModel(float(lib.voltage), tick, I, load * cu, arc_gate, np.array(fanin, dtype=np.int64),
                         np.array(e_rise, dtype=np.float64), np.array(e_fall, dtype=np.float64), e_in_rise, e_in_fall,
                         leak)
    dr = np.clip(np.rint(dr_ticks), 1, MAX_DELAY).astype(np.int64)
    df = np.clip(np.rint(df_ticks), 1, MAX_DELAY).astype(np.int64)
    if init is None:
        init_arr = np.zeros(N, dtype=np.uint8)
    elif isinstance(init, dict):
        init_arr = np.zeros(N, dtype=np.uint8)
        for n, v in init.items():
            init_arr[index[n]] = v
    else:
        init_arr = np.asarray(init, dtype=np.uint8)
    for j, n in enumerate(const_nets):
        init_arr[index[n]] = nl.consts[n]
    c = Circuit(I, np.concatenate([[0], np.cumsum(k)]) if G else np.zeros(1), np.array(fanin, dtype=np.int64),
                np.array(tts, dtype=np.uint64), dr, df, init_arr)
    sources = [(g[0].name, g[2].name) for g in gates] + [("<const>", n) for n in const_nets]
    return Annotated(c, names, index, sources, load, slew, tick, dr_ticks, df_ticks, gate_instance, emodel, nl,
                     options)
