"""Physical instruments: the idealization gap and energy per decision of a circuit under device variation, with
energy from SKY130 Liberty cells (freerun's energy model, SPEC §11; variation, SPEC §12). See ~/freerun/docs/SPEC.md.

Two timing modes (PhysicalConfig.timing):

timing: arcs (default) -- the search substrate's own timing
  The circuit runs with its own per-arc rise/fall delays, read as SKY130 hd delays at `substrate_tick` (10 ps, the
  convention of grown.sim.devices and the search: arcs in [3, 12] ticks = 30-120 ps), at the nominal corner. The
  *twin* is that circuit, unchanged: exactly what the search substrate (FreerunSubstrate, physical: None or select:
  twin) runs, so the twin's outputs, duties and task error equal the search's bit for bit (tested on random cyclic
  circuits with asymmetric arcs and on a saved E3 elite). Each *device* is drawn and applied by grown.sim.devices
  itself (`draw` with seed = PhysicalConfig.seed and cid = the device index, `apply_devices`): a corner box (process,
  T, V) entering as the calibrated corner delay ratio, a die-to-die scale, per-gate rise/fall mismatch, delays
  rounded and clamped to >= 1, and per-transition jitter; the same per-arc model the search's device lanes use, and
  the same (corner, scale) per device index as freerun.variation.population. `subticks` (default 1) runs every
  circuit at a finer time base (all delays and the step multiplied by it; the twin is unchanged up to that time
  scaling, tested) so that variation of short arcs is not rounded away; 1 is the search's own resolution.
  Liberty cells are used for energy only: each gate is mapped to one SKY130 cell (`map_cells(chains=False)`, the
  table below) and the cell netlist is annotated with freerun (loads = fanout pin capacitances, slews by fixed point),
  at the nominal corner for the twin and at each device's corner (freerun.variation.Design.at, interpolated as
  LibraryFamily.weights) for the devices. That annotation's energy model is carried over to the circuit: net
  capacitance and input-pin internal energy per net, internal energy per arc (an arc on a pin that the gate's
  function ignores, which the cell does not have, takes the mean of the gate's cell arcs), leakage per gate (a
  constant gate: none). Its delays are not used.

timing: cells -- a real cell netlist (the pre-2026-10-10 behaviour, kept for unit-delay exports)
  Cell mapping (circuit -> sky130_fd_sc_hd cell netlist), `map_cells`: the circuit is flattened, each gate's pins
  reduced to the distinct nets its function depends on, then a constant becomes a tie net; a function one `_1` cell
  implements up to a pin permutation becomes that cell (all 16 two-input functions, every 1-input function and 55 of
  the 256 three-input functions); any other function (arity 3..6) a same-arity *proxy* cell with the gate's own truth
  table substituted (3: mux2_1, 4: a22o_1, 5: a221o_1, 6: a222oi_1); a gate whose largest arc is d > 1 ticks is
  followed by d - 1 buf_1 cells. The circuit's per-pin and rise/fall-asymmetric arcs are not kept (each gate keeps
  only its largest arc, as a chain): right for a unit-delay export whose clocked semantics is the point, wrong for a
  searched circuit tuned to its arcs (NOTES 2026-10-10 E4: E3 elites 0.073 on their substrate, 0.355 in this twin).
  One substrate tick is `unit` freerun ticks (default: the rounded mean nominal arc delay of the mapped circuit at
  tt, 25 C, 1.80 V; freerun ticks are `tick` = 1 fs). The *uniform twin* (every arc exactly `unit`) is the
  stand-in substrate's semantics scaled in time (tested); the *nominal twin* has the cells' real delays at the
  nominal corner; devices are freerun.variation.population of the cell design (reproducible from (seed, device)).

Common to both: the step duration is fixed in physical time (a slow device does not get a slower clock). Outputs
are sampled at the end of each step, (s + 1) K unit; the duty of step s is the fraction of [s K unit + unit,
(s + 1) K unit + unit) the wire is high (the stand-in's convention scaled by `unit`; in arcs mode unit = subticks).

Idealization gap (against the twin: arcs: the circuit itself; cells: the uniform twin), per device and summarized
  disagree    fraction of (step, output wire) samples at the end of each step that differ from the twin
  duty_diff   mean |duty - twin duty| per step and wire (duty: fraction of the step's time the wire is high)
  error_diff  task error of the device minus that of the twin, when a score function (duty -> error) is given
  The nominal twin is reported against the twin too (cells: real cell delays at nominal conditions; arcs: the same
  circuit as the twin, so 0).

Energy (freerun.energy, SPEC §11): switching + internal + leakage joules over the run, divided by the number of
control steps (one decision per step), for every device and both twins, plus leakage power so the cost of
holding the circuit at a slower control rate can be added for circuits that are quiescent between decisions.
Primary-input switching is excluded, as in freerun. Slow delay elements (grown.sim.slow, freerun SPEC §14; read out
of the netlist by their delays) carry their SPICE table's energy per output transition (crossover current of their
restoring stages included) and their static power (bias branches + leakage), which freerun reports separately
(`static`, `static_power` per step and per second); at a device's corner both take the table's ratios.

`measure` runs everything in one simulator batch; `idealization_gap`, `energy` and `descriptor` are views of it.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field

import numpy as np

from ..circuit.flatten import Netlist, flatten
from .freerun_backend import import_freerun

PROXIES = {3: "sky130_fd_sc_hd__mux2_1", 4: "sky130_fd_sc_hd__a22o_1", 5: "sky130_fd_sc_hd__a221o_1",
           6: "sky130_fd_sc_hd__a222oi_1"}
BUF = "sky130_fd_sc_hd__buf_1"
EXCLUDE = ("lpflow", "clk", "dly", "probe", "bufbuf", "bufinv", "ebuf", "einv", "conb", "tap", "fill", "decap",
           "diode", "macro", "sdl", "dlx", "dfx", "dfr", "dfs", "dfb", "edf", "sdf", "dlr", "dlc", "lowlv")


def libdir() -> str:
    d = os.environ.get("GROWN_SKY130_LIBDIR")
    if d:
        return d
    from .freerun_backend import default_freerun_home
    # The SKY130 PDK is not part of this repository (2 GB): freerun/scripts/build_tools.sh fetches it into
    # freerun/tools/, or point $GROWN_SKY130_LIBDIR at an existing sky130_fd_sc_hd/lib directory.
    return os.path.join(default_freerun_home(), "tools/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib")


def corner_boxes(preset: str = "corners") -> tuple:
    """Condition boxes. "corners": ss/tt/ff over temperature and voltage (freerun's idealization-gap example);
    "tt": the nominal process only (local variation: scale, mismatch, jitter); "nominal": tt 25 C 1.80 V."""
    import_freerun()
    import freerun.variation as fv
    if preset == "corners":
        return (fv.Box("ss", (-40, 100), (1.40, 1.60)), fv.Box("tt", (25, 100), 1.80),
                fv.Box("ff", (-40, 100), (1.65, 1.95)))
    if preset in ("tt", "nominal"):
        return (fv.Box("tt", 25.0, 1.80),)
    raise ValueError(f"unknown corner preset {preset!r}")


TIMING = ("arcs", "cells")


@dataclass(frozen=True)
class PhysicalConfig:
    n_devices: int = 16
    seed: int = 1
    preset: str = "corners"          # condition boxes, see corner_boxes (= grown.sim.devices.PRESETS)
    scale_sigma: float = 0.03        # die-to-die delay scale (lognormal sigma)
    mismatch_sigma: float = 0.05     # per-gate (cells: per-cell) rise/fall mismatch (lognormal sigma)
    jitter_rel: float = 0.01         # per-transition jitter sigma relative to the gate's delay
    timing: str = "arcs"             # "arcs": the circuit's own arcs (the search substrate); "cells": cell-mapped
    subticks: int = 1                # arcs: freerun ticks per substrate tick (1 = the search's resolution)
    substrate_tick: float = 1e-11    # arcs: seconds per substrate tick (grown.sim.devices' SKY130 convention)
    unit: int | None = None          # cells: freerun ticks per substrate tick; None = rounded mean nominal arc delay
    tick: float = 1e-15              # cells: seconds per freerun tick
    backend: str = "cpu"
    threads: int = 4

    def variation(self) -> dict:
        """cells: freerun.variation.population keywords."""
        return dict(corners=corner_boxes(self.preset), scale_sigma=self.scale_sigma,
                    mismatch_sigma=self.mismatch_sigma, jitter_rel=self.jitter_rel)

    def spec(self):
        """arcs: this population as a grown.sim.devices.PhysicalSpec."""
        from .devices import PhysicalSpec
        return PhysicalSpec(select="device", seed=int(self.seed), preset=self.preset, scale_sigma=self.scale_sigma,
                            mismatch_sigma=self.mismatch_sigma, jitter_rel=self.jitter_rel,
                            tick=self.substrate_tick)


# ---- cell libraries and the function -> cell table ------------------------------------------------------------------

_FAMILIES: dict = {}
_TABLES: dict = {}


def family(path: str | None = None):
    import_freerun()
    from freerun.variation import LibraryFamily
    path = path or libdir()
    if path not in _FAMILIES:
        _FAMILIES[path] = LibraryFamily(path)
    return _FAMILIES[path]


def cell_table(lib) -> dict:
    """(arity, truth table over our pin order) -> (cell name, perm) with our pin i on cell pin cell.inputs[perm[i]].
    Single-output combinational `_1` cells, EXCLUDE filtered, smallest area wins."""
    key = getattr(lib, "path", None) or lib.name
    if key in _TABLES:
        return _TABLES[key]
    table: dict = {}
    cands = []
    for name, c in lib.cells.items():
        base = name.split("__")[-1]
        if not base.endswith("_1") or len(c.outputs) != 1 or any(x in base for x in EXCLUDE):
            continue
        k = len(c.inputs)
        if not 1 <= k <= 6:
            continue
        cands.append((c.area, name, c))
    for area, name, c in sorted(cands, key=lambda x: (x[0], x[1])):
        k = len(c.inputs)
        tt = int(c.outputs[0].tt)
        if k > 4:
            perms = [tuple(range(k))]
        else:
            perms = itertools.permutations(range(k))
        for perm in perms:
            t2 = 0
            for r in range(1 << k):
                cr = sum(((r >> i) & 1) << perm[i] for i in range(k))
                t2 |= ((tt >> cr) & 1) << r
            table.setdefault((k, t2), (name, tuple(perm)))
    _TABLES[key] = table
    return table


def reduce_gate(tt: int, pins: list) -> tuple[int, list]:
    """Merge pins on the same net and drop pins the function ignores. Returns (tt, distinct support nets)."""
    nets = list(dict.fromkeys(pins))
    k0, k = len(pins), len(nets)
    pos = [nets.index(p) for p in pins]
    t = 0
    for r in range(1 << k):
        row = sum(((r >> pos[i]) & 1) << i for i in range(k0))
        t |= ((tt >> row) & 1) << r
    keep = []
    for i in range(k):
        if any(((t >> r) & 1) != ((t >> (r | (1 << i))) & 1) for r in range(1 << k) if not r >> i & 1):
            keep.append(i)
    t2 = 0
    for r in range(1 << len(keep)):
        row = sum(((r >> j) & 1) << keep[j] for j in range(len(keep)))
        t2 |= ((t >> row) & 1) << r
    return t2, [nets[i] for i in keep]


@dataclass
class Mapping:
    netlist: object                  # freerun.cells.CellNetlist
    init: dict                       # net name -> initial value
    outputs: list                    # net name per circuit output
    proxy_tt: dict                   # instance name -> truth table to substitute (proxies)
    info: dict = field(default_factory=dict)


def map_cells(net: Netlist, lib, chains: bool = True) -> Mapping:
    """Our flattened netlist -> a SKY130 cell netlist (module docstring). chains=False: one cell per gate and no
    buffer chains (timing: arcs, energy only; gate g is instance "g<g>" driving net "n<I+g>")."""
    import_freerun()
    from freerun import cells
    table = cell_table(lib)
    I = net.n_inputs
    names = [f"i{j}" for j in range(I)] + [f"n{I + g}" for g in range(net.n_gates)]
    insts, consts, init = [], {}, {}
    proxy_tt: dict = {}
    counts = dict(cells=0, exact=0, proxy=0, const=0, chain_buffers=0)
    by_cell: dict = {}
    for g in range(net.n_gates):
        s = slice(int(net.fanin_ptr[g]), int(net.fanin_ptr[g + 1]))
        pins = [int(x) for x in net.fanin[s]]
        dmax = int(max(net.dr[s].max(initial=1), net.df[s].max(initial=1))) if chains else 1
        out = names[I + g]
        v0 = int(net.init[I + g])
        tt, sup = reduce_gate(int(net.tt[g]), pins)
        if not sup:
            consts[out] = tt & 1
            counts["const"] += 1
            continue
        cell_out = out if dmax <= 1 else out + "_c"
        hit = table.get((len(sup), tt))
        if hit is not None:
            cname, perm = hit
            cell = lib.cells[cname]
            conns = {cell.inputs[perm[i]]: names[p] for i, p in enumerate(sup)}
            counts["exact"] += 1
        else:
            cname = PROXIES[len(sup)]
            cell = lib.cells[cname]
            conns = {cell.inputs[i]: names[p] for i, p in enumerate(sup)}
            proxy_tt[f"g{g}"] = tt
            counts["proxy"] += 1
        conns[cell.outputs[0].name] = cell_out
        insts.append(cells.Instance(f"g{g}", cname, conns))
        by_cell[cname] = by_cell.get(cname, 0) + 1
        init[cell_out] = v0
        prev = cell_out
        for m in range(dmax - 1):
            nxt = out if m == dmax - 2 else f"{out}_b{m}"
            insts.append(cells.Instance(f"g{g}_b{m}", BUF, {"A": prev, "X": nxt}))
            init[nxt] = v0
            prev = nxt
            counts["chain_buffers"] += 1
    counts["cells"] = len(insts)
    outs = [names[int(o)] for o in net.outputs]
    nl = cells.CellNetlist(inputs=names[:I], outputs=sorted(set(o for o in outs if not o.startswith("i"))),
                           instances=insts, consts=consts)
    return Mapping(nl, init, outs, proxy_tt, dict(counts, by_cell=by_cell))


class _EnergyMap:
    """timing: arcs. Our circuit's nets, arcs and gates -> those of its energy-only cell netlist (map_cells with
    chains=False, annotated by freerun), so the cell netlist's freerun EnergyModel at any corner carries over to our
    circuit (module docstring)."""

    def __init__(self, net: Netlist, ann):
        I, G = net.n_inputs, net.n_gates
        names = [f"i{j}" for j in range(I)] + [f"n{I + g}" for g in range(G)]
        self.n_inputs = I
        self.net_src = np.array([ann.net_index[n] for n in names], np.int64)
        gi_of = {inst: gi for gi, (inst, _pin) in enumerate(ann.gate_source) if inst != "<const>"}
        afp = np.asarray(ann.circuit.fanin_ptr, np.int64)
        afi = np.asarray(ann.circuit.fanin, np.int64)
        E = len(net.fanin)
        self.arc_gate = np.repeat(np.arange(G, dtype=np.int64), np.diff(np.asarray(net.fanin_ptr, np.int64)))
        self.arc_net = np.asarray(net.fanin, np.int64)
        self.arc_src = np.full(E, -1, np.int64)        # the cell arc of each of our arcs (-1: a pin the cell lacks)
        self.arc_cell = np.full(E, -1, np.int64)       # the cell gate of each of our arcs (-1: a constant gate)
        self.gate_src = np.full(G, -1, np.int64)
        for g in range(G):
            gi = gi_of.get(f"g{g}")
            if gi is None:                             # a constant: a tie net, no cell
                continue
            self.gate_src[g] = gi
            src = {int(afi[a]): a for a in range(int(afp[gi]), int(afp[gi + 1]))}
            for e in range(int(net.fanin_ptr[g]), int(net.fanin_ptr[g + 1])):
                self.arc_src[e] = src.get(int(self.net_src[int(net.fanin[e])]), -1)
                self.arc_cell[e] = gi

    def model(self, em, tick: float, slow=None):
        """Our circuit's freerun EnergyModel from the cell netlist's model em (at any corner). slow: None, or
        (levels [G], energy ratio [G], static ratio [G]) for slow delay elements (grown.sim.slow): their arcs take
        the SPICE table's energy per output rise / fall, their leakage is replaced by the table's static power (bias
        branches + leakage), reported as freerun's separate `static` term."""
        from freerun.energy import EnergyModel
        Ga = len(em.leak)
        k = np.bincount(np.asarray(em.arc_gate, np.int64), minlength=Ga).astype(np.float64)

        def arcs(x):
            x = np.asarray(x, np.float64)
            if not len(x):
                return np.zeros(len(self.arc_src))
            mean = np.bincount(np.asarray(em.arc_gate, np.int64), x, minlength=Ga) / np.maximum(k, 1.0)
            fill = np.where(self.arc_cell >= 0, mean[np.maximum(self.arc_cell, 0)], 0.0)
            return np.where(self.arc_src >= 0, x[np.maximum(self.arc_src, 0)], fill)
        leak = np.where(self.gate_src >= 0, np.asarray(em.leak, np.float64)[np.maximum(self.gate_src, 0)], 0.0)
        e_rise, e_fall = arcs(em.e_rise), arcs(em.e_fall)
        kw = {}
        if slow is not None and (slow[0] >= 0).any():
            from .slow import table
            t = table()
            lev, r_e, r_s = slow
            sg = lev >= 0
            ga = self.arc_gate
            sa = sg[ga]
            # the table's energy is all the element draws per output transition (its stages and their crossover
            # current, driving one inv_1): it replaces the buffer cell's internal energy and its output net's
            # switching energy (the net's capacitance stays: it is the fanout's load, part of the next stage)
            e_rise = np.where(sa, t.e_rise_j[np.maximum(lev, 0)][ga] * r_e[ga], e_rise)
            e_fall = np.where(sa, t.e_fall_j[np.maximum(lev, 0)][ga] * r_e[ga], e_fall)
            leak = np.where(sg, 0.0, leak)
            kw["static"] = np.where(sg, t.p_static_w[np.maximum(lev, 0)] * r_s, 0.0)
        return EnergyModel(float(em.vdd), float(tick), self.n_inputs, np.asarray(em.cap, np.float64)[self.net_src],
                           self.arc_gate, self.arc_net, e_rise, e_fall,
                           np.asarray(em.e_in_rise, np.float64)[self.net_src],
                           np.asarray(em.e_in_fall, np.float64)[self.net_src], leak, **kw)


# ---- the physical view of one circuit -------------------------------------------------------------------------------

class Physical:
    """A circuit, its twins, its device population and their energy models (module docstring; cfg.timing)."""

    def __init__(self, circuit, cfg: PhysicalConfig = PhysicalConfig(), library=None, netlist: Netlist | None = None):
        import_freerun()
        if cfg.timing not in TIMING:
            raise ValueError(f"unknown timing {cfg.timing!r} (one of {TIMING})")
        self.cfg = cfg
        self.timing = cfg.timing
        self.net = netlist if netlist is not None else flatten(circuit, library)
        if cfg.timing == "cells":
            self._init_cells()
        else:
            self._init_arcs()
        self.devices = self._population(cfg)

    def _init_cells(self):
        from freerun.netlist import Circuit as FrCircuit
        from freerun.variation import Design, ideal_twin
        cfg = self.cfg
        fam = family()
        self.mapping = m = map_cells(self.net, fam.library(fam.nominal))
        self.design = Design(m.netlist, fam, init=m.init, tick=cfg.tick)
        ann = self.design.nominal
        if m.proxy_tt:                                   # substitute the LUT functions into the proxy cells
            tt = ann.circuit.tt.copy()
            for g, (inst, _pin) in enumerate(ann.gate_source):
                if inst in m.proxy_tt:
                    tt[g] = m.proxy_tt[inst]
            c = ann.circuit
            ann.circuit = FrCircuit(c.n_inputs, c.fanin_ptr, c.fanin, tt, c.dr, c.df, c.init, c.jitter)
        self.annotated = ann
        self.out_nets = [ann.net_index[o] for o in m.outputs]
        nom = ann.circuit
        self.unit = int(cfg.unit) if cfg.unit else (max(1, int(round((nom.dr.mean() + nom.df.mean()) / 2)))
                                                    if len(nom.fanin) else 1)
        self.tick = cfg.tick
        self.nominal_twin = ideal_twin(nom)
        self.uniform_twin = self.twin = ideal_twin(nom, uniform=self.unit)
        self.nominal_energy = ann.energy

    def _init_arcs(self):
        from dataclasses import replace
        from freerun.variation import Design
        from .freerun_backend import to_freerun
        cfg = self.cfg
        if cfg.unit is not None:
            raise ValueError("unit is a timing: cells option (timing: arcs runs at subticks freerun ticks per tick)")
        m = int(cfg.subticks)
        if m < 1:
            raise ValueError("subticks must be >= 1")
        self.unit = m
        self.tick = cfg.substrate_tick / m
        self.scaled = replace(self.net, dr=np.asarray(self.net.dr, np.int64) * m,
                              df=np.asarray(self.net.df, np.int64) * m)
        self.twin = self.nominal_twin = self.uniform_twin = to_freerun(self.scaled)
        self.out_nets = [int(o) for o in self.net.outputs]
        fam = family()
        self.mapping = mp = map_cells(self.net, fam.library(fam.nominal), chains=False)
        self.design = Design(mp.netlist, fam, init=mp.init, tick=self.tick)
        self.annotated = self.design.nominal              # energy only: the cells' delays are not used
        self._emap = _EnergyMap(self.net, self.annotated)
        from .slow import TABLE, table
        G = self.net.n_gates
        self.slow_levels = np.full(G, -1, np.int64)
        if os.path.exists(os.environ.get("GROWN_SLOW_TABLE", TABLE)):
            ga = np.repeat(np.arange(G), np.diff(np.asarray(self.net.fanin_ptr, np.int64)))
            self.slow_levels = table().levels_of(np.asarray(self.net.tt), np.diff(np.asarray(self.net.fanin_ptr)),
                                                 self.net.dr, self.net.df, ga)
        one = np.ones(G)
        self.nominal_energy = self._emap.model(self.annotated.energy, self.tick, (self.slow_levels, one, one))

    def _population(self, cfg: PhysicalConfig) -> list:
        if not cfg.n_devices:
            return []
        if cfg.timing == "cells":
            from freerun.variation import population
            return population(self.design, cfg.n_devices, cfg.seed, **cfg.variation())
        from freerun.variation import Conditions, Corner, Device
        from .devices import apply_devices, draw
        from .stepper import Job, LaneSet
        D = int(cfg.n_devices)
        spec = cfg.spec()
        x = np.zeros((D, 1, self.net.n_inputs), np.uint8)
        lanes = LaneSet.build([Job([self.scaled], np.zeros(D, np.int64), x, 1)])
        seed = np.full(D, int(cfg.seed) & 0xFFFFFFFFFFFFFFFF, np.uint64)
        cid = np.arange(D, dtype=np.uint32)
        cond = draw(lanes, spec, seed, cid)               # = freerun.variation.draw_conditions(seed, d) per device d
        apply_devices(lanes, np.arange(D), cond, spec)
        out = []
        for d, c in enumerate(lanes.circuits()):
            corner = Corner(str(cond["process"][d]), float(cond["temp"][d]), float(cond["voltage"][d]))
            slow = None
            if (self.slow_levels >= 0).any():
                from .slow import table
                t = table()
                ci = int(t.condition_index([corner.process], [corner.temp], [corner.voltage])[0])
                lv = np.maximum(self.slow_levels, 0)
                slow = ((self.slow_levels,) + ((t.r_energy[ci][lv], t.r_static[ci][lv]) if ci >= 0
                                               else (np.ones(len(lv)), np.ones(len(lv)))))
            em = self._emap.model(self.design.at(corner).energy, self.tick, slow)
            cd = Conditions(corner, float(cond["scale"][d]), spec.mismatch_sigma, int(seed[d]), int(cid[d]),
                            spec.jitter_rel)
            out.append(Device(c.with_init(self.twin.init), self.twin, cd, em, self.nominal_energy, None,
                              dict(delay_factor=float(cond["delay"][d]))))
        return out

    def set_population(self, cfg: PhysicalConfig) -> "Physical":
        """Another device population (preset, seed, sigmas) for the same circuit, twins and energy mapping."""
        keep = ("timing", "subticks", "substrate_tick", "unit", "tick")
        if any(getattr(cfg, k) != getattr(self.cfg, k) for k in keep):
            raise ValueError(f"set_population cannot change {keep}; build a new Physical")
        self.cfg = cfg
        self.devices = self._population(cfg)
        return self

    @property
    def n_inputs(self) -> int:
        return self.net.n_inputs

    def circuits(self) -> list:
        """[twin, nominal twin, device 0, ..., device D-1] as freerun circuits (cells: the twin is the uniform twin;
        arcs: both twins are the circuit itself)."""
        return [self.twin, self.nominal_twin] + [d.circuit for d in self.devices]

    def energy_models(self) -> list:
        return [self.nominal_energy, self.nominal_energy] + [d.energy for d in self.devices]

    def labels(self) -> list:
        return ((["uniform_twin"] if self.timing == "cells" else ["twin"]) + ["nominal_twin"]
                + [str(d.conditions.corner) for d in self.devices])

    def run(self, inputs, K: int, per_tick: bool = False, circuits=None, models=None, settle_steps: int = 0) -> dict:
        """Simulate every circuit (default: twins and devices) on every episode of inputs [E, S, n_in] (or
        [E, S, K, n_in] per tick). Returns outputs uint8 [C, E, S, n_out], high float32 [C, E, S, n_out], the
        freerun energies (list [C][E]) over steps settle_steps .. S-1 (the window [settle_steps * K * unit,
        S * K * unit)) with the gate-net transitions they count (gate_toggles int64 [C, E]), and the traces of the
        watched output nets."""
        fr = import_freerun()
        from freerun.energy import energy as fr_energy
        circuits = self.circuits() if circuits is None else circuits
        models = self.energy_models() if models is None else models
        x = np.asarray(inputs, np.uint8)
        if x.ndim == (3 if per_tick else 2):
            x = x[None]
        if not per_tick:
            x = np.repeat(x[:, :, None, :], K, axis=2)
        E, S, _, I = x.shape
        u, step = self.unit, K * self.unit
        xt = x.reshape(E, S * K, I)
        lanes = [(c, e) for c in range(len(circuits)) for e in range(E)]
        frs = [circuits[c].with_init(_init(circuits[c].init, xt[e, 0])) for c, e in lanes]
        gate_outs = sorted({n for n in self.out_nets if n >= I})
        sim = fr.sim.Simulator(frs, backend=self.cfg.backend, watch=[gate_outs] * len(frs), threads=self.cfg.threads)
        try:
            for b, (c, e) in enumerate(lanes):
                if S * K > 1:
                    t, i = np.nonzero(xt[e, 1:] != xt[e, :-1])
                    t = t + 1
                    sim.add_stimulus(b, zip((t * u).tolist(), i.tolist(), xt[e, t, i].tolist()))
            w0 = sim.run(settle_steps * step) if settle_steps else None
            wins = sim.run(S * step)
            tail = sim.run(S * step + u)             # the last step's duty window ends one unit late
        finally:
            sim.close()
        n_out = len(self.out_nets)
        out = np.zeros((len(circuits), E, S, n_out), np.uint8)
        high = np.zeros((len(circuits), E, S, n_out), np.float32)
        ends = (np.arange(S, dtype=np.int64) + 1) * step
        starts = ends - step
        en = [[None] * E for _ in circuits]
        tog = np.zeros((len(circuits), E), np.int64)
        for b, (c, e) in enumerate(lanes):
            w = wins[b]
            tr = np.concatenate(([w0[b].trace] if w0 else []) + [w.trace, tail[b].trace])
            for j, net in enumerate(self.out_nets):
                if net < I:
                    out[c, e, :, j] = x[e, :, -1, net]
                    high[c, e, :, j] = x[e, :, :, net].mean(-1)
                    continue
                rows = tr[tr[:, 1] == net]
                v0 = int(frs[b].init[net])
                out[c, e, :, j] = _sample(rows[:, 0], rows[:, 2], v0, ends)
                high[c, e, :, j] = (_integral(rows[:, 0], rows[:, 2], v0, ends + u)
                                    - _integral(rows[:, 0], rows[:, 2], v0, starts + u)) / step
            en[c][e] = fr_energy(w, models[c])
            tog[c, e] = int(np.asarray(w.toggles)[I:].sum())
        return dict(outputs=out, high=high, energy=en, gate_toggles=tog, steps=S, step_ticks=step, unit=u,
                    energy_steps=S - settle_steps, traces=[[np.concatenate(([w0[c * E + e].trace] if w0 else []) + [wins[c * E + e].trace])
                             for e in range(E)] for c in range(len(circuits))])


def _init(init, x0):
    a = np.asarray(init, np.uint8).copy()
    a[:len(x0)] = x0
    return a


def _sample(times, vals, v0, points):
    if len(times) == 0:
        return np.full(len(points), v0)
    i = np.searchsorted(times, points, side="right") - 1
    return np.where(i >= 0, vals[np.maximum(i, 0)], v0)


def _integral(times, vals, v0, points):
    bt = np.concatenate([[0], times]).astype(np.int64)
    bv = np.concatenate([[v0], vals]).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(np.diff(bt) * bv[:-1])])
    i = np.searchsorted(bt, points, side="right") - 1
    return cum[i] + bv[i] * (points - bt[i])


# ---- the instruments ------------------------------------------------------------------------------------------------

def _stats(v) -> dict:
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return dict(mean=float("nan"), median=float("nan"), q90=float("nan"), max=float("nan"), min=float("nan"))
    return dict(mean=float(v.mean()), median=float(np.median(v)), q90=float(np.quantile(v, 0.9)),
                max=float(v.max()), min=float(v.min()))


def compare(ref: dict, other_out, other_high) -> dict:
    """Behavioral distance of one circuit's run (outputs [E, S, n_out], high) from a reference run."""
    return dict(disagree=float((other_out != ref["outputs"]).mean()),
                duty_diff=float(np.abs(other_high - ref["high"]).mean()))


def measure(circuit, inputs, K: int, cfg: PhysicalConfig = PhysicalConfig(), *, per_tick: bool = False,
            score_fn=None, library=None, physical: Physical | None = None, settle_steps: int = 0) -> dict:
    """Run the twins and the device population on inputs and compute both instruments. score_fn(duty [E, S, n_out])
    -> mean task error (e.g. baselines.common.EpisodeView.error_of_wires) adds task-score differences."""
    ph = physical or Physical(circuit, cfg, library)
    r = ph.run(inputs, K, per_tick, settle_steps=settle_steps)
    out, high, en = r["outputs"], r["high"], r["energy"]
    S, E = r["energy_steps"], out.shape[1]
    labels = ph.labels()
    twin = dict(outputs=out[0], high=high[0])
    err = [float(score_fn(high[c])) if score_fn is not None else float("nan") for c in range(len(labels))]

    def e_per_step(c):
        tot = sum(x.total for x in en[c]) / (E * S)
        sw = sum(x.switching for x in en[c]) / (E * S)
        it = sum(x.internal for x in en[c]) / (E * S)
        lk = sum(x.leakage for x in en[c]) / (E * S)
        st = sum(getattr(x, "static", 0.0) for x in en[c]) / (E * S)
        pw = float(np.mean([x.leakage / x.duration for x in en[c]]))
        ps = float(np.mean([getattr(x, "static", 0.0) / x.duration for x in en[c]]))
        return dict(total=tot, switching=sw, internal=it, leakage=lk, static=st, leak_power=pw, static_power=ps)

    rows = []
    for c, lab in enumerate(labels):
        row = dict(label=lab, **compare(twin, out[c], high[c]), error=err[c], error_diff=err[c] - err[0],
                   energy=e_per_step(c))
        if c >= 2:
            d = ph.devices[c - 2]
            row.update(process=d.conditions.corner.process, temp=d.conditions.corner.temp,
                       voltage=d.conditions.corner.voltage, scale=d.conditions.scale)
        rows.append(row)
    devs = rows[2:]
    gap = dict(disagree=_stats([d["disagree"] for d in devs]), duty_diff=_stats([d["duty_diff"] for d in devs]),
               error_diff=_stats([d["error_diff"] for d in devs]) if score_fn is not None else None,
               device_error=_stats([d["error"] for d in devs]) if score_fn is not None else None)
    by_proc: dict = {}
    for d in devs:
        by_proc.setdefault(d["process"], []).append(d)
    gap["by_process"] = {p: dict(n=len(v), disagree=float(np.mean([x["disagree"] for x in v])),
                                 error=float(np.mean([x["error"] for x in v])))
                         for p, v in sorted(by_proc.items())}
    step_s = r["step_ticks"] * ph.tick
    energy = dict(per_decision_twin=rows[0]["energy"]["total"], per_decision_nominal=rows[1]["energy"]["total"],
                  per_decision_devices=_stats([d["energy"]["total"] for d in devs]),
                  leak_power_nominal=rows[1]["energy"]["leak_power"], twin=rows[0]["energy"],
                  point_twin=(rows[0]["energy"]["total"], err[0]),
                  point_devices=(float(np.mean([d["energy"]["total"] for d in devs])) if devs else float("nan"),
                                 float(np.mean([d["error"] for d in devs])) if devs else float("nan")))
    return dict(unit_ticks=ph.unit, step_seconds=step_s, K=K, steps=S, episodes=E, n_devices=len(devs),
                timing=ph.timing, config=dict(ph.cfg.__dict__), mapping=ph.mapping.info, twin_error=err[0],
                nominal=dict(disagree=rows[1]["disagree"], duty_diff=rows[1]["duty_diff"], error=err[1],
                             error_diff=rows[1]["error_diff"]),
                gap=gap, energy=energy, rows=rows)


def idealization_gap(circuit, inputs, K: int, cfg: PhysicalConfig = PhysicalConfig(), **kw) -> dict:
    """Behavioral distance between the idealized twin and the device population (see module docstring)."""
    m = measure(circuit, inputs, K, cfg, **kw)
    return dict(twin_error=m["twin_error"], nominal=m["nominal"], n_devices=m["n_devices"], **m["gap"])


def energy(circuit, inputs, K: int, cfg: PhysicalConfig = PhysicalConfig(), **kw) -> dict:
    """Joules per control step (= per decision) for the twins and the devices, leakage power, and the
    energy-vs-score points (see module docstring)."""
    m = measure(circuit, inputs, K, cfg, **kw)
    return dict(step_seconds=m["step_seconds"], unit_ticks=m["unit_ticks"], **m["energy"])


DESCRIPTOR_NAMES = ("gap_disagree", "gap_duty", "gap_error", "nominal_disagree", "log10_energy_twin",
                    "log10_energy_devices")


def descriptor(m: dict) -> np.ndarray:
    """float32 [6] behavioral descriptor for an archive axis, from a `measure` result (DESCRIPTOR_NAMES; a missing
    score gives gap_error 0)."""
    g, e = m["gap"], m["energy"]
    ge = g["error_diff"]["mean"] if g.get("error_diff") else 0.0
    lg = lambda x: float(np.log10(max(x, 1e-30)))  # noqa: E731
    return np.array([g["disagree"]["mean"], g["duty_diff"]["mean"], ge, m["nominal"]["disagree"],
                     lg(e["per_decision_twin"]), lg(e["per_decision_devices"]["mean"])], np.float32)
