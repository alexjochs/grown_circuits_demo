"""Liberty (.lib) standard-cell libraries: parsing, cell functions as truth tables, NLDM delay annotation.

A small self-contained parser (no third-party dependency) covering what freerun needs:
  * combinational cells: input pins with capacitance, output pins with a boolean `function`;
  * timing arcs per (input pin -> output pin): cell_rise / cell_fall delay tables and rise/fall_transition tables,
    indexed by input transition (index_1) and output load capacitance (index_2), with timing_sense and optional
    `when` conditions (state-dependent arcs, e.g. XOR or MUX).
Sequential cells (ff, latch groups) are reported as unsupported; a level-sensitive latch can be built from a
cyclic gate instead.

A multi-output cell (e.g. a half adder) becomes one freerun gate per output pin, sharing the same inputs.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# Generic Liberty syntax: group  name (args) { statements }, simple attribute  name : value ;
# complex attribute  name (args) ;   Comments /* */ and // ; line continuation backslash.

_TOKEN = re.compile(r'\s+|/\*.*?\*/|//[^\n]*|\\\n|"(?:[^"\\]|\\.)*"|[{}();:,]|[^\s{}();:,"]+', re.S)


@dataclass
class Group:
    kind: str
    args: list
    attrs: dict = field(default_factory=dict)       # simple attributes: name -> value (last wins)
    complex: dict = field(default_factory=dict)     # complex attributes: name -> list of arg lists
    groups: list = field(default_factory=list)

    def sub(self, kind: str):
        return [g for g in self.groups if g.kind == kind]


def _tokens(text: str):
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        if tok[0].isspace() or tok.startswith("/*") or tok.startswith("//") or tok == "\\\n":
            continue
        yield tok


def _unquote(tok: str) -> str:
    return tok[1:-1] if tok.startswith('"') else tok


def parse(text: str) -> Group:
    toks = list(_tokens(text))
    pos = 0

    def statement(parent: Group):
        nonlocal pos
        name = toks[pos]
        pos += 1
        if toks[pos] == ":":
            pos += 1
            vals = []
            while toks[pos] != ";" and toks[pos] != "}":
                vals.append(_unquote(toks[pos]))
                pos += 1
            if toks[pos] == ";":
                pos += 1
            parent.attrs[name] = " ".join(vals)
            return
        if toks[pos] != "(":
            raise ValueError(f"liberty: unexpected token {toks[pos]!r} after {name!r}")
        pos += 1
        args = []
        while toks[pos] != ")":
            if toks[pos] != ",":
                args.append(_unquote(toks[pos]))
            pos += 1
        pos += 1
        if pos < len(toks) and toks[pos] == "{":
            pos += 1
            g = Group(name, args)
            while toks[pos] != "}":
                statement(g)
            pos += 1
            parent.groups.append(g)
        else:
            if pos < len(toks) and toks[pos] == ";":
                pos += 1
            parent.complex.setdefault(name, []).append(args)

    root = Group("<root>", [])
    while pos < len(toks):
        statement(root)
    if len(root.groups) != 1 or root.groups[0].kind != "library":
        raise ValueError("liberty: expected exactly one library group")
    return root.groups[0]


# ---------------------------------------------------------------------------------------------------------------------
# Boolean functions (Liberty syntax): ! and postfix ' (not), & * and juxtaposition (and), | + (or), ^ (xor), ( ), 0 1

_FTOK = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_\[\]\.]*|[01]|[!'&*|+^()])")


def parse_function(expr: str):
    """Returns a function env -> 0/1 and the set of variable names used."""
    toks = []
    i = 0
    while i < len(expr):
        m = _FTOK.match(expr, i)
        if not m:
            if expr[i:].strip() == "":
                break
            raise ValueError(f"bad function {expr!r} at {expr[i:]!r}")
        toks.append(m.group(1))
        i = m.end()
    pos = 0
    names = set()

    def peek():
        return toks[pos] if pos < len(toks) else None

    def p_or():
        nonlocal pos
        a = p_xor()
        while peek() in ("|", "+"):
            pos += 1
            b = p_xor()
            a = (lambda x, y: lambda e: x(e) | y(e))(a, b)
        return a

    def p_xor():
        nonlocal pos
        a = p_and()
        while peek() == "^":
            pos += 1
            b = p_and()
            a = (lambda x, y: lambda e: x(e) ^ y(e))(a, b)
        return a

    def p_and():
        nonlocal pos
        a = p_unary()
        while True:
            t = peek()
            if t in ("&", "*"):
                pos += 1
            elif t is None or t in ("|", "+", "^", ")"):
                return a
            b = p_unary()  # explicit or implicit (juxtaposition) AND
            a = (lambda x, y: lambda e: x(e) & y(e))(a, b)

    def p_unary():
        nonlocal pos
        if peek() == "!":
            pos += 1
            a = p_unary()
            return lambda e: 1 - a(e)
        a = p_atom()
        while peek() == "'":
            pos += 1
            a = (lambda x: lambda e: 1 - x(e))(a)
        return a

    def p_atom():
        nonlocal pos
        t = peek()
        pos += 1
        if t == "(":
            a = p_or()
            if peek() != ")":
                raise ValueError(f"unbalanced parentheses in {expr!r}")
            pos += 1
            return a
        if t in ("0", "1"):
            v = int(t)
            return lambda e: v
        if t is None or not (t[0].isalpha() or t[0] == "_"):
            raise ValueError(f"unexpected {t!r} in {expr!r}")
        names.add(t)
        return lambda e: e[t]

    f = p_or()
    if pos != len(toks):
        raise ValueError(f"trailing tokens in {expr!r}")
    return f, names


def function_truth_table(expr: str, pins: list[str]) -> int:
    f, names = parse_function(expr)
    unknown = names - set(pins)
    if unknown:
        raise ValueError(f"function {expr!r} uses unknown pins {sorted(unknown)}")
    tt = 0
    for r in range(1 << len(pins)):
        env = {p: (r >> i) & 1 for i, p in enumerate(pins)}
        tt |= (f(env) & 1) << r
    return tt


# ---------------------------------------------------------------------------------------------------------------------
# Library model

@dataclass
class Table:
    """NLDM lookup table: values[i1, i2] with index_1 = input transition, index_2 = output load (library units)."""
    index_1: np.ndarray
    index_2: np.ndarray
    values: np.ndarray

    def __call__(self, slew: float, load: float) -> float:
        """Bilinear interpolation, linear extrapolation outside the grid (as in common STA practice)."""
        return _bilinear(self.index_1, self.index_2, self.values, slew, load)


def _segment(axis: np.ndarray, x: float):
    if len(axis) == 1:
        return 0, 0, 0.0
    i = int(np.clip(np.searchsorted(axis, x) - 1, 0, len(axis) - 2))
    x0, x1 = axis[i], axis[i + 1]
    return i, i + 1, (x - x0) / (x1 - x0)


def _bilinear(i1, i2, v, x, y) -> float:
    if v.ndim == 1:  # one-dimensional table: indexed by load if index_1 is absent
        a, b, u = _segment(i1, x if len(i1) == len(v) else y)
        return float(v[a] + (v[b] - v[a]) * u)
    a, b, u = _segment(i1, x)
    c, d, w = _segment(i2, y)
    return float((1 - u) * (1 - w) * v[a, c] + u * (1 - w) * v[b, c] + (1 - u) * w * v[a, d] + u * w * v[b, d])


@dataclass
class Arc:
    related_pin: str
    sense: str                       # positive_unate | negative_unate | non_unate
    when: str | None
    cell_rise: Table | None
    cell_fall: Table | None
    rise_transition: Table | None
    fall_transition: Table | None


@dataclass
class PowerArc:
    """internal_power group: energy per transition (library energy unit = cap unit x voltage unit^2).
    Output pins: tables over (input transition of related_pin, output load). Input pins: over input transition."""
    related_pin: str | None
    when: str | None
    rise: Table | None               # energy for a rising transition of the pin owning the group
    fall: Table | None


@dataclass
class OutputPin:
    name: str
    function: str
    tt: int                          # over the cell's input pins, in Cell.inputs order
    arcs: dict[str, list[Arc]]       # related input pin -> arcs (several when state-dependent)
    power: dict[str, list[PowerArc]] = field(default_factory=dict)  # related input pin -> internal power


@dataclass
class Cell:
    name: str
    area: float
    inputs: list[str]                # pin order of the freerun gates made from this cell
    cap: dict[str, float]            # input pin capacitance (library cap unit)
    outputs: list[OutputPin]
    leakage: float = 0.0             # cell_leakage_power (library leakage unit), the state average
    input_power: dict[str, list[PowerArc]] = field(default_factory=dict)  # input pin -> internal power
    leakage_states: list = field(default_factory=list)  # [(when, value)] state-dependent leakage_power groups


@dataclass
class WireLoad:
    """Liberty wire_load model: length(fanout) from the fanout_length table, extrapolated with slope."""
    cap_per_length: float            # library cap unit per length unit
    slope: float
    fanout_length: dict[int, float]

    def cap(self, fanout: int) -> float:
        if fanout <= 0:
            return 0.0
        fl = self.fanout_length
        if fanout in fl:
            length = fl[fanout]
        else:
            top = max(fl) if fl else 0
            length = (fl[top] if fl else 0.0) + self.slope * (fanout - top)
        return self.cap_per_length * length


@dataclass
class Library:
    name: str
    time_unit_s: float
    cap_unit_f: float
    voltage: float
    cells: dict[str, Cell]
    unsupported: dict[str, str]      # cell name -> reason
    temperature: float = float("nan")
    leakage_unit_w: float = 1e-9     # leakage_power_unit
    voltage_unit_v: float = 1.0
    wire_loads: dict[str, WireLoad] = field(default_factory=dict)
    default_wire_load: str | None = None
    path: str | None = None

    def __getitem__(self, name: str) -> Cell:
        return self.cells[name]

    @property
    def energy_unit_j(self) -> float:
        """Unit of internal_power tables: capacitive_load_unit x voltage_unit^2."""
        return self.cap_unit_f * self.voltage_unit_v ** 2


_UNITS = {"fs": 1e-15, "ps": 1e-12, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0,
          "ff": 1e-15, "pf": 1e-12, "nf": 1e-9,
          "pw": 1e-12, "nw": 1e-9, "uw": 1e-6, "mw": 1e-3, "w": 1.0, "mv": 1e-3, "v": 1.0}


def _unit(s: str) -> float:
    m = re.fullmatch(r"\s*([0-9.eE+-]*)\s*([a-zA-Z]+)\s*", s)
    if not m:
        raise ValueError(f"bad unit {s!r}")
    return float(m.group(1) or 1) * _UNITS[m.group(2).lower()]


def _floats(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.replace("\\", " ").replace(",", " ").split()], dtype=np.float64)


def _table(g: Group | None, templates: dict) -> Table | None:
    if g is None:
        return None
    tpl = templates.get(g.args[0] if g.args else "", {})
    i1 = _floats(g.complex["index_1"][0][0]) if "index_1" in g.complex else tpl.get("index_1")
    i2 = _floats(g.complex["index_2"][0][0]) if "index_2" in g.complex else tpl.get("index_2")
    rows = [_floats(r) for r in g.complex["values"][0]]
    vals = np.array(rows) if len(rows) > 1 else rows[0]
    if g.args and g.args[0] == "scalar":
        return Table(np.array([0.0]), np.array([0.0]), np.array([vals.ravel()[0]]))
    var1 = tpl.get("variable_1", "input_net_transition")
    if var1 == "total_output_net_capacitance":  # normalise so index_1 is always the input transition
        i1, i2 = i2, i1
        vals = vals.T if vals.ndim == 2 else vals
        if i1 is None:  # load-only table
            return Table(np.array([0.0]), i2, vals.reshape(1, -1))
    if i2 is None:
        return Table(i1, np.array([0.0]), vals.reshape(-1, 1))
    return Table(i1, i2, vals.reshape(len(i1), len(i2)))


def load(path) -> Library:
    """Parse a Liberty file into a Library of combinational cells."""
    lib = parse(Path(path).read_text())
    a = lib.attrs
    time_unit = _unit(a.get("time_unit", "1ns"))
    cl = lib.complex.get("capacitive_load_unit", [["1", "pf"]])[0]
    cap_unit = float(cl[0]) * _UNITS[cl[1].lower()]
    voltage = float(a.get("nom_voltage", "nan"))
    temperature = float(a.get("nom_temperature", "nan"))
    leakage_unit = _unit(a.get("leakage_power_unit", "1nW"))
    voltage_unit = _unit(a.get("voltage_unit", "1V"))
    templates = {}
    for t in lib.sub("lu_table_template") + lib.sub("power_lut_template"):
        d = {k: v for k, v in t.attrs.items() if k.startswith("variable_")}
        for k in ("index_1", "index_2"):
            if k in t.complex:
                d[k] = _floats(t.complex[k][0][0])
        templates[t.args[0]] = d
    wire_loads = {}
    for w in lib.sub("wire_load"):
        fl = {int(float(x[0])): float(x[1]) for x in w.complex.get("fanout_length", [])}
        wire_loads[w.args[0]] = WireLoad(float(w.attrs.get("capacitance", 0.0)), float(w.attrs.get("slope", 0.0)), fl)

    def power_arcs(pin_group, related: bool) -> dict[str, list[PowerArc]]:
        out: dict[str, list[PowerArc]] = {}
        for pg in pin_group.sub("internal_power"):
            rise = _table(pg.sub("rise_power")[0], templates) if pg.sub("rise_power") else None
            fall = _table(pg.sub("fall_power")[0], templates) if pg.sub("fall_power") else None
            if pg.sub("power"):  # one table for both directions
                rise = fall = _table(pg.sub("power")[0], templates)
            rps = pg.attrs.get("related_pin", "").split() if related else [pin_group.args[0]]
            for rp in rps:
                out.setdefault(rp, []).append(PowerArc(rp if related else None, pg.attrs.get("when"), rise, fall))
        return out
    cells, unsupported = {}, {}
    for cg in lib.sub("cell"):
        name = cg.args[0]
        if cg.sub("ff") or cg.sub("latch") or cg.sub("ff_bank") or cg.sub("latch_bank") or cg.sub("statetable"):
            unsupported[name] = "sequential"
            continue
        pins = cg.sub("pin")
        ins = [p for p in pins if p.attrs.get("direction") == "input"]
        outs = [p for p in pins if p.attrs.get("direction") == "output" and "function" in p.attrs]
        if not outs:
            unsupported[name] = "no output with a function"
            continue
        if any("three_state" in p.attrs for p in pins):
            unsupported[name] = "three-state output"
            continue
        inputs = [p.args[0] for p in ins]
        if len(inputs) > 6:
            unsupported[name] = f"{len(inputs)} inputs (max 6)"
            continue
        cap = {p.args[0]: float(p.attrs.get("capacitance", 0.0)) for p in ins}
        outputs = []
        try:
            for op in outs:
                fn = op.attrs["function"]
                tt = function_truth_table(fn, inputs)
                arcs: dict[str, list[Arc]] = {}
                for tg in op.sub("timing"):
                    if tg.attrs.get("timing_type", "combinational") not in ("combinational", "combinational_rise",
                                                                            "combinational_fall"):
                        continue
                    first = {k: (tg.sub(k)[0] if tg.sub(k) else None)
                             for k in ("cell_rise", "cell_fall", "rise_transition", "fall_transition")}
                    for rp in tg.attrs["related_pin"].split():
                        arcs.setdefault(rp, []).append(Arc(
                            rp, tg.attrs.get("timing_sense", "non_unate"), tg.attrs.get("when"),
                            *(_table(first[k], templates) for k in ("cell_rise", "cell_fall", "rise_transition",
                                                                     "fall_transition"))))
                outputs.append(OutputPin(op.args[0], fn, tt, arcs, power_arcs(op, True)))
        except ValueError as e:
            unsupported[name] = str(e)
            continue
        in_power = {}
        for p in ins:
            in_power.update(power_arcs(p, False))
        states = [(lg.attrs.get("when"), float(lg.attrs.get("value", 0.0))) for lg in cg.sub("leakage_power")]
        cells[name] = Cell(name, float(cg.attrs.get("area", "nan")), inputs, cap, outputs,
                           float(cg.attrs.get("cell_leakage_power", 0.0)), in_power, states)
    return Library(lib.args[0] if lib.args else "", time_unit, cap_unit, voltage, cells, unsupported, temperature,
                   leakage_unit, voltage_unit, wire_loads, a.get("default_wire_load"), str(path))


# ---------------------------------------------------------------------------------------------------------------------
# NLDM delay calculation

def arc_delays(lib: Library, cell: Cell, out: OutputPin, pin: str, slew_rise: float, slew_fall: float, load: float,
               combine: str = "mean"):
    """(rise delay, fall delay, rise transition, fall transition) of the arc pin -> out, in library time units.

    Input slew is chosen by unateness: an output rise follows an input fall for a negative-unate arc, an input rise
    for a positive-unate arc, and either (the worse of the two) for a non-unate arc. State-dependent arcs (several
    `when` conditions for the same pin) are combined with `combine` ("mean", "max" or "min")."""
    arcs = out.arcs.get(pin)
    if not arcs:
        raise ValueError(f"{cell.name}: no timing arc {pin} -> {out.name}")
    red = {"mean": np.mean, "max": np.max, "min": np.min}[combine]

    def one(arc: Arc, table: Table | None, out_rises: bool) -> float:
        if table is None:
            return math.nan
        if arc.sense == "negative_unate":
            s = slew_fall if out_rises else slew_rise
            return table(s, load)
        if arc.sense == "positive_unate":
            s = slew_rise if out_rises else slew_fall
            return table(s, load)
        return max(table(slew_rise, load), table(slew_fall, load))

    vals = []
    for get, rises in (("cell_rise", True), ("cell_fall", False), ("rise_transition", True),
                       ("fall_transition", False)):
        xs = [one(a, getattr(a, get), rises) for a in arcs]
        xs = [x for x in xs if not math.isnan(x)]
        vals.append(float(red(xs)) if xs else math.nan)
    return tuple(vals)


# ---------------------------------------------------------------------------------------------------------------------
# Internal energy (docs/SPEC.md §11)

def output_internal_energy(cell: Cell, out: OutputPin, pin: str, slew_rise: float, slew_fall: float,
                           load: float) -> tuple[float, float]:
    """(energy per output rise, per output fall) caused through related input pin `pin`, in library energy units,
    averaged over `when` conditions. slew_rise / slew_fall: of the pin's input net; the slew is chosen by the
    unateness of the pin's timing arc, as for delays. No power data gives (0, 0)."""
    groups = out.power.get(pin)
    if not groups:
        return 0.0, 0.0
    arcs = out.arcs.get(pin)
    sense = arcs[0].sense if arcs else "non_unate"

    def at(table, out_rises):
        if table is None:
            return math.nan
        if sense == "negative_unate":
            return table(slew_fall if out_rises else slew_rise, load)
        if sense == "positive_unate":
            return table(slew_rise if out_rises else slew_fall, load)
        return 0.5 * (table(slew_rise, load) + table(slew_fall, load))

    r = [x for x in (at(g.rise, True) for g in groups) if not math.isnan(x)]
    f = [x for x in (at(g.fall, False) for g in groups) if not math.isnan(x)]
    return (float(np.mean(r)) if r else 0.0, float(np.mean(f)) if f else 0.0)


def input_internal_energy(cell: Cell, pin: str, slew_rise: float, slew_fall: float) -> tuple[float, float]:
    """(energy per rise, per fall) of an input pin, library energy units, averaged over `when` conditions."""
    groups = cell.input_power.get(pin, [])
    r = [g.rise(slew_rise, 0.0) for g in groups if g.rise is not None]
    f = [g.fall(slew_fall, 0.0) for g in groups if g.fall is not None]
    return (float(np.mean(r)) if r else 0.0, float(np.mean(f)) if f else 0.0)
