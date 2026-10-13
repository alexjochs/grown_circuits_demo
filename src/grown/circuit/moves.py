"""The edit vocabulary: legal transformations of a circuit graph, a uniform operator over them, a `Proposal`
interface for a learned editor, and the `Edit` record.

A `Move` is fully specified (type, location, parameter), so `apply_move(circuit, move)` is deterministic and an
edit can be replayed from the log. Fields:

  TT_REPLACE     node = gate; a = new truth table (int64 view)
  TT_FLIP        node = gate; a = row index whose bit flips
  REWIRE         node = consumer (OUT = -2 for a circuit output), pin; (a, b) = new source ref
  ADD_NODE       node = new index (len(nodes)); data = encoded node (codec.encode_node);
                 (a, b) = pin spliced to read the new node's port 0, or a = NONE (-3) for a dangling node.
                 When spliced, the new node's pin 0 reads what the spliced pin read (an insertion on that edge).
  DELETE_NODE    node; (a, b) = fallback source. Deletion rule: every pin reading port p of the deleted node is
                 reconnected to the node's input pin min(p, k-1) (a bypass: deletion inverts a spliced insertion).
                 When the node has no inputs, or that pin reads the node itself, the fallback source is used.
                 Later nodes are renumbered down by one.
  DUPLICATE      node = seed; data = the copied node set; copies keep their internal wiring and read the same
                 external sources; copies are appended in set order. (a, b) = one external pin of an original
                 port that is moved to the copy of that port (a = NONE: the copy dangles).
  INSERT_MODULE  node = new index; data = (hash_lo, hash_hi, out_port, n_in, src pairs...); (a, b) = spliced pin
                 that reads the instance's `out_port`, or a = NONE. Like ADD_NODE, instance pin 0 takes over the
                 spliced pin's old source.
  ADJUST_PARAM   node; pin = arc pin or slot index; a = field (RISE, FALL, INIT, DRIVE, SLOT); b = new value
  REGENERATE     data = region (node set). Reserved hook for a learned generator: a no-op unless the operator has
                 a regenerator, whose output is not replayable from the move (future work).
  INLINE_MODULE  node = module instance; replaced by its body (inverse of INSERT_MODULE).

INLINE_MODULE is an addition to the list in the brief: without it a module instance can never be tuned inside.
"""
from __future__ import annotations

import dataclasses
from bisect import bisect_right
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

import numpy as np

from . import codec
from .flatten import inline
from .graph import (DELAY, GATE, INPUT, MAX_ARITY, MODULE, OUT, SLOW_DRIVE, WIRE, Circuit, InvalidCircuit, Node,
                    delete_nodes, random_tt, set_node, set_pin, tt_mask, validate)

NONE = -3


class MoveType(IntEnum):
    TT_REPLACE = 0
    TT_FLIP = 1
    REWIRE = 2
    ADD_NODE = 3
    DELETE_NODE = 4
    DUPLICATE = 5
    INSERT_MODULE = 6
    ADJUST_PARAM = 7
    REGENERATE = 8
    INLINE_MODULE = 9


N_MOVES = len(MoveType)
RISE, FALL, INIT, DRIVE, SLOT = range(5)


def _slow_table():
    from ..sim.slow import table
    return table()
_FIELD = {RISE: "rise", FALL: "fall", INIT: "init", DRIVE: "drive"}
_SLOT_FIELD = {"rise": RISE, "fall": FALL, "init": INIT, "drive": DRIVE}


class Source(IntEnum):
    UNIFORM = 0
    LEARNED = 1
    FALLBACK = 2      # learned proposal failed (None or illegal); a uniform move was used instead


class Fate(IntEnum):
    PENDING = 0
    INVALID = 1       # child failed validation or evaluation
    CULLED = 2        # evaluated, not inserted (lost to the cell's elite or to a sibling)
    SURVIVED = 3      # kept by selection without becoming an elite (for archives that keep non-elites)
    ELITE_NEW = 4     # became the elite of an empty cell
    ELITE_REPLACE = 5  # displaced the cell's elite (its hash is recorded in Edit.displaced)


@dataclass(frozen=True)
class Move:
    type: int
    node: int = -1
    pin: int = -1
    a: int = 0
    b: int = 0
    data: tuple = ()


ZERO_HASH = bytes(16)


@dataclass
class Edit:
    parent: bytes
    child: bytes
    move: Move
    source: int = Source.UNIFORM
    fate: int = Fate.PENDING
    task: int = 0
    generation: int = 0
    parent_fitness: float = float("nan")
    child_fitness: float = float("nan")
    cell: int = -1
    displaced: bytes = ZERO_HASH
    parent_sig: np.ndarray | None = None
    child_sig: np.ndarray | None = None


# --- application --------------------------------------------------------------------------------------------------

def apply_move(c: Circuit, mv: Move, library=None) -> Circuit:
    """Apply a fully specified move. Raises InvalidCircuit (or IndexError / ValueError) if it does not apply."""
    t = mv.type
    if t in (MoveType.TT_REPLACE, MoveType.TT_FLIP):
        nd = c.nodes[mv.node]
        if nd.op != GATE:
            raise InvalidCircuit("truth-table move on a non-gate")
        tt = codec.u64(mv.a) if t == MoveType.TT_REPLACE else nd.tt ^ (1 << mv.a)
        if tt & ~tt_mask(nd.k):
            raise InvalidCircuit("truth table out of range")
        init = tt & 1 if nd.k == 0 else nd.init
        return set_node(c, mv.node, dataclasses.replace(nd, tt=tt, init=init))
    if t == MoveType.REWIRE:
        return set_pin(c, mv.node, mv.pin, (mv.a, mv.b))
    if t == MoveType.ADD_NODE:
        nd, _ = codec.decode_node(mv.data)
        if mv.node != len(c.nodes):
            raise InvalidCircuit("ADD_NODE index must be len(nodes)")
        c2 = Circuit(c.n_in, c.nodes + (nd,), c.outputs)
        return c2 if mv.a == NONE else set_pin(c2, mv.a, mv.b, (mv.node, 0))
    if t == MoveType.DELETE_NODE:
        i = mv.node
        nd = c.nodes[i]
        fallback = (mv.a, mv.b)
        if fallback[0] == i:
            raise InvalidCircuit("fallback source on the deleted node")
        redirect = {}
        for p in range(nd.n_out):
            r = nd.ins[min(p, nd.k - 1)] if nd.k else None
            redirect[(i, p)] = r if r is not None and r[0] != i else fallback
        return delete_nodes(c, [i], redirect)
    if t == MoveType.DUPLICATE:
        S = list(mv.data)
        base = len(c.nodes)
        pos = {s: base + j for j, s in enumerate(S)}
        m = lambda r: (pos[r[0]], r[1]) if r[0] in pos else r  # noqa: E731
        c2 = Circuit(c.n_in, c.nodes + tuple(c.nodes[s].with_ins(m(r) for r in c.nodes[s].ins) for s in S),
                     c.outputs)
        if mv.a == NONE:
            return c2
        r = c.ref_at(mv.a, mv.b)
        if r[0] not in pos or mv.a in pos:
            raise InvalidCircuit("DUPLICATE redirect must move an external pin of a copied port")
        return set_pin(c2, mv.a, mv.b, m(r))
    if t == MoveType.INSERT_MODULE:
        if library is None:
            raise InvalidCircuit("INSERT_MODULE needs a library")
        lo, hi, out_port, n_in = mv.data[:4]
        src = mv.data[4:]
        nd = library.instance(codec.ints_to_hash(lo, hi), [(src[2 * j], src[2 * j + 1]) for j in range(n_in)])
        if mv.node != len(c.nodes):
            raise InvalidCircuit("INSERT_MODULE index must be len(nodes)")
        c2 = Circuit(c.n_in, c.nodes + (nd,), c.outputs)
        return c2 if mv.a == NONE else set_pin(c2, mv.a, mv.b, (mv.node, out_port))
    if t == MoveType.ADJUST_PARAM:
        nd = c.nodes[mv.node]
        if mv.a == SLOT:
            if nd.op != MODULE:
                raise InvalidCircuit("slot adjustment on a non-module")
            d = library.get(nd.module)
            params = list(nd.params or d.defaults())
            params[mv.pin] = mv.b
            return set_node(c, mv.node, dataclasses.replace(nd, params=tuple(params)))
        if nd.op not in (GATE, DELAY):
            raise InvalidCircuit("parameter move on a token without parameters")
        f = _FIELD[mv.a]
        if nd.op == DELAY and nd.drive == SLOW_DRIVE:   # a slow delay: rise carries its level, the rest is fixed
            if f == "rise":
                r, fl = _slow_table().ticks(int(mv.b))
                return set_node(c, mv.node, dataclasses.replace(nd, rise=(r,), fall=(fl,)))
            if f != "init":
                raise InvalidCircuit("a slow delay's delays and drive follow its level")
        if f in ("rise", "fall"):
            v = list(getattr(nd, f))
            v[mv.pin] = mv.b
            return set_node(c, mv.node, dataclasses.replace(nd, **{f: tuple(v)}))
        if f == "init" and nd.op == GATE and nd.k == 0:
            raise InvalidCircuit("constant init is fixed by its truth table")
        return set_node(c, mv.node, dataclasses.replace(nd, **{f: mv.b}))
    if t == MoveType.REGENERATE:
        return c
    if t == MoveType.INLINE_MODULE:
        return inline(c, mv.node, library)
    raise ValueError(f"unknown move type {t}")


# --- proposals ----------------------------------------------------------------------------------------------------

class Proposal(Protocol):
    """A learned editor: a distribution over the same moves. `propose` returns a fully specified Move (or None to
    defer to the uniform operator)."""

    def propose(self, circuit: Circuit, rng: np.random.Generator) -> Move | None: ...


@dataclass
class OperatorConfig:
    weights: dict = field(default_factory=dict)   # MoveType -> relative weight (default 1)
    arity_max: int = 4
    p_const: float = 0.02                         # new gates of arity 0
    kind_weights: tuple = (0.8, 0.1, 0.1)         # new node: gate, delay, wire
    delay_range: tuple = (1, 1)                   # new arcs and delay elements, in substrate ticks
    delay_sigma: float = 0.5                      # ADJUST: delay *= 2 ** N(0, sigma)
    delay_clamp: bool = False                     # ADJUST keeps delays inside delay_range (a fixed delay scale)
    drive_max: int = 8
    dup_max: int = 4                              # DUPLICATE / REGENERATE region size bound
    max_nodes: int = 512                          # growth moves are disabled at this size
    p_splice: float = 0.5
    regen_noop: bool = False                      # offer REGENERATE even without a regenerator
    attempts: int = 32
    fast: bool = True                             # packed path for module-free parents (same moves, draws and
                                                  # children as the Node path; see circuit.packed)
    slow_weight: float = 0.0                      # ADD_NODE weight of a slow delay element (grown.sim.slow), next
                                                  # to kind_weights; > 0 puts it in the vocabulary (Node path only)
    slow_step: int = 2                            # a slow delay's level moves by 1 .. slow_step (or is redrawn)


def _cdf(w) -> list:
    """numpy Generator.choice(len(w), p=w / w.sum())'s cumulative table: choice draws one rng.random() u and
    returns bisect_right(cdf, u). Replaying it with the same table gives the same index from the same draw."""
    w = np.asarray(w, float)
    cdf = (w / w.sum()).cumsum()
    cdf /= cdf[-1]
    return cdf.tolist()


class UniformOperator:
    """Samples a move type (uniformly over the applicable ones, or by `weights`), then a location and a parameter
    uniformly, applies it and validates the child.

    Two implementations of the same operator: the Node path (`sample`, `apply`) works on any circuit; the packed
    path (`sample_packed`, `apply_packed_move`, used by `mutate` when the parent is module-free and valid, see
    circuit.packed) consumes exactly the same random draws, produces the same moves and equal children, and costs
    O(edit) Python work instead of O(circuit). tests/grown/test_packed.py checks the two against each other."""

    def __init__(self, config: OperatorConfig | None = None, library=None, regenerator=None):
        self.cfg = config or OperatorConfig()
        self.library = library
        self.regenerator = regenerator
        self._type_tables: dict = {}
        self._kind_cdf = (_cdf(np.array(self.cfg.kind_weights) / sum(self.cfg.kind_weights))
                          if sum(self.cfg.kind_weights) > 0 else None)

    # -- applicability
    def applicable(self, c: Circuit) -> list:
        n = len(c.nodes)
        has_gate = any(nd.op == GATE for nd in c.nodes)
        grow = n < self.cfg.max_nodes
        lib = self.library is not None and len(self.library.active()) > 0
        ok = {
            MoveType.TT_REPLACE: has_gate,
            MoveType.TT_FLIP: has_gate,
            MoveType.REWIRE: bool(c.pins()) and c.n_ports() > 1,
            MoveType.ADD_NODE: grow,
            MoveType.DELETE_NODE: n > 0,
            MoveType.DUPLICATE: n > 0 and grow,
            MoveType.INSERT_MODULE: lib and grow,
            MoveType.ADJUST_PARAM: any(self._params(nd) for nd in c.nodes),
            MoveType.REGENERATE: n > 0 and (self.regenerator is not None or self.cfg.regen_noop),
            MoveType.INLINE_MODULE: self.library is not None and any(nd.op == MODULE for nd in c.nodes),
        }
        return [t for t in MoveType if ok[t] and self.cfg.weights.get(t, 1.0) > 0]

    def _params(self, nd: Node) -> list:
        """(field, pin) pairs adjustable on this token."""
        if nd.op == GATE:
            out = [(f, p) for f in (RISE, FALL) for p in range(nd.k)] + [(DRIVE, 0)]
            return out + ([(INIT, 0)] if nd.k else [])
        if nd.op == DELAY:
            if nd.drive == SLOW_DRIVE:
                return [(RISE, 0), (INIT, 0)]      # RISE: the slow element's bias level (its delays follow)
            return [(RISE, 0), (FALL, 0), (INIT, 0), (DRIVE, 0)]
        if nd.op == MODULE and self.library is not None:
            return [(SLOT, s) for s in range(len(self.library.get(nd.module).slots))]
        return []

    # -- sampling
    def sample(self, c: Circuit, rng: np.random.Generator) -> Move:
        types = self.applicable(c)
        if not types:
            raise InvalidCircuit("no applicable move")
        w = np.array([self.cfg.weights.get(t, 1.0) for t in types], float)
        for _ in range(self.cfg.attempts):
            t = types[int(rng.choice(len(types), p=w / w.sum()))]
            mv = getattr(self, f"_s_{t.name.lower()}")(c, rng)
            if mv is not None:
                return mv
        raise InvalidCircuit("could not sample a move")

    def _port(self, c, rng, n_extra_nodes=0, exclude=None):
        """A uniform source port among inputs, node ports and (n_extra_nodes) single-port nodes being added."""
        ports = c.ports() + [(len(c.nodes) + j, 0) for j in range(n_extra_nodes)]
        if exclude is not None:
            ports = [p for p in ports if p[0] != exclude]
        if not ports:
            return None
        return ports[int(rng.integers(len(ports)))]

    def _gate_ix(self, c, rng):
        g = [i for i, nd in enumerate(c.nodes) if nd.op == GATE]
        return g[int(rng.integers(len(g)))] if g else None

    def _s_tt_replace(self, c, rng):
        i = self._gate_ix(c, rng)
        if i is None:
            return None
        k = c.nodes[i].k
        tt = random_tt(k, rng)
        if tt == c.nodes[i].tt:
            tt ^= 1 << int(rng.integers(1 << k))
        return Move(MoveType.TT_REPLACE, i, a=codec.s64(tt))

    def _s_tt_flip(self, c, rng):
        i = self._gate_ix(c, rng)
        if i is None:
            return None
        return Move(MoveType.TT_FLIP, i, a=int(rng.integers(1 << c.nodes[i].k)))

    def _s_rewire(self, c, rng):
        pins = c.pins()
        consumer, pin = pins[int(rng.integers(len(pins)))]
        old = c.ref_at(consumer, pin)
        ports = [p for p in c.ports() if p != old]
        if not ports:
            return None
        s = ports[int(rng.integers(len(ports)))]
        return Move(MoveType.REWIRE, consumer, pin, s[0], s[1])

    def _splice(self, c, rng):
        if rng.random() >= self.cfg.p_splice:
            return None
        pins = c.pins()
        return pins[int(rng.integers(len(pins)))] if pins else None

    def _delay(self, rng):
        lo, hi = self.cfg.delay_range
        if lo == hi:
            return int(lo)               # rng.integers(lo, lo + 1) returns lo without consuming a draw
        return int(rng.integers(lo, hi + 1))

    def _s_add_node(self, c, rng):
        if self.cfg.slow_weight > 0:
            w = np.array(tuple(self.cfg.kind_weights) + (self.cfg.slow_weight,), float)
            kind = int(rng.choice(4, p=w / w.sum()))
        else:
            kind = int(rng.choice(3, p=np.array(self.cfg.kind_weights) / sum(self.cfg.kind_weights)))
        if kind == 0:
            k = 0 if rng.random() < self.cfg.p_const else int(rng.integers(1, self.cfg.arity_max + 1))
        else:
            k = 1
        ins = [self._port(c, rng, 1) for _ in range(k)]
        splice = self._splice(c, rng)
        if splice is not None and k:
            ins[0] = c.ref_at(*splice)
        if kind == 0:
            tt = random_tt(k, rng)
            nd = Node(GATE, tuple(ins), tt, tuple(self._delay(rng) for _ in range(k)),
                      tuple(self._delay(rng) for _ in range(k)), tt & 1 if k == 0 else int(rng.integers(2)))
        elif kind == 1:
            nd = Node(DELAY, tuple(ins), 0, (self._delay(rng),), (self._delay(rng),), int(rng.integers(2)))
        elif kind == 3:                            # a slow delay element at a uniformly drawn bias level
            r, f = _slow_table().ticks(int(rng.integers(_slow_table().n_levels)))
            nd = Node(DELAY, tuple(ins), 0, (r,), (f,), int(rng.integers(2)), SLOW_DRIVE)
        else:
            nd = Node(WIRE, tuple(ins))
        a, b = splice if splice is not None else (NONE, 0)
        return Move(MoveType.ADD_NODE, len(c.nodes), -1, a, b, tuple(codec.encode_node(nd)))

    def _s_delete_node(self, c, rng):
        i = int(rng.integers(len(c.nodes)))
        fb = self._port(c, rng, exclude=i)
        if fb is None:
            return None
        return Move(MoveType.DELETE_NODE, i, -1, fb[0], fb[1])

    def _region(self, c, rng, seed):
        size = int(rng.integers(1, self.cfg.dup_max + 1))
        nbr = [set() for _ in c.nodes]
        for i, nd in enumerate(c.nodes):
            for s, _ in nd.ins:
                if s != INPUT and s != i:
                    nbr[i].add(s)
                    nbr[s].add(i)
        S, frontier = [seed], set(nbr[seed])
        while len(S) < size and frontier:
            f = sorted(frontier)
            j = f[int(rng.integers(len(f)))]
            S.append(j)
            frontier |= nbr[j]
            frontier -= set(S)
        return sorted(S)

    def _s_duplicate(self, c, rng):
        S = self._region(c, rng, int(rng.integers(len(c.nodes))))
        if len(c.nodes) + len(S) > self.cfg.max_nodes:
            return None
        Sset = set(S)
        ext = [(ci, p) for ci, p in c.pins() if ci not in Sset and c.ref_at(ci, p)[0] in Sset]
        a, b = ext[int(rng.integers(len(ext)))] if ext and rng.random() < 0.5 else (NONE, 0)
        return Move(MoveType.DUPLICATE, S[0], -1, a, b, tuple(S))

    def _s_insert_module(self, c, rng):
        act = self.library.active()
        h = act[int(rng.integers(len(act)))]
        d = self.library.get(h)
        src = [self._port(c, rng, 1) for _ in range(d.n_in)]
        splice = self._splice(c, rng)
        if splice is not None and d.n_in:
            src[0] = c.ref_at(*splice)
        out_port = int(rng.integers(d.n_out)) if d.n_out else 0
        if d.n_out == 0:
            splice = None
        a, b = splice if splice is not None else (NONE, 0)
        lo, hi = codec.hash_to_ints(h)
        flat = tuple(x for r in src for x in r)
        return Move(MoveType.INSERT_MODULE, len(c.nodes), -1, a, b, (lo, hi, out_port, d.n_in) + flat)

    def _s_adjust_param(self, c, rng):
        cand = [i for i, nd in enumerate(c.nodes) if self._params(nd)]
        if not cand:
            return None
        i = cand[int(rng.integers(len(cand)))]
        nd = c.nodes[i]
        opts = self._params(nd)
        f, p = opts[int(rng.integers(len(opts)))]
        if f == SLOT:
            d = self.library.get(nd.module)
            sf = _SLOT_FIELD[self.library.slot_field(nd.module, p)]
            old = (nd.params or d.defaults())[p]
            return Move(MoveType.ADJUST_PARAM, i, p, SLOT, self._new_value(sf, old, rng))
        if nd.op == DELAY and nd.drive == SLOW_DRIVE and f == RISE:
            return Move(MoveType.ADJUST_PARAM, i, 0, RISE, self._new_level(nd, rng))
        old = getattr(nd, _FIELD[f])
        old = old[p] if f in (RISE, FALL) else old
        return Move(MoveType.ADJUST_PARAM, i, p, f, self._new_value(f, old, rng))

    def _new_level(self, nd, rng) -> int:
        """A slow delay's new bias level: a uniform redraw with probability 1/4, else a step of 1 .. slow_step up or
        down (reflected at the ends)."""
        t = _slow_table()
        n = t.n_levels
        old = t.level_of(nd.rise[0], nd.fall[0])
        if old < 0 or rng.random() < 0.25:
            v = int(rng.integers(n))
            return v if v != old or n == 1 else (v + 1) % n
        step = int(rng.integers(1, max(int(self.cfg.slow_step), 1) + 1)) * (1 if rng.random() < 0.5 else -1)
        v = old + step
        if not 0 <= v < n:
            v = old - step
        return int(min(max(v, 0), n - 1))

    def _new_value(self, f, old, rng):
        if f == INIT:
            return 1 - old
        if f == DRIVE:
            v = old + (1 if rng.random() < 0.5 else -1)
            v = min(max(v, 1), self.cfg.drive_max)
            return v if v != old else (old % self.cfg.drive_max) + 1
        v = int(round(old * 2.0 ** rng.normal(0, self.cfg.delay_sigma)))
        if v == old:
            v = old + (1 if rng.random() < 0.5 or old == 1 else -1)
        if self.cfg.delay_clamp:
            lo, hi = self.cfg.delay_range
            return min(max(v, int(lo)), int(hi))
        return min(max(v, 1), 2**31 - 1)

    def _s_regenerate(self, c, rng):
        S = self._region(c, rng, int(rng.integers(len(c.nodes))))
        return Move(MoveType.REGENERATE, S[0], -1, 0, 0, tuple(S))

    def _s_inline_module(self, c, rng):
        m = [i for i, nd in enumerate(c.nodes) if nd.op == MODULE]
        return Move(MoveType.INLINE_MODULE, m[int(rng.integers(len(m)))]) if m else None

    # -- the packed path: the same samplers on circuit.packed arrays ------------------------------------------
    def _types_packed(self, p):
        """(types, cdf, samplers) for a packed (module-free) circuit: `applicable` and sample's choice table."""
        c = p.cache()
        facts = c.get("_move_facts")
        if facts is None:
            facts = c["_move_facts"] = (len(p.gate_ix()) > 0, p.P + p.n_out > 0 and p.n_in + p.n > 1)
        n = p.n
        grow = n < self.cfg.max_nodes
        lib = self.library is not None and len(self.library.active()) > 0
        key = (facts[0], facts[1], grow, n > 0, n > 0 and grow, lib and grow, n > p.n_wire,
               n > 0 and (self.regenerator is not None or self.cfg.regen_noop))
        tab = self._type_tables.get(key)
        if tab is None:
            has_gate, rewire, grow, nonempty, dup, ins_mod, params, regen = key
            ok = {MoveType.TT_REPLACE: has_gate, MoveType.TT_FLIP: has_gate, MoveType.REWIRE: rewire,
                  MoveType.ADD_NODE: grow, MoveType.DELETE_NODE: nonempty, MoveType.DUPLICATE: dup,
                  MoveType.INSERT_MODULE: ins_mod, MoveType.ADJUST_PARAM: params, MoveType.REGENERATE: regen,
                  MoveType.INLINE_MODULE: False}
            types = [t for t in MoveType if ok[t] and self.cfg.weights.get(t, 1.0) > 0]
            cdf = _cdf([self.cfg.weights.get(t, 1.0) for t in types]) if types else []
            fns = [getattr(self, f"_f_{t.name.lower()}") for t in types]
            tab = self._type_tables[key] = (types, cdf, fns)
        return tab

    def sample_packed(self, p, rng: np.random.Generator) -> Move:
        types, cdf, fns = self._types_packed(p)
        if not types:
            raise InvalidCircuit("no applicable move")
        for _ in range(self.cfg.attempts):
            mv = fns[bisect_right(cdf, rng.random())](p, rng)
            if mv is not None:
                return mv
        raise InvalidCircuit("could not sample a move")

    def _f_port(self, p, rng, n_extra_nodes=0, exclude=None):
        N = p.n_in + p.n + n_extra_nodes - (1 if exclude is not None and 0 <= exclude < p.n else 0)
        if N <= 0:
            return None
        j = int(rng.integers(N))
        if j < p.n_in:
            return (INPUT, j)
        j -= p.n_in
        if exclude is not None and 0 <= exclude <= j:
            j += 1
        return (j, 0)

    def _f_gate(self, p, rng):
        g = p.gate_ix()
        return int(g[int(rng.integers(len(g)))]) if len(g) else None

    def _f_tt_replace(self, p, rng):
        i = self._f_gate(p, rng)
        if i is None:
            return None
        k = int(p.ptr[i + 1] - p.ptr[i])
        tt = random_tt(k, rng)
        if tt == int(p.tt[i]):
            tt ^= 1 << int(rng.integers(1 << k))
        return Move(MoveType.TT_REPLACE, i, a=codec.s64(tt))

    def _f_tt_flip(self, p, rng):
        i = self._f_gate(p, rng)
        if i is None:
            return None
        return Move(MoveType.TT_FLIP, i, a=int(rng.integers(1 << int(p.ptr[i + 1] - p.ptr[i]))))

    def _f_rewire(self, p, rng):
        consumer, pin = p.pin_at(int(rng.integers(p.P + p.n_out)))
        old = p.ref_at(consumer, pin)
        pos_old = old[1] if old[0] == INPUT else p.n_in + old[0]
        m = p.n_in + p.n - 1                       # every port but `old` (a valid parent's refs are ports)
        if m <= 0:
            return None
        j = int(rng.integers(m))
        if j >= pos_old:
            j += 1
        s = (INPUT, j) if j < p.n_in else (j - p.n_in, 0)
        return Move(MoveType.REWIRE, consumer, pin, s[0], s[1])

    def _f_splice(self, p, rng):
        if rng.random() >= self.cfg.p_splice:
            return None
        npins = p.P + p.n_out
        return p.pin_at(int(rng.integers(npins))) if npins else None

    def _f_add_node(self, p, rng):
        kind = bisect_right(self._kind_cdf, rng.random())
        if kind == 0:
            k = 0 if rng.random() < self.cfg.p_const else int(rng.integers(1, self.cfg.arity_max + 1))
        else:
            k = 1
        ins = [self._f_port(p, rng, 1) for _ in range(k)]
        splice = self._f_splice(p, rng)
        if splice is not None and k:
            ins[0] = p.ref_at(*splice)
        flat = [x for r in ins for x in r]
        if kind == 0:
            tt = random_tt(k, rng)
            rise = [self._delay(rng) for _ in range(k)]
            fall = [self._delay(rng) for _ in range(k)]
            init = tt & 1 if k == 0 else int(rng.integers(2))
            data = (GATE, k, *flat, codec.s64(tt), init, 1, *rise, *fall)
        elif kind == 1:
            r, f = self._delay(rng), self._delay(rng)
            data = (DELAY, 1, *flat, int(rng.integers(2)), 1, r, f)
        else:
            data = (WIRE, 1, *flat)
        a, b = splice if splice is not None else (NONE, 0)
        return Move(MoveType.ADD_NODE, p.n, -1, a, b, data)

    def _f_delete_node(self, p, rng):
        i = int(rng.integers(p.n))
        fb = self._f_port(p, rng, exclude=i)
        if fb is None:
            return None
        return Move(MoveType.DELETE_NODE, i, -1, fb[0], fb[1])

    def _f_region(self, p, rng, seed):
        size = int(rng.integers(1, self.cfg.dup_max + 1))
        S, frontier = [seed], set(p.neighbours(seed)) if size > 1 else set()
        while len(S) < size and frontier:
            f = sorted(frontier)
            j = f[int(rng.integers(len(f)))]
            S.append(j)
            frontier |= p.neighbours(j)
            frontier -= set(S)
        return sorted(S)

    def _f_duplicate(self, p, rng):
        S = self._f_region(p, rng, int(rng.integers(p.n)))
        if p.n + len(S) > self.cfg.max_nodes:
            return None
        off, order = p.consumers()
        nop = p.node_of_pin()
        Sset = set(S)
        pins = sorted(q for s in S for q in order[off[s]:off[s + 1]].tolist() if int(nop[q]) not in Sset)
        ext = [(int(nop[q]), q - int(p.ptr[int(nop[q])])) for q in pins]
        ext += [(OUT, j) for j, s in enumerate(p.out_src.tolist()) if s in Sset]
        a, b = ext[int(rng.integers(len(ext)))] if ext and rng.random() < 0.5 else (NONE, 0)
        return Move(MoveType.DUPLICATE, S[0], -1, a, b, tuple(S))

    def _f_insert_module(self, p, rng):
        act = self.library.active()
        h = act[int(rng.integers(len(act)))]
        d = self.library.get(h)
        src = [self._f_port(p, rng, 1) for _ in range(d.n_in)]
        splice = self._f_splice(p, rng)
        if splice is not None and d.n_in:
            src[0] = p.ref_at(*splice)
        out_port = int(rng.integers(d.n_out)) if d.n_out else 0
        if d.n_out == 0:
            splice = None
        a, b = splice if splice is not None else (NONE, 0)
        lo, hi = codec.hash_to_ints(h)
        flat = tuple(x for r in src for x in r)
        return Move(MoveType.INSERT_MODULE, p.n, -1, a, b, (lo, hi, out_port, d.n_in) + flat)

    def _f_adjust_param(self, p, rng):
        cand = p.param_ix()
        if cand is None:
            i = int(rng.integers(p.n))
        elif not len(cand):
            return None
        else:
            i = int(cand[int(rng.integers(len(cand)))])
        a = int(p.ptr[i])
        k = int(p.ptr[i + 1]) - a
        if int(p.op[i]) == GATE:
            o = int(rng.integers(2 * k + 1 + (1 if k else 0)))
            f, pin = (RISE, o) if o < k else (FALL, o - k) if o < 2 * k else (DRIVE, 0) if o == 2 * k else (INIT, 0)
        else:
            f, pin = ((RISE, 0), (FALL, 0), (INIT, 0), (DRIVE, 0))[int(rng.integers(4))]
        old = int(p.rise[a + pin]) if f == RISE else int(p.fall[a + pin]) if f == FALL else \
            int(p.init[i]) if f == INIT else int(p.drive[i])
        return Move(MoveType.ADJUST_PARAM, i, pin, f, self._new_value(f, old, rng))

    def _f_regenerate(self, p, rng):
        S = self._f_region(p, rng, int(rng.integers(p.n)))
        return Move(MoveType.REGENERATE, S[0], -1, 0, 0, tuple(S))

    def _f_inline_module(self, p, rng):
        return None

    def apply_packed_move(self, c: Circuit, p, mv: Move, rng=None) -> Circuit:
        """Apply a move to a valid module-free circuit `c` with packed form `p`, validating incrementally. Moves
        that leave the packed layout (module insertion, a regenerator) go through `apply`. Raises like `apply`."""
        from . import packed as P
        t = mv.type
        if t == MoveType.TT_REPLACE or t == MoveType.TT_FLIP:
            i = mv.node
            if not -p.n <= i < p.n:
                raise IndexError("node out of range")
            i %= p.n
            if int(p.op[i]) != GATE:
                raise InvalidCircuit("truth-table move on a non-gate")
            tt = codec.u64(mv.a) if t == MoveType.TT_REPLACE else int(p.tt[i]) ^ (1 << mv.a)
            q = P.p_tt(p, i, tt)
        elif t == MoveType.REWIRE:
            q = P.p_rewire(p, mv.node, mv.pin, mv.a, mv.b)
        elif t == MoveType.ADD_NODE:
            if mv.node != p.n:
                raise InvalidCircuit("ADD_NODE index must be len(nodes)")
            d = mv.data
            op, k = int(d[0]), int(d[1])
            if op not in (GATE, DELAY, WIRE) or k > MAX_ARITY or (op != GATE and k != 1):
                return self.apply(c, mv, rng)
            ins = [(int(d[2 + 2 * j]), int(d[3 + 2 * j])) for j in range(k)]
            x = 2 + 2 * k
            splice = None if mv.a == NONE else (mv.a, mv.b)
            if op == GATE:
                q = P.p_add(p, GATE, ins, codec.u64(int(d[x])), d[x + 3:x + 3 + k], d[x + 3 + k:x + 3 + 2 * k],
                            int(d[x + 1]), int(d[x + 2]), splice)
            elif op == DELAY:
                q = P.p_add(p, DELAY, ins, 0, d[x + 2:x + 3], d[x + 3:x + 4], int(d[x]), int(d[x + 1]), splice)
            else:
                q = P.p_add(p, WIRE, ins, 0, (), (), 0, 1, splice)
        elif t == MoveType.DELETE_NODE:
            q = P.p_delete(p, mv.node, (mv.a, mv.b))
        elif t == MoveType.DUPLICATE:
            q = P.p_duplicate(p, mv.data, None if mv.a == NONE else (mv.a, mv.b))
        elif t == MoveType.ADJUST_PARAM and mv.a != SLOT:
            q = P.p_adjust(p, mv.node, _FIELD[mv.a], mv.pin, mv.b)
        elif t == MoveType.REGENERATE and self.regenerator is None:
            return c
        else:
            return self.apply(c, mv, rng)
        q.valid = True
        return P.PackedCircuit(q)

    def _mutate_packed(self, c, p, rng, proposal, eps):
        if proposal is not None and rng.random() >= eps:
            mv = proposal.propose(c, rng)
            if mv is not None:
                try:
                    return self.apply(c, mv, rng), mv, Source.LEARNED
                except (InvalidCircuit, IndexError, ValueError, KeyError):
                    pass
            source = Source.FALLBACK
        else:
            source = Source.UNIFORM
        for _ in range(self.cfg.attempts):
            mv = self.sample_packed(p, rng)
            try:
                return self.apply_packed_move(c, p, mv, rng), mv, source
            except (InvalidCircuit, IndexError, ValueError, KeyError):
                continue
        raise InvalidCircuit(f"no valid move found in {self.cfg.attempts} attempts")

    # -- application with validation
    def apply(self, c: Circuit, mv: Move, rng: np.random.Generator | None = None) -> Circuit:
        child = apply_move(c, mv, self.library)
        if mv.type == MoveType.REGENERATE and self.regenerator is not None:
            child = self.regenerator(c, list(mv.data), rng)
        validate(child, self.library)
        return child

    def mutate(self, c: Circuit, rng: np.random.Generator, proposal: Proposal | None = None, eps: float = 0.1):
        """-> (child, move, source). With a proposal, the move comes from it with probability 1 - eps and from the
        uniform operator otherwise (the uniform floor keeps every edit possible). Module-free valid parents take
        the packed path (children are PackedCircuits); the result is the same either way."""
        if self.cfg.fast and self.cfg.slow_weight <= 0:
            from .packed import valid_packed
            p = valid_packed(c, self.library)
            if p is not None:
                return self._mutate_packed(c, p, rng, proposal, eps)
        if proposal is not None and rng.random() >= eps:
            mv = proposal.propose(c, rng)
            if mv is not None:
                try:
                    return self.apply(c, mv, rng), mv, Source.LEARNED
                except (InvalidCircuit, IndexError, ValueError, KeyError):
                    pass
            source = Source.FALLBACK
        else:
            source = Source.UNIFORM
        for _ in range(self.cfg.attempts):
            mv = self.sample(c, rng)
            try:
                return self.apply(c, mv, rng), mv, source
            except (InvalidCircuit, IndexError, ValueError, KeyError):
                continue
        raise InvalidCircuit(f"no valid move found in {self.cfg.attempts} attempts")

