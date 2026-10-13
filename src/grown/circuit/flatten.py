"""Flattening: token graph -> primitive netlist in freerun's data model (docs/SPEC.md §2 of freerun).

Nets 0..I-1 are primary inputs and net I+g is the output of gate g. GATE and DELAY tokens become gates (a DELAY is
a buffer with its rise/fall delay). WIRE tokens and module ports are aliases and disappear. Module instances are
expanded recursively with their slot values applied. Gate order is a depth-first walk in node order, so the
result is deterministic for a given circuit (not canonical: relabeled circuits flatten to relabeled netlists).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import DELAY, GATE, INPUT, MODULE, WIRE, Circuit, InvalidCircuit, Node, delete_nodes

BUF = 0b10


@dataclass(eq=False)
class Netlist:
    n_inputs: int
    fanin_ptr: np.ndarray   # int64 [G+1]
    fanin: np.ndarray       # int32 [E]
    tt: np.ndarray          # uint64 [G]
    dr: np.ndarray          # int32 [E]
    df: np.ndarray          # int32 [E]
    init: np.ndarray        # uint8 [N] (input nets: 0)
    outputs: np.ndarray     # int64 [n_out] net per circuit output
    origin: list            # per gate: path of node indices from the top level (module nesting)

    @property
    def n_gates(self) -> int:
        return len(self.tt)

    @property
    def n_nets(self) -> int:
        return self.n_inputs + self.n_gates

    @property
    def arity(self) -> np.ndarray:
        return np.diff(self.fanin_ptr)

    def same_as(self, o: "Netlist") -> bool:
        return self.n_inputs == o.n_inputs and all(
            np.array_equal(getattr(self, f), getattr(o, f))
            for f in ("fanin_ptr", "fanin", "tt", "dr", "df", "init", "outputs"))


def flatten(c: Circuit, library=None) -> Netlist:
    """The primitive netlist of c. Module-free circuits whose packed form is at hand (circuit.packed) take the
    compiled path in netbatch; the result is the same."""
    if "packed" in c.__dict__ and c.__dict__["packed"] is not None:
        from .. import _ckern
        if _ckern.lib() is not None:
            from .netbatch import flatten_packed
            return flatten_packed([c.__dict__["packed"]]).netlist(0)
    return flatten_nodes(c, library)


def flatten_nodes(c: Circuit, library=None) -> Netlist:
    """flatten by a walk over the Node form (module instances expanded recursively)."""
    gates: list = []          # [tt, pin handles, dr, df, init, origin]
    alias: dict = {}          # alias id -> handle
    I = c.n_in

    def expand(g: Circuit, scope_in: list, prefix: tuple) -> list:
        handles: list = [None] * len(g.nodes)
        bind: list = []       # (node index, kind, target)
        for i, nd in enumerate(g.nodes):
            if nd.op in (GATE, DELAY):
                gi = len(gates)
                tt = nd.tt if nd.op == GATE else BUF
                gates.append([tt, None, list(nd.rise), list(nd.fall), nd.init, prefix + (i,)])
                handles[i] = [("g", gi)]
                bind.append((i, "g", gi))
            elif nd.op == WIRE:
                aid = len(alias)
                alias[aid] = None
                handles[i] = [("a", aid)]
                bind.append((i, "a", [aid]))
            elif nd.op == MODULE:
                if library is None or nd.module not in library:
                    raise InvalidCircuit(f"module {nd.module.hex()[:12]} not in library")
                d = library.get(nd.module)
                ids = []
                for _ in nd.ins:
                    aid = len(alias)
                    alias[aid] = None
                    ids.append(aid)
                handles[i] = expand(d.resolved_body(nd.params), [("a", a) for a in ids], prefix + (i,))
                bind.append((i, "a", ids))
            else:
                raise InvalidCircuit(f"unknown op {nd.op}")

        def ref(r):
            return scope_in[r[1]] if r[0] == INPUT else handles[r[0]][r[1]]

        for i, kind, target in bind:
            pins = [ref(r) for r in g.nodes[i].ins]
            if kind == "g":
                gates[target][1] = pins
            else:
                for aid, h in zip(target, pins):
                    alias[aid] = h
        return [ref(r) for r in g.outputs]

    outs = expand(c, [("in", i) for i in range(I)], ())

    def resolve(h):
        seen = 0
        while h[0] == "a":
            h = alias[h[1]]
            seen += 1
            if seen > len(alias):
                raise InvalidCircuit("zero-delay loop through wires or module pass-throughs")
        return h[1] if h[0] == "in" else I + h[1]

    k = [len(g[1]) for g in gates]
    ptr = np.zeros(len(gates) + 1, np.int64)
    ptr[1:] = np.cumsum(k)
    fanin = np.array([resolve(h) for g in gates for h in g[1]], np.int32)
    init = np.zeros(I + len(gates), np.uint8)
    init[I:] = [g[4] for g in gates]
    return Netlist(I, ptr, fanin, np.array([g[0] for g in gates], np.uint64),
                   np.array([d for g in gates for d in g[2]], np.int32),
                   np.array([d for g in gates for d in g[3]], np.int32), init,
                   np.array([resolve(h) for h in outs], np.int64), [g[5] for g in gates])


def from_netlist(n: Netlist) -> Circuit:
    """Netlist -> circuit of GATE tokens only (lossless: flatten(from_netlist(n)) reproduces n's arrays)."""
    I = n.n_inputs
    ref = lambda net: (INPUT, int(net)) if net < I else (int(net) - I, 0)  # noqa: E731
    nodes = []
    for g in range(n.n_gates):
        s = slice(int(n.fanin_ptr[g]), int(n.fanin_ptr[g + 1]))
        nodes.append(Node(GATE, tuple(ref(x) for x in n.fanin[s]), int(n.tt[g]),
                          tuple(int(x) for x in n.dr[s]), tuple(int(x) for x in n.df[s]), int(n.init[I + g])))
    return Circuit(I, tuple(nodes), tuple(ref(x) for x in n.outputs))


def inline(c: Circuit, i: int, library) -> Circuit:
    """Replace module instance `i` by its body (one level). Body nodes are appended; consumers of the instance's
    ports are pointed at the body's corresponding refs; the instance is removed."""
    nd = c.nodes[i]
    if nd.op != MODULE:
        raise ValueError(f"node {i} is not a module instance")
    body = library.get(nd.module).resolved_body(nd.params)
    base = len(c.nodes)

    def m(r):
        return nd.ins[r[1]] if r[0] == INPUT else (base + r[0], r[1])

    new = [b.with_ins(m(r) for r in b.ins) for b in body.nodes]
    out_map = {(i, p): m(r) for p, r in enumerate(body.outputs)}
    c2 = Circuit(c.n_in, c.nodes + tuple(new), c.outputs)
    return delete_nodes(c2, [i], out_map)


def inline_all(c: Circuit, library) -> Circuit:
    """Inline every module instance recursively: a circuit of primitives only."""
    while True:
        idx = next((i for i, nd in enumerate(c.nodes) if nd.op == MODULE), None)
        if idx is None:
            return c
        c = inline(c, idx, library)

