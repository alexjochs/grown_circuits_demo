"""Canonical forms modulo parameters and gate pin order: the equivalence behind library mining modulo parameters
(archive.mining, mode "param") and the verification hash of a parametric rewrite.

Equivalence ("parametric pattern"). Two connected node sets are the same pattern when a bijection of their nodes,
boundary inputs and exposed ports exists such that
  - corresponding tokens have the same op and arity, and: GATEs compute the same Boolean function up to a
    permutation of their pins (the P class of canon_cells.p_canon) with the same drive; MODULE instances name the
    same definition with the same number of slot values; DELAY matches DELAY and WIRE matches WIRE;
  - wiring is preserved through the pin permutation, where pins that the function treats symmetrically (swapping
    them leaves the truth table unchanged) are interchangeable;
  - the same ports are exposed (read outside the set, or by a circuit output).
Parameters are ignored: every arc's rise and fall delay, every init value and every nested instance's slot values
become slots of the module (`library.param_slots`), and each instance carries its own values. Truth tables are
exact up to pin order. Input and output negations (the N of NPN) are not abstracted: they change the function, so an
instance could not reproduce its occurrence without extra inverters, and a match "up to an inverter" is a different
(larger) pattern that mining finds by itself when it recurs. Pin order is abstracted because it is pure labelling:
the uniform operator wires new gates' pins in random order, so AND(x, internal) and AND(internal, x), or a & ~b and
~b & a, are one gate written twice.

Canonical form. Every gate is rewritten with its P-canonical truth table (the smallest truth table over pin
permutations; the first minimizing permutation in itertools order); canonical positions related by a symmetric
transposition form a class (the relation is an equivalence, and any permutation inside a class is a symmetry). The
vertex graph (canon._Graph) has one vertex per boundary input and per node, labelled by the node's parametric key
and exposure mask; an edge source port -> consumer pin carries the pin's class label (the smallest canonical position
of its class) instead of the pin index. canon._Search gives the canonical labelling. The canonical body orders each
gate's pins by canonical position, and the pins inside one class by the canonical rank of their source (then source
port, then original pin), so all occurrences of one pattern give bodies that are literally equal once their
parameters are set alike: `ParamSubgraph.values` are the occurrence's parameters in `param_slots(body)` order and
`ModuleDef.resolved_body(values)` reproduces the occurrence's body exactly.

`pinperm_hash` is the full-circuit counterpart with parameters kept (edge labels carry (class, rise, fall)): equal
hashes mean the flattened netlists are equal up to node relabelling and pin order inside gates, i.e. the same
circuit on any substrate. The compress phase uses it to verify parametric rewrites.
"""
from __future__ import annotations

import hashlib
import itertools
import struct
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .canon import MAX_LEAVES, _Graph, _h8, _Search, _vb
from .graph import DELAY, GATE, INPUT, MODULE, OUT, SLOW_DRIVE, WIRE, Circuit, Node
from .library import param_slots, slot_value


# --- gates up to pin order ------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _pmaps(k: int):
    """[k!, 2^k] original row read by each row of the permuted function, and the permutations (q[i] = original pin
    at position i), in itertools order."""
    perms = list(itertools.permutations(range(k)))
    rows = np.arange(1 << k)
    bits = (rows[:, None] >> np.arange(k)) & 1                     # [R, k]: bit i of row r'
    maps = np.array([(bits << np.array(q)).sum(1) for q in perms], np.int64).reshape(len(perms), 1 << k)
    return maps, perms


def _apply(tt: int, k: int, rowmap) -> int:
    return int(sum(((tt >> int(r)) & 1) << j for j, r in enumerate(rowmap)))


@lru_cache(maxsize=1 << 16)
def gate_pcanon(tt: int, k: int):
    """-> (canonical truth table, q, cls): q[i] is the original pin placed at canonical position i, cls[i] the
    class label of position i (smallest position of its symmetry class)."""
    if k <= 1:
        return tt, tuple(range(k)), tuple(range(k))
    maps, perms = _pmaps(k)
    b = np.array([(tt >> r) & 1 for r in range(1 << k)], np.uint64)
    w = np.uint64(1) << np.arange(1 << k, dtype=np.uint64)
    vals = (b[maps] * w).sum(1, dtype=np.uint64)                   # <= 64 distinct bits: exact in uint64
    best = int(np.argmin(vals))                                    # first minimizer
    tt2 = int(vals[best])
    q = perms[best]
    cls = list(range(k))
    for i in range(k):
        for j in range(i + 1, k):
            if cls[j] != j:
                continue
            sw = list(range(k))
            sw[i], sw[j] = j, i
            m = (((np.arange(1 << k)[:, None] >> np.arange(k)) & 1) << np.array(sw)).sum(1)
            if _apply(tt2, k, m) == tt2:
                cls[j] = cls[i]
    return tt2, tuple(q), tuple(cls)


def _explicit(nd: Node, library) -> Node:
    """A module instance with its slot values written out (the definition's defaults when empty)."""
    if nd.op == MODULE and not nd.params and library is not None:
        d = library.get(nd.module)
        if d is not None and d.slots:
            nd2 = object.__new__(Node)
            nd2.__dict__.update({k: v for k, v in nd.__dict__.items() if k != "key"})
            nd2.__dict__["params"] = d.defaults()
            return nd2
    return nd


def _pkey(nd: Node, tt2: int) -> bytes:
    """Token bytes without parameters (gates by their P-canonical truth table)."""
    if nd.op == GATE:
        return struct.pack("<BBQi", GATE, len(nd.ins), tt2, nd.drive)
    if nd.op == DELAY:
        if nd.drive == SLOW_DRIVE:                 # a slow element's level (its delays) is structure, not a slot
            return struct.pack("<Biii", DELAY, nd.drive, nd.rise[0], nd.fall[0])
        return struct.pack("<Bi", DELAY, nd.drive)
    if nd.op == WIRE:
        return b"\x02"
    return struct.pack("<B16sHHH", MODULE, nd.module, len(nd.ins), nd.n_out, len(nd.params))


def node_form(nd: Node, pin_perm: bool = True):
    """(parametric key bytes, canonical tt, q, cls, pin class label per ORIGINAL pin)."""
    k = len(nd.ins)
    if nd.op == GATE and pin_perm and k > 1:
        tt2, q, cls = gate_pcanon(nd.tt, k)
    else:
        tt2, q, cls = nd.tt, tuple(range(k)), tuple(range(k))
    qinv = [0] * k
    for pos, p in enumerate(q):
        qinv[p] = pos
    return _pkey(nd, tt2), tt2, q, cls, tuple(cls[qinv[p]] for p in range(k))


# --- parametric subgraphs -------------------------------------------------------------------------------------------

@dataclass
class ParamSubgraph:
    hash: bytes              # the pattern (parameters and pin order abstracted)
    body: Circuit            # canonical body with this occurrence's parameters
    nodes: list              # host node indices in body order
    ext: list                # module input i <- this host ref
    exposed: list            # module output j = this host (node, port)
    exact: bool
    slots: tuple             # library.param_slots(body)
    values: tuple            # this occurrence's slot values


def param_subgraph(c: Circuit, nodes, cons: dict | None = None, library=None, pin_perm: bool = True,
                   max_leaves: int = MAX_LEAVES) -> ParamSubgraph:
    """Canonical parametric form of the node set `nodes` of `c` (module docstring). Boundary inputs and exposed
    outputs are as in canon.subgraph."""
    if cons is None:
        cons = c.consumers()
    S = sorted(set(int(i) for i in nodes))
    inS = {i: k for k, i in enumerate(S)}
    ext: list = []
    ext_ix: dict = {}
    for i in S:
        for r in c.nodes[i].ins:
            if r[0] == INPUT or r[0] not in inS:
                if r not in ext_ix:
                    ext_ix[r] = len(ext)
                    ext.append(r)
    X = len(ext)
    vbytes = [_vb(b"\xffX")] * X
    flags, forms, nds = [], [], []
    for i in S:
        nd = _explicit(c.nodes[i], library)
        nds.append(nd)
        mask = 0
        for p in range(nd.n_out):
            if any(ci == OUT or ci not in inS for ci, _ in cons.get((i, p), ())):
                mask |= 1 << p
        flags.append(mask)
        f = node_form(nd, pin_perm)
        forms.append(f)
        vbytes.append(_vb(f[0] + struct.pack("<Q", mask)))
    dst, pin, src, port = [], [], [], []
    for k, i in enumerate(S):
        plab = forms[k][4]
        for p, r in enumerate(nds[k].ins):
            dst.append(X + k), pin.append(plab[p])
            if r in ext_ix:
                src.append(ext_ix[r]), port.append(0)
            else:
                src.append(X + inS[r[0]]), port.append(r[1])
    a = lambda x: np.array(x, np.int64)  # noqa: E731
    g = _Graph(vbytes, np.array([_h8(b) for b in vbytes], np.uint64), a(dst), a(pin), a(src), a(port), [], b"P")
    ser, order, exact = _Search(g, max_leaves).run()
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    ext_order = sorted(range(X), key=lambda x: rank[x])
    node_order = sorted(range(len(S)), key=lambda k: rank[X + k])
    new_in = {x: n for n, x in enumerate(ext_order)}
    new_node = {S[k]: n for n, k in enumerate(node_order)}

    def m(r):
        return (INPUT, new_in[ext_ix[r]]) if r in ext_ix else (new_node[r[0]], r[1])

    def src_key(r):
        return (int(rank[ext_ix[r]]), 0) if r in ext_ix else (int(rank[X + inS[r[0]]]), r[1])

    body_nodes = []
    for k in node_order:
        nd = nds[k]
        _, tt2, q, cls, plab = forms[k]
        if nd.op == GATE and len(nd.ins) > 1 and pin_perm:
            kk = len(nd.ins)
            newpin = [0] * kk
            for lab in sorted(set(cls)):
                pos = [i for i in range(kk) if cls[i] == lab]
                pins = sorted((p for p in range(kk) if plab[p] == lab), key=lambda p: (src_key(nd.ins[p]), p))
                for ps, p in zip(pos, pins):
                    newpin[ps] = p
            body_nodes.append(Node(GATE, tuple(m(nd.ins[p]) for p in newpin), tt2,
                                   tuple(nd.rise[p] for p in newpin), tuple(nd.fall[p] for p in newpin), nd.init,
                                   nd.drive))
        else:
            body_nodes.append(nd.with_ins(m(r) for r in nd.ins))
    exposed = [(S[k], p) for k in node_order for p in range(nds[k].n_out) if flags[k] >> p & 1]
    body = Circuit(X, tuple(body_nodes), tuple((new_node[i], p) for i, p in exposed))
    slots = param_slots(body)
    values = tuple(int(slot_value(body.nodes[i], f, p)) for i, f, p in slots)
    return ParamSubgraph(hashlib.blake2b(ser, digest_size=16).digest(), body, [S[k] for k in node_order],
                         [ext[x] for x in ext_order], exposed, exact, slots, values)


def pattern_hash(body: Circuit, library=None, pin_perm: bool = True) -> bytes:
    """The parametric pattern of a whole module body (its inputs as boundary, its outputs as the exposed ports):
    equals the hash of the occurrences the body was mined from."""
    return param_subgraph(body, range(len(body.nodes)), library=library, pin_perm=pin_perm).hash


# --- full circuits with parameters, modulo pin order ----------------------------------------------------------------

def pinperm_hash(c: Circuit, library=None) -> bytes:
    """Canonical hash of c's flattened netlist with every parameter, invariant under node relabelling and under pin
    permutations inside gates that preserve the gate (module docstring)."""
    from .flatten import flatten, from_netlist
    f = from_netlist(flatten(c, library))
    I = f.n_in
    vbytes = [_vb(b"\xffI" + struct.pack("<I", i)) for i in range(I)]
    edges = []
    for i, nd in enumerate(f.nodes):
        key, tt2, q, cls, plab = node_form(nd, True)
        vbytes.append(_vb(key + struct.pack("<B", nd.init)))
        for p, (s, port) in enumerate(nd.ins):
            edges.append((I + i, (plab[p], nd.rise[p], nd.fall[p]), port if s != INPUT else 0,
                          port if s == INPUT else I + s))
    table = sorted({e[1] for e in edges})               # (class, rise, fall): its values go into the header
    labels = {t: n for n, t in enumerate(table)}
    lab0 = [_h8(b) for b in vbytes]
    outputs = []
    for j, (s, q) in enumerate(f.outputs):
        v, q = (q, 0) if s == INPUT else (I + s, q)
        outputs.append((v, q))
        lab0[v] = (lab0[v] + _h8(struct.pack("<cII", b"O", j, q))) & 0xFFFFFFFFFFFFFFFF
    a = lambda x: np.array(x, np.int64)  # noqa: E731
    g = _Graph(vbytes, np.array(lab0, np.uint64), a([e[0] for e in edges]), a([labels[e[1]] for e in edges]),
               a([e[3] for e in edges]), a([e[2] for e in edges]), outputs,
               struct.pack("<cII", b"Q", I, len(outputs)) + np.array(table, "<i8").tobytes())
    ser, _, _ = _Search(g, MAX_LEAVES).run()
    return hashlib.blake2b(ser, digest_size=16).digest()
