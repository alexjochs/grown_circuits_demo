"""Canonical forms and hashes of circuits and subgraphs.

Method: colour refinement plus individualization (the scheme of nauty / bliss, without their pruning heuristics
beyond automorphism pruning).

1. Every vertex (primary input, node, or boundary placeholder) gets an initial label from its token bytes (op and
   parameters) and, in circuit mode, from its input index and from which circuit outputs read it.
2. Refinement: repeatedly relabel each vertex from its own label, the labels of its sources (with pin and port),
   and the multiset of its consumers (with pin and port), until the number of classes stops growing. Directed
   edges with pin order and per-node truth tables make most circuits discrete here.
3. If ties remain (symmetric dead code, duplicated subgraphs, rings), individualize each member of the smallest
   tied class in turn, refine, recurse, and keep the lexicographically smallest serialization over all leaves.
   Two leaves with equal serializations give an automorphism, which prunes siblings in the same orbit.

The result does not depend on node order. It is exact unless the search exceeds `max_leaves` (pathologically
symmetric graphs: many interchangeable copies), in which case the smallest leaf found so far is used and
`exact=False` is reported; such hashes may then differ between relabelings of the same circuit.

Hashes are 16-byte BLAKE2b digests of the canonical serialization (which contains every token's full parameter
bytes, so equal hashes mean equal canonical forms up to a 2^-128 collision).
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from .. import _ckern
from .graph import INPUT, OUT, Circuit

_M = np.uint64
_P = [_M(x) for x in (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93,
                      0xA0761D6478BD642F, 0xE7037ED1A0B428DB, 0x8EBC6AF09C88C6E3, 0x589965CC75374CC3,
                      0x1D8E4E27C47D124F)]
MAX_LEAVES = 1024


def _mix(x):
    x = x ^ (x >> _M(30))
    x = x * _M(0xBF58476D1CE4E5B9)
    x = x ^ (x >> _M(27))
    x = x * _M(0x94D049BB133111EB)
    return x ^ (x >> _M(31))


def _h8(b: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


@dataclass
class _Graph:
    vbytes: list            # per vertex: length-prefixed token bytes (part of the serialization)
    lab0: np.ndarray        # uint64 [V]
    e_dst: np.ndarray       # int64 [E]  consumer vertex
    e_pin: np.ndarray
    e_src: np.ndarray       # source vertex
    e_port: np.ndarray
    outputs: list           # ordered (vertex, port) (circuit mode) or []
    header: bytes


def _refine(g: _Graph, lab: np.ndarray, want_count: bool = False):
    """Refine `lab` to the last labelling that increased the number of classes (compiled when available; the
    numpy twin _refine_numpy computes the same values). With want_count, -> (lab, number of classes or None)."""
    if _ckern.lib() is not None:
        out, n = _ckern.refine(lab, g.e_src, g.e_dst, g.e_pin, g.e_port, count=True)
        return (out, n) if want_count else out
    out = _refine_numpy(g, lab)
    return (out, None) if want_count else out


def _refine_numpy(g: _Graph, lab: np.ndarray) -> np.ndarray:
    V = len(lab)
    if V == 0:
        return lab
    ncls = len(np.unique(lab))
    src, dst = g.e_src, g.e_dst
    pin, port = g.e_pin.astype(np.uint64), g.e_port.astype(np.uint64)
    with np.errstate(over="ignore"):
        while True:
            acc_in = np.zeros(V, np.uint64)
            acc_out = np.zeros(V, np.uint64)
            if len(src):
                np.add.at(acc_in, dst, _mix(lab[src] * _P[0] + port * _P[1] + pin * _P[2] + _P[3]))
                np.add.at(acc_out, src, _mix(lab[dst] * _P[4] + pin * _P[5] + port * _P[6] + _P[7]))
            new = _mix(lab * _P[8] + _mix(acc_in + _P[0]) * _P[1] + _mix(acc_out + _P[2]))
            n = len(np.unique(new))
            if n <= ncls:
                return lab
            lab, ncls = new, n


def _edge_blocks(g: _Graph):
    """(start, count) of each vertex's in-edges when the edge arrays are sorted by (dst, pin) (both graph builders
    emit them that way), else None."""
    if not hasattr(g, "_blocks"):
        E, V = len(g.e_dst), len(g.lab0)
        blocks = None
        if E:
            key = g.e_dst * (int(g.e_pin.max()) + 1) + g.e_pin
            if np.all(key[1:] > key[:-1]):
                cnt = np.bincount(g.e_dst, minlength=V)
                start = np.zeros(V, np.int64)
                np.cumsum(cnt[:-1], out=start[1:])
                blocks = (start, cnt)
        g._blocks = blocks
    return g._blocks


def _serialize(g: _Graph, order: np.ndarray) -> bytes:
    V = len(order)
    rank = np.empty(V, np.int64)
    rank[order] = np.arange(V)
    E = len(g.e_dst)
    if E:
        blocks = _edge_blocks(g)
        if blocks is not None:          # rows sorted by (rank[dst], pin): each vertex's block, in canonical order
            start, cnt = blocks
            c = cnt[order]
            ends = np.cumsum(c)
            idx = np.repeat(start[order] - (ends - c), c) + np.arange(E)
        else:       # (dst, pin) need not be unique (pin classes, circuit.pcanon): ties sort by source rank, port
            idx = np.lexsort((g.e_port, rank[g.e_src], g.e_pin, rank[g.e_dst]))
        rows = np.empty((E, 4), "<u4")
        rows[:, 0] = rank[g.e_dst[idx]]
        rows[:, 1] = g.e_pin[idx]
        rows[:, 2] = rank[g.e_src[idx]]
        rows[:, 3] = g.e_port[idx]
    else:
        rows = np.zeros((0, 4), "<u4")
    outs = np.array([(rank[v], p) for v, p in g.outputs], np.int64).reshape(-1, 2)
    vb = g.vbytes
    return b"".join([g.header, struct.pack("<II", V, E), *[vb[v] for v in order.tolist()], rows.tobytes(),
                     outs.astype("<u4").tobytes()])


class _Search:
    def __init__(self, g: _Graph, max_leaves: int):
        self.g = g
        self.max_leaves = max_leaves
        self.leaves: dict = {}
        self.best: bytes | None = None
        self.best_order = None
        self.autos: list = []
        self.exact = True
        self._sig = None

    def run(self):
        lab, ncls = _refine(self.g, self.g.lab0.copy(), want_count=True)
        if ncls is not None and ncls == len(lab):
            self._leaf(lab)                         # refinement is discrete: one leaf, no search
        else:
            self._rec(lab, [])
        return self.best, self.best_order, self.exact

    def _leaf(self, lab):
        order = np.argsort(lab, kind="stable")
        ser = _serialize(self.g, order)
        prev = self.leaves.get(ser)
        if prev is not None:
            gamma = np.empty(len(order), np.int64)
            gamma[prev] = order                 # vertex prev[i] <-> order[i]: an automorphism
            self.autos.append(gamma)
        else:
            self.leaves[ser] = order
        if self.best is None or ser < self.best:
            self.best, self.best_order = ser, order

    def _twin(self, v):
        """Vertex v's (in-edges, out-edges) signature and its set of adjacent vertices. Two same-label vertices
        with equal signatures and no edge between them are exchanged by an automorphism fixing everything else.
        Computed per queried vertex from edge lists grouped by endpoint (only members of tied cells are asked)."""
        if self._sig is None:
            g = self.g
            V = len(g.lab0)
            by_dst = np.argsort(g.e_dst, kind="stable")
            by_src = np.argsort(g.e_src, kind="stable")
            off_d = np.zeros(V + 1, np.int64)
            off_s = np.zeros(V + 1, np.int64)
            np.cumsum(np.bincount(g.e_dst, minlength=V), out=off_d[1:])
            np.cumsum(np.bincount(g.e_src, minlength=V), out=off_s[1:])
            self._sig = {}
            self._csr = (by_dst, off_d, by_src, off_s)
        hit = self._sig.get(v)
        if hit is None:
            g = self.g
            by_dst, off_d, by_src, off_s = self._csr
            ei = by_dst[off_d[v]:off_d[v + 1]]
            eo = by_src[off_s[v]:off_s[v + 1]]
            ins = sorted(zip(g.e_pin[ei].tolist(), g.e_src[ei].tolist(), g.e_port[ei].tolist()))
            outs = sorted(zip(g.e_dst[eo].tolist(), g.e_pin[eo].tolist(), g.e_port[eo].tolist()))
            hit = self._sig[v] = ((tuple(ins), tuple(outs)), {x[0] for x in outs} | {x[1] for x in ins})
        return hit

    def _rec(self, lab, prefix):
        vals, counts = np.unique(lab, return_counts=True)
        tied = counts > 1
        if not tied.any():
            self._leaf(lab)
            return
        # target cell: smallest size, then smallest label (an isomorphism-invariant choice)
        cand = np.nonzero(tied)[0]
        t = cand[np.lexsort((vals[cand], counts[cand]))[0]]
        members = np.nonzero(lab == vals[t])[0].tolist()
        mset = set(members)
        explored, twin_sigs = [], set()
        orbits, n_autos = None, -1
        for v in members:
            if len(self.leaves) + len(self.autos) >= self.max_leaves:
                self.exact = False
                return
            sig_v, adj_v = self._twin(v)
            if sig_v in twin_sigs and not (adj_v & mset):
                continue                                   # a twin of an explored member: same orbit
            if explored and self.autos:
                if n_autos != len(self.autos):
                    orbits, n_autos = self._orbits(prefix), len(self.autos)
                if orbits is not None and any(orbits[u] == orbits[v] for u in explored):
                    continue
            lab2 = lab.copy()
            with np.errstate(over="ignore"):
                lab2[v] = _mix(lab[v] + _P[4] + _M(len(prefix)))
            self._rec(_refine(self.g, lab2), prefix + [v])
            explored.append(v)
            if not (adj_v & mset):
                twin_sigs.add(sig_v)

    def _orbits(self, prefix):
        """Orbit representative of every vertex under the known automorphisms fixing `prefix` pointwise."""
        gens = [a for a in self.autos if all(a[p] == p for p in prefix)]
        if not gens:
            return None
        parent = list(range(len(self.g.lab0)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in gens:
            for x in np.nonzero(a != np.arange(len(a)))[0].tolist():
                rx, ry = find(x), find(int(a[x]))
                if rx != ry:
                    parent[rx] = ry
        return [find(x) for x in range(len(parent))]


def _vb(key: bytes) -> bytes:
    return struct.pack("<H", len(key)) + key


def _circuit_graph(c: Circuit) -> _Graph:
    I = c.n_in
    vbytes = [_vb(b"\xffI" + struct.pack("<I", i)) for i in range(I)] + [_vb(n.key) for n in c.nodes]
    lab0 = [_h8(b) for b in vbytes]
    dst, pin, src, port = [], [], [], []
    for i, nd in enumerate(c.nodes):
        for p, (s, q) in enumerate(nd.ins):
            dst.append(I + i), pin.append(p)
            if s == INPUT:
                src.append(q), port.append(0)
            else:
                src.append(I + s), port.append(q)
    outputs = []
    for j, (s, q) in enumerate(c.outputs):
        v, q = (q, 0) if s == INPUT else (I + s, q)
        outputs.append((v, q))
        lab0[v] = (lab0[v] + _h8(struct.pack("<cII", b"O", j, q))) & 0xFFFFFFFFFFFFFFFF
    a = lambda x: np.array(x, np.int64)  # noqa: E731
    return _Graph(vbytes, np.array(lab0, np.uint64), a(dst), a(pin), a(src), a(port), outputs,
                  struct.pack("<cII", b"C", I, len(c.outputs)))


def _graph(c: Circuit) -> _Graph:
    """The vertex graph of c: from its packed arrays when it is module-free (same values as _circuit_graph, which
    builds them from the Node form and is used for circuits with module instances)."""
    from .packed import canon_arrays, packed_of
    p = packed_of(c)
    if p is None:
        return _circuit_graph(c)
    return _Graph(*canon_arrays(p))


def canonical(c: Circuit, max_leaves: int = MAX_LEAVES):
    """-> (serialization bytes, canonical node order (list of node indices), exact)."""
    g = _graph(c)
    ser, order, exact = _Search(g, max_leaves).run()
    I = c.n_in
    node_order = [] if order is None else (np.asarray(order)[np.asarray(order) >= I] - I).tolist()
    return ser, node_order, exact


def circuit_hash(c: Circuit) -> bytes:
    """BLAKE2b-128 of the canonical serialization. Module-free circuits whose refinement is discrete are hashed by
    the compiled kernel (circuit._cpacked.grown_hash: the same serialization and digest); the rest run the search
    here."""
    from .packed import packed_of
    p = packed_of(c)
    if p is not None:
        h = p.fast_hash()
        if h is not None:
            return h
    ser, _, _ = _Search(_graph(c), MAX_LEAVES).run()
    return hashlib.blake2b(ser, digest_size=16).digest()


def canonical_circuit(c: Circuit) -> Circuit:
    """The same circuit with nodes in canonical order (so equal hashes give literally equal circuits)."""
    _, order, _ = canonical(c)
    pos = {old: new for new, old in enumerate(order)}
    m = lambda r: r if r[0] == INPUT else (pos[r[0]], r[1])  # noqa: E731
    nodes = tuple(c.nodes[i].with_ins(m(r) for r in c.nodes[i].ins) for i in order)
    return Circuit(c.n_in, nodes, tuple(m(r) for r in c.outputs))


# --- subgraphs ----------------------------------------------------------------------------------------------------

@dataclass
class Subgraph:
    """A connected node set seen as a module: boundary inputs and exposed outputs are unordered in the hash; the
    canonical labeling fixes their order (the port signature)."""
    hash: bytes
    body: Circuit            # nodes in canonical order; inputs = `ext` in order; outputs = `exposed` in order
    nodes: list              # the circuit's node indices, in body order
    ext: list                # module input i <- this ref of the host circuit
    exposed: list            # module output j = this (node, port) of the host circuit
    exact: bool


def subgraph(c: Circuit, nodes, cons: dict | None = None, max_leaves: int = MAX_LEAVES) -> Subgraph:
    """Canonical form of the node set `nodes` of `c` (structure plus parameters). Each distinct external ref read
    by the set becomes one boundary input; each port of the set read from outside (or by a circuit output) is
    exposed."""
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
    exposed_flags = []
    for i in S:
        nd = c.nodes[i]
        mask = 0
        for p in range(nd.n_out):
            if any(ci == OUT or ci not in inS for ci, _ in cons.get((i, p), ())):
                mask |= 1 << p
        exposed_flags.append(mask)
        vbytes.append(_vb(nd.key + struct.pack("<Q", mask)))
    dst, pin, src, port = [], [], [], []
    for k, i in enumerate(S):
        for p, r in enumerate(c.nodes[i].ins):
            dst.append(X + k), pin.append(p)
            if r in ext_ix:
                src.append(ext_ix[r]), port.append(0)
            else:
                src.append(X + inS[r[0]]), port.append(r[1])
    a = lambda x: np.array(x, np.int64)  # noqa: E731
    g = _Graph(vbytes, np.array([_h8(b) for b in vbytes], np.uint64), a(dst), a(pin), a(src), a(port), [], b"S")
    ser, order, exact = _Search(g, max_leaves).run()
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    ext_order = sorted(range(X), key=lambda x: rank[x])
    node_order = sorted(range(len(S)), key=lambda k: rank[X + k])
    new_in = {x: n for n, x in enumerate(ext_order)}
    new_node = {S[k]: n for n, k in enumerate(node_order)}

    def m(r):
        return (INPUT, new_in[ext_ix[r]]) if r in ext_ix else (new_node[r[0]], r[1])

    body_nodes = tuple(c.nodes[S[k]].with_ins(m(r) for r in c.nodes[S[k]].ins) for k in node_order)
    exposed = [(S[k], p) for k in node_order for p in range(c.nodes[S[k]].n_out) if exposed_flags[k] >> p & 1]
    body = Circuit(X, body_nodes, tuple((new_node[i], p) for i, p in exposed))
    return Subgraph(hashlib.blake2b(ser, digest_size=16).digest(), body, [S[k] for k in node_order],
                    [ext[x] for x in ext_order], exposed, exact)
