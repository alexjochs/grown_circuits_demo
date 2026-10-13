"""Structural descriptors of circuits (behavioral ones are supplied by the caller as arrays).

All graph descriptors except the token counts are computed on the flattened primitive netlist (module internals
included), with an edge from each gate's input net to the gate.

  n_nodes        top-level tokens
  n_gates        primitive gates after flattening (DELAY elements count as gates)
  dl             description length in tokens (Circuit.description_length)
  fvs            size of a feedback vertex set found greedily (an upper bound on the minimum; exact minimum is
                 NP-hard). 0 means combinational.
  cycle_frac     fraction of gates on a cycle (in a strongly connected component of size > 1, or self-looped)
  n_scc          number of cyclic strongly connected components
  max_scc        size of the largest one
  depth          longest path, in gates, through the condensation DAG from any primary input (a cyclic SCC
                 counts as one step)
  module_frac    fraction of top-level tokens that are module instances
  live_frac      fraction of top-level tokens in the backward cone of the outputs
"""
from __future__ import annotations

import numpy as np

from ..circuit.flatten import flatten
from ..circuit.graph import MODULE, Circuit, live_nodes

STRUCTURAL = ("n_nodes", "n_gates", "dl", "fvs", "cycle_frac", "n_scc", "max_scc", "depth", "module_frac",
              "live_frac")
_GRAPH = {"fvs", "cycle_frac", "n_scc", "max_scc", "depth"}


def _gate_graph(net):
    """Successor lists over gates: gate a -> gate g when a's output net feeds g."""
    I, G = net.n_inputs, net.n_gates
    succ = [[] for _ in range(G)]
    gate_of_pin = np.repeat(np.arange(G), net.arity)
    for src, g in zip(net.fanin.tolist(), gate_of_pin.tolist()):
        if src >= I:
            succ[src - I].append(g)
    return succ


def scc(succ) -> list:
    """Tarjan's algorithm, iterative. -> list of components (lists of vertices)."""
    n = len(succ)
    index = [-1] * n
    low = [0] * n
    on = [False] * n
    stack, comps = [], []
    counter = 0
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on[v] = True
            recurse = False
            for j in range(i, len(succ[v])):
                w = succ[v][j]
                if index[w] == -1:
                    work.append((v, j + 1))
                    work.append((w, 0))
                    recurse = True
                    break
                if on[w]:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on[w] = False
                    comp.append(w)
                    if w == v:
                        break
                comps.append(comp)
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
    return comps


def _cyclic(comps, succ):
    return [c for c in comps if len(c) > 1 or c[0] in succ[c[0]]]


def fvs_greedy(succ) -> int:
    """Greedy feedback vertex set: per cyclic SCC remove a self-looped vertex if any, else the vertex with the
    largest in-SCC (in-degree x out-degree); repeat until acyclic. Upper bound on the minimum."""
    succ = [set(s) for s in succ]
    removed = 0
    while True:
        cyc = _cyclic(scc([sorted(s) for s in succ]), succ)
        if not cyc:
            return removed
        for comp in cyc:
            cs = set(comp)
            self_loop = [v for v in comp if v in succ[v]]
            if self_loop:
                v = min(self_loop)
            else:
                indeg = {v: 0 for v in comp}
                for u in comp:
                    for w in succ[u]:
                        if w in cs:
                            indeg[w] += 1
                v = max(comp, key=lambda x: (indeg[x] * len(succ[x] & cs), -x))
            for u in range(len(succ)):
                succ[u].discard(v)
            succ[v] = set()
            removed += 1


def _depth(succ, comps, net) -> int:
    comp_of = {}
    for ci, comp in enumerate(comps):
        for v in comp:
            comp_of[v] = ci
    # Tarjan emits components in reverse topological order: process sources first
    I = net.n_inputs
    gate_of_pin = np.repeat(np.arange(net.n_gates), net.arity)
    best = np.zeros(len(comps), np.int64)
    driven = set(gate_of_pin[net.fanin < I].tolist())
    for ci in reversed(range(len(comps))):
        comp = comps[ci]
        if best[ci] == 0 and any(v in driven for v in comp):
            best[ci] = 1
        if best[ci] == 0:
            continue
        for v in comp:
            for w in succ[v]:
                cw = comp_of[w]
                if cw != ci:
                    best[cw] = max(best[cw], best[ci] + 1)
    return int(best.max()) if len(best) else 0


def structural(c: Circuit, names=STRUCTURAL, library=None) -> np.ndarray:
    out = np.zeros(len(names), np.float64)
    n = len(c.nodes)
    net = flatten(c, library) if (set(names) & (_GRAPH | {"n_gates"})) else None
    succ = comps = cyc = None
    if net is not None and set(names) & _GRAPH:
        succ = _gate_graph(net)
        comps = scc(succ)
        cyc = _cyclic(comps, succ)
    for j, name in enumerate(names):
        if name == "n_nodes":
            v = n
        elif name == "n_gates":
            v = net.n_gates
        elif name == "dl":
            v = c.description_length()
        elif name == "fvs":
            v = fvs_greedy(succ)
        elif name == "cycle_frac":
            v = sum(len(x) for x in cyc) / max(net.n_gates, 1)
        elif name == "n_scc":
            v = len(cyc)
        elif name == "max_scc":
            v = max((len(x) for x in cyc), default=0)
        elif name == "depth":
            v = _depth(succ, comps, net)
        elif name == "module_frac":
            v = sum(nd.op == MODULE for nd in c.nodes) / max(n, 1)
        elif name == "live_frac":
            v = live_nodes(c).sum() / max(n, 1)
        else:
            raise KeyError(f"unknown structural descriptor {name!r}")
        out[j] = v
    return out


def structural_batch(circuits, names=STRUCTURAL, library=None, nets=None, skip=None) -> np.ndarray:
    """[B, len(names)] for a batch, equal to `structural` per circuit. Only the requested names are computed. The
    graph descriptors come from one flattened NetBatch (`nets`, reused from evaluation when given) and a compiled
    SCC pass; without a compiler, per circuit in Python. Rows with skip[b] true are left NaN (the caller knows
    them, e.g. a parameter edit's child has its parent's descriptors)."""
    from .. import _ckern
    B = len(circuits)
    names = list(names)
    if not B:
        return np.zeros((0, len(names)))
    if _ckern.lib() is None:
        out = np.full((B, len(names)), np.nan)
        for b in range(B):
            if skip is None or not skip[b]:
                out[b] = structural(circuits[b], names, library)
        return out
    from ..circuit.packed import packed_of
    out = np.zeros((B, len(names)), np.float64)
    tab = circuits.region_table() if hasattr(circuits, "region_table") else None
    if tab is not None:
        cn, cP = tab[1], tab[2]
        m = circuits.cb.m
        sizes = cn.astype(np.float64)
        dls = (cn + cP + m).astype(np.float64)
    else:
        sizes = np.array([len(c) for c in circuits], np.float64)
        dls = None
    need = set(names)
    if nets is None and need & (_GRAPH | {"n_gates"}):
        from ..circuit.netbatch import flatten_batch
        nets = flatten_batch(circuits, library)
    G = nets.n_gates if nets is not None else None
    scc_stats = None
    if need & {"cycle_frac", "n_scc", "max_scc", "depth"}:
        scc_stats = _ckern.scc_batch(nets.n_inputs, nets.goff, nets.poff, nets.fanin, "depth" in need,
                                     None if skip is None else ~np.asarray(skip, bool))
    for j, name in enumerate(names):
        if name == "n_nodes":
            out[:, j] = sizes
        elif name == "n_gates":
            out[:, j] = G
        elif name == "dl":
            out[:, j] = dls if dls is not None else [c.description_length() for c in circuits]
        elif name == "cycle_frac":
            out[:, j] = scc_stats[:, 0] / np.maximum(G, 1)
        elif name == "n_scc":
            out[:, j] = scc_stats[:, 1]
        elif name == "max_scc":
            out[:, j] = scc_stats[:, 2]
        elif name == "depth":
            out[:, j] = scc_stats[:, 3]
        elif name == "fvs":
            out[:, j] = [np.nan if skip is not None and skip[b] else fvs_greedy(_gate_graph(nets.netlist(b)))
                         for b in range(B)]
        elif name == "module_frac":
            out[:, j] = 0.0 if tab is not None else \
                [0.0 if packed_of(c) is not None else sum(nd.op == MODULE for nd in c.nodes) / max(len(c), 1)
                 for c in circuits]
        elif name == "live_frac" and tab is not None:
            from ..circuit._cpacked import lib as plib
            cnt = np.zeros(B, np.int64)
            a = lambda x: x.ctypes.data  # noqa: E731
            addr, nn, pp = (np.ascontiguousarray(x) for x in tab)
            if plib().grown_live_regions(B, m, a(addr), a(nn), a(pp), a(cnt)):
                raise MemoryError("grown_live_regions: allocation failed")
            out[:, j] = cnt / np.maximum(cn, 1)
        elif name == "live_frac":
            col = []
            for c in circuits:
                p = packed_of(c)
                if p is None:
                    col.append(live_nodes(c).sum() / max(len(c), 1))
                else:
                    col.append(_ckern.live(p.n, p.ptr, p.src, p.out_src).sum() / max(p.n, 1))
            out[:, j] = col
        else:
            raise KeyError(f"unknown structural descriptor {name!r}")
    if skip is not None:
        out[np.asarray(skip, bool)] = np.nan
    return out
