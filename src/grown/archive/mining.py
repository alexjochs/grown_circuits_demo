"""Library mining: recurring connected subgraphs scored by description-length saving.

Two equivalences (`mode`):
  exact  subgraphs match by token identity: truth table, init, every delay (canon.subgraph). A module has no slots.
  param  subgraphs match modulo parameters and gate pin order (circuit.pcanon: same structure, same gate functions
         up to pin permutation, delays / init values / nested slot values abstracted into the module's parameter
         slots, `library.param_slots`). Each instance carries its own slot values. WIRE and DELAY tokens take part
         like gates.

Algorithm
  1. Enumerate every connected node set of size k_min..k_max of every circuit (top-level tokens, connectivity
     ignoring direction) with ESU (Wernicke 2006), which emits each connected set exactly once.
  2. Pass 1 computes a cheap isomorphism invariant per set (sorted token labels, sorted internal edges as
     (src token, port, dst token, pin), boundary pins, exposed ports; in param mode the token label is its
     parametric key and the pin its symmetry-class label) and records which lineages contain it. Equivalent sets
     have equal invariants, so this filter loses nothing.
  3. Pass 2 re-enumerates (reusing pass 1's invariants, kept per circuit in ESU order) and computes the canonical
     form (exact or parametric) only for sets whose invariant occurs in at least `min_lineages` lineages.
  4. Per canonical hash: distinct lineages, distinct circuits, and occurrences (per circuit, a greedy maximal set
     of node-disjoint occurrences in enumeration order); in param mode each occurrence's slot values.
  5. Score = occurrences * (tokens of the occurrence - tokens of one instance) - tokens of the definition.
     Tokens are Circuit.description_length: a node, a pin pointer, an output pointer, and with `dl_params` also
     one per parameter value (graph.param_tokens). An instance costs 1 + its inputs, plus (dl_params) its slot
     values: exact modules have none (their parameters are paid once, in the definition); parametric modules pay
     every slot per instance, which cancels the occurrence's own parameter tokens, so the parametric saving per
     occurrence is the structural one under either DL. Candidates with a positive score and enough lineages are
     returned ranked.

Complexity: the number of connected sets of size <= k in a graph with n nodes and maximum degree d is
O(n * (e*d)^(k-1)); each costs O(k*d) for the invariant. Canonical forms cost refinement on k + |boundary| vertices
each, only for the filtered sets. Memory is one dict entry per distinct invariant. `stats` (a dict) receives the
counts per pass (sets, distinct invariants, kept invariants, canonical forms computed, patterns) and seconds.

Port signature: a candidate's boundary inputs and exposed outputs, in the canonical order of its body. Which
ports are exposed is part of the pattern (two occurrences that differ only in which internal port is read from
outside are different patterns).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..circuit.canon import _h8, subgraph
from ..circuit.graph import INPUT, OUT, Circuit, delete_nodes, param_tokens
from ..circuit.pcanon import _explicit, node_form, param_subgraph

MODES = ("exact", "param")


@dataclass
class Candidate:
    hash: bytes
    body: Circuit
    size: int
    n_in: int
    n_out: int
    in_types: tuple
    out_types: tuple
    occurrences: int
    lineages: int
    circuits: int
    per_occurrence: int        # tokens saved per replaced occurrence
    def_cost: int              # tokens to define the module once
    saving: int                # occurrences * per_occurrence - def_cost
    where: dict = field(default_factory=dict)   # circuit index -> list of disjoint occurrences (node lists)
    mode: str = "exact"
    slots: tuple = ()          # param mode: library.param_slots(body)
    values: dict = field(default_factory=dict)  # param mode: circuit index -> slot values per occurrence (as where)


def neighbours(c: Circuit) -> list:
    nbr = [set() for _ in c.nodes]
    for i, nd in enumerate(c.nodes):
        for s, _ in nd.ins:
            if s != INPUT and s != i:
                nbr[i].add(s)
                nbr[s].add(i)
    return nbr


def connected_sets(nbr, k_max: int, k_min: int = 1):
    """ESU: every connected vertex set of size k_min..k_max exactly once (as a list, seed first)."""
    def extend(S, ext, v, nb):
        if len(S) >= k_min:
            yield S
        if len(S) == k_max:
            return
        ext = set(ext)
        while ext:
            w = min(ext)
            ext.discard(w)
            new_ext = ext | {u for u in nbr[w] if u > v and u not in nb}
            yield from extend(S + [w], new_ext, v, nb | nbr[w])

    for v in range(len(nbr)):
        yield from extend([v], {u for u in nbr[v] if u > v}, v, nbr[v] | {v})


class _Labels:
    """Per circuit: an int label per node (exact: its token bytes; param: its parametric key) and a label per pin
    (exact: the pin index; param: its symmetry-class label). Labels are BLAKE2b digests, so invariants (hashes of
    int tuples) are the same in every process."""

    def __init__(self, c: Circuit, param: bool, library=None, pin_perm: bool = True):
        self.lab, self.plab = [], []
        for nd in c.nodes:
            if param:
                key, _, _, _, plab = node_form(_explicit(nd, library), pin_perm)
            else:
                key, plab = nd.key, tuple(range(len(nd.ins)))
            self.lab.append(_h8(key))
            self.plab.append(plab)


def _invariant(c: Circuit, S, cons, L: _Labels) -> int:
    inS = set(S)
    lab, plab = L.lab, L.plab
    keys, edges, exposed = [], [], []
    ext_refs = set()
    for i in S:
        nd = c.nodes[i]
        li = lab[i]
        keys.append(li)
        pl = plab[i]
        for p, r in enumerate(nd.ins):
            if r[0] != INPUT and r[0] in inS:
                edges.append((lab[r[0]], r[1], li, pl[p]))
            else:
                edges.append((0, -1, li, pl[p]))
                ext_refs.add(r)
        for p in range(nd.n_out):
            if any(ci == OUT or ci not in inS for ci, _ in cons.get((i, p), ())):
                exposed.append((li, p))
    keys.sort()
    edges.sort()
    exposed.sort()
    return hash((tuple(keys), tuple(edges), len(ext_refs), tuple(exposed)))


def circuit_invariants(c: Circuit, k_max: int, k_min: int, param: bool, library=None,
                       pin_perm: bool = True) -> np.ndarray:
    """int64 invariant of every connected set of c, in ESU order (pass 1 for one circuit)."""
    cons = c.consumers()
    L = _Labels(c, param, library, pin_perm)
    return np.fromiter((_invariant(c, S, cons, L) for S in connected_sets(neighbours(c), k_max, k_min)), np.int64)


_POOL_STATE: dict = {}


def _pass1_chunk(idx):
    s = _POOL_STATE
    return [circuit_invariants(s["circuits"][i], s["k_max"], s["k_min"], s["param"], s["library"], s["pin_perm"])
            for i in idx]


def _pass1(circuits, todo, k_max, k_min, param, library, pin_perm, workers):
    """Invariants of circuits[todo], in a forked process pool when workers > 1."""
    if workers <= 1 or len(todo) < 64:
        return [circuit_invariants(circuits[i], k_max, k_min, param, library, pin_perm) for i in todo]
    import multiprocessing as mp
    _POOL_STATE.update(circuits=circuits, k_max=k_max, k_min=k_min, param=param, library=library,
                       pin_perm=pin_perm)
    chunks = [todo[j:j + 256] for j in range(0, len(todo), 256)]
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_pass1_chunk, chunks)
    finally:
        _POOL_STATE.clear()
    return [x for part in parts for x in part]


def circuit_occurrences(c: Circuit, inv: np.ndarray, keep: np.ndarray, k_max: int, k_min: int, param: bool,
                        library=None, pin_perm: bool = True):
    """Pass 2 for one circuit: canonical forms of its sets whose invariant is in `keep` (sorted). -> (first
    canonical form per pattern hash, greedy node-disjoint occurrences [(hash, nodes, slot values)] in ESU order,
    number of canonical forms computed)."""
    hit = np.isin(inv, keep, assume_unique=False).tolist()
    cons = c.consumers()
    subs: dict = {}
    occ: list = []
    used: dict = {}            # hash -> set of nodes already covered in this circuit
    n = 0
    for k, S in enumerate(connected_sets(neighbours(c), k_max, k_min)):
        if not hit[k]:
            continue
        n += 1
        sg = param_subgraph(c, S, cons, library, pin_perm) if param else subgraph(c, S, cons)
        subs.setdefault(sg.hash, sg)
        cover = used.setdefault(sg.hash, set())
        if cover.isdisjoint(S):
            cover.update(S)
            occ.append((sg.hash, sg.nodes, sg.values if param else None))
    return subs, occ, n


def _pass2_chunk(idx):
    s = _POOL_STATE
    return [circuit_occurrences(s["circuits"][i], s["invs"][i], s["keep"], s["k_max"], s["k_min"], s["param"],
                                s["library"], s["pin_perm"]) for i in idx]


def _pass2(circuits, invs, keep, todo, k_max, k_min, param, library, pin_perm, workers):
    if workers <= 1 or len(todo) < 64:
        return [circuit_occurrences(circuits[i], invs[i], keep, k_max, k_min, param, library, pin_perm) for i in todo]
    import multiprocessing as mp
    _POOL_STATE.update(circuits=circuits, invs=invs, keep=keep, k_max=k_max, k_min=k_min, param=param,
                       library=library, pin_perm=pin_perm)
    chunks = [todo[j:j + 64] for j in range(0, len(todo), 64)]
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_pass2_chunk, chunks)
    finally:
        _POOL_STATE.clear()
    return [x for part in parts for x in part]


def _tokens(c: Circuit, S, params: bool = False) -> int:
    n = sum(1 + len(c.nodes[i].ins) for i in S)
    return n + sum(param_tokens(c.nodes[i]) for i in S) if params else n


def instance_tokens(n_in: int, n_slots: int, mode: str, dl_params: bool) -> int:
    """Tokens of one instance: the token, its input pointers and (dl_params, param mode) its slot values."""
    return 1 + n_in + (n_slots if (dl_params and mode == "param") else 0)


def mine(circuits, lineages=None, k_max: int = 4, k_min: int = 2, min_lineages: int = 2, top: int | None = 50,
         min_saving: int = 1, mode: str = "exact", dl_params: bool = False, library=None, pin_perm: bool = True,
         stats: dict | None = None, workers: int = 1, cache: dict | None = None) -> list:
    """Ranked recurring-subgraph candidates (see module docstring). `lineages[i]` labels circuit i's lineage
    (default: every circuit is its own lineage). `library`: needed in param mode when circuits hold module
    instances with default (empty) params. `workers`: processes for pass 1 (fork). `cache`: a dict kept across
    calls (compress rounds) holding each circuit object's pass-1 invariants, so unchanged circuits are not
    re-enumerated."""
    if mode not in MODES:
        raise ValueError(f"mining mode must be one of {MODES}, not {mode!r}")
    t0 = time.time()
    param = mode == "param"
    circuits = list(circuits)
    lineages = list(range(len(circuits))) if lineages is None else [int(x) for x in lineages]

    # pass 1: invariants per circuit (cached by circuit object), then the invariants seen in >= min_lineages
    key = (k_max, k_min, mode, pin_perm)
    invs: list = [None] * len(circuits)
    todo = []
    for ci, c in enumerate(circuits):
        hit = cache.get((id(c),) + key) if cache is not None else None
        if hit is not None and hit[0] is c:
            invs[ci] = hit[1]
        else:
            todo.append(ci)
    for ci, arr in zip(todo, _pass1(circuits, todo, k_max, k_min, param, library, pin_perm, workers)):
        invs[ci] = arr
        if cache is not None:
            cache[(id(circuits[ci]),) + key] = (circuits[ci], arr)
    if cache is not None:                      # forget circuits that are gone (rewritten elites)
        live = {id(c) for c in circuits}
        for k in [k for k in cache if k[1:] == key and k[0] not in live]:
            del cache[k]
    lens = np.fromiter((len(a) for a in invs), np.int64, len(invs))
    n_sets = int(lens.sum())
    allinv = np.concatenate(invs) if n_sets else np.zeros(0, np.int64)
    alllin = np.repeat(np.asarray(lineages, np.int64), lens)
    order = np.lexsort((alllin, allinv))
    si, sl = allinv[order], alllin[order]
    del allinv, alllin, order
    new_inv = np.r_[True, si[1:] != si[:-1]] if len(si) else np.zeros(0, bool)
    new_pair = new_inv | np.r_[True, sl[1:] != sl[:-1]] if len(si) else new_inv
    starts = np.flatnonzero(new_inv)
    n_lin = np.add.reduceat(new_pair.astype(np.int64), starts) if len(starts) else np.zeros(0, np.int64)
    keep = si[starts[n_lin >= min_lineages]]
    n_inv = len(starts)
    del si, sl, new_inv, new_pair
    t1 = time.time()

    # pass 2: canonical forms of the kept sets, per circuit (in a pool when workers > 1), merged in circuit order
    keep = np.sort(keep)
    todo = [ci for ci in range(len(circuits)) if np.isin(invs[ci], keep).any()]
    pats: dict = {}            # hash -> dict(sub, lineages, where, values)
    n_canon = 0
    for ci, (subs, occ, n) in zip(todo, _pass2(circuits, invs, keep, todo, k_max, k_min, param, library, pin_perm,
                                               workers)):
        n_canon += n
        for h, sg in subs.items():
            p = pats.get(h)
            if p is None:
                p = pats[h] = {"sub": sg, "lineages": set(), "where": {}, "values": {}}
            p["lineages"].add(lineages[ci])
        for h, nodes, values in occ:
            p = pats[h]
            p["where"].setdefault(ci, []).append(nodes)
            if param:
                p["values"].setdefault(ci, []).append(values)

    out = []
    for h, p in pats.items():
        if len(p["lineages"]) < min_lineages:
            continue
        sg = p["sub"]
        body = sg.body
        occ = sum(len(v) for v in p["where"].values())
        slots = sg.slots if param else ()
        per = (_tokens(body, range(len(body.nodes)), dl_params)
               - instance_tokens(body.n_in, len(slots), mode, dl_params))
        cost = body.description_length(params=dl_params)
        saving = occ * per - cost
        if saving < min_saving:
            continue
        out.append(Candidate(h, body, len(body.nodes), body.n_in, body.n_out, ("bit",) * body.n_in,
                             ("bit",) * body.n_out, occ, len(p["lineages"]), len(p["where"]), per, cost, saving,
                             p["where"], mode, slots, p["values"]))
    out.sort(key=lambda x: (-x.saving, -x.lineages, x.hash))
    if stats is not None:
        stats.update(mode=mode, circuits=len(circuits), sets=n_sets, invariants=n_inv, kept_invariants=len(keep),
                     canonical_forms=n_canon, patterns=len(pats), candidates=len(out),
                     pass1_seconds=t1 - t0, pass2_seconds=time.time() - t1)
    return out if top is None else out[:top]


def mine_archive(archive, **kw) -> list:
    """Mine the elites of an archive; lineages from the archive's lineage column (cells without one count as
    their own lineage). Candidate.where indexes `archive.filled_cells()`."""
    cells = archive.filled_cells()
    lin = archive.lineage[cells].copy()
    own = lin < 0
    lin[own] = -1 - np.arange(own.sum())
    kw.setdefault("library", archive.library)
    return mine([archive.items[c] for c in cells], lin, **kw)


def param_defaults(values) -> tuple:
    """Default slot values of a parametric module: per slot the lower median over its occurrences (for 0/1 init
    values, the majority with ties to 0)."""
    v = np.asarray(list(values), np.int64)
    if v.ndim != 2 or not v.shape[1]:
        return ()
    return tuple(int(x) for x in np.sort(v, 0)[(len(v) - 1) // 2])


def param_body(cand: Candidate, values=None):
    """(body with default slot values, slots) for admitting a parametric candidate; `values` defaults to every
    occurrence's slot values."""
    from ..circuit.library import set_slot
    if values is None:
        values = [x for vs in cand.values.values() for x in vs]
    d = param_defaults(values)
    nodes = list(cand.body.nodes)
    for (i, f, p), x in zip(cand.slots, d):
        nodes[i] = set_slot(nodes[i], f, p, x)
    return Circuit(cand.body.n_in, tuple(nodes), cand.body.outputs), cand.slots


def admit(library, cand: Candidate, expose_delays: bool = False, values=None):
    """Admission is explicit: add a candidate to the vocabulary. Returns the ModuleDef. Parametric candidates are
    admitted with every parameter as a slot and the occurrences' median values as defaults."""
    if cand.mode == "param":
        body, slots = param_body(cand, values)
        return library.admit(body, cand.in_types, cand.out_types, slots=slots)
    return library.admit(cand.body, cand.in_types, cand.out_types, expose_delays=expose_delays)


def compress(c: Circuit, nodes, mdef, library, mode: str = "exact", pin_perm: bool = True) -> Circuit:
    """Replace the occurrence `nodes` of `mdef`'s pattern in `c` by one instance of it (param mode: with the
    occurrence's slot values; the definition's body with those values must reproduce the occurrence exactly)."""
    if mode == "param":
        sg = param_subgraph(c, nodes, library=library, pin_perm=pin_perm)
        if len(sg.values) != len(mdef.slots) or not mdef.resolved_body(sg.values).same_as(sg.body):
            raise ValueError("node set is not an occurrence of this module")
        inst = library.instance(mdef.hash, sg.ext, sg.values)
    else:
        sg = subgraph(c, nodes)
        if not sg.body.same_as(mdef.body):
            raise ValueError("node set is not an occurrence of this module")
        inst = library.instance(mdef.hash, sg.ext)
    idx = len(c.nodes)
    c2 = Circuit(c.n_in, c.nodes + (inst,), c.outputs)
    return delete_nodes(c2, sg.nodes, {r: (idx, j) for j, r in enumerate(sg.exposed)})
