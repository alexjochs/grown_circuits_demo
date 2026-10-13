"""A generation's children bred by the compiled kernel (_cbreed): one buffer of regions, materialized as
PackedCircuits only when something asks for a child (in the search: the few that enter the archive).

`breed_compiled(operator, parents, rng)` -> ChildBatch, or None when the compiled path does not apply (then the
caller runs UniformOperator.mutate per parent; the results are the same). A ChildBatch gives:

    ok        bool [B]          the parent's mutation succeeded (mutate did not raise)
    children  ChildList         the valid children, in parent order (lazy; the same objects on repeated access)
    moves     MoveColumns       one Move per parent (Move(-1) for failures), also as columns for the edit log
    same_graph bool [B]         the edit left the gate graph unchanged (truth table / parameter moves), so
                                structural descriptors equal the parent's
"""
from __future__ import annotations

import numpy as np

from .graph import Circuit
from .moves import Move, MoveType

_PARAMETRIC = (MoveType.TT_REPLACE, MoveType.TT_FLIP, MoveType.ADJUST_PARAM, MoveType.REGENERATE)
_PAYCAP = 64


class MoveColumns:
    """Moves as columns (type, node, pin, a, b, payload values / offsets) that also behave as a list of Move."""

    def __init__(self, cols: np.ndarray, pay: np.ndarray, pay_n: np.ndarray):
        self.type, self.node, self.pin, self.a, self.b = (cols[:, j].copy() for j in range(5))
        self.pay_offsets = np.zeros(len(cols) + 1, np.int64)
        np.cumsum(pay_n, out=self.pay_offsets[1:])
        self.pay_values = pay[np.arange(pay.shape[1])[None, :] < pay_n[:, None]]
        self._moves = None

    def __len__(self):
        return len(self.type)

    def _list(self):
        if self._moves is None:
            o = self.pay_offsets
            pv = self.pay_values.tolist()
            self._moves = [Move(t, nd, p, a, b, tuple(pv[o[i]:o[i + 1]])) if t >= 0 else Move(-1)
                           for i, (t, nd, p, a, b) in enumerate(zip(self.type.tolist(), self.node.tolist(),
                                                                    self.pin.tolist(), self.a.tolist(),
                                                                    self.b.tolist()))]
        return self._moves

    def __getitem__(self, i):
        return self._list()[i]

    def __iter__(self):
        return iter(self._list())


class ChildList:
    """The valid children of a ChildBatch as a lazy sequence of Circuits."""

    def __init__(self, cb: "ChildBatch"):
        self.cb = cb
        self.idx = np.flatnonzero(cb.ok)          # parent slot of each valid child
        self._items: dict = {}

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        it = self._items.get(i)
        if it is None:
            it = self._items[i] = self.cb.child(int(self.idx[i]))
        return it

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def region_table(self):
        """(addresses uint64, n, P) of the valid children's regions (in the batch buffer or the parent's)."""
        cb = self.cb
        return cb.addr[self.idx], cb.cn[self.idx], cb.cP[self.idx]

    def n_wire(self):
        return self.cb.cnw[self.idx]


class ChildBatch:
    def __init__(self, parents, packs, buf, status, coff, cn, cP, cnw, moves, n_in, m):
        self.parents, self.packs, self.buf = parents, packs, buf
        self.status = status
        self.ok = status > 0
        self.coff, self.cn, self.cP, self.cnw = coff, cn, cP, cnw
        self.n_in, self.m = n_in, m
        self.moves = moves
        base = buf.ctypes.data
        self.addr = np.where(status == 2, np.array([p.region()[2] for p in packs], np.uint64),
                             np.uint64(base) + coff.astype(np.uint64))
        self.same_graph = np.isin(moves.type, [int(t) for t in _PARAMETRIC]) & self.ok
        self.children = ChildList(self)

    def child(self, b: int) -> Circuit:
        """Child of parent slot b as a PackedCircuit (its region copied out of the batch buffer)."""
        from .packed import Packed, PackedCircuit
        if self.status[b] == 2:
            return self.parents[b]
        from ._cpacked import region_size
        n, P = int(self.cn[b]), int(self.cP[b])
        o = int(self.coff[b])
        buf = self.buf[o:o + region_size(n, P, self.m)].copy()
        return PackedCircuit(Packed.from_region(buf, 0, n, P, self.m, self.n_in, int(self.cnw[b])))


def compiled_ok(operator) -> bool:
    """The compiled kernel reproduces UniformOperator.mutate itself: not a wrapper or subclass that overrides it
    (e.g. library.usage.ModuleTracker records every mutation)."""
    from .moves import UniformOperator
    if not isinstance(operator, UniformOperator) or type(operator).mutate is not UniformOperator.mutate:
        return False
    cfg = operator.cfg
    lib_active = operator.library is not None and len(operator.library.active()) > 0
    return bool(cfg.fast and not lib_active and operator.regenerator is None and 1 <= cfg.arity_max <= 12
                and cfg.dup_max <= 60 and getattr(cfg, "slow_weight", 0.0) <= 0)


def _tables(operator):
    """Per applicability key (8 bits, see _cbreed): the types and choice table, as UniformOperator builds them."""
    t = getattr(operator, "_c_tables", None)
    if t is None:
        from .moves import _cdf
        cfg = operator.cfg
        tab_n = np.zeros(256, np.int64)
        tab_types = np.zeros((256, 10), np.int64)
        tab_cdf = np.zeros((256, 10), np.float64)
        for key in range(256):
            bits = [(key >> i) & 1 for i in range(8)]
            has_gate, rewire, grow, nonempty, dup, ins_mod, params, regen = bits
            ok = {MoveType.TT_REPLACE: has_gate, MoveType.TT_FLIP: has_gate, MoveType.REWIRE: rewire,
                  MoveType.ADD_NODE: grow, MoveType.DELETE_NODE: nonempty, MoveType.DUPLICATE: dup,
                  MoveType.INSERT_MODULE: ins_mod, MoveType.ADJUST_PARAM: params, MoveType.REGENERATE: regen,
                  MoveType.INLINE_MODULE: 0}
            types = [x for x in MoveType if ok[x] and cfg.weights.get(x, 1.0) > 0]
            tab_n[key] = len(types)
            if types:
                tab_types[key, :len(types)] = [int(x) for x in types]
                tab_cdf[key, :len(types)] = _cdf([cfg.weights.get(x, 1.0) for x in types])
        ci = np.array([cfg.max_nodes, cfg.arity_max, cfg.delay_range[0], cfg.delay_range[1], cfg.drive_max,
                       cfg.dup_max, int(cfg.regen_noop), cfg.attempts, int(getattr(cfg, "delay_clamp", False))],
                      np.int64)
        cd = np.array([cfg.p_const, cfg.p_splice, cfg.delay_sigma], np.float64)
        t = operator._c_tables = (tab_n, tab_types, tab_cdf, ci, cd, np.array(operator._kind_cdf, np.float64))
    return t


def breed_compiled(operator, parents, rng) -> ChildBatch | None:
    from ._cbreed import lib as breed_lib
    from ._cpacked import region_size
    from .packed import valid_packed
    if not parents or not compiled_ok(operator):
        return None
    L = breed_lib()
    if L is None:
        return None
    packs = [valid_packed(c, operator.library) for c in parents]
    if any(p is None for p in packs):
        return None
    n_in, m = packs[0].n_in, packs[0].n_out
    if any(p.n_in != n_in or p.n_out != m for p in packs):
        return None
    B = len(packs)
    regs = [p.region_labels() for p in packs]
    paddr = np.array([r[2] for r in regs], np.uint64)
    pn = np.fromiter((p.n for p in packs), np.int64, B)
    pP = np.fromiter((p.P for p in packs), np.int64, B)
    pnw = np.fromiter((p.n_wire for p in packs), np.int64, B)
    cfg = operator.cfg
    dn = max(cfg.dup_max, 1)
    cap = int(sum(region_size(int(n) + dn + 1, int(P) + 6 * dn + 6, m) for n, P in zip(pn.tolist(), pP.tolist())))
    tab_n, tab_types, tab_cdf, ci, cd, kcdf = _tables(operator)
    buf = np.empty(max(cap, 8), np.uint8)
    status = np.zeros(B, np.int64)
    coff, cn, cP, cnw = (np.zeros(B, np.int64) for _ in range(4))
    mvc = np.zeros((B, 5), np.int64)
    pay = np.zeros((B, _PAYCAP), np.int64)
    pay_n = np.zeros(B, np.int64)
    a = lambda x: x.ctypes.data  # noqa: E731
    bg = rng.bit_generator
    with bg.lock:
        rc = L.grown_breed(B, n_in, m, a(paddr), a(pn), a(pP), a(pnw), a(ci), a(cd), a(kcdf), a(tab_n), a(tab_types),
                           a(tab_cdf), bg.ctypes.bit_generator.value, a(buf), len(buf), a(status), a(coff), a(cn),
                           a(cP), a(cnw), a(mvc), a(pay), _PAYCAP, a(pay_n))
    if rc != 0:
        raise RuntimeError(f"grown_breed failed ({rc}): the random stream has advanced, the batch cannot be retried")
    return ChildBatch(parents, packs, buf, status, coff, cn, cP, cnw, MoveColumns(mvc, pay, pay_n), n_in, m)
