"""A batch of flattened netlists in concatenated arrays: what a substrate evaluates in one call.

    goff   int64  [B+1]   gate offsets: circuit b's gates are goff[b]..goff[b+1]-1
    poff   int64  [G+1]   pin offsets per gate (global): gate g's pins are poff[g]..poff[g+1]-1
    fanin  int32  [Pg]    source net of each pin, local to its circuit (0..I-1 primary inputs, I+j gate j)
    tt     uint64 [G], dr / df int32 [Pg], init uint8 [G] (gate nets), outs int32 [B, n_out] (local nets)
    gnode  int64  [G]     the top-level node each gate came from (Netlist.origin)

`flatten_batch(circuits)` builds it in one compiled pass over packed circuits (circuit.packed), with the same
arrays `flatten` produces circuit by circuit; circuits with module instances (or without a compiler) go through
`flatten` and are concatenated. `NetBatch.netlist(b)` is circuit b's `Netlist`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flatten import Netlist
from .graph import InvalidCircuit


class _Origin:
    """Netlist.origin for a flat circuit: gate g came from top-level node gnode[g] (a lazy list of 1-tuples)."""

    def __init__(self, gnode):
        self.gnode = gnode

    def __len__(self):
        return len(self.gnode)

    def __getitem__(self, g):
        if isinstance(g, slice):
            return [(int(x),) for x in self.gnode[g]]
        return (int(self.gnode[g]),)

    def __iter__(self):
        return ((x,) for x in self.gnode.tolist())

    def __eq__(self, other):
        return list(self) == list(other)


@dataclass(eq=False)
class NetBatch:
    n_inputs: int
    n_out: int
    goff: np.ndarray
    poff: np.ndarray
    fanin: np.ndarray
    tt: np.ndarray
    dr: np.ndarray
    df: np.ndarray
    init: np.ndarray
    outs: np.ndarray
    gnode: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.goff) - 1

    @property
    def n_gates(self) -> np.ndarray:
        return np.diff(self.goff)

    def netlist(self, b: int) -> Netlist:
        g0, g1 = int(self.goff[b]), int(self.goff[b + 1])
        p0, p1 = int(self.poff[g0]), int(self.poff[g1])
        I = self.n_inputs
        init = np.zeros(I + g1 - g0, np.uint8)
        init[I:] = self.init[g0:g1]
        origin = _Origin(self.gnode[g0:g1]) if self.gnode is not None else [()] * (g1 - g0)
        return Netlist(I, self.poff[g0:g1 + 1] - p0, self.fanin[p0:p1].copy(), self.tt[g0:g1].copy(),
                       self.dr[p0:p1].copy(), self.df[p0:p1].copy(), init, self.outs[b].astype(np.int64), origin)

    def netlists(self) -> list:
        return [self.netlist(b) for b in range(len(self))]

    def select(self, b0: int, b1: int) -> "NetBatch":
        """Circuits b0..b1-1 as their own batch (offsets rebased, arrays shared where possible)."""
        g0, g1 = int(self.goff[b0]), int(self.goff[b1])
        p0, p1 = int(self.poff[g0]), int(self.poff[g1])
        return NetBatch(self.n_inputs, self.n_out, self.goff[b0:b1 + 1] - g0, self.poff[g0:g1 + 1] - p0,
                        self.fanin[p0:p1], self.tt[g0:g1], self.dr[p0:p1], self.df[p0:p1], self.init[g0:g1],
                        self.outs[b0:b1], None if self.gnode is None else self.gnode[g0:g1])

    @staticmethod
    def from_netlists(nets) -> "NetBatch":
        if not nets:
            return NetBatch(0, 0, np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(0, np.int32),
                            np.zeros(0, np.uint64), np.zeros(0, np.int32), np.zeros(0, np.int32),
                            np.zeros(0, np.uint8), np.zeros((0, 0), np.int32))
        I = nets[0].n_inputs
        n_out = len(nets[0].outputs)
        if any(n.n_inputs != I or len(n.outputs) != n_out for n in nets):
            raise ValueError("all circuits in a batch need the same number of inputs and outputs")
        G = np.array([n.n_gates for n in nets], np.int64)
        goff = np.zeros(len(nets) + 1, np.int64)
        np.cumsum(G, out=goff[1:])
        npins = np.concatenate([np.diff(n.fanin_ptr) for n in nets])
        poff = np.zeros(len(npins) + 1, np.int64)
        np.cumsum(npins, out=poff[1:])
        cat = np.concatenate
        return NetBatch(I, n_out, goff, poff, cat([n.fanin for n in nets]).astype(np.int32),
                        cat([n.tt for n in nets]).astype(np.uint64), cat([n.dr for n in nets]).astype(np.int32),
                        cat([n.df for n in nets]).astype(np.int32), cat([n.init[I:] for n in nets]).astype(np.uint8),
                        np.stack([np.asarray(n.outputs) for n in nets]).astype(np.int32).reshape(len(nets), n_out))


def region_table(circuits):
    """(addresses uint64, n, P) of the circuits' regions (circuit._cpacked), or None if one is not module-free.
    A ChildList (circuit.childbatch) answers from its batch buffer without materializing children."""
    if hasattr(circuits, "region_table"):
        return circuits.region_table()
    from .packed import packed_of
    packs = [packed_of(c) for c in circuits]
    if any(p is None for p in packs):
        return None
    return (np.array([p.region()[2] for p in packs], np.uint64), np.fromiter((p.n for p in packs), np.int64, len(packs)),
            np.fromiter((p.P for p in packs), np.int64, len(packs)))


def flatten_batch(circuits, library=None) -> NetBatch:
    """Flatten circuits (same n_in and n_out) into one NetBatch."""
    return flatten_seqs([circuits], library)


def _io(seq):
    return (seq.cb.n_in, seq.cb.m) if hasattr(seq, "cb") else (seq[0].n_in, seq[0].n_out)


def flatten_seqs(seqs, library=None) -> NetBatch:
    """Flatten several sequences of circuits (e.g. several tasks' children; same n_in and n_out) into one
    NetBatch, in order: one compiled pass over their regions, or the Node walk when one has module instances."""
    from ._cpacked import lib as plib
    seqs = [s for s in seqs if len(s)]
    if not seqs:
        return NetBatch.from_netlists([])
    tabs = [region_table(s) for s in seqs] if plib() is not None else [None]
    if any(t is None for t in tabs):
        from .flatten import flatten
        return NetBatch.from_netlists([flatten(c, library) for s in seqs for c in s])
    addr = np.concatenate([t[0] for t in tabs])
    nn = np.concatenate([t[1] for t in tabs])
    pp = np.concatenate([t[2] for t in tabs])
    I, m = _io(seqs[0])
    return flatten_regions(I, m, addr, nn, pp)


def flatten_regions(I: int, m: int, addr, nn, pp) -> NetBatch:
    """NetBatch of B regions in one compiled pass (circuit._cpacked.grown_flatten_regions)."""
    from ._cpacked import lib as plib
    B = len(addr)
    gcap = int(nn.sum())
    pcap = int(pp.sum())
    goff = np.zeros(B + 1, np.int64)
    poff = np.zeros(gcap + 1, np.int64)
    fanin = np.empty(pcap, np.int32)
    tt = np.empty(gcap, np.uint64)
    dr = np.empty(pcap, np.int32)
    df = np.empty(pcap, np.int32)
    init = np.empty(gcap, np.uint8)
    outs = np.empty((B, m), np.int32)
    gnode = np.empty(gcap, np.int64)
    addr = np.ascontiguousarray(addr, np.uint64)
    nn = np.ascontiguousarray(nn, np.int64)
    pp = np.ascontiguousarray(pp, np.int64)
    a = lambda x: x.ctypes.data  # noqa: E731
    rc = plib().grown_flatten_regions(B, I, m, a(addr), a(nn), a(pp), gcap, pcap, a(goff), a(poff), a(fanin), a(tt),
                                      a(dr), a(df), a(init), a(outs), a(gnode))
    if rc == -1:
        raise MemoryError("grown_flatten_regions: allocation failed")
    if rc == -2:
        raise ValueError("grown_flatten_regions: capacity exceeded (malformed region)")
    if rc > 0:
        raise InvalidCircuit("zero-delay loop through wires or module pass-throughs")
    G = int(goff[-1])
    Pg = int(poff[G])
    return NetBatch(I, m, goff, poff[:G + 1], fanin[:Pg], tt[:G], dr[:Pg], df[:Pg], init[:G], outs, gnode[:G])


def flatten_packed(packs) -> NetBatch:
    from .. import _ckern
    from .._cbuild import ptr
    B = len(packs)
    I, n_out = packs[0].n_in, packs[0].n_out
    if any(p.n_in != I or p.n_out != n_out for p in packs):
        raise ValueError("all circuits in a batch need the same number of inputs and outputs")
    nn = np.fromiter((p.n for p in packs), np.int64, B)
    gcap = int(sum(p.n - p.n_wire for p in packs))
    pcap = int(sum(p.P for p in packs))
    T = np.fromiter((ptr(a) for p in packs for a in (p.op, p.ptr, p.src, p.port, p.tt, p.init, p.rise, p.fall,
                                                      p.out_src, p.out_port)), np.uint64, 10 * B)
    goff = np.zeros(B + 1, np.int64)
    poff = np.zeros(gcap + 1, np.int64)
    fanin = np.empty(pcap, np.int32)
    tt = np.empty(gcap, np.uint64)
    dr = np.empty(pcap, np.int32)
    df = np.empty(pcap, np.int32)
    init = np.empty(gcap, np.uint8)
    outs = np.empty((B, n_out), np.int32)
    gnode = np.empty(gcap, np.int64)
    rc = _ckern.lib().grown_flatten_batch(B, I, n_out, ptr(nn), ptr(T), gcap, pcap, ptr(goff), ptr(poff), ptr(fanin),
                                          ptr(tt), ptr(dr), ptr(df), ptr(init), ptr(outs), ptr(gnode))
    if rc == -1:
        raise MemoryError("grown_flatten_batch: allocation failed")
    if rc == -2:
        raise ValueError("grown_flatten_batch: capacity exceeded (malformed packed circuit)")
    if rc > 0:
        raise InvalidCircuit("zero-delay loop through wires or module pass-throughs")
    G, Pg = int(goff[-1]), int(poff[int(goff[-1])])
    if G != gcap:
        poff, tt, init, gnode = poff[:G + 1], tt[:G], init[:G], gnode[:G]
    if Pg != pcap:
        fanin, dr, df = fanin[:Pg], dr[:Pg], df[:Pg]
    return NetBatch(I, n_out, goff, poff, fanin, tt, dr, df, init, outs, gnode)
