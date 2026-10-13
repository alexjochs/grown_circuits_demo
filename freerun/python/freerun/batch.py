"""Batch packing: concatenate circuits into the global arrays described in csrc/freerun.h."""
from __future__ import annotations

from functools import cached_property

import numpy as np

from .netlist import Circuit


def _offsets(sizes) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


def _local_ptrs(counts: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per-segment CSR pointers, concatenated: for each segment b, [0, cumsum(counts[starts[b]:starts[b+1]])]."""
    sizes = np.diff(starts) + 1
    seg = np.repeat(np.arange(len(sizes)), sizes)
    within = np.arange(int(sizes.sum())) - np.repeat(_offsets(sizes)[:-1], sizes)
    cs = _offsets(counts)
    return cs[starts[seg] + within] - cs[starts[seg]]


class Batch:
    def __init__(self, circuits):
        self.circuits = list(circuits)
        if not self.circuits:
            raise ValueError("empty batch")
        cs = self.circuits
        B = len(cs)
        self.B = B
        n_nets = np.array([c.n_nets for c in cs], dtype=np.int64)
        n_gates = np.array([c.n_gates for c in cs], dtype=np.int64)
        n_edges = np.array([len(c.fanin) for c in cs], dtype=np.int64)
        for name, n in (("nets", n_nets), ("gates", n_gates), ("edges", n_edges)):
            if np.any(n >= 2**31 - 1):
                raise ValueError(f"a circuit has too many {name} for 32-bit local indices")
        self.net_off, self.gate_off, self.edge_off = _offsets(n_nets), _offsets(n_gates), _offsets(n_edges)
        self.n_inputs = np.array([c.n_inputs for c in cs], dtype=np.int32)
        self.fanin_ptr = np.concatenate([c.fanin_ptr for c in cs]).astype(np.int32)
        self.fanin = np.concatenate([c.fanin for c in cs]).astype(np.int32)
        self.dr = np.concatenate([c.dr for c in cs]).astype(np.int32)
        self.df = np.concatenate([c.df for c in cs]).astype(np.int32)
        self.tt = np.concatenate([c.tt for c in cs]).astype(np.uint64)
        self.init = np.concatenate([c.init for c in cs]).astype(np.uint8)
        # Jitter (SPEC §10): per circuit seed / cid / dmin (dmin 0 = no jitter), per gate amplitude.
        self.has_jitter = any(c.jitter is not None for c in cs)
        self.jit_seed = np.array([c.jitter.seed if c.jitter is not None else 0 for c in cs], dtype=np.uint64)
        self.jit_cid = np.array([c.jitter.cid if c.jitter is not None else 0 for c in cs], dtype=np.uint32)
        self.jit_dmin = np.array([c.jitter.dmin if c.jitter is not None else 0 for c in cs], dtype=np.int32)
        self.jit_amp = (np.concatenate([c.jitter.amp if c.jitter is not None else np.zeros(c.n_gates, np.int64)
                                        for c in cs]).astype(np.int64) if self.has_jitter
                        else np.zeros(int(self.gate_off[-1]), np.int64))

    # The fanout arrays are optional in the C ABI (the engines derive them), so they are only built on request.
    @property
    def fo_ptr(self) -> np.ndarray:
        return self._fanout[0]

    @property
    def fo_edge(self) -> np.ndarray:
        return self._fanout[1]

    @property
    def edge_gate(self) -> np.ndarray:
        return self._fanout[2]

    @cached_property
    def _fanout(self):
        """Fanout CSR of every circuit at once: entries of a net sorted by (gate, pin), local indices."""
        cs, B = self.circuits, self.B
        n_edges = np.diff(self.edge_off)
        arity = np.concatenate([c.arity for c in cs]).astype(np.int64)
        circ_of_edge = np.repeat(np.arange(B), n_edges)
        gate_of_edge = np.repeat(np.arange(len(arity)), arity) - self.gate_off[circ_of_edge]  # local gate
        pin = np.arange(len(circ_of_edge)) - np.repeat(_offsets(arity)[:-1], arity)
        net = self.net_off[circ_of_edge] + self.fanin  # global net
        order = np.lexsort((pin, gate_of_edge, net))
        return (_local_ptrs(np.bincount(net, minlength=self.n_nets), self.net_off).astype(np.int32),
                (order - self.edge_off[circ_of_edge[order]]).astype(np.int32), gate_of_edge.astype(np.int32))

    @property
    def n_nets(self) -> int:
        return int(self.net_off[-1])

    @property
    def n_gates(self) -> int:
        return int(self.gate_off[-1])

    def nets(self, b: int) -> slice:
        return slice(int(self.net_off[b]), int(self.net_off[b + 1]))

    def gates(self, b: int) -> slice:
        return slice(int(self.gate_off[b]), int(self.gate_off[b + 1]))

    def __len__(self):
        return self.B

    def __getitem__(self, b) -> Circuit:
        return self.circuits[b]
