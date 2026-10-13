"""Circuits by canonical hash, so hindsight training can rebuild every parent named in an edit log.

The edit log stores 16-byte hashes, not circuits, and moves refer to the parent's literal node order. A
`CircuitStore` keeps the literal circuit (the object that was mutated) per hash and is saved beside the log as
`circuits.npz` (codec-encoded values and offsets plus the hashes). When two literal circuits share a hash
(isomorphic relabelings), the first one stored wins; training checks every example by replaying the move and
comparing the child's hash, and drops mismatches.

`RecordingArchive` is a MAP-Elites `Archive` that adds every inserted elite to a store. Parents in `grown.loop` are
always archive elites, so a run with a recording archive can be replayed in full without touching the loop.
"""
from __future__ import annotations

import os

import numpy as np

from ..archive.map_elites import Archive
from ..circuit import codec
from ..circuit.graph import Circuit

FILE = "circuits.npz"


class CircuitStore:
    def __init__(self):
        self.items: dict = {}

    def __len__(self):
        return len(self.items)

    def __contains__(self, h: bytes) -> bool:
        return h in self.items

    def add(self, c: Circuit) -> bytes:
        h = c.hash
        self.items.setdefault(h, c)
        return h

    def extend(self, circuits) -> None:
        for c in circuits:
            if isinstance(c, Circuit):
                self.add(c)

    def get(self, h: bytes):
        return self.items.get(bytes(h))

    def save(self, path) -> None:
        path = str(path)
        if os.path.isdir(path):
            path = os.path.join(path, FILE)
        hs = list(self.items)
        vals, off = codec.encode_many([self.items[h] for h in hs])
        H = np.frombuffer(b"".join(hs), np.uint8).reshape(-1, 16) if hs else np.zeros((0, 16), np.uint8)
        np.savez_compressed(path, hashes=H, values=vals, offsets=off)

    @staticmethod
    def load(path) -> "CircuitStore":
        path = str(path)
        if os.path.isdir(path):
            path = os.path.join(path, FILE)
        z = np.load(path)
        s = CircuitStore()
        for h, c in zip(z["hashes"], codec.decode_many(z["values"], z["offsets"])):
            s.items[bytes(h)] = c
        return s


class RecordingArchive(Archive):
    """An Archive that records every inserted circuit in `self.store`."""

    def __init__(self, *args, store: CircuitStore | None = None, **kw):
        super().__init__(*args, **kw)
        self.store = store if store is not None else CircuitStore()

    def insert(self, desc, fitness, items=None, **kw):
        res = super().insert(desc, fitness, items=items, **kw)
        if items is not None:
            for i in np.nonzero(res.inserted)[0].tolist():
                self.store.add(items[i])
        return res
