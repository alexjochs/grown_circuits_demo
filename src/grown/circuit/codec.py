"""Flat integer encoding of circuits and nodes (storage in archives, libraries and edit-log payloads).

All values fit in int64. Truth tables and hash halves are stored as their two's-complement int64 views.

    circuit: [VERSION, n_in, n_out, n_nodes, (out_node, out_port) * n_out, node * n_nodes]
    node:    [op, k, (src_node, src_port) * k, payload]
      GATE   payload = tt, init, drive, rise * k, fall * k
      DELAY  payload = init, drive, rise, fall
      WIRE   payload = (none)
      MODULE payload = hash_lo, hash_hi, n_out, n_params, params * n_params
"""
from __future__ import annotations

import numpy as np

from .graph import DELAY, GATE, MODULE, WIRE, Circuit, Node

VERSION = 1
_TWO64 = 1 << 64


def s64(x: int) -> int:
    x &= _TWO64 - 1
    return x - _TWO64 if x >= 1 << 63 else x


def u64(x: int) -> int:
    return x & (_TWO64 - 1)


def hash_to_ints(h: bytes) -> tuple:
    return s64(int.from_bytes(h[:8], "little")), s64(int.from_bytes(h[8:], "little"))


def ints_to_hash(lo: int, hi: int) -> bytes:
    return u64(lo).to_bytes(8, "little") + u64(hi).to_bytes(8, "little")


def encode_node(nd: Node, out: list | None = None) -> list:
    out = [] if out is None else out
    k = len(nd.ins)
    out += [nd.op, k]
    for s, p in nd.ins:
        out += [s, p]
    if nd.op == GATE:
        out += [s64(nd.tt), nd.init, nd.drive, *nd.rise, *nd.fall]
    elif nd.op == DELAY:
        out += [nd.init, nd.drive, nd.rise[0], nd.fall[0]]
    elif nd.op == MODULE:
        out += [*hash_to_ints(nd.module), nd.n_out, len(nd.params), *nd.params]
    return out


def decode_node(a, i: int = 0):
    """-> (Node, next index)."""
    op, k = int(a[i]), int(a[i + 1])
    i += 2
    ins = tuple((int(a[i + 2 * j]), int(a[i + 2 * j + 1])) for j in range(k))
    i += 2 * k
    if op == GATE:
        tt, init, drive = u64(int(a[i])), int(a[i + 1]), int(a[i + 2])
        i += 3
        rise = tuple(int(x) for x in a[i:i + k])
        fall = tuple(int(x) for x in a[i + k:i + 2 * k])
        return Node(GATE, ins, tt, rise, fall, init, drive), i + 2 * k
    if op == DELAY:
        return Node(DELAY, ins, 0, (int(a[i + 2]),), (int(a[i + 3]),), int(a[i]), int(a[i + 1])), i + 4
    if op == WIRE:
        return Node(WIRE, ins), i
    if op == MODULE:
        h = ints_to_hash(int(a[i]), int(a[i + 1]))
        n_out, n_p = int(a[i + 2]), int(a[i + 3])
        params = tuple(int(x) for x in a[i + 4:i + 4 + n_p])
        return Node(MODULE, ins, module=h, n_out=n_out, params=params), i + 4 + n_p
    raise ValueError(f"unknown op {op}")


def encode(c: Circuit) -> list:
    out = [VERSION, c.n_in, len(c.outputs), len(c.nodes)]
    for s, p in c.outputs:
        out += [s, p]
    for nd in c.nodes:
        encode_node(nd, out)
    return out


def decode(a) -> Circuit:
    a = [int(x) for x in a]
    if a[0] != VERSION:
        raise ValueError(f"codec version {a[0]} != {VERSION}")
    n_in, n_out, n_nodes = a[1], a[2], a[3]
    outs = tuple((a[4 + 2 * j], a[5 + 2 * j]) for j in range(n_out))
    i = 4 + 2 * n_out
    nodes = []
    for _ in range(n_nodes):
        nd, i = decode_node(a, i)
        nodes.append(nd)
    if i != len(a):
        raise ValueError("trailing data in circuit encoding")
    return Circuit(n_in, tuple(nodes), outs)


def encode_many(circuits) -> tuple:
    """-> (int64 values, int64 offsets[len+1])."""
    parts = [np.asarray(encode(c), np.int64) for c in circuits]
    off = np.zeros(len(parts) + 1, np.int64)
    off[1:] = np.cumsum([len(p) for p in parts])
    return (np.concatenate(parts) if parts else np.zeros(0, np.int64)), off


def decode_many(values, offsets) -> list:
    return [decode(values[offsets[i]:offsets[i + 1]]) for i in range(len(offsets) - 1)]
