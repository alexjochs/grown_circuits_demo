"""Packed circuits: a module-free circuit as flat numpy arrays, and edits applied to the arrays directly.

The search loop produces a child per edit and keeps only a few of them. Building every child as a tuple of `Node`
objects, flattening it with a Python walk and hashing it with a Python graph builder costs O(n) interpreter work per
edit. A `Packed` circuit holds the same information as arrays:

    op     int8   [n]     GATE, DELAY or WIRE (MODULE instances are not packed; such circuits use the Node path)
    ptr    int64  [n+1]   pin offsets: node i's pins are ptr[i]..ptr[i+1]-1
    src    int32  [P]     source node of each pin (-1 = primary input, as graph.INPUT)
    port   int32  [P]     source port (the input index for primary inputs)
    tt     uint64 [n]     truth table (0 for DELAY and WIRE)
    init   uint8  [n]
    drive  int64  [n]
    rise, fall int64 [P]  per-pin delays (0 on WIRE pins)
    out_src, out_port int32 [n_out]

`apply_packed(p, move)` edits the arrays (sharing every array the move does not touch) and checks validity
incrementally: the parent is valid, so only the touched nodes and any wire chain through a changed wire pin can make
the child invalid. The result equals `pack(apply_move(unpack(p), move))` exactly, and raises exactly when
`apply_move` plus `validate` would (tests/grown/test_packed.py checks both on random circuits and moves).

`PackedCircuit` is a `Circuit` backed by a Packed: `nodes` and `outputs` are built on first access, so children
that are evaluated and discarded never become Node tuples. Token labels for canonical hashing (the per-node bytes
and their BLAKE2b digests) are derived from the parent's labels on demand, recomputing only the edited nodes.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np

from .graph import DELAY, GATE, INPUT, MAX_ARITY, MAX_DELAY, OUT, WIRE, Circuit, InvalidCircuit, Node, tt_mask

_MASK64 = (1 << 64) - 1
_NODE_FIELDS = ("op", "ins", "tt", "rise", "fall", "init", "drive", "module", "n_out", "params")


def make_node(op, ins, tt=0, rise=(), fall=(), init=0, drive=1) -> Node:
    """A Node without the frozen-dataclass __init__ overhead (same fields, same equality and hash)."""
    nd = object.__new__(Node)
    nd.__dict__.update(op=op, ins=ins, tt=tt, rise=rise, fall=fall, init=init, drive=drive, module=b"", n_out=1,
                       params=())
    return nd


class Packed:
    __slots__ = ("n_in", "n", "P", "op", "ptr", "src", "port", "tt", "init", "drive", "rise", "fall", "out_src",
                 "out_port", "n_wire", "valid", "_vb", "_lab", "_lab_from", "_cache", "_region", "_region_lab")

    def __init__(self, n_in, op, ptr, src, port, tt, init, drive, rise, fall, out_src, out_port, n_wire=None,
                 valid=False):
        self.n_in = n_in
        self.op, self.ptr, self.src, self.port = op, ptr, src, port
        self.tt, self.init, self.drive, self.rise, self.fall = tt, init, drive, rise, fall
        self.out_src, self.out_port = out_src, out_port
        self.n = len(op)
        self.P = int(ptr[-1])
        self.n_wire = int(np.count_nonzero(op == WIRE)) if n_wire is None else n_wire
        self.valid = valid            # known to satisfy graph.validate (set by validation, inherited by edits)
        self._vb = self._lab = self._lab_from = None
        self._cache = self._region = None
        self._region_lab = False

    @property
    def n_out(self) -> int:
        return len(self.out_src)

    def derive(self, **changes) -> "Packed":
        """A copy with some arrays replaced (the others are shared: arrays are never written after creation)."""
        q = object.__new__(Packed)
        for k in ("n_in", "op", "ptr", "src", "port", "tt", "init", "drive", "rise", "fall", "out_src", "out_port",
                  "n_wire"):
            setattr(q, k, changes.get(k, getattr(self, k)))
        q.n = len(q.op)
        q.P = int(q.ptr[-1]) if "ptr" in changes else self.P
        q.valid = False
        q._vb = q._lab = q._lab_from = q._cache = q._region = None
        q._region_lab = False
        return q

    # --- derived, cached ----------------------------------------------------------------------------------------
    def cache(self) -> dict:
        if self._cache is None:
            self._cache = {}
        return self._cache

    def arity(self) -> np.ndarray:
        c = self.cache()
        a = c.get("arity")
        if a is None:
            a = c["arity"] = np.diff(self.ptr)
        return a

    def gate_ix(self) -> np.ndarray:
        c = self.cache()
        g = c.get("gate_ix")
        if g is None:
            g = c["gate_ix"] = np.flatnonzero(self.op == GATE)
        return g

    def param_ix(self) -> np.ndarray:
        """Nodes with adjustable parameters (GATE and DELAY)."""
        c = self.cache()
        g = c.get("param_ix")
        if g is None:
            g = c["param_ix"] = np.flatnonzero(self.op != WIRE) if self.n_wire else None
        return g

    def node_of_pin(self) -> np.ndarray:
        c = self.cache()
        a = c.get("node_of_pin")
        if a is None:
            a = c["node_of_pin"] = np.repeat(np.arange(self.n, dtype=np.int64), self.arity())
        return a

    def consumers(self):
        """(offsets [n+1], pin indices sorted by source node then pin) over node pins reading a node."""
        c = self.cache()
        v = c.get("consumers")
        if v is None:
            s = self.src.astype(np.int64)
            order = np.argsort(s, kind="stable")
            order = order[s[order] >= 0]
            off = np.zeros(self.n + 1, np.int64)
            np.cumsum(np.bincount(s[order], minlength=self.n), out=off[1:])
            v = c["consumers"] = (off, order)
        return v

    def neighbours(self, j: int) -> set:
        """Nodes adjacent to j (sources and consumers, not j itself, not inputs): moves._region's neighbour sets."""
        a, b = int(self.ptr[j]), int(self.ptr[j + 1])
        out = {s for s in self.src[a:b].tolist() if s >= 0 and s != j}
        off, order = self.consumers()
        if off[j + 1] > off[j]:
            nop = self.node_of_pin()
            out.update(i for i in nop[order[off[j]:off[j + 1]]].tolist() if i != j)
        return out

    def ref_at(self, consumer: int, pin: int) -> tuple:
        if consumer == OUT:
            return (int(self.out_src[pin]), int(self.out_port[pin]))
        a = int(self.ptr[consumer])
        if not 0 <= pin < int(self.ptr[consumer + 1]) - a:
            raise IndexError("pin out of range")
        return (int(self.src[a + pin]), int(self.port[a + pin]))

    def pin_at(self, idx: int) -> tuple:
        """The idx-th entry of Circuit.pins(): node pins in node order, then circuit outputs."""
        if idx >= self.P:
            return (OUT, idx - self.P)
        i = int(np.searchsorted(self.ptr, idx, side="right")) - 1
        return (i, idx - int(self.ptr[i]))

    def description_length(self, params: bool = False) -> int:
        n = self.n + self.P + self.n_out
        if not params:
            return n
        k = np.diff(self.ptr[:self.n + 1])
        op = np.asarray(self.op[:self.n])
        return n + int(np.where((op == GATE) & (k > 0), 2 * k + 1, 0).sum() + 3 * (op == DELAY).sum())

    # --- token labels for hashing ---------------------------------------------------------------------------------
    def lab_array(self) -> np.ndarray:
        """uint64 [n]: each node's token label, the 8-byte BLAKE2b of its length-prefixed Node.key (as
        canon._circuit_graph computes them). Derived from the parent's labels when this circuit came from an edit
        (only edited nodes are recomputed), else computed from the arrays."""
        if self._lab is None:
            src = self._lab_from
            if src is None:
                from ._cpacked import lib
                if lib() is not None:
                    self.region_labels()
                    return self._lab
                vb = [_node_vb(self, i) for i in range(self.n)]
                self._vb, self._lab = vb, _labs(vb)
                return self._lab
            parent, kind, arg = src
            plab = parent.lab_array()
            if kind == "same":
                lab = plab
            elif kind == "set":
                lab = plab.copy()
                lab[arg] = _h8(_node_vb(self, arg))
            elif kind == "del":
                lab = np.concatenate((plab[:arg], plab[arg + 1:]))
            else:                                       # "append": parent indices to copy, -1 = new node
                n0 = len(plab)
                new = [int(plab[j]) if j >= 0 else _h8(_node_vb(self, n0 + t)) for t, j in enumerate(arg)]
                lab = np.concatenate((plab, np.array(new, np.uint64)))
            self._lab = lab
            if parent._vb is None:
                self._lab_from = None                   # keep no chain of parents alive for the bytes
        return self._lab

    def vbytes(self) -> list:
        """Each node's length-prefixed token bytes (for canon's Python serialization)."""
        if self._vb is None:
            src = self._lab_from
            if src is not None and src[0]._vb is not None:
                parent, kind, arg = src
                pvb = parent._vb
                if kind == "same":
                    vb = pvb
                elif kind == "set":
                    vb = list(pvb)
                    vb[arg] = _node_vb(self, arg)
                elif kind == "del":
                    vb = pvb[:arg] + pvb[arg + 1:]
                else:
                    n0 = len(pvb)
                    vb = list(pvb) + [pvb[j] if j >= 0 else _node_vb(self, n0 + t) for t, j in enumerate(arg)]
            else:
                vb = [_node_vb(self, i) for i in range(self.n)]
            self._vb = vb
            self._lab_from = None
        return self._vb

    def labels(self):
        """(vbytes list [n], lab uint64 [n])."""
        lab = self.lab_array()
        return self.vbytes(), lab

    # --- regions (one contiguous buffer per circuit, read by the compiled kernels; see _cpacked) ----------------
    def region(self):
        """(buffer, byte offset, address) of this circuit's region, built on first use by copying the arrays.
        Token labels are included when known (region_labels makes sure they are)."""
        if self._region is None:
            from ._cpacked import region_size, region_views
            m = self.n_out
            buf = np.empty(region_size(self.n, self.P, m), np.uint8)
            v = region_views(buf, 0, self.n, self.P, m)
            for k in ("ptr", "tt", "drive", "rise", "fall", "src", "port", "out_src", "out_port", "op", "init"):
                v[k][...] = getattr(self, k)
            if self._lab is not None:
                v["lab"][...] = self._lab
                self._region_lab = True
            self._region = (buf, 0, buf.ctypes.data)
        return self._region

    def region_labels(self):
        """The region, with token labels filled in (computed by the compiled kernel when not known)."""
        buf, off, addr = self.region()
        if not self._region_lab:
            from ._cpacked import lib, region_views
            if self._lab is None and self._lab_from is not None:
                lab = self.lab_array()
                region_views(buf, off, self.n, self.P, self.n_out)["lab"][...] = lab
            else:
                lib().grown_labels(addr, self.n, self.P, self.n_out)
                self._lab = region_views(buf, off, self.n, self.P, self.n_out)["lab"]
            self._region_lab = True
        return self._region

    @staticmethod
    def from_region(buf: np.ndarray, off: int, n: int, P: int, m: int, n_in: int, n_wire: int) -> "Packed":
        """A Packed whose arrays are views of a region (with token labels) in `buf` at byte offset `off`."""
        from ._cpacked import region_views
        v = region_views(buf, off, n, P, m)
        q = object.__new__(Packed)
        q.n_in, q.n, q.P, q.n_wire = n_in, n, P, n_wire
        q.op, q.ptr, q.src, q.port = v["op"], v["ptr"], v["src"], v["port"]
        q.tt, q.init, q.drive, q.rise, q.fall = v["tt"], v["init"], v["drive"], v["rise"], v["fall"]
        q.out_src, q.out_port = v["out_src"], v["out_port"]
        q.valid = True
        q._vb = q._lab_from = q._cache = None
        q._lab = v["lab"]
        q._region = (buf, off, buf.ctypes.data + off)
        q._region_lab = True
        return q

    def fast_hash(self) -> bytes | None:
        """The canonical hash from the compiled kernel when colour refinement is discrete, else None (the caller
        runs canon's search)."""
        from ._cpacked import lib
        L = lib()
        if L is None:
            return None
        buf, off, addr = self.region_labels()
        out = np.empty(16, np.uint8)
        rc = L.grown_hash(addr, self.n, self.P, self.n_out, self.n_in, out.ctypes.data)
        if rc < 0:
            raise MemoryError("grown_hash: allocation failed")
        return out.tobytes() if rc == 0 else None


def _h8(b: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


def _labs(vb) -> np.ndarray:
    return np.fromiter((_h8(b) for b in vb), np.uint64, len(vb))


def node_key(op, k, tt, init, drive, rise, fall) -> bytes:
    """Node.key from field values."""
    if op == GATE:
        return struct.pack(f"<BBQBi{2 * k}i", GATE, k, tt, init, drive, *rise, *fall)
    if op == DELAY:
        return struct.pack("<BBiii", DELAY, init, drive, rise[0], fall[0])
    return b"\x02"


def _node_vb(p: Packed, i: int) -> bytes:
    a, b = int(p.ptr[i]), int(p.ptr[i + 1])
    op = int(p.op[i])
    if op == WIRE:
        key = b"\x02"
    else:
        key = node_key(op, b - a, int(p.tt[i]), int(p.init[i]), int(p.drive[i]), p.rise[a:b].tolist(),
                       p.fall[a:b].tolist())
    return struct.pack("<H", len(key)) + key


# --- packing and unpacking ------------------------------------------------------------------------------------------

def pack(c: Circuit) -> Packed | None:
    """Arrays of a module-free circuit; None if any node is a module instance or does not fit the packed layout
    (such circuits stay on the Node path). Validity is not checked here."""
    nodes = c.nodes
    n = len(nodes)
    ops, ks, refs, rise, fall, tts, inits, drives = [], [], [], [], [], [], [], []
    for nd in nodes:
        op, k = nd.op, len(nd.ins)
        if nd.n_out != 1 or nd.module or nd.params:
            return None
        if op == GATE:
            if len(nd.rise) != k or len(nd.fall) != k:
                return None
            rise.extend(nd.rise)
            fall.extend(nd.fall)
        elif op == DELAY:
            if len(nd.rise) != 1 or len(nd.fall) != 1 or k != 1 or nd.tt != 0:
                return None
            rise.extend(nd.rise)
            fall.extend(nd.fall)
        elif op == WIRE:
            if nd.rise or nd.fall or nd.tt or nd.init or nd.drive != 1:
                return None
            rise.extend([0] * k)
            fall.extend([0] * k)
        else:
            return None
        for r in nd.ins:
            if len(r) != 2:
                return None
            refs.extend(r)
        ops.append(op)
        ks.append(k)
        tts.append(nd.tt)
        inits.append(nd.init)
        drives.append(nd.drive)
    outs = [x for r in c.outputs for x in r]
    if len(outs) != 2 * len(c.outputs):
        return None
    try:
        ptr = np.zeros(n + 1, np.int64)
        np.cumsum(ks, out=ptr[1:])
        r = np.array(refs, np.int64).reshape(-1, 2)
        o = np.array(outs, np.int64).reshape(-1, 2)
        p = Packed(int(c.n_in), np.array(ops, np.int8), ptr, r[:, 0].astype(np.int32), r[:, 1].astype(np.int32),
                   np.array(tts, np.uint64), np.array(inits, np.uint8), np.array(drives, np.int64),
                   np.array(rise, np.int64), np.array(fall, np.int64), o[:, 0].astype(np.int32),
                   o[:, 1].astype(np.int32))
    except (OverflowError, ValueError, TypeError):
        return None
    if (np.any(r != np.stack([p.src, p.port], 1)) or np.any(o != np.stack([p.out_src, p.out_port], 1))
            or any(int(x) != y for x, y in zip(p.init.tolist(), inits))):
        return None                                 # a value did not survive the narrowing
    return p


def unpack(p: Packed) -> tuple:
    """-> (nodes, outputs) as Node tuples (the inverse of pack)."""
    op, ptr = p.op.tolist(), p.ptr.tolist()
    refs = list(zip(p.src.tolist(), p.port.tolist()))
    tt, init, drive = p.tt.tolist(), p.init.tolist(), p.drive.tolist()
    rise, fall = p.rise.tolist(), p.fall.tolist()
    nodes = []
    for i in range(p.n):
        a, b = ptr[i], ptr[i + 1]
        o = op[i]
        if o == WIRE:
            nodes.append(make_node(WIRE, tuple(refs[a:b])))
        else:
            nodes.append(make_node(o, tuple(refs[a:b]), tt[i], tuple(rise[a:b]), tuple(fall[a:b]), init[i],
                                   drive[i]))
    return tuple(nodes), tuple(zip(p.out_src.tolist(), p.out_port.tolist()))


class PackedCircuit(Circuit):
    """A Circuit backed by a Packed; `nodes` and `outputs` are built on first access."""

    def __init__(self, p: Packed):
        d = self.__dict__
        d["n_in"] = p.n_in
        d["packed"] = p

    def _unpack(self):
        d = self.__dict__
        d["_nodes"], d["_outputs"] = unpack(d["packed"])

    @property
    def nodes(self):
        d = self.__dict__
        if "_nodes" not in d:
            self._unpack()
        return d["_nodes"]

    @property
    def outputs(self):
        d = self.__dict__
        if "_outputs" not in d:
            self._unpack()
        return d["_outputs"]

    @property
    def n_out(self) -> int:
        return self.__dict__["packed"].n_out

    def __len__(self) -> int:
        return self.__dict__["packed"].n

    def description_length(self, params: bool = False) -> int:
        return self.__dict__["packed"].description_length(params)

    def __reduce__(self):
        return (PackedCircuit, (self.__dict__["packed"],))


def packed_of(c: Circuit) -> Packed | None:
    """The packed form of c (cached on the circuit), or None for circuits with module instances."""
    d = c.__dict__
    if "packed" in d:
        return d["packed"]
    p = d["packed"] = pack(c)
    return p


def valid_packed(c: Circuit, library=None) -> Packed | None:
    """The packed form of c if c is module-free and valid (validated once with graph.validate and remembered)."""
    p = packed_of(c)
    if p is None:
        return None
    if not p.valid:
        from .graph import is_valid
        if not is_valid(c, library):
            return None
        p.valid = True
    return p


# --- edits on packed arrays -----------------------------------------------------------------------------------------

def _set(a: np.ndarray, i: int, v) -> np.ndarray:
    b = a.copy()
    b[i] = v
    return b


def _check_ref(p: Packed, s: int, q: int) -> None:
    if s == INPUT:
        if not 0 <= q < p.n_in:
            raise InvalidCircuit(f"input {q} out of range")
    elif not 0 <= s < p.n or q != 0:
        raise InvalidCircuit(f"ref {(s, q)} out of range")


def _check_node(p: Packed, i: int) -> None:
    """graph.validate's per-node checks for node i (refs included)."""
    a, b = int(p.ptr[i]), int(p.ptr[i + 1])
    k = b - a
    for s, q in zip(p.src[a:b].tolist(), p.port[a:b].tolist()):
        _check_ref(p, s, q)
    op = int(p.op[i])
    if op == GATE:
        if k > MAX_ARITY:
            raise InvalidCircuit(f"node {i}: arity {k} > {MAX_ARITY}")
        tt = int(p.tt[i])
        if tt & ~tt_mask(k):
            raise InvalidCircuit(f"node {i}: truth table out of range for arity {k}")
        if k == 0 and int(p.init[i]) != (tt & 1):
            raise InvalidCircuit(f"node {i}: constant init must equal tt bit 0")
    elif k != 1:
        raise InvalidCircuit(f"node {i}: delay element / wire takes one pin")
    if op != WIRE:
        if int(p.init[i]) not in (0, 1):
            raise InvalidCircuit(f"node {i}: init must be 0 or 1")
        if k and (int(p.rise[a:b].min()) < 1 or int(p.fall[a:b].min()) < 1 or int(p.rise[a:b].max()) > MAX_DELAY
                  or int(p.fall[a:b].max()) > MAX_DELAY):
            raise InvalidCircuit(f"node {i}: delays must be in [1, 2^31-1]")


def _check_wires(p: Packed, wires) -> None:
    """No zero-delay loop through any of `wires` (WIRE nodes whose pin changed, or new ones). Any new wire-only
    cycle contains a changed wire pin, so following the alias chain from each one finds it."""
    if not p.n_wire:
        return
    op, ptr, src = p.op, p.ptr, p.src
    for w in wires:
        if int(op[w]) != WIRE:
            continue
        cur, steps = w, 0
        while True:
            s = int(src[int(ptr[cur])])
            if s < 0 or int(op[s]) != WIRE:
                break
            if s == w or steps > p.n_wire:
                raise InvalidCircuit(f"zero-delay loop through node {w} port 0")
            cur, steps = s, steps + 1


def p_tt(p: Packed, i: int, tt: int) -> Packed:
    if int(p.op[i]) != GATE:
        raise InvalidCircuit("truth-table move on a non-gate")
    k = int(p.ptr[i + 1] - p.ptr[i])
    if tt & ~tt_mask(k):
        raise InvalidCircuit("truth table out of range")
    if k == 0:
        q = p.derive(tt=_set(p.tt, i, tt), init=_set(p.init, i, tt & 1))
    else:
        q = p.derive(tt=_set(p.tt, i, tt))
    q._lab_from = (p, "set", i)
    q.valid = p.valid
    return q


def p_rewire(p: Packed, consumer: int, pin: int, s: int, port: int) -> Packed:
    if consumer == OUT:
        if not 0 <= pin < p.n_out:
            raise IndexError("output out of range")
        q = p.derive(out_src=_set(p.out_src, pin, s), out_port=_set(p.out_port, pin, port))
    else:
        a = int(p.ptr[consumer])
        if not 0 <= pin < int(p.ptr[consumer + 1]) - a:
            raise IndexError("pin out of range")
        q = p.derive(src=_set(p.src, a + pin, s), port=_set(p.port, a + pin, port))
    q._lab_from = (p, "same", None)
    _check_ref(q, s, port)
    if consumer != OUT:
        _check_wires(q, (consumer,))
    q.valid = p.valid
    return q


def p_add(p: Packed, op: int, ins, tt: int, rise, fall, init: int, drive: int, splice) -> Packed:
    """Append one node; splice = (consumer, pin) re-pointed at the new node's port 0, or None."""
    n, k = p.n, len(ins)
    ptr = np.empty(n + 2, np.int64)
    ptr[:n + 1] = p.ptr
    ptr[n + 1] = p.P + k
    if k:
        srcs = [r[0] for r in ins]
        ports = [r[1] for r in ins]
        src = np.concatenate((p.src, np.array(srcs, np.int32)))
        port = np.concatenate((p.port, np.array(ports, np.int32)))
        rise_a = np.concatenate((p.rise, np.array(rise if op != WIRE else [0] * k, np.int64)))
        fall_a = np.concatenate((p.fall, np.array(fall if op != WIRE else [0] * k, np.int64)))
    else:
        src, port, rise_a, fall_a = p.src, p.port, p.rise, p.fall
    out_src, out_port = p.out_src, p.out_port
    if splice is not None:
        c, pin = splice
        if c == OUT:
            if not 0 <= pin < p.n_out:
                raise IndexError("output out of range")
            out_src, out_port = _set(out_src, pin, n), _set(out_port, pin, 0)
        else:
            a = int(ptr[c]) if 0 <= c <= n else -1
            if c < 0 or not 0 <= pin < int(ptr[c + 1]) - a:
                raise IndexError("pin out of range")
            src, port = _set(src, a + pin, n), _set(port, a + pin, 0)
    q = Packed(p.n_in, np.append(p.op, np.int8(op)), ptr, src, port, np.append(p.tt, np.uint64(tt)),
               np.append(p.init, np.uint8(init)), np.append(p.drive, np.int64(drive)), rise_a, fall_a, out_src,
               out_port, p.n_wire + (op == WIRE))
    q._lab_from = (p, "append", (-1,))
    _check_node(q, n)
    wires = [n]
    if splice is not None and splice[0] != OUT:
        wires.append(splice[0])
    _check_wires(q, wires)
    q.valid = p.valid
    return q


def p_delete(p: Packed, i: int, fallback) -> Packed:
    """moves.DELETE_NODE: pins reading node i read its pin 0's source (or `fallback` when it has no pins or pin 0
    reads i itself); later nodes are renumbered down by one."""
    if not 0 <= i < p.n:
        raise IndexError("node out of range")
    if fallback[0] == i:
        raise InvalidCircuit("fallback source on the deleted node")
    a, b = int(p.ptr[i]), int(p.ptr[i + 1])
    r = fallback
    if b > a:
        s0 = int(p.src[a])
        if s0 != i:
            r = (s0, int(p.port[a]))
    rs = r[0] - 1 if r[0] > i else r[0]
    ptr = np.concatenate((p.ptr[:i + 1], p.ptr[i + 2:] - (b - a)))
    src = np.concatenate((p.src[:a], p.src[b:]))
    port = np.concatenate((p.port[:a], p.port[b:]))
    hit = src == i
    wires = ()
    if hit.any() or (p.out_src == i).any():
        _check_ref(p, r[0], r[1])                   # the redirect target, in the parent's numbering (r != i)
    if hit.any():
        port = port.copy()
        port[hit] = r[1]
        if p.n_wire:
            nop = np.repeat(np.arange(p.n - 1, dtype=np.int64), np.diff(ptr))
            wires = np.unique(nop[hit]).tolist()
    src = np.where(src > i, src - 1, src).astype(np.int32)
    if hit.any():
        src[hit] = rs
    os_ = p.out_src
    ohit = os_ == i
    out_port = p.out_port
    if ohit.any():
        out_port = out_port.copy()
        out_port[ohit] = r[1]
    out_src = np.where(os_ > i, os_ - 1, os_).astype(np.int32)
    if ohit.any():
        out_src[ohit] = rs
    q = Packed(p.n_in, np.concatenate((p.op[:i], p.op[i + 1:])), ptr, src, port,
               np.concatenate((p.tt[:i], p.tt[i + 1:])), np.concatenate((p.init[:i], p.init[i + 1:])),
               np.concatenate((p.drive[:i], p.drive[i + 1:])), np.concatenate((p.rise[:a], p.rise[b:])),
               np.concatenate((p.fall[:a], p.fall[b:])), out_src, out_port, p.n_wire - (int(p.op[i]) == WIRE))
    q._lab_from = (p, "del", i)
    if wires:
        _check_wires(q, wires)
    q.valid = p.valid
    return q


def p_duplicate(p: Packed, S, redirect) -> Packed:
    """moves.DUPLICATE: copies of the node set S (internal wiring kept, external sources shared) appended in S
    order; redirect = (consumer, pin) of an original port moved to the copy, or None."""
    S = list(S)
    n = p.n
    Sa = np.array(S, np.int64)
    if len(S) and (Sa.min() < 0 or Sa.max() >= n):
        raise IndexError("node out of range")
    pos = np.full(n, -1, np.int64)
    pos[Sa] = n + np.arange(len(S))
    k = p.ptr[Sa + 1] - p.ptr[Sa]
    pin_ix = (np.repeat(p.ptr[Sa] - np.concatenate(([0], np.cumsum(k)[:-1])), k) + np.arange(int(k.sum()))) \
        if len(S) else np.zeros(0, np.int64)
    cs = p.src[pin_ix].astype(np.int64)
    inS = np.zeros(len(cs), bool)
    m = cs >= 0
    inS[m] = pos[cs[m]] >= 0
    cs = np.where(inS, pos[np.maximum(cs, 0)], cs)
    ptr = np.concatenate((p.ptr, p.P + np.cumsum(k)))
    src = np.concatenate((p.src, cs.astype(np.int32)))
    port = np.concatenate((p.port, p.port[pin_ix]))
    out_src, out_port = p.out_src, p.out_port
    wires = [n + j for j in range(len(S)) if int(p.op[S[j]]) == WIRE] if p.n_wire else []
    if redirect is not None:
        ca, cb = redirect
        r = p.ref_at(ca, cb)
        if r[0] < 0 or pos[r[0]] < 0 or (ca >= 0 and pos[ca] >= 0):
            raise InvalidCircuit("DUPLICATE redirect must move an external pin of a copied port")
        new = int(pos[r[0]])
        if ca == OUT:
            out_src = _set(out_src, cb, new)
        else:
            src = _set(src, int(p.ptr[ca]) + cb, new)
            wires.append(ca)
    q = Packed(p.n_in, np.concatenate((p.op, p.op[Sa])), ptr, src, port, np.concatenate((p.tt, p.tt[Sa])),
               np.concatenate((p.init, p.init[Sa])), np.concatenate((p.drive, p.drive[Sa])),
               np.concatenate((p.rise, p.rise[pin_ix])), np.concatenate((p.fall, p.fall[pin_ix])), out_src, out_port,
               p.n_wire + sum(1 for s in S if int(p.op[s]) == WIRE) if p.n_wire else 0)
    q._lab_from = (p, "append", tuple(S))
    _check_wires(q, wires)
    q.valid = p.valid
    return q


def p_adjust(p: Packed, i: int, field: str, pin: int, v: int) -> Packed:
    """moves.ADJUST_PARAM on a GATE or DELAY: field in rise, fall, init, drive."""
    op = int(p.op[i])
    if op == WIRE:
        raise InvalidCircuit("parameter move on a token without parameters")
    if field in ("rise", "fall"):
        a = int(p.ptr[i])
        k = int(p.ptr[i + 1]) - a
        if not -k <= pin < k:
            raise IndexError("pin out of range")
        j = a + (pin % k)
        q = p.derive(**{field: _set(getattr(p, field), j, v)})
    elif field == "init":
        if op == GATE and int(p.ptr[i + 1] - p.ptr[i]) == 0:
            raise InvalidCircuit("constant init is fixed by its truth table")
        if not 0 <= v < 256:
            raise InvalidCircuit("init must be 0 or 1")
        q = p.derive(init=_set(p.init, i, v))
    else:
        if not -2**31 <= v < 2**31:
            raise InvalidCircuit("drive out of range")
        q = p.derive(drive=_set(p.drive, i, v))
    q._lab_from = (p, "set", i)
    _check_node(q, i)
    q.valid = p.valid
    return q


# --- canonical-hash graph -------------------------------------------------------------------------------------------

_IN_LABELS: dict = {}
_OUT_LABELS: dict = {}


def _input_labels(I: int):
    v = _IN_LABELS.get(I)
    if v is None:
        vb = [struct.pack("<H", 6) + b"\xffI" + struct.pack("<I", i) for i in range(I)]
        v = _IN_LABELS[I] = (vb, _labs(vb))
    return v


def _out_label(j: int, q: int) -> int:
    v = _OUT_LABELS.get((j, q))
    if v is None:
        v = _OUT_LABELS[(j, q)] = _h8(struct.pack("<cII", b"O", j, q))
    return v


def canon_arrays(p: Packed):
    """The pieces of canon._Graph for a packed circuit: (vbytes, lab0, dst, pin, src, port, outputs, header), equal
    to what canon._circuit_graph builds from the Node form."""
    I, n = p.n_in, p.n
    in_vb, in_lab = _input_labels(I)
    vb, lab = p.labels()
    vbytes = in_vb + vb
    lab0 = np.concatenate((in_lab, lab))
    nop = p.node_of_pin()
    dst = I + nop
    pin = np.arange(p.P, dtype=np.int64) - p.ptr[nop]
    s = p.src.astype(np.int64)
    isn = s >= 0
    src = np.where(isn, I + s, p.port)
    port = np.where(isn, p.port, 0).astype(np.int64)
    outputs = []
    for j, (s_, q) in enumerate(zip(p.out_src.tolist(), p.out_port.tolist())):
        v, q = (q, 0) if s_ == INPUT else (I + s_, q)
        outputs.append((v, q))
        lab0[v] = (int(lab0[v]) + _out_label(j, q)) & _MASK64
    return vbytes, lab0, dst, pin, src, port, outputs, struct.pack("<cII", b"C", I, p.n_out)
