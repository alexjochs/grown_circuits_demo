"""Circuit graphs: the token-level representation.

A `Circuit` has `n_in` primary inputs, a tuple of `Node`s and a tuple of output `Ref`s. Every node is one token of
the vocabulary:

  GATE    primitive: a truth table over k <= 6 input pins (pin 0 is the least significant bit of the row index,
          as in freerun), a rise and a fall delay per pin (the arc pin -> output, in substrate ticks), the initial
          value of its output, and a reserved drive-strength slot. Arity 0 is a constant (init == tt & 1).
  DELAY   primitive: an explicit delay element, one input, identity function, one rise and one fall delay. With
          drive == SLOW_DRIVE it is a *slow delay element* (a current-starved stage characterized in SKY130,
          grown.sim.slow): its rise and fall are its delays at one bias level of the table, and only the level is
          a parameter (the operator moves it; rise/fall follow).
  WIRE    primitive: one input, identity, zero delay. Flattening aliases it away. Interconnect delay lives on the
          consuming gate's arcs (that is how freerun models it).
  MODULE  library token: an instance of a `ModuleDef` (see library.py), referenced by the definition's canonical
          hash, with typed ports and optional parameter-slot values.

A `Ref` is a pointer to an output port: `(node, port)` with `node == INPUT` (-1) meaning primary input `port`.
Pointers may target any port, including later nodes and the node itself, because clockless circuits need
feedback (latches, oscillators). Node order carries no meaning: canonical hashing (canon.py) is invariant under
relabeling, and flattening is deterministic for a given order.

Everything is immutable. Edits build new tuples that share unchanged `Node` objects with the parent.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import numpy as np

GATE, DELAY, WIRE, MODULE = 0, 1, 2, 3
OP_NAMES = ("gate", "delay", "wire", "module")
INPUT = -1          # Ref node id of a primary input (or of a module definition's input port)
OUT = -2            # consumer id of a circuit output (used by edit helpers and moves)
MAX_ARITY = 6
MAX_DELAY = 2**31 - 1
SLOW_DRIVE = 1024   # DELAY.drive of a slow delay element (grown.sim.slow): its delays follow its bias level

Ref = tuple  # (node, port)


def tt_mask(k: int) -> int:
    return (1 << (1 << k)) - 1


@dataclass(frozen=True, eq=True)
class Node:
    op: int
    ins: tuple = ()                 # tuple[Ref, ...], one per input pin
    tt: int = 0                     # GATE: truth table
    rise: tuple = ()                # GATE: per pin; DELAY: (d,)
    fall: tuple = ()
    init: int = 0                   # GATE / DELAY: initial output value
    drive: int = 1                  # reserved (drive strength); hashed, ignored by the simulators for now
    module: bytes = b""             # MODULE: definition hash (16 bytes)
    n_out: int = 1                  # number of output ports (MODULE: from the definition)
    params: tuple = ()              # MODULE: parameter-slot values (empty = definition defaults)

    @property
    def k(self) -> int:
        return len(self.ins)

    @cached_property
    def key(self) -> bytes:
        """Bytes identifying the token and its parameters (not its connections). Used by hashing."""
        k = len(self.ins)
        if self.op == GATE:
            return struct.pack(f"<BBQBi{2 * k}i", GATE, k, self.tt, self.init, self.drive, *self.rise, *self.fall)
        if self.op == DELAY:
            return struct.pack("<BBiii", DELAY, self.init, self.drive, self.rise[0], self.fall[0])
        if self.op == WIRE:
            return b"\x02"
        return struct.pack(f"<B16sHHH{len(self.params)}q", MODULE, self.module, k, self.n_out, len(self.params),
                           *self.params)

    def with_ins(self, ins) -> "Node":
        """A copy with new input refs (the cached token key carries over: it does not depend on connections)."""
        nd = object.__new__(Node)
        nd.__dict__.update(self.__dict__)
        nd.__dict__["ins"] = tuple(ins)
        return nd


def param_tokens(nd: Node) -> int:
    """Parameter values a token carries, for the description length with parameters: a GATE of arity k > 0 its
    k rise and k fall delays and its init (a constant's init is its truth table), a DELAY rise, fall and init (a slow
    delay element its init: its level is part of the token), a
    WIRE none, a MODULE instance its explicit slot values (none for the definition's defaults). The truth table
    and drive are part of the token itself."""
    if nd.op == GATE:
        return 2 * len(nd.ins) + 1 if nd.ins else 0
    if nd.op == DELAY:
        return 1 if nd.drive == SLOW_DRIVE else 3
    if nd.op == MODULE:
        return len(nd.params)
    return 0


def gate(tt: int, ins, rise=1, fall=1, init: int = 0, drive: int = 1) -> Node:
    ins = tuple(tuple(r) for r in ins)
    k = len(ins)
    rise = tuple(int(x) for x in rise) if np.ndim(rise) else (int(rise),) * k
    fall = tuple(int(x) for x in fall) if np.ndim(fall) else (int(fall),) * k
    if k == 0:
        init = int(tt) & 1
    return Node(GATE, ins, int(tt), rise, fall, int(init), int(drive))


def delay(src, rise: int = 1, fall: int = 1, init: int = 0) -> Node:
    return Node(DELAY, (tuple(src),), 0, (int(rise),), (int(fall),), int(init))


def wire(src) -> Node:
    return Node(WIRE, (tuple(src),))


def inp(i: int) -> Ref:
    return (INPUT, i)


@dataclass(frozen=True, eq=False)
class Circuit:
    n_in: int
    nodes: tuple = ()
    outputs: tuple = ()

    @property
    def n_out(self) -> int:
        return len(self.outputs)

    def __len__(self) -> int:
        return len(self.nodes)

    @cached_property
    def hash(self) -> bytes:
        """Canonical 16-byte hash: structure plus parameters, invariant under node relabeling."""
        from .canon import circuit_hash
        return circuit_hash(self)

    def __eq__(self, other) -> bool:
        return isinstance(other, Circuit) and self.hash == other.hash

    def __hash__(self) -> int:
        return int.from_bytes(self.hash[:8], "little")

    def same_as(self, other: "Circuit") -> bool:
        """Literal equality (same node order), as opposed to `==` (isomorphism)."""
        return self.n_in == other.n_in and self.nodes == other.nodes and self.outputs == other.outputs

    def ports(self) -> list:
        """Every output port a pin may point to: primary inputs, then each node's ports."""
        out = [(INPUT, i) for i in range(self.n_in)]
        for i, n in enumerate(self.nodes):
            out.extend((i, p) for p in range(n.n_out))
        return out

    def n_ports(self) -> int:
        return self.n_in + sum(n.n_out for n in self.nodes)

    def consumers(self) -> dict:
        """(node, port) -> list of (consumer node or OUT, pin). Ports with no consumer are absent."""
        cons: dict = {}
        for i, n in enumerate(self.nodes):
            for p, r in enumerate(n.ins):
                cons.setdefault(r, []).append((i, p))
        for j, r in enumerate(self.outputs):
            cons.setdefault(r, []).append((OUT, j))
        return cons

    def pins(self) -> list:
        """Every input pin (consumer, pin): node pins then circuit outputs."""
        out = [(i, p) for i, n in enumerate(self.nodes) for p in range(len(n.ins))]
        out.extend((OUT, j) for j in range(len(self.outputs)))
        return out

    def ref_at(self, consumer: int, pin: int) -> Ref:
        return self.outputs[pin] if consumer == OUT else self.nodes[consumer].ins[pin]

    def description_length(self, params: bool = False) -> int:
        """Tokens to write the circuit down: one per node, one per pin pointer, one per output pointer. With
        `params`, also one per parameter value (`param_tokens`)."""
        n = sum(1 + len(nd.ins) for nd in self.nodes) + len(self.outputs)
        return n + sum(param_tokens(nd) for nd in self.nodes) if params else n

    def __repr__(self) -> str:
        return f"Circuit(n_in={self.n_in}, nodes={len(self.nodes)}, n_out={self.n_out}, hash={self.hash.hex()[:12]})"


class InvalidCircuit(ValueError):
    pass


def validate(c: Circuit, library=None) -> None:
    """Raise InvalidCircuit unless `c` is well formed. With a library, module instances are checked against their
    definitions and zero-delay alias loops (cycles through wires and module pass-throughs only) are rejected."""
    if c.n_in < 0:
        raise InvalidCircuit("n_in < 0")
    n = len(c.nodes)

    def check_ref(r, where):
        if len(r) != 2:
            raise InvalidCircuit(f"{where}: malformed ref {r}")
        node, port = r
        if node == INPUT:
            if not 0 <= port < c.n_in:
                raise InvalidCircuit(f"{where}: input {port} out of range")
        elif not 0 <= node < n or not 0 <= port < c.nodes[node].n_out:
            raise InvalidCircuit(f"{where}: ref {r} out of range")

    for i, nd in enumerate(c.nodes):
        k = len(nd.ins)
        for p, r in enumerate(nd.ins):
            check_ref(r, f"node {i} pin {p}")
        if nd.op == GATE:
            if k > MAX_ARITY:
                raise InvalidCircuit(f"node {i}: arity {k} > {MAX_ARITY}")
            if nd.tt < 0 or nd.tt & ~tt_mask(k):
                raise InvalidCircuit(f"node {i}: truth table out of range for arity {k}")
            if len(nd.rise) != k or len(nd.fall) != k:
                raise InvalidCircuit(f"node {i}: delays must align with pins")
            if k == 0 and nd.init != (nd.tt & 1):
                raise InvalidCircuit(f"node {i}: constant init must equal tt bit 0")
        elif nd.op == DELAY:
            if k != 1 or len(nd.rise) != 1 or len(nd.fall) != 1:
                raise InvalidCircuit(f"node {i}: delay element takes one pin and one rise/fall delay")
        elif nd.op == WIRE:
            if k != 1:
                raise InvalidCircuit(f"node {i}: wire takes one pin")
        elif nd.op == MODULE:
            if len(nd.module) != 16:
                raise InvalidCircuit(f"node {i}: module hash must be 16 bytes")
            if library is not None:
                d = library.get(nd.module)
                if d is None:
                    raise InvalidCircuit(f"node {i}: unknown module {nd.module.hex()[:12]}")
                if k != d.n_in or nd.n_out != d.n_out:
                    raise InvalidCircuit(f"node {i}: ports do not match module definition")
                if nd.params and len(nd.params) != len(d.slots):
                    raise InvalidCircuit(f"node {i}: {len(nd.params)} params for {len(d.slots)} slots")
        else:
            raise InvalidCircuit(f"node {i}: unknown op {nd.op}")
        if nd.op in (GATE, DELAY):
            if nd.init not in (0, 1):
                raise InvalidCircuit(f"node {i}: init must be 0 or 1")
            if any(not 1 <= d <= MAX_DELAY for d in nd.rise + nd.fall):
                raise InvalidCircuit(f"node {i}: delays must be in [1, 2^31-1]")
    for j, r in enumerate(c.outputs):
        check_ref(r, f"output {j}")
    if any(nd.op == WIRE for nd in c.nodes) or (library is not None and any(nd.op == MODULE for nd in c.nodes)):
        _check_alias_loops(c, library)


def alias_target(c: Circuit, ref: Ref, library=None):
    """Follow wires (and module pass-throughs, given a library) from `ref`. Returns the first ref that is a
    primary input or a delaying port, or None on a zero-delay loop."""
    seen = set()
    while ref[0] != INPUT:
        nd = c.nodes[ref[0]]
        if nd.op == WIRE:
            nxt = nd.ins[0]
        elif nd.op == MODULE and library is not None:
            pt = library.get(nd.module).passthrough.get(ref[1])
            if pt is None:
                return ref
            nxt = nd.ins[pt]
        else:
            return ref
        if ref in seen:
            return None
        seen.add(ref)
        ref = nxt
    return ref


def _check_alias_loops(c: Circuit, library) -> None:
    for i, nd in enumerate(c.nodes):
        if nd.op == WIRE or nd.op == MODULE:
            for p in range(nd.n_out):
                if alias_target(c, (i, p), library) is None:
                    raise InvalidCircuit(f"zero-delay loop through node {i} port {p}")


def is_valid(c: Circuit, library=None) -> bool:
    try:
        validate(c, library)
    except InvalidCircuit:
        return False
    return True


# --- immutable edit helpers -------------------------------------------------------------------------------------

def set_node(c: Circuit, i: int, node: Node) -> Circuit:
    nodes = list(c.nodes)
    nodes[i] = node
    return Circuit(c.n_in, tuple(nodes), c.outputs)


def set_pin(c: Circuit, consumer: int, pin: int, ref: Ref) -> Circuit:
    ref = tuple(ref)
    if consumer == OUT:
        outs = list(c.outputs)
        outs[pin] = ref
        return Circuit(c.n_in, c.nodes, tuple(outs))
    nd = c.nodes[consumer]
    ins = list(nd.ins)
    ins[pin] = ref
    return set_node(c, consumer, nd.with_ins(ins))


def set_pins(c: Circuit, changes: Iterable) -> Circuit:
    """changes: iterable of (consumer, pin, ref)."""
    nodes, outs = list(c.nodes), list(c.outputs)
    pending: dict = {}
    for consumer, pin, ref in changes:
        if consumer == OUT:
            outs[pin] = tuple(ref)
        else:
            pending.setdefault(consumer, {})[pin] = tuple(ref)
    for i, pins in pending.items():
        ins = list(nodes[i].ins)
        for p, r in pins.items():
            ins[p] = r
        nodes[i] = nodes[i].with_ins(ins)
    return Circuit(c.n_in, tuple(nodes), tuple(outs))


def append_nodes(c: Circuit, new: Iterable[Node]) -> Circuit:
    return Circuit(c.n_in, c.nodes + tuple(new), c.outputs)


def delete_nodes(c: Circuit, dead: Iterable[int], redirect: dict | None = None) -> Circuit:
    """Remove nodes `dead`. Every pin that pointed at a removed port is pointed at `redirect[(node, port)]`, which
    must be a ref to a surviving port (it may point through other removed ports; chains are followed). Surviving
    nodes are renumbered in order."""
    dead = set(dead)
    redirect = redirect or {}
    new_index = {}
    j = 0
    for i in range(len(c.nodes)):
        if i not in dead:
            new_index[i] = j
            j += 1

    def remap(r):
        seen = 0
        while r[0] != INPUT and r[0] in dead:
            if r not in redirect:
                raise InvalidCircuit(f"ref {r} points at a deleted node and has no redirect")
            r = redirect[r]
            seen += 1
            if seen > len(redirect) + 1:
                raise InvalidCircuit("redirect cycle")
        return r if r[0] == INPUT else (new_index[r[0]], r[1])

    nodes = tuple(nd.with_ins(remap(r) for r in nd.ins) if nd.ins else nd
                  for i, nd in enumerate(c.nodes) if i not in dead)
    return Circuit(c.n_in, nodes, tuple(remap(r) for r in c.outputs))


def live_nodes(c: Circuit) -> np.ndarray:
    """Boolean mask of nodes in the backward cone of the outputs."""
    live = np.zeros(len(c.nodes), bool)
    stack = [r[0] for r in c.outputs if r[0] != INPUT]
    while stack:
        i = stack.pop()
        if live[i]:
            continue
        live[i] = True
        stack.extend(r[0] for r in c.nodes[i].ins if r[0] != INPUT and not live[r[0]])
    return live


# --- random circuits --------------------------------------------------------------------------------------------

def random_tt(k: int, rng: np.random.Generator) -> int:
    nb = 1 << k
    bits = rng.integers(0, 2, nb)
    return int.from_bytes(np.packbits(bits.astype(np.uint8), bitorder="little").tobytes(), "little")


def random_circuit(n_in: int, n_out: int, n_nodes: int, rng: np.random.Generator, arity=(1, 3),
                   delay_range=(1, 1), feedback: bool = True) -> Circuit:
    """Random gate network. Sources are uniform over all ports (feedback=True) or over inputs and earlier nodes."""
    nodes = []
    for i in range(n_nodes):
        k = int(rng.integers(arity[0], arity[1] + 1))
        lim = n_in + (n_nodes if feedback else i)
        if lim == 0:
            k = 0
        ins = []
        for _ in range(k):
            s = int(rng.integers(0, lim))
            ins.append((INPUT, s) if s < n_in else (s - n_in, 0))
        d = lambda: int(rng.integers(delay_range[0], delay_range[1] + 1))  # noqa: E731
        nodes.append(gate(random_tt(k, rng), ins, [d() for _ in range(k)], [d() for _ in range(k)],
                          init=int(rng.integers(0, 2))))
    lim = n_in + n_nodes
    outs = []
    for _ in range(n_out):
        s = int(rng.integers(0, lim))
        outs.append((INPUT, s) if s < n_in else (s - n_in, 0))
    return Circuit(n_in, tuple(nodes), tuple(outs))
