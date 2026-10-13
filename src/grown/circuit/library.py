"""Library tokens: modules with typed ports, defined as token graphs of other tokens.

A `ModuleDef` wraps a body `Circuit` whose primary inputs are the module's input ports and whose outputs are its
output ports. Bodies may instantiate other modules, so definitions are hierarchical; they bottom out in
primitives. A definition is identified by a canonical hash of its body (ordered ports), port types and parameter
slots, so the same module admitted twice is one token.

Parameter slots expose body parameters to instances: slot s = (body node, field, pin) with field in
{"rise", "fall", "init", "drive"}, or "param" for slot `pin` of a nested module instance in the body (whose `params`
must then be explicit). An instance's `params` tuple (one value per slot, or empty for the defaults) overrides those
fields when the instance is flattened or inlined. Exactly mined modules have no slots (their parameters are part of
the pattern); `expose_delays=True` exposes every arc delay; modules mined modulo parameters (archive.mining, mode
"param") expose every parameter of the body (`param_slots`).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field

from .graph import DELAY, GATE, INPUT, MODULE, SLOW_DRIVE, Circuit, Node, validate

FIELDS = ("rise", "fall", "init", "drive", "param")


@dataclass(frozen=True, eq=False)
class ModuleDef:
    body: Circuit
    in_types: tuple
    out_types: tuple
    slots: tuple = ()               # ((node, field, pin), ...)
    name: str = ""
    hash: bytes = b""
    passthrough: dict = field(default_factory=dict)   # out port -> in port, where the output is a zero-delay alias
    n_prims: int = 0                # primitive gates after full flattening

    @property
    def n_in(self) -> int:
        return self.body.n_in

    @property
    def n_out(self) -> int:
        return self.body.n_out

    def defaults(self) -> tuple:
        return tuple(slot_value(self.body.nodes[i], f, p) for i, f, p in self.slots)

    def resolved_body(self, params=()) -> Circuit:
        """The body with an instance's slot values applied."""
        if not params:
            return self.body
        nodes = list(self.body.nodes)
        for (i, f, p), v in zip(self.slots, params):
            nodes[i] = set_slot(nodes[i], f, p, v)
        return Circuit(self.body.n_in, tuple(nodes), self.body.outputs)


def slot_value(node: Node, f: str, pin: int) -> int:
    if f in ("rise", "fall"):
        return getattr(node, f)[pin]
    if f == "param":
        return node.params[pin]
    return getattr(node, f)


def set_slot(node: Node, f: str, pin: int, v: int) -> Node:
    if f in ("rise", "fall", "param"):
        name = "params" if f == "param" else f
        t = list(getattr(node, name))
        t[pin] = int(v)
        return dataclasses.replace(node, **{name: tuple(t)})
    return dataclasses.replace(node, **{f: int(v)})


def param_slots(body: Circuit) -> tuple:
    """Every parameter of a body as a slot, in body order: per GATE of arity k > 0 its k rise and k fall delays,
    then its init; per DELAY rise, fall, init (a slow delay element: init only, its level is structure); per nested
    MODULE instance each of its (explicit) params. WIRE and
    constant gates (init fixed by the truth table) have none; drive (reserved, ignored by the simulators) stays
    structure. len(param_slots(b)) == b.description_length(params=True) - b.description_length()."""
    out = []
    for i, nd in enumerate(body.nodes):
        if nd.op == GATE and nd.k:
            out += [(i, "rise", p) for p in range(nd.k)] + [(i, "fall", p) for p in range(nd.k)] + [(i, "init", 0)]
        elif nd.op == DELAY and nd.drive == SLOW_DRIVE:   # a slow element's delays are its level: structure
            out += [(i, "init", 0)]
        elif nd.op == DELAY:
            out += [(i, "rise", 0), (i, "fall", 0), (i, "init", 0)]
        elif nd.op == MODULE:
            out += [(i, "param", j) for j in range(len(nd.params))]
    return tuple(out)


def def_hash(body: Circuit, in_types, out_types, slots) -> bytes:
    """Hash of a definition: the body's canonical form with ordered ports. Slots are stored by canonical node
    position so that relabeling the body does not change the hash."""
    from .canon import canonical
    ser, order, _ = canonical(body)
    pos = {old: new for new, old in enumerate(order)}
    extra = json.dumps([list(in_types), list(out_types), sorted([pos[i], f, p] for i, f, p in slots)]).encode()
    return hashlib.blake2b(b"M" + ser + extra, digest_size=16).digest()


class Library:
    """The vocabulary's library part. Primitives (GATE, DELAY, WIRE) are implicit and never leave."""

    def __init__(self):
        self.defs: dict = {}         # hash -> ModuleDef, in admission order
        self.retired: set = set()
        self.uses: dict = {}         # hash -> count of accepted edits that inserted it (maintained by the caller)

    def __len__(self):
        return len(self.defs)

    def __contains__(self, h):
        return h in self.defs

    def get(self, h: bytes):
        return self.defs.get(h)

    def active(self) -> list:
        return [h for h in self.defs if h not in self.retired]

    def admit(self, body: Circuit, in_types=None, out_types=None, slots=(), name: str = "",
              expose_delays: bool = False) -> ModuleDef:
        """Add a module whose ports are `body`'s inputs and outputs. Idempotent: returns the existing definition
        when an identical one is present. Every module the body instantiates must already be in the library."""
        validate(body, self)
        in_types = tuple(in_types or ("bit",) * body.n_in)
        out_types = tuple(out_types or ("bit",) * body.n_out)
        if expose_delays and not slots:
            slots = tuple((i, f, p) for i, nd in enumerate(body.nodes) if nd.op in (GATE, DELAY)
                          for f in ("rise", "fall") for p in range(len(getattr(nd, f))))
        slots = tuple(tuple(s) for s in slots)
        h = def_hash(body, in_types, out_types, slots)
        if h in self.defs:
            self.retired.discard(h)
            return self.defs[h]
        d = ModuleDef(body, in_types, out_types, slots, name or f"m{len(self.defs)}_{h.hex()[:8]}", h,
                      self._passthrough(body), self._count_prims(body))
        self.defs[h] = d
        self.uses.setdefault(h, 0)
        return d

    def slot_field(self, h: bytes, s: int) -> str:
        """The primitive field ("rise", "fall", "init", "drive") behind slot s of module h; nested "param" slots
        are followed down to the module that owns the parameter."""
        d = self.defs[h]
        i, f, p = d.slots[s]
        while f == "param":
            d = self.defs[d.body.nodes[i].module]
            i, f, p = d.slots[p]
        return f

    def retire(self, h: bytes) -> None:
        """Stop offering a module to the edit operator. Existing instances keep flattening."""
        self.retired.add(h)

    def instance(self, h: bytes, ins, params=()) -> Node:
        d = self.defs[h]
        ins = tuple(tuple(r) for r in ins)
        if len(ins) != d.n_in:
            raise ValueError(f"module {d.name} takes {d.n_in} inputs, got {len(ins)}")
        return Node(MODULE, ins, module=h, n_out=d.n_out, params=tuple(int(x) for x in params))

    def _passthrough(self, body: Circuit) -> dict:
        from .graph import alias_target
        pt = {}
        for j, r in enumerate(body.outputs):
            t = alias_target(body, r, self)
            if t is not None and t[0] == INPUT:
                pt[j] = t[1]
        return pt

    def _count_prims(self, body: Circuit) -> int:
        n = 0
        for nd in body.nodes:
            if nd.op in (GATE, DELAY):
                n += 1
            elif nd.op == MODULE:
                n += self.defs[nd.module].n_prims
        return n

    # --- persistence ----------------------------------------------------------------------------------------------
    def to_json(self) -> str:
        from .codec import encode
        return json.dumps({"defs": [{"hash": d.hash.hex(), "name": d.name, "in_types": list(d.in_types),
                                     "out_types": list(d.out_types), "slots": [list(s) for s in d.slots],
                                     "body": encode(d.body), "retired": d.hash in self.retired,
                                     "uses": self.uses.get(d.hash, 0)} for d in self.defs.values()]})

    @staticmethod
    def from_json(s: str) -> "Library":
        from .codec import decode
        lib = Library()
        for e in json.loads(s)["defs"]:
            d = lib.admit(decode(e["body"]), e["in_types"], e["out_types"], [tuple(x) for x in e["slots"]], e["name"])
            if d.hash.hex() != e["hash"]:
                raise ValueError(f"module {e['name']}: hash mismatch on load (format change?)")
            if e.get("retired"):
                lib.retired.add(d.hash)
            lib.uses[d.hash] = e.get("uses", 0)
        return lib

    def save(self, path) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path) -> "Library":
        with open(path) as f:
            return Library.from_json(f.read())


def module_body_len(lib: Library, h: bytes) -> int:
    """Description length (tokens) of a definition's body, the one-time cost of naming it."""
    return lib.defs[h].body.description_length()

