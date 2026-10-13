"""Reader for flat structural (gate-level) Verilog netlists of standard cells, as written by yosys or OpenROAD.

Supported: one module (or a named one); input/output/wire declarations with optional [msb:lsb] ranges; cell
instances with named port connections (.PIN(net), .PIN(bus[3]), .PIN(1'b0), .PIN()); continuous assignments
between single-bit nets or constants (aliases). Power pins (VPWR, VGND, VPB, VNB) are ignored. Not supported:
behavioural code, hierarchy, concatenations in connections, positional port connections.
"""
from __future__ import annotations

import re

from .cells import CellNetlist, Instance

POWER_PINS = {"VPWR", "VGND", "VPB", "VNB", "VDD", "VSS"}
_TOK = re.compile(r"\s+|//[^\n]*|/\*.*?\*/|\\\S+\s|[A-Za-z_][A-Za-z0-9_$]*|\d+'[bBhHdD][0-9a-fA-FxXzZ_]+|\d+|.", re.S)


def _tokens(text):
    for m in _TOK.finditer(text):
        t = m.group(0)
        if t.isspace() or t.startswith("//") or t.startswith("/*"):
            continue
        yield t.strip()[1:] if t.startswith("\\") else t  # \name is the same identifier as name (IEEE 1364 3.7.1)


def _const(tok):
    m = re.fullmatch(r"1'[bB]([01])", tok)
    return int(m.group(1)) if m else None


def read(text: str, module: str | None = None) -> CellNetlist:
    toks = list(_tokens(text))
    pos = 0

    def expect(t):
        nonlocal pos
        if toks[pos] != t:
            raise ValueError(f"verilog: expected {t!r}, found {toks[pos]!r} (token {pos})")
        pos += 1

    # find the module
    while True:
        while pos < len(toks) and toks[pos] != "module":
            pos += 1
        if pos >= len(toks):
            raise ValueError(f"verilog: module {module!r} not found")
        pos += 1
        name = toks[pos]
        if module is None or name == module:
            break
    pos += 1
    if toks[pos] == "(":
        while toks[pos] != ")":
            pos += 1
        pos += 1
    expect(";")

    decl = {"input": [], "output": [], "wire": []}
    instances, assigns = [], []

    def net_ref():
        """A single-bit net reference: name, name[i], or a constant. Returns ('net', name) or ('const', v)."""
        nonlocal pos
        t = toks[pos]
        pos += 1
        c = _const(t)
        if c is not None:
            return ("const", c)
        if toks[pos] == "[":
            pos += 1
            idx = toks[pos]
            pos += 1
            expect("]")
            return ("net", f"{t}[{idx}]")
        return ("net", t)

    while toks[pos] != "endmodule":
        t = toks[pos]
        if t in decl:
            pos += 1
            rng = None
            if toks[pos] == "[":
                msb, lsb = int(toks[pos + 1]), int(toks[pos + 3])
                if toks[pos + 2] != ":" or toks[pos + 4] != "]":
                    raise ValueError("verilog: bad range")
                rng = (msb, lsb)
                pos += 5
            while True:
                n = toks[pos]
                pos += 1
                if rng is None:
                    decl[t].append(n)
                else:
                    step = -1 if rng[0] >= rng[1] else 1
                    decl[t].extend(f"{n}[{i}]" for i in range(rng[0], rng[1] + step, step))
                if toks[pos] == ",":
                    pos += 1
                    continue
                expect(";")
                break
        elif t == "assign":
            pos += 1
            lhs = net_ref()
            expect("=")
            rhs = net_ref()
            expect(";")
            if lhs[0] != "net":
                raise ValueError("verilog: assignment to a constant")
            assigns.append((lhs[1], rhs))
        else:  # cell instance
            cell = t
            pos += 1
            if toks[pos] == "#":
                raise ValueError(f"verilog: parameterised instance of {cell} not supported")
            inst = toks[pos]
            pos += 1
            expect("(")
            conns = {}
            while toks[pos] != ")":
                if toks[pos] == ",":
                    pos += 1
                    continue
                expect(".")
                pin = toks[pos]
                pos += 1
                expect("(")
                if toks[pos] == ")":
                    pos += 1
                    continue
                if toks[pos] == "{":
                    raise ValueError(f"verilog: concatenation on {inst}.{pin} not supported")
                ref = net_ref()
                expect(")")
                if pin not in POWER_PINS:
                    conns[pin] = ref
            pos += 1
            expect(";")
            instances.append((cell, inst, conns))

    # aliases (assign a = b) via union-find; constants become tie nets
    parent = {}

    def find(n):
        parent.setdefault(n, n)
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    ports = set(decl["input"]) | set(decl["output"])
    consts = {}
    for lhs, rhs in assigns:
        if rhs[0] == "const":
            consts[lhs] = rhs[1]
            continue
        a, b = find(lhs), find(rhs[1])
        if a != b:  # prefer port names as representatives
            if a in ports and b not in ports:
                parent[b] = a
            else:
                parent[a] = b
    const_nets = {}
    for n, v in consts.items():
        const_nets[find(n)] = v
    n_const = 0
    out_instances = []
    for cell, inst, conns in instances:
        c2 = {}
        for pin, ref in conns.items():
            if ref[0] == "const":
                name = f"$const{ref[1]}"
                const_nets[name] = ref[1]
                n_const += 1
                c2[pin] = name
            else:
                c2[pin] = find(ref[1])
        out_instances.append(Instance(inst, cell, c2))
    return CellNetlist(inputs=[find(n) for n in decl["input"]], outputs=[find(n) for n in decl["output"]],
                       instances=out_instances, consts=const_nets)
