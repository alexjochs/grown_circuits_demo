"""Export circuits to HDL testbenches (for cross-checking against independent simulators) and read VCD files.

Verilog: one sequential UDP per distinct (function, arity, initial value), instanced with #(rise, fall). Per-gate
delays only (Verilog primitives have no per-arc delays). Timescale 1 fs, so 1 tick = 1 fs.

VHDL (2008): one process per gate implementing SPEC §3 with `after` (inertial, reject = delay) and a zero-delay
transport self-assignment to cancel. Per-arc delays are supported through 'event on each pin.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from .netlist import Circuit


def _net(n: int) -> str:
    return f"n{n}"


def gate_delays(c: Circuit, g: int) -> tuple[int, int]:
    """(rise, fall) of a gate whose arcs all share one delay pair; raises otherwise."""
    p = c.pins(g)
    dr, df = set(c.dr[p].tolist()), set(c.df[p].tolist())
    if len(dr) > 1 or len(df) > 1:
        raise ValueError(f"gate {g} has per-arc delays, which Verilog primitives cannot express")
    return dr.pop(), df.pop()


def verilog_settle_time(c: Circuit) -> int:
    """Length of the reset phase prepended by to_verilog (all gate outputs reach their initial values by then)."""
    return 2 * int(max(c.dr.max(initial=0), c.df.max(initial=0))) + 1


def to_verilog(c: Circuit, stimulus, T: int, vcd_path: str = "out.vcd") -> str:
    """Testbench simulating [0, T) of the spec, shifted by S = verilog_settle_time(c).

    Icarus propagates a UDP's initial value through the instance delay, so nets start at X. Every UDP therefore
    gets an extra first input `rst`. While rst = 1 the output is driven to the gate's initial value, and by time S
    every output has settled there. Releasing rst at S makes every gate evaluate against the initial values of its
    inputs at once, which is exactly the spec's power-up evaluation (SPEC §5) with per-gate delays. Spec time t
    maps to Verilog time S + t."""
    I, G = c.n_inputs, c.n_gates
    k = c.arity
    S = verilog_settle_time(c)
    out = ["`timescale 1fs/1fs", ""]
    udps = {}
    for g in range(G):
        if k[g] == 0:
            continue
        key = (int(k[g]), int(c.tt[g]), int(c.init[I + g]))
        if key not in udps:
            name = f"fr_k{key[0]}_{key[1]:x}_{key[2]}"
            udps[key] = name
            kk, tt, y0 = key
            ins = ["rst"] + [f"i{i}" for i in range(kk)]
            out.append(f"primitive {name} (y, {', '.join(ins)});")
            out.append(f"  output y; reg y; input {', '.join(ins)};")
            out.append(f"  initial y = 1'b{y0};")
            out.append("  table")
            out.append(f"    1 {' '.join('?' * kk)} : ? : {y0};")
            for r in range(1 << kk):
                cols = " ".join(str((r >> i) & 1) for i in range(kk))
                out.append(f"    0 {cols} : ? : {(tt >> r) & 1};")
            out.append("  endtable")
            out.append("endprimitive")
            out.append("")
    out.append("module top;")
    out.append("  reg rst = 1'b1;")
    for n in range(I):
        out.append(f"  reg {_net(n)} = 1'b{int(c.init[n])};")
    for g in range(G):
        n = I + g
        if k[g] == 0:
            out.append(f"  wire {_net(n)} = 1'b{int(c.tt[g]) & 1};")
        else:
            out.append(f"  wire {_net(n)};")
    for g in range(G):
        if k[g] == 0:
            continue
        r, f = gate_delays(c, g)
        name = udps[(int(k[g]), int(c.tt[g]), int(c.init[I + g]))]
        pins = ", ".join(_net(int(x)) for x in c.fanin[c.pins(g)])
        out.append(f"  {name} #({r}, {f}) g{g} ({_net(I + g)}, rst, {pins});")
    out.append(f"  initial begin #{S}; rst = 1'b0; end")
    out.append("  initial begin")
    t_prev = 0
    for t, n, v in sorted(stimulus):
        if t <= 0:
            raise ValueError("Verilog export: stimulus at time 0 coincides with the power-up evaluation")
        if S + t > t_prev:
            out.append(f"    #{S + t - t_prev};")
            t_prev = S + t
        out.append(f"    {_net(n)} = 1'b{v};")
    out.append("  end")
    out.append(f'  initial begin $dumpfile("{vcd_path}"); $dumpvars(0, top); #{S + T} $finish; end')
    out.append("endmodule")
    return "\n".join(out) + "\n"


def to_vhdl(c: Circuit, stimulus) -> str:
    I, G = c.n_inputs, c.n_gates
    k = c.arity
    out = ["entity top is", "end entity;", "", "architecture sim of top is"]
    for n in range(c.n_nets):
        out.append(f"  signal {_net(n)} : bit := '{int(c.init[n])}';")
    out.append("begin")
    for g in range(G):
        if k[g] == 0:
            continue
        y = _net(I + g)
        pins = [int(x) for x in c.fanin[c.pins(g)]]
        sens = ", ".join(_net(n) for n in dict.fromkeys(pins))
        tt = int(c.tt[g])
        tt_bits = "".join(str((tt >> r) & 1) for r in range(1 << int(k[g])))
        dr, df = c.dr[c.pins(g)], c.df[c.pins(g)]
        out.append(f"  g{g}: process ({sens})")
        out.append(f"    constant TT : bit_vector(0 to {len(tt_bits) - 1}) := \"{tt_bits}\";")
        out.append("    variable first : boolean := true;")
        out.append("    variable idx : natural;")
        out.append("    variable v : bit;")
        out.append("    variable d : time;")
        out.append("  begin")
        out.append("    idx := 0;")
        for i, n in enumerate(pins):
            out.append(f"    if {_net(n)} = '1' then idx := idx + {1 << i}; end if;")
        out.append("    v := TT(idx);")
        out.append(f"    if v = {y} then")
        out.append(f"      {y} <= transport {y};")
        out.append("    else")
        out.append("      d := time'high;")
        for i, n in enumerate(pins):
            out.append(f"      if first or {_net(n)}'event then")
            out.append(f"        if v = '1' then d := minimum(d, {int(dr[i])} fs); "
                       f"else d := minimum(d, {int(df[i])} fs); end if;")
            out.append("      end if;")
        out.append(f"      {y} <= v after d;")
        out.append("    end if;")
        out.append("    first := false;")
        out.append("  end process;")
    out.append("  stim: process")
    out.append("  begin")
    for t, n, v in sorted(stimulus, key=lambda e: (e[1], e[0])):
        out.append(f"    {_net(n)} <= transport '{v}' after {int(t)} fs;")
    out.append("    wait;")
    out.append("  end process;")
    out.append("end architecture;")
    return "\n".join(out) + "\n"


_SCALE = {"fs": 1, "ps": 10**3, "ns": 10**6, "us": 10**9, "ms": 10**12, "s": 10**15}


def read_vcd(path: str, n_nets: int):
    """Parse a VCD written for an exported circuit. Returns (initial values at time 0, edges).

    edges: int64[M, 3] rows (t, net, value) with t > 0, one row per actual value change, ordered by (t, net).
    Times are converted to fs (= ticks). Nets are matched by name n<index>; other signals are ignored."""
    ids = defaultdict(list)
    scale = 1
    with open(path) as fh:
        text = fh.read()
    head, _, body = text.partition("$enddefinitions")
    m = re.search(r"\$timescale\s+(\d+)\s*(\w+)\s+\$end", head)
    if m:
        scale = int(m.group(1)) * _SCALE[m.group(2)]
    for m in re.finditer(r"\$var\s+\S+\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+\[[^\]]*\])?\s+\$end", head):
        width, code, name = m.groups()
        mm = re.fullmatch(r"n(\d+)", name.lower())
        if mm and int(width) == 1 and int(mm.group(1)) < n_nets:
            ids[code].append(int(mm.group(1)))
    t = 0
    cur = {}
    last = {}  # net -> value at the end of the current timestamp
    rows = []
    init = np.full(n_nets, -1, dtype=np.int64)

    def flush(time):
        for n, v in sorted(cur.items()):
            if time == 0:
                init[n] = v
            elif last.get(n, init[n]) != v:
                rows.append((time, n, v))
            last[n] = v
        cur.clear()

    for tok in body.split()[1:]:
        if tok.startswith("#"):
            flush(t)
            t = int(tok[1:]) * scale
        elif tok[0] in "01xXzZ" and len(tok) > 1 and tok[1:] in ids:
            v = {"0": 0, "1": 1}.get(tok[0], -1)
            for n in ids[tok[1:]]:
                cur[n] = v
        elif tok.startswith("$"):
            continue
    flush(t)
    return init, np.array(rows, dtype=np.int64).reshape(-1, 3)
