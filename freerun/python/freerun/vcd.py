"""VCD (IEEE 1364 value change dump) export of a run window, for waveform viewers such as GTKWave or Surfer.

    win = sim.run(T)[b]
    vcd.write("out.vcd", win, names=annotated.net_names)          # all nets in the trace
    vcd.write("out.vcd", win, nets=[0, 5, 7], timescale="1ps")    # a subset; 1 tick = 1 timescale unit

The window must carry its final values (Window.values), from which the values at its start are derived; the trace
must contain every change of the dumped nets (watch them, or use watch="all").
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _ident(k: int) -> str:
    """Short VCD identifier code from printable ASCII 33..126."""
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 94)
        s = chr(33 + r) + s
    return s


def default_names(n_nets: int, n_inputs: int) -> list[str]:
    return [f"in{i}" if i < n_inputs else f"g{i - n_inputs}" for i in range(n_nets)]


def write(dest, window, names=None, nets=None, n_inputs: int = 0, timescale: str = "1fs", scope: str = "freerun"):
    """Write a VCD of `window`. names: net index -> name (default in<i> / g<j>, which needs n_inputs). nets: the
    net indices to dump (default: every net that appears in the trace). dest: path or text file object."""
    if window.values is None:
        raise ValueError("the window has no final values (Window.values)")
    if not re.fullmatch(r"\s*(1|10|100)\s*(s|ms|us|ns|ps|fs)\s*", timescale):
        raise ValueError(f"bad VCD timescale {timescale!r}")
    N = len(window.values)
    names = list(names) if names is not None else default_names(N, n_inputs)
    trace = np.asarray(window.trace, dtype=np.int64).reshape(-1, 3)
    if nets is None:
        nets = sorted(set(trace[:, 1].tolist())) if len(trace) else []
    nets = [int(n) for n in nets]
    ident = {n: _ident(k) for k, n in enumerate(nets)}
    init = window.initial
    out = io.StringIO()
    out.write(f"$date {datetime.now(timezone.utc).isoformat(timespec='seconds')} $end\n")
    out.write("$version freerun $end\n")
    out.write(f"$timescale {timescale.replace(' ', '')} $end\n")
    out.write(f"$scope module {scope} $end\n")
    for n in nets:
        nm = re.sub(r"\s+", "_", str(names[n]))
        out.write(f"$var wire 1 {ident[n]} {nm} $end\n")
    out.write("$upscope $end\n$enddefinitions $end\n")
    out.write(f"#{window.t0}\n$dumpvars\n")
    for n in nets:
        out.write(f"{int(init[n])}{ident[n]}\n")
    out.write("$end\n")
    keep = np.isin(trace[:, 1], nets) if len(trace) else np.zeros(0, bool)
    last = None
    for t, n, v in trace[keep]:
        if t != last:
            out.write(f"#{int(t)}\n")
            last = t
        out.write(f"{int(v)}{ident[int(n)]}\n")
    if last is None or last < window.t1:
        out.write(f"#{window.t1}\n")
    text = out.getvalue()
    if hasattr(dest, "write"):
        dest.write(text)
    else:
        Path(dest).write_text(text)
    return text


def read(src) -> tuple[dict[str, list[tuple[int, int]]], str]:
    """Minimal reader for single-bit VCDs (as written above): name -> [(time, value)] including the initial dump,
    and the timescale."""
    text = src.read() if hasattr(src, "read") else Path(src).read_text()
    names, out, ts = {}, {}, ""
    t = 0
    toks = text.split("\n")
    in_header = True
    for line in toks:
        line = line.strip()
        if not line:
            continue
        if in_header:
            m = re.match(r"\$var\s+\S+\s+1\s+(\S+)\s+(\S+)", line)
            if m:
                names[m.group(1)] = m.group(2)
                out[m.group(2)] = []
            m = re.match(r"\$timescale\s+(\S+)", line)
            if m:
                ts = m.group(1)
            if line.startswith("$enddefinitions"):
                in_header = False
            continue
        if line.startswith("#"):
            t = int(line[1:])
        elif line[0] in "01" and line[1:] in names:
            out[names[line[1:]]].append((t, int(line[0])))
    return out, ts
