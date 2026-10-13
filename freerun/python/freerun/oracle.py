"""Run exported circuits on external HDL simulators (Icarus Verilog, GHDL, nvc) and return their edge lists.

Tool locations: $FREERUN_TOOLS (default: <repo>/tools), as built by scripts/build_tools.sh.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import hdl
from .netlist import Circuit

TOOLS = Path(os.environ.get("FREERUN_TOOLS", Path(__file__).resolve().parents[2] / "tools"))

BINARIES = {
    "iverilog": TOOLS / "iverilog/bin/iverilog",
    "vvp": TOOLS / "iverilog/bin/vvp",
    "ghdl": TOOLS / "ghdl/bin/ghdl",
    "nvc": TOOLS / "nvc/usr/bin/nvc",
}
SIMULATORS = ("icarus", "ghdl", "nvc")


def available(sim: str) -> bool:
    need = {"icarus": ("iverilog", "vvp"), "ghdl": ("ghdl",), "nvc": ("nvc",)}[sim]
    return all(BINARIES[b].exists() for b in need)


def _run(cmd, cwd, timeout):
    r = subprocess.run([str(x) for x in cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


def run(sim: str, c: Circuit, stimulus, T: int, workdir, timeout: float = 600):
    """Simulate [0, T) and return (initial values, edges int64[M, 3] of (t, net, value) with 0 < t < T)."""
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    vcd = wd / "out.vcd"
    if sim == "icarus":
        (wd / "top.v").write_text(hdl.to_verilog(c, stimulus, T, vcd_path="out.vcd"))
        _run([BINARIES["iverilog"], "-o", "top.vvp", "top.v"], wd, timeout)
        _run([BINARIES["vvp"], "-n", "top.vvp"], wd, timeout)
    elif sim == "ghdl":
        (wd / "top.vhd").write_text(hdl.to_vhdl(c, stimulus))
        g = BINARIES["ghdl"]
        _run([g, "-a", "--std=08", "top.vhd"], wd, timeout)
        _run([g, "-e", "--std=08", "top"], wd, timeout)
        _run([g, "-r", "--std=08", "top", f"--vcd={vcd.name}", f"--stop-time={T}fs"], wd, timeout)
    elif sim == "nvc":
        (wd / "top.vhd").write_text(hdl.to_vhdl(c, stimulus))
        _run([BINARIES["nvc"], "--std=2008", "-L", TOOLS / "nvc/usr/lib/x86_64-linux-gnu/nvc", "--work=work:work", "-a", "top.vhd", "-e", "top", "-r",
              f"--stop-time={T}fs", "--format=vcd", f"--wave={vcd.name}"], wd, timeout)
    else:
        raise ValueError(sim)
    init, edges = hdl.read_vcd(str(vcd), c.n_nets)
    if sim == "icarus":  # undo the reset phase (see hdl.to_verilog)
        S = hdl.verilog_settle_time(c)
        before = edges[edges[:, 0] <= S]
        init = init.copy()
        for t, n, v in before:  # values at the end of the reset phase are the spec's initial values
            init[n] = v
        edges = edges[edges[:, 0] > S].copy()
        edges[:, 0] -= S
    return init, edges[edges[:, 0] < T]
