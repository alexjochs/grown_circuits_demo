#!/usr/bin/env python
"""Differential fuzzing of the reference simulator against Icarus Verilog, GHDL and nvc.

Usage: scripts/fuzz_oracles.py [--cases N] [--jobs J] [--seed S] [--out DIR]

Each case is freerun.gen.fuzz_case(seed): a random cyclic circuit and stimulus from a mix of regimes chosen to
exercise the inertial rules.

The VHDL simulators must agree exactly. Icarus is compared on per-gate cases up to the first tie (SPEC §8). Failing
cases are saved to --out as circuit .npz + stimulus .json for replay.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from freerun import gen, oracle  # noqa: E402
from freerun.ref import simulate  # noqa: E402

MAX_EVENTS = 60_000


def make_case(seed):
    c, stim, T, meta = gen.fuzz_case(seed)
    # keep the HDL runs small: shorten T if the circuit is very active
    w = simulate(c, T, stim)
    while w.events > MAX_EVENTS and T > 10:
        T //= 2
        stim = [e for e in stim if e[0] < T]
        w = simulate(c, T, stim)
    return c, stim, T, w, dict(meta, T=T)


def run_case(args):
    seed, outdir = args
    c, stim, T, w, meta = make_case(seed)
    res = dict(meta, events=w.events, cancels=w.cancels, advances=w.advances, ties=w.ties, mismatches=[])
    sims = ["ghdl", "nvc"] + ([] if meta["per_arc"] else ["icarus"])
    for sim in sims:
        upto = w.first_tie if (sim == "icarus" and w.ties) else None
        with tempfile.TemporaryDirectory(dir=os.environ.get("FUZZ_TMP")) as d:
            try:
                init, e = oracle.run(sim, c, stim, T, d)
            except Exception as ex:  # tool failure counts as a mismatch, with the message
                res["mismatches"].append((sim, f"error: {str(ex)[:300]}"))
                continue
        ref = w.trace
        if upto is not None:
            ref, e = ref[ref[:, 0] <= upto], e[e[:, 0] <= upto]
        if not np.array_equal(init, c.init) or not np.array_equal(e, ref):
            first = _first_diff(ref, e)
            res["mismatches"].append((sim, f"first difference: {first}"))
    if res["mismatches"]:
        p = Path(outdir)
        p.mkdir(parents=True, exist_ok=True)
        c.save(p / f"case_{seed}.npz")
        (p / f"case_{seed}.json").write_text(json.dumps(dict(res, stimulus=[list(map(int, x)) for x in stim])))
    return res


def _first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if not np.array_equal(a[i], b[i]):
            return dict(index=i, ref=a[i].tolist(), other=b[i].tolist())
    return dict(index=n, ref_len=len(a), other_len=len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/fuzz_failures")
    a = ap.parse_args()
    for sim in oracle.SIMULATORS:
        if not oracle.available(sim):
            sys.exit(f"{sim} not available; run scripts/build_tools.sh")
    t0 = time.time()
    tot = dict(cases=0, events=0, cancels=0, advances=0, ties=0, comparisons=0, failed=0)
    with mp.Pool(a.jobs) as pool:
        for r in pool.imap_unordered(run_case, [(a.seed + i, a.out) for i in range(a.cases)]):
            tot["cases"] += 1
            for k in ("events", "cancels", "advances", "ties"):
                tot[k] += r[k]
            tot["comparisons"] += 2 + (0 if r["per_arc"] else 1)
            if r["mismatches"]:
                tot["failed"] += 1
                print("MISMATCH", json.dumps({k: r[k] for k in ("seed", "regime", "per_arc", "mismatches")}),
                      flush=True)
            if tot["cases"] % 50 == 0:
                print(json.dumps(tot), f"{time.time() - t0:.0f}s", flush=True)
    print("DONE", json.dumps(tot), f"{time.time() - t0:.0f}s")
    sys.exit(1 if tot["failed"] else 0)


if __name__ == "__main__":
    main()
