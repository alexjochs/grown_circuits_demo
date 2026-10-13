#!/usr/bin/env python
"""Jitter of the current-starved delay element (docs/slow_delay.md): transient noise in ngspice.

The noise that sets the timing of a starved stage is the current noise of its starving transistor: in weak
inversion its channel noise is shot-like, S_i = 2 q I (one-sided, A^2/Hz), and in moderate inversion within a factor
of ~1.5 of that. It is injected as a white current source on the slow node (ngspice TRNOISE with rms
na = sqrt(2 q I * B), bandwidth B = 1 / (2 nt)), and the delay is measured on N runs with different seeds. The
analytic expectation for a ramp that integrates white current noise for the delay d at the bias current I is
sigma_d / d = sqrt(q / (I d)) (= sqrt(q / Q) with Q = I d the charge moved to the switching threshold).

    nice -n 10 .venv/bin/python scripts/slow_delay_jitter.py [--runs 24]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slow_delay_char as sd  # noqa: E402

Q = 1.602176634e-19


def delays(p, runs):
    out = []
    d0 = sd.est_delay(p)
    for k in range(runs):
        q = dict(p, seed=1000 + k)
        q["noise_nt"] = d0 / 2000
        q["noise_na"] = float(np.sqrt(2 * Q * p["Ib"] / (2 * q["noise_nt"])))
        r = sd.edges_point(q)
        out.append((r["d_rise"], r["d_fall"]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=24)
    ap.add_argument("--out", default="runs/slow_delay/jitter.json")
    a = ap.parse_args()
    res = []
    for Ib, C in ((1e-6, 1e-13), (1e-8, 0.0), (1e-9, 1e-13)):
        p = dict(kind="cs", Ib=Ib, C=C, corner="tt", temp=25.0, vdd=1.8)
        d = delays(p, a.runs)
        m, s = d.mean(0), d.std(0, ddof=1)
        expect = np.sqrt(Q / (Ib * m))
        row = dict(Ib=Ib, C=C, runs=a.runs, d_rise=m[0], d_fall=m[1], sigma_rise=s[0], sigma_fall=s[1],
                   rel_rise=s[0] / m[0], rel_fall=s[1] / m[1], expect_rel_rise=expect[0], expect_rel_fall=expect[1])
        res.append(row)
        print(json.dumps(row), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
