"""Smoke run of the grown-circuits plumbing on a toy target.

Toy task: two random input bit streams; output 0 must be in0 XOR in1 (combinational), output 1 must be in0 of the
previous step (one step of memory). Per-step error is the mean mismatch of the sampled outputs.

    .venv/bin/python scripts/grown_loop.py --generations 200 --backend sync --out runs/grown_smoke
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from grown import loop  # noqa: E402
from grown.archive import Archive, Axis  # noqa: E402
from grown.circuit import Library, OperatorConfig, UniformOperator  # noqa: E402
from grown.log import EditLog  # noqa: E402
from grown.sim import FreerunSubstrate, SyncSubstrate  # noqa: E402


class ToyTarget:
    n_out = 2

    def __init__(self, steps: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.inputs = rng.integers(0, 2, (steps, 2)).astype(np.uint8)
        x = self.inputs
        prev = np.concatenate([[0], x[:-1, 0]])
        self.target = np.stack([x[:, 0] ^ x[:, 1], prev], 1).astype(np.uint8)

    def __call__(self, res):
        return (res.outputs != self.target[None]).mean(2)          # [B, S]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--backend", choices=["sync", "freerun"], default="sync")
    ap.add_argument("--sampling", choices=["uniform", "curiosity", "lineage"], default="curiosity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()

    target = ToyTarget(args.steps, args.seed)
    substrate = SyncSubstrate() if args.backend == "sync" else FreerunSubstrate()
    library = Library()
    archive = Archive([Axis("n_gates", 0, 64, 16), Axis("cycle_frac", 0, 1, 8),
                       Axis("toggle_rate", 0, 0.5, 8, source="behavior")],
                      sig_dim=loop.SIG_DIM, qd_offset=-1.0, library=library)
    op = UniformOperator(OperatorConfig(max_nodes=64), library)
    log = EditLog(os.path.join(args.out, "edits"), sig_dim=loop.SIG_DIM) if args.out else None
    cfg = loop.LoopConfig(generations=args.generations, batch=args.batch, K=args.K, sampling=args.sampling,
                          seed=args.seed)
    t0 = time.time()

    def report(st):
        g = st["generation"]
        if g % args.every == 0 or g == args.generations - 1:
            print(f"gen {g:5d}  coverage {st['coverage']:.4f}  qd {st['qd_score']:9.3f}  "
                  f"best {st['max_fitness']:+.4f}  accepted {st['accepted']:3d}/{args.batch}  "
                  f"{time.time() - t0:7.1f}s", flush=True)

    hist = loop.run(substrate, target, archive, log, target.inputs, op, cfg, library=library, callback=report)
    if args.out:
        log.close()
        archive.save(os.path.join(args.out, "archive"))
        library.save(os.path.join(args.out, "library.json"))
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(hist, f)
        print(json.dumps(log.summary()["accept_rate"], indent=1))


if __name__ == "__main__":
    main()
