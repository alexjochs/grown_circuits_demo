"""Task families from the command line.

  python scripts/grown_tasks.py sample   --family signal --n 20 --seed 0 [--split extrap]
  python scripts/grown_tasks.py coverage --family signal --n 5000 [--json cov.json]
  python scripts/grown_tasks.py teach    --family bodies --n 8 --iterations 100 [--cache DIR]
  python scripts/grown_tasks.py dataset  --family signal --n 16 --episodes 4 --out runs/datasets/signal
  python scripts/grown_tasks.py dataset  --catalog thompson_tone --episodes 4 --out runs/datasets/catalog

Families: signal, bodies, named_bodies, sensorimotor. Body datasets need teachers: `teach` first, or pass
--train-missing to `dataset`.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from grown.tasks import AXES, TeacherDataset, demand_coverage, get_family  # noqa: E402


def _tasks(a):
    if getattr(a, "catalog", None):
        from grown.tasks.signal import catalog
        cat = catalog()
        return [cat[n] for n in a.catalog.split(",")] if a.catalog != "all" else list(cat.values())
    kw = {}
    if a.family == "signal" and a.kinds:
        kw["kinds"] = a.kinds.split(",")
    if a.family in ("bodies", "named_bodies") and a.horizon:
        kw["horizon"] = a.horizon
    return get_family(a.family, **kw).sample(a.seed, a.n, a.split)


def _fmt(v):
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def cmd_sample(a):
    tasks = _tasks(a)
    print("\t".join(["name", "kind", *AXES, "wires_in", "wires_out", "dt", "horizon"]))
    for t in tasks:
        g = t.tags
        print("\t".join([t.name, str(g.extra.get("kind")), *(_fmt(float(getattr(g, x))) for x in AXES),
                         str(g.extra.get("in_wires")), str(g.extra.get("out_wires")), _fmt(t.dt), str(t.horizon)]))
        if a.verbose:
            print("   ", json.dumps(t.params)[:400])


def _bar(counts, width=40):
    m = max(max(counts), 1)
    return ["#" * int(round(width * c / m)) for c in counts]


def cmd_coverage(a):
    t0 = time.time()
    tasks = _tasks(a)
    cov = demand_coverage(tasks)
    print(f"{cov['n']} tasks ({time.time() - t0:.1f}s), {cov['cells']} occupied demand cells")
    print("families:", cov["families"])
    print("kinds:", cov["kinds"])
    for ax, s in cov["axes"].items():
        print(f"\n{ax}: min {_fmt(s['min'])} q05 {_fmt(s['q05'])} median {_fmt(s['q50'])} q95 {_fmt(s['q95'])} "
              f"max {_fmt(s['max'])}{' (log2 bins)' if s['log2'] else ''}")
        for lo, hi, c, b in zip(s["edges"][:-1], s["edges"][1:], s["hist"], _bar(s["hist"])):
            print(f"  [{_fmt(lo):>8}, {_fmt(hi):>8}) {c:6d} {b}")
    if a.json:
        json.dump(cov, open(a.json, "w"), indent=1)


def _teacher_cfg(a):
    from grown.tasks.teachers import ARSConfig
    return ARSConfig(iterations=a.iterations, directions=a.directions, top=a.top, horizon=a.train_horizon,
                     seed=a.seed)


def cmd_teach(a):
    from grown.tasks.teachers import ensure_teacher
    cfg = _teacher_cfg(a)
    for t in _tasks(a):
        t0 = time.time()
        pol = ensure_teacher(t, cfg, cache=a.cache,
                             log=(lambda r: print("   ", json.dumps(r), flush=True)) if a.verbose else None)
        print(json.dumps(dict(task=t.name, physics=t.physics_id, cached=pol.info.get("cached"),
                              eval_return=pol.info.get("eval_return"), zero_return=pol.info.get("zero_return"),
                              seconds=round(time.time() - t0, 1))), flush=True)


def cmd_dataset(a):
    os.makedirs(a.out, exist_ok=True)
    tasks = _tasks(a)
    for i, t in enumerate(tasks):
        if hasattr(t, "has_teacher") and not t.has_teacher:
            from grown.tasks.teachers import ensure_teacher
            ensure_teacher(t, _teacher_cfg(a), cache=a.cache, train=a.train_missing)
        t0 = time.time()
        ds = TeacherDataset.from_rollouts(t, keys=a.seed * 1_000_003 + i, n_episodes=a.episodes)
        stem = ds.save(os.path.join(a.out, f"{t.name}"))
        print(json.dumps(dict(task=t.name, path=stem, steps=len(ds), episodes=ds.n_episodes,
                              act_mean=np.round(ds.act.mean(0), 3).tolist(), seconds=round(time.time() - t0, 1))),
              flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("sample", cmd_sample), ("coverage", cmd_coverage), ("teach", cmd_teach),
                     ("dataset", cmd_dataset)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--family", default="bodies" if name == "teach" else "signal",
                       choices=["signal", "bodies", "named_bodies", "sensorimotor"])
        s.add_argument("--n", type=int, default=20 if name != "coverage" else 2000)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--split", default="train", choices=["train", "interp", "extrap", "nominal"])
        s.add_argument("--kinds", default=None, help="signal kinds, comma separated")
        s.add_argument("--catalog", default=None, help="signal catalog names, comma separated, or 'all'")
        s.add_argument("--horizon", type=int, default=None, help="episode length for body families")
        s.add_argument("--verbose", "-v", action="store_true")
        if name == "coverage":
            s.add_argument("--json", default=None)
        if name in ("teach", "dataset"):
            s.add_argument("--cache", default=None, help="teacher cache dir (default $GROWN_TEACHER_CACHE)")
            s.add_argument("--iterations", type=int, default=100)
            s.add_argument("--directions", type=int, default=16)
            s.add_argument("--top", type=int, default=8)
            s.add_argument("--train-horizon", type=int, default=None)
        if name == "dataset":
            s.add_argument("--episodes", type=int, default=4)
            s.add_argument("--out", required=True)
            s.add_argument("--train-missing", action="store_true")
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
