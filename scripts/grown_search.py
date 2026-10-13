"""Teacher-matched quality-diversity circuit search from a config file (see grown.config for the schema).

    .venv/bin/python scripts/grown_search.py --config configs/grown/signal_small.yaml --seed 0 \
        --out runs/grown/signal_small_s0 [--workers 4] [--generations 50] [--set search.batch=32]

    # continue a run (or a compressed copy of its archives) for --generations MORE generations, optionally with a
    # library: its modules join the vocabulary (INSERT_MODULE) and module usage is tracked (ModuleTracker)
    .venv/bin/python scripts/grown_search.py --config configs/grown/exp/e3_b1e6.yaml --seed 0 \
        --resume runs/e5/compress_param/s0 --library runs/e5/compress_param/library.json \
        --generations 1563 --out runs/e5/wake2_lib_s0

Writes <out>/config.json (resolved config), tasks.json (training and held-out task records), archives/<task>.npz/json
and <task>.state.json (every archive.save_every generations and at the end), history/<task>.json, edits/<task>/
(edit log; tracker.json with a library), summary.json, and with --resume resume.json / library.json. One status line
per generation: mean archive coverage, summed QD score, mean best fitness, mean best imitation error, tasks solved,
surviving lineages (summed over tasks), edits per second, mean blend weight. A run is reproducible from (config,
seed), and a run resumed from its own saved state is bit-identical to an uninterrupted one. --config defaults to the
resumed run's config.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from grown import config as C  # noqa: E402


def parse_set(items):
    """--set a.b=value (value parsed as YAML/JSON scalar or flow collection)."""
    out: dict = {}
    for it in items or []:
        k, v = it.split("=", 1)
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = C._flow(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="config file (default with --resume: <resume>/config.json)")
    ap.add_argument("--resume", default=None, help="run directory whose archives/ to continue from")
    ap.add_argument("--library", default=None, help="library json whose active modules join the vocabulary")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="run directory (default runs/grown/<name>_s<seed>)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], help="override a config key, e.g. search.batch=32")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--checkpoint-resume", action="store_true",
                    help="if --out holds checkpoints of every task, continue from the newest shared one up to "
                         "search.generations in total (bit-identical to an unstopped run); else start fresh")
    a = ap.parse_args()

    over = parse_set(a.set)
    if a.seed is not None:
        over["seed"] = a.seed
    if a.generations is not None:
        over.setdefault("search", {})["generations"] = a.generations
    if a.config is None:
        if not a.resume:
            ap.error("--config is required unless --resume is given")
        a.config = os.path.join(a.resume, "config.json")
    cfg = C.load(a.config, over)
    out = a.out or os.path.join("runs", "grown", f"{cfg['name']}_s{cfg['seed']}" + ("_resumed" if a.resume else ""))
    if a.resume and os.path.abspath(out) == os.path.abspath(a.resume):
        ap.error("--out must differ from --resume (the resumed run directory is read, not written)")

    import jax
    jax.config.update("jax_platform_name", "cpu")
    from grown import search
    s = search.run(cfg, out, workers=a.workers, verbose=not a.quiet, resume=a.resume, library=a.library,
                   checkpoint_resume=a.checkpoint_resume)
    print(f"\n{s['solved']}/{s['n_tasks']} solved (imitation error <= {cfg['solved']}), {s['edits']} edits in "
          f"{s['loop_seconds']:.1f} s ({s['edits_per_s']:.0f} edits/s), wall {s['wall_seconds']:.1f} s")
    print(f"{'task':28s} {'kind':12s} {'best err':>9s} {'trivial':>8s} {'progress':>8s} {'return':>7s} "
          f"{'nodes':>5s} {'filled':>6s}")
    for t in s["tasks"]:
        print(f"{t['task'][:28]:28s} {str(t['kind'])[:12]:12s} {t['best_error']:9.4f} {t['trivial']:8.4f} "
              f"{t['progress']:8.3f} {t['best_return']:7.3f} {t['best_nodes']:5d} {t['filled']:6d}"
              f"{'  solved' if t['solved'] else ''}")
    print(f"\nrun directory: {out}")
    if not a.quiet:
        print(json.dumps({k: s[k] for k in ("solved", "n_tasks", "edits", "edits_per_s")}))


if __name__ == "__main__":
    main()
