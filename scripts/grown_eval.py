"""Evaluate a grown_search run: elites on fresh episodes of their training tasks, zero-shot on held-out tasks
(interp / extrap splits or a held-out demand region), coverage, and a table by demand-tag quartile.

    .venv/bin/python scripts/grown_eval.py --run runs/grown/signal_family_s0 [--episodes 4] [--top 16]
    .venv/bin/python scripts/grown_eval.py --archive runs/grown/signal_small_s0/archives/xor2

Writes <run>/eval.json, or <archive prefix>.eval.json for a single archive. See grown.evaluation.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def _f(x, w=7):
    return f"{x:{w}.3f}" if isinstance(x, (int, float)) and x == x else f"{'n/a':>{w}s}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run directory written by grown_search.py")
    ap.add_argument("--archive", default=None, help="one archive prefix (<run>/archives/<task>)")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--top", type=int, default=16, help="elites per archive tried on each held-out task")
    ap.add_argument("--splits", default=None, help="held-out splits to evaluate, comma separated (default all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None, help="a checkpoint generation or 'latest': evaluate a running (or "
                    "stopped) search from its checkpoints; writes <run>/eval_ckpt_g<generation>.json")
    ap.add_argument("--backend", default=None, help="override the run's substrate backend (e.g. cpu, to leave the "
                    "GPU to a running search); lane placement is turned off with it")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if not a.run and not a.archive:
        ap.error("--run or --archive is required")
    if a.archive and a.archive.endswith((".npz", ".json")):
        a.archive = a.archive.rsplit(".", 1)[0]
    run = a.run or os.path.dirname(os.path.dirname(os.path.abspath(a.archive)))

    import jax
    jax.config.update("jax_platform_name", "cpu")
    from grown.evaluation import evaluate_run
    sub = {"backend": a.backend} if a.backend else None
    res = evaluate_run(run, a.archive, a.episodes, a.horizon, a.top, a.splits.split(",") if a.splits else None,
                       a.out, verbose=not a.quiet, checkpoint=a.checkpoint, substrate=sub)
    print()
    for g, d in res["groups"].items():
        c = d["coverage"]
        print(f"{g:18s} tasks {c['n']:3d}  compatible {c['compatible']:3d}  solved {c['solved']:3d}  "
              f"improved {c['improved']:3d}  mean best error {_f(c['mean_error'])}")
    for g, d in res["groups"].items():
        if not d["by_tag"]:
            continue
        print(f"\n{g}: mean best error / mean progress / solved by demand-tag quartile")
        for ax, table in d["by_tag"].items():
            cells = "  ".join(f"[{t['lo']:.3g}, {t['hi']:.3g}] n={t['n']} err={t['mean_error']:.3f} "
                              f"prog={_f(t['mean_progress'], 5).strip()} solved={t['solved']}" for t in table)
            print(f"  {ax:16s} {cells}")
    print(f"\nwrote {res['path']}")


if __name__ == "__main__":
    main()
