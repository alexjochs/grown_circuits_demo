"""Learned editor: generate hindsight logs on the toy target, train, and compare against the uniform operator.

Toy target and loop settings are those of scripts/grown_loop.py (out0 = in0 XOR in1, out1 = in0 one step ago; sync
stand-in; archive n_gates x cycle_frac x toggle_rate; operator max_nodes 64). The target stream is fixed
(--target-seed) and loop seeds vary, so training logs and evaluation runs share the task but not the trajectory.

    # 1. hindsight logs: uniform operator, one directory per loop seed (edits/, circuits.npz, library.json, ...)
    .venv/bin/python scripts/grown_editor.py gen-log --seeds 1-8 --generations 200 --out runs/editor/logs
    # 2. train (optionally with dream pretraining of the encoder first)
    .venv/bin/python scripts/grown_editor.py train --logs runs/editor/logs/seed_* --out runs/editor/editor.npz
    .venv/bin/python scripts/grown_editor.py train --logs ... --dream 20000 --out runs/editor/editor_dream.npz
    # 3. uniform vs learned at several eps, same seeds ("eval" is an alias of "compare")
    .venv/bin/python scripts/grown_editor.py compare --ckpt runs/editor/editor.npz --seeds 100-104 --eps 0.1 0.3 0.5
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grown_loop import ToyTarget  # noqa: E402

from grown import loop  # noqa: E402
from grown.archive import Axis  # noqa: E402
from grown.circuit import Fate, Library, OperatorConfig, Source, UniformOperator  # noqa: E402
from grown.editor import EditorConfig, LearnedProposal, RecordingArchive, init_params, load_params, save_params  # noqa: E402
from grown.editor import dream as D  # noqa: E402
from grown.editor import train as TR  # noqa: E402
from grown.editor.model import n_params  # noqa: E402
from grown.log import EditLog  # noqa: E402
from grown.sim import SyncSubstrate  # noqa: E402

MAX_NODES = 64


def seeds_arg(s: str) -> list:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def op_config() -> OperatorConfig:
    return OperatorConfig(max_nodes=MAX_NODES)


def run_toy(seed, generations, batch, target, out=None, proposal_factory=None, eps=0.1, every=0, label="",
            op_weights=None):
    """One smoke-loop run on the toy target. Returns (history, archive, proposal stats per generation)."""
    library = Library()
    archive = RecordingArchive([Axis("n_gates", 0, 64, 16), Axis("cycle_frac", 0, 1, 8),
                                Axis("toggle_rate", 0, 0.5, 8, source="behavior")],
                               sig_dim=loop.SIG_DIM, qd_offset=-1.0, library=library)
    cfg_op = op_config()
    if op_weights:
        cfg_op.weights = dict(op_weights)
    op = UniformOperator(cfg_op, library)
    log = EditLog(os.path.join(out, "edits"), sig_dim=loop.SIG_DIM) if out else None
    prop = proposal_factory(op, archive) if proposal_factory else None
    cfg = loop.LoopConfig(generations=generations, batch=batch, K=4, sampling="curiosity", seed=seed, eps=eps)
    pstats = []
    t0 = time.time()

    def cb(st):
        if prop is not None:
            ps = prop.pop_stats()
            pstats.append(ps)
            prop.prepare([archive.items[c] for c in archive.filled_cells()])
        g = st["generation"]
        if every and (g % every == 0 or g == generations - 1):
            extra = f"  H_mc {pstats[-1].get('H_mc', float('nan')):.2f}" if prop is not None else ""
            print(f"{label} gen {g:4d} cov {st['coverage']:.4f} qd {st['qd_score']:8.2f} best {st['max_fitness']:+.4f} "
                  f"acc {st['accepted']:3d}/{batch}{extra} {time.time() - t0:6.1f}s", flush=True)

    hist = loop.run(SyncSubstrate(), target, archive, log, target.inputs, op, cfg, proposal=prop, library=library,
                    callback=cb)
    if out:
        log.close()
        archive.store.save(os.path.join(out, "circuits.npz"))
        archive.save(os.path.join(out, "archive"))
        library.save(os.path.join(out, "library.json"))
        with open(os.path.join(out, "history.json"), "w") as f:
            json.dump(hist, f)
    return hist, archive, pstats, log


def cmd_gen_log(a):
    target = ToyTarget(a.steps, a.target_seed)
    total = 0
    for s in seeds_arg(a.seeds):
        out = os.path.join(a.out, f"seed_{s}")
        if os.path.exists(os.path.join(out, "circuits.npz")) and not a.force:
            print(f"{out} exists, skipping")
            continue
        hist, arch, _, log = run_toy(s, a.generations, a.batch, target, out, every=a.every, label=f"seed {s}")
        total += len(log)
        print(f"seed {s}: {len(log)} edits, store {len(arch.store)} circuits, final coverage "
              f"{hist[-1]['coverage']:.4f} qd {hist[-1]['qd_score']:.2f}", flush=True)
    print(f"total {total} edits")


def cmd_train(a):
    logs = sorted(set(sum((glob.glob(x) for x in a.logs), [])))
    print(f"logs: {logs}")
    ds, op, counts = TR.build(logs, op_config(), sig_dim=loop.SIG_DIM, verify=not a.no_verify, max_rows=a.max_rows)
    tr, te = TR.split(ds, a.holdout, a.seed)
    print(f"train {len(tr)} ({int((tr.weight > 0).sum())} pos)  held-out {len(te)} ({int((te.weight > 0).sum())} pos)")
    n_in, n_out = 2, 2
    cfg = EditorConfig(d=a.d, layers=a.layers, heads=a.heads, sig_dim=loop.SIG_DIM,
                       dream_dim=D.dream_dim(n_in, n_out) if a.dream else 0, seed=a.seed)
    p = init_params(cfg)
    print(f"model: {n_params(p)} parameters, d={cfg.d}, layers={cfg.layers}")
    report = {"logs": logs, "counts": counts, "model": cfg.to_dict(), "n_params": n_params(p),
              "n_train": len(tr), "n_heldout": len(te)}
    if a.dream:
        t0 = time.time()
        circs, sigs, _ = D.dream_pairs(op, a.dream, n_in, n_out, seed=a.seed, steps=a.steps)
        print(f"dreams: {len(circs)} circuits in {time.time() - t0:.0f}s, signature dim {sigs.shape[1]}")
        p, dh = D.pretrain(p, cfg, op, circs, sigs, steps=a.dream_steps, seed=a.seed)
        report["dream"] = {"n": len(circs), "history": dh}
    tc = TR.TrainConfig(steps=a.train_steps, lr=a.lr, neg_weight=a.neg_weight, holdout=a.holdout, seed=a.seed,
                        eval_every=a.eval_every)
    p, hist = TR.train(p, cfg, tr, te, tc, ckpt=a.out)
    ent = TR.proposal_entropy(p, cfg, te, n=200, seed=a.seed)
    ent0 = TR.proposal_entropy(init_params(cfg), cfg, te, n=200, seed=a.seed)
    print("held-out proposal entropy (trained):", json.dumps({k: v for k, v in ent.items() if not isinstance(v, dict)}))
    print("held-out proposal entropy (init):   ", json.dumps({k: v for k, v in ent0.items() if not isinstance(v, dict)}))
    report.update(train=vars(tc) if not hasattr(tc, "__dataclass_fields__") else tc.__dict__, history=hist,
                  entropy=ent, entropy_init=ent0)
    save_params(a.out, p, cfg, {"report": "see .json"})
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(report, f, indent=1, default=_jsonable)
    sel = hist[-1].get("selected_step", None) if hist else None
    last = next((h for h in hist if "n" in h and h.get("step") == sel), hist[-1] if hist else {})
    print(f"held-out: pos loglik {hist[0].get('pos_loglik', float('nan')):.3f} -> {last.get('pos_loglik', float('nan')):.3f}"
          f" nats/move; coarse lift pos {last.get('pos_coarse_lift', float('nan')):+.3f} "
          f"neg {last.get('neg_coarse_lift', float('nan')):+.3f}")


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _accept_by_source(log) -> dict:
    out = {}
    rows = [b for b in log.scan(columns=["source", "fate"])]
    if not rows:
        return out
    src = np.concatenate([b["source"] for b in rows])
    fate = np.concatenate([b["fate"] for b in rows])
    acc = (fate == Fate.ELITE_NEW) | (fate == Fate.ELITE_REPLACE)
    for s in Source:
        m = src == s
        if m.any():
            out[s.name] = {"n": int(m.sum()), "frac": float(m.mean()), "accept": float(acc[m].mean())}
    return out


def type_prior(log_dirs) -> dict:
    """Per-type acceptance rate over logs of uniform runs: the type-level optimum of the hindsight objective
    (q*(t) proportional to u(t) P(accept | t), with u uniform over types), as operator weights."""
    from grown.circuit import MoveType
    acc, tot = {}, {}
    for d in log_dirs:
        for b in EditLog(os.path.join(d, "edits")).scan(columns=["move", "fate"]):
            ok = (b["fate"] == Fate.ELITE_NEW) | (b["fate"] == Fate.ELITE_REPLACE)
            for t in np.unique(b["move"]):
                if t < 0:
                    continue
                m = b["move"] == t
                acc[int(t)] = acc.get(int(t), 0) + int(ok[m].sum())
                tot[int(t)] = tot.get(int(t), 0) + int(m.sum())
    w = {MoveType(t): max(acc[t] / tot[t], 1e-3) for t in tot}
    for t in MoveType:
        w.setdefault(t, 1e-3)
    return w


def cmd_eval(a):
    target = ToyTarget(a.steps, a.target_seed)
    p, cfg, _ = load_params(a.ckpt)
    results = []
    arms = [("uniform", None)] + [(f"learned eps={e}", e) for e in a.eps]
    type_w = None
    if a.type_prior:
        type_w = type_prior(a.type_prior)
        print("type prior (per-type acceptance in", a.type_prior, "):", {k.name: round(v, 4) for k, v in type_w.items()})
        arms.insert(1, ("uniform+typeprior", None))
    for s in seeds_arg(a.seeds):
        for name, eps in arms:
            out = os.path.join(a.out, f"seed_{s}", name.replace(" ", "_").replace("=", ""))
            factory = None
            if eps is not None:
                def factory(op, archive):
                    return LearnedProposal(p, cfg, op, temperature=a.temperature, archive=archive, cache_size=a.cache)
            t0 = time.time()
            hist, arch, pst, log = run_toy(s, a.generations, a.batch, target, out, factory, eps if eps is not None
                                           else 0.1, every=a.every, label=f"[{name} s{s}]",
                                           op_weights=type_w if name == "uniform+typeprior" else None)
            r = {"arm": name, "eps": eps, "seed": s, "seconds": time.time() - t0,
                 "accept_rate": [h["accepted"] / a.batch for h in hist], "coverage": [h["coverage"] for h in hist],
                 "qd": [h["qd_score"] for h in hist], "best": [h["max_fitness"] for h in hist],
                 "by_source": _accept_by_source(log), "proposal": pst}
            results.append(r)
            print(f"[{name} s{s}] final cov {r['coverage'][-1]:.4f} qd {r['qd'][-1]:.2f} best {r['best'][-1]:+.4f} "
                  f"mean accept {np.mean(r['accept_rate']):.4f}  {r['seconds']:.0f}s  by source "
                  f"{json.dumps(r['by_source'])}", flush=True)
            with open(os.path.join(a.out, "eval.json"), "w") as f:
                json.dump({"ckpt": a.ckpt, "args": vars(a), "results": results}, f, default=_jsonable)
    summarize(results, a.generations)


def summarize(results, G):
    print("\narm                 seeds  accept(mean)   coverage(final)   qd(final)        best(final)    "
          "H_mc(first->last)  KL_coarse  top10")
    for arm in dict.fromkeys(r["arm"] for r in results):
        rs = [r for r in results if r["arm"] == arm]
        acc = [np.mean(r["accept_rate"]) for r in rs]
        cov = [r["coverage"][-1] for r in rs]
        qd = [r["qd"][-1] for r in rs]
        best = [r["best"][-1] for r in rs]
        line = (f"{arm:18s}  {len(rs):5d}  {np.mean(acc):.4f}±{np.std(acc):.4f}  {np.mean(cov):.4f}±{np.std(cov):.4f}  "
                f"{np.mean(qd):7.2f}±{np.std(qd):5.2f}  {np.mean(best):+.4f}±{np.std(best):.4f}")
        if rs[0]["proposal"]:
            h0 = np.mean([r["proposal"][0].get("H_mc", np.nan) for r in rs])
            h1 = np.mean([r["proposal"][-1].get("H_mc", np.nan) for r in rs])
            kl = np.nanmean([np.nanmean([q.get("KL_coarse", np.nan) for q in r["proposal"]]) for r in rs])
            t10 = np.nanmean([np.nanmean([q.get("top10", np.nan) for q in r["proposal"]]) for r in rs])
            line += f"  {h0:6.2f} -> {h1:6.2f}     {kl:6.3f}   {t10:.3f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen-log", "train", "compare"):
        s = sub.add_parser(name, aliases=["eval"] if name == "compare" else [])
        s.add_argument("--steps", type=int, default=64, help="toy target length (control steps)")
        s.add_argument("--target-seed", type=int, default=0)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--every", type=int, default=50)
    g = sub.choices["gen-log"]
    g.add_argument("--seeds", default="1-8")
    g.add_argument("--generations", type=int, default=200)
    g.add_argument("--batch", type=int, default=64)
    g.add_argument("--out", required=True)
    g.add_argument("--force", action="store_true")
    t = sub.choices["train"]
    t.add_argument("--logs", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--d", type=int, default=64)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--train-steps", type=int, default=3000)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--neg-weight", type=float, default=0.1)
    t.add_argument("--holdout", type=float, default=0.125)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--dream", type=int, default=0, help="number of dream pairs for inverse pretraining (0: none)")
    t.add_argument("--dream-steps", type=int, default=1500)
    t.add_argument("--max-rows", type=int, default=0)
    t.add_argument("--no-verify", action="store_true")
    e = sub.choices["compare"]
    e.add_argument("--ckpt", required=True)
    e.add_argument("--seeds", default="100-104")
    e.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    e.add_argument("--temperature", type=float, default=1.0)
    e.add_argument("--type-prior", nargs="*", default=[], help="log dirs: add a uniform arm with per-type "
                   "acceptance-rate weights (static type prior control)")
    e.add_argument("--cache", type=int, default=1536, help="per-circuit table cache (archive elites)")
    e.add_argument("--generations", type=int, default=200)
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    if getattr(a, "max_rows", 0) == 0 and hasattr(a, "max_rows"):
        a.max_rows = None
    {"gen-log": cmd_gen_log, "train": cmd_train, "compare": cmd_eval, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
