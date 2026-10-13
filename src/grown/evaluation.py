"""Evaluation of a search run's archives on fresh episodes and on held-out tasks.

  train      per training task: every elite of its archive re-evaluated on fresh episodes (keys never used in
             training); best imitation error, the trivial constant-output error on the same episodes, progress
             = 1 - best / trivial, solved (best <= cfg["solved"]), archive coverage and QD score; the best fresh
             elite's cell, size (top-level tokens, flattened gates), cycle_frac, live_frac and depth
  held out   per held-out task (Family.splits interp / extrap, a demand-box complement, or catalog names):
             zero-shot transfer. Candidates are the top `top` elites (by stored imitation error) of every training
             archive whose wire interface matches the task's (same n_in and n_out); the best candidate's error on
             fresh episodes is the task's error. Tasks with no compatible archive are counted, not scored.
  coverage   per group: fraction of tasks with a compatible archive, solved, improved (progress > 0.1)
  by_tag     per group and demand axis: tasks binned by quartiles of the tag over the group; n, tag range, mean best
             error, mean progress, solved count per bin

Results go to <run>/eval.json (run mode) or <archive prefix>.eval.json (single-archive mode).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .archive.descriptors import structural
from .archive.map_elites import Archive
from .circuit.library import Library
from .search import (_P_EVAL, TaskData, derive_seed, evaluate, make_substrate, rebuild_task, task_seed_word,
                     trivial_error)
from .tasks import AXES, TeacherDataset
from .tasks.base import jsonable


def _fresh(task, cfg, episodes, horizon):
    seed = int(cfg.get("seed", 0))
    ds = TeacherDataset.from_rollouts(task, keys=derive_seed(seed, task_seed_word(task), _P_EVAL),
                                      n_episodes=episodes, horizon=horizon)
    return TaskData(task, ds)


def _best(substrate, circuits, data, library, metric, chunk=256):
    errs = []
    for i in range(0, len(circuits), chunk):
        errs.append(evaluate(substrate, circuits[i:i + chunk], data, library, metric).imitation)
    e = np.concatenate(errs)
    j = int(np.argmin(e))
    return float(e[j]), j, e


def _row(rec_task, best, trivial, thr):
    return dict(task=rec_task.name, kind=rec_task.tags.extra.get("kind"), uid=rec_task.uid, best_error=best,
                trivial=trivial, progress=(1.0 - best / trivial) if trivial > 0 and best == best else float("nan"),
                solved=bool(best <= thr), tags=rec_task.tags.to_dict())


def by_tag(rows: list, q=(0.0, 0.25, 0.5, 0.75, 1.0)) -> dict:
    """Per demand axis: rows binned by quantiles of the tag value over these rows."""
    rows = [r for r in rows if r.get("best_error") == r.get("best_error") and r.get("best_error") is not None]
    out = {}
    if not rows:
        return out
    for ax in AXES:
        v = np.array([float(r["tags"][ax]) for r in rows])
        edges = np.unique(np.quantile(v, q))
        if len(edges) == 1:
            bins = np.zeros(len(v), int)
            edges = np.array([edges[0], edges[0]])
        else:
            bins = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
        table = []
        for b in range(len(edges) - 1):
            m = bins == b
            if not m.any():
                continue
            e = np.array([rows[i]["best_error"] for i in np.flatnonzero(m)])
            p = np.array([rows[i]["progress"] for i in np.flatnonzero(m)], float)
            table.append(dict(bin=b, lo=float(v[m].min()), hi=float(v[m].max()), n=int(m.sum()),
                              mean_error=float(e.mean()), mean_progress=float(np.nanmean(p)) if np.isfinite(p).any()
                              else float("nan"), solved=int(sum(rows[i]["solved"] for i in np.flatnonzero(m)))))
        out[ax] = table
    return out


def _coverage(rows):
    n = len(rows)
    scored = [r for r in rows if r.get("best_error") is not None and r["best_error"] == r["best_error"]]
    return dict(n=n, compatible=len(scored), solved=sum(r["solved"] for r in scored),
                improved=sum((r["progress"] or 0) > 0.1 for r in scored),
                frac_compatible=len(scored) / n if n else float("nan"),
                frac_solved=sum(r["solved"] for r in scored) / n if n else float("nan"),
                mean_error=float(np.mean([r["best_error"] for r in scored])) if scored else float("nan"))


def evaluate_run(run: str, archive: str | None = None, episodes: int | None = None, horizon: int | None = None,
                 top: int = 16, splits=None, out: str | None = None, verbose: bool = True, checkpoint=None,
                 substrate: dict | None = None) -> dict:
    """Evaluate a run directory (all archives) or one archive prefix inside it. Returns and saves the results.

    checkpoint: evaluate the archives of a checkpoint (search.TaskSearch.save) instead of the run's current files:
    a generation, or "latest" (the newest generation every task has); safe while the search is running (checkpoints
    are written once and renamed into place). Default output <run>/eval_ckpt_g<generation>.json. substrate: config
    overrides of the run's substrate section (e.g. {"backend": "cpu", "placement": "none"} to stay off the GPU a
    running search uses)."""
    with open(os.path.join(run, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run, "tasks.json")) as f:
        recs = json.load(f)
    dc = cfg["data"]
    episodes = int(episodes or dc.get("eval_episodes", 4))
    horizon = int(horizon or dc.get("eval_horizon", dc.get("horizon", 256)))
    metric = dc.get("metric", "abs")
    thr = float(cfg.get("solved", 0.01))
    sub_cfg = dict(cfg.get("substrate", {}), **(substrate or {}))
    if substrate and sub_cfg.get("name") == "freerun" and "placement" not in substrate:
        sub_cfg["placement"] = "none"                        # an overridden backend: no CPU/GPU split
    substrate = make_substrate(sub_cfg)
    lib_path = os.path.join(run, "library.json")           # a run resumed with a library (module instances)
    library = Library.load(lib_path) if os.path.exists(lib_path) else Library()
    train = [(r, rebuild_task(r)) for r in recs if r["role"] == "train"]
    if archive is not None:
        name = os.path.basename(archive)
        train = [(r, t) for r, t in train if r["name"] == name]
    if checkpoint is not None:
        from .search import checkpoint_dir, common_checkpoint
        if checkpoint == "latest":
            checkpoint = common_checkpoint(run, [r["name"] for r, _ in train])
            if checkpoint is None:
                raise FileNotFoundError(f"no checkpoint shared by every task under {run}/checkpoints")
        checkpoint = int(checkpoint)
    archives = {}
    for r, t in train:
        prefix = (os.path.join(checkpoint_dir(run, r["name"], checkpoint), r["name"]) if checkpoint is not None
                  else os.path.join(run, "archives", r["name"]))
        if os.path.exists(prefix + ".npz"):
            archives[r["name"]] = (t, Archive.load(prefix, library))
    res: dict = dict(run=os.path.abspath(run), episodes=episodes, horizon=horizon, metric=metric, solved_threshold=thr,
                     top=top, groups={}, checkpoint=checkpoint)
    # training tasks on fresh episodes
    rows = []
    for name, (t, a) in archives.items():
        cells = a.filled_cells()
        data = _fresh(t, cfg, episodes, horizon)
        best, j, _ = _best(substrate, [a.items[c] for c in cells], data, library, metric)
        row = _row(t, best, trivial_error(data.ds, metric), thr)
        row.update(stored_best_error=float(a.sig[cells, 0].min()), coverage=a.coverage(), qd_score=a.qd_score(),
                   filled=int(len(cells)), best_nodes=len(a.items[cells[j]].nodes), best_cell=int(cells[j]))
        st = structural(a.items[cells[j]], ("n_gates", "cycle_frac", "live_frac", "depth"), library)
        row.update(best_gates=int(st[0]), best_cycle_frac=float(st[1]), best_live_frac=float(st[2]),
                   best_depth=int(st[3]))
        rows.append(row)
        if verbose:
            print(f"train  {name:28s} fresh {best:.4f}  stored {row['stored_best_error']:.4f}  trivial "
                  f"{row['trivial']:.4f}{'  solved' if row['solved'] else ''}", flush=True)
    res["groups"]["train"] = dict(rows=rows, coverage=_coverage(rows), by_tag=by_tag(rows))
    # held-out tasks: zero-shot transfer of compatible elites
    held = [r for r in recs if r["role"] != "train" and (splits is None or r["role"] in splits)]
    pools = {}
    for name, (t, a) in archives.items():
        io = t.io_spec()
        cells = a.filled_cells()
        cells = cells[np.lexsort((cells, a.sig[cells, 0]))][:top]
        pools.setdefault((io.n_in, io.n_out), []).extend((name, a.items[c]) for c in cells)
    for split in sorted({r["role"] for r in held}):
        rows = []
        for r in [r for r in held if r["role"] == split]:
            t = rebuild_task(r)
            io = t.io_spec()
            cand = pools.get((io.n_in, io.n_out), [])
            data = _fresh(t, cfg, episodes, horizon)
            triv = trivial_error(data.ds, metric)
            if cand:
                best, j, _ = _best(substrate, [c for _, c in cand], data, library, metric)
                row = _row(t, best, triv, thr)
                row.update(source=cand[j][0], candidates=len(cand))
            else:
                row = _row(t, float("nan"), triv, thr)
                row.update(best_error=None, progress=None, solved=False, source=None, candidates=0)
            rows.append(row)
            if verbose:
                be = "   n/a" if row["best_error"] is None else f"{row['best_error']:.4f}"
                print(f"{split:6s} {t.name:28s} best {be}  trivial {triv:.4f}  candidates {row['candidates']:4d}"
                      f"{'  from ' + row['source'] if row['source'] else ''}", flush=True)
        res["groups"][split] = dict(rows=rows, coverage=_coverage(rows), by_tag=by_tag(rows))
    path = out or (archive + ".eval.json" if archive else
                   os.path.join(run, f"eval_ckpt_g{checkpoint}.json" if checkpoint is not None else "eval.json"))
    with open(path, "w") as f:
        json.dump(jsonable(res), f, indent=1)
    res["path"] = path
    return res
