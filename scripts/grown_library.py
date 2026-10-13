"""Library learning: wake / compress / dream cycles, the canonical-cell report, and a mining dry run.

    # several cycles on toy targets (XOR plus one step of delay, one archive per input-stream seed)
    .venv/bin/python scripts/grown_library.py run --out runs/grown_lib --cycles 4 --wake-generations 50

    # the canonical-cell table (research question 1) for a saved library, usage-weighted by archive elites
    .venv/bin/python scripts/grown_library.py report --library runs/grown_lib/library.json --archives runs/grown_lib

    # what compression would admit from saved archives (a grown_loop / grown_search run dir, a program dir, or
    # archive prefixes), without changing anything
    .venv/bin/python scripts/grown_library.py mine --archives runs/grown/x --min-lineages 2

    # one compress phase over saved archives, written to a new directory (archives, library, events, cell table)
    .venv/bin/python scripts/grown_library.py compress --archives runs/grown/x --out runs/grown/x_lib
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from grown.archive import Archive  # noqa: E402
from grown.archive.mining import mine  # noqa: E402
from grown.circuit import Library, flatten  # noqa: E402
from grown.library import CompressConfig, Program, ProgramConfig, Task, UsageConfig  # noqa: E402
from grown.library.canon_cells import classify_netlist, library_report, write_report  # noqa: E402
from grown.library.compress import _index, check_module, compress_phase, description_length  # noqa: E402


class ToyTarget:
    """Two random input bit streams; out0 = in0 XOR in1, out1 = in0 one step ago (as scripts/grown_loop.py)."""
    n_out = 2

    def __init__(self, steps: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.inputs = rng.integers(0, 2, (steps, 2)).astype(np.uint8)
        x = self.inputs
        prev = np.concatenate([[0], x[:-1, 0]])
        self.target = np.stack([x[:, 0] ^ x[:, 1], prev], 1).astype(np.uint8)

    def __call__(self, res):
        return (res.outputs != self.target[None]).mean(2)


def archive_prefixes(paths) -> list:
    out = []
    for p in paths:
        if os.path.isdir(p):
            for sub in ("archives", "ckpt", ""):
                found = sorted(glob.glob(os.path.join(p, sub, "*.json")))
                found = [f[:-5] for f in found if os.path.exists(f[:-5] + ".npz")
                         and not f.endswith((".data.json", ".state.json"))]      # search state, DAgger data
                if found:
                    out += found
                    break
        else:
            out.append(os.path.splitext(p)[0] if p.endswith((".json", ".npz")) else p)
    return out


def load_library(path, archives_paths=()) -> Library:
    if path:
        return Library.load(path)
    for p in archives_paths:
        for cand in (os.path.join(p, "library.json"), os.path.join(p, "ckpt", "library.json")):
            if os.path.isdir(p) and os.path.exists(cand):
                return Library.load(cand)
    return Library()


def cmd_run(a):
    tasks = []
    for i in range(a.tasks):
        t = ToyTarget(a.steps, a.seed + i)
        tasks.append(Task(f"toy{i}", t, t.inputs))
    cfg = ProgramConfig(cycles=a.cycles, wake_generations=a.wake_generations, batch=a.batch, K=a.K,
                        sampling=a.sampling, max_nodes=a.max_nodes, seed=a.seed,
                        compress=CompressConfig(k_max=a.k_max, min_lineages=a.min_lineages, threshold=a.threshold,
                                                max_admissions=a.max_admissions, max_rounds=a.max_rounds),
                        usage=UsageConfig(window=a.window), freerun=a.freerun)
    t0 = time.time()
    p = Program(a.out, tasks, cfg, log=lambda s: print(s, flush=True))
    p.run()
    rep = library_report(p.library, p.archives, freerun=a.freerun, cache=p.cell_cache)
    write_report(rep, os.path.join(a.out, "cells"), f"Canonical cells: {a.out}")
    print(f"\n{len(p.library)} modules admitted, {len(p.library.retired)} retired, "
          f"{rep['matched']}/{rep['modules']} active modules match a human cell; report in {a.out}/cells.md "
          f"({time.time() - t0:.1f} s)")


def cmd_report(a):
    lib = load_library(a.library, a.archives)
    archives = [Archive.load(p, lib) for p in archive_prefixes(a.archives)]
    mods = list(lib.defs) if a.all else None
    t0 = time.time()
    rep = library_report(lib, archives, modules=mods, freerun=a.freerun)
    out = a.out or os.path.splitext(a.library or "cells")[0] + "_cells"
    write_report(rep, out, a.title or f"Canonical cells: {a.library}")
    with open(out + ".md") as f:
        print(f.read())
    print(f"({time.time() - t0:.1f} s; written to {out}.md / .json)")


def cmd_mine(a):
    lib = load_library(a.library, a.archives)
    prefixes = archive_prefixes(a.archives)
    archives = [Archive.load(p, lib) for p in prefixes]
    circuits, refs, lineages = _index(archives)
    print(f"{len(archives)} archives, {len(circuits)} elites, {len(set(lineages))} lineages, "
          f"DL {description_length(archives, lib)} tokens, library {len(lib)} modules")
    t0 = time.time()
    stats: dict = {}
    cands = mine(circuits, lineages, k_max=a.k_max, k_min=a.k_min, min_lineages=a.min_lineages, top=a.top,
                 min_saving=1, mode=a.mode, dl_params=a.dl_params, library=lib, stats=stats, workers=a.workers)
    print(f"mined {len(cands)} candidates in {time.time() - t0:.1f} s ({a.mode}): {json.dumps(stats)}\n")
    rows = []
    print(f"{'#':>3} {'save':>6} {'occ':>5} {'lin':>4} {'arch':>4} {'circ':>5} {'size':>4} {'in/out':>6}  "
          f"{'valid':<28} cell")
    for i, c in enumerate(cands):
        reason = check_module(c.body, lib)
        try:
            r = classify_netlist(flatten(c.body, lib))
            cell = f"{r['status']}: {r['cell'] or r['kind']}" + (f" ({r['fsm_states']} states)"
                                                                 if r["kind"] == "sequential" else "")
        except Exception as e:  # noqa: BLE001
            cell = f"error: {e}"
        n_arch = len({refs[ci][0] for ci in c.where})
        print(f"{i:3d} {c.saving:6d} {c.occurrences:5d} {c.lineages:4d} {n_arch:4d} {c.circuits:5d} {c.size:4d} "
              f"{c.n_in:>3}/{c.n_out:<2}  {(reason or 'ok')[:28]:<28} {cell}")
        rows.append({"hash": c.hash.hex(), "saving": c.saving, "occurrences": c.occurrences, "lineages": c.lineages,
                     "archives": n_arch,
                     "circuits": c.circuits, "size": c.size, "n_in": c.n_in, "n_out": c.n_out, "valid": reason,
                     "cell": cell})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


def out_prefix(out, p) -> str:
    """Where a compressed archive goes: archives of a run directory (<run>/archives/<task>) keep their run's name
    (<out>/<run>/archives/<task>, resumable with grown_search --resume <out>/<run>); others go to <out>/archives."""
    d, name = os.path.split(p)
    if os.path.basename(d) == "archives":
        return os.path.join(out, os.path.basename(os.path.dirname(d)), "archives", name)
    return os.path.join(out, "archives", name)


def cmd_compress(a):
    import shutil
    lib = load_library(a.library, a.archives)
    prefixes = archive_prefixes(a.archives)
    archives = [Archive.load(p, lib) for p in prefixes]
    targets = [out_prefix(a.out, p) for p in prefixes]
    if len(set(targets)) != len(targets):
        raise SystemExit("two archives map to the same output path")
    cfg = CompressConfig(k_min=a.k_min, k_max=a.k_max, min_lineages=a.min_lineages, threshold=a.threshold,
                         max_admissions=a.max_admissions, max_rounds=a.max_rounds, top=a.top, mode=a.mode,
                         dl_params=a.dl_params, verify=a.verify, workers=a.workers)
    rep = compress_phase(archives, lib, cfg, phase=a.phase, events_dir=a.out,
                         log=lambda s: print(s, flush=True))
    for p, t, arc in zip(prefixes, targets, archives):
        os.makedirs(os.path.dirname(t), exist_ok=True)
        arc.save(t)
        for ext in (".state.json", ".data.npz", ".data.json"):        # search state travels with the archive
            if os.path.exists(p + ext):
                shutil.copyfile(p + ext, t + ext)
        run = os.path.dirname(os.path.dirname(p))
        for f in ("config.json", "tasks.json", "summary.json"):        # resume reads the recorded generation
            src, dst = os.path.join(run, f), os.path.join(os.path.dirname(os.path.dirname(t)), f)
            if os.path.basename(os.path.dirname(p)) == "archives" and os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
    lib.save(os.path.join(a.out, "library.json"))
    summ = rep.summary()
    with open(os.path.join(a.out, "compress_report.json"), "w") as f:
        json.dump({**summ, "config": vars(a), "admitted": [vars(e) for e in rep.admissions]}, f, indent=1,
                  default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "mining"}))
    cells = library_report(lib, archives, freerun=a.freerun)
    write_report(cells, os.path.join(a.out, "cells"), f"Canonical cells: {', '.join(a.archives)}")
    print(f"{cells['matched']}/{cells['modules']} modules match a human cell "
          f"({cells['matched_nontrivial']} non-trivial); {a.out}/cells.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--cycles", type=int, default=4)
    r.add_argument("--wake-generations", type=int, default=50)
    r.add_argument("--tasks", type=int, default=4, help="toy targets (input-stream seeds), one archive each")
    r.add_argument("--batch", type=int, default=64)
    r.add_argument("--steps", type=int, default=64)
    r.add_argument("--K", type=int, default=4)
    r.add_argument("--max-nodes", type=int, default=64)
    r.add_argument("--sampling", choices=["uniform", "curiosity", "lineage"], default="curiosity")
    r.add_argument("--k-max", type=int, default=4)
    r.add_argument("--min-lineages", type=int, default=2)
    r.add_argument("--threshold", type=int, default=8)
    r.add_argument("--max-admissions", type=int, default=16)
    r.add_argument("--max-rounds", type=int, default=4)
    r.add_argument("--window", type=int, default=50)
    r.add_argument("--freerun", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--library", default="")
    p.add_argument("--archives", nargs="*", default=[])
    p.add_argument("--out", default="")
    p.add_argument("--title", default="")
    p.add_argument("--all", action="store_true", help="include retired modules")
    p.add_argument("--freerun", action="store_true")
    p.set_defaults(fn=cmd_report)
    m = sub.add_parser("mine")
    m.add_argument("--archives", nargs="+", required=True)
    m.add_argument("--library", default="")
    m.add_argument("--k-min", type=int, default=2)
    m.add_argument("--k-max", type=int, default=4)
    m.add_argument("--min-lineages", type=int, default=2)
    m.add_argument("--top", type=int, default=30)
    m.add_argument("--json", default="")
    m.add_argument("--mode", choices=["exact", "param"], default="exact",
                   help="exact token identity, or modulo parameters and gate pin order (slots)")
    m.add_argument("--dl-params", action="store_true", help="parameter values count as tokens")
    m.add_argument("--workers", type=int, default=1, help="processes for mining pass 1")
    m.set_defaults(fn=cmd_mine)
    c = sub.add_parser("compress", help="compress saved archives into --out (archives, library, events, cells)")
    c.add_argument("--archives", nargs="+", required=True)
    c.add_argument("--library", default="")
    c.add_argument("--out", required=True)
    c.add_argument("--k-min", type=int, default=2)
    c.add_argument("--k-max", type=int, default=4)
    c.add_argument("--min-lineages", type=int, default=2)
    c.add_argument("--threshold", type=int, default=8)
    c.add_argument("--max-admissions", type=int, default=64)
    c.add_argument("--max-rounds", type=int, default=6)
    c.add_argument("--top", type=int, default=64, help="candidates considered per round")
    c.add_argument("--mode", choices=["exact", "param"], default="exact")
    c.add_argument("--dl-params", action="store_true", help="parameter values count as tokens (slot costs)")
    c.add_argument("--verify", default="auto", choices=["auto", "sim", "hash", "phash", "none"])
    c.add_argument("--workers", type=int, default=1, help="processes for mining pass 1")
    c.add_argument("--phase", type=int, default=0)
    c.add_argument("--freerun", action="store_true")
    c.set_defaults(fn=cmd_compress)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
