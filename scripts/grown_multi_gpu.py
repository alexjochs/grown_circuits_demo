"""Run one grown_search config on several GPUs: the (seed, task) pairs are split into one shard per GPU, each shard is
an ordinary grown_search run in its own process with CUDA_VISIBLE_DEVICES set to that GPU, and a table of the shards
is printed at the end.

    python scripts/grown_multi_gpu.py --config configs/grown/exp/e0_fr_unit_cuda.yaml            # every visible GPU
    python scripts/grown_multi_gpu.py --config ... --gpus 2 --generations 200 --eval --out runs/demo
    python scripts/grown_multi_gpu.py --config ... --gpu-ids 0,1 --backend cpu                   # dry run on CPUs

There is no communication between GPUs: every task's search is independent (its random streams derive from the seed
and the task, not from the shard it lands in), so a shard is the same search it would be on one GPU. Shards are
contiguous blocks of the (seed, task) list; with --seeds auto (default) the launcher adds seeds until every GPU has
at least one task. Catalog configs (tasks.source: catalog) are split by task and seed; other task sources by seed only.

Writes <out>/gpu<k>_s<seed>/ (a normal grown_search run directory each, with search.log) and <out>/multi_gpu.json.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from grown import config as C  # noqa: E402


def visible_gpus() -> list:
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is not None and v.strip() not in ("", "NoDevFiles"):
        return [g.strip() for g in v.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return [str(i) for i, ln in enumerate(out.splitlines()) if ln.startswith("GPU ")]
    except (OSError, subprocess.CalledProcessError):
        return []


def gpu_names() -> dict:
    """physical GPU index (as nvidia-smi numbers them, i.e. as CUDA_VISIBLE_DEVICES names them) -> model name"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"], capture_output=True,
                             text=True, check=True).stdout
        return {ln.split(",")[0].strip(): ln.split(",", 1)[1].strip() for ln in out.splitlines() if "," in ln}
    except (OSError, subprocess.CalledProcessError):
        return {}


def split(units: list, n: int) -> list:
    """n contiguous, near-equal blocks (sizes differ by at most one; empty blocks when n > len(units))"""
    q, r = divmod(len(units), n)
    out, i = [], 0
    for k in range(n):
        m = q + (k < r)
        out.append(units[i:i + m])
        i += m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpus", type=int, default=None, help="number of GPUs (default: every visible GPU)")
    ap.add_argument("--gpu-ids", default=None, help="comma separated GPU ids (default: CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--seeds", nargs="+", default=["auto"],
                    help="search seeds, or 'auto': the config's seed, plus more until every GPU has work")
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--backend", default="cuda", help="freerun backend for every shard (default cuda)")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes per shard (default: its task count, capped by the CPUs per GPU)")
    ap.add_argument("--out", default=None, help="default runs/multi_gpu/<config name>_<time>")
    ap.add_argument("--eval", action="store_true", help="run grown_eval on every shard afterwards")
    ap.add_argument("--set", action="append", default=[], help="passed to grown_search (config override)")
    a = ap.parse_args()

    ids = a.gpu_ids.split(",") if a.gpu_ids else visible_gpus()
    if a.gpus is not None:
        if a.gpus > len(ids):
            ap.error(f"--gpus {a.gpus} but only {len(ids)} GPU(s) visible ({ids}); use --gpu-ids to name them")
        ids = ids[:a.gpus]
    if not ids:
        ap.error("no GPU visible (set CUDA_VISIBLE_DEVICES or --gpu-ids)")
    n = len(ids)

    cfg = C.load(a.config)
    catalog = cfg["tasks"].get("source", "catalog") == "catalog"
    tasks = list(cfg["tasks"]["names"]) if catalog else [None]
    if a.seeds == ["auto"]:
        s0 = int(cfg.get("seed", 0))
        seeds = list(range(s0, s0 + max(1, math.ceil(n / len(tasks)))))
    else:
        seeds = [int(s) for s in a.seeds]
    units = [(s, t) for s in seeds for t in tasks]
    shards = split(units, n)
    out = a.out or os.path.join("runs", "multi_gpu", f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    per_gpu = max(1, cpus // n)
    names = gpu_names()
    print(f"{cfg['name']}: {len(tasks) if catalog else 'all'} task(s) x {len(seeds)} seed(s) on {n} GPU(s) {ids}, "
          f"backend {a.backend}, {per_gpu} CPU(s) per GPU, out {out}", flush=True)

    def runs_of(k):
        """the grown_search runs of shard k: one per seed, with that seed's tasks"""
        by_seed: dict = {}
        for s, t in shards[k]:
            by_seed.setdefault(s, []).append(t)
        return [(s, ts, os.path.join(out, f"gpu{k}_s{s}")) for s, ts in by_seed.items()]

    results = [dict(shard=k, gpu=ids[k], gpu_name=names.get(ids[k], ""), runs=[]) for k in range(n)]

    def shard(k):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=ids[k], PYTHONUNBUFFERED="1")
        t0 = time.time()
        for s, ts, rd in runs_of(k):
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "grown_search.py"), "--config", a.config,
                   "--seed", str(s), "--out", rd, "--set", f"substrate.backend={a.backend}"]
            if catalog:
                cmd += ["--set", "tasks.names=[" + ", ".join(ts) + "]"]
            w = a.workers or min(len(ts) if catalog else int(cfg.get("workers", 1)), per_gpu)
            cmd += ["--workers", str(w)]
            if a.generations is not None:
                cmd += ["--generations", str(a.generations)]
            for x in a.set:
                cmd += ["--set", x]
            os.makedirs(rd, exist_ok=True)
            print(f"[gpu {ids[k]}] seed {s}: {', '.join(ts) if catalog else 'all tasks'} ({w} worker(s))", flush=True)
            with open(os.path.join(rd, "search.log"), "w") as log:
                rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            rec = dict(seed=s, tasks=ts if catalog else None, workers=w, out=rd, exit_code=rc)
            if rc == 0:
                sm = json.load(open(os.path.join(rd, "summary.json")))
                rec.update(edits=sm["edits"], loop_seconds=sm["loop_seconds"], edits_per_s=sm["edits_per_s"],
                           solved=sm["solved"], n_tasks=sm["n_tasks"], wall_seconds=sm["wall_seconds"])
                if a.eval:
                    ev = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "grown_eval.py"), "--run", rd,
                                         "--quiet"], env=env, capture_output=True, text=True)
                    if ev.returncode == 0:
                        e = json.load(open(os.path.join(rd, "eval.json")))
                        rec["solved_fresh"] = e["groups"]["train"]["coverage"]["solved"]
                    rec["eval_exit_code"] = ev.returncode
            else:
                print(f"[gpu {ids[k]}] seed {s}: grown_search exit {rc}, see {rd}/search.log", flush=True)
            results[k]["runs"].append(rec)
        results[k]["wall_seconds"] = time.time() - t0

    t0 = time.time()
    threads = [threading.Thread(target=shard, args=(k,)) for k in range(n) if shards[k]]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t0

    print(f"\n{'shard':>5s} {'GPU':>4s}  {'model':24s} {'seeds':6s} {'tasks':>5s} {'edits':>9s} {'loop s':>7s} "
          f"{'edits/s':>8s} {'wall s':>7s} {'solved':>6s}")
    tot_e = tot_eps = 0.0
    bad = 0
    for r in results:
        ok = [x for x in r["runs"] if x.get("exit_code") == 0]
        bad += len(r["runs"]) - len(ok)
        if not r["runs"]:
            print(f"{r['shard']:5d} {r['gpu']:>4s}  (no tasks)")
            continue
        e = sum(x["edits"] for x in ok)
        ls = sum(x["loop_seconds"] for x in ok)
        nt = sum(x["n_tasks"] for x in ok)
        sv = sum(x.get("solved_fresh", x["solved"]) for x in ok)
        r.update(edits=e, loop_seconds=ls, edits_per_s=e / ls if ls else 0.0, n_tasks=nt, solved=sv)
        tot_e += e
        tot_eps += r["edits_per_s"]
        seeds_s = ",".join(str(x["seed"]) for x in r["runs"])
        print(f"{r['shard']:5d} {r['gpu']:>4s}  {r['gpu_name'][:24]:24s} {seeds_s:6s} {nt:5d} {e:9d} {ls:7.1f} "
              f"{r['edits_per_s']:8.0f} {r['wall_seconds']:7.1f} {sv:6d}")
    print(f"total: {int(tot_e)} edits on {n} GPU(s), {tot_eps:.0f} edits/s summed over GPUs, wall {wall:.1f} s"
          + (f", {bad} run(s) FAILED" if bad else ""))
    rec = dict(config=a.config, name=cfg["name"], backend=a.backend, gpus=ids, seeds=seeds, generations=a.generations,
               job_id=os.environ.get("SLURM_JOB_ID"), wall_seconds=wall, edits=int(tot_e), edits_per_s_sum=tot_eps,
               shards=results)
    with open(os.path.join(out, "multi_gpu.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {os.path.join(out, 'multi_gpu.json')}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
