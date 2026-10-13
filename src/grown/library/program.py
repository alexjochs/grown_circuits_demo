"""The wake / compress / dream program, with checkpoints.

One cycle:
  wake      `grown.loop.run` for `wake_generations` generations on every task (one archive per task, one shared
            library), the operator wrapped in a `ModuleTracker`; edits go to `edits/phase_CCC/<task>/`.
  compress  usage update from the wake logs, `compress_phase` on all archives, then the retirement rule.
  dream     a hook `dream(program, cycle) -> dict | None` (stub: does nothing; the learner fills it).

State is checkpointed after every step into `<out>/ckpt/` (archives, library, usage, classification cache,
program state), written to `ckpt.tmp/` and swapped in, so a killed run resumes from the last finished step. The
append-only event files (`admissions.jsonl`, `rewrites.jsonl`, `retirements.jsonl`, `status.jsonl`) are truncated
on resume to the line counts recorded in the checkpoint, and a wake step that was interrupted is re-run from
scratch (its edit-log directory is removed first). Every random stream is derived from (seed, cycle, task), so a
resumed run is bit-identical to an uninterrupted one.

Status line per step: coverage and QD score summed over tasks, library size (active / total), admissions and
retirements of the cycle, module-token fraction of elites, the canonical-cell fraction of active modules, seconds.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .. import loop
from ..archive import Archive, Axis
from ..circuit import Library, OperatorConfig, UniformOperator
from ..log import EditLog
from .canon_cells import library_report
from .compress import CompressConfig, compress_phase, description_length
from .usage import ModuleTracker, Usage, UsageConfig

STEPS = ("wake", "compress", "dream")
EVENT_FILES = ("admissions.jsonl", "rewrites.jsonl", "retirements.jsonl", "status.jsonl")


@dataclass
class Task:
    name: str
    score_fn: object                 # EvalResult -> per-step error [B, S]; may carry n_out
    inputs: np.ndarray               # uint8 [S, n_in]


@dataclass
class ProgramConfig:
    cycles: int = 4
    wake_generations: int = 50
    batch: int = 64
    K: int = 4
    sampling: str = "curiosity"
    n_init: int = 64
    init_nodes: int = 8
    max_nodes: int = 64
    seed: int = 0
    axes: list = field(default_factory=lambda: [
        {"name": "n_gates", "lo": 0, "hi": 64, "bins": 16}, {"name": "cycle_frac", "lo": 0, "hi": 1, "bins": 8},
        {"name": "toggle_rate", "lo": 0, "hi": 0.5, "bins": 8, "source": "behavior"}])
    compress: CompressConfig = field(default_factory=CompressConfig)
    usage: UsageConfig = field(default_factory=UsageConfig)
    classify: bool = True
    freerun: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ProgramConfig":
        d = dict(d)
        d["compress"] = CompressConfig(**d.get("compress", {}))
        d["usage"] = UsageConfig(**d.get("usage", {}))
        return ProgramConfig(**d)


def derive_seed(seed: int, *path) -> int:
    h = hashlib.blake2b(json.dumps([seed, *path]).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & ((1 << 63) - 1)


def no_dream(program, cycle: int):
    return None


def search_wake(search_cfg: dict, workers: int = 1):
    """A wake hook running `grown.search` (teacher-matched search over the config's tasks) on the program's archives:
    they are written in the search layout under `<phase dir>/start/archives` together with the previous search
    wake's state files (random stream, blend, DAgger data), resumed with the program's library for
    `wake_generations` more generations (`search.run(resume=..., library=...)`), and read back. Edit logs and the
    ModuleTracker records land in `<phase dir>/search/edits/<task>/`, where `compress` finds them. The program's
    task names must be the config's training task names (`search_tasks`)."""
    def wake(program, cycle: int) -> dict:
        from .. import search
        from ..config import merge
        d = program.phase_dir(cycle)
        start, out = os.path.join(d, "start"), os.path.join(d, "search")
        prev = getattr(program, "_search_out", None)
        if prev is None and cycle > 0:
            cand = os.path.join(program.phase_dir(cycle - 1), "search")
            prev = cand if os.path.isdir(cand) else None
        os.makedirs(os.path.join(start, "archives"), exist_ok=True)
        for t, a in zip(program.tasks, program.archives):
            a.save(os.path.join(start, "archives", t.name))
            for ext in (".state.json", ".data.npz", ".data.json"):
                src = prev and os.path.join(prev, "archives", t.name + ext)
                if src and os.path.exists(src):
                    shutil.copyfile(src, os.path.join(start, "archives", t.name + ext))
        cfg = merge(search_cfg, {"search": {"generations": program.cfg.wake_generations}})
        res = search.run(cfg, out, workers=workers, verbose=False, resume=start, library=program.library)
        names = [t["task"] for t in res["tasks"]]
        if names != [t.name for t in program.tasks]:
            raise ValueError(f"search tasks {names} differ from the program's {[t.name for t in program.tasks]}")
        program.archives = [Archive.load(os.path.join(out, "archives", t.name), program.library)
                            for t in program.tasks]
        records = {}
        for t in program.tasks:
            p = os.path.join(out, "edits", t.name, "tracker.json")
            if os.path.exists(p):
                with open(p) as f:
                    records[t.name] = json.load(f)
        with open(os.path.join(d, "tracker.json"), "w") as f:
            json.dump(records, f)
        program._search_out = out
        return {"tasks": {t["task"]: {"best_error": t["best_error"], "solved": t["solved"],
                                      "lineages": t["lineages"]} for t in res["tasks"]},
                "edits_per_s": res["edits_per_s"], "solved": res["solved"]}
    return wake


def search_tasks(search_cfg: dict) -> list:
    """Program task placeholders named after a search config's training tasks (the search scores them itself)."""
    from ..search import build_tasks
    return [Task(t.name, None, np.zeros((0, 0), np.uint8)) for t in build_tasks(search_cfg)["train"]]


class Program:
    def __init__(self, out: str, tasks, cfg: ProgramConfig | None = None, substrate=None, dream=no_dream,
                 log=print, wake=None, archives=None, library=None, first_step: str = "wake",
                 generation: int = 0):
        """wake: a hook `wake(program, cycle) -> dict` replacing the built-in `grown.loop` wake (e.g. `search_wake`:
        grown.search resumed on the program's archives). archives / library / first_step / generation: the
        starting state of a fresh program (e.g. the archives of a finished search run, entering at "compress")."""
        from ..sim import SyncSubstrate
        self.out = str(out)
        self.tasks = list(tasks)
        self.substrate = substrate or SyncSubstrate()
        self.dream_hook = dream
        self.wake_hook = wake
        self.log = log
        os.makedirs(self.out, exist_ok=True)
        self.ckpt = os.path.join(self.out, "ckpt")
        if os.path.exists(os.path.join(self.ckpt, "state.json")):
            self._load(cfg)
        else:
            self.cfg = cfg or ProgramConfig()
            self.library = library if library is not None else Library()
            self.archives = list(archives) if archives is not None else [self._new_archive(i)
                                                                         for i in range(len(self.tasks))]
            for a in self.archives:
                a.library = self.library
            self.usage = Usage(self.cfg.usage)
            self.cycle, self.step_i, self.generation = 0, STEPS.index(first_step), int(generation)
            self.history: list = []
            self.cell_cache: dict = {}
            self.cycle_counts = {"admitted": 0, "retired": 0}
            for f in EVENT_FILES:
                open(os.path.join(self.out, f), "w").close()
            self._checkpoint()
        self.operator = ModuleTracker(UniformOperator(OperatorConfig(max_nodes=self.cfg.max_nodes), self.library))

    def _new_archive(self, i: int) -> Archive:
        return Archive([Axis(**a) for a in self.cfg.axes], sig_dim=loop.SIG_DIM, qd_offset=-1.0,
                       library=self.library, task=i)

    # --- checkpoints ------------------------------------------------------------------------------------------
    def _checkpoint(self) -> None:
        tmp = self.ckpt + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for i, a in enumerate(self.archives):
            a.save(os.path.join(tmp, f"archive_{i:03d}"))
        self.library.save(os.path.join(tmp, "library.json"))
        self.usage.save(os.path.join(tmp, "usage.json"))
        with open(os.path.join(tmp, "cells.json"), "w") as f:
            json.dump(self.cell_cache, f)
        lines = {}
        for fn in EVENT_FILES:
            p = os.path.join(self.out, fn)
            lines[fn] = sum(1 for _ in open(p)) if os.path.exists(p) else 0
        state = {"cycle": self.cycle, "step": self.step_i, "generation": self.generation, "history": self.history,
                 "cfg": self.cfg.to_dict(), "tasks": [t.name for t in self.tasks], "event_lines": lines,
                 "cycle_counts": self.cycle_counts}
        with open(os.path.join(tmp, "state.json"), "w") as f:
            json.dump(state, f, default=float)
        old = self.ckpt + ".old"
        shutil.rmtree(old, ignore_errors=True)
        if os.path.exists(self.ckpt):
            os.replace(self.ckpt, old)
        os.replace(tmp, self.ckpt)
        shutil.rmtree(old, ignore_errors=True)
        # the library file next to the event logs is the current vocabulary
        self.library.save(os.path.join(self.out, "library.json"))

    def _load(self, cfg) -> None:
        with open(os.path.join(self.ckpt, "state.json")) as f:
            st = json.load(f)
        self.cfg = ProgramConfig.from_dict(st["cfg"])
        if cfg is not None and dataclasses.replace(cfg, cycles=self.cfg.cycles).to_dict() != self.cfg.to_dict():
            raise ValueError("checkpoint was written with a different configuration")
        if cfg is not None:
            self.cfg.cycles = cfg.cycles             # extending a run is allowed
        if st["tasks"] != [t.name for t in self.tasks]:
            raise ValueError(f"checkpoint tasks {st['tasks']} differ from {[t.name for t in self.tasks]}")
        self.library = Library.load(os.path.join(self.ckpt, "library.json"))
        self.archives = [Archive.load(os.path.join(self.ckpt, f"archive_{i:03d}"), self.library)
                         for i in range(len(self.tasks))]
        self.usage = Usage.load(os.path.join(self.ckpt, "usage.json"))
        with open(os.path.join(self.ckpt, "cells.json")) as f:
            self.cell_cache = json.load(f)
        self.cycle, self.step_i, self.generation = st["cycle"], st["step"], st["generation"]
        self.history = st["history"]
        self.cycle_counts = st["cycle_counts"]
        for fn, n in st["event_lines"].items():
            p = os.path.join(self.out, fn)
            lines = open(p).readlines()[:n] if os.path.exists(p) else []
            with open(p, "w") as f:
                f.writelines(lines)
        self.log(f"resumed from {self.ckpt}: cycle {self.cycle}, next step {STEPS[self.step_i]}, "
                 f"generation {self.generation}")

    # --- steps --------------------------------------------------------------------------------------------------
    def phase_dir(self, cycle: int) -> str:
        return os.path.join(self.out, "edits", f"phase_{cycle:03d}")

    def wake(self) -> dict:
        c = self.cycle
        d = self.phase_dir(c)
        shutil.rmtree(d, ignore_errors=True)
        if self.wake_hook is not None:
            r = self.wake_hook(self, c)
            self.generation += self.cfg.wake_generations
            return r
        records = {}
        hist = {}
        for i, (t, a) in enumerate(zip(self.tasks, self.archives)):
            elog = EditLog(os.path.join(d, t.name), sig_dim=loop.SIG_DIM)
            lc = loop.LoopConfig(generations=self.cfg.wake_generations, batch=self.cfg.batch, K=self.cfg.K,
                                 sampling=self.cfg.sampling, n_init=self.cfg.n_init, init_nodes=self.cfg.init_nodes,
                                 task=i, seed=derive_seed(self.cfg.seed, "wake", c, i))
            self.operator.take()
            h = loop.run(self.substrate, t.score_fn, a, elog, t.inputs, self.operator, lc, library=self.library)
            elog.close()
            records[t.name] = ModuleTracker.to_json(self.operator.take())
            hist[t.name] = {"best": h[-1]["max_fitness"], "accepted": int(sum(x["accepted"] for x in h))}
        with open(os.path.join(d, "tracker.json"), "w") as f:
            json.dump(records, f)
        self.generation += self.cfg.wake_generations
        return {"tasks": hist}

    def compress(self) -> dict:
        c = self.cycle
        d = self.phase_dir(c)
        start = self.generation - self.cfg.wake_generations
        counts = {}
        if os.path.exists(os.path.join(d, "tracker.json")):
            with open(os.path.join(d, "tracker.json")) as f:
                records = json.load(f)
            ed = os.path.join(d, "search", "edits")          # search_wake's layout
            ed = ed if os.path.isdir(ed) else d
            for t in self.tasks:
                elog = EditLog(os.path.join(ed, t.name), sig_dim=loop.SIG_DIM if ed == d else 0)
                recs = records.get(t.name)
                counts[t.name] = self.usage.update(elog, None if recs is None else ModuleTracker.from_json(recs),
                                                   offset=start, generations=self.cfg.wake_generations)
        rep = compress_phase(self.archives, self.library, self.cfg.compress, phase=c, events_dir=self.out,
                             log=self.log)
        self.usage.note_admissions([e.hash for e in rep.admissions], self.generation)
        retired = self.usage.retire(self.library)
        with open(os.path.join(self.out, "retirements.jsonl"), "a") as f:
            for h in retired:
                f.write(json.dumps({"phase": c, "generation": self.generation, "hash": h.hex(),
                                    "name": self.library.get(h).name,
                                    "admitted": self.usage.admitted.get(h.hex())}) + "\n")
        self.cycle_counts = {"admitted": len(rep.admissions), "retired": len(retired)}
        turn = self.usage.turnover(self.archives, self.library, c, len(rep.admissions), len(retired))
        return {"compress": rep.summary(), "usage": counts, "turnover": turn,
                "admitted_modules": [asdict(e) for e in rep.admissions], "retired_modules": [h.hex() for h in retired]}

    def dream(self) -> dict:
        r = self.dream_hook(self, self.cycle)
        return {"dream": r}

    # --- running ------------------------------------------------------------------------------------------------
    def cells(self) -> dict:
        if not self.cfg.classify or not len(self.library.active()):
            return {"matched_frac": float("nan"), "modules": 0}
        return library_report(self.library, self.archives, freerun=self.cfg.freerun, cache=self.cell_cache)

    def status(self, step: str, seconds: float, extra: dict) -> dict:
        st = [a.stats() for a in self.archives]
        rep = self.cells()
        row = {"cycle": self.cycle, "step": step, "generation": self.generation,
               "coverage": float(np.mean([s["coverage"] for s in st])),
               "qd_score": float(sum(s["qd_score"] for s in st)),
               "best": [s["max_fitness"] for s in st], "library_active": len(self.library.active()),
               "library_total": len(self.library), **self.cycle_counts,
               "module_token_frac": (self.usage.phases[-1]["module_token_frac"] if self.usage.phases else 0.0),
               "dl": description_length(self.archives, self.library, self.cfg.compress.dl_params),
               "cells_matched": rep.get("matched", 0), "cells_frac": rep["matched_frac"], "seconds": seconds}
        self.log(f"cycle {self.cycle} {step:8s} gen {self.generation:5d}  coverage {row['coverage']:.4f}  "
                 f"qd {row['qd_score']:9.3f}  lib {row['library_active']}/{row['library_total']}  "
                 f"+{row['admitted']} -{row['retired']}  modtok {row['module_token_frac']:.3f}  "
                 f"dl {row['dl']}  cells {row['cells_matched']}/{rep.get('modules', 0)}  {seconds:6.1f}s")
        with open(os.path.join(self.out, "status.jsonl"), "a") as f:
            f.write(json.dumps({**row, **extra}, default=float) + "\n")
        return row

    def run_step(self) -> dict:
        step = STEPS[self.step_i]
        if step == "wake":
            self.cycle_counts = {"admitted": 0, "retired": 0}
        t0 = time.time()
        extra = getattr(self, step)()
        row = self.status(step, time.time() - t0, extra)
        self.history.append(row)
        self.step_i += 1
        if self.step_i == len(STEPS):
            self.step_i = 0
            self.cycle += 1
        self._checkpoint()
        return row

    def run(self, max_steps: int | None = None) -> list:
        n = 0
        while self.cycle < self.cfg.cycles and (max_steps is None or n < max_steps):
            self.run_step()
            n += 1
        return self.history
