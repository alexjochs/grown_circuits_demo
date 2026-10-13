"""Teacher-matched quality-diversity search: circuits from grown.circuit, tasks and teachers from grown.tasks,
evaluated through a grown.sim Substrate, kept in per-task MAP-Elites archives, every edit logged.

Per task (TaskSearch):
  data       TeacherDataset.from_rollouts(task) -> TaskData: per-episode input event streams
             (IOSpec.encode_events; legacy tasks: the exact old per-tick bits of IOSpec.encode_episode, as events)
             and row bookkeeping
  evaluate   every circuit on every episode as one substrate lane (state reset per episode), sent as an
             EventLaneBatch (any K; dense substrates get per-tick bits for K <= 4096); output wires' duty over each
             control step (EvalResult.high, never traces) -> IOSpec.decode -> actions; grown.tasks.score per row
             against the teacher -> per-step error [B, N]
  budget     the task's TimeModel event budget per step (docs/grown_time.md): lanes whose events in some step exceed
             it are flagged (EvalResult.over_budget); budget_mode fail (default) scores the circuit as failed
             (error 1, fitness -1), penalize subtracts budget_penalty * the fraction of its lanes over budget,
             ignore only logs. Behaviour values events_per_step and over_budget go through EvalResult.extra
  fitness    -((1 - w) * imitation error + w * (1 - task return)), w from a Blend schedule (0 = pure imitation)
  signature  SIG_NAMES: error, error on teacher-event / steady steps, error trend, toggle rate, output duty mean
             and spread, sampled-output flip rate; behavioural archive axes are drawn from the same names
  step       breed (uniform operator), evaluate, insert, log (grown.loop helpers)
  generation SearchGroup steps several tasks together: every task breeds, all children of compatible tasks
             (same widths, K and episode length) are flattened into one NetBatch and evaluated in one substrate
             call, then each task scores, inserts and logs its own. With search.overlap (default on) the tasks are
             split in two halves and one half's substrate call runs in a background thread while the other half
             breeds and settles. Tasks are independent, so results are the same as stepping them one by one.
  DAgger     every `dagger.every` generations: fresh episodes driven by the top elites, relabelled by the teacher,
             appended to the dataset (open-loop tasks only; closed-loop bodies are a stub that raises)
  refresh    whenever the objective changes (blend weight, dataset), the elites are re-scored and re-inserted,
             so stored fitness always means the current objective

`run(cfg, out)` drives a whole config (grown.config) over its training tasks, sequentially or across worker
processes (tasks are independent: results do not depend on the number of workers), printing one status line per
generation. See docs/grown_io.md for the io conventions and docs/grown_performance.md for the throughput work.

Search options read here (not in grown.config.DEFAULTS): search.child_hashes ("inserted", default: the edit log
records canonical hashes of children that enter the archive and zeros for the others; "all": every valid child,
as before 2026-10-09's performance work, several times slower on large circuits), search.overlap (default: on
for thread-safe substrates, see SearchGroup).
"""
from __future__ import annotations

import json
import math
import os
import queue
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import loop
from .archive.map_elites import Archive, Axis
from .circuit.graph import random_circuit
from .circuit.library import Library
from .circuit.moves import OperatorConfig, UniformOperator
from .sim.base import (EvalResult, concat_streams, evaluate_lanes, events_from_dense, events_to_dense, flag_budget,
                       step_events_from_toggles)

SIG_NAMES = ("error", "err_event", "err_steady", "err_trend", "toggle_rate", "duty_mean", "duty_std", "flip_rate")
SIG_DIM = len(SIG_NAMES)
QD_OFFSET = -1.0                 # fitness lies in [-1, 0]; QD score = sum over elites of (fitness + 1)

# purposes of derived seeds
_P_RNG, _P_DATA, _P_DAGGER, _P_EVAL = 0, 1, 2, 3


def derive_seed(seed: int, *path) -> int:
    """A 31-bit int that is a pure function of (seed, *path) (path items: ints or strings)."""
    words = [int(seed)]
    for p in path:
        words.append(int(p, 16) if isinstance(p, str) else int(p))
    return int(np.random.SeedSequence(words).generate_state(1)[0] & 0x7FFFFFFF)


def task_seed_word(task) -> str:
    return task.uid[:12]


# ---- the objective schedule --------------------------------------------------------------------------------------

@dataclass
class Blend:
    """Weight w of task return in the fitness: fitness = -((1 - w) * imitation_error + w * (1 - return)).

    imitation  w = 0 always
    linear     w ramps from 0 at generation `start` to `w_max` at `end`, in `steps` equal increments (each change
               re-scores the archive, so the ramp is quantized)
    threshold  w = 0 until the best elite's imitation error is <= `threshold` and the generation is >= `start`, then
               w_max from then on (sticky)
    """
    schedule: str = "imitation"
    w_max: float = 0.5
    start: int = 0
    end: int = 0
    steps: int = 4
    threshold: float = 0.05
    switched: bool = False

    def weight(self, generation: int, best_error: float) -> float:
        if self.schedule == "imitation":
            return 0.0
        if self.schedule == "linear":
            if generation < self.start:
                return 0.0
            span = max(self.end - self.start, 1)
            f = min((generation - self.start) / span, 1.0)
            f = math.floor(f * self.steps + 1e-9) / self.steps
            return float(self.w_max * f)
        if self.schedule == "threshold":
            if not self.switched and generation >= self.start and best_error <= self.threshold:
                self.switched = True
            return float(self.w_max) if self.switched else 0.0
        raise ValueError(f"unknown schedule {self.schedule!r}")

    @classmethod
    def from_config(cls, d: dict) -> "Blend":
        return cls(d.get("schedule", "imitation"), float(d.get("w_max", 0.5)), int(d.get("start", 0)),
                   int(d.get("end", 0)), int(d.get("steps", 4)), float(d.get("threshold", 0.05)))


# ---- data ---------------------------------------------------------------------------------------------------------

class TaskData:
    """The circuit-facing view of a TeacherDataset.

    events sim.stepper.EventStreams, one per episode: the input wire changes in ticks (padded episodes repeat their last
           observation; padded steps are never scored)
    bits   uint8 [E, T, K, n_in]: input wire values at every circuit tick (dense; computed on first use and only for
           K <= sim.base.DENSE_MAX_K: the stand-in, baselines and tests)
    row_ep, row_t  [N]: episode index and step of every dataset row (results are gathered to dataset row order,
           so grown.tasks.score applies directly)
    event  bool [N]: the teacher's action changed from the previous step of the same episode
    late   bool [N]: the row lies in the second half of its episode
    """

    def __init__(self, task, ds):
        self.task, self.ds, self.io = task, ds, task.io_spec()
        ids, inv = np.unique(ds.episode_id, return_inverse=True)
        order = np.lexsort((ds.step, inv))
        counts = np.bincount(inv)
        first = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pos = np.empty(len(ds), np.int64)
        pos[order] = np.arange(len(ds)) - first[inv[order]]
        self.E, self.T = len(ids), int(counts.max()) if len(ds) else 0
        self.row_ep, self.row_t, self.lengths = inv.astype(np.int64), pos, counts
        D = ds.obs.shape[1]
        obs = np.zeros((self.E, self.T, D), np.float32)
        obs[inv, pos] = ds.obs
        for e in np.flatnonzero(counts < self.T):                    # pad with the last observation
            obs[e, counts[e]:] = obs[e, counts[e] - 1]
        self.obs = obs
        self.time = getattr(task, "time", None)
        self._bits = None
        if self.time is None or self.time.is_legacy:
            # the pre-time-model encoder, bit for bit (float32 sigma-delta at the tick rate), carried as events
            self._bits = self.io.encode_episode(obs)
            self.events = events_from_dense(self._bits, self.io.K)
        else:
            self.events = self.io.encode_events(obs)
        prev = np.full(len(ds), -1, np.int64)
        prev[order[1:]] = order[:-1]
        same = (prev >= 0) & (inv[np.maximum(prev, 0)] == inv)
        self.event = same & np.any(ds.act != ds.act[np.maximum(prev, 0)], -1)
        self.late = pos >= counts[inv] / 2

    @property
    def bits(self) -> np.ndarray:
        if self._bits is None:
            self._bits = events_to_dense(self.events, self.T, self.io.K, self.io.n_in)
            self.events.dense = self._bits
        return self._bits

    @property
    def budget(self):
        """Events allowed per step (None: unlimited) and the mode, from the task's time model."""
        if self.time is None:
            return None, "ignore"
        return self.time.budget(self.task.dt), self.time.budget_mode

    @property
    def cap(self):
        """The compute guard (TimeModel.sim_event_cap): events per step above which a lane stops being simulated."""
        return None if self.time is None else self.time.cap(self.task.dt)

    @property
    def stop_events(self):
        """The early-stop value the substrate gets (EventLaneBatch.budget): the budget in fail mode, the compute cap,
        the smaller of the two, or None."""
        budget, mode = self.budget
        c = [v for v in ((budget if mode == "fail" else None), self.cap) if v is not None]
        return min(c) if c else None

    def __len__(self):
        return len(self.ds)

    def episode_rows(self, e: int) -> np.ndarray:
        r = np.flatnonzero(self.row_ep == e)
        return r[np.argsort(self.row_t[r])]


def trivial_error(ds, metric: str = "abs") -> float:
    """Error of the best constant action per dim (the median for abs; the mean for sq; the mode for bit)."""
    from .tasks import score
    y = ds.act
    if metric == "sq":
        c = y.mean(0)
    elif metric == "bit":
        lo = np.asarray(ds.meta.get("act_lo", np.full(y.shape[1], -1.0)), np.float32)
        hi = np.asarray(ds.meta.get("act_hi", np.full(y.shape[1], 1.0)), np.float32)
        c = np.where((y > (lo + hi) / 2).mean(0) > 0.5, hi, lo)
    else:
        c = np.median(y, 0)
    return float(score(np.broadcast_to(c, y.shape), ds, metric).mean())


# ---- evaluation ---------------------------------------------------------------------------------------------------

@dataclass
class Evaluation:
    err: np.ndarray          # [B, N] per-step imitation error, dataset row order
    actions: np.ndarray      # [B, N, act_dim] decoded actions, dataset row order
    imitation: np.ndarray    # [B] mean imitation error
    ret: np.ndarray          # [B] mean task return in [0, 1] (NaN when not computed)
    fitness: np.ndarray      # [B]
    sig: np.ndarray          # [B, SIG_DIM]
    behavior: dict           # name -> [B], SIG_NAMES


def evaluate(substrate, circuits, data: TaskData, library=None, metric: str = "abs", w: float = 0.0,
             need_return: bool = False, nets=None) -> Evaluation:
    """Every circuit on every episode of `data` (one LaneBatch through sim.base.evaluate_requests, the same path
    the generation loop uses), scored against the teacher."""
    from .circuit.netbatch import flatten_batch
    from .sim.base import EventLaneBatch, evaluate_requests
    B, E = len(circuits), data.E
    nb = nets if nets is not None else flatten_batch(circuits, library)
    req = EventLaneBatch(nb, np.repeat(np.arange(B, dtype=np.int64), E), data.events,
                         np.tile(np.arange(E, dtype=np.int64), B), data.io.K, data.T, data.stop_events)
    res, = evaluate_requests(substrate, [req])
    return evaluate_results(res, B, data, metric, w, need_return)


def evaluate_results(res, B: int, data: TaskData, metric: str = "abs", w: float = 0.0,
                     need_return: bool = False) -> Evaluation:
    """Score substrate lanes (circuit b on episode e is lane b * E + e) against the task's teacher, then apply the
    task's event budget (module doc)."""
    from .tasks import score
    io, task = data.io, data.task
    E, T = data.E, data.T
    duty = res.high.reshape(B, E, T, -1)
    act = np.asarray(io.decode(duty), np.float32)                   # [B, E, T, A]
    rows = act[:, data.row_ep, data.row_t]                           # [B, N, A]
    err = np.asarray(score(rows, data.ds, metric), np.float64)       # [B, N]
    budget, mode = data.budget
    cap = data.cap
    ev = np.asarray(res.events if res.events is not None else step_events_from_toggles(res), np.float64)
    L = B * E
    capped = np.zeros(L, bool)
    if cap is not None:
        # the substrate's stop flags mean "over the cap" when the cap was the stop value (not the fail-mode budget)
        stopped = None if res.over_budget is None else np.asarray(res.over_budget, bool)
        fail_stop = mode == "fail" and budget is not None and budget <= cap
        if stopped is not None and not fail_stop:
            capped |= stopped
            res.over_budget = stopped if (budget is not None and budget <= cap) else None
        if ev.ndim == 2 and ev.shape[1]:
            capped |= ev.max(1) > cap
    flag_budget(res, budget)
    over = np.asarray(res.over_budget, bool).reshape(B, E)
    res.extra = dict(res.extra or {}, events_per_step=ev.mean(1) if ev.shape[1] else np.zeros(L),
                     over_budget=over.ravel().astype(np.float64), capped=capped.astype(np.float64))
    failed = over.any(1) if (budget is not None and mode == "fail") else np.zeros(B, bool)
    failed = failed | capped.reshape(B, E).any(1)                  # unscorable: not simulated to the end
    if failed.any():
        err[failed] = 1.0
    imit = err.mean(1)
    ret = np.full(B, np.nan)
    if w > 0 or need_return:
        ret = np.array([np.mean([task_return(task, data, e, act[b, e, :data.lengths[e]]) for e in range(E)])
                        for b in range(B)])
        ret[failed] = 0.0
    fit = -((1.0 - w) * imit + (w * (1.0 - ret) if w > 0 else 0.0))
    fit[failed] = -1.0
    if budget is not None and mode == "penalize":
        fit = np.maximum(fit - data.time.budget_penalty * over.mean(1), -1.0)

    def masked_mean(m):
        return err[:, m].mean(1) if m.any() else imit
    tog = res.toggles.reshape(B, E).sum(1)
    beh = {
        "error": imit,
        "err_event": masked_mean(data.event),
        "err_steady": masked_mean(~data.event),
        "err_trend": masked_mean(data.late) - masked_mean(~data.late),
        "toggle_rate": tog / (np.maximum(res.n_gates.reshape(B, E)[:, 0], 1) * E * T * io.K),
        "duty_mean": duty.mean((1, 2, 3)),
        "duty_std": duty.std(2).mean((1, 2)),
        "flip_rate": (np.diff(res.outputs.reshape(B, E, T, -1).astype(np.int8), axis=2) != 0).mean((1, 2, 3))
        if T > 1 else np.zeros(B),
    }
    for k, v in (getattr(res, "extra", None) or {}).items():      # per-lane extras: mean over episodes
        beh[k] = np.asarray(v, np.float64).reshape(B, E).mean(1)
    sig = np.stack([beh[k] for k in SIG_NAMES], 1).astype(np.float32)
    return Evaluation(err, rows, imit, ret, fit, sig, beh)


def _group_key(data: TaskData, library):
    """Tasks whose children can share one EventLaneBatch: same widths, K, episode length and early-stop budget (and
    library, when it defines modules the circuits may use)."""
    lib = id(library) if library is not None and len(library) else None
    return (data.io.n_in, data.io.n_out, data.io.K, data.T, data.stop_events, lib)


def launch_many(jobs, pool=None):
    """Start evaluating several tasks' circuits: jobs = [(substrate, circuits, data, library)]. Every job's lanes
    (circuit b on episode e) go into EventLaneBatch requests, one per group of compatible tasks (_group_key), and all
    requests to one substrate go out in ONE `sim.base.evaluate_requests` call (flattening included), in `pool` (a
    thread) when given. -> a handle for `collect_many`."""
    from .sim.base import EventLaneBatch, evaluate_requests
    by_sub: dict = {}
    for j, (sub, circs, data, lib) in enumerate(jobs):
        if len(circs):
            by_sub.setdefault(id(sub), (sub, {}))[1].setdefault(_group_key(data, lib), []).append(j)
    runs = []
    for sub, groups in by_sub.values():
        specs = []
        for js in groups.values():
            sizes = [len(jobs[j][1]) for j in js]
            E = [jobs[j][2].E for j in js]
            off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            soff = np.concatenate([[0], np.cumsum(E)]).astype(np.int64)
            lane_circ = np.concatenate([np.repeat(np.arange(b, dtype=np.int64) + o, e)
                                        for b, o, e in zip(sizes, off[:-1], E)])
            lane_stream = np.concatenate([np.tile(np.arange(e, dtype=np.int64) + so, b)
                                          for b, so, e in zip(sizes, soff[:-1], E)])
            streams = concat_streams([jobs[j][2].events for j in js])
            d0 = jobs[js[0]][2]
            specs.append(dict(js=js, sizes=sizes, off=off, E=E, circs=[jobs[j][1] for j in js],
                              lib=jobs[js[0]][3], K=d0.io.K, S=d0.T, budget=d0.stop_events,
                              lane_circ=lane_circ, lane_stream=lane_stream, streams=streams))

        def work(sub=sub, specs=specs):
            from .circuit.netbatch import flatten_seqs
            t = time.time()
            reqs = [EventLaneBatch(flatten_seqs(sp["circs"], sp["lib"]), sp["lane_circ"], sp["streams"],
                                   sp["lane_stream"], sp["K"], sp["S"], sp["budget"]) for sp in specs]
            res = evaluate_requests(sub, reqs)
            return reqs, res, time.time() - t
        runs.append((specs, pool.submit(work) if pool is not None else _Done(work())))
    return jobs, runs


class _Done:
    def __init__(self, v):
        self.v = v

    def result(self):
        return self.v


def collect_many(handle, metric="abs", ws=None) -> list:
    """Finish `launch_many`: per job (Evaluation or None, NetBatch or None, substrate seconds)."""
    jobs, runs = handle
    out: list = [(None, None, 0.0)] * len(jobs)
    ws = ws or [0.0] * len(jobs)
    for specs, fut in runs:
        reqs, results, secs = fut.result()
        L_all = sum(len(r.lane_circ) for r in reqs)
        for sp, req, res in zip(specs, reqs, results):
            lo = 0
            for j, b, o, e in zip(sp["js"], sp["sizes"], sp["off"][:-1], sp["E"]):
                hi = lo + b * e
                cut = lambda a: None if a is None else np.asarray(a)[lo:hi]  # noqa: E731
                r = EvalResult(res.outputs[lo:hi], res.high[lo:hi], res.toggles[lo:hi], res.n_gates[lo:hi],
                               res.ticks, events=cut(getattr(res, "events", None)),
                               over_budget=cut(getattr(res, "over_budget", None)))
                extra = getattr(res, "extra", None)
                if extra:
                    r.extra = {k: np.asarray(v)[lo:hi] for k, v in extra.items()}
                ev = evaluate_results(r, b, jobs[j][2], metric[j] if isinstance(metric, list) else metric, ws[j])
                out[j] = (ev, req.nets.select(int(o), int(o) + b), secs * b * e / max(L_all, 1))
                lo = hi
    return out


def task_return(task, data: TaskData, e: int, ya) -> float:
    """The task's own episode criterion in [0, 1] for student actions ya [T_e, act_dim] on episode e."""
    if hasattr(task, "stream_return"):
        r = data.episode_rows(e)
        return task.stream_return(data.ds.obs[r], data.ds.act[r], ya)
    raise NotImplementedError(f"{type(task).__name__} has no open-loop return; closed-loop task return needs the "
                              "circuit stepped inside the task rollout (see circuit_policy)")


# ---- DAgger -------------------------------------------------------------------------------------------------------

def require_open_loop(task) -> None:
    if not getattr(task, "open_loop", False):
        raise NotImplementedError(
            f"closed-loop DAgger is not implemented for {getattr(task, 'name', task)!r} "
            f"(family {getattr(task, 'family', '?')!r}): its observations depend on the actions, so the circuit "
            "must be stepped inside the task rollout as the policy (circuit_policy), which needs a substrate that "
            "steps one control step at a time and carries net state and sigma-delta carry between steps")


def circuit_policy(circuit, io, substrate, library=None):
    """Hook: a circuit as a rollout policy (obs, carry) -> (action, carry) for closed-loop tasks (bodies,
    sensorimotor). The carry would hold the circuit's net state and the sigma-delta encoder carry
    (IOSpec.encode_ticks). Not implemented: the stand-in and freerun adapters evaluate whole streams, not single
    steps, and the policy must run inside jax.lax.scan."""
    raise NotImplementedError("circuit_policy: closed-loop circuit policies need a stepping substrate API "
                              "(freerun's device-resident step(inputs) -> outputs); open-loop signal tasks do not "
                              "need one")


# ---- per-task search ----------------------------------------------------------------------------------------------

def behavior_names() -> tuple:
    """Names a behavioural archive axis may use: SIG_NAMES plus the substrate's per-lane extras
    (grown.sim.base.EXTRA_BEHAVIOR, e.g. freerun's idealization gap and energy, filled through EvalResult.extra)."""
    from .sim import base
    return SIG_NAMES + tuple(getattr(base, "EXTRA_BEHAVIOR", ()))


def make_axes(cfg_axes) -> list:
    axes = [a if isinstance(a, Axis) else Axis(**a) for a in cfg_axes]
    names = behavior_names()
    for a in axes:
        if a.source == "behavior" and a.name not in names:
            raise KeyError(f"behavioral axis {a.name!r} not in {names}")
    return axes


def make_substrate(d: dict):
    from .sim import FreerunSubstrate, SyncSubstrate
    name = d.get("name", "sync")
    if name == "sync":
        return SyncSubstrate()
    if name == "freerun_events":
        from .sim.freerun_events import EventSubstrate
        return EventSubstrate.from_config(d)
    if name == "freerun":
        if hasattr(FreerunSubstrate, "from_config"):          # the adapter reads its own config keys
            return FreerunSubstrate.from_config(d)
        return FreerunSubstrate(d.get("backend", "cpu"), int(d.get("threads", 0)), d.get("home"), d.get("libdir"),
                                d.get("stimulus", "packed"))
    raise ValueError(f"unknown substrate {name!r}")


def _write_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def checkpoint_dir(out: str, task: str, generation: int) -> str:
    return os.path.join(out, "checkpoints", task, f"g{int(generation):08d}")


def checkpoint_generations(out: str, task: str) -> list:
    """Generations of the complete checkpoints of a task under <out>/checkpoints/<task>, ascending."""
    root = os.path.join(out, "checkpoints", task)
    if not os.path.isdir(root):
        return []
    return sorted(int(n[1:]) for n in os.listdir(root) if n.startswith("g") and n[1:].isdigit()
                  and os.path.isdir(os.path.join(root, n)))


def common_checkpoint(out: str, tasks) -> int | None:
    """The newest generation at which every task has a complete checkpoint (None: none)."""
    gens = None
    for t in tasks:
        g = set(checkpoint_generations(out, t))
        gens = g if gens is None else gens & g
    return max(gens) if gens else None


class TaskSearch:
    def __init__(self, task, cfg: dict, index: int = 0, substrate=None, library=None, out: str | None = None,
                 resume: str | None = None, defer_seed: bool = False, checkpoint: int | None = None):
        """`resume`: a run directory (or a compressed copy of one) to continue from instead of seeding: see
        `resume_from`. `defer_seed`: draw the seed population now but leave its evaluation to `SearchGroup.seed`
        (all tasks' seeds in one engine call; the archives are the same). `checkpoint`: with `resume`, the generation
        of the checkpoint under `<resume>/checkpoints/<task>/` to continue from (see `save`). With a `library` that has active modules the operator offers INSERT_MODULE and is wrapped in
        a library.usage.ModuleTracker (records written next to the edit log at close)."""
        from .tasks import TeacherDataset
        self.task, self.cfg, self.index, self.out = task, cfg, index, out
        self.io = task.io_spec()
        self.seed = int(cfg.get("seed", 0))
        self.word = task_seed_word(task)
        sc, dc = cfg["search"], cfg["data"]
        self.rng = np.random.default_rng(derive_seed(self.seed, self.word, _P_RNG))
        self.substrate = substrate or make_substrate(cfg.get("substrate", {}))
        self.library = library if library is not None else Library()
        self.metric = dc.get("metric", "abs")
        self.horizon = int(dc["horizon"]) if dc.get("horizon") else None
        ds = TeacherDataset.from_rollouts(task, keys=derive_seed(self.seed, self.word, _P_DATA),
                                          n_episodes=int(dc["episodes"]), horizon=self.horizon)
        self.n_base_episodes = ds.n_episodes
        self.data = TaskData(task, ds)
        self.trivial = trivial_error(ds, self.metric)
        self.archive = Archive(make_axes(cfg["archive"]["axes"]), sig_dim=SIG_DIM, qd_offset=QD_OFFSET,
                               library=self.library, task=index)
        self.operator = UniformOperator(OperatorConfig(**_op_cfg(sc.get("operator", {}))), self.library)
        self.blend = Blend.from_config(cfg.get("objective", {}))
        self.w = 0.0
        self.generation = 0
        self.dagger_rounds = 0
        self.history: list = []
        self.edits = 0
        self.eval_seconds = 0.0
        self.child_hashes = str(sc.get("child_hashes", "inserted"))
        self.log = None
        if out:
            from .log import EditLog
            self.log = EditLog(os.path.join(out, "edits", task.name), sig_dim=SIG_DIM)
        self.tracker = None
        if len(self.library.active()):
            from .library.usage import ModuleTracker
            self.operator = self.tracker = ModuleTracker(self.operator)
            self.child_hashes = "all"        # the tracker hashes every child; Usage aligns its records on them
        self.start_generation = 0
        self.resumed_from = None
        self.pending_seeds = None
        if resume:
            self.resume_from(resume, checkpoint)
        elif defer_seed:
            self.pending_seeds = self.seed_circuits()
        else:
            self._seed_population()

    # -- resume (E5's second wake: continue a run from its saved, possibly compressed, archives) -----------------
    def state(self) -> dict:
        """What `save` writes next to the archive so that `resume_from` continues bit-identically."""
        st = {"generation": self.generation, "edits": self.edits, "w": self.w, "blend_switched": self.blend.switched,
              "dagger_rounds": self.dagger_rounds, "eval_seconds": self.eval_seconds,
              "rng": self.rng.bit_generator.state, "dataset": self.dagger_rounds > 0, "task_uid": self.task.uid}
        if self.log is not None:                     # the edit log as of this state (flushed by save)
            st.update(log_chunks=len(self.log.chunks), log_rows=int(self.log.next_id))
        return st

    def resume_from(self, path: str, checkpoint: int | None = None) -> None:
        """Continue from `<path>/archives/<task>` (Archive.save layout, as a search run or a compress phase writes
        it; module instances need the library this search was given). With the `<task>.state.json` that `save`
        writes (generation, blend state, DAgger dataset, the random generator's state) the continuation is
        bit-identical to an uninterrupted run. Runs saved before the state file existed resume at their recorded
        generation (history file, else summary.json, else the archive's batch count) with a fresh random stream
        derived from (seed, task, generation)."""
        from .tasks import TeacherDataset
        if checkpoint is not None:
            ck = checkpoint_dir(path, self.name, checkpoint)
            prefix, hist_path = os.path.join(ck, self.name), os.path.join(ck, "history.json")
        else:
            prefix = os.path.join(path, "archives", self.name)
            hist_path = os.path.join(path, "history", self.name + ".json")
        if not os.path.exists(prefix + ".npz"):
            raise FileNotFoundError(f"resume: no archive for task {self.name!r} at {prefix}")
        a = Archive.load(prefix, self.library)
        if [x.name for x in a.axes] != [x.name for x in self.archive.axes] or a.shape != self.archive.shape:
            raise ValueError(f"resume: archive axes of {prefix} differ from the config's")
        a.task = self.index
        self.archive = a
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                self.history = json.load(f)
        st = None
        if os.path.exists(prefix + ".state.json"):
            with open(prefix + ".state.json") as f:
                st = json.load(f)
        if st is not None:
            if st.get("task_uid", self.task.uid) != self.task.uid:
                raise ValueError(f"resume: {prefix} was saved for another task")
            self.generation, self.w = int(st["generation"]), float(st["w"])
            self.blend.switched, self.dagger_rounds = bool(st["blend_switched"]), int(st["dagger_rounds"])
            self.eval_seconds = float(st.get("eval_seconds", 0.0))
            self.rng.bit_generator.state = st["rng"]
            if (self.log is not None and "log_chunks" in st and self.out
                    and os.path.abspath(self.out) == os.path.abspath(path)):
                self.log.truncate(int(st["log_chunks"]), int(st["log_rows"]))   # drop edits logged after the state
            if st.get("dataset"):
                ds = TeacherDataset.load(prefix + ".data")
                self.data = TaskData(self.task, ds)
                self.trivial = trivial_error(ds, self.metric)
        else:
            gen = len(self.history) if self.history else None
            if gen is None and os.path.exists(os.path.join(path, "summary.json")):
                with open(os.path.join(path, "summary.json")) as f:
                    gen = next((int(t["generations"]) for t in json.load(f).get("tasks", [])
                                if t.get("task") == self.name), None)
            self.generation = int(gen if gen is not None else max(a.n_batches - 1, 0))
            self.rng = np.random.default_rng(derive_seed(self.seed, self.word, _P_RNG, self.generation))
        self.history = self.history[:self.generation]
        self.start_generation = self.generation
        self.resumed_from = os.path.abspath(path)

    @property
    def name(self) -> str:
        return self.task.name

    @property
    def dataset(self):
        return self.data.ds

    def evaluate(self, circuits, data: TaskData | None = None) -> Evaluation:
        t = time.time()
        ev = evaluate(self.substrate, circuits, data or self.data, self.library, self.metric, self.w)
        self.eval_seconds += time.time() - t
        return ev

    def seed_circuits(self) -> list:
        """The random seed population (draws from the search's generator)."""
        sc = self.cfg["search"]
        lo, hi = sc.get("init_arity", [1, 3])
        dr = tuple(int(v) for v in sc.get("operator", {}).get("delay_range", (1, 1)))
        return [random_circuit(self.io.n_in, self.io.n_out, int(sc["init_nodes"]), self.rng, (int(lo), int(hi)),
                               delay_range=dr, feedback=bool(sc.get("init_feedback", True)))
                for _ in range(int(sc["n_init"]))]

    def insert_seeds(self, seeds, ev) -> None:
        self.archive.insert(self.archive.descriptors(seeds, ev.behavior), ev.fitness, items=seeds, sigs=ev.sig,
                            lineage=np.arange(len(seeds)))
        self.pending_seeds = None

    def _seed_population(self):
        seeds = self.seed_circuits()
        self.insert_seeds(seeds, self.evaluate(seeds))

    def best_error(self) -> float:
        f = self.archive.filled
        return float(self.archive.sig[f, 0].min()) if f.any() else float("inf")

    def best_cell(self) -> int:
        cells = self.archive.filled_cells()
        return int(cells[np.argmin(self.archive.sig[cells, 0])])

    def elites(self, n: int | None = None, by: str = "fitness") -> np.ndarray:
        """Filled cells, best first (by fitness, or by imitation error with by="error")."""
        cells = self.archive.filled_cells()
        key = -self.archive.fitness[cells] if by == "fitness" else self.archive.sig[cells, 0]
        cells = cells[np.lexsort((cells, key))]
        return cells if n is None else cells[:n]

    def refresh(self) -> None:
        """Re-score every elite under the current objective and data. archive.rescore = "inplace" (default) keeps
        each elite in its cell and updates fitness and signature (behavioural descriptors stay as measured at
        insertion); "rebin" clears the archive and re-inserts the elites with fresh descriptors (elites whose
        descriptors collide are lost)."""
        a = self.archive
        cells = a.filled_cells()
        if not len(cells):
            return
        items = [a.items[c] for c in cells]
        ev = self.evaluate(items)
        if self.cfg["archive"].get("rescore", "inplace") == "inplace":
            a.fitness[cells] = ev.fitness
            a.sig[cells] = ev.sig
            return
        lineage = a.lineage[cells].copy()
        counters = (a.n_candidates, a.n_inserted, a.n_batches)
        a.clear()
        a.insert(a.descriptors(items, ev.behavior), ev.fitness, items=items, sigs=ev.sig, lineage=lineage)
        a.n_candidates, a.n_inserted, a.n_batches = counters

    def dagger_round(self) -> int:
        """One DAgger round: fresh episodes driven by the top elites (round robin), labelled by the teacher, appended
        to the dataset; the archive is re-scored. Returns the number of rows appended."""
        from .tasks import TeacherDataset, score
        require_open_loop(self.task)
        dc = self.cfg["dagger"]
        n_ep = int(dc.get("episodes", 2))
        drivers = self.elites(max(int(dc.get("elites", 4)), 1))
        key = derive_seed(self.seed, self.word, _P_DAGGER, self.dagger_rounds)
        new = TeacherDataset.from_rollouts(self.task, keys=key, n_episodes=n_ep, horizon=self.horizon)
        nd = TaskData(self.task, new)
        ev = self.evaluate([self.archive.items[c] for c in drivers], nd)
        who = np.arange(nd.E) % len(drivers)                          # episode e is driven by elite who[e]
        execd = ev.actions[who[nd.row_ep], np.arange(len(new))]
        new.exec_act = execd.astype(np.float32)
        new.reward = -score(execd, new, self.metric).astype(np.float32)
        new.meta["provenance"].update(policy="circuit", round=self.dagger_rounds, generation=self.generation,
                                      drivers=[bytes(self.archive.hash[c].astype("<u8")).hex() for c in drivers])
        ds = self.data.ds.concat(new)
        cap = int(dc.get("max_episodes", 0) or 0)
        if cap and ds.n_episodes > cap:
            ids = np.unique(ds.episode_id)
            keep = np.isin(ds.episode_id, ids[-cap:])
            ds = type(ds)(ds.obs[keep], ds.act[keep], ds.episode_id[keep], ds.step[keep], ds.meta,
                          None if ds.exec_act is None else ds.exec_act[keep],
                          None if ds.reward is None else ds.reward[keep])
        self.data = TaskData(self.task, ds)
        self.trivial = trivial_error(ds, self.metric)
        self.dagger_rounds += 1
        self.refresh()
        return len(new)

    # -- one generation, in phases (SearchGroup interleaves the phases of several tasks) --------------------------
    def begin_generation(self) -> int:
        """Objective schedule and DAgger for the coming generation. Returns the DAgger rows appended."""
        g = self.generation
        w = self.blend.weight(g, self.best_error())
        if w != self.w:
            self.w = w
            self.refresh()
        dag = int(self.cfg.get("dagger", {}).get("every", 0) or 0)
        self._appended = self.dagger_round() if dag and g > 0 and g % dag == 0 else 0
        return self._appended

    def breed(self) -> dict:
        sc = self.cfg["search"]
        return loop.breed(self.archive, self.operator, self.rng, int(sc["batch"]), sc.get("sampling", "curiosity"),
                          None, float(sc.get("eps", 0.1)))

    def eval_job(self, br: dict):
        """(substrate, circuits, data, library) for launch_many: the valid children of `breed`."""
        return (self.substrate, br["children"] if br["ok"].any() else [], self.data, self.library)

    def finish_generation(self, br: dict, ev, nets, eval_seconds: float = 0.0, seconds: float = 0.0) -> dict:
        """Descriptors, insertion, log and stats for the evaluated children of `breed`."""
        g = self.generation
        self.eval_seconds += eval_seconds
        desc = fit = sig = None
        if br["ok"].any():
            same = br.get("same_graph")
            same_as = None if same is None else np.where(same[br["ok"]], br["cells"][br["ok"]], -1)
            desc = self.archive.descriptors(br["children"], ev.behavior, nets, same_as)
            fit, sig = ev.fitness, ev.sig
        st = loop.settle(self.archive, br, desc, fit, sig, self.child_hashes)
        if self.log is not None:
            loop.log_edits(self.log, g, self.index, br, st)
        self.edits += len(br["ok"])
        stats = self.archive.stats()
        fate = st["fate"]
        ev_ps = over = capped = float("nan")
        if br["ok"].any() and ev is not None:
            ev_ps = float(np.mean(ev.behavior["events_per_step"]))
            over = float(np.mean(ev.behavior["over_budget"]))
            capped = float(np.mean(ev.behavior["capped"]))
        stats.update(task=self.name, generation=g, seconds=seconds, edits=len(br["ok"]),
                     invalid=int((~br["ok"]).sum()), accepted=int((fate >= 4).sum()), best_error=self.best_error(),
                     trivial=self.trivial, w=self.w, rows=len(self.data), dagger_rows=self._appended,
                     child_events_per_step=ev_ps, child_over_budget=over, child_capped=capped,
                     eval_seconds=eval_seconds)
        self.history.append(stats)
        self.generation += 1
        every = int(self.cfg["archive"].get("save_every", 0) or 0)
        if self.out and every and self.generation % every == 0:
            self.save()
        return stats

    def step(self) -> dict:
        t0 = time.time()
        self.begin_generation()
        br = self.breed()
        (ev, nets, secs), = collect_many(launch_many([self.eval_job(br)]), self.metric, [self.w])
        return self.finish_generation(br, ev, nets, secs, time.time() - t0)

    def save(self) -> None:
        """A checkpoint: the archive, its state (generation, random generator, objective, DAgger dataset, edit log
        position) and history, as one directory `<out>/checkpoints/<task>/g<generation>` (written under a temporary
        name and renamed, so it is complete or absent; `LATEST` names the newest; the last `archive.keep_checkpoints`
        (3) are kept), and as the run's current files `<out>/archives/<task>.*` and `<out>/history/<task>.json`
        (each replaced atomically, so tools such as grown_eval can read a running search). `resume_from(out,
        checkpoint=g)` continues bit-identically."""
        if not self.out:
            return
        if self.log is not None:
            self.log.flush()
        g = self.generation
        root = os.path.join(self.out, "checkpoints", self.name)
        final = checkpoint_dir(self.out, self.name, g)
        if not os.path.isdir(final):
            tmp = final + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            self._write_files(os.path.join(tmp, self.name), os.path.join(tmp, "history.json"))
            os.rename(tmp, final)
            _write_atomic(os.path.join(root, "LATEST"), str(g))
            keep = max(int(self.cfg["archive"].get("keep_checkpoints", 3) or 3), 1)
            for old in checkpoint_generations(self.out, self.name)[:-keep]:
                shutil.rmtree(checkpoint_dir(self.out, self.name, old), ignore_errors=True)
        os.makedirs(os.path.join(self.out, "archives"), exist_ok=True)
        os.makedirs(os.path.join(self.out, "history"), exist_ok=True)
        self._write_files(os.path.join(self.out, "archives", self.name),
                          os.path.join(self.out, "history", self.name + ".json"), atomic=True)

    def _write_files(self, prefix: str, hist_path: str, atomic: bool = False) -> None:
        """Archive (prefix.npz / .json), DAgger dataset (prefix.data.*), state (prefix.state.json), history."""
        d, base = os.path.split(prefix)
        stage = os.path.join(d, f".{base}.tmp") if atomic else prefix
        self.archive.save(stage)
        files = [(".npz", stage + ".npz"), (".json", stage + ".json")]
        if self.dagger_rounds:
            self.data.ds.save(stage + ".data")
            files += [(".data.npz", stage + ".data.npz"), (".data.json", stage + ".data.json")]
        with open(stage + ".state.json", "w") as f:
            json.dump(self.state(), f)
        files.append((".state.json", stage + ".state.json"))
        if atomic:
            for ext, src in files:
                os.replace(src, prefix + ext)
            _write_atomic(hist_path, json.dumps(self.history))
        else:
            with open(hist_path, "w") as f:
                json.dump(self.history, f)

    def close(self) -> None:
        self.save()
        if self.log is not None:
            self.log.close()
            self.log = None
        if self.tracker is not None and self.out:
            from .library.usage import ModuleTracker
            with open(os.path.join(self.out, "edits", self.name, "tracker.json"), "w") as f:
                json.dump(ModuleTracker.to_json(self.tracker.take()), f)

    def _module_elites(self) -> int:
        """Elites holding at least one library module instance."""
        from .circuit.graph import MODULE
        a = self.archive
        return int(sum(any(nd.op == MODULE for nd in a.items[c].nodes) for c in a.filled_cells()
                       if "packed" not in a.items[c].__dict__ or a.items[c].__dict__["packed"] is None))

    def summary(self) -> dict:
        c = self.best_cell()
        best = self.archive.items[c]
        ev = evaluate(self.substrate, [best], self.data, self.library, self.metric, 0.0, need_return=True)
        tags = self.task.tags
        return dict(task=self.name, uid=self.task.uid, kind=tags.extra.get("kind"), index=self.index,
                    dt=self.task.dt, K=self.io.K, best_events_per_step=float(ev.behavior["events_per_step"][0]),
                    best_over_budget=float(ev.behavior["over_budget"][0]),
                    generations=self.generation, edits=self.edits, best_error=self.best_error(), trivial=self.trivial,
                    progress=1.0 - self.best_error() / self.trivial if self.trivial > 0 else float("nan"),
                    solved=bool(self.best_error() <= float(self.cfg.get("solved", 0.01))),
                    best_return=float(ev.ret[0]), best_nodes=len(best.nodes),
                    best_hash=best.hash.hex(), coverage=self.archive.coverage(), qd_score=self.archive.qd_score(),
                    filled=int(self.archive.filled.sum()), w=self.w, dagger_rounds=self.dagger_rounds,
                    lineages=self.archive.lineages(), start_generation=self.start_generation,
                    resumed_from=self.resumed_from, module_elites=self._module_elites(),
                    rows=len(self.data), eval_seconds=round(self.eval_seconds, 2), tags=tags.to_dict())


class SearchGroup:
    """Several TaskSearches stepped together (see the module docstring). `step()` -> one stats dict per task, the
    same as calling each task's `step()` in turn. Tasks sharing a substrate object get one substrate call per
    generation (two with overlap); `run` gives all of a process's tasks the same substrate.

    overlap: one half of the tasks' substrate call runs in a background thread while the other half breeds and
    settles. Default (search.overlap unset): on for substrates that declare `threadsafe = True` (the compiled sync
    stand-in releases the GIL), off otherwise."""

    def __init__(self, searches, overlap: bool | None = None):
        self.searches = list(searches)
        self.seed_seconds = 0.0
        self.engine_log: list = []          # placement records of the last seed() / step() (placement_summary)
        cfg = self.searches[0].cfg if self.searches else {"archive": {}}
        self.save_seconds = float(cfg["archive"].get("save_seconds", 0) or 0)   # also checkpoint every so often
        self._last_save = time.time()
        if overlap is None and self.searches:
            overlap = self.searches[0].cfg["search"].get("overlap")
            if overlap is None:
                overlap = all(getattr(t.substrate, "threadsafe", False) for t in self.searches)
        overlap = bool(overlap)
        n = len(self.searches)
        self.parts = [list(range(n))] if not overlap or n < 2 else [list(range(n // 2)), list(range(n // 2, n))]
        self.pool = ThreadPoolExecutor(1) if overlap and n >= 2 else None

    def seed(self) -> None:
        """Evaluate every deferred seed population (TaskSearch(defer_seed=True)) in one launch (one engine call per
        substrate) and insert it; the archives are those of seeding each task on its own."""
        todo = [t for t in self.searches if t.pending_seeds is not None]
        if not todo:
            return
        t0 = time.time()
        res = collect_many(launch_many([(t.substrate, t.pending_seeds, t.data, t.library) for t in todo]),
                           [t.metric for t in todo], [t.w for t in todo])
        for t, (ev, nets, secs) in zip(todo, res):
            t.eval_seconds += secs
            t.insert_seeds(t.pending_seeds, ev)
        self.seed_seconds = time.time() - t0
        self.engine_log = self._take_logs()

    def _take_logs(self) -> list:
        subs = {id(t.substrate): t.substrate for t in self.searches}
        return [r for sub in subs.values() if hasattr(sub, "take_placement_log") for r in sub.take_placement_log()]

    def step(self) -> list:
        self.seed()
        t0 = time.time()
        for ts in self.searches:
            ts.begin_generation()
        stats: list = [None] * len(self.searches)
        pending = None
        for part in self.parts:
            brs = [self.searches[i].breed() for i in part]
            handle = launch_many([self.searches[i].eval_job(br) for i, br in zip(part, brs)], self.pool)
            if pending is not None:
                self._finish(*pending, stats)
            pending = (part, brs, handle)
        self._finish(*pending, stats)
        dt = (time.time() - t0) / max(len(self.searches), 1)
        for s in stats:
            s["seconds"] = dt
        self.engine_log = self._take_logs()
        if self.save_seconds and time.time() - self._last_save >= self.save_seconds:
            for ts in self.searches:          # all tasks at the same generation: a common checkpoint
                ts.save()
            self._last_save = time.time()
        return stats

    def _finish(self, part, brs, handle, stats):
        ts = [self.searches[i] for i in part]
        res = collect_many(handle, [t.metric for t in ts], [t.w for t in ts])
        for i, t, br, (ev, nets, secs) in zip(part, ts, brs, res):
            stats[i] = t.finish_generation(br, ev, nets, secs)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None


def placement_summary(records: list) -> dict | None:
    """One generation's (or the seeding's) lane placement records (FreerunSubstrate.placement_log) in a line: events
    in total and per side, the busiest lane and where it ran, lanes per side, seconds (probe, CPU side, GPU side, wall
    of the placed calls). None without records."""
    if not records:
        return None
    ev = sum(r["events"] for r in records)
    ecpu = sum(r["events_cpu"] for r in records)
    big = max(records, key=lambda r: r["busiest_lane"])
    return dict(calls=len(records), lanes=sum(r["lanes"] for r in records),
                cpu_lanes=sum(r["cpu_lanes"] for r in records), gpu_lanes=sum(r["gpu_lanes"] for r in records),
                events=ev, events_cpu=ecpu, events_gpu=ev - ecpu, cpu_share=ecpu / ev if ev else 0.0,
                busiest_lane=big["busiest_lane"], busiest_on=big["busiest_on"], busiest_K=big["busiest_K"],
                max_gpu_lane=max(r["max_gpu_lane"] for r in records),
                probe_s=sum(r["probe_s"] for r in records), cpu_s=sum(r["cpu_s"] for r in records),
                gpu_s=sum(r["gpu_s"] for r in records), wall_s=sum(r["wall_s"] for r in records))


def placement_line(tag, p: dict, seconds: float | None = None) -> str:
    sec = f"{seconds:.1f} s, " if seconds is not None else ""
    return (f"placement {tag}: {sec}{p['events']:.3g} events ({100 * p['cpu_share']:.1f} % on {p['cpu_lanes']} CPU "
            f"lanes, rest on {p['gpu_lanes']} GPU lanes), busiest lane {p['busiest_lane']:.3g} on {p['busiest_on']} "
            f"(K {p['busiest_K']}), busiest GPU lane {p['max_gpu_lane']:.3g}; probe {p['probe_s']:.1f} s, CPU "
            f"{p['cpu_s']:.1f} s, GPU {p['gpu_s']:.1f} s, calls {p['wall_s']:.1f} s")


def _op_cfg(d: dict) -> dict:
    d = dict(d)
    for k in ("delay_range", "kind_weights"):
        if k in d:
            d[k] = tuple(d[k])
    if "weights" in d:
        from .circuit.moves import MoveType
        d["weights"] = {MoveType[k.upper()] if isinstance(k, str) else MoveType(k): v for k, v in d["weights"].items()}
    return d


# ---- tasks from a config ------------------------------------------------------------------------------------------

def build_tasks(cfg: dict) -> dict:
    """-> {"train": [Task], "heldout": {split name: [Task]}} from cfg["tasks"] and cfg["time"] (see grown.config).
    Every task gets the config's TimeModel; its K is derived from its own dt."""
    from .config import check
    from .tasks.base import TimeModel
    check(cfg)
    tc = cfg["tasks"]
    tm = TimeModel.from_config(cfg.get("time"))
    src = tc.get("source", "catalog")
    horizon = int(tc.get("horizon", 1024))
    dt = tc.get("dt")
    if src == "catalog":
        from .tasks.signal import LEGACY_DT, SignalTask, catalog
        if isinstance(dt, (list, tuple)):
            raise ValueError("tasks.dt for the catalog is one number (or null: each entry's own rate)")
        cat = catalog(horizon=horizon, time=tm, dt=None if dt is None else float(dt))
        for spec in tc.get("extra") or []:                 # tasks defined in the config: {name, kind, params, dt}
            d = spec.get("dt", dt)
            if d is None:
                if not tm.is_legacy:
                    raise ValueError(f"tasks.extra {spec['name']!r} needs a dt (its sample period in seconds), or set "
                                     "tasks.dt")
                d = LEGACY_DT
            cat[spec["name"]] = SignalTask(spec["kind"], dict(spec.get("params") or {}), horizon=horizon, dt=float(d),
                                           name=spec["name"], time=tm)
        train = [cat[n] for n in tc.get("names", [])]
        held = [cat[n] for n in tc.get("heldout", []) or []]
        return {"train": train, "heldout": {"heldout": held} if held else {}}
    from .tasks import get_family
    fam_kw = dict(horizon=horizon, time=tm, dt=dt)
    if tc.get("kinds"):
        fam_kw["kinds"] = list(tc["kinds"])
    if tc.get("family", "signal") != "signal":
        raise NotImplementedError("only the signal family is wired to the search (bodies need closed-loop rollout)")
    fam = get_family("signal", **fam_kw)
    key = int(tc.get("family_seed", 0))
    if src == "family":
        sp = fam.splits(key, int(tc.get("n_train", 32)), int(tc.get("n_interp", 8)), int(tc.get("n_extrap", 8)))
        return {"train": list(sp.train), "heldout": {"interp": list(sp.interp), "extrap": list(sp.extrap)}}
    if src == "demand_holdout":
        from .tasks import demand_holdout
        pool = fam.sample(key, int(tc.get("n_pool", 64)), "train")
        box = {k: tuple(v) for k, v in (tc.get("box") or {}).items()}
        inside, out = demand_holdout(pool, box)
        return {"train": inside[:int(tc.get("max_train", 16))],
                "heldout": {"demand_complement": out[:int(tc.get("max_heldout", 16))]}}
    raise ValueError(f"unknown task source {src!r}")


def task_record(task, role: str) -> dict:
    from .tasks import task_meta
    m = task_meta(task)
    m["role"] = role
    m["K"] = task.io_spec().K
    tm = getattr(task, "time", None)
    if tm is not None:
        m["time"] = tm.to_dict()
    return m


def rebuild_task(rec: dict):
    """A task from its task_record (signal family). Records without a time model are pre-time-model runs: legacy
    K."""
    from .tasks.base import TimeModel
    from .tasks.signal import SignalTask
    tm = TimeModel.from_config(rec["time"]) if rec.get("time") else TimeModel.legacy(int(rec["K"]))
    t = SignalTask(rec["params"]["kind"], rec["params"], horizon=rec["horizon"], dt=rec["dt"], name=rec["name"],
                   time=tm)
    if t.K != int(rec["K"]):
        raise ValueError(f"task {rec['name']!r}: rebuilt K {t.K} != recorded K {rec['K']}")
    return t


# ---- the run ------------------------------------------------------------------------------------------------------

def _aggregate(stats: list, solved_thr: float) -> dict:
    nanmean = lambda k: float(np.nanmean([s.get(k, np.nan) for s in stats])) if any(  # noqa: E731
        np.isfinite(s.get(k, np.nan)) for s in stats) else float("nan")
    return dict(child_events_per_step=nanmean("child_events_per_step"), child_over_budget=nanmean("child_over_budget"),
                child_capped=nanmean("child_capped"),
                coverage=float(np.mean([s["coverage"] for s in stats])),
                qd=float(np.sum([s["qd_score"] for s in stats])),
                best=float(np.mean([s["max_fitness"] for s in stats])),
                err=float(np.mean([s["best_error"] for s in stats])),
                solved=int(sum(s["best_error"] <= solved_thr for s in stats)), n=len(stats),
                lineages=int(sum(s.get("lineages", 0) for s in stats)),
                edits=int(sum(s["edits"] for s in stats)), w=float(np.mean([s["w"] for s in stats])))


def status_line(g: int, agg: dict, eps: float) -> str:
    return (f"gen {g:5d}  coverage {agg['coverage']:.4f}  qd {agg['qd']:10.2f}  best {agg['best']:+.4f}  "
            f"err {agg['err']:.4f}  solved {agg['solved']}/{agg['n']}  lin {agg.get('lineages', 0)}  {eps:7.0f} edits/s  "
            f"w {agg['w']:.2f}")


def _worker(cfg, out, indices, queue, resume=None, library_path=None, checkpoint=None, generations=None):
    import jax
    jax.config.update("jax_platform_name", "cpu")
    tasks = build_tasks(cfg)["train"]
    substrate = make_substrate(cfg.get("substrate", {}))
    library = Library.load(library_path) if library_path else None
    searches = [TaskSearch(tasks[i], cfg, i, substrate=substrate, out=out, library=library, resume=resume,
                           defer_seed=True, checkpoint=checkpoint) for i in indices]
    group = SearchGroup(searches)
    group.seed()
    queue.put(("ready", indices, None))
    for _ in range(int(cfg["search"]["generations"]) if generations is None else int(generations)):
        for i, st in zip(indices, group.step()):
            queue.put(("stats", i, st))
    group.close()
    for i, ts in zip(indices, searches):
        ts.close()
        queue.put(("done", i, ts.summary()))


def run(cfg: dict, out: str | None = None, workers: int | None = None, verbose: bool = True,
        callback=None, resume: str | None = None, library=None, checkpoint_resume: bool = False) -> dict:
    """Run the search for a config. Returns the summary (also written to <out>/summary.json).

    checkpoint_resume: if <out> holds checkpoints (TaskSearch.save) of every training task, continue from the newest
    generation they share up to search.generations in total, bit-identically to a run that was never stopped (a
    multi-day run that is killed or requeued restarts with the same command); otherwise start fresh.

    resume: continue every task from `<resume>/archives` (TaskSearch.resume_from) for search.generations MORE
    generations. library: a Library or a library json path; its active modules join the vocabulary (INSERT_MODULE,
    ModuleTracker records in <out>/edits/<task>/tracker.json) and it is copied to <out>/library.json."""
    from .tasks.base import jsonable
    t_start = time.time()
    tasks = build_tasks(cfg)
    train = tasks["train"]
    G = int(cfg["search"]["generations"])
    workers = int(workers if workers is not None else cfg.get("workers", 1))
    workers = max(1, min(workers, len(train)))
    thr = float(cfg.get("solved", 0.01))
    every = max(int(cfg.get("status_every", 1)), 1)
    lib_obj = Library.load(library) if isinstance(library, str) else library
    lib_path = library if isinstance(library, str) else None
    if lib_obj is not None and out:
        lib_path = os.path.join(out, "library.json")
        os.makedirs(out, exist_ok=True)
        lib_obj.save(lib_path)
    elif lib_obj is not None and workers > 1 and lib_path is None:
        raise ValueError("a Library object with several workers needs an out directory (the workers load it)")
    ck = None
    if checkpoint_resume and out and resume is None:
        ck = common_checkpoint(out, [t.name for t in train])
        if ck is not None:
            resume = out
            if verbose:
                print(f"resuming from the checkpoint at generation {ck} in {out}", flush=True)
            with open(os.path.join(out, "resumes.jsonl"), "a") as f:
                f.write(json.dumps(dict(time=time.time(), generation=ck, target=G)) + "\n")
            pj = os.path.join(out, "placement.jsonl")
            if os.path.exists(pj):                  # placement records of generations after the checkpoint
                with open(pj) as f:
                    keep = [ln for ln in f if json.loads(ln)["generation"] == "seed"
                            or int(json.loads(ln)["generation"]) < ck]
                _write_atomic(pj, "".join(keep))
    if checkpoint_resume and out and ck is None:
        stale = [n for n in ("archives", "history", "edits", "checkpoints", "placement.jsonl")
                 if os.path.exists(os.path.join(out, n))]
        if stale:                      # an attempt that died before its first checkpoint: start clean, keep its files
            aside = os.path.join(out, f"aborted_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(aside)
            for n in stale:
                shutil.move(os.path.join(out, n), os.path.join(aside, n))
            if verbose:
                print(f"no checkpoint shared by every task in {out}: starting fresh, earlier files moved to {aside}",
                      flush=True)
    start = ck or 0
    G_run = max(G - start, 0)
    if out and resume and ck is None:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "resume.json"), "w") as f:
            json.dump({"resume": os.path.abspath(resume), "library": lib_path and os.path.abspath(lib_path),
                       "generations": G}, f, indent=1)
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(jsonable(cfg), f, indent=1)
        recs = [task_record(t, "train") for t in train]
        for split, ts in tasks["heldout"].items():
            recs += [task_record(t, split) for t in ts]
        with open(os.path.join(out, "tasks.json"), "w") as f:
            json.dump(jsonable(recs), f, indent=1)
    if verbose:
        held = {k: len(v) for k, v in tasks["heldout"].items()}
        print(f"{cfg.get('name', 'run')}: {len(train)} training tasks, held out {held}, {G} generations x "
              f"{cfg['search']['batch']} edits per task, {workers} worker(s)", flush=True)
    per_gen: dict = {}
    summaries: dict = {}
    history = []
    last = [time.time(), 0.0]                      # time of the last emitted generation, loop start

    def emit(g, stats):
        agg = _aggregate(stats, thr)
        now = time.time()
        eps = agg["edits"] / max(now - last[0], 1e-9)
        last[0] = now
        agg.update(generation=g, edits_per_s=eps, wall=now - t_start)
        history.append(agg)
        if verbose and (g % every == 0 or g == G - 1):
            print(status_line(g, agg, eps), flush=True)
        if callback is not None:
            callback(agg)

    if workers == 1:
        substrate = make_substrate(cfg.get("substrate", {}))
        searches = [TaskSearch(t, cfg, i, substrate=substrate, out=out, library=lib_obj, resume=resume,
                               defer_seed=True, checkpoint=ck) for i, t in enumerate(train)]
        group = SearchGroup(searches)
        group.seed()

        def log_placement(tag, seconds):
            p = placement_summary(group.engine_log)
            if p is None:
                return
            if out:
                with open(os.path.join(out, "placement.jsonl"), "a") as f:
                    f.write(json.dumps(dict(generation=tag, seconds=seconds, **p, records=group.engine_log)) + "\n")
            if verbose and (tag == "seed" or tag % every == 0 or tag == G - 1):
                print(placement_line(tag, p, seconds), flush=True)
        if ck is None:
            log_placement("seed", group.seed_seconds)
        last[0] = last[1] = time.time()
        for g in range(start, start + G_run):
            tg = time.time()
            emit(g, group.step())
            log_placement(searches[0].generation - 1, time.time() - tg)
        group.close()
        for i, ts in enumerate(searches):
            ts.close()
            summaries[i] = ts.summary()
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        parts = [list(range(len(train)))[w::workers] for w in range(workers)]
        procs = [ctx.Process(target=_worker, args=(cfg, out, p, q, resume, lib_path, ck, G_run), daemon=True)
                 for p in parts]
        n_stats = [0] * len(train)                 # generations reported per task (resumed tasks may start apart)
        for p in procs:
            p.start()
        ready = 0
        try:
            while len(summaries) < len(train):
                try:
                    kind, i, payload = q.get(timeout=5)
                except queue.Empty:
                    dead = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
                    if dead:
                        raise RuntimeError(f"search worker died (exit codes {dead})")
                    continue
                if kind == "ready":
                    ready += 1
                    if ready == workers:
                        last[0] = last[1] = time.time()
                elif kind == "stats":
                    g = start + n_stats[i]
                    n_stats[i] += 1
                    per_gen.setdefault(g, {})[i] = payload
                    if len(per_gen[g]) == len(train):
                        emit(g, [per_gen[g][j] for j in range(len(train))])
                        del per_gen[g]
                else:
                    summaries[i] = payload
        finally:
            for p in procs:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()
        bad = [p.exitcode for p in procs if p.exitcode]
        if bad:
            raise RuntimeError(f"worker exit codes {bad}")
    wall = time.time() - t_start
    loop_seconds = (t_start + history[-1]["wall"] - last[1]) if history else 0.0
    total = sum(s["edits"] for s in summaries.values())
    summary = dict(name=cfg.get("name"), seed=cfg.get("seed"), generations=G, workers=workers, wall_seconds=wall,
                   edits=total, loop_seconds=loop_seconds, edits_per_s=total / max(loop_seconds, 1e-9),
                   solved=sum(s["solved"] for s in summaries.values()), n_tasks=len(train),
                   tasks=[summaries[i] for i in range(len(train))], curve=history,
                   resumed_from=resume and os.path.abspath(resume), resumed_checkpoint=ck, library=lib_path,
                   library_modules=len(lib_obj.active()) if lib_obj is not None else 0)
    if out:
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(jsonable(summary), f, indent=1)
    return summary
