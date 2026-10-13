"""Frontier-generated tasks: a POET-style proposer that mutates existing tasks toward the edge of solvability.

The archive side injects `score_fn(task) -> float`: the current best student return on the task (same units as
the task's return). The proposer
  1. mutates parents with their family's `mutate(task, key)` (parameters may leave the original box),
  2. measures the teacher's return and a trivial baseline's return (best constant action),
  3. applies the minimal criterion, judged by these returns:
       valid:      teacher_return - trivial_return >= margin  (the teacher really solves something; a spec whose
                   output is constant, or a body where doing nothing is as good as the teacher, is degenerate)
       progress p = (student - trivial) / (teacher - trivial)
       not trivial: p <= hi    not impossible: p >= lo
  4. ranks the accepted children by novelty in demand-tag space (mean distance to the k nearest tasks in the pool)
     and admits the most novel.

Every proposal is logged with its fate (accepted / reason), parent and numbers; this is training data for later.
Default teacher / trivial returns: exact and cheap for signal tasks (the teacher is perfect, the trivial baseline is
computed from the reference stream); a generic rollout otherwise. Both can be injected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .base import AXES, as_key, np_rng, rollout, stateless

_LOG = ("memory_depth", "timing_precision", "n_in", "n_out")


@dataclass
class FrontierConfig:
    lo: float = 0.1                 # minimal progress: below this the task is (currently) impossible
    hi: float = 0.9                 # maximal progress: above this it is already solved
    margin: float = 1e-3            # teacher must beat the trivial baseline by at least this (raw return units)
    children_per_parent: int = 4
    novelty_k: int = 5
    eval_episodes: int = 2


@dataclass
class Proposal:
    task: object
    parent: str | None
    teacher_return: float
    trivial_return: float
    student_return: float
    progress: float
    accepted: bool
    reason: str
    novelty: float = 0.0

    def record(self) -> dict:
        return dict(uid=self.task.uid, name=self.task.name, parent=self.parent, teacher_return=self.teacher_return,
                    trivial_return=self.trivial_return, student_return=self.student_return, progress=self.progress,
                    accepted=self.accepted, reason=self.reason, novelty=self.novelty,
                    tags=self.task.tags.to_dict())


def tag_features(tags) -> np.ndarray:
    """Tag vector with log2 on the scale axes (so novelty is about octaves, not raw counts)."""
    v = tags.vector().copy()
    for i, a in enumerate(AXES):
        if a in _LOG:
            v[i] = math.log2(max(v[i], 1e-6))
    return v


def novelty(tags, pool_features: np.ndarray, k: int = 5) -> float:
    if len(pool_features) == 0:
        return float("inf")
    d = np.linalg.norm(pool_features - tag_features(tags), axis=1)
    return float(np.sort(d)[:k].mean())


def default_returns(task, key, episodes: int = 2) -> tuple[float, float]:
    """(teacher return, trivial return): the trivial policy is the best constant action among lo / mid / hi of the
    action range, applied to every action dim."""
    io = task.io_spec()
    lo, hi = io.act_lo, io.act_hi
    consts = [lo, (lo + hi) / 2, hi]
    keys = jax.random.split(as_key(key), episodes)
    if hasattr(task, "reference"):                    # signal task: teacher exact, trivial from the stream
        ys = np.stack([task.reference(jax.random.split(k)[0])[1] for k in keys])
        errs = [np.mean(np.abs(c - ys) / (hi - lo), -1).sum(-1).mean() for c in consts]
        return 0.0, -float(min(errs))

    def ret(policy):
        tr = jax.jit(jax.vmap(lambda k: rollout(task, k, policy)))(keys)
        return float(np.asarray(tr["reward"]).sum(-1).mean())

    teacher = ret(None)
    trivial = max(ret(stateless(lambda o, c=jnp.asarray(c): c)) for c in consts)
    return teacher, trivial


class FrontierSampler:
    """POET-style task proposer. family: provides mutate(task, key). score_fn: task -> best student return (injected
    by the archive). returns_fn: (task, key) -> (teacher return, trivial return), default `default_returns`."""

    def __init__(self, family, score_fn: Callable, cfg: FrontierConfig = FrontierConfig(),
                 returns_fn: Callable | None = None):
        self.family, self.score_fn, self.cfg = family, score_fn, cfg
        self.returns_fn = returns_fn or (lambda t, k: default_returns(t, k, cfg.eval_episodes))
        self.pool: list = []
        self.log: list[dict] = []

    def evaluate(self, task, key, parent: str | None = None) -> Proposal:
        """Measure a task and apply the minimal criterion (no novelty ranking)."""
        tr, triv = self.returns_fn(task, key)
        st = float(self.score_fn(task))
        gap = tr - triv
        if not np.isfinite(gap) or gap < self.cfg.margin:
            return Proposal(task, parent, tr, triv, st, float("nan"), False, "degenerate: teacher not above trivial")
        p = (st - triv) / gap
        if p > self.cfg.hi:
            return Proposal(task, parent, tr, triv, st, p, False, "trivial: already solved")
        if p < self.cfg.lo:
            return Proposal(task, parent, tr, triv, st, p, False, "impossible: no progress")
        return Proposal(task, parent, tr, triv, st, p, True, "frontier")

    def propose(self, parents, key, n_accept: int | None = None) -> list[Proposal]:
        """Mutate each parent children_per_parent times, evaluate, and accept (by novelty against the pool) up to
        n_accept children that pass the minimal criterion. Returns every proposal; accepted ones join the pool."""
        props, seen = [], {t.uid for t in self.pool} | {t.uid for t in parents}
        for i, parent in enumerate(parents):
            for j in range(self.cfg.children_per_parent):
                k = jax.random.fold_in(jax.random.fold_in(as_key(key), i), j)
                child = self.family.mutate(parent, k)
                if child.uid in seen:
                    props.append(Proposal(child, parent.uid, float("nan"), float("nan"), float("nan"), float("nan"),
                                          False, "duplicate"))
                    continue
                seen.add(child.uid)
                props.append(self.evaluate(child, jax.random.fold_in(k, 7), parent.uid))
        feats = np.stack([tag_features(t.tags) for t in self.pool]) if self.pool else np.zeros((0, len(AXES)))
        ok = [p for p in props if p.accepted]
        for p in ok:
            p.novelty = novelty(p.task.tags, feats, self.cfg.novelty_k)
        ok.sort(key=lambda p: -p.novelty)
        if n_accept is not None:
            for p in ok[n_accept:]:
                p.accepted, p.reason = False, "not novel enough"
            ok = ok[:n_accept]
        self.pool.extend(p.task for p in ok)
        self.log.extend(p.record() for p in props)
        return props

    def seed(self, tasks, key) -> list[Proposal]:
        """Evaluate initial tasks; those meeting the minimal criterion form the starting pool."""
        props = [self.evaluate(t, jax.random.fold_in(as_key(key), i)) for i, t in enumerate(tasks)]
        self.pool.extend(p.task for p in props if p.accepted)
        self.log.extend(p.record() for p in props)
        return props

    def step(self, key, n_parents: int = 4, n_accept: int | None = None) -> list[Proposal]:
        """One generation: pick parents uniformly from the pool and propose."""
        if not self.pool:
            return []
        rng = np_rng(key, 9)
        idx = rng.choice(len(self.pool), size=min(n_parents, len(self.pool)), replace=False)
        return self.propose([self.pool[i] for i in idx], key, n_accept)
