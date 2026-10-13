"""Frozen teachers for body tasks: a compact ARS V2-t (Mania, Guy & Recht 2018) linear policy, trained once per
physics hash on the task's physical view (identity interface, full velocities) and cached on disk.

Policy: c = clip(W (o - mu) / sd, -1, 1) on the physical observation. Per iteration: N Gaussian directions,
antithetic rollouts W +/- nu*delta, keep the top b directions by max(r+, r-), step along them scaled by the std of
the used returns; running state statistics from every live state seen (as evocirc.ars, which is tied to its own
env registry; this one runs on any Task). Training returns subtract a per-step `shift` (survival bonus).

Cache: <cache>/<key>.npz (W, mu, sd) + <key>.json (config, physics id, eval return, zero-policy return), with
key = stable_hash(physics_id, ARS config). cache defaults to $GROWN_TEACHER_CACHE or ~/.cache/grown/teachers.

TODO: short PPO teachers (`train_ppo`) for bodies where a linear policy is too weak.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .base import jsonable, stable_hash, tree_where

SHIFT = {"gym_hopper": 1.0, "gym_humanoid": 5.0}


@dataclass(frozen=True)
class ARSConfig:
    iterations: int = 100
    directions: int = 16
    top: int = 8
    step_size: float = 0.02
    noise: float = 0.03
    shift: float | None = None        # None: by objective (SHIFT), else 0
    horizon: int | None = None        # training episode length (None: the task's horizon)
    eval_episodes: int = 8
    seed: int = 0


@dataclass
class LinearPolicy:
    W: np.ndarray          # [A, D]
    mu: np.ndarray         # [D]
    sd: np.ndarray         # [D]
    info: dict = dataclasses.field(default_factory=dict)

    def __call__(self, o):
        return jnp.clip(jnp.asarray(self.W) @ ((o - jnp.asarray(self.mu)) / jnp.asarray(self.sd)), -1.0, 1.0)

    def save(self, stem):
        np.savez(stem + ".npz", W=self.W, mu=self.mu, sd=self.sd)
        with open(stem + ".json", "w") as f:
            json.dump(jsonable(self.info), f, indent=1)

    @classmethod
    def load(cls, stem):
        z = np.load(stem + ".npz")
        info = json.load(open(stem + ".json")) if os.path.exists(stem + ".json") else {}
        return cls(z["W"], z["mu"], z["sd"], info)


def _episode_fn(task, horizon):
    def episode(W, mu, sd, key, shift):
        s = task.reset(key)

        def body(c, _):
            s, alive, ret, sret, n, s1, s2 = c
            o = task.physical_obs(s)
            a = jnp.clip(W @ ((o - mu) / sd), -1.0, 1.0)
            s_new, _, r, done = task.step(s, a)
            w = alive.astype(jnp.float32)
            return (tree_where(alive, s_new, s), alive & ~done, ret + r * w, sret + (r - shift) * w, n + w,
                    s1 + o * w, s2 + o * o * w), None

        D = mu.shape[0]
        c0 = (s, jnp.bool_(True), 0.0, 0.0, 0.0, jnp.zeros(D), jnp.zeros(D))
        (_, _, ret, sret, n, s1, s2), _ = jax.lax.scan(body, c0, None, length=horizon)
        return ret, sret, n, s1, s2

    return jax.jit(jax.vmap(episode, in_axes=(0, None, None, 0, None)))


def train_ars(task, cfg: ARSConfig = ARSConfig(), log=None) -> LinearPolicy:
    """Train on task.physical() (identity interface, velocities visible). Returns the policy with an info dict."""
    phys = task.physical() if hasattr(task, "physical") else task
    D, A, N, b = phys.obs_dim, phys.act_dim, cfg.directions, min(cfg.top, cfg.directions)
    H = cfg.horizon or phys.horizon
    shift = cfg.shift if cfg.shift is not None else SHIFT.get(phys.params.get("objective"), 0.0)
    ep = _episode_fn(phys, H)
    key = jax.random.PRNGKey(cfg.seed)
    W = jnp.zeros((A, D))
    n, s1, s2 = 0.0, np.zeros(D), np.zeros(D)
    mu, sd = jnp.zeros(D), jnp.ones(D)
    t0, steps = time.time(), 0.0

    def evaluate(W, k):
        Ws = jnp.broadcast_to(W, (cfg.eval_episodes, A, D))
        ret, *_ = ep(Ws, mu, sd, jax.random.split(k, cfg.eval_episodes), 0.0)
        return float(ret.mean())

    key, k0 = jax.random.split(key)
    zero_return = evaluate(jnp.zeros((A, D)), k0)
    hist = []
    for it in range(cfg.iterations):
        key, kd, kr = jax.random.split(key, 3)
        delta = jax.random.normal(kd, (N, A, D))
        Ws = jnp.concatenate([W + cfg.noise * delta, W - cfg.noise * delta])
        kk = jax.random.split(kr, N)
        _, sret, dn, ds1, ds2 = ep(Ws, mu, sd, jnp.concatenate([kk, kk]), shift)   # common random numbers
        rp, rm = sret[:N], sret[N:]
        idx = jnp.argsort(-jnp.maximum(rp, rm))[:b]
        used = jnp.concatenate([rp[idx], rm[idx]])
        W = W + cfg.step_size / (b * (used.std() + 1e-8)) * jnp.einsum("n,nad->ad", rp[idx] - rm[idx], delta[idx])
        n, s1, s2 = n + float(dn.sum()), s1 + np.asarray(ds1.sum(0)), s2 + np.asarray(ds2.sum(0))
        steps += float(dn.sum())
        mu = jnp.asarray(s1 / n, jnp.float32)
        var = np.maximum(s2 / n - (s1 / n) ** 2, 0.0)
        sd = jnp.asarray(np.where(var < 1e-8, 1.0, np.sqrt(var)), jnp.float32)
        hist.append(float(jnp.maximum(rp, rm).mean()))
        if log:
            log(dict(it=it, train_return=hist[-1], env_steps=steps))
    key, ke = jax.random.split(key)
    pol = LinearPolicy(np.asarray(W, np.float32), np.asarray(mu, np.float32), np.asarray(sd, np.float32))
    pol.info = dict(config=dataclasses.asdict(cfg), physics_id=phys.physics_id, task=phys.name,
                    eval_return=evaluate(W, ke), zero_return=zero_return, env_steps=steps,
                    train_curve=hist, seconds=round(time.time() - t0, 2))
    return pol


def train_ppo(task, **kw):
    """TODO: a short PPO teacher (nonlinear) for bodies a linear ARS policy cannot handle."""
    raise NotImplementedError("PPO teachers are not implemented yet; use train_ars")


def cache_dir(path: str | None = None) -> str:
    d = path or os.environ.get("GROWN_TEACHER_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "grown",
                                                                      "teachers")
    os.makedirs(d, exist_ok=True)
    return d


def teacher_key(task, cfg: ARSConfig) -> str:
    return stable_hash(dict(physics=task.physics_id, ars=dataclasses.asdict(cfg)))


def ensure_teacher(task, cfg: ARSConfig = ARSConfig(), cache: str | None = None, train: bool = True,
                   log=None) -> LinearPolicy:
    """Load the cached teacher for task's physics (or train and cache it) and attach it to the task. Wrappers
    (sensorimotor) are unwrapped: the teacher belongs to the innermost body."""
    while hasattr(task, "inner"):
        task = task.inner
    stem = os.path.join(cache_dir(cache), teacher_key(task, cfg))
    if os.path.exists(stem + ".npz"):
        pol = LinearPolicy.load(stem)
        pol.info["cached"] = True
    elif train:
        pol = train_ars(task, cfg, log)
        tmp = stem + f".tmp{os.getpid()}"
        pol.save(tmp)
        os.replace(tmp + ".npz", stem + ".npz")       # atomic: concurrent trainers of one body cannot tear a file
        os.replace(tmp + ".json", stem + ".json")
        pol.info["cached"] = False
    else:
        raise FileNotFoundError(f"no cached teacher for {task.name} at {stem}")
    task.attach_teacher(pol)
    return pol
