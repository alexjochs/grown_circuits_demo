"""Dreams: random circuits simulated on the stand-in substrate, for inverse-model pretraining of the encoder.

`dream_pairs` runs random walks of uniform edits from a seed population (random gate networks, or given seeds;
when the operator's library has active modules, INSERT_MODULE moves put library tokens into the dreams), evaluates
the circuits on `grown.sim.SyncSubstrate` under a random input stream drawn per batch, and returns (circuit,
behaviour signature) pairs. Everything derives from one numpy seed, so the pairs are reproducible.

The dream signature is task-free (no teacher): per output, the fraction of time high and the flip rate; per
(output, input, lag in 0..LAGS-1) the agreement rate P(out_j(s) == in_i(s - lag)) minus one half; and the gate
toggle rate. Its size is n_out * (2 + n_in * LAGS) + 1.

`pretrain` fits the dream head (G -> signature, MSE on standardized targets) together with the encoder. The
editor's behaviour-signature input is switched off during this (sig_on = 0), so the encoder has to read the
behaviour off the structure. Fine-tuning on hindsight edits then starts from this encoder.

`generate` (behaviour -> circuit) is the generator head of project.md and is a stub: see its docstring.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..circuit.graph import InvalidCircuit, random_circuit
from ..sim import SyncSubstrate
from . import featurize as F
from . import model as Mdl
from .train import adam_init, adam_update

LAGS = 3


def dream_dim(n_in: int, n_out: int) -> int:
    return n_out * (2 + n_in * LAGS) + 1


def dream_signature(res, x) -> np.ndarray:
    """res: EvalResult for B circuits; x: inputs uint8 [S, n_in] (shared). -> float32 [B, dream_dim]."""
    out = res.outputs.astype(np.float32)                         # [B, S, n_out]
    B, S, n_out = out.shape
    xf = np.asarray(x, np.float32)
    feats = [res.high.mean(1)]
    flips = (np.abs(np.diff(out, axis=1)).mean(1) if S > 1 else np.zeros((B, n_out), np.float32))
    feats.append(flips)
    for lag in range(LAGS):
        xs = xf[:S - lag] if lag else xf                          # in(s - lag) aligned with out(s), s >= lag
        o = out[:, lag:]
        agree = (o[:, :, :, None] == xs[None, :, None, :]).mean(1) - 0.5   # [B, n_out, n_in]
        feats.append(agree.reshape(B, -1))
    feats.append(res.toggle_rate[:, None])
    return np.concatenate([f.reshape(B, -1) for f in feats], 1).astype(np.float32)


def dream_pairs(operator, n: int, n_in: int, n_out: int, seed: int = 0, seeds=None, steps: int = 32, K: int = 4,
                walk=(1, 8), batch: int = 64, pool_max: int = 4096, init_nodes: int = 8, substrate=None):
    """-> (circuits, signatures [n, dream_dim], inputs per batch). Reproducible from `seed`."""
    rng = np.random.default_rng(seed)
    sub = substrate or SyncSubstrate()
    pool = list(seeds) if seeds else [random_circuit(n_in, n_out, init_nodes, rng) for _ in range(64)]
    circuits, sigs, streams = [], [], []
    while len(circuits) < n:
        B = min(batch, n - len(circuits))
        chunk = []
        while len(chunk) < B:
            c = pool[int(rng.integers(len(pool)))]
            for _ in range(int(rng.integers(walk[0], walk[1] + 1))):
                try:
                    c, _, _ = operator.mutate(c, rng)
                except InvalidCircuit:
                    break
            chunk.append(c)
        x = rng.integers(0, 2, (steps, n_in)).astype(np.uint8)
        res = sub.evaluate(chunk, x, K, operator.library)
        sigs.append(dream_signature(res, x))
        streams.append(x)
        circuits += chunk
        for c in chunk:
            if len(pool) < pool_max:
                pool.append(c)
            else:
                pool[int(rng.integers(pool_max))] = c
    return circuits, np.concatenate(sigs)[:n], streams


def pretrain(p, cfg: Mdl.EditorConfig, operator, circuits, sigs, steps: int = 1000, batch: int = 64,
             lr: float = 1e-3, seed: int = 0, holdout: float = 0.1, log_fn=print, every: int = 250):
    """Inverse-model pretraining (circuit -> dream signature). Returns (params, history)."""
    if not cfg.dream_dim or cfg.dream_dim != sigs.shape[1]:
        raise ValueError(f"config dream_dim {cfg.dream_dim} != signature size {sigs.shape[1]}")
    ctxs = [F.featurize(c, operator) for c in circuits]
    dims = F.padded_dims([c.arrays for c in ctxs])
    data = F.collate([c.arrays for c in ctxs], dims, cfg.sig_dim)
    mu, sd = sigs.mean(0), sigs.std(0) + 1e-3
    y = (sigs - mu) / sd
    rng = np.random.default_rng(seed)
    n = len(circuits)
    te = rng.random(n) < holdout
    tr_ix, te_ix = np.nonzero(~te)[0], np.nonzero(te)[0]

    def loss(p, f, y):
        pred = jax.vmap(lambda f: Mdl.dream_pred(p, f, cfg))(f)
        return ((pred - y) ** 2).mean()

    @jax.jit
    def step(p, st, f, y, lr):
        l, g = jax.value_and_grad(loss)(p, f, y)
        p, st, _ = adam_update(p, g, st, lr)
        return p, st, l

    ev = jax.jit(loss)
    st = adam_init(p)
    hist = []
    t0 = time.time()
    run = []

    def held_out(p):
        ls = []
        for i in range(0, len(te_ix) - batch + 1, batch):
            ix = te_ix[i:i + batch]
            ls.append(float(ev(p, {k: v[ix] for k, v in data.items()}, y[ix])))
        return float(np.mean(ls)) if ls else float("nan")

    for s in range(1, steps + 1):
        ix = rng.choice(tr_ix, batch)
        f = {k: v[ix] for k, v in data.items()}
        cur = lr * 0.5 * (1 + np.cos(np.pi * s / steps))
        p, st, l = step(p, st, f, jnp.asarray(y[ix]), np.float32(cur))
        run.append(float(l))
        if s % every == 0 or s == steps:
            h = {"step": s, "train_mse": float(np.mean(run)), "heldout_mse": held_out(p)}
            run = []
            hist.append(h)
            log_fn(f"dream step {s} train mse {h['train_mse']:.3f} held-out mse {h['heldout_mse']:.3f} "
                   f"(1.0 = predicting the mean) {time.time() - t0:.0f}s")
    return p, hist


def generate(params, cfg, signature, n_in: int, n_out: int, rng=None):
    """Generator head (behaviour -> circuit). STUB, not implemented.

    Planned shape (project.md, "Learner"): the same transformer decodes a circuit token by token conditioned on a
    target behaviour signature (and task embedding): each step emits a token type (gate / delay / wire / library
    module), its parameters with the editor's value heads, and its pin pointers over the ports emitted so far (and
    forward pointers for feedback, resolved at the end), then the output pointers. It is trained on dream pairs
    in the inverse direction (signature -> circuit, teacher-forced log-likelihood of the dreamed circuits in
    canonical node order, so isomorphic circuits share one target sequence) and later on archive elites. Until it
    exists, REGENERATE moves stay no-ops unless an operator is given a regenerator.
    """
    raise NotImplementedError("the generator head (behaviour -> circuit) is not implemented yet; see docstring")
