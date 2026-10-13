"""Hindsight training of the editor from the search's own edit logs.

Data. For each log directory (an `EditLog` plus `circuits.npz` from a `CircuitStore`, and optionally
`library.json`), every row with a move is turned into an example (parent circuit, parent signature, task, move,
fate). Parents come from the store by hash; each example is verified by replaying the move and comparing the
child's canonical hash with the logged one (mismatches, e.g. an isomorphic relabeling stored under the same hash,
are dropped and counted). Moves outside the model's support (featurize.Unrepresentable) are dropped and counted.

Objective (the choice and why). Accepted edits are positives with weights by fate: ELITE_NEW 1.0, ELITE_REPLACE
0.75, SURVIVED 0.5. CULLED and INVALID are negatives.

  L = - sum_pos w_i log q(m_i | c_i) / sum_pos w_i
      + lam * mean_neg [ log q_c(m_j | c_j) - log u_c(m_j | c_j) ]_+

The first term is weighted maximum likelihood of accepted moves over the full factorized move. When the logged
moves were drawn from the uniform operator u, its optimum is q*(m | c) proportional to u(m | c) P(accept | m, c):
the uniform prior reweighted by acceptance. Its acceptance rate, E_u[P^2] / E_u[P], is at least the uniform's
E_u[P] (Cauchy-Schwarz), so an accurate model provably accepts more, and q* > 0 wherever u > 0 and P > 0. The
second term is a bounded rejection hinge on the coarse (type, location) marginal q_c: a culled move costs only
when the model puts more mass on its coarse choice than the uniform operator u_c did. It never pushes below the
uniform reference, so it cannot drive probabilities to zero (unlikelihood on the full move would: a single
move's probability is tiny and -log(1 - q) carries no signal, while -log q diverges), and it agrees with q*,
which sits below u exactly where acceptance is below average. lam is small (0.1 by default) because negatives
outnumber positives five to ten to one. u_c is exact for every coarse choice except the DUPLICATE / REGENERATE
anchors, approximated as uniform over nodes.

Reported on held-out data (whole log directories when there are several, else 10% of parents by hash): the mean
log-likelihood of accepted moves (nats per move), the coarse lift E[log q_c - log u_c] on positives and on
negatives, and the proposal's entropy (Monte Carlo over the full move, exact over the coarse marginal).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..circuit.graph import InvalidCircuit
from ..circuit.library import Library
from ..circuit.moves import Fate, Move, MoveType, OperatorConfig, UniformOperator, apply_move
from ..log import EditLog, decode_hashes
from . import featurize as F
from . import model as Mdl
from .store import CircuitStore

FATE_WEIGHT = {Fate.ELITE_NEW: 1.0, Fate.ELITE_REPLACE: 0.75, Fate.SURVIVED: 0.5}
NEG_FATES = (Fate.CULLED, Fate.INVALID)


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_pos: int = 48
    batch_neg: int = 16
    lr: float = 1e-3
    warmup: int = 100
    weight_decay: float = 1e-4
    clip: float = 1.0
    neg_weight: float = 0.1
    holdout: float = 0.1
    eval_every: int = 500
    seed: int = 0


# --- data ---------------------------------------------------------------------------------------------------------

def read_log(log_dir: str, operator=None, verify: bool = True, max_rows: int | None = None):
    """-> (examples, counts). Example: dict(parent=Circuit, sig, task, move, fate, generation, source, group)."""
    store = CircuitStore.load(log_dir)
    log = EditLog(os.path.join(log_dir, "edits") if os.path.isdir(os.path.join(log_dir, "edits")) else log_dir)
    lib = operator.library if operator is not None else None
    cnt = {"rows": 0, "no_move": 0, "no_parent": 0, "mismatch": 0}
    out = []
    for b in log.scan():
        ph, chh = decode_hashes(b, "parent"), decode_hashes(b, "child")
        off, vals = b["payload_offsets"], b["payload_values"]
        for i in range(len(b["move"])):
            cnt["rows"] += 1
            if b["move"][i] < 0:
                cnt["no_move"] += 1
                continue
            parent = store.get(ph[i])
            if parent is None:
                cnt["no_parent"] += 1
                continue
            mv = Move(int(b["move"][i]), int(b["node"][i]), int(b["pin"][i]), int(b["a"][i]), int(b["b"][i]),
                      tuple(int(x) for x in vals[off[i]:off[i + 1]]))
            if verify and any(chh[i]):
                try:
                    ok = apply_move(parent, mv, lib).hash == chh[i]
                except (InvalidCircuit, IndexError, ValueError, KeyError):
                    ok = False
                if not ok:
                    cnt["mismatch"] += 1
                    continue
            out.append(dict(parent=parent, sig=b["parent_sig"][i], task=int(b["task"][i]), move=mv,
                            fate=int(b["fate"][i]), generation=int(b["generation"][i]), source=int(b["source"][i]),
                            group=log_dir))
            if max_rows and len(out) >= max_rows:
                return out, cnt
    return out, cnt


def operator_for(log_dir: str, op_cfg: OperatorConfig | None = None) -> UniformOperator:
    lib_path = os.path.join(log_dir, "library.json")
    lib = Library.load(lib_path) if os.path.exists(lib_path) else Library()
    return UniformOperator(op_cfg or OperatorConfig(), lib)


class Dataset:
    """Featurized examples: per distinct parent, padded arrays (shared); per example, action arrays."""

    def __init__(self, examples, operator, sig_dim: int, dims: dict | None = None):
        self.op = operator
        self.sig_dim = sig_dim
        ctxs, parent_ix, acts, keep = {}, [], [], []
        self.skipped = 0
        for j, ex in enumerate(examples):
            h = ex["parent"].hash
            if h not in ctxs:
                ctxs[h] = F.featurize(ex["parent"], operator)
            ctx = ctxs[h]
            try:
                acts.append(F.encode_move(ctx, ex["move"]))
            except (F.Unrepresentable, KeyError, IndexError):
                self.skipped += 1
                continue
            parent_ix.append(h)
            keep.append(j)
        self.examples = [examples[j] for j in keep]
        hs = list(ctxs)
        self.ctxs = [ctxs[h] for h in hs]
        pos = {h: i for i, h in enumerate(hs)}
        self.parent = np.array([pos[h] for h in parent_ix], np.int64)
        self.dims = dims or F.padded_dims([c.arrays for c in self.ctxs])
        P = [{k: v for k, v in c.arrays.items() if k not in ("sig", "sig_on", "task")} for c in self.ctxs]
        self.parents = F.collate(P, self.dims, sig_dim) if P else {}
        self.acts = F.collate(acts, self.dims, sig_dim) if acts else {}
        self.sig = np.zeros((len(self.examples), sig_dim), np.float32)
        self.sig_on = np.zeros(len(self.examples), np.float32)
        for i, ex in enumerate(self.examples):
            s = np.asarray(ex["sig"], np.float32).reshape(-1)[:sig_dim]
            if s.size and not np.isnan(s).all():
                self.sig[i, :len(s)] = np.nan_to_num(s)
                self.sig_on[i] = 1.0
        self.task = np.array([ex["task"] for ex in self.examples], np.int32)
        self.fate = np.array([ex["fate"] for ex in self.examples], np.int8)
        self.weight = np.array([FATE_WEIGHT.get(int(f), 0.0) for f in self.fate], np.float32)
        self.neg = np.isin(self.fate, NEG_FATES)
        self.type = np.array([ex["move"].type for ex in self.examples], np.int8)

    def __len__(self):
        return len(self.examples)

    def batch(self, idx) -> dict:
        idx = np.asarray(idx)
        f = {k: v[self.parent[idx]] for k, v in self.parents.items()}
        f.update({k: v[idx] for k, v in self.acts.items()})
        f["sig"], f["sig_on"], f["task"] = self.sig[idx], self.sig_on[idx], self.task[idx]
        f["w"] = self.weight[idx]
        f["neg"] = self.neg[idx].astype(np.float32)
        return f

    def subset(self, mask) -> "Dataset":
        d = object.__new__(Dataset)
        d.__dict__.update(self.__dict__)
        ix = np.nonzero(mask)[0]
        d.examples = [self.examples[i] for i in ix]
        d.parent = self.parent[ix]
        d.acts = {k: v[ix] for k, v in self.acts.items()}
        for k in ("sig", "sig_on", "task", "fate", "weight", "neg", "type"):
            setattr(d, k, getattr(self, k)[ix])
        return d


def split(ds: Dataset, holdout: float, seed: int = 0):
    groups = sorted({ex["group"] for ex in ds.examples})
    if len(groups) > 1:
        held = groups[-max(1, int(round(len(groups) * holdout))):]
        mask = np.array([ex["group"] in held for ex in ds.examples])
    else:
        rng = np.random.default_rng(seed)
        u = rng.random(len(ds.ctxs))
        mask = u[ds.parent] < holdout
    return ds.subset(~mask), ds.subset(mask)


# --- optimisation -------------------------------------------------------------------------------------------------

def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return {"m": z, "v": jax.tree_util.tree_map(jnp.zeros_like, p), "t": jnp.zeros((), jnp.int32)}


def adam_update(p, g, st, lr, wd=0.0, b1=0.9, b2=0.999, clip=1.0):
    gn = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(g)))
    scale = jnp.minimum(1.0, clip / (gn + 1e-9))
    g = jax.tree_util.tree_map(lambda x: x * scale, g)
    t = st["t"] + 1
    m = jax.tree_util.tree_map(lambda m, x: b1 * m + (1 - b1) * x, st["m"], g)
    v = jax.tree_util.tree_map(lambda v, x: b2 * v + (1 - b2) * x * x, st["v"], g)
    c1, c2 = 1 - b1 ** t, 1 - b2 ** t
    p = jax.tree_util.tree_map(lambda p, m, v: p - lr * (m / c1 / (jnp.sqrt(v / c2) + 1e-8) + wd * p), p, m, v)
    return p, {"m": m, "v": v, "t": t}, gn


def make_loss(cfg: Mdl.EditorConfig, neg_weight: float):
    def per_ex(p, f):
        return Mdl.move_logprob(p, f, f, cfg)

    def loss(p, f):
        lp, lc, lu = jax.vmap(per_ex, in_axes=(None, 0))(p, f)
        w = f["w"]
        l_pos = -(w * lp).sum() / jnp.maximum(w.sum(), 1e-6)
        l_neg = (f["neg"] * jax.nn.relu(lc - lu)).sum() / jnp.maximum(f["neg"].sum(), 1.0)
        return l_pos + neg_weight * l_neg, (l_pos, l_neg)

    return loss


def evaluate(p, cfg: Mdl.EditorConfig, ds: Dataset, batch: int = 128) -> dict:
    """Held-out metrics: acceptance log-likelihood and coarse lift over uniform, per sign and per move type."""
    fn = jax.jit(jax.vmap(lambda p, f: Mdl.move_logprob(p, f, f, cfg), in_axes=(None, 0)))
    lps, lcs, lus = [], [], []
    for i in range(0, len(ds), batch):
        idx = np.arange(i, min(i + batch, len(ds)))
        pad = np.concatenate([idx, np.full(batch - len(idx), idx[0])])
        lp, lc, lu = (np.asarray(x)[:len(idx)] for x in fn(p, ds.batch(pad)))
        lps.append(lp), lcs.append(lc), lus.append(lu)
    if not lps:
        return {}
    lp, lc, lu = np.concatenate(lps), np.concatenate(lcs), np.concatenate(lus)
    pos = ds.weight > 0
    out = {"n": int(len(ds)), "n_pos": int(pos.sum()), "n_neg": int(ds.neg.sum())}
    if pos.any():
        out["pos_loglik"] = float(np.average(lp[pos], weights=ds.weight[pos]))
        out["pos_loglik_unweighted"] = float(lp[pos].mean())
        out["pos_coarse_lift"] = float((lc - lu)[pos].mean())
        out["pos_coarse_loglik"] = float(lc[pos].mean())
        out["pos_coarse_loglik_uniform"] = float(lu[pos].mean())
    if ds.neg.any():
        out["neg_loglik"] = float(lp[ds.neg].mean())
        out["neg_coarse_lift"] = float((lc - lu)[ds.neg].mean())
    out["by_type"] = {MoveType(t).name: {"pos_loglik": float(lp[pos & (ds.type == t)].mean()),
                                        "pos_lift": float((lc - lu)[pos & (ds.type == t)].mean()),
                                        "n_pos": int((pos & (ds.type == t)).sum())}
                      for t in np.unique(ds.type) if (pos & (ds.type == t)).any()}
    return out


def proposal_entropy(p, cfg, ds: Dataset, n: int = 200, seed: int = 0) -> dict:
    """Entropy instruments of the proposal on up to n held-out parents (one sample each)."""
    from .proposal import LearnedProposal
    rng = np.random.default_rng(seed)
    prop = LearnedProposal(p, cfg, ds.op, cache_size=n + 1)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False) if len(ds) else []
    for i in idx:
        ex = ds.examples[i]
        prop._entry(ex["parent"])
        prop.propose(ex["parent"], rng)
    return prop.pop_stats()


def train(p, cfg: Mdl.EditorConfig, tr: Dataset, te: Dataset | None, tc: TrainConfig, log_fn=print,
          ckpt: str | None = None, keep_best: bool = True) -> tuple:
    """Train; with held-out data and keep_best, returns the parameters of the best held-out acceptance
    log-likelihood among the evaluations (early stopping). hist: one dict per evaluation."""
    rng = np.random.default_rng(tc.seed)
    loss = make_loss(cfg, tc.neg_weight)

    @jax.jit
    def step(p, st, f, lr):
        (l, (lp, ln_)), g = jax.value_and_grad(loss, has_aux=True)(p, f)
        p, st, gn = adam_update(p, g, st, lr, tc.weight_decay, clip=tc.clip)
        return p, st, l, lp, ln_, gn

    st = adam_init(p)
    pos = np.nonzero(tr.weight > 0)[0]
    neg = np.nonzero(tr.neg)[0]
    hist = []
    best = (-np.inf, p, 0)
    if te is not None and len(te):
        ev = evaluate(p, cfg, te)
        ev["step"] = 0
        hist.append(ev)
        best = (ev.get("pos_loglik", -np.inf), p, 0)
        log_fn(f"step 0 held-out pos loglik {ev.get('pos_loglik', float('nan')):.3f}  "
               f"lift pos {ev.get('pos_coarse_lift', float('nan')):+.3f} neg {ev.get('neg_coarse_lift', float('nan')):+.3f}")
    t0 = time.time()
    run = []
    for s in range(1, tc.steps + 1):
        bp = rng.choice(pos, tc.batch_pos) if len(pos) else np.zeros(0, np.int64)
        bn = rng.choice(neg, tc.batch_neg) if len(neg) else rng.choice(pos, tc.batch_neg)
        f = tr.batch(np.concatenate([bp, bn]))
        lr = tc.lr * min(1.0, s / max(tc.warmup, 1)) * 0.5 * (1 + np.cos(np.pi * min(s / tc.steps, 1.0)))
        p, st, l, lp, ln_, gn = step(p, st, f, np.float32(lr))
        run.append((float(l), float(lp), float(ln_), float(gn)))
        if s % tc.eval_every == 0 or s == tc.steps:
            r = np.mean(run, 0)
            run = []
            msg = f"step {s} loss {r[0]:.3f} (pos {r[1]:.3f} neg {r[2]:.3f}) |g| {r[3]:.2f} {time.time() - t0:.0f}s"
            if te is not None and len(te):
                ev = evaluate(p, cfg, te)
                ev.update(step=s, train_loss=float(r[0]))
                hist.append(ev)
                if ev.get("pos_loglik", -np.inf) > best[0]:
                    best = (ev["pos_loglik"], p, s)
                msg += (f"  held-out pos loglik {ev.get('pos_loglik', float('nan')):.3f} lift pos "
                        f"{ev.get('pos_coarse_lift', float('nan')):+.3f} neg {ev.get('neg_coarse_lift', float('nan')):+.3f}")
            log_fn(msg)
            if ckpt:
                Mdl.save_params(ckpt, best[1] if keep_best else p, cfg, {"train": asdict(tc), "step": best[2]})
    if keep_best and te is not None and len(te):
        log_fn(f"keeping step {best[2]} (best held-out pos loglik {best[0]:.3f})")
        hist.append({"selected_step": best[2], "pos_loglik": best[0]})
        return best[1], hist
    return p, hist


def build(log_dirs, op_cfg: OperatorConfig | None = None, sig_dim: int = 4, verify: bool = True,
          max_rows: int | None = None, log_fn=print):
    """Read logs, build the operator (library from the first log dir) and the dataset."""
    op = operator_for(log_dirs[0], op_cfg)
    exs, counts = [], {}
    for d in log_dirs:
        e, c = read_log(d, op, verify, max_rows)
        exs += e
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
    t0 = time.time()
    ds = Dataset(exs, op, sig_dim)
    counts["unrepresentable"] = ds.skipped
    counts["examples"] = len(ds)
    counts["parents"] = len(ds.ctxs)
    counts["positives"] = int((ds.weight > 0).sum())
    counts["negatives"] = int(ds.neg.sum())
    log_fn(f"dataset {json.dumps(counts)} dims {ds.dims} ({time.time() - t0:.0f}s featurizing)")
    return ds, op, counts
