"""Collapse instruments for a move proposal.

Exact, per circuit, over the coarse marginal (move type, location), where the location is the node pointer for
node-located types, the pin for REWIRE, the splice (NONE or pin) for ADD_NODE and the module for INSERT_MODULE:
  H_type          entropy of the type distribution (nats)
  H_loc[type]     entropy of the location given the type
  H_coarse        entropy of the joint (type, location)
  KL_coarse       KL(proposal || uniform operator) on that joint
  top{1,5,10}     probability mass on the k most likely (type, location) pairs
Monte Carlo, per sampled move (exact log-probabilities of the samples):
  H_mc            -mean log q(move): an unbiased estimate of the entropy of the full move distribution
  H_mc[type]      the same restricted to samples of each type (entropy of the move given its type, plus -log p(type))
"""
from __future__ import annotations

import numpy as np

from ..circuit.moves import MoveType
from . import featurize as F

TOPK = (1, 5, 10)


def _norm(lp, mask):
    if not np.any(mask):
        return np.zeros(len(mask))
    w = np.where(mask, np.exp(np.asarray(lp, np.float64) - np.max(np.where(mask, lp, -np.inf))), 0.0)
    s = w.sum()
    return w / s if s > 0 else w


def _H(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def coarse_metrics(ctx: F.Ctx, tab: dict, op) -> dict:
    A = ctx.arrays
    tm = A["type_mask"]
    pt = _norm(tab["type"], tm)
    w = np.where(tm, A["type_w"], 0.0)
    ut = w / w.sum() if w.sum() > 0 else w
    q_joint, u_joint, h_loc = [], [], {}
    for t in MoveType:
        if not tm[t]:
            continue
        if t in F.PTR:
            mask = A["ptr_mask"][F.PTR[t]]
            lp = tab["node"][F.PTR[t]]
            u = mask / mask.sum()
        elif t == MoveType.REWIRE:
            mask = A["pin_valid"].copy()
            mask[0] = False
            lp = tab["rw_pin"]
            u = mask / mask.sum()
        elif t == MoveType.ADD_NODE:
            mask = A["pin_valid"]
            lp = tab["add_splice"]
            n = mask.sum() - 1
            u = np.where(mask, A["p_splice"] / max(n, 1), 0.0)
            u[0] = 1 - A["p_splice"] if n else 1.0
        else:  # INSERT_MODULE: location = module
            mask = A["mod_valid"]
            lp = tab["ins_mod"]
            u = mask / mask.sum()
        q = _norm(lp, mask)
        h_loc[t.name] = _H(q)
        q_joint.append(pt[t] * q)
        u_joint.append(ut[t] * u)
    q = np.concatenate(q_joint) if q_joint else np.zeros(1)
    u = np.concatenate(u_joint) if u_joint else np.ones(1)
    nz = q > 0
    kl = float((q[nz] * (np.log(q[nz]) - np.log(np.maximum(u[nz], 1e-300)))).sum())
    qs = np.sort(q)[::-1]
    out = {"H_type": _H(pt), "H_coarse": _H(q), "H_coarse_uniform": _H(u), "KL_coarse": kl,
           "p_type": pt, "H_loc": h_loc}
    for k in TOPK:
        out[f"top{k}"] = float(qs[:k].sum())
    return out


def aggregate(samples, coarse_log) -> dict:
    """samples: list of (type, log q); coarse_log: list of coarse_metrics dicts (one per propose call)."""
    st: dict = {"n": len(samples)}
    if samples:
        t = np.array([s[0] for s in samples])
        lp = np.array([s[1] for s in samples])
        st["H_mc"] = float(-lp.mean())
        st["H_mc_by_type"] = {MoveType(k).name: float(-lp[t == k].mean()) for k in np.unique(t)}
        st["type_freq"] = {MoveType(k).name: float((t == k).mean()) for k in np.unique(t)}
    if coarse_log:
        for k in ("H_type", "H_coarse", "H_coarse_uniform", "KL_coarse") + tuple(f"top{k}" for k in TOPK):
            st[k] = float(np.mean([c[k] for c in coarse_log]))
        names = sorted({n for c in coarse_log for n in c["H_loc"]})
        st["H_loc"] = {n: float(np.mean([c["H_loc"][n] for c in coarse_log if n in c["H_loc"]])) for n in names}
        st["p_type"] = {MoveType(i).name: float(v) for i, v in
                        enumerate(np.mean([c["p_type"] for c in coarse_log], 0)) if v > 0}
    return st
