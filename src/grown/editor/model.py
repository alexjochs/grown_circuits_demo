"""The editor network, in plain JAX (a parameter pytree of arrays; no framework dependency).

Encoder. Tokens (CLS, primary inputs, nodes; see featurize.py) are embedded from their feature vectors and ids. The
CLS token additionally receives an MLP of the behaviour signature and a task embedding. Each of `layers` blocks
applies, with pre-layer-norm residuals:
  1. message passing over the wiring: per-pin weights for fan-in messages (src -> dst through pin p) and one
     mean-aggregated fan-out message (consumers -> src);
  2. multi-head self-attention over all tokens with a learned per-head bias for the relation of each token pair
     (none, i reads j, j reads i, both, self);
  3. an MLP.
G = the final CLS state. Pin slots get an embedding from their consumer, source, pin position and output index
(slot 0 = NONE); port slots from their token and port position (slot 0 = NEW, the node being added).

Heads. One function per factor of the move distribution (featurize.py lists the factorization). Every head is
written for one conditioning choice and used two ways: gathered at a logged move's indices for training
(`move_logprob`), and vmapped over all choices for sampling (`tables`). Both paths therefore compute identical
log-probabilities. Pointers are scaled dot products between a query and keys (tokens, pins, ports, modules).

Dream head. An MLP from G to a behaviour signature (inverse-model pretraining on dreamed circuits).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..circuit.moves import MoveType, N_MOVES
from . import featurize as F

NEG = -1e9


@dataclass
class EditorConfig:
    d: int = 64
    layers: int = 3
    heads: int = 4
    sig_dim: int = 4
    n_tasks: int = 64
    dream_dim: int = 0            # 0: no dream head
    seed: int = 0

    def to_dict(self):
        return asdict(self)


# --- parameters ---------------------------------------------------------------------------------------------------

class _Init:
    def __init__(self, key):
        self.key = key

    def next(self):
        self.key, k = jax.random.split(self.key)
        return k

    def dense(self, din, dout, scale=1.0):
        w = jax.random.normal(self.next(), (din, dout)) * (scale / math.sqrt(din))
        return {"w": w.astype(jnp.float32), "b": jnp.zeros(dout, jnp.float32)}

    def emb(self, n, d, scale=0.02):
        return (jax.random.normal(self.next(), (n, d)) * scale).astype(jnp.float32)

    def mlp(self, din, dh, dout, out_scale=1.0):
        return {"l1": self.dense(din, dh), "l2": self.dense(dh, dout, out_scale)}

    def ln(self, d):
        return {"g": jnp.ones(d, jnp.float32), "b": jnp.zeros(d, jnp.float32)}


def init_params(cfg: EditorConfig) -> dict:
    r = _Init(jax.random.PRNGKey(cfg.seed))
    d, H = cfg.d, cfg.heads
    p = {
        "tok_in": r.dense(F.N_FEAT, d),
        "inidx_emb": r.emb(F.N_INIDX + 1, d),
        "mod_emb": r.emb(F.N_MODBUCKET + 1, d),
        "cls": r.emb(1, d)[0],
        "sig_mlp": r.mlp(cfg.sig_dim + 1, d, d),
        "task_emb": r.emb(cfg.n_tasks, d),
        "layers": [],
        "ln_f": r.ln(d),
        # pins / ports / modules
        "pin_c": r.dense(d, d), "pin_s": r.dense(d, d, 0.5), "pin_pos": r.emb(F.PINMAX, d),
        "pin_outpos": r.emb(F.OUTMAX + 1, d), "pin_srcport": r.emb(F.PINMAX, d), "pin_none": r.emb(1, d)[0],
        "port_t": r.dense(d, d), "port_pos": r.emb(F.PINMAX, d), "port_new": r.emb(1, d)[0], "port_newg": r.dense(d, d),
        "mod_k": r.mlp(F.N_MODFEAT, d, d), "mod_bk": r.emb(F.N_MODBUCKET + 1, d),
        # heads
        "type": r.mlp(d, d, N_MOVES, 0.1),
        "ptr_q": r.dense(d, F.N_PTR * d), "ptr_k": r.dense(d, d),
        "rw_q": r.dense(d, d), "rw_k": r.dense(d, d),
        "rws_a": r.dense(d, d), "rws_b": r.dense(d, d), "rws_k": r.dense(d, d),
        "del_a": r.dense(d, d), "del_b": r.dense(d, d), "del_k": r.dense(d, d),
        "ttr": r.mlp(d, d, F.TTB, 0.1), "ttf": r.mlp(d, d, F.TTB, 0.1),
        "reg_a": r.dense(d, d), "reg_b": r.dense(d, d), "reg_type": r.emb(2, d), "reg_bias": jnp.full(2, -3.0),
        "dr_a": r.dense(d, d), "dr_b": r.dense(d, d), "dr_k": r.dense(d, d),
        "add_sq": r.dense(d, d), "add_sk": r.dense(d, d), "add_c": r.dense(d, d),
        "add_kind": r.mlp(d, d, 3, 0.1), "add_ar": r.mlp(d, d, F.ARITY, 0.1), "add_ar_emb": r.emb(F.ARITY, d),
        "add_tt": r.mlp(d, d, F.TTB, 0.1), "add_srcq": r.dense(d, d), "add_srcpos": r.emb(F.NEWPIN, d),
        "add_srck": r.dense(d, d), "add_val": r.mlp(d, d, 3 * 2 * F.VB, 0.1), "add_init": r.dense(d, 3, 0.1),
        "ins_q": r.dense(d, d), "ins_k": r.dense(d, d), "ins_out": r.mlp(d, d, F.OUTPORT, 0.1),
        "ins_sa": r.dense(d, d), "ins_sb": r.dense(d, d), "ins_sk": r.dense(d, d),
        "ins_qa": r.dense(d, d), "ins_qb": r.dense(d, d), "ins_pos": r.emb(F.MODPIN, d), "ins_qk": r.dense(d, d),
        "adj_opt": r.mlp(d, d, F.NOPT, 0.1), "adj_a": r.dense(d, d), "adj_oemb": r.emb(F.NOPT, d),
        "adj_bemb": r.emb(F.VB, d), "adj_val": r.mlp(d, d, F.VB, 0.1),
    }
    for _ in range(cfg.layers):
        p["layers"].append({
            "ln1": r.ln(d), "w_in": (jax.random.normal(r.next(), (F.PINMAX, d, d)) / math.sqrt(d) * 0.5),
            "w_out": r.dense(d, d, 0.5), "w_mp": r.dense(d, d, 0.5),
            "ln2": r.ln(d), "qkv": r.dense(d, 3 * d), "o": r.dense(d, d, 0.5), "rel": r.emb(5, H, 0.1),
            "ln3": r.ln(d), "mlp": r.mlp(d, 2 * d, d, 0.5),
        })
    if cfg.dream_dim:
        p["dream"] = r.mlp(d, 2 * d, cfg.dream_dim)
    return p


def n_params(p) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))


# --- primitives ---------------------------------------------------------------------------------------------------

def dense(p, x):
    return x @ p["w"] + p["b"]


def mlp(p, x):
    return dense(p["l2"], jax.nn.gelu(dense(p["l1"], x)))


def ln(p, x):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + 1e-5) * p["g"] + p["b"]


def lsm(logits, mask):
    return jax.nn.log_softmax(jnp.where(mask, logits, NEG), axis=-1)


def bern(logits, bits, n):
    """Sum of Bernoulli log-probs of `bits` under `logits` over the first n entries."""
    on = jnp.arange(logits.shape[-1]) < n
    lp = jnp.where(bits > 0, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits))
    return jnp.where(on, lp, 0.0).sum(-1)


# --- encoder ------------------------------------------------------------------------------------------------------

def encode(p, f, cfg: EditorConfig) -> dict:
    T = f["tok_f"].shape[0]
    d, H = cfg.d, cfg.heads
    x = dense(p["tok_in"], f["tok_f"]) + p["inidx_emb"][f["tok_inidx"]] + p["mod_emb"][f["tok_mod"]]
    sigv = jnp.concatenate([f["sig"] * f["sig_on"], f["sig_on"][None]])
    cls = p["cls"] + mlp(p["sig_mlp"], sigv) + p["task_emb"][f["task"] % cfg.n_tasks]
    x = x.at[0].add(cls)
    A = jnp.zeros((F.PINMAX, T, T), jnp.float32).at[f["e_pin"], f["e_dst"], f["e_src"]].add(
        f["e_valid"].astype(jnp.float32))
    As = A.sum(0)                                         # [dst, src]
    feeds = As > 0
    rel = feeds.astype(jnp.int32) + 2 * feeds.T.astype(jnp.int32)
    rel = jnp.where(jnp.eye(T, dtype=bool), 4, rel)
    deg_out = jnp.maximum(As.sum(0), 1.0)
    valid = f["tok_valid"]
    hd = d // H
    for lp in p["layers"]:
        h = ln(lp["ln1"], x)
        Hp = jnp.einsum("td,pde->pte", h, lp["w_in"])
        m_in = jnp.einsum("pts,psd->td", A, Hp)
        m_out = (As.T @ dense(lp["w_out"], h)) / deg_out[:, None]
        x = x + dense(lp["w_mp"], jax.nn.gelu(m_in + m_out))
        h = ln(lp["ln2"], x)
        q, k, v = jnp.split(dense(lp["qkv"], h), 3, -1)
        q, k, v = (z.reshape(T, H, hd) for z in (q, k, v))
        s = jnp.einsum("thd,shd->hts", q, k) / math.sqrt(hd) + lp["rel"][rel].transpose(2, 0, 1)
        s = jnp.where(valid[None, None, :], s, NEG)
        a = jax.nn.softmax(s, -1)
        x = x + dense(lp["o"], jnp.einsum("hts,shd->thd", a, v).reshape(T, d))
        x = x + mlp(lp["mlp"], ln(lp["ln3"], x))
    h = ln(p["ln_f"], x)
    G = h[0]
    is_out = f["pin_out"] >= 0
    pe = jax.nn.gelu(dense(p["pin_c"], h[f["pin_cons"]]) + dense(p["pin_s"], h[f["pin_src"]])
                     + p["pin_pos"][f["pin_pos"]] + p["pin_srcport"][f["pin_srcport"]]
                     + jnp.where(is_out[:, None], p["pin_outpos"][jnp.maximum(f["pin_out"], 0)], 0.0))
    pe = pe.at[0].set(p["pin_none"])
    qe = dense(p["port_t"], h[f["port_tok"]]) + p["port_pos"][f["port_pos"]]
    qe = qe.at[0].set(p["port_new"] + dense(p["port_newg"], G))
    mk = mlp(p["mod_k"], f["mod_f"]) + p["mod_bk"][f["mod_bucket"]]
    return {"h": h, "G": G, "pe": pe, "qe": qe, "mk": mk}


def _ptr(q, K):
    return K @ q / math.sqrt(q.shape[-1])


# --- heads (single conditioning choice each) ----------------------------------------------------------------------

def h_type(p, e, f):
    return lsm(mlp(p["type"], e["G"]), f["type_mask"])


def h_node(p, e, f):
    d = e["G"].shape[0]
    Q = dense(p["ptr_q"], e["G"]).reshape(F.N_PTR, d)
    K = dense(p["ptr_k"], e["h"])
    return lsm(Q @ K.T / math.sqrt(d), f["ptr_mask"])          # [N_PTR, T]


def h_rw_pin(p, e, f):
    mask = f["pin_valid"] & (jnp.arange(f["pin_valid"].shape[0]) > 0)
    return lsm(_ptr(dense(p["rw_q"], e["G"]), dense(p["rw_k"], e["pe"])), mask)


def h_rw_src(p, e, f, s):
    q = dense(p["rws_a"], e["pe"][s]) + dense(p["rws_b"], e["G"])
    idx = jnp.arange(f["port_valid"].shape[0])
    mask = f["port_valid"] & (idx > 0) & (idx != f["pin_src_slot"][s])
    return lsm(_ptr(q, dense(p["rws_k"], e["qe"])), mask)


def h_del_fb(p, e, f, n):
    q = dense(p["del_a"], e["h"][n]) + dense(p["del_b"], e["G"])
    idx = jnp.arange(f["port_valid"].shape[0])
    mask = f["port_valid"] & (idx > 0) & (f["port_node"] != n)
    return lsm(_ptr(q, dense(p["del_k"], e["qe"])), mask)


def h_ttr(p, e, f, n):
    return mlp(p["ttr"], e["h"][n])                              # Bernoulli logits [TTB]


def h_ttf(p, e, f, n):
    return lsm(mlp(p["ttf"], e["h"][n]), jnp.arange(F.TTB) < (1 << f["node_k"][n]))


def h_region(p, e, f, n, which):
    """Inclusion logits [T] for anchor token n and the candidate mask (valid tokens after n)."""
    d = e["G"].shape[0]
    q = dense(p["reg_a"], e["h"][n]) + p["reg_type"][which]
    logits = dense(p["reg_b"], e["h"]) @ q / math.sqrt(d) + p["reg_bias"][which]
    cand = f["tok_valid"] & (jnp.arange(f["tok_valid"].shape[0]) > n)
    return logits, cand


def region_logz(logits, cand, budget):
    """log P(number of included candidates <= budget - 1) under independent Bernoullis (exact DP)."""
    pj = jnp.where(cand, jax.nn.sigmoid(logits), 0.0)

    def step(c, pr):
        sh = jnp.concatenate([jnp.zeros(1, c.dtype), c[:-1]])
        return c * (1 - pr) + sh * pr, None

    c0 = jnp.zeros(F.DUPMAX, jnp.float32).at[0].set(1.0)
    c, _ = jax.lax.scan(step, c0, pj)
    Z = jnp.where(jnp.arange(F.DUPMAX) < budget, c, 0.0).sum()
    return jnp.log(jnp.maximum(Z, 1e-30))


def region_lp(p, e, f, n, which, region, budget):
    logits, cand = h_region(p, e, f, n, which)
    lp = jnp.where(cand, jnp.where(region, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits)), 0.0).sum()
    return lp - region_logz(logits, cand, budget)


def h_dup_redir(p, e, f, n):
    """Unnormalized redirect logits [P] for anchor n (the mask depends on the sampled set)."""
    q = dense(p["dr_a"], e["G"]) + dense(p["dr_b"], e["h"][n])
    return _ptr(q, dense(p["dr_k"], e["pe"]))


def h_add_splice(p, e, f):
    return lsm(_ptr(dense(p["add_sq"], e["G"]), dense(p["add_sk"], e["pe"])), f["pin_valid"])


def _add_ctx(p, e, s):
    return e["G"] + dense(p["add_c"], e["pe"][s])


def h_add_kind(p, e, f, s):
    return jax.nn.log_softmax(mlp(p["add_kind"], _add_ctx(p, e, s)))


def h_add_arity(p, e, f, s):
    return lsm(mlp(p["add_ar"], _add_ctx(p, e, s)), jnp.arange(F.ARITY) <= f["arity_max"])


def h_add_tt(p, e, f, s, k):
    return mlp(p["add_tt"], _add_ctx(p, e, s) + p["add_ar_emb"][k])       # Bernoulli logits [TTB]


def h_add_src(p, e, f):
    q = dense(p["add_srcq"], e["G"])[None] + p["add_srcpos"]                 # [NEWPIN, d]
    K = dense(p["add_srck"], e["qe"])
    return lsm(q @ K.T / math.sqrt(q.shape[-1]), f["port_valid"][None])      # [NEWPIN, Q]


def h_add_val(p, e, f):
    l = mlp(p["add_val"], e["G"]).reshape(3, 2, F.VB)
    return lsm(l, f["bin_mask"][F.VK_NEW_DELAY][None, None])                 # [kind, rise/fall, VB]


def h_add_init(p, e, f):
    return dense(p["add_init"], e["G"])                                      # Bernoulli logits [3]


def h_ins_mod(p, e, f):
    return lsm(_ptr(dense(p["ins_q"], e["G"]), dense(p["ins_k"], e["mk"])), f["mod_valid"])


def h_ins_out(p, e, f, m):
    return lsm(mlp(p["ins_out"], e["mk"][m]), jnp.arange(F.OUTPORT) < f["mod_nout"][m])


def h_ins_splice(p, e, f, m):
    q = dense(p["ins_sa"], e["G"]) + dense(p["ins_sb"], e["mk"][m])
    idx = jnp.arange(f["pin_valid"].shape[0])
    return lsm(_ptr(q, dense(p["ins_sk"], e["pe"])), f["pin_valid"] & ((idx == 0) | (f["mod_nout"][m] > 0)))


def h_ins_src(p, e, f, m):
    q = (dense(p["ins_qa"], e["G"]) + dense(p["ins_qb"], e["mk"][m]))[None] + p["ins_pos"]
    K = dense(p["ins_qk"], e["qe"])
    return lsm(q @ K.T / math.sqrt(q.shape[-1]), f["port_valid"][None])      # [MODPIN, Q]


def h_adj_opt(p, e, f, n):
    return lsm(mlp(p["adj_opt"], e["h"][n]), f["opt_mask"][n])


def h_adj_val(p, e, f, n, o):
    z = dense(p["adj_a"], e["h"][n]) + p["adj_oemb"][o] + p["adj_bemb"][f["old_bin"][n, o]]
    vk = jnp.minimum(f["vkind"][n, o], 2)
    return lsm(mlp(p["adj_val"], z), f["bin_mask"][vk])


# --- log-probability of one move (training) ------------------------------------------------------------------------

def uniform_coarse_lp(f, t, loc_ok):
    """log-probability of the coarse choice (type, location) under the uniform operator (approximate for the
    anchors of DUPLICATE / REGENERATE, which the operator does not draw uniformly)."""
    w = jnp.where(f["type_mask"], f["type_w"], 0.0)
    lt = jnp.log(jnp.maximum(w[t], 1e-30)) - jnp.log(jnp.maximum(w.sum(), 1e-30))
    n_ptr = f["ptr_mask"].sum(1)
    n_pins = f["pin_valid"].sum() - 1
    n_mod = f["mod_valid"].sum()
    ptr_slot = jnp.array([F.PTR.get(MoveType(i), 0) for i in range(N_MOVES)])
    l_node = -jnp.log(jnp.maximum(n_ptr[ptr_slot[t]], 1))
    l_pin = -jnp.log(jnp.maximum(n_pins, 1))
    l_splice = jnp.where(loc_ok, jnp.log(f["p_splice"]) + l_pin, jnp.log(1 - f["p_splice"]))
    l_mod = -jnp.log(jnp.maximum(n_mod, 1))
    ll = jnp.select([t == MoveType.REWIRE, t == MoveType.ADD_NODE, t == MoveType.INSERT_MODULE],
                    [l_pin, l_splice, l_mod], l_node)
    return lt + ll


def move_logprob(p, f, a, cfg: EditorConfig, enc=None):
    """-> (log q(move), log q(coarse type and location), log u(coarse) under the uniform operator)."""
    e = encode(p, f, cfg) if enc is None else enc
    t = a["type"]
    n = a["node"]
    lt = h_type(p, e, f)[t]
    nodes = h_node(p, e, f)
    T = MoveType
    # TT_REPLACE / TT_FLIP
    k_n = f["node_k"][n]
    l_ttr = nodes[0, n] + bern(h_ttr(p, e, f, n), a["tt_bits"], 1 << k_n)
    l_ttf = nodes[1, n] + h_ttf(p, e, f, n)[a["flip_row"]]
    # REWIRE
    l_rwp = h_rw_pin(p, e, f)[a["rw_pin"]]
    l_rw = l_rwp + h_rw_src(p, e, f, a["rw_pin"])[a["rw_src"]]
    # DELETE
    l_del = nodes[2, n] + h_del_fb(p, e, f, n)[a["del_fb"]]
    # DUPLICATE / REGENERATE
    l_dup = (nodes[3, n] + region_lp(p, e, f, n, 0, a["region"], f["dup_budget"])
             + lsm(h_dup_redir(p, e, f, n), a["redir_mask"])[a["dup_redir"]])
    l_reg = nodes[5, n] + region_lp(p, e, f, n, 1, a["region"], f["regen_budget"])
    # ADD_NODE
    s = a["add_splice"]
    l_spl = h_add_splice(p, e, f)[s]
    kind = a["add_kind"]
    k = a["add_k"]
    l_kind = h_add_kind(p, e, f, s)[kind]
    l_gate = h_add_arity(p, e, f, s)[k] + bern(h_add_tt(p, e, f, s, k), a["add_tt"], 1 << k)
    src = h_add_src(p, e, f)
    l_src = jnp.where(a["add_src_on"], src[jnp.arange(F.NEWPIN), a["add_src"]], 0.0).sum()
    val = h_add_val(p, e, f)
    lw = f["bin_lw"][F.VK_NEW_DELAY]
    kc = jnp.minimum(kind, 1)
    l_arc = (val[kc, 0, a["add_rise"]] - lw[a["add_rise"]] + val[kc, 1, a["add_fall"]] - lw[a["add_fall"]])
    l_arc = jnp.where(a["add_arc_on"], l_arc, 0.0).sum()
    il = h_add_init(p, e, f)[kind]
    l_init = jnp.where(a["add_init_on"], jnp.where(a["add_init"] > 0, jax.nn.log_sigmoid(il),
                                                   jax.nn.log_sigmoid(-il)), 0.0)
    l_add = l_spl + l_kind + jnp.where(kind == 0, l_gate, 0.0) + l_src + l_arc + l_init
    # INSERT_MODULE
    m = a["ins_mod"]
    l_mod = h_ins_mod(p, e, f)[m]
    l_ins = (l_mod + h_ins_out(p, e, f, m)[a["ins_out"]] + h_ins_splice(p, e, f, m)[a["ins_splice"]]
             + jnp.where(a["ins_src_on"], h_ins_src(p, e, f, m)[jnp.arange(F.MODPIN), a["ins_src"]], 0.0).sum())
    # ADJUST_PARAM
    o = a["adj_opt"]
    vk = jnp.minimum(f["vkind"][n, o], 2)
    l_val = h_adj_val(p, e, f, n, o)[a["adj_bin"]] - f["bin_lw"][vk, a["adj_bin"]]
    l_adj = nodes[4, n] + h_adj_opt(p, e, f, n)[o] + jnp.where(a["adj_on"], l_val, 0.0)
    l_inl = nodes[6, n]
    body = jnp.select([t == T.TT_REPLACE, t == T.TT_FLIP, t == T.REWIRE, t == T.ADD_NODE, t == T.DELETE_NODE,
                       t == T.DUPLICATE, t == T.INSERT_MODULE, t == T.ADJUST_PARAM, t == T.REGENERATE],
                      [l_ttr, l_ttf, l_rw, l_add, l_del, l_dup, l_ins, l_adj, l_reg], l_inl)
    ptr_slot = jnp.array([F.PTR.get(MoveType(i), 0) for i in range(N_MOVES)])
    coarse_loc = jnp.select([t == T.REWIRE, t == T.ADD_NODE, t == T.INSERT_MODULE],
                            [l_rwp, l_spl, l_mod], nodes[ptr_slot[t], n])
    splice_on = jnp.where(t == T.ADD_NODE, s, a["ins_splice"]) > 0
    return lt + body, lt + coarse_loc, uniform_coarse_lp(f, t, splice_on)


def dream_pred(p, f, cfg: EditorConfig, enc=None):
    e = encode(p, f, cfg) if enc is None else enc
    return mlp(p["dream"], e["G"])


# --- full tables (sampling) ---------------------------------------------------------------------------------------

def tables(p, f, cfg: EditorConfig) -> dict:
    """Every factor's log-probabilities (or Bernoulli logits) over all conditioning choices, for one circuit."""
    e = encode(p, f, cfg)
    T = f["tok_f"].shape[0]
    P = f["pin_valid"].shape[0]
    M = f["mod_valid"].shape[0]
    toks = jnp.arange(T)
    pins = jnp.arange(P)
    mods = jnp.arange(M)
    vm = jax.vmap
    reg0 = vm(lambda n: h_region(p, e, f, n, 0)[0])(toks)
    reg1 = vm(lambda n: h_region(p, e, f, n, 1)[0])(toks)
    return {
        "type": h_type(p, e, f),
        "node": h_node(p, e, f),
        "rw_pin": h_rw_pin(p, e, f),
        "rw_src": vm(lambda s: h_rw_src(p, e, f, s))(pins),
        "del_fb": vm(lambda n: h_del_fb(p, e, f, n))(toks),
        "ttr": vm(lambda n: h_ttr(p, e, f, n))(toks),
        "ttf": vm(lambda n: h_ttf(p, e, f, n))(toks),
        "region": jnp.stack([reg0, reg1]),                                   # [2, T(anchor), T]
        "dup_redir": vm(lambda n: h_dup_redir(p, e, f, n))(toks),            # [T, P] unnormalized
        "add_splice": h_add_splice(p, e, f),
        "add_kind": vm(lambda s: h_add_kind(p, e, f, s))(pins),
        "add_arity": vm(lambda s: h_add_arity(p, e, f, s))(pins),
        "add_tt": vm(lambda s: vm(lambda k: h_add_tt(p, e, f, s, k))(jnp.arange(F.ARITY)))(pins),
        "add_src": h_add_src(p, e, f),
        "add_val": h_add_val(p, e, f),
        "add_init": h_add_init(p, e, f),
        "ins_mod": h_ins_mod(p, e, f),
        "ins_out": vm(lambda m: h_ins_out(p, e, f, m))(mods),
        "ins_splice": vm(lambda m: h_ins_splice(p, e, f, m))(mods),
        "ins_src": vm(lambda m: h_ins_src(p, e, f, m))(mods),
        "adj_opt": vm(lambda n: h_adj_opt(p, e, f, n))(toks),
        "adj_val": vm(lambda n: vm(lambda o: h_adj_val(p, e, f, n, o))(jnp.arange(F.NOPT)))(toks),
    }


# --- checkpoints --------------------------------------------------------------------------------------------------

def save_params(path, p, cfg: EditorConfig, extra: dict | None = None) -> None:
    import json
    flat, tree = jax.tree_util.tree_flatten(p)
    paths = [jax.tree_util.keystr(k) for k, _ in jax.tree_util.tree_flatten_with_path(p)[0]]
    np.savez_compressed(path, **{f"p{i}": np.asarray(x) for i, x in enumerate(flat)},
                        __meta__=np.frombuffer(json.dumps({"cfg": cfg.to_dict(), "paths": paths,
                                                           "extra": extra or {}}).encode(), np.uint8))


def load_params(path) -> tuple:
    import json
    z = np.load(path)
    meta = json.loads(bytes(z["__meta__"]).decode())
    cfg = EditorConfig(**meta["cfg"])
    ref = init_params(cfg)
    paths = [jax.tree_util.keystr(k) for k, _ in jax.tree_util.tree_flatten_with_path(ref)[0]]
    if paths != meta["paths"]:
        raise ValueError("checkpoint parameter layout does not match this code version")
    _, tree = jax.tree_util.tree_flatten(ref)
    leaves = [jnp.asarray(z[f"p{i}"]) for i in range(len(paths))]
    return jax.tree_util.tree_unflatten(tree, leaves), cfg, meta.get("extra", {})
