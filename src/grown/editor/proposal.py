"""`LearnedProposal`: the editor model as a `moves.Proposal`, plus the numpy sampler over its factor tables.

`propose(circuit, rng)` featurizes the circuit (with its behaviour signature, looked up in an attached archive by
hash), computes every factor table in one jitted forward pass (cached per circuit), samples a move factor by
factor at `temperature`, decodes it to a `Move` and checks that it applies and validates (up to `attempts`
samples; then None, which makes `UniformOperator.mutate` fall back to a uniform move). No-op samples (a truth
table or value equal to the current one) count as failed attempts.

It plugs into `UniformOperator.mutate(c, rng, proposal, eps)` unchanged: the operator calls `propose` with
probability 1 - eps and samples uniformly otherwise, so every move keeps probability at least eps * uniform.

Collapse instruments (see metrics.py): every call records the type drawn, the sample's log-probability (its mean
negation estimates the entropy of the full move distribution) and, per circuit, the exact entropy, top-k mass and
KL to the uniform operator of the coarse (type, location) marginal. `pop_stats()` aggregates and resets them.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass

import jax
import numpy as np

from ..circuit import codec
from ..circuit.graph import DELAY, GATE, WIRE, InvalidCircuit, Node
from ..circuit.moves import NONE, SLOT, Move, MoveType
from . import featurize as F
from . import metrics as Mx
from . import model as Mdl

_BAD = (InvalidCircuit, IndexError, ValueError, KeyError)


# --- sampling from tables -----------------------------------------------------------------------------------------

class _S:
    """Tempered sampling helpers that accumulate the log-probability of what they draw."""

    def __init__(self, rng, tau):
        self.rng, self.tau, self.lp = rng, tau, 0.0

    def cat(self, lp, mask=None):
        lp = np.asarray(lp, np.float64)
        if mask is not None:
            lp = np.where(mask, lp, -1e9)
        z = lp / self.tau
        z = z - z.max()
        w = np.exp(z)
        w /= w.sum()
        i = int(self.rng.choice(len(w), p=w))
        self.lp += math.log(max(w[i], 1e-300))
        return i

    def bern(self, logits):
        x = np.asarray(logits, np.float64) / self.tau
        pr = 1.0 / (1.0 + np.exp(-x))
        b = (self.rng.random(len(pr)) < pr).astype(np.int64)
        self.lp += float(np.where(b > 0, np.log(np.maximum(pr, 1e-300)), np.log(np.maximum(1 - pr, 1e-300))).sum())
        return b

    def region(self, logits, cand, budget):
        """Exact sample of independent Bernoullis over `cand` conditioned on at most budget - 1 inclusions."""
        x = np.asarray(logits, np.float64) / self.tau
        pr = np.where(cand, 1.0 / (1.0 + np.exp(-x)), 0.0)
        m = int(budget) - 1
        n = len(pr)
        R = np.ones((n + 1, m + 1))                     # R[j, r] = P(inclusions in j.. <= r)
        for j in range(n - 1, -1, -1):
            R[j] = (1 - pr[j]) * R[j + 1]
            R[j, 1:] += pr[j] * R[j + 1, :-1]
        out, r = [], m
        for j in range(n):
            if pr[j] == 0:
                continue
            q = pr[j] * (R[j + 1, r - 1] if r > 0 else 0.0) / max(R[j, r], 1e-300)
            if self.rng.random() < q:
                out.append(j)
                r -= 1
                self.lp += math.log(max(q, 1e-300))
            else:
                self.lp += math.log(max(1 - q, 1e-300))
        return out

    def value(self, lp_bins, vk, A):
        b = self.cat(lp_bins, A["bin_mask"][vk])
        lo, hi = int(A["bin_vlo"][vk, b]), int(A["bin_vhi"][vk, b])
        self.lp -= math.log(hi - lo + 1)
        return int(self.rng.integers(lo, hi + 1))


def sample_move(ctx: F.Ctx, tab: dict, rng: np.random.Generator, tau: float = 1.0):
    """-> (Move, log q(move) at temperature tau, type)."""
    c = ctx.circuit
    A = ctx.arrays
    s = _S(rng, tau)
    t = MoveType(s.cat(tab["type"], A["type_mask"]))
    node = lambda tok: ctx.node(tok)  # noqa: E731
    if t in (MoveType.TT_REPLACE, MoveType.TT_FLIP, MoveType.DELETE_NODE, MoveType.ADJUST_PARAM,
             MoveType.INLINE_MODULE, MoveType.DUPLICATE, MoveType.REGENERATE):
        n = s.cat(tab["node"][F.PTR[t]], A["ptr_mask"][F.PTR[t]])
    if t == MoveType.TT_REPLACE:
        k = c.nodes[node(n)].k
        bits = s.bern(tab["ttr"][n][:1 << k])
        mv = Move(t, node(n), a=codec.s64(F.bits_to_int(bits)))
    elif t == MoveType.TT_FLIP:
        mv = Move(t, node(n), a=s.cat(tab["ttf"][n]))
    elif t == MoveType.REWIRE:
        p = s.cat(tab["rw_pin"])
        q = s.cat(tab["rw_src"][p])
        ci, pin = F.decode_pin(ctx, p)
        mv = Move(t, ci, pin, *F.decode_port(ctx, q))
    elif t == MoveType.DELETE_NODE:
        q = s.cat(tab["del_fb"][n])
        mv = Move(t, node(n), -1, *F.decode_port(ctx, q))
    elif t in (MoveType.DUPLICATE, MoveType.REGENERATE):
        w = 0 if t == MoveType.DUPLICATE else 1
        cand = np.arange(ctx.T) > n
        budget = A["dup_budget"] if w == 0 else A["regen_budget"]
        S = [node(n)] + [node(j) for j in s.region(tab["region"][w, n], cand, budget)]
        if t == MoveType.DUPLICATE:
            r = s.cat(tab["dup_redir"][n], F.redirect_mask(ctx, S))
            mv = Move(t, S[0], -1, *F.decode_pin(ctx, r), tuple(S))
        else:
            mv = Move(t, S[0], -1, 0, 0, tuple(S))
    elif t == MoveType.ADD_NODE:
        sp = s.cat(tab["add_splice"])
        kind = s.cat(tab["add_kind"][sp])
        spliced = sp > 0
        a, b = F.decode_pin(ctx, sp)
        if kind == 0:
            k = s.cat(tab["add_arity"][sp])
            tt = F.bits_to_int(s.bern(tab["add_tt"][sp, k][:1 << k]))
        else:
            k, tt = 1, 0
        ins = []
        for i in range(k):
            if i == 0 and spliced:
                ins.append(c.ref_at(a, b))
            else:
                ins.append(F.decode_port(ctx, s.cat(tab["add_src"][i])))
        rise, fall = [], []
        if kind in (0, 1):
            for i in range(k):
                rise.append(s.value(tab["add_val"][kind, 0], F.VK_NEW_DELAY, A))
                fall.append(s.value(tab["add_val"][kind, 1], F.VK_NEW_DELAY, A))
            if kind == 0 and k == 0:
                init = tt & 1
            else:
                init = int(s.bern(tab["add_init"][kind:kind + 1])[0])
        if kind == 0:
            nd = Node(GATE, tuple(ins), tt, tuple(rise), tuple(fall), init)
        elif kind == 1:
            nd = Node(DELAY, tuple(ins), 0, tuple(rise), tuple(fall), init)
        else:
            nd = Node(WIRE, tuple(ins))
        mv = Move(t, len(c.nodes), -1, a, b, tuple(codec.encode_node(nd)))
    elif t == MoveType.INSERT_MODULE:
        m = s.cat(tab["ins_mod"], A["mod_valid"])
        h = ctx.modules[m]
        d = ctx.library.get(h)
        o = s.cat(tab["ins_out"][m])
        sp = s.cat(tab["ins_splice"][m])
        a, b = F.decode_pin(ctx, sp)
        src = []
        for j in range(d.n_in):
            if j == 0 and sp > 0:
                src.append(c.ref_at(a, b))
            else:
                src.append(F.decode_port(ctx, s.cat(tab["ins_src"][m, j])))
        lo, hi = codec.hash_to_ints(h)
        mv = Move(t, len(c.nodes), -1, a, b, (lo, hi, o, d.n_in) + tuple(x for r in src for x in r))
    elif t == MoveType.ADJUST_PARAM:
        o = s.cat(tab["adj_opt"][n])
        field, pin = F.opt_field(o)
        vk = int(A["vkind"][n, o])
        if vk == F.VK_TOGGLE:
            old, _ = F._old_value(c.nodes[node(n)], field, pin, ctx.library)
            val = 1 - old
        else:
            val = s.value(tab["adj_val"][n, o], vk, A)
        mv = Move(t, node(n), pin, field, val)
    else:  # INLINE_MODULE
        mv = Move(t, node(n))
    return mv, s.lp, t


def is_noop(c, mv: Move, library=None) -> bool:
    t = mv.type
    if t == MoveType.TT_REPLACE:
        return codec.u64(mv.a) == c.nodes[mv.node].tt
    if t == MoveType.ADJUST_PARAM:
        nd = c.nodes[mv.node]
        if mv.a == SLOT:
            d = library.get(nd.module)
            return (nd.params or d.defaults())[mv.pin] == mv.b
        old, _ = F._old_value(nd, mv.a, mv.pin, library)
        return old == mv.b
    return False


# --- the proposal -------------------------------------------------------------------------------------------------

@dataclass
class _Entry:
    ctx: F.Ctx
    tab: dict
    coarse: dict


class LearnedProposal:
    def __init__(self, params, cfg: Mdl.EditorConfig, operator, temperature: float = 1.0, archive=None,
                 task: int = 0, cache_size: int = 512, attempts: int = 8, batch: int = 16):
        self.params, self.cfg, self.op = params, cfg, operator
        self.temperature = temperature
        self.task = task
        self.cache_size = cache_size
        self.attempts = attempts
        self.batch = batch
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self._tables = jax.jit(lambda p, f: Mdl.tables(p, f, cfg))
        self._tables_b = jax.jit(jax.vmap(lambda p, f: Mdl.tables(p, f, cfg), in_axes=(None, 0)))
        self.archive = None
        self._sig_key = None
        self._sigs: dict = {}
        if archive is not None:
            self.attach(archive)
        self.reset_stats()

    # -- signatures
    def attach(self, archive) -> None:
        """Look up parents' behaviour signatures in `archive` (by canonical hash), as the edit log records them."""
        self.archive = archive
        self._sig_key = None

    def _sig(self, c):
        a = self.archive
        if a is None or not a.sig_dim:
            return None
        key = (a.n_candidates, a.n_batches)
        if key != self._sig_key:
            cells = a.filled_cells()
            self._sigs = {bytes(a.hash[x].astype("<u8")): a.sig[x] for x in cells.tolist()}
            self._sig_key = key
        return self._sigs.get(c.hash)

    # -- encodings
    def _entry(self, c) -> _Entry:
        e = self.cache.get(c.hash)
        if e is not None and e.ctx.circuit.same_as(c):
            self.cache.move_to_end(c.hash)
            self.hits += 1
            return e
        self.misses += 1
        ctx = F.featurize(c, self.op, self._sig(c), self.task)
        dims = self._dims([ctx])
        f = F.collate([ctx.arrays], dims, self.cfg.sig_dim)
        f = {k: v[0] for k, v in f.items()}
        tab = jax.tree_util.tree_map(np.asarray, self._tables(self.params, f))
        return self._store(c, ctx, tab)

    def _store(self, c, ctx, tab) -> _Entry:
        tab = _slice(tab, ctx)
        e = _Entry(ctx, tab, Mx.coarse_metrics(ctx, tab, self.op))
        self.cache[c.hash] = e
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return e

    def prepare(self, circuits) -> int:
        """Batch-compute and cache the tables of every circuit not cached yet. Returns how many were computed."""
        todo, seen = [], set()
        for c in circuits:
            if c is None or c.hash in seen:
                continue
            e = self.cache.get(c.hash)
            if e is None or not e.ctx.circuit.same_as(c):
                todo.append(c)
                seen.add(c.hash)
        todo = todo[-self.cache_size:]
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            ctxs = [F.featurize(c, self.op, self._sig(c), self.task) for c in chunk]
            dims = self._dims(ctxs)
            items = [x.arrays for x in ctxs]
            items += [items[0]] * (self.batch - len(items))
            f = F.collate(items, dims, self.cfg.sig_dim)
            tb = jax.tree_util.tree_map(np.asarray, self._tables_b(self.params, f))
            for j, (c, ctx) in enumerate(zip(chunk, ctxs)):
                self._store(c, ctx, {k: v[j] for k, v in tb.items()})
        return len(todo)

    def _dims(self, ctxs) -> dict:
        """Padded shapes: at least a floor derived from the operator's size limit, so that one compiled shape
        serves nearly every circuit of a run (jit retraces per shape)."""
        d = F.padded_dims([x.arrays for x in ctxs])
        c = ctxs[0].circuit
        cfg = self.op.cfg
        N = cfg.max_nodes + cfg.dup_max
        if N > 128:                     # large limits: per-shape buckets are cheaper than padding everything
            return d
        floor = {"T": 1 + c.n_in + N, "P": 1 + N * min(cfg.arity_max, 4) + c.n_out, "Q": 1 + c.n_in + N,
                 "E": N * min(cfg.arity_max, 4), "M": 1}
        return {k: max(v, F.bucket(floor[k])) for k, v in d.items()}

    # -- proposing
    def propose(self, circuit, rng) -> Move | None:
        e = self._entry(circuit)
        self.coarse_log.append(e.coarse)
        for _ in range(self.attempts):
            mv, lp, t = sample_move(e.ctx, e.tab, rng, self.temperature)
            if is_noop(circuit, mv, self.op.library):
                self.rejected += 1
                continue
            try:
                self.op.apply(circuit, mv, rng)
            except _BAD:
                self.rejected += 1
                continue
            self.samples.append((int(t), lp))
            return mv
        self.failed += 1
        return None

    def propose_batch(self, circuits, rng) -> list:
        self.prepare(circuits)
        return [self.propose(c, rng) for c in circuits]

    def log_prob(self, circuit, mv: Move) -> float:
        """log q(mv) at temperature 1 (exact; not conditioned on validity)."""
        e = self._entry(circuit)
        act = F.encode_move(e.ctx, mv)
        dims = F.padded_dims([e.ctx.arrays])
        f = F.collate([{**e.ctx.arrays, **act}], dims, self.cfg.sig_dim)
        f = {k: v[0] for k, v in f.items()}
        return float(Mdl.move_logprob(self.params, f, f, self.cfg)[0])

    # -- instruments
    def reset_stats(self) -> None:
        self.samples, self.coarse_log = [], []
        self.hits = self.misses = self.rejected = self.failed = 0

    def pop_stats(self) -> dict:
        st = Mx.aggregate(self.samples, self.coarse_log)
        st.update(cache_hits=self.hits, cache_misses=self.misses, rejected=self.rejected, failed=self.failed)
        self.reset_stats()
        return st


def _slice(tab: dict, ctx: F.Ctx) -> dict:
    """Drop padding from a padded table dict (so sampling sees only real tokens, pins, ports, modules)."""
    d = F.dims_of(ctx.arrays)
    T, P, Q, M = d["T"], d["P"], d["Q"], d["M"]
    sl = {
        "node": (slice(None), slice(0, T)), "rw_pin": (slice(0, P),), "rw_src": (slice(0, P), slice(0, Q)),
        "del_fb": (slice(0, T), slice(0, Q)), "ttr": (slice(0, T),), "ttf": (slice(0, T),),
        "region": (slice(None), slice(0, T), slice(0, T)), "dup_redir": (slice(0, T), slice(0, P)),
        "add_splice": (slice(0, P),), "add_kind": (slice(0, P),), "add_arity": (slice(0, P),),
        "add_tt": (slice(0, P),), "add_src": (slice(None), slice(0, Q)), "ins_mod": (slice(0, M),),
        "ins_out": (slice(0, M),), "ins_splice": (slice(0, M), slice(0, P)),
        "ins_src": (slice(0, M), slice(None), slice(0, Q)), "adj_opt": (slice(0, T),), "adj_val": (slice(0, T),),
    }
    return {k: (v[sl[k]] if k in sl else v) for k, v in tab.items()}
