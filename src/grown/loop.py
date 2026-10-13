"""A thin end-to-end loop: sample parents, mutate, evaluate, score, insert, log. A smoke test of the plumbing, not
the research loop (no teachers, no DAgger, no learner, one task).

Per generation:
  1. sample `batch` parent cells from the archive (uniform or curiosity);
  2. mutate each parent once with the operator (uniform, or a learned proposal mixed with the uniform floor);
  3. evaluate the children on the substrate with the task's per-step inputs;
  4. score: `score_fn(result) -> per-step error [B, S]`; fitness = -mean error;
  5. insert into the archive (structural descriptors computed, behavioral ones from `behavior_fn`);
  6. log every edit with its fate, the parent's and child's fitness, cell, and behaviour signatures.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .circuit.graph import InvalidCircuit, random_circuit
from .circuit.moves import Fate, Move, Source
from .archive.map_elites import Archive, hashes_to_array
from .sim.base import EvalResult

SIG_DIM = 4


def default_behavior(res: EvalResult, err: np.ndarray) -> dict:
    flips = (np.diff(res.outputs.astype(np.int8), axis=1) != 0).mean((1, 2)) if res.outputs.shape[1] > 1 else \
        np.zeros(len(err))
    return {"toggle_rate": res.toggle_rate, "mean_high": res.high.mean((1, 2)), "output_flip_rate": flips,
            "error": err.mean(1)}


def default_signature(res: EvalResult, err: np.ndarray) -> np.ndarray:
    b = default_behavior(res, err)
    return np.stack([b["error"], b["toggle_rate"], b["mean_high"], b["output_flip_rate"]], 1).astype(np.float32)


@dataclass
class LoopConfig:
    generations: int = 20
    batch: int = 64
    K: int = 4                      # substrate ticks per control step
    eps: float = 0.1                # uniform floor when a proposal is given
    sampling: str = "curiosity"     # or "uniform", "lineage" (archive.map_elites)
    n_init: int = 64                # random seed circuits
    init_nodes: int = 8
    init_arity: tuple = (1, 3)
    task: int = 0
    seed: int = 0


def evaluate(substrate, circuits, inputs, K, score_fn, library, behavior_fn, sig_fn):
    res = substrate.evaluate(circuits, inputs, K, library)
    err = np.asarray(score_fn(res), np.float64)
    return res, err, -err.mean(1), behavior_fn(res, err), sig_fn(res, err)


def breed(archive: Archive, operator, rng, batch: int, sampling: str, proposal=None, eps: float = 0.1) -> dict:
    """Sample `batch` parent cells and mutate each once. Returns the parents' cells, hashes, fitness, signatures and
    lineage, the children (valid ones only), the moves and sources (one per parent) and the validity mask."""
    cells = archive.sample(batch, rng, sampling)
    if proposal is None:
        from .circuit.childbatch import breed_compiled
        cb = breed_compiled(operator, [archive.items[c] for c in cells.tolist()], rng)
        if cb is not None:                       # same moves, children and draws as the loop below
            return dict(cells=cells, p_hash=archive.hash[cells].copy(), p_fit=archive.fitness[cells].copy(),
                        p_sig=archive.sig[cells].copy(), p_lin=archive.lineage[cells].copy(),
                        children=cb.children, moves=cb.moves, sources=np.zeros(len(cells), np.int8), ok=cb.ok,
                        same_graph=cb.same_graph)
    children, moves, sources, ok = [], [], [], []
    for c in cells.tolist():
        try:
            child, mv, src = operator.mutate(archive.items[c], rng, proposal, eps)
            children.append(child), moves.append(mv), sources.append(src), ok.append(True)
        except InvalidCircuit:
            moves.append(Move(-1)), sources.append(Source.UNIFORM), ok.append(False)
    return dict(cells=cells, p_hash=archive.hash[cells].copy(), p_fit=archive.fitness[cells].copy(),
                p_sig=archive.sig[cells].copy(), p_lin=archive.lineage[cells].copy(), children=children,
                moves=moves, sources=sources, ok=np.array(ok, bool))


def settle(archive: Archive, br: dict, desc, fit, sig, child_hashes: str = "all") -> dict:
    """Insert the evaluated valid children of `breed` (desc, fit, sig over br["children"]) and return per-edit
    columns over all parents: fate, cell, displaced hash, child fitness, signature and hash.

    child_hashes: "all" computes every valid child's canonical hash for the log; "inserted" only those that enter
    the archive (their hashes are needed anyway; the others are logged as zeros: a child is identified by its
    parent's hash and the replayable move). Archive contents are the same either way."""
    ok, children = br["ok"], br["children"]
    B = len(ok)
    fate = np.full(B, Fate.INVALID, np.int8)
    cell = np.full(B, -1, np.int64)
    displaced = np.zeros((B, 2), np.uint64)
    c_fit = np.full(B, np.nan, np.float32)
    c_sig = np.full((B, archive.sig_dim), np.nan, np.float32)
    c_hash = np.zeros((B, 2), np.uint64)
    if ok.any():
        ins = archive.insert(desc, fit, items=children, sigs=sig, lineage=br["p_lin"][ok],
                             parent_cells=br["cells"][ok])
        fate[ok], cell[ok], displaced[ok] = ins.fate, ins.cell, ins.displaced
        c_fit[ok], c_sig[ok] = fit, sig
        if child_hashes == "all":
            c_hash[ok] = hashes_to_array([c.hash for c in children])
        elif child_hashes == "inserted":
            entered = np.flatnonzero(ins.inserted)
            vi = np.flatnonzero(ok)
            c_hash[vi[entered]] = hashes_to_array([children[i].hash for i in entered.tolist()])
        else:
            raise ValueError(f"child_hashes must be 'all' or 'inserted', not {child_hashes!r}")
    return dict(fate=fate, cell=cell, displaced=displaced, c_fit=c_fit, c_sig=c_sig, c_hash=c_hash)


def log_edits(log, generation: int, task: int, br: dict, st: dict) -> None:
    moves = br["moves"]
    if hasattr(moves, "pay_values"):             # MoveColumns from the compiled breed
        log.append_batch(generation=generation, task=task, move=moves.type, source=br["sources"], fate=st["fate"],
                         node=moves.node, pin=moves.pin, a=moves.a, b=moves.b,
                         payload=(moves.pay_values, moves.pay_offsets), parent_hash=br["p_hash"],
                         child_hash=st["c_hash"], displaced_hash=st["displaced"], cell=st["cell"],
                         parent_fitness=br["p_fit"], child_fitness=st["c_fit"], parent_sig=br["p_sig"],
                         child_sig=st["c_sig"])
        return
    log.append_batch(generation=generation, task=task, move=[m.type for m in moves], source=br["sources"],
                     fate=st["fate"], node=[m.node for m in moves], pin=[m.pin for m in moves],
                     a=[m.a for m in moves], b=[m.b for m in moves], payload=[m.data for m in moves],
                     parent_hash=br["p_hash"], child_hash=st["c_hash"], displaced_hash=st["displaced"],
                     cell=st["cell"], parent_fitness=br["p_fit"], child_fitness=st["c_fit"], parent_sig=br["p_sig"],
                     child_sig=st["c_sig"])


def run(substrate, score_fn, archive: Archive, log, inputs, operator, cfg: LoopConfig, proposal=None,
        library=None, seeds=None, behavior_fn=default_behavior, sig_fn=default_signature, callback=None) -> list:
    """Returns per-generation stats. `inputs`: uint8 [S, n_in]; `log` may be None."""
    rng = np.random.default_rng(cfg.seed)
    inputs = np.asarray(inputs, np.uint8)
    n_in = inputs.shape[-1]
    history = []
    if not archive.filled.any():
        if seeds is None:
            n_out = score_fn.n_out if hasattr(score_fn, "n_out") else 1
            seeds = [random_circuit(n_in, n_out, cfg.init_nodes, rng, cfg.init_arity) for _ in range(cfg.n_init)]
        _, _, fit, beh, sig = evaluate(substrate, seeds, inputs, cfg.K, score_fn, library, behavior_fn, sig_fn)
        archive.insert(archive.descriptors(seeds, beh), fit, items=seeds, sigs=sig, lineage=np.arange(len(seeds)))
    for g in range(cfg.generations):
        t0 = time.time()
        br = breed(archive, operator, rng, cfg.batch, cfg.sampling, proposal, cfg.eps)
        desc = fit = sig = None
        if br["ok"].any():
            res, err, fit, beh, sig = evaluate(substrate, br["children"], inputs, cfg.K, score_fn, library,
                                               behavior_fn, sig_fn)
            desc = archive.descriptors(br["children"], beh)
        st = settle(archive, br, desc, fit, sig)
        if log is not None:
            log_edits(log, g, cfg.task, br, st)
        stats = archive.stats()
        fate = st["fate"]
        stats.update(generation=g, seconds=time.time() - t0, invalid=int((~br["ok"]).sum()),
                     accepted=int(((fate == Fate.ELITE_NEW) | (fate == Fate.ELITE_REPLACE)).sum()))
        history.append(stats)
        if callback is not None:
            callback(stats)
    return history
