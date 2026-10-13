"""MAP-Elites archive: a discretized descriptor space, one elite per cell by fitness.

Cells live in dense numpy arrays (fitness, hash, descriptors, behaviour signature, curiosity, lineage, birth), so
insertion of a batch is a handful of vectorized operations and archives of 10^5 to 10^7 cells are fine. Elites
themselves are kept in an object array (any payload; normally `Circuit`s).

Insertion semantics for a batch:
  - candidates with NaN fitness are INVALID;
  - candidates in the same cell compete; only the fittest (first on ties) can enter;
  - it enters if the cell is empty (ELITE_NEW) or its fitness beats the elite (ELITE_REPLACE; `replace_ties`
    also lets equal fitness replace, for neutral drift, unless the hash is the elite's own);
  - every other candidate is CULLED.

Curiosity (Cully & Demiris 2018): each elite carries a score; when a candidate with parent cell p enters the
archive p gains `curiosity[0]`, otherwise it gains `curiosity[1]` (negative). A replaced elite's score resets to
0. Curiosity sampling picks filled cells with probability proportional to (score - min score + 1).

Lineage sampling ("lineage"): pick a surviving lineage uniformly (a lineage is the `lineage` label children inherit
from their parent; seeds are lineages 0..n_init-1; a cell without a label is its own lineage), then a cell of that
lineage by curiosity within the lineage (score - the lineage's min score + 1). Every lineage still in the archive gets
the same share of parents however many cells it holds, so a lineage that falls behind keeps breeding instead of
being sampled out; it still dies when all of its cells are taken by fitter children. No task information is used.
`lineages()` counts the distinct lineages among the elites (Archive.stats reports it).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from ..circuit import codec
from ..circuit.graph import Circuit
from ..circuit.moves import Fate
from . import descriptors as D


@dataclass
class Axis:
    name: str
    lo: float
    hi: float
    bins: int
    source: str = "structural"     # "structural" (computed from the circuit) or "behavior" (supplied by caller)
    log: bool = False              # bin log(1 + x) instead of x

    def bin(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64)
        lo, hi = self.lo, self.hi
        if self.log:
            x, lo, hi = np.log1p(np.maximum(x, 0)), np.log1p(max(lo, 0)), np.log1p(hi)
        b = np.floor((x - lo) / (hi - lo) * self.bins)
        return np.clip(np.nan_to_num(b, nan=0), 0, self.bins - 1).astype(np.int64)


@dataclass
class InsertResult:
    fate: np.ndarray        # int8 [B] Fate
    cell: np.ndarray        # int64 [B]
    displaced: np.ndarray   # uint64 [B, 2] hash of the elite displaced by an ELITE_REPLACE (zeros otherwise)

    @property
    def inserted(self) -> np.ndarray:
        return (self.fate == Fate.ELITE_NEW) | (self.fate == Fate.ELITE_REPLACE)


def hashes_to_array(hashes) -> np.ndarray:
    if len(hashes) == 0:
        return np.zeros((0, 2), np.uint64)
    return np.frombuffer(b"".join(hashes), dtype="<u8").reshape(-1, 2).copy()


def array_to_hash(a) -> bytes:
    return np.asarray(a, "<u8").tobytes()


class Archive:
    MAX_CELLS = 1 << 26

    def __init__(self, axes, sig_dim: int = 0, qd_offset: float = 0.0, replace_ties: bool = False,
                 curiosity=(1.0, -0.5), library=None, task: int = 0):
        self.axes = [a if isinstance(a, Axis) else Axis(**a) for a in axes]
        self.shape = tuple(a.bins for a in self.axes)
        self.n_cells = int(np.prod(self.shape))
        if self.n_cells > self.MAX_CELLS:
            raise ValueError(f"{self.n_cells} cells: use fewer bins (dense storage limit {self.MAX_CELLS})")
        self.sig_dim = sig_dim
        self.qd_offset = qd_offset
        self.replace_ties = replace_ties
        self.curiosity_delta = tuple(curiosity)
        self.library = library
        self.task = task
        n, d = self.n_cells, len(self.axes)
        self.fitness = np.full(n, -np.inf)
        self.filled = np.zeros(n, bool)
        self.hash = np.zeros((n, 2), np.uint64)
        self.desc = np.zeros((n, d), np.float32)
        self.sig = np.zeros((n, sig_dim), np.float32)
        self.curiosity = np.zeros(n, np.float32)
        self.lineage = np.full(n, -1, np.int64)
        self.born = np.full(n, -1, np.int64)            # insertion counter value when the elite entered
        self.items = np.empty(n, object)
        self._s_ix = [j for j, a in enumerate(self.axes) if a.source == "structural"]
        self.sdesc = np.full((n, len(self._s_ix)), np.nan)   # float64 structural descriptors of each elite
        self.n_candidates = 0
        self.n_inserted = 0
        self.n_batches = 0

    # --- descriptors ------------------------------------------------------------------------------------------
    @property
    def names(self) -> list:
        return [a.name for a in self.axes]

    def descriptors(self, circuits, behavior: dict | None = None, nets=None, same_as=None) -> np.ndarray:
        """[B, n_axes] raw descriptor values: structural axes computed (only those the axes use; graph descriptors
        from `nets`, the circuits' flattened NetBatch, when given), behavioral axes from `behavior`.
        same_as [B]: a filled cell whose elite has the same structure as circuit b (a parameter edit's parent), or
        -1; those rows reuse the elite's stored structural values instead of recomputing them."""
        behavior = behavior or {}
        B = len(circuits)
        out = np.zeros((B, len(self.axes)))
        s_ix = self._s_ix
        if s_ix:
            skip = None
            if same_as is not None:
                same_as = np.asarray(same_as, np.int64)
                known = same_as >= 0
                known[known] = ~np.isnan(self.sdesc[same_as[known]]).any(1)
                skip = known if known.any() else None
            vals = D.structural_batch(circuits, [self.axes[j].name for j in s_ix], self.library, nets, skip)
            if skip is not None:
                vals[skip] = self.sdesc[same_as[skip]]
            out[:, s_ix] = vals
        for j, a in enumerate(self.axes):
            if a.source == "behavior":
                if a.name not in behavior:
                    raise KeyError(f"behavioral descriptor {a.name!r} not supplied")
                out[:, j] = np.asarray(behavior[a.name], np.float64).reshape(B)
        return out

    def cells_of(self, desc: np.ndarray) -> np.ndarray:
        desc = np.atleast_2d(desc)
        idx = [a.bin(desc[:, j]) for j, a in enumerate(self.axes)]
        return np.ravel_multi_index(idx, self.shape).astype(np.int64)

    # --- insertion --------------------------------------------------------------------------------------------
    def insert(self, desc, fitness, items=None, hashes=None, sigs=None, lineage=None, parent_cells=None) -> InsertResult:
        """Vectorized batch insertion. desc [B, n_axes]; fitness [B]; items: payloads (Circuits); hashes: list of
        16-byte hashes or uint64 [B, 2] (default: items' .hash, computed only for the candidates that enter, or that
        tie with an elite under replace_ties); sigs [B, sig_dim]; lineage [B]; parent_cells [B] (-1 for none)
        credit curiosity."""
        fitness = np.asarray(fitness, np.float64).reshape(-1)
        B = len(fitness)
        desc = np.asarray(desc, np.float64).reshape(B, len(self.axes))
        lazy = hashes is None and items is not None and B > 0 and isinstance(items[0], Circuit)
        if lazy:
            H = None
        elif hashes is None:
            H = np.zeros((B, 2), np.uint64)
        elif isinstance(hashes, np.ndarray):
            H = hashes.astype(np.uint64).reshape(B, 2)
        else:
            H = hashes_to_array(hashes)

        def H_of(idx):
            if H is not None:
                return H[idx]
            return hashes_to_array([items[i].hash for i in idx.tolist()])
        cells = self.cells_of(desc) if B else np.zeros(0, np.int64)
        fate = np.full(B, Fate.CULLED, np.int8)
        displaced = np.zeros((B, 2), np.uint64)
        valid = ~np.isnan(fitness)
        fate[~valid] = Fate.INVALID
        vi = np.nonzero(valid)[0]
        if len(vi):
            order = vi[np.lexsort((np.arange(len(vi)), -fitness[vi], cells[vi]))]
            cs = cells[order]
            first = np.ones(len(order), bool)
            first[1:] = cs[1:] != cs[:-1]
            win = order[first]
            wc = cells[win]
            cur = self.fitness[wc]
            filled = self.filled[wc]
            better = fitness[win] > cur
            if self.replace_ties:
                tie = (fitness[win] == cur) & filled
                same = np.zeros(len(win), bool)
                if tie.any():
                    same[tie] = (self.hash[wc[tie]] == H_of(win[tie])).all(1)
                better |= tie & ~same
            ins = win[better]
            ic = cells[ins]
            was = self.filled[ic]
            fate[ins] = np.where(was, Fate.ELITE_REPLACE, Fate.ELITE_NEW)
            displaced[ins[was]] = self.hash[ic[was]]
            # curiosity: credit parents first (the credit belongs to the parent individual), then reset replaced
            if parent_cells is not None:
                pc = np.asarray(parent_cells, np.int64).reshape(B)
                ok = (pc >= 0) & valid
                entered = (fate == Fate.ELITE_NEW) | (fate == Fate.ELITE_REPLACE)
                delta = np.where(entered, *self.curiosity_delta).astype(np.float32)
                np.add.at(self.curiosity, pc[ok], delta[ok])
            self.curiosity[ic] = 0.0
            self.fitness[ic] = fitness[ins]
            self.filled[ic] = True
            self.hash[ic] = H_of(ins)
            self.desc[ic] = desc[ins]
            if self._s_ix:
                self.sdesc[ic] = desc[ins][:, self._s_ix]
            if self.sig_dim and sigs is not None:
                self.sig[ic] = np.asarray(sigs, np.float32).reshape(B, self.sig_dim)[ins]
            if lineage is not None:
                self.lineage[ic] = np.asarray(lineage, np.int64).reshape(B)[ins]
            self.born[ic] = self.n_candidates + ins
            if items is not None:
                for i, c in zip(ins.tolist(), ic.tolist()):
                    self.items[c] = items[i]
        self.n_candidates += B
        self.n_inserted += int(((fate == Fate.ELITE_NEW) | (fate == Fate.ELITE_REPLACE)).sum())
        self.n_batches += 1
        return InsertResult(fate, cells, displaced)

    def clear(self) -> None:
        """Empty every cell. Configuration and the candidate / insertion counters are kept (used to rebuild the
        archive when the objective changes: re-score the old elites and insert them again)."""
        self.fitness[:] = -np.inf
        self.filled[:] = False
        self.hash[:] = 0
        self.desc[:] = 0
        self.sig[:] = 0
        self.curiosity[:] = 0
        self.lineage[:] = -1
        self.born[:] = -1
        self.items[:] = None
        self.sdesc[:] = np.nan

    def add(self, circuits, fitness, behavior=None, **kw) -> InsertResult:
        """Compute descriptors and insert."""
        return self.insert(self.descriptors(circuits, behavior), fitness, items=circuits, **kw)

    # --- sampling ---------------------------------------------------------------------------------------------
    def filled_cells(self) -> np.ndarray:
        return np.flatnonzero(self.filled)

    def sample(self, n: int, rng: np.random.Generator, method: str = "uniform") -> np.ndarray:
        cells = self.filled_cells()
        if len(cells) == 0:
            raise ValueError("archive is empty")
        if method == "uniform":
            return cells[rng.integers(0, len(cells), n)]
        if method == "curiosity":
            c = self.curiosity[cells].astype(np.float64)
            w = c - c.min() + 1.0
            return cells[rng.choice(len(cells), n, p=w / w.sum())]
        if method == "lineage":
            lin = self._lineage_ids(cells)
            order = np.lexsort((cells, lin))                    # cells grouped by lineage
            lin_s, cells_s = lin[order], cells[order]
            start = np.flatnonzero(np.r_[True, lin_s[1:] != lin_s[:-1]])
            stop = np.r_[start[1:], len(cells_s)]
            c = self.curiosity[cells_s].astype(np.float64)
            gmin = np.minimum.reduceat(c, start)
            w = c - np.repeat(gmin, stop - start) + 1.0
            cum = np.cumsum(w)
            base = np.r_[0.0, cum][start]
            tot = cum[stop - 1] - base
            g = rng.integers(0, len(start), n)                  # a lineage, uniformly
            u = rng.random(n)
            j = np.searchsorted(cum, base[g] + u * tot[g], side="right")
            return cells_s[np.clip(j, start[g], stop[g] - 1)]
        raise ValueError(f"unknown sampling method {method!r}")

    def _lineage_ids(self, cells) -> np.ndarray:
        lin = self.lineage[cells].copy()
        own = lin < 0
        if own.any():
            lin[own] = np.iinfo(np.int64).min + cells[own]       # unlabelled cells: a lineage each
        return lin

    def lineages(self) -> int:
        """Distinct lineages among the elites."""
        cells = self.filled_cells()
        return int(len(np.unique(self._lineage_ids(cells)))) if len(cells) else 0

    def __getitem__(self, cell):
        return self.items[cell]

    # --- metrics ----------------------------------------------------------------------------------------------
    def coverage(self) -> float:
        return float(self.filled.sum()) / self.n_cells

    def qd_score(self) -> float:
        return float((self.fitness[self.filled] - self.qd_offset).sum())

    def stats(self) -> dict:
        f = self.fitness[self.filled]
        return {"filled": int(self.filled.sum()), "cells": self.n_cells, "coverage": self.coverage(),
                "qd_score": self.qd_score(), "max_fitness": float(f.max()) if len(f) else float("nan"),
                "mean_fitness": float(f.mean()) if len(f) else float("nan"), "candidates": self.n_candidates,
                "inserted": self.n_inserted, "lineages": self.lineages()}

    # --- persistence ------------------------------------------------------------------------------------------
    def save(self, prefix) -> None:
        """Writes <prefix>.npz (filled cells' arrays and encoded circuits) and <prefix>.json (config, counters)."""
        prefix = str(prefix)
        cells = self.filled_cells()
        items = [self.items[c] for c in cells]
        is_c = np.array([isinstance(it, Circuit) for it in items], bool)
        vals, off = codec.encode_many([it if isinstance(it, Circuit) else Circuit(0) for it in items])
        np.savez_compressed(prefix + ".npz", cells=cells, fitness=self.fitness[cells], hash=self.hash[cells],
                            desc=self.desc[cells], sig=self.sig[cells], curiosity=self.curiosity[cells],
                            lineage=self.lineage[cells], born=self.born[cells], item_is_circuit=is_c,
                            circ_values=vals, circ_offsets=off)
        meta = {"axes": [asdict(a) for a in self.axes], "sig_dim": self.sig_dim, "qd_offset": self.qd_offset,
                "replace_ties": self.replace_ties, "curiosity": list(self.curiosity_delta), "task": self.task,
                "n_candidates": self.n_candidates, "n_inserted": self.n_inserted, "n_batches": self.n_batches,
                "codec_version": codec.VERSION}
        with open(prefix + ".json", "w") as f:
            json.dump(meta, f, indent=1)

    @staticmethod
    def load(prefix, library=None) -> "Archive":
        prefix = str(prefix)
        with open(prefix + ".json") as f:
            meta = json.load(f)
        a = Archive(meta["axes"], meta["sig_dim"], meta["qd_offset"], meta["replace_ties"], meta["curiosity"],
                    library, meta["task"])
        z = np.load(prefix + ".npz", allow_pickle=False)
        cells = z["cells"]
        a.filled[cells] = True
        for name in ("fitness", "hash", "desc", "sig", "curiosity", "lineage", "born"):
            getattr(a, name)[cells] = z[name]
        circs = codec.decode_many(z["circ_values"], z["circ_offsets"])
        for c, is_c, circ in zip(cells.tolist(), z["item_is_circuit"].tolist(), circs):
            a.items[c] = circ if is_c else None
        a.n_candidates, a.n_inserted, a.n_batches = meta["n_candidates"], meta["n_inserted"], meta["n_batches"]
        return a
