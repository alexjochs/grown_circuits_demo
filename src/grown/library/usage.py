"""Library usage, decay and vocabulary turnover.

Usage comes from the edit log. INSERT_MODULE rows name their module in the move payload (hash halves). INLINE_MODULE
rows only name the instance's node index, so the module is recovered from a side table written during the wake
phase by `ModuleTracker`, an operator wrapper that records, for every successful mutation, the parent and child
hashes, the move type, the inserted or inlined module and the modules the child instantiates (top level). Tracker
records align one-to-one, in order, with the log rows whose move is valid (rows with move -1 are failed mutations
that never reached the operator's return); the alignment is checked on the child hashes.

Per module and generation the statistics are
  proposed     INSERT_MODULE edits of it (valid moves)
  accepted     ... whose child survived selection (SURVIVED, ELITE_NEW, ELITE_REPLACE)
  elite        ... whose child became an elite (ELITE_NEW, ELITE_REPLACE)
  inlined      INLINE_MODULE edits of it, and `inlined_accepted`
  carried      accepted edits whose child contains an instance of it (the module "appears in an accepted edit"
               without being freshly inserted; needs the tracker)
kept over a sliding window of `window` generations (older generations are folded into totals).

Retirement rule: a module is retired when it is active, was admitted at least `window` generations ago, has no
use in the last `window` generations (use = elite insertions, plus carried appearances when `count_carried`), and
is not a dependency (transitively) of any module that was used in the window. `Library.retire` only stops the
operator from offering it; existing instances keep flattening. Primitives are not library entries and never
retire.

Generations: `grown.loop.run` numbers generations from 0 in every call, so a wake phase's log is read with the
phase's global generation offset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..circuit import codec
from ..circuit.graph import MODULE
from ..circuit.moves import Fate, MoveType

STATS = ("proposed", "accepted", "elite", "inlined", "inlined_accepted", "carried")
_ACCEPTED = (Fate.SURVIVED, Fate.ELITE_NEW, Fate.ELITE_REPLACE)
_ELITE = (Fate.ELITE_NEW, Fate.ELITE_REPLACE)


class ModuleTracker:
    """Wraps an edit operator; records module facts per successful mutation (see module docstring)."""

    def __init__(self, operator):
        self.operator = operator
        self.records: list = []

    def __getattr__(self, k):
        return getattr(self.operator, k)

    def mutate(self, c, rng, proposal=None, eps: float = 0.1):
        child, mv, src = self.operator.mutate(c, rng, proposal, eps)
        ins = inl = b""
        if mv.type == MoveType.INSERT_MODULE:
            ins = codec.ints_to_hash(mv.data[0], mv.data[1])
        elif mv.type == MoveType.INLINE_MODULE:
            inl = c.nodes[mv.node].module
        contained = tuple(sorted({nd.module for nd in child.nodes if nd.op == MODULE}))
        self.records.append((child.hash, int(mv.type), ins, inl, contained))
        return child, mv, src

    def take(self) -> list:
        r, self.records = self.records, []
        return r

    @staticmethod
    def to_json(records) -> list:
        return [[a.hex(), t, b.hex(), c.hex(), [m.hex() for m in ms]] for a, t, b, c, ms in records]

    @staticmethod
    def from_json(rows) -> list:
        return [(bytes.fromhex(a), int(t), bytes.fromhex(b), bytes.fromhex(c), tuple(bytes.fromhex(m) for m in ms))
                for a, t, b, c, ms in rows]


@dataclass
class UsageConfig:
    window: int = 50                 # generations
    count_carried: bool = True       # carried appearances count as use


class Usage:
    def __init__(self, cfg: UsageConfig | None = None):
        self.cfg = cfg or UsageConfig()
        self.generation = 0                    # global generations completed
        self.admitted: dict = {}               # hex -> generation admitted
        self.retired_at: dict = {}             # hex -> generation retired
        self.window: dict = {}                 # hex -> {generation: [counts per STATS]}
        self.totals: dict = {}                 # hex -> [counts per STATS] over all time
        self.last_used: dict = {}              # hex -> last generation with a use
        self.phases: list = []                 # turnover rows

    # --- bookkeeping ------------------------------------------------------------------------------------------
    def note_admissions(self, hashes, generation: int | None = None) -> None:
        g = self.generation if generation is None else generation
        for h in hashes:
            hx = h if isinstance(h, str) else h.hex()
            self.admitted.setdefault(hx, g)
            self.retired_at.pop(hx, None)

    def _bump(self, hx: str, g: int, k: int, n: int = 1) -> None:
        row = self.window.setdefault(hx, {}).setdefault(g, [0] * len(STATS))
        row[k] += n
        tot = self.totals.setdefault(hx, [0] * len(STATS))
        tot[k] += n

    def update(self, log, records=None, offset: int = 0, generations: int | None = None) -> dict:
        """Scan a wake phase's edit log (generations offset by `offset`) and the tracker records of that phase.
        Advances `self.generation` to offset + generations (default: the log's last generation + 1)."""
        cols = ["move", "fate", "generation", "child_h0", "child_h1", "payload"]
        recs = list(records or [])
        ri = 0
        last = -1
        counts = {k: 0 for k in STATS}
        for b in log.scan(columns=cols):
            mv, fate, gen = b["move"].astype(np.int64), b["fate"].astype(np.int64), b["generation"].astype(np.int64)
            off, vals = b["payload_offsets"], b["payload_values"]
            ch = np.stack([b["child_h0"], b["child_h1"]], 1).astype("<u8")
            last = max(last, int(gen.max()) if len(gen) else -1)
            for r in range(len(mv)):
                t = int(mv[r])
                if t < 0:
                    continue
                g = int(gen[r]) + offset
                acc = int(fate[r]) in _ACCEPTED
                eli = int(fate[r]) in _ELITE
                rec = None
                if records is not None:
                    if ri >= len(recs):
                        raise ValueError("tracker records are shorter than the log")
                    rec = recs[ri]
                    ri += 1
                    if rec[0] != ch[r].tobytes():
                        raise ValueError(f"tracker record {ri - 1} does not match log row (child hash)")
                if t == MoveType.INSERT_MODULE:
                    p = vals[off[r]:off[r] + 2]
                    hx = codec.ints_to_hash(int(p[0]), int(p[1])).hex()
                    self._bump(hx, g, 0)
                    counts["proposed"] += 1
                    if acc:
                        self._bump(hx, g, 1)
                        counts["accepted"] += 1
                    if eli:
                        self._bump(hx, g, 2)
                        counts["elite"] += 1
                        self.last_used[hx] = g
                elif t == MoveType.INLINE_MODULE and rec is not None:
                    hx = rec[3].hex()
                    self._bump(hx, g, 3)
                    counts["inlined"] += 1
                    if acc:
                        self._bump(hx, g, 4)
                        counts["inlined_accepted"] += 1
                if rec is not None and acc:
                    for m in rec[4]:
                        hx = m.hex()
                        self._bump(hx, g, 5)
                        counts["carried"] += 1
                        if self.cfg.count_carried:
                            self.last_used[hx] = g
        if records is not None and ri != len(recs):
            raise ValueError(f"{len(recs) - ri} tracker records left over after the log")
        n_gen = generations if generations is not None else last + 1
        self.generation = max(self.generation, offset + n_gen)
        self._prune()
        return counts

    def _prune(self) -> None:
        lo = self.generation - self.cfg.window
        for hx in list(self.window):
            w = self.window[hx]
            for g in [g for g in w if g < lo]:
                del w[g]
            if not w:
                del self.window[hx]

    def in_window(self, hx: str) -> list:
        tot = [0] * len(STATS)
        for row in self.window.get(hx, {}).values():
            for k, v in enumerate(row):
                tot[k] += v
        return tot

    def used(self, hx: str) -> bool:
        w = self.in_window(hx)
        return w[2] > 0 or (self.cfg.count_carried and w[5] > 0)

    # --- retirement -------------------------------------------------------------------------------------------
    def retire(self, library) -> list:
        """Apply the retirement rule. -> list of retired hashes (bytes)."""
        W = self.cfg.window
        active = library.active()
        used = {h for h in library.defs if self.used(h.hex())}
        protected = set()
        stack = list(used)
        while stack:
            h = stack.pop()
            for nd in library.get(h).body.nodes:
                if nd.op == MODULE and nd.module not in protected:
                    protected.add(nd.module)
                    stack.append(nd.module)
        out = []
        for h in active:
            hx = h.hex()
            age = self.generation - self.admitted.get(hx, self.generation)
            if age < W or h in used or h in protected:
                continue
            if h not in library.defs:          # never true for library entries; primitives are not entries
                continue
            library.retire(h)
            self.retired_at[hx] = self.generation
            out.append(h)
        for h in library.defs:
            library.uses[h] = int(self.totals.get(h.hex(), [0] * len(STATS))[2])
        return out

    # --- turnover ---------------------------------------------------------------------------------------------
    def turnover(self, archives, library, phase: int, admitted: int, retired: int) -> dict:
        from .compress import module_depth
        active = library.active()
        ages = [self.generation - self.admitted.get(h.hex(), self.generation) for h in active]
        memo: dict = {}
        depths = [module_depth(library, h, memo) for h in active]
        tokens = mods = elites_with = n_el = 0
        for a in archives:
            for c in a.filled_cells():
                nodes = a.items[c].nodes
                m = sum(nd.op == MODULE for nd in nodes)
                tokens += len(nodes)
                mods += m
                elites_with += m > 0
                n_el += 1
        row = {"phase": phase, "generation": self.generation, "admitted": admitted, "retired": retired,
               "library_active": len(active), "library_total": len(library),
               "median_age": float(np.median(ages)) if ages else float("nan"),
               "max_depth": max(depths, default=0), "mean_depth": float(np.mean(depths)) if depths else 0.0,
               "module_token_frac": mods / tokens if tokens else 0.0,
               "elites_with_modules": elites_with / n_el if n_el else 0.0}
        self.phases.append(row)
        return row

    # --- persistence ------------------------------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({"cfg": {"window": self.cfg.window, "count_carried": self.cfg.count_carried},
                           "generation": self.generation, "admitted": self.admitted, "retired_at": self.retired_at,
                           "window": {h: {str(g): v for g, v in w.items()} for h, w in self.window.items()},
                           "totals": self.totals, "last_used": self.last_used, "phases": self.phases},
                          sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "Usage":
        d = json.loads(s)
        u = Usage(UsageConfig(**d["cfg"]))
        u.generation = d["generation"]
        u.admitted, u.retired_at, u.totals, u.last_used = d["admitted"], d["retired_at"], d["totals"], d["last_used"]
        u.window = {h: {int(g): v for g, v in w.items()} for h, w in d["window"].items()}
        u.phases = d["phases"]
        return u

    def save(self, path) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path) -> "Usage":
        with open(path) as f:
            return Usage.from_json(f.read())
