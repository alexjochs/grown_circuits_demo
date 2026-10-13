"""The compress phase: admit modules that shorten the description of the archives, and rewrite the archives to use
them.

Description length. DL(archives, library) = sum over archive elites of Circuit.description_length() (one token per
node, per pin pointer and per output pointer) + sum over library definitions of their bodies' description length.
Naming a module costs its body once; each replaced occurrence saves (tokens of the occurrence) - (1 + module
inputs). Primitives are free (they are the alphabet). With `dl_params` every parameter value is a token too
(graph.param_tokens: a gate's delays and init, a module instance's explicit slot values), so a parametric module's
instances pay their slot values and an exact module's parameters are paid once in its body.

Mining mode (`mode`, archive.mining): "exact" (token identity, modules without slots, the first version) or
"param" (modulo parameters and gate pin order, circuit.pcanon; every parameter of the body is a slot, defaults are
the occurrences' medians, each rewritten occurrence becomes an instance with its own slot values). A parametric
candidate whose pattern equals an admitted parametric module's (pcanon.pattern_hash of its body) reuses it.

Admission rule (per round):
  1. Mine every elite of every archive together (`archive.mining.mine`: ESU connected sets of k_min..k_max top-level
     tokens, exact or parametric canonical forms, greedy disjoint occurrences per circuit). Lineages are (archive,
     lineage) pairs, so recurrence must span `min_lineages` unrelated lineages.
  2. Walk the candidates by decreasing saving. Drop occurrences that overlap nodes already claimed this round by a
     better candidate; recompute the saving from the remaining occurrences, with a definition cost of zero when the
     module is already in the library. Admit while that net saving is >= max(threshold, rel_threshold * DL) and
     the body passes `check_module` (connected, typed bit ports with every input read, at least one output, every
     node in the backward cone of an output, not a wrapper of a single primitive or a single existing module, no
     zero-delay loop).
  3. Rewrite every elite with a claimed occurrence (`mining.compress`, one occurrence at a time with node indices
     remapped), verify behavior (optional: the synchronous stand-in on random inputs, canonical hashes of the
     flattened netlists, or "phash": canonical hashes of the flattened netlists with every parameter, modulo pin
     order inside gates, which proves the rewrite is the same circuit on any substrate; "auto" is "sim" for exact
     and "phash" for param mode), re-hash and put the elite back in the archive (same cell unless a structural
     axis moved it; a moved elite competes for its new cell as an insertion would).
  4. Re-mine. Later rounds see the new module tokens, so a candidate may contain modules admitted earlier (nested
     admission). Stop when a round admits nothing or after `max_rounds` / `max_admissions`.

The realized change in DL equals the sum of the admitted savings exactly (checked by the tests), unless an
occurrence cannot be rewritten (non-exact canonical form) or fails verification, in which case it is skipped and
the admission records the realized saving.

Events: one `Admission` json line per admission in `<events_dir>/admissions.jsonl` (saving, occurrences, lineages,
module hash, ...) and one line per rewritten elite in `<events_dir>/rewrites.jsonl` (old and new hash), so edit-log
lineages stay traceable across a compress phase.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..archive import descriptors as D
from ..archive.mining import compress, mine, param_body
from ..circuit.canon import circuit_hash
from ..circuit.flatten import flatten, from_netlist
from ..circuit.graph import INPUT, MODULE, WIRE, Circuit, InvalidCircuit, live_nodes, validate
from ..circuit.library import def_hash, param_slots
from ..circuit.pcanon import pattern_hash, pinperm_hash

LINEAGE_SHIFT = 40


@dataclass
class CompressConfig:
    k_min: int = 2
    k_max: int = 4
    min_lineages: int = 2
    threshold: int = 8               # minimum net saving (tokens) to admit
    rel_threshold: float = 0.0       # ... or this fraction of the current DL, whichever is larger
    max_rounds: int = 4
    max_admissions: int = 16         # per phase
    top: int | None = 64             # valid candidates (passing check_module) considered per round
    verify: str = "auto"             # "sim" (stand-in on random inputs), "hash" (flattened netlists), "phash"
                                     # (flattened, modulo gate pin order), "none"; "auto": sim (exact) / phash (param)
    verify_steps: int = 32
    verify_K: int = 4
    seed: int = 0
    expose_delays: bool = False
    mode: str = "exact"              # mining equivalence: "exact" or "param" (archive.mining)
    dl_params: bool = False          # count parameter values as tokens in the description length
    pin_perm: bool = True            # param mode: gates match up to pin order
    workers: int = 1                 # mining pass-1 processes (fork)


@dataclass
class Admission:
    phase: int
    round: int
    hash: str
    name: str
    saving: int                      # realized net saving in tokens (predicted == realized unless skips)
    predicted: int
    per_occurrence: int
    def_cost: int                    # 0 when the module was already in the library
    occurrences: int
    skipped: int
    lineages: int
    circuits: int
    size: int                        # top-level tokens in the body
    n_prims: int
    n_in: int
    n_out: int
    depth: int                       # hierarchy depth (1 = primitives only)
    contains: list                   # module hashes the body instantiates
    reused: bool
    dl_before: int
    dl_after: int
    time: float
    mode: str = "exact"
    n_slots: int = 0


@dataclass
class CompressReport:
    phase: int
    admissions: list = field(default_factory=list)
    rounds: int = 0
    candidates: int = 0
    rejected: dict = field(default_factory=dict)
    dl_before: int = 0
    dl_after: int = 0
    rewritten: int = 0
    moved: int = 0
    lost: int = 0
    verify_failures: int = 0
    seconds: float = 0.0
    mode: str = "exact"
    dl_params: bool = False
    mining: list = field(default_factory=list)      # per round: archive.mining stats

    def summary(self) -> dict:
        d = asdict(self)
        d["admissions"] = len(self.admissions)
        d["saving"] = self.dl_before - self.dl_after
        return d


# --- description length -----------------------------------------------------------------------------------------

def library_dl(library, params: bool = False) -> int:
    return sum(d.body.description_length(params) for d in library.defs.values())


def archive_dl(archives, params: bool = False) -> int:
    return sum(a.items[c].description_length(params) for a in archives for c in a.filled_cells())


def description_length(archives, library, params: bool = False) -> int:
    return archive_dl(archives, params) + library_dl(library, params)


# --- validity ---------------------------------------------------------------------------------------------------

def check_module(body: Circuit, library) -> str | None:
    """None if `body` may be admitted, else the reason it may not."""
    try:
        validate(body, library)
    except InvalidCircuit as e:
        return f"invalid: {e}"
    if body.n_out == 0:
        return "no outputs"
    n = len(body.nodes)
    read = {r[1] for nd in body.nodes for r in nd.ins if r[0] == INPUT}
    if read != set(range(body.n_in)):
        return "unused input port"
    # connected (ignoring direction) over internal edges
    nbr = [set() for _ in range(n)]
    for i, nd in enumerate(body.nodes):
        for s, _ in nd.ins:
            if s != INPUT and s != i:
                nbr[i].add(s)
                nbr[s].add(i)
    seen, stack = set(), [0] if n else []
    while stack:
        v = stack.pop()
        if v not in seen:
            seen.add(v)
            stack.extend(nbr[v] - seen)
    if len(seen) != n:
        return "disconnected"
    live = live_nodes(body)
    if not live.all():
        return "dead nodes (read by no output)"
    try:
        net = flatten(body, library)
    except InvalidCircuit as e:
        return f"invalid: {e}"
    if net.n_gates < 2:
        return "trivial: fewer than two primitive gates"
    core = [nd for nd in body.nodes if nd.op != WIRE]
    if len(core) == 1 and core[0].op == MODULE:
        return "trivial: wrapper of one module"
    if all(int(o) < body.n_in for o in net.outputs):
        return "trivial: outputs are input aliases"
    return None


def module_depth(library, h: bytes, memo=None) -> int:
    memo = {} if memo is None else memo
    if h not in memo:
        memo[h] = 1 + max((module_depth(library, nd.module, memo) for nd in library.get(h).body.nodes
                           if nd.op == MODULE), default=0)
    return memo[h]


# --- rewriting ----------------------------------------------------------------------------------------------------

def rewrite(c: Circuit, occurrences, library, mode: str = "exact", pin_perm: bool = True) -> tuple:
    """Replace disjoint occurrences [(ModuleDef, node list)] in `c`, one at a time through `mining.compress`,
    remapping the node indices of the remaining ones. -> (circuit, per-occurrence success flags)."""
    pending = [(k, d, list(nodes)) for k, (d, nodes) in enumerate(occurrences)]
    ok = [False] * len(pending)
    while pending:
        k, d, nodes = pending.pop(0)
        try:
            c = compress(c, nodes, d, library, mode, pin_perm)
        except (ValueError, InvalidCircuit):
            continue
        ok[k] = True
        dead = sorted(nodes)
        pending = [(k2, d2, [i - int(np.searchsorted(dead, i)) for i in n2]) for k2, d2, n2 in pending]
    return c, ok


def equivalent(old: list, new: list, library, mode: str = "sim", steps: int = 32, K: int = 4, seed: int = 0) -> list:
    """Per pair: True if the rewritten circuit behaves like the original."""
    if mode == "none" or not old:
        return [True] * len(old)
    if mode == "hash":
        return [circuit_hash(from_netlist(flatten(a, library))) == circuit_hash(from_netlist(flatten(b, library)))
                for a, b in zip(old, new)]
    if mode == "phash":
        return [pinperm_hash(a, library) == pinperm_hash(b, library) for a, b in zip(old, new)]
    from ..sim.sync import SyncSubstrate
    sub = SyncSubstrate()
    out = [True] * len(old)
    groups: dict = {}
    for i, c in enumerate(old):
        groups.setdefault((c.n_in, c.n_out), []).append(i)
    rng = np.random.default_rng(seed)
    for (n_in, _), idx in groups.items():
        x = rng.integers(0, 2, (len(idx), steps, n_in)).astype(np.uint8)
        xx = np.concatenate([x, x])
        r = sub.evaluate([old[i] for i in idx] + [new[i] for i in idx], xx, K, library)
        B = len(idx)
        same = ((r.outputs[:B] == r.outputs[B:]).all((1, 2)) & (r.high[:B] == r.high[B:]).all((1, 2))
                & (r.toggles[:B] == r.toggles[B:]))
        for i, ok in zip(idx, same.tolist()):
            out[i] = bool(ok)
    return out


def _struct_axes(archive):
    return [j for j, a in enumerate(archive.axes) if a.source == "structural"]


def replace_elites(archive, cells, circuits, library) -> tuple:
    """Put rewritten elites back. Structural descriptors are recomputed; an elite whose cell changes moves there if
    the cell is empty or it is fitter, and is dropped otherwise. -> (moved, lost)."""
    sj = _struct_axes(archive)
    names = [archive.axes[j].name for j in sj]
    moved = lost = 0
    for cell, c in zip(cells, circuits):
        cell = int(cell)
        new_h = np.frombuffer(c.hash, "<u8")
        if sj:
            old_v = D.structural(archive.items[cell], names, library)
            new_v = D.structural(c, names, library)
        if not sj or np.array_equal(old_v, new_v):
            archive.items[cell] = c
            archive.hash[cell] = new_h
            continue
        desc = archive.desc[cell].astype(np.float64).copy()
        desc[sj] = new_v
        target = int(archive.cells_of(desc[None])[0])
        if target == cell:
            archive.items[cell], archive.hash[cell], archive.desc[cell] = c, new_h, desc
            continue
        if not archive.filled[target] or archive.fitness[cell] > archive.fitness[target]:
            for name in ("fitness", "sig", "curiosity", "lineage", "born"):
                getattr(archive, name)[target] = getattr(archive, name)[cell]
            archive.filled[target] = True
            archive.items[target], archive.hash[target], archive.desc[target] = c, new_h, desc
            moved += 1
        else:
            lost += 1
        archive.filled[cell] = False
        archive.items[cell] = None
        archive.fitness[cell] = -np.inf
        archive.hash[cell] = 0
        archive.lineage[cell] = -1
        archive.born[cell] = -1
        archive.curiosity[cell] = 0
    return moved, lost


# --- the phase ----------------------------------------------------------------------------------------------------

def _index(archives):
    circuits, refs, lineages = [], [], []
    own = 0
    for ai, a in enumerate(archives):
        for c in a.filled_cells().tolist():
            circuits.append(a.items[c])
            refs.append((ai, c))
            lin = int(a.lineage[c])
            if lin < 0:
                own += 1
                lin = -own
                lineages.append(lin)
            else:
                lineages.append((ai << LINEAGE_SHIFT) | lin)
    return circuits, refs, lineages


def _append_jsonl(path, rows) -> None:
    if not path or not rows:
        return
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def compress_phase(archives, library, cfg: CompressConfig | None = None, phase: int = 0, events_dir=None,
                   log=print) -> CompressReport:
    """Run the compress phase in place on `archives` (a list of Archive) and `library`."""
    cfg = cfg or CompressConfig()
    archives = list(archives)
    t_start = time.time()
    rep = CompressReport(phase)
    if cfg.mode not in ("exact", "param"):
        raise ValueError(f"compress mode must be 'exact' or 'param', not {cfg.mode!r}")
    param = cfg.mode == "param"
    verify = cfg.verify if cfg.verify != "auto" else ("phash" if param else "sim")
    rep.mode, rep.dl_params = cfg.mode, cfg.dl_params
    rep.dl_before = dl = description_length(archives, library, cfg.dl_params)
    adm_path = os.path.join(events_dir, "admissions.jsonl") if events_dir else None
    rw_path = os.path.join(events_dir, "rewrites.jsonl") if events_dir else None
    if events_dir:
        os.makedirs(events_dir, exist_ok=True)
    depth_memo: dict = {}
    inv_cache: dict = {}             # pass-1 invariants of elites that a round did not rewrite
    for rnd in range(cfg.max_rounds):
        if len(rep.admissions) >= cfg.max_admissions:
            break
        circuits, refs, lineages = _index(archives)
        if not circuits:
            break
        mstats: dict = {}
        cands = mine(circuits, lineages, k_max=cfg.k_max, k_min=cfg.k_min, min_lineages=cfg.min_lineages,
                     top=None, min_saving=1, mode=cfg.mode, dl_params=cfg.dl_params, library=library,
                     pin_perm=cfg.pin_perm, stats=mstats, workers=cfg.workers, cache=inv_cache)
        rep.mining.append(mstats)
        log(f"compress: round {rnd} mined {mstats['sets']} sets, {mstats['kept_invariants']} kept invariants, "
            f"{mstats['canonical_forms']} canonical forms, {mstats['candidates']} candidates "
            f"({mstats['pass1_seconds']:.0f} + {mstats['pass2_seconds']:.0f} s)")
        rep.candidates += len(cands)
        rep.rounds = rnd + 1
        thr = max(cfg.threshold, cfg.rel_threshold * dl)
        if param:
            patterns = {pattern_hash(d.body, library, cfg.pin_perm): h for h, d in library.defs.items()
                        if d.slots == param_slots(d.body)}
        claimed: dict = {}
        planned: list = []                 # (cand, where, predicted, def_cost, exists, values)
        planned_hashes: set = set()
        n_valid = 0
        for cand in cands:
            if len(rep.admissions) + len(planned) >= cfg.max_admissions:
                break
            reason = check_module(cand.body, library)
            if reason is not None:
                key = reason.split(":")[0] if reason.startswith("invalid") else reason
                rep.rejected[key] = rep.rejected.get(key, 0) + 1
                continue
            n_valid += 1                   # `top` bounds the valid candidates considered (dead-code patterns,
            if cfg.top is not None and n_valid > cfg.top:      # rejected above, do not use it up)
                break
            where, values = {}, []
            for ci, occs in cand.where.items():
                cl = claimed.get(ci, set())
                keep = [j for j, o in enumerate(occs) if cl.isdisjoint(o)]
                if keep:
                    where[ci] = [occs[j] for j in keep]
                    if param:
                        values += [cand.values[ci][j] for j in keep]
            n_occ = sum(len(v) for v in where.values())
            dh = cand.hash if param else def_hash(cand.body, cand.in_types, cand.out_types, ())
            exists = (dh in patterns if param else dh in library.defs) or dh in planned_hashes
            def_cost = 0 if exists else cand.def_cost
            saving = n_occ * cand.per_occurrence - def_cost
            if saving < thr:
                rep.rejected["below threshold"] = rep.rejected.get("below threshold", 0) + 1
                continue
            for ci, occs in where.items():
                cl = claimed.setdefault(ci, set())
                for o in occs:
                    cl.update(o)
            planned.append((cand, where, saving, def_cost, exists, values))
            planned_hashes.add(dh)
        if not planned:
            break
        # admit, then rewrite each touched elite once with all of its occurrences
        defs = []
        for cand, where, saving, def_cost, exists, values in planned:
            if param and cand.hash in patterns:
                mdef = library.get(patterns[cand.hash])
                library.retired.discard(mdef.hash)
            elif param:
                body, slots = param_body(cand, values)
                mdef = library.admit(body, cand.in_types, cand.out_types, slots=slots)
                patterns[cand.hash] = mdef.hash
            else:
                mdef = library.admit(cand.body, cand.in_types, cand.out_types, expose_delays=cfg.expose_delays)
            defs.append(mdef)
        per_circuit: dict = {}
        for k, (cand, where, *_rest) in enumerate(planned):
            for ci, occs in where.items():
                for o in occs:
                    per_circuit.setdefault(ci, []).append((k, o))
        skip = np.zeros(len(planned), np.int64)
        touched, olds, news = [], [], []
        for ci, items in sorted(per_circuit.items()):
            c1, flags = rewrite(circuits[ci], [(defs[k], o) for k, o in items], library, cfg.mode, cfg.pin_perm)
            ok_k = []
            for (k, _), f in zip(items, flags):
                if f:
                    ok_k.append(k)
                else:
                    skip[k] += 1
            touched.append((ci, ok_k))
            olds.append(circuits[ci])
            news.append(c1)
        ok = equivalent(olds, news, library, verify, cfg.verify_steps, cfg.verify_K, cfg.seed + rnd)
        by_archive: dict = {}
        rewrites = []
        for (ci, ks), c0, c1, good in zip(touched, olds, news, ok):
            if not good:
                rep.verify_failures += 1
                for k in ks:
                    skip[k] += 1
                log(f"compress: rewrite of elite {refs[ci]} failed verification; kept the original")
                continue
            ai, cell = refs[ci]
            by_archive.setdefault(ai, ([], []))
            by_archive[ai][0].append(cell)
            by_archive[ai][1].append(c1)
            rewrites.append({"phase": phase, "round": rnd, "archive": ai, "cell": int(cell), "old": c0.hash.hex(),
                             "new": c1.hash.hex()})
        for ai, (cells, cs) in by_archive.items():
            mv, lo = replace_elites(archives[ai], cells, cs, library)
            rep.moved += mv
            rep.lost += lo
        rep.rewritten += len(rewrites)
        _append_jsonl(rw_path, rewrites)
        # realized savings: recompute exactly from the occurrence counts that were rewritten
        events = []
        for k, (cand, where, predicted, def_cost, exists, _values) in enumerate(planned):
            mdef = defs[k]
            n_occ = sum(len(v) for v in where.values())
            realized_occ = n_occ - int(skip[k])
            saving = realized_occ * cand.per_occurrence - def_cost
            dl_after = dl - saving
            events.append(Admission(
                phase, rnd, mdef.hash.hex(), mdef.name, int(saving), int(predicted), cand.per_occurrence, def_cost,
                realized_occ, int(skip[k]), cand.lineages, len(where), len(cand.body.nodes), mdef.n_prims,
                mdef.n_in, mdef.n_out, module_depth(library, mdef.hash, depth_memo),
                sorted({nd.module.hex() for nd in cand.body.nodes if nd.op == MODULE}), exists, int(dl),
                int(dl_after), time.time(), cfg.mode, len(mdef.slots)))
            dl = dl_after
        rep.admissions += events
        _append_jsonl(adm_path, [asdict(e) for e in events])
        for e in events:
            log(f"compress: admitted {e.name} ({e.size} tokens, {e.n_prims} prims, {e.n_in}->{e.n_out}, depth "
                f"{e.depth}) x{e.occurrences} in {e.circuits} elites / {e.lineages} lineages, saving {e.saving}"
                + (" (reused)" if e.reused else ""))
    rep.dl_after = description_length(archives, library, cfg.dl_params)
    rep.seconds = time.time() - t_start
    return rep
