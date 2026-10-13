"""Canonical-cell instrument: does a library module do what a human building block does?

Every module is flattened (with the library) and simulated in isolation on the stand-in semantics: a synchronous
unit-delay network, y_g(t+1) = f_g(nets at t), as in `grown.sim.sync`. Classification is behavioral only; the
structure of the module is used just to name a ring oscillator.

**Fundamental-mode machine.** Inputs are held until the module settles. Settling from internal state s under
input x iterates the unit-delay update until a state repeats: period 1 is a stable state; a longer period is an
oscillation. An oscillation from which some stable state is reachable when gates fire one at a time (the
asynchronous, any-delay model) is a race that real delays resolve (two cross-coupled NANDs released together in
lockstep); its outcome is the set of those stable states (found by 32 random firing walks, plus an exhaustive
search of up to 4096 states for modules of <= 16 gates) and the module is nondeterministic there. An oscillation
with no reachable stable state is genuine (OSC); its period is taken from the limit cycle reached from the calmest
asynchronously reachable state (fewest excited gates), the mode real delays sustain (2L ticks for an L-stage ring,
not the unit-delay lockstep). A genuine oscillation that no output sees is treated as settled (flagged
`internal_oscillation`). Total states are pairs (stable internal state, input vector);
the output is read from a total state. Inputs change one bit at a time (single-input change, the textbook
assumption under which latches and flip-flops are specified). The reachable machine is explored breadth first from
the power-up states (the module's own init values settled under every input vector), bounded by `max_states` total
states and `max_ticks` per settle. Its size is reported after minimization (partition refinement over outputs and
successor classes) as `fsm_states`: the largest number of distinct classes among total states that share an input
vector, i.e. the number of memory states (1 = combinational). Modules with more than 8 inputs are not simulated.

**Combinational modules** (fsm_states == 1, no oscillation) are classified by their settled truth tables, reduced
to the inputs they depend on: the exact name up to input permutation (P class: NAND2, MAJ3, ...) and the NPN class
(input permutation, input negation, output negation). Multi-output modules are matched jointly (one input
transform for all outputs) against multi-output cells (half and full adder), with output permutation and negation;
a module whose outputs are all catalog functions but not one joint cell is a `composite`. Modules whose settled
function is trivial (a constant, or one input passed or inverted) are then examined tick by tick: a delay line
reproduces one input exactly L >= 2 ticks later for any input stream; an edge detector emits one pulse per rising
(falling, or any) edge of one input and nothing on the other inputs' edges.

**Sequential modules** are matched against catalog machines up to input permutation, input negation, state
relabeling and an output map that sends every module output to a catalog output, possibly negated (so a module
that also exports Q-bar still matches a latch). The check is a product exploration of total-state pairs from the
power-up states over the catalog's allowed inputs (an SR latch's S = R = 1 is excluded); it succeeds when every
reachable pair agrees under one output map, which for strongly connected catalog machines is trace equivalence.
A module matches when this holds from at least one of its power-up states (a circuit implements a cell from a
reset state; a Johnson counter powered into its unused cycle is a T flip-flop there); the number of conforming
power-up states is reported, and among several matching cells the one that explains the most power-up states wins
(then cells without forbidden inputs, then the larger machine: a C-element restricted to three input vectors is an
SR latch). A module that matches on a subset of its outputs is reported as `partial`.

**Oscillators.** A genuine oscillation with a toggling output under some held input is `clock-like` (free-running if
it oscillates under every input, gated otherwise). If the flattened module has exactly one cyclic component, that
component is a simple cycle of L gates and the period is 2L ticks with a 50% duty cycle, it is a ring oscillator.

**freerun (optional).** With `freerun=True` the module is also run on freerun (CPU, its real per-arc delays): a
random single-input-change stimulus with long holds is compared with the fundamental-mode prediction (agreement
fraction), and delay-line latency and oscillation period are measured in freerun ticks.
"""
from __future__ import annotations

import itertools
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np

from ..archive.descriptors import _cyclic, _gate_graph, scc
from ..circuit.flatten import Netlist, flatten
from ..circuit.graph import MODULE

MAX_STATES = 4096
MAX_ASYNC = 4096
MAX_INPUTS = 8       # exhaustive simulation over 2^n input vectors
MAX_NPN = 6          # NPN canonical forms by enumeration (6! * 2^6 * 2 transforms)


# --- unit-delay model ---------------------------------------------------------------------------------------------

class UnitDelay:
    """The stand-in semantics of one flattened netlist, one state at a time."""

    def __init__(self, net: Netlist):
        self.net = net
        self.I, self.G = net.n_inputs, net.n_gates
        A = max([int(net.arity.max()) if self.G else 0, 1])
        self.fan = np.full((self.G, A), self.I + self.G, np.intp)     # padding pins read a constant-0 net
        for g in range(self.G):
            s = net.fanin[net.fanin_ptr[g]:net.fanin_ptr[g + 1]]
            self.fan[g, :len(s)] = s
        self.shift = np.arange(A, dtype=np.uint64)
        self.tt = net.tt.astype(np.uint64)
        self.init = net.init[self.I:].astype(np.uint8)
        self.outputs = net.outputs.astype(np.intp)

    def xvec(self, x: int) -> np.ndarray:
        return np.array([(x >> i) & 1 for i in range(self.I)], np.uint8)

    def vals(self, s: np.ndarray, xv: np.ndarray) -> np.ndarray:
        return np.concatenate([xv, s, np.zeros(1, np.uint8)])

    def tick(self, s: np.ndarray, xv: np.ndarray) -> np.ndarray:
        if not self.G:
            return s
        v = self.vals(s, xv)
        idx = (v[self.fan].astype(np.uint64) << self.shift).sum(1, dtype=np.uint64)
        return ((self.tt >> idx) & np.uint64(1)).astype(np.uint8)

    def out(self, s: np.ndarray, xv: np.ndarray) -> tuple:
        return tuple(int(b) for b in self.vals(s, xv)[self.outputs])


@dataclass(frozen=True)
class Osc:
    period: int
    wave: tuple            # output vectors over one period, starting at the smallest state of the cycle
    resolved: bool         # False when the asynchronous search for a stable state hit its bound

    @property
    def toggling(self) -> tuple:
        return tuple(j for j in range(len(self.wave[0])) if len({w[j] for w in self.wave}) > 1) if self.wave else ()


class ModuleFSM:
    """Fundamental-mode machine of a netlist (see module docstring). States are bytes of gate values; inputs are
    ints (bit i = input i)."""

    def __init__(self, net: Netlist, max_states: int = MAX_STATES, max_ticks: int | None = None):
        self.m = UnitDelay(net)
        self.I, self.G = self.m.I, self.m.G
        self.max_states = max_states
        self.max_ticks = max_ticks or max(64, 8 * self.G + 8)
        self._cache: dict = {}
        self._outs: dict = {}
        self.truncated = False
        self.races = 0
        self.internal_osc = 0

    def arr(self, s: bytes) -> np.ndarray:
        return np.frombuffer(s, np.uint8)

    def out(self, s: bytes, x: int) -> tuple:
        k = (s, x)
        o = self._outs.get(k)
        if o is None:
            o = self._outs[k] = self.m.out(self.arr(s), self.m.xvec(x))
        return o

    def settle(self, s: bytes, x: int):
        """-> ("ok", tuple of stable states) or ("osc", Osc)."""
        key = (s, x)
        r = self._cache.get(key)
        if r is not None:
            return r
        xv = self.m.xvec(x)
        cur = self.arr(s)
        seen = {s: 0}
        traj = [s]
        r = None
        for t in range(1, self.max_ticks + 1):
            cur = self.m.tick(cur, xv)
            b = cur.tobytes()
            if b in seen:
                start = seen[b]
                if t - start == 1:
                    r = ("ok", (b,))
                else:
                    r = self._oscillation(traj[start:], x)
                break
            seen[b] = t
            traj.append(b)
        if r is None:                                       # no repeat within the bound: treat as oscillating
            self.truncated = True
            r = ("osc", Osc(0, (), False))
        self._cache[key] = r
        return r

    def _oscillation(self, cycle: list, x: int):
        fixed, complete, calm = self._async_resolve(cycle[0], x)
        if fixed:
            self.races += 1
            return ("ok", tuple(sorted(fixed)))
        # genuine: report the limit cycle reached from the calmest asynchronously reachable state (fewest excited
        # gates), which is the oscillation real delays sustain (a ring's 2L-tick mode, not unit-delay lockstep)
        cyc = self._sync_cycle(calm, x) or cycle
        i0 = min(range(len(cyc)), key=lambda i: cyc[i])
        cyc = cyc[i0:] + cyc[:i0]
        osc = Osc(len(cyc), tuple(self.out(s, x) for s in cyc), complete)
        if not osc.toggling:
            # an internal oscillation the outputs never see: treat the cycle (represented by its smallest state) as
            # settled; its phase when the next input arrives is arbitrary in a real circuit
            self.internal_osc += 1
            return ("ok", (cyc[0],))
        return ("osc", osc)

    def _sync_cycle(self, s: bytes, x: int):
        xv = self.m.xvec(x)
        cur = self.arr(s)
        seen = {s: 0}
        traj = [s]
        for t in range(1, self.max_ticks + 1):
            cur = self.m.tick(cur, xv)
            b = cur.tobytes()
            if b in seen:
                return traj[seen[b]:] if t - seen[b] > 1 else None
            seen[b] = t
            traj.append(b)
        return None

    def _excited(self, a: np.ndarray, xv) -> np.ndarray:
        return np.nonzero(self.m.tick(a, xv) != a)[0]

    def _async_resolve(self, s0: bytes, x: int, walks: int = 32):
        """Stable states reachable when one excited gate fires at a time (the any-delay model). Random firing walks
        find them; for small modules a bounded breadth-first search makes the set exact. -> (stable states, search
        exhaustive, calmest state seen)."""
        import hashlib
        xv = self.m.xvec(x)
        rng = np.random.default_rng(int.from_bytes(hashlib.blake2b(s0 + bytes([x & 255, x >> 8 & 255]),
                                                                   digest_size=8).digest(), "little"))
        fixed = set()
        calm, calm_n = s0, len(self._excited(self.arr(s0), xv))
        steps = 32 * self.G + 64
        for _ in range(walks):
            a = self.arr(s0).copy()
            for _ in range(steps):
                exc = self._excited(a, xv)
                if not len(exc):
                    fixed.add(a.tobytes())
                    break
                if len(exc) < calm_n:
                    calm, calm_n = a.tobytes(), len(exc)
                a[exc[rng.integers(len(exc))]] ^= 1
        complete = False
        if self.G <= 16:
            seen = {s0}
            q = deque([s0])
            complete = True
            while q:
                st = q.popleft()
                a = self.arr(st)
                exc = self._excited(a, xv)
                if not len(exc):
                    fixed.add(st)
                    continue
                for g in exc.tolist():
                    b = a.copy()
                    b[g] ^= 1
                    bb = b.tobytes()
                    if bb not in seen:
                        if len(seen) >= MAX_ASYNC:
                            complete = False
                            q.clear()
                            break
                        seen.add(bb)
                        q.append(bb)
        return fixed, complete, calm

    def power_up(self, x: int):
        return self.settle(self.m.init.tobytes(), x)

    # --- full exploration ------------------------------------------------------------------------------------
    def explore(self) -> "Explored":
        """Reachable total states over every input vector with single-input changes."""
        n_x = 1 << self.I
        states: dict = {}          # (s, x) -> index
        succ: list = []            # per total state: per input bit: tuple of successor indices, or Osc
        oscs: list = []            # (x, Osc, reached from power-up or transition)
        q = deque()

        def add(s, x):
            k = (s, x)
            if k not in states:
                if len(states) >= self.max_states:
                    self.truncated = True
                    return None
                states[k] = len(states)
                succ.append(None)
                q.append(k)
            return states[k]

        power = []
        for x in range(n_x):
            r = self.power_up(x)
            if r[0] == "osc":
                oscs.append((x, r[1]))
                power.append(r[1])
            else:
                power.append(tuple(i for i in (add(s, x) for s in r[1]) if i is not None))
        while q:
            s, x = q.popleft()
            row = []
            for i in range(self.I):
                x2 = x ^ (1 << i)
                r = self.settle(s, x2)
                if r[0] == "osc":
                    oscs.append((x2, r[1]))
                    row.append(r[1])
                else:
                    row.append(tuple(j for j in (add(t, x2) for t in r[1]) if j is not None))
            succ[states[(s, x)]] = row
        keys = list(states)
        return Explored(self, keys, [self.out(s, x) for s, x in keys], succ, power, oscs)


@dataclass
class Explored:
    fsm: ModuleFSM
    keys: list                  # (state bytes, x)
    outs: list
    succ: list
    power: list                 # per x: tuple of state indices, or Osc
    oscs: list

    def minimize(self) -> list:
        """Partition refinement: class id per total state."""
        n = len(self.keys)
        cls = [hash((self.keys[i][1], self.outs[i])) for i in range(n)]
        n_cls = len(set(cls))
        while True:
            sig = []
            for i in range(n):
                row = self.succ[i] or ()
                sig.append((cls[i], tuple(("osc", r.period, r.wave) if isinstance(r, Osc)
                                          else frozenset(cls[j] for j in r) for r in row)))
            ids: dict = {}
            new = [ids.setdefault(s, len(ids)) for s in sig]
            if len(ids) == n_cls:
                return new
            cls, n_cls = new, len(ids)

    def memory_states(self) -> int:
        if not self.keys:
            return 0
        cls = self.minimize()
        per_x: dict = {}
        for (s, x), c in zip(self.keys, cls):
            per_x.setdefault(x, set()).add(c)
        return max(len(v) for v in per_x.values())

    def truth_table(self) -> list:
        """Settled output per input vector (valid when memory_states() == 1 and nothing oscillates)."""
        by_x = {}
        for (s, x), o in zip(self.keys, self.outs):
            by_x.setdefault(x, o)
        return [by_x[x] for x in range(1 << self.fsm.I)]


# --- boolean functions: P and NPN classes -------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _rowmaps(n: int):
    """[T, 2^n] source row for every (permutation, input negation) and the flags per transform."""
    rows = np.arange(1 << n)
    bits = (rows[:, None] >> np.arange(n)) & 1                          # [R, n]
    maps, meta = [], []
    for perm in itertools.permutations(range(n)):
        for neg in range(1 << n):
            nb = (neg >> np.arange(n)) & 1
            src_bits = bits[:, list(perm)] ^ nb                          # input i of the transformed function
            maps.append((src_bits << np.arange(n)).sum(1))               # reads input perm[i] of the original
            meta.append((perm, neg))
    return np.array(maps, np.int64), meta


def tt_bits(tt: int, n: int) -> np.ndarray:
    return np.array([(tt >> r) & 1 for r in range(1 << n)], np.uint8)


def bits_tt(b) -> int:
    return int(sum(int(v) << r for r, v in enumerate(b)))


def _pack(b: np.ndarray) -> np.ndarray:
    """[T, R] bits -> python ints per row (R <= 64), as an object array for exact comparison."""
    w = np.uint64(1) << np.arange(b.shape[1], dtype=np.uint64)
    return (b.astype(np.uint64) * w).sum(1, dtype=np.uint64)


def p_canon(tt: int, n: int) -> int:
    maps, meta = _rowmaps(n)
    keep = [i for i, (_, neg) in enumerate(meta) if neg == 0]
    return int(_pack(tt_bits(tt, n)[maps[keep]]).min())


def npn_canon(tt: int, n: int) -> int:
    maps, _ = _rowmaps(n)
    b = tt_bits(tt, n)[maps]
    return int(min(_pack(b).min(), _pack(1 - b).min()))


def support(tts: list, n: int) -> list:
    """Inputs any of the truth tables depends on."""
    out = []
    for i in range(n):
        rows = np.arange(1 << n)
        lo = rows[(rows >> i) & 1 == 0]
        if any(any((tt >> int(r)) & 1 != (tt >> int(r | (1 << i))) & 1 for r in lo) for tt in tts):
            out.append(i)
    return out


def restrict(tt: int, n: int, keep: list) -> int:
    """Truth table over inputs `keep` (others at 0; valid when the others are irrelevant)."""
    out = 0
    for r in range(1 << len(keep)):
        src = sum(((r >> j) & 1) << i for j, i in enumerate(keep))
        out |= ((tt >> src) & 1) << r
    return out


def _f(n, fn) -> int:
    return bits_tt([fn(*[(r >> i) & 1 for i in range(n)]) & 1 for r in range(1 << n)])


def _and(*a):
    return int(all(a))


def _or(*a):
    return int(any(a))


def _xor(*a):
    return sum(a) & 1


# single-output combinational catalog: (name, n, truth table, family)
COMB = [("CONST0", 0, 0, "constant"), ("CONST1", 0, 1, "constant"),
        ("BUF", 1, 0b10, "buffer/inverter"), ("NOT", 1, 0b01, "buffer/inverter")]
COMB += [("AND2", 2, _f(2, _and), "AND/OR"), ("NAND2", 2, _f(2, lambda a, b: 1 - _and(a, b)), "AND/OR"),
         ("OR2", 2, _f(2, _or), "AND/OR"), ("NOR2", 2, _f(2, lambda a, b: 1 - _or(a, b)), "AND/OR"),
         ("ANDN2 (a and not b)", 2, _f(2, lambda a, b: a & (1 - b)), "AND/OR"),
         ("ORN2 (a or not b)", 2, _f(2, lambda a, b: a | (1 - b)), "AND/OR"),
         ("XOR2", 2, _f(2, _xor), "parity"), ("XNOR2", 2, _f(2, lambda a, b: 1 - _xor(a, b)), "parity")]
for _n in range(3, 7):
    COMB += [(f"AND{_n}", _n, _f(_n, _and), "AND/OR"), (f"NAND{_n}", _n, _f(_n, lambda *a: 1 - _and(*a)), "AND/OR"),
             (f"OR{_n}", _n, _f(_n, _or), "AND/OR"), (f"NOR{_n}", _n, _f(_n, lambda *a: 1 - _or(*a)), "AND/OR"),
             (f"XOR{_n} (parity tree)", _n, _f(_n, _xor), "parity"),
             (f"XNOR{_n} (parity tree)", _n, _f(_n, lambda *a: 1 - _xor(*a)), "parity")]
COMB += [("MAJ3", 3, _f(3, lambda a, b, c: (a & b) | (a & c) | (b & c)), "majority"),
         ("MIN3 (inverted majority)", 3, _f(3, lambda a, b, c: 1 - ((a & b) | (a & c) | (b & c))), "majority"),
         ("MUX2 (s ? b : a)", 3, _f(3, lambda s, a, b: b if s else a), "multiplexer"),
         ("AO21 (a.b + c)", 3, _f(3, lambda a, b, c: (a & b) | c), "and-or"),
         ("AOI21", 3, _f(3, lambda a, b, c: 1 - ((a & b) | c)), "and-or"),
         ("OA21 ((a + b).c)", 3, _f(3, lambda a, b, c: (a | b) & c), "and-or"),
         ("OAI21", 3, _f(3, lambda a, b, c: 1 - ((a | b) & c)), "and-or"),
         ("AO22 (a.b + c.d)", 4, _f(4, lambda a, b, c, d: (a & b) | (c & d)), "and-or"),
         ("AOI22", 4, _f(4, lambda a, b, c, d: 1 - ((a & b) | (c & d))), "and-or"),
         ("OA22", 4, _f(4, lambda a, b, c, d: (a | b) & (c | d)), "and-or"),
         ("OAI22", 4, _f(4, lambda a, b, c, d: 1 - ((a | b) & (c | d))), "and-or"),
         ("MUX4", 6, _f(6, lambda s0, s1, a, b, c, d: (a, b, c, d)[s0 | (s1 << 1)]), "multiplexer")]

# multi-output combinational cells: (name, n, [truth tables])
MULTI = [("half adder", 2, [_f(2, _xor), _f(2, _and)]),
         ("full adder", 3, [_f(3, _xor), _f(3, lambda a, b, c: (a & b) | (a & c) | (b & c))])]


@lru_cache(maxsize=None)
def _comb_index():
    p_names, npn_fams = {}, {}
    for name, n, tt, fam in COMB:
        p_names.setdefault((n, p_canon(tt, n)), name)
        npn_fams.setdefault((n, npn_canon(tt, n)), f"{fam} ({n}-input)" if n > 1 else fam)
    return p_names, npn_fams


def name_function(tt: int, n: int) -> dict:
    """Name a single-output function (already reduced to its support)."""
    if n > MAX_NPN:
        return {"n": n, "tt": hex(tt), "npn": None, "name": None, "npn_class": None, "human": False}
    p_names, npn_fams = _comb_index()
    p = p_names.get((n, p_canon(tt, n)))
    npn = npn_canon(tt, n)
    fam = npn_fams.get((n, npn))
    return {"n": n, "tt": hex(tt), "npn": hex(npn), "name": p, "npn_class": fam, "human": fam is not None}


def match_multi(tts: list, n: int):
    """Joint match of a multi-output function (over its support) to a multi-output cell. -> (cell, transform)."""
    maps, meta = _rowmaps(n)
    m_bits = [tt_bits(t, n)[maps] for t in tts]                         # per output [T, R]
    m_int = [_pack(b) for b in m_bits]
    m_neg = [_pack(1 - b) for b in m_bits]
    for name, cn, ctts in MULTI:
        if cn != n:
            continue
        cint = [np.uint64(bits_tt(tt_bits(t, n))) for t in ctts]
        for ti in range(len(meta)):
            cand = []
            for j in range(len(tts)):
                cand.append([(k, ng) for k, c in enumerate(cint) for ng, v in ((0, m_int[j][ti]), (1, m_neg[j][ti]))
                             if v == c])
            assign = _surjection(cand, len(ctts))
            if assign is not None:
                perm, neg = meta[ti]
                return name, {"input_perm": list(perm), "input_neg": neg, "outputs": assign}
    return None


def _surjection(cand: list, m: int):
    """Pick one (k, neg) per module output so that every catalog output 0..m-1 is hit. None if impossible."""
    if any(not c for c in cand):
        return None
    for choice in itertools.product(*cand):
        if {k for k, _ in choice} == set(range(m)):
            return [list(c) for c in choice]
    return None


# --- sequential catalog -------------------------------------------------------------------------------------------

@dataclass
class SeqCell:
    """A catalog machine in fundamental mode. `delta(q, x_old, x_new) -> q`, `out(q, x) -> tuple`; inputs are bit
    tuples in the cell's port order. `allowed(x)` excludes forbidden input vectors."""
    name: str
    n_in: int
    n_out: int
    states: tuple
    delta: object
    out: object
    allowed: object = None
    ports: tuple = ()

    def ok(self, x: tuple) -> bool:
        return self.allowed is None or bool(self.allowed(x))

    def stable(self, q, x) -> bool:
        return self.delta(q, x, x) == q


def _rise(xo, xn, i):
    return xo[i] == 0 and xn[i] == 1


def _counter(bits, code="binary"):
    n_st = (1 << bits) if code != "johnson" else 2 * bits

    def word(q):
        if code == "binary":
            return tuple((q >> b) & 1 for b in range(bits))
        if code == "gray":
            g = q ^ (q >> 1)
            return tuple((g >> b) & 1 for b in range(bits))
        return tuple(1 if (q > b and q <= b + bits) else 0 for b in range(bits))     # johnson ring

    return SeqCell(f"{bits}-bit {code} counter", 1, bits, tuple(range(n_st)),
                   lambda q, xo, xn: (q + 1) % n_st if _rise(xo, xn, 0) else q,
                   lambda q, x: word(q), ports=("clk",))


SEQ = [
    SeqCell("SR latch", 2, 1, (0, 1),
            lambda q, xo, xn: 1 if xn == (1, 0) else 0 if xn == (0, 1) else q,
            lambda q, x: (q,), allowed=lambda x: x != (1, 1), ports=("S", "R")),
    SeqCell("gated D latch", 2, 1, (0, 1), lambda q, xo, xn: xn[0] if xn[1] else q, lambda q, x: (q,),
            ports=("D", "G")),
    SeqCell("D flip-flop (edge-triggered)", 2, 1, (0, 1),
            lambda q, xo, xn: xn[0] if _rise(xo, xn, 1) else q, lambda q, x: (q,), ports=("D", "clk")),
    SeqCell("T flip-flop (toggle on edge)", 1, 1, (0, 1), lambda q, xo, xn: 1 - q if _rise(xo, xn, 0) else q,
            lambda q, x: (q,), ports=("clk",)),
    SeqCell("T flip-flop with enable", 2, 1, (0, 1),
            lambda q, xo, xn: q ^ xn[0] if _rise(xo, xn, 1) else q, lambda q, x: (q,), ports=("T", "clk")),
    SeqCell("JK flip-flop", 3, 1, (0, 1),
            lambda q, xo, xn: ((xn[0] & (1 - q)) | ((1 - xn[1]) & q)) if _rise(xo, xn, 2) else q,
            lambda q, x: (q,), ports=("J", "K", "clk")),
    SeqCell("Muller C-element", 2, 1, (0, 1), lambda q, xo, xn: xn[0] if xn[0] == xn[1] else q,
            lambda q, x: (q,), ports=("A", "B")),
    _counter(2, "binary"), _counter(3, "binary"), _counter(2, "gray"), _counter(3, "johnson"),
]


def _xbits(x: int, n: int) -> tuple:
    return tuple((x >> i) & 1 for i in range(n))


def _xint(bits) -> int:
    return sum(int(b) << i for i, b in enumerate(bits))


def match_sequential(fsm: ModuleFSM, cell: SeqCell, n_out: int, max_pairs: int = 20000):
    """-> (status, transform) with status "match" (every module output explained) or "partial", or None."""
    n = cell.n_in
    if fsm.I != n:
        return None
    allowed = [xc for xc in range(1 << n) if cell.ok(_xbits(xc, n))]
    best_partial = None
    for perm in itertools.permutations(range(n)):
        for neg in range(1 << n):
            def xm(xc, perm=perm, neg=neg):
                b = _xbits(xc ^ neg, n)
                return sum(b[i] << perm[i] for i in range(n))
            res = _match_transform(fsm, cell, n_out, allowed, xm, max_pairs)
            if res is None:
                continue
            status, outmap, n_ok, n_start = res
            tf = {"input_perm": list(perm), "input_neg": neg, "outputs": outmap, "powerup_conforming": n_ok,
                  "powerup_states": n_start}
            if status == "match":
                return "match", tf
            best_partial = best_partial or ("partial", tf)
    return best_partial


def _match_transform(fsm, cell, n_out, allowed, xm, max_pairs):
    n = cell.n_in
    # power-up: the first allowed input fixes the pairing; the others are checked against it below
    groups = []
    for xc in allowed:
        r = fsm.power_up(xm(xc))
        if r[0] == "osc":
            return None
        groups.append((xc, r[1]))
    starts = [(xc, m) for xc, ms in groups for m in ms]
    partial = None
    for k, (xc0, m0) in enumerate(starts):
        for q0 in [q for q in cell.states if cell.stable(q, _xbits(xc0, n))]:
            pairs = _product(fsm, cell, [(m0, xc0, q0)], allowed, xm, max_pairs)
            if pairs is None:
                continue
            outmap = _output_map(pairs, n_out, cell.n_out)
            if outmap is None:
                continue
            status, omap = outmap
            if status == "partial":
                partial = partial or ("partial", omap, 0, len(starts))
                continue
            n_ok = 1 + sum(_conforms(fsm, cell, st, allowed, xm, omap, max_pairs) for st in starts[:k] + starts[k + 1:])
            return status, omap, n_ok, len(starts)
    return partial


def _product(fsm, cell, start, allowed, xm, max_pairs):
    """Explore (module state, input, catalog state) triples. -> list of (module outputs, catalog outputs), or None
    when the module oscillates where the catalog does not."""
    n = cell.n_in
    allowed_set = set(allowed)
    seen = set(start)
    q = deque(start)
    obs = []
    while q:
        m, xc, c = q.popleft()
        xb = _xbits(xc, n)
        obs.append((fsm.out(m, xm(xc)), cell.out(c, xb)))
        for i in range(n):
            xc2 = xc ^ (1 << i)
            if xc2 not in allowed_set:
                continue
            r = fsm.settle(m, xm(xc2))
            if r[0] == "osc":
                return None
            c2 = cell.delta(c, xb, _xbits(xc2, n))
            for m2 in r[1]:
                t = (m2, xc2, c2)
                if t not in seen:
                    if len(seen) >= max_pairs:
                        return None
                    seen.add(t)
                    q.append(t)
    return obs


def _output_map(obs, n_out, c_out):
    mo = np.array([o[0] for o in obs], np.uint8).reshape(len(obs), n_out)
    co = np.array([o[1] for o in obs], np.uint8).reshape(len(obs), c_out)
    cand = [[(k, ng) for k in range(c_out) for ng in (0, 1) if np.array_equal(mo[:, j], co[:, k] ^ ng)]
            for j in range(n_out)]
    full = _surjection(cand, c_out)
    if full is not None:
        return "match", full
    covered = {k for c in cand for k, _ in c}
    if covered == set(range(c_out)):
        return "partial", [list(c[0]) if c else None for c in cand]
    return None


def _conforms(fsm, cell, start, allowed, xm, omap, max_pairs) -> bool:
    """Does the module, from power-up total state `start`, behave like the cell from some state (fixed output map)?"""
    xc, m = start
    for q in cell.states:
        if not cell.stable(q, _xbits(xc, cell.n_in)):
            continue
        obs = _product(fsm, cell, [(m, xc, q)], allowed, xm, max_pairs)
        if obs is not None and all(all(mo[j] == co[k] ^ ng for j, (k, ng) in enumerate(omap)) for mo, co in obs):
            return True
    return False


# --- tick-level behavior: delay lines and edge detectors -------------------------------------------------------

def _waveform(m: UnitDelay, s: np.ndarray, x_seq) -> np.ndarray:
    out = []
    for xv in x_seq:
        s = m.tick(s, xv)
        out.append(m.out(s, xv))
    return np.array(out, np.uint8).reshape(len(out), len(m.outputs))


def delay_line(ex: Explored, i: int, inverted: bool, rng, T: int = 256):
    """Latency L if output j reproduces input i exactly L ticks later on a random stream (every output), else None."""
    m = ex.fsm.m
    x = rng.integers(0, 2, (T, m.I)).astype(np.uint8)
    r = ex.fsm.power_up(_xint(x[0]))                # start settled under x[0]
    if r[0] != "ok":
        return None
    s = np.frombuffer(r[1][0], np.uint8).copy()
    w = _waveform(m, s, x)                        # w[t] = outputs after tick t+1 under x[t]
    src = x[:, i] ^ int(inverted)
    for L in range(1, min(T // 4, ex.fsm.max_ticks) + 1):
        # output after tick t (index t-1) equals input applied at tick t-L+1 ... i.e. w[t] == src[t - L + 1]
        if all(np.array_equal(w[L - 1:, j], src[:T - L + 1]) for j in range(w.shape[1])):
            return L
    return None


def edge_response(ex: Explored, i: int, j: int, settle_ticks: int):
    """Per held configuration of the other inputs, the response of output j to rising and falling edges of input i:
    -> (kind, polarity) with kind in {"rising", "falling", "both"} or None if not an edge detector."""
    fsm, m = ex.fsm, ex.fsm.m
    kinds = set()
    pol = None
    for xr in range(1 << m.I):
        if (xr >> i) & 1:
            continue
        resp = {}
        for edge, (xa, xb) in (("rising", (xr, xr | (1 << i))), ("falling", (xr | (1 << i), xr))):
            r = fsm.power_up(xa)
            if r[0] != "ok":
                return None
            s = np.frombuffer(r[1][0], np.uint8).copy()
            base = fsm.out(r[1][0], xa)[j]
            w = _waveform(m, s, [m.xvec(xb)] * settle_ticks)[:, j]
            if w[-1] != base:
                return None
            exc = (w != base).astype(np.int8)
            n_pulses = int((np.diff(np.concatenate([[0], exc])) == 1).sum())
            if n_pulses > 1:
                return None
            resp[edge] = n_pulses
            if pol is None:
                pol = base
            elif pol != base:
                return None
        kinds.add((resp["rising"], resp["falling"]))
    if len(kinds) != 1:                          # the same response for every configuration of the other inputs
        return None
    rising, falling = kinds.pop()
    if not rising and not falling:
        return None
    # edges of the other inputs must not pulse output j
    for k in range(m.I):
        if k == i:
            continue
        for xr in range(1 << m.I):
            if (xr >> k) & 1:
                continue
            for xa, xb in ((xr, xr | (1 << k)), (xr | (1 << k), xr)):
                r = fsm.power_up(xa)
                s = np.frombuffer(r[1][0], np.uint8).copy()
                w = _waveform(m, s, [m.xvec(xb)] * settle_ticks)[:, j]
                if (w != pol).any():
                    return None
    kind = "both" if rising and falling else "rising" if rising else "falling"
    return kind, int(pol)


# --- structure: ring oscillators ----------------------------------------------------------------------------------

def ring_length(net: Netlist):
    """L if the netlist has exactly one cyclic component and it is a simple cycle of L gates, else None."""
    succ = _gate_graph(net)
    cyc = _cyclic(scc(succ), succ)
    if len(cyc) != 1:
        return None
    comp = set(cyc[0])
    if all(sum(w in comp for w in succ[v]) == 1 for v in comp):
        return len(comp)
    return None


# --- classification -----------------------------------------------------------------------------------------------

@dataclass
class CellMatch:
    module: str                  # hash hex
    name: str
    n_prims: int
    n_in: int
    n_out: int
    depth: int
    kind: str                    # combinational | sequential | oscillator | timing | unknown
    status: str                  # match | composite | partial | unmatched
    cell: str | None = None
    transform: dict = field(default_factory=dict)
    fsm_states: int = 0
    total_states: int = 0
    truncated: bool = False
    races: int = 0
    detail: dict = field(default_factory=dict)
    freerun: dict | None = None
    uses: int = 0                # module tokens in elites (filled in by the report)

    @property
    def matched(self) -> bool:
        return self.status == "match"


def slow_elements(net: Netlist) -> np.ndarray:
    """Per gate, its slow delay level (grown.sim.slow), -1 for ordinary gates (all -1 without a table)."""
    from ..sim.slow import TABLE, table
    import os
    G = len(net.tt)
    if not os.path.exists(os.environ.get("GROWN_SLOW_TABLE", TABLE)) or not G:
        return np.full(G, -1, np.int64)
    ga = np.repeat(np.arange(G), np.diff(np.asarray(net.fanin_ptr, np.int64)))
    return table().levels_of(np.asarray(net.tt), np.diff(np.asarray(net.fanin_ptr)), net.dr, net.df, ga)


def classify_netlist(net: Netlist, max_states: int = MAX_STATES, seed: int = 0) -> dict:
    """Classify a flattened module. -> dict(kind, status, cell, transform, fsm_states, total_states, ...).
    Slow delay elements (grown.sim.slow) are buffers to the unit-delay semantics, so they classify like delays; the
    result counts them (`slow_elements`) and an oscillator that contains one is named "... with slow delays", with
    its period estimated from the elements' physical delays (`detail.period_s`, a ring: the sum of its stages'
    rise and fall delays at 10 ps per tick)."""
    res = _classify_netlist(net, max_states, seed)
    lev = slow_elements(net)
    res["slow_elements"] = int((lev >= 0).sum())
    if res["slow_elements"] and res.get("kind") == "oscillator":
        res["cell"] = f"{res['cell']} with slow delays"
        if res.get("detail", {}).get("ring_stages"):
            per = float(np.sum(np.asarray(net.dr, np.float64)) + np.sum(np.asarray(net.df, np.float64)))
            res["detail"]["period_s"] = per * 1e-11
    return res


def _classify_netlist(net: Netlist, max_states: int = MAX_STATES, seed: int = 0) -> dict:
    if not len(net.outputs):
        return {"kind": "unknown", "status": "unmatched", "cell": None, "transform": {}, "total_states": 0,
                "fsm_states": 0, "detail": {"reason": "no outputs"}}
    if net.n_inputs > MAX_INPUTS:
        return {"kind": "unknown", "status": "unmatched", "cell": None, "transform": {}, "total_states": 0,
                "fsm_states": 0, "detail": {"reason": f"{net.n_inputs} inputs: too wide for exhaustive simulation "
                                                      f"(limit {MAX_INPUTS})"}}
    fsm = ModuleFSM(net, max_states)
    ex = fsm.explore()
    n_out = len(net.outputs)
    res = {"kind": "unknown", "status": "unmatched", "cell": None, "transform": {}, "detail": {},
           "total_states": len(ex.keys)}
    genuine = [(x, o) for x, o in ex.oscs if o.period > 0]   # (output-invisible oscillations settle)
    if ex.oscs and not genuine:
        res.update(fsm_states=ex.memory_states(), truncated=True, races=fsm.races,
                   detail={"reason": f"did not settle or repeat within {fsm.max_ticks} ticks"})
        return res
    if genuine:
        res["fsm_states"] = ex.memory_states()
        xs = sorted({x for x, _ in genuine})
        o = max((o for _, o in genuine), key=lambda o: o.period)
        free = len(xs) == (1 << net.n_inputs)
        L = ring_length(net)
        duty = [float(np.mean([w[j] for w in o.wave])) for j in range(n_out)] if o.wave else []
        square = bool(o.toggling) and all(abs(duty[j] - 0.5) < 1e-9 for j in o.toggling)
        if L is not None and o.period == 2 * L and square:
            cell = f"ring oscillator ({L} stages)"
        else:
            cell = f"clock-like oscillator (period {o.period} ticks)"
        res.update(kind="oscillator", status="match", cell=("gated " if not free else "") + cell,
                   detail={"period": o.period, "duty": duty, "enabled_inputs": xs, "free_running": free,
                           "ring_stages": L, "resolved": o.resolved})
        res.update(truncated=fsm.truncated, races=fsm.races)
        return res
    mem = ex.memory_states()
    res.update(fsm_states=mem, truncated=fsm.truncated, races=fsm.races)
    if fsm.internal_osc:
        res["detail"]["internal_oscillation"] = fsm.internal_osc
    if mem == 1:
        res["kind"] = "combinational"
        _classify_comb(ex, net, res, seed)
        return res
    res["kind"] = "sequential"
    # several cells can match from different power-up states (a Johnson counter powered into its unused cycle is a
    # T flip-flop): keep the one that explains the most power-up states, then the larger machine
    full, part = [], []
    for cell in sorted(SEQ, key=lambda c: c.allowed is not None):
        if cell.n_in != net.n_inputs:
            continue
        m = match_sequential(fsm, cell, n_out)
        if m is not None:
            (full if m[0] == "match" else part).append((cell, m[1]))
    if full:
        full.sort(key=lambda ct: (-ct[1]["powerup_conforming"], ct[0].allowed is not None, -len(ct[0].states)))
        cell, tf = full[0]
        res.update(status="match", cell=cell.name, transform=_describe(tf, cell.ports))
        if len(full) > 1:
            res["detail"]["also_matches"] = [c.name for c, _ in full[1:]]
    elif part:
        cell, tf = part[0]
        res.update(status="partial", cell=cell.name, transform=_describe(tf, cell.ports))
    return res


def _describe(tf: dict, ports=()) -> dict:
    perm, neg = tf["input_perm"], tf["input_neg"]
    out = dict(tf)
    if ports:
        out["inputs"] = {ports[i]: ("~" if (neg >> i) & 1 else "") + f"in{perm[i]}" for i in range(len(perm))}
    return out


def _classify_comb(ex: Explored, net: Netlist, res: dict, seed: int) -> None:
    n = net.n_inputs
    table = ex.truth_table()
    n_out = len(net.outputs)
    tts = [bits_tt([table[x][j] for x in range(1 << n)]) for j in range(n_out)]
    sup = support(tts, n)
    red = [restrict(t, n, sup) for t in tts]
    names = []
    for t in tts:                                         # each output named over its own support
        sj = support([t], n)
        names.append({**name_function(restrict(t, n, sj), len(sj)), "support": sj})
    res["detail"].update(support=sup, functions=names)
    k = len(sup)
    rng = np.random.default_rng(seed)
    # trivial settled functions: look at timing
    if k <= 1:
        settle = ex.fsm.max_ticks
        if k == 1 and all(nm["name"] in ("BUF", "NOT") for nm in names):
            i = sup[0]
            invs = {nm["name"] == "NOT" for nm in names}
            if len(invs) == 1:
                L = delay_line(ex, i, invs.pop(), rng)
                if L is not None and L >= 2:
                    res.update(kind="timing", status="match", cell=f"delay line ({L} ticks)",
                               transform={"input": i, "inverted": names[0]["name"] == "NOT"})
                    res["detail"]["latency"] = L
                    return
        edges = [[edge_response(ex, i, j, settle) for i in range(n)] for j in range(n_out)]
        found = {}
        for j in range(n_out):
            hits = [(i, e) for i, e in enumerate(edges[j]) if e is not None]
            if len(hits) == 1:
                found[j] = hits[0]
        if found and len({(i, e[0]) for i, e in found.values()}) == 1 and len(found) == n_out:
            i, (kind, pol) = next(iter(found.values()))
            label = {"rising": "rising-edge", "falling": "falling-edge", "both": "any-edge"}[kind]
            res.update(kind="timing", status="match", cell=f"{label} detector (pulse)",
                       transform={"input": i, "active_low": [found[j][1][1] == 1 for j in range(n_out)]})
            return
    if n_out > 1 and k <= MAX_NPN:
        m = match_multi(red, k)
        if m is not None:
            res.update(status="match", cell=m[0], transform={**m[1], "support": sup})
            return
    if all(nm["human"] for nm in names):
        distinct = {nm["npn"] for nm in names}
        full = (1 << (1 << k)) - 1
        if n_out == 1 or (len(distinct) == 1 and all(t in (red[0], red[0] ^ full) for t in red)
                          and len({tuple(nm["support"]) for nm in names}) == 1):
            nm = names[0]
            res.update(status="match", cell=nm["name"] or nm["npn_class"], transform={"support": sup})
        else:
            res.update(status="composite", cell=" + ".join(nm["name"] or nm["npn_class"] for nm in names),
                       transform={"support": sup})


# --- library-level report -----------------------------------------------------------------------------------------

TRIVIAL = {"CONST0", "CONST1", "BUF", "NOT"}      # catalog cells, but they say nothing about abstraction

def module_netlist(library, h: bytes) -> Netlist:
    return flatten(library.get(h).body, library)


def hierarchy_depth(library, h: bytes, _memo=None) -> int:
    memo = {} if _memo is None else _memo
    if h in memo:
        return memo[h]
    d = library.get(h)
    sub = [hierarchy_depth(library, nd.module, memo) for nd in d.body.nodes if nd.op == MODULE]
    memo[h] = 1 + max(sub, default=0)
    return memo[h]


def classify_module(library, h: bytes, freerun: bool = False, max_states: int = MAX_STATES) -> CellMatch:
    d = library.get(h)
    net = module_netlist(library, h)
    r = classify_netlist(net, max_states)
    cm = CellMatch(h.hex(), d.name, int(net.n_gates), d.n_in, d.n_out, hierarchy_depth(library, h), r["kind"],
                   r["status"], r["cell"], r.get("transform", {}), int(r.get("fsm_states", 0)),
                   int(r.get("total_states", 0)), bool(r.get("truncated", False)), int(r.get("races", 0)),
                   r.get("detail", {}))
    if freerun:
        try:
            cm.freerun = freerun_check(net, r)
        except Exception as e:  # pragma: no cover - environment dependent
            cm.freerun = {"error": f"{type(e).__name__}: {e}"}
    return cm


def elite_module_counts(archives, library) -> dict:
    """hash -> number of instances in archive elites, counting nested instances (a module used inside an instance of
    another counts once per enclosing instance)."""
    direct: dict = {}
    for a in archives:
        for c in a.filled_cells():
            for nd in a.items[c].nodes:
                if nd.op == MODULE:
                    direct[nd.module] = direct.get(nd.module, 0) + 1
    total = dict(direct)
    memo: dict = {}

    def inner(h):
        if h not in memo:
            cnt: dict = {}
            for nd in library.get(h).body.nodes:
                if nd.op == MODULE:
                    cnt[nd.module] = cnt.get(nd.module, 0) + 1
                    for k, v in inner(nd.module).items():
                        cnt[k] = cnt.get(k, 0) + v
            memo[h] = cnt
        return memo[h]

    for h, n in direct.items():
        for k, v in inner(h).items():
            total[k] = total.get(k, 0) + n * v
    return {"direct": direct, "total": total}


def library_report(library, archives=(), modules=None, freerun: bool = False, cache: dict | None = None) -> dict:
    """Classify every module (default: the active ones). -> json-able report with per-module rows and summary."""
    hs = list(modules) if modules is not None else library.active()
    rows = []
    for h in hs:
        key = h.hex() + (":fr" if freerun else "")
        if cache is not None and key in cache:
            cm = CellMatch(**cache[key])
        else:
            cm = classify_module(library, h, freerun)
            if cache is not None:
                cache[key] = asdict(cm)
        rows.append(cm)
    counts = elite_module_counts(archives, library) if archives else {"direct": {}, "total": {}}
    for cm in rows:
        cm.uses = int(counts["direct"].get(bytes.fromhex(cm.module), 0))
    n = len(rows)
    matched = [r for r in rows if r.matched]
    tok = sum(r.uses for r in rows)
    tok_m = sum(r.uses for r in matched)
    elite_tokens = sum(len(a.items[c].nodes) for a in archives for c in a.filled_cells())
    nontrivial = [r for r in matched if r.cell not in TRIVIAL]
    by_cell: dict = {}
    for r in matched:
        e = by_cell.setdefault(r.cell, {"modules": 0, "uses": 0})
        e["modules"] += 1
        e["uses"] += r.uses
    return {
        "modules": n, "matched": len(matched), "matched_frac": len(matched) / n if n else float("nan"),
        "matched_nontrivial": len(nontrivial),
        "matched_nontrivial_frac": len(nontrivial) / n if n else float("nan"),
        "composite": sum(r.status == "composite" for r in rows), "partial": sum(r.status == "partial" for r in rows),
        "module_tokens_in_elites": tok, "matched_tokens": tok_m,
        "matched_token_frac": tok_m / tok if tok else float("nan"),
        "elite_tokens": elite_tokens, "matched_frac_of_all_elite_tokens": tok_m / elite_tokens if elite_tokens else
        float("nan"),
        "by_cell": dict(sorted(by_cell.items(), key=lambda kv: (-kv[1]["uses"], kv[0]))),
        "rows": [asdict(r) for r in rows],
    }


def _fmt_tf(r: dict) -> str:
    t = r.get("transform") or {}
    if "inputs" in t:
        s = ", ".join(f"{k}={v}" for k, v in t["inputs"].items())
        if t.get("outputs"):
            s += "; out " + ", ".join(f"{'~' if ng else ''}{k}" for k, ng in (o for o in t["outputs"] if o))
        return s
    if "input_perm" in t:
        return f"perm {t['input_perm']} neg {t['input_neg']:b}"
    if "support" in t:
        return f"inputs {t['support']}"
    if "input" in t:
        return f"input {t['input']}"
    return ""


def report_markdown(rep: dict, title: str = "Library canonical-cell report") -> str:
    L = [f"# {title}", ""]
    n = rep["modules"]
    L.append(f"- modules classified: **{n}**; matching a human cell: **{rep['matched']}** "
             f"({100 * rep['matched_frac']:.1f}%)" if n else "- no modules")
    if n:
        L.append(f"- excluding constants, buffers and inverters: **{rep['matched_nontrivial']}** "
                 f"({100 * rep['matched_nontrivial_frac']:.1f}%) match a non-trivial human cell")
        L.append(f"- composites of human cells (several catalog functions side by side): {rep['composite']}; "
                 f"partial matches (some outputs unexplained): {rep['partial']}")
        if rep["module_tokens_in_elites"]:
            L.append(f"- module tokens in archive elites: {rep['module_tokens_in_elites']}, of which "
                     f"{rep['matched_tokens']} ({100 * rep['matched_token_frac']:.1f}%) are matched cells; "
                     f"matched cells are {100 * rep['matched_frac_of_all_elite_tokens']:.2f}% of all "
                     f"{rep['elite_tokens']} elite tokens")
        L += ["", "## Matched cells", "", "| cell | modules | uses in elites |", "|---|---:|---:|"]
        for cell, e in rep["by_cell"].items():
            L.append(f"| {cell} | {e['modules']} | {e['uses']} |")
        L += ["", "## All modules", "",
              "| module | prims | in/out | depth | uses | kind | status | cell | transform | FSM states |",
              "|---|---:|---:|---:|---:|---|---|---|---|---:|"]
        for r in sorted(rep["rows"], key=lambda r: (-r["uses"], r["name"])):
            fs = f"{r['fsm_states']}" + (" (bound hit)" if r["truncated"] else "")
            L.append(f"| {r['name']} | {r['n_prims']} | {r['n_in']}/{r['n_out']} | {r['depth']} | {r['uses']} | "
                     f"{r['kind']} | {r['status']} | {r['cell'] or ''} | {_fmt_tf(r)} | {fs} |")
        un = [r for r in rep["rows"] if r["status"] != "match"]
        if un:
            L += ["", "## Unmatched", "", "| module | prims | in/out | kind | FSM states | total states | what it is |",
                  "|---|---:|---:|---|---:|---:|---|"]
            for r in sorted(un, key=lambda r: (-r["uses"], r["name"])):
                fn = r["detail"].get("functions")
                what = "; ".join(f"npn {f['npn']} over {f['n']} inputs" + (f" ({f['name'] or f['npn_class']})"
                                                                           if f["human"] else "")
                                 for f in fn) if fn else (r["cell"] or "")
                L.append(f"| {r['name']} | {r['n_prims']} | {r['n_in']}/{r['n_out']} | {r['kind']} | "
                         f"{r['fsm_states']} | {r['total_states']} | {what} |")
    return "\n".join(L) + "\n"


def write_report(rep: dict, prefix: str, title: str = "Library canonical-cell report") -> None:
    with open(prefix + ".json", "w") as f:
        json.dump(rep, f, indent=1, default=str)
    with open(prefix + ".md", "w") as f:
        f.write(report_markdown(rep, title))


# --- freerun (optional) -------------------------------------------------------------------------------------------

def freerun_check(net: Netlist, r: dict, steps: int = 48, seed: int = 0) -> dict:
    """Run the module on freerun (CPU, real per-arc delays). Compares settled outputs under a random single-input-
    change stimulus with the stand-in's fundamental-mode prediction, and measures latency / period in ticks."""
    from ..sim.freerun_backend import FreerunSubstrate
    sub = FreerunSubstrate("cpu")
    hold = int(4 * (int(net.dr.sum()) + int(net.df.sum())) + 16)
    out = {"hold_ticks": hold}
    rng = np.random.default_rng(seed)
    I = net.n_inputs
    fsm = ModuleFSM(net)
    x = np.zeros((steps, I), np.uint8)
    x[0] = rng.integers(0, 2, I)
    for s in range(1, steps):
        x[s] = x[s - 1]
        if I:
            x[s, rng.integers(I)] ^= 1
    res = sub.evaluate_netlists([net], x[None], hold)
    got = res.outputs[0]
    # prediction (deterministic positions only)
    agree = total = 0
    r0 = fsm.power_up(_xint(x[0]))
    states = set(r0[1]) if r0[0] == "ok" else None
    for s in range(steps):
        if states is None:
            break
        if s:
            nxt = set()
            for st in states:
                rr = fsm.settle(st, _xint(x[s]))
                if rr[0] == "osc":
                    nxt = None
                    break
                nxt |= set(rr[1])
            states = nxt
            if states is None:
                break
        outs = {fsm.out(st, _xint(x[s])) for st in states}
        if len(outs) == 1:
            total += 1
            agree += int(tuple(got[s]) == outs.pop())
    out["agreement"] = agree / total if total else float("nan")
    out["compared_steps"] = total
    if r.get("kind") == "oscillator" and len(net.outputs):
        T = 64 * hold
        xe = (r.get("detail", {}).get("enabled_inputs") or [0])[0]           # an input that enables it
        held = np.zeros((T, I), np.uint8) + np.array(_xbits(xe, I), np.uint8)
        res = sub.evaluate_netlists([net], held[None], 1)
        out["period_ticks"] = None
        for j in range(len(net.outputs)):
            rises = np.nonzero(np.diff(res.outputs[0, :, j].astype(np.int8)) == 1)[0]
            if len(rises) > 2:
                out["period_ticks"] = float(np.diff(rises).mean())
                break
    if r.get("cell", "") and str(r.get("cell")).startswith("delay line"):
        i = r["transform"]["input"]
        T = 4 * hold
        xs = np.zeros((T, I), np.uint8)
        xs[T // 2:, i] = 1
        res = sub.evaluate_netlists([net], xs[None], 1)
        w = res.outputs[0, :, 0]
        ch = np.nonzero(w[T // 2:] != w[T // 2 - 1])[0]
        out["latency_ticks"] = int(ch[0]) + 1 if len(ch) else None
    return out
