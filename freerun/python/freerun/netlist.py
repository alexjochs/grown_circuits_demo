"""Circuit data model (SPEC §2): flat arrays, one circuit or a batch of independent circuits.

Net numbering: nets 0..I-1 are primary inputs, net I+g is the output of gate g. Gate g's pins are
fanin[fanin_ptr[g]:fanin_ptr[g+1]] (pin 0 first); the per-pin arc delays dr/df are aligned with fanin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_ARITY = 6
MAX_DELAY = 2**31 - 1


@dataclass(frozen=True, eq=False)
class Jitter:
    """Per-transition delay jitter parameters of one circuit (SPEC §10).

    amp:  int64[G] fixed-point amplitudes a_g (sigma_g ~= a_g / jitter.K ticks); 0 disables a gate's jitter.
    seed: 64-bit stream key, cid: 32-bit circuit id. Both belong to the circuit, never to its batch position.
    dmin: floor on effective delays, in [1, the circuit's minimum arc delay].
    Build from standard deviations with Jitter.from_sigma."""
    amp: np.ndarray
    seed: int = 0
    cid: int = 0
    dmin: int = 1

    def __post_init__(self):
        object.__setattr__(self, "amp", np.ascontiguousarray(self.amp, dtype=np.int64))
        for name, hi in (("seed", 2**64), ("cid", 2**32)):
            v = int(getattr(self, name))
            if not 0 <= v < hi:
                raise ValueError(f"jitter {name} must be in [0, {hi})")
            object.__setattr__(self, name, v)
        object.__setattr__(self, "dmin", int(self.dmin))
        if self.amp.ndim != 1 or np.any(self.amp < 0) or np.any(self.amp >= 2**44):
            raise ValueError("jitter amplitudes must be a 1-D array of integers in [0, 2^44)")
        if self.dmin < 1:
            raise ValueError("jitter dmin must be >= 1")

    @staticmethod
    def from_sigma(sigma, n_gates: int | None = None, seed: int = 0, cid: int = 0, dmin: int = 1) -> "Jitter":
        """sigma: standard deviation in ticks, a scalar (needs n_gates) or one value per gate."""
        from .jitter import amplitude
        s = np.asarray(sigma, dtype=np.float64)
        if s.ndim == 0:
            if n_gates is None:
                raise ValueError("a scalar sigma needs n_gates")
            s = np.full(n_gates, float(s))
        return Jitter(amplitude(s), seed, cid, dmin)

    @property
    def sigma(self) -> np.ndarray:
        from .jitter import sigma
        return sigma(self.amp)

    def replace(self, **kw) -> "Jitter":
        d = dict(amp=self.amp, seed=self.seed, cid=self.cid, dmin=self.dmin)
        d.update(kw)
        return Jitter(**d)

    def __eq__(self, other) -> bool:
        return isinstance(other, Jitter) and np.array_equal(self.amp, other.amp) and \
            (self.seed, self.cid, self.dmin) == (other.seed, other.cid, other.dmin)


@dataclass(frozen=True, eq=False)
class Circuit:
    n_inputs: int
    fanin_ptr: np.ndarray  # int64[G+1]
    fanin: np.ndarray      # int32[E]   net index per pin
    tt: np.ndarray         # uint64[G]  truth table, bit idx(x) with pin 0 = LSB
    dr: np.ndarray         # int32[E]   rise delay of arc pin -> output
    df: np.ndarray         # int32[E]   fall delay
    init: np.ndarray       # uint8[N]   initial net values
    jitter: Jitter | None = None  # per-transition jitter (SPEC §10); None = none

    def __post_init__(self):
        for name, dt in (("fanin_ptr", np.int64), ("fanin", np.int32), ("tt", np.uint64),
                         ("dr", np.int32), ("df", np.int32), ("init", np.uint8)):
            object.__setattr__(self, name, np.ascontiguousarray(getattr(self, name), dtype=dt))
        self.validate()

    @property
    def n_gates(self) -> int:
        return len(self.tt)

    @property
    def n_nets(self) -> int:
        return self.n_inputs + self.n_gates

    @property
    def arity(self) -> np.ndarray:
        return np.diff(self.fanin_ptr)

    def pins(self, g: int) -> slice:
        return slice(int(self.fanin_ptr[g]), int(self.fanin_ptr[g + 1]))

    def validate(self) -> None:
        I, G, N = self.n_inputs, self.n_gates, self.n_nets
        E = len(self.fanin)
        if I < 0:
            raise ValueError("n_inputs must be >= 0")
        if self.fanin_ptr.shape != (G + 1,) or self.fanin_ptr[0] != 0 or self.fanin_ptr[-1] != E:
            raise ValueError("fanin_ptr must have G+1 entries from 0 to len(fanin)")
        k = self.arity
        if np.any(k < 0) or np.any(k > MAX_ARITY):
            raise ValueError(f"gate arity must be in [0, {MAX_ARITY}]")
        if self.dr.shape != (E,) or self.df.shape != (E,):
            raise ValueError("dr and df must align with fanin")
        if self.init.shape != (N,) or np.any(self.init > 1):
            raise ValueError("init must hold one 0/1 value per net")
        if E and (self.fanin.min() < 0 or self.fanin.max() >= N):
            raise ValueError("fanin references a net out of range")
        if E and (min(self.dr.min(), self.df.min()) < 1):
            raise ValueError("delays must be >= 1 tick")
        width = np.left_shift(np.uint64(1), k.astype(np.uint64))  # 2^k (k <= 6, so 2^k <= 64 fits)
        mask = np.where(k == 6, np.uint64(0), ~((np.uint64(1) << width) - np.uint64(1)))
        if np.any(self.tt & mask):
            raise ValueError("truth table has bits set at or above 2^arity")
        const = np.nonzero(k == 0)[0]
        if len(const) and np.any(self.init[I + const] != (self.tt[const] & np.uint64(1))):
            raise ValueError("constant (arity-0) gate init must equal its truth table bit 0")
        j = self.jitter
        if j is not None:
            if not isinstance(j, Jitter):
                raise ValueError("jitter must be a Jitter or None")
            if j.amp.shape != (G,):
                raise ValueError("jitter amplitudes must have one entry per gate")
            if E and j.dmin > min(self.dr.min(), self.df.min()):
                raise ValueError("jitter dmin must not exceed the minimum arc delay (SPEC §10.1)")

    def fanout(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CSR fanout: for net n, entries fo_ptr[n]:fo_ptr[n+1] of (gate, pin), sorted by (gate, pin)."""
        gate_of_edge = np.repeat(np.arange(self.n_gates, dtype=np.int32), self.arity)
        pin_of_edge = np.arange(len(self.fanin), dtype=np.int64) - np.repeat(self.fanin_ptr[:-1], self.arity)
        order = np.lexsort((pin_of_edge, gate_of_edge, self.fanin))
        counts = np.bincount(self.fanin, minlength=self.n_nets)
        fo_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return fo_ptr, gate_of_edge[order], pin_of_edge[order].astype(np.int32)

    def permute_gates(self, perm: np.ndarray) -> "Circuit":
        """Relabel gates: new gate j is old gate perm[j]. Nets are remapped consistently (SPEC §7.4)."""
        perm = np.asarray(perm, dtype=np.int64)
        I = self.n_inputs
        net_map = np.concatenate([np.arange(I), I + np.argsort(perm)])  # old net -> new net
        k = self.arity[perm]
        ptr = np.concatenate([[0], np.cumsum(k)])
        idx = np.concatenate([np.arange(self.fanin_ptr[g], self.fanin_ptr[g + 1]) for g in perm] or [[]]).astype(np.int64)
        init = np.empty_like(self.init)
        init[net_map] = self.init
        jit = None if self.jitter is None else self.jitter.replace(amp=self.jitter.amp[perm])
        return Circuit(I, ptr, net_map[self.fanin[idx]], self.tt[perm], self.dr[idx], self.df[idx], init, jit)

    def with_init(self, init) -> "Circuit":
        return Circuit(self.n_inputs, self.fanin_ptr, self.fanin, self.tt, self.dr, self.df, init, self.jitter)

    def with_delays(self, dr, df) -> "Circuit":
        """Same topology, functions, initial values and jitter, new arc delays."""
        return Circuit(self.n_inputs, self.fanin_ptr, self.fanin, self.tt, dr, df, self.init, self.jitter)

    def with_jitter(self, jitter: Jitter | None) -> "Circuit":
        return Circuit(self.n_inputs, self.fanin_ptr, self.fanin, self.tt, self.dr, self.df, self.init, jitter)

    def save(self, path) -> None:
        extra = {}
        if self.jitter is not None:
            j = self.jitter
            extra = dict(jit_amp=j.amp, jit_seed=np.uint64(j.seed), jit_cid=np.uint32(j.cid), jit_dmin=j.dmin)
        np.savez_compressed(path, n_inputs=self.n_inputs, fanin_ptr=self.fanin_ptr, fanin=self.fanin, tt=self.tt,
                            dr=self.dr, df=self.df, init=self.init, **extra)

    @staticmethod
    def load(path) -> "Circuit":
        z = np.load(path)
        jit = None
        if "jit_amp" in z:
            jit = Jitter(z["jit_amp"], int(z["jit_seed"]), int(z["jit_cid"]), int(z["jit_dmin"]))
        return Circuit(int(z["n_inputs"]), z["fanin_ptr"], z["fanin"], z["tt"], z["dr"], z["df"], z["init"], jit)

    def __eq__(self, other) -> bool:
        return isinstance(other, Circuit) and self.n_inputs == other.n_inputs and all(
            np.array_equal(getattr(self, f), getattr(other, f)) for f in ("fanin_ptr", "fanin", "tt", "dr", "df", "init")
        ) and self.jitter == other.jitter


class Net:
    """Handle to a net inside a Builder; resolved to an index at build time (allows forward references)."""
    __slots__ = ("kind", "index")

    def __init__(self, kind: str, index: int):
        self.kind, self.index = kind, index

    def __repr__(self):
        return f"Net({self.kind}{self.index})"


@dataclass
class _Gate:
    tt: int
    ins: list
    dr: list
    df: list
    init: int


@dataclass
class Builder:
    """Incremental construction with forward references, so cycles are easy to write:

        b = Builder(); a = b.input()
        g1 = b.gate(NAND2, k=2); g2 = b.gate(NAND2, k=2)
        b.connect(g1, [a, g2], rise=50, fall=40); b.connect(g2, [g1, a], rise=50, fall=40)
    """
    _inputs: list = field(default_factory=list)
    _gates: list = field(default_factory=list)

    def input(self, init: int = 0) -> Net:
        self._inputs.append(int(init))
        return Net("i", len(self._inputs) - 1)

    def gate(self, tt: int, ins=None, rise=1, fall=1, init: int = 0, k: int | None = None) -> Net:
        """Add a gate. Delays may be scalars (all arcs) or per-pin lists. If ins is None, call connect() later."""
        n = Net("g", len(self._gates))
        self._gates.append(_Gate(int(tt), [], [], [], int(init)))
        if ins is not None:
            self.connect(n, ins, rise, fall)
        elif k == 0:
            self.connect(n, [], rise, fall)
        return n

    def const(self, value: int) -> Net:
        return self.gate(int(value), [], init=int(value))

    def connect(self, gate: Net, ins, rise=1, fall=1) -> None:
        if gate.kind != "g":
            raise ValueError("can only connect gates")
        k = len(ins)
        rise = list(rise) if np.ndim(rise) else [rise] * k
        fall = list(fall) if np.ndim(fall) else [fall] * k
        if len(rise) != k or len(fall) != k:
            raise ValueError("per-pin delay lists must match the number of inputs")
        g = self._gates[gate.index]
        g.ins, g.dr, g.df = list(ins), [int(x) for x in rise], [int(x) for x in fall]

    def set_init(self, net: Net, value: int) -> None:
        if net.kind == "i":
            self._inputs[net.index] = int(value)
        else:
            self._gates[net.index].init = int(value)

    def index(self, net: Net) -> int:
        return net.index if net.kind == "i" else len(self._inputs) + net.index

    def build(self) -> Circuit:
        I = len(self._inputs)
        k = [len(g.ins) for g in self._gates]
        return Circuit(
            n_inputs=I,
            fanin_ptr=np.concatenate([[0], np.cumsum(k)]) if k else np.zeros(1),
            fanin=np.array([self.index(n) for g in self._gates for n in g.ins], dtype=np.int64),
            tt=np.array([g.tt for g in self._gates], dtype=np.uint64),
            dr=np.array([d for g in self._gates for d in g.dr], dtype=np.int64),
            df=np.array([d for g in self._gates for d in g.df], dtype=np.int64),
            init=np.array(self._inputs + [g.init for g in self._gates], dtype=np.uint8),
        )


def truth_table(fn, k: int) -> int:
    """Truth table of a Python function of k bits (pin 0 = LSB of the row index)."""
    return sum(int(bool(fn(*[(r >> i) & 1 for i in range(k)]))) << r for r in range(1 << k))


# Common functions. Pin order follows the argument order.
BUF = truth_table(lambda a: a, 1)
INV = truth_table(lambda a: 1 - a, 1)
AND2 = truth_table(lambda a, b: a & b, 2)
NAND2 = truth_table(lambda a, b: 1 - (a & b), 2)
OR2 = truth_table(lambda a, b: a | b, 2)
NOR2 = truth_table(lambda a, b: 1 - (a | b), 2)
XOR2 = truth_table(lambda a, b: a ^ b, 2)
XNOR2 = truth_table(lambda a, b: 1 - (a ^ b), 2)
NAND3 = truth_table(lambda a, b, c: 1 - (a & b & c), 3)
NOR3 = truth_table(lambda a, b, c: 1 - (a | b | c), 3)
MUX2 = truth_table(lambda a0, a1, s: a1 if s else a0, 3)  # sky130 mux2 pin order: A0, A1, S
