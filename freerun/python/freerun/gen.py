"""Circuit and stimulus generators for tests and benchmarks."""
from __future__ import annotations

import numpy as np

from .netlist import INV, NAND2, Builder, Circuit, MAX_ARITY


def ring(stages: int, rise=10, fall=10, init=None) -> Circuit:
    """Ring of `stages` inverters (stage i drives stage i+1, the last drives the first). No inputs.

    rise/fall: scalars or per-stage sequences. init: per-stage initial outputs (default all 0)."""
    rise = np.broadcast_to(rise, stages)
    fall = np.broadcast_to(fall, stages)
    init = np.zeros(stages, dtype=int) if init is None else np.asarray(init)
    b = Builder()
    gs = [b.gate(INV, k=1, init=int(init[i])) for i in range(stages)]
    for i in range(stages):
        b.connect(gs[i], [gs[i - 1]], rise=int(rise[i]), fall=int(fall[i]))
    return b.build()


def sr_latch(delay=10) -> Circuit:
    """Cross-coupled NAND latch. Inputs: 0 = S_n, 1 = R_n (active low). Gates: 0 = Q, 1 = Q_n."""
    b = Builder()
    s_n, r_n = b.input(1), b.input(1)
    q, qn = b.gate(NAND2, k=2, init=0), b.gate(NAND2, k=2, init=1)
    b.connect(q, [s_n, qn], delay, delay)
    b.connect(qn, [r_n, q], delay, delay)
    return b.build()


def random_circuit(rng: np.random.Generator, n_inputs: int, n_gates: int, *, max_arity: int = 3,
                   min_arity: int = 1, delay=(5, 50), per_arc: bool = True, feedforward: bool = False,
                   locality: int | None = None, random_init: bool = True, tt_pool=None) -> Circuit:
    """Random netlist with uniformly random truth tables.

    feedforward: gate g may only read inputs and gates < g (acyclic). Otherwise any net, including itself.
    locality:    if set, gate inputs are drawn from gates within +-locality of g (plus primary inputs), which gives
                 many short loops, like a placed circuit.
    delay:       (lo, hi) inclusive range of arc delays in ticks.
    per_arc:     independent delays per pin; otherwise one rise and one fall delay per gate.
    tt_pool:     optional list of (truth_table, arity) to draw gate functions from instead of random tables.
    """
    if not 0 <= min_arity <= max_arity <= MAX_ARITY:
        raise ValueError("bad arity range")
    I, G = n_inputs, n_gates
    if tt_pool is not None:
        choice = rng.integers(0, len(tt_pool), size=G)
        k = np.array([tt_pool[i][1] for i in choice], dtype=np.int64)
    else:
        k = rng.integers(min_arity, max_arity + 1, size=G)
    ptr = np.concatenate([[0], np.cumsum(k)])
    E = int(ptr[-1])
    eg = np.repeat(np.arange(G), k)  # gate of each edge
    if feedforward:
        if E and I == 0 and np.any(eg == 0):
            raise ValueError("a feedforward circuit with no inputs cannot start with a gate of arity >= 1")
        fanin = rng.integers(0, I + eg) if E else np.zeros(0, dtype=np.int64)
    elif locality is None:
        fanin = rng.integers(0, I + G, size=E)
    else:  # primary inputs, or gates within +-locality (wrapping around)
        r = rng.integers(0, I + 2 * locality + 1, size=E)
        fanin = np.where(r < I, r, I + (eg + r - I - locality) % G)
    lo, hi = delay
    if per_arc:
        dr = rng.integers(lo, hi + 1, size=E)
        df = rng.integers(lo, hi + 1, size=E)
    else:
        dr = np.repeat(rng.integers(lo, hi + 1, size=G), k)
        df = np.repeat(rng.integers(lo, hi + 1, size=G), k)
    if tt_pool is not None:
        tt = np.array([tt_pool[i][0] for i in choice], dtype=np.uint64)
    else:
        r = rng.integers(0, 2**64, size=G, dtype=np.uint64, endpoint=False)
        width = np.left_shift(np.uint64(1), np.minimum(k, 5).astype(np.uint64))  # 2^k, k clipped to keep shifts < 64
        mask = np.where(k == 6, ~np.uint64(0), (np.uint64(1) << width) - np.uint64(1))
        tt = r & mask
    init = rng.integers(0, 2, size=I + G) if random_init else np.zeros(I + G, dtype=int)
    const = np.nonzero(k == 0)[0]
    init[I + const] = tt[const] & np.uint64(1)
    return Circuit(I, ptr, fanin, tt, dr, df, init)


# Function pools (truth table, arity)
from .netlist import AND2, MUX2, NAND3, NOR2, NOR3, OR2, XNOR2, XOR2  # noqa: E402

SKY130_LIKE = [(INV, 1), (NAND2, 2), (NOR2, 2), (NAND3, 3), (NOR3, 3), (XOR2, 2), (XNOR2, 2), (MUX2, 3)]
XOR_HEAVY = [(INV, 1), (XOR2, 2), (XNOR2, 2), (NAND2, 2), (MUX2, 3)]


def random_stimulus(rng: np.random.Generator, n_inputs: int, t_end: int, mean_interval: float, t_start: int = 1):
    """Independent random toggles per input, roughly every mean_interval ticks, in [t_start, t_end)."""
    ev = []
    for n in range(n_inputs):
        t = t_start
        v = int(rng.integers(0, 2))
        while True:
            t += 1 + int(rng.exponential(mean_interval))
            if t >= t_end:
                break
            v ^= 1
            ev.append((t, n, v))
    return sorted(ev)


def fuzz_case(seed: int, max_gates: int = 300, max_inputs: int = 7):
    """A random (circuit, stimulus, T) from a mix of regimes chosen to exercise the inertial rules: tiny delays (many
    ties), wide and narrow delay ranges, stimulus intervals comparable to gate delays, several function pools,
    per-arc and per-gate delays, global and local wiring. Returns (circuit, stimulus, T, info dict)."""
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["tiny", "wide", "narrow"]))
    delay = {"tiny": (1, 6), "wide": (20, 3000), "narrow": (900, 1100)}[regime]
    pools = [None, SKY130_LIKE, XOR_HEAVY]
    pool_i = int(rng.integers(0, 3))
    per_arc = bool(rng.integers(0, 2))
    n_in = int(rng.integers(1, max_inputs + 1))
    n_gates = int(rng.integers(3, max_gates + 1))
    c = random_circuit(rng, n_in, n_gates, max_arity=int(rng.integers(1, 7)), delay=delay, per_arc=per_arc,
                       locality=None if rng.integers(0, 2) else int(rng.integers(1, 10)), tt_pool=pools[pool_i])
    mean_d = sum(delay) / 2
    interval = float(mean_d * rng.choice([0.3, 1.0, 3.0, 30.0]))
    T = int(mean_d * rng.integers(20, 400))
    stim = random_stimulus(rng, n_in, T, interval)
    info = dict(seed=int(seed), regime=regime, per_arc=per_arc, pool=pool_i, n_in=n_in, n_gates=n_gates, T=T)
    return c, stim, T, info
