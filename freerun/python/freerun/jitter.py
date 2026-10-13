"""Per-transition delay jitter (SPEC §10): the counter-based RNG and the integer offset arithmetic.

Everything that decides a simulated delay is integer arithmetic, written to match csrc/jitter.h bit for bit:

    (r0..r3) = philox4x32_10(counter=(k_lo, k_hi, gate, cid), key=(seed_lo, seed_hi))
    c        = sum(2 * (r_i >> 16) + 1) - 2**18            # symmetric Irwin-Hall(4), integer
    j        = (a_g * c + 2**31) >> 32                      # floor division by 2**32
    D        = min(2**31 - 1, max(dmin, d + j))

Host-side helpers convert a standard deviation in ticks to the fixed-point amplitude a_g and back.
"""
from __future__ import annotations

import math

import numpy as np

M32 = 0xFFFFFFFF
MAX_DELAY = 2**31 - 1
AMP_LIMIT = 2**44                      # a_g must be < 2**44 so that |a_g * c| < 2**62
C_MAX = 2**18 - 4                      # |c| <= C_MAX
K = 2**31 * math.sqrt(3.0 / (2**32 - 1))  # a_g = round(sigma_ticks * K)


def philox4x32_10(ctr, key):
    """Philox4x32-10 (Salmon et al., SC'11). ctr: 4 uint32 words, key: 2 uint32 words. Returns 4 uint32 words."""
    c0, c1, c2, c3 = (int(x) & M32 for x in ctr)
    k0, k1 = (int(x) & M32 for x in key)
    for r in range(10):
        if r:
            k0 = (k0 + 0x9E3779B9) & M32
            k1 = (k1 + 0xBB67AE85) & M32
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & M32, p1 & M32, ((p0 >> 32) ^ c3 ^ k1) & M32, p0 & M32
    return c0, c1, c2, c3


def philox_np(ctr: np.ndarray, key) -> np.ndarray:
    """Vectorized Philox4x32-10: ctr uint64[..., 4] holding uint32 values, key (k0, k1). Returns uint64[..., 4]."""
    c = np.asarray(ctr, dtype=np.uint64) & np.uint64(M32)
    c0, c1, c2, c3 = (c[..., i].copy() for i in range(4))
    k0, k1 = np.uint64(int(key[0]) & M32), np.uint64(int(key[1]) & M32)
    m32 = np.uint64(M32)
    s32 = np.uint64(32)
    for r in range(10):
        if r:
            k0 = (k0 + np.uint64(0x9E3779B9)) & m32
            k1 = (k1 + np.uint64(0xBB67AE85)) & m32
        p0 = np.uint64(0xD2511F53) * c0  # < 2**64: exact in uint64
        p1 = np.uint64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = ((p1 >> s32) ^ c1 ^ k0) & m32, p1 & m32, ((p0 >> s32) ^ c3 ^ k1) & m32, p0 & m32
    return np.stack([c0, c1, c2, c3], axis=-1)


def centered(r) -> int:
    """c = sum_i (2 * (r_i >> 16) + 1) - 2**18, in [-C_MAX, C_MAX]."""
    return sum(2 * (int(x) >> 16) + 1 for x in r) - 2**18


def offset(amp: int, seed: int, cid: int, gate: int, k: int) -> int:
    """Jitter offset j_g(k) in ticks (SPEC §10.3)."""
    if amp == 0:
        return 0
    r = philox4x32_10((k & M32, (k >> 32) & M32, gate, cid), (seed & M32, (seed >> 32) & M32))
    return (amp * centered(r) + 2**31) >> 32


def offsets(amp, seed: int, cid: int, gate, k) -> np.ndarray:
    """Vectorized offset(): amp, gate and k broadcast against each other (k < 2**63)."""
    amp, gate, k = np.broadcast_arrays(np.asarray(amp, np.int64), np.asarray(gate, np.int64), np.asarray(k, np.int64))
    ctr = np.stack([k & M32, (k >> 32) & M32, gate & M32, np.full(k.shape, cid & M32)], axis=-1).astype(np.uint64)
    r = philox_np(ctr, (seed & M32, (seed >> 32) & M32))
    c = (2 * (r >> np.uint64(16)) + np.uint64(1)).sum(axis=-1).astype(np.int64) - 2**18
    return (amp * c + np.int64(2**31)) >> np.int64(32)


def effective(d: int, j: int, dmin: int) -> int:
    return min(MAX_DELAY, max(dmin, d + j))


def min_offset(amp) -> np.ndarray | int:
    """Smallest possible offset for amplitude amp (SPEC §10.6). Works on ints and int64 arrays."""
    if isinstance(amp, (int, np.integer)):
        return (-int(amp) * C_MAX + 2**31) >> 32
    a = np.asarray(amp, dtype=np.int64)
    return (-a * np.int64(C_MAX) + np.int64(2**31)) >> np.int64(32)


def max_offset(amp: int) -> int:
    return (int(amp) * C_MAX + 2**31) >> 32


def amplitude(sigma) -> np.ndarray:
    """Fixed-point amplitude a_g for a standard deviation in ticks (scalar or array)."""
    a = np.rint(np.asarray(sigma, dtype=np.float64) * K)
    if np.any(a < 0) or np.any(a >= AMP_LIMIT):
        raise ValueError(f"jitter sigma must be in [0, {(AMP_LIMIT - 1) / K:.3g}] ticks")
    return a.astype(np.int64)


def sigma(amp) -> np.ndarray:
    """Standard deviation in ticks represented by amplitude a_g (excluding the 1/12 tick^2 rounding variance)."""
    return np.asarray(amp, dtype=np.float64) / K


def sigma_relative(circuit, rel: float) -> np.ndarray:
    """Per-gate sigma (ticks) = rel x the mean of the gate's rise and fall arc delays (0 for constants)."""
    k = circuit.arity
    G = circuit.n_gates
    tot = np.zeros(G)
    if len(circuit.fanin):
        eg = np.repeat(np.arange(G), k)
        np.add.at(tot, eg, circuit.dr.astype(np.float64) + circuit.df.astype(np.float64))
    mean = np.divide(tot, 2 * k, out=np.zeros(G), where=k > 0)
    return rel * mean


def lookahead(circuit) -> int:
    """Window width L for the windowed algorithm (SPEC §10.6): a lower bound on every effective delay."""
    E = len(circuit.fanin)
    if not E:
        return 1 << 40
    base = np.minimum(circuit.dr, circuit.df).astype(np.int64)
    jit = circuit.jitter
    if jit is None:
        return int(base.min())
    eg = np.repeat(np.arange(circuit.n_gates), circuit.arity)
    eff = np.minimum(MAX_DELAY, np.maximum(jit.dmin, base + min_offset(jit.amp)[eg]))
    return int(eff.min())
