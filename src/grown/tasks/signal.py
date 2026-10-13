"""Signal tasks: no physics, the specification is the teacher.

A task is a spec kind plus parameters. Each kind supplies
  draw(rng, split)      parameters from the train box, or outside it along one demand axis for "extrap"
  gen(a, key, T)        the exogenous input stream xs [T, n_in] (and latent zs [T, n_z]) from a key, in JAX
  init / ref            the reference: a causal state machine (ts, x_t, z_t) -> (ts, y_t), in JAX
  tags(p)               DemandTags axes, analytically from the parameters
  io(p)                 the bit interface (InCode per input channel, OutCode per output channel)
  mutate(p, rng)        a nearby parameter set (frontier sampler)

Streams are sampled at 1 / dt, a physical sample rate (the time model, docs/grown_time.md): the catalog gives every
named task its own realistic dt (1 kHz to 1 MHz, see CATALOG), the family draws one per task from SAMPLE_RATES, and
the substrate ticks per sample are derived, K = round(dt / tick) (10^5 to 10^8 at 10 ps). Legacy mode
(TimeModel.legacy_K, or the K= shorthand) reproduces the pre-time-model tasks (dt = 1 ms, fixed K) and their uids.
Demand tags are dimensionless: timing_precision = dt / tau_res with the spec's resolution tau_res in samples, so it
does not change with dt; tags.extra carries the physical values (dt, rate_hz, tau_res_s, K). The teacher at step t is the
reference applied causally to x_0..x_t; inputs never depend on actions, so DAgger relabelling is the identity here
and the dataset of a task is just (xs, ys). Bit-valued outputs use the duty code on [0, 1] so the per-step error is
graded (the fraction of the step the output wire was wrong).

Kinds (n_in -> n_out):
  fir           FIR low/high/band pass on a quantized level stream (1 -> 1)
  iir           RBJ biquad low/high/band pass (1 -> 1)
  oscillator    square wave of half-period h = m / R samples while `enable` is high, phase reset on enable (1 -> 1);
                fractional h puts edges inside a sample, so the duty target needs sub-sample timing
  counter       pulses counted mod n with a reset line; output the wrap pulse or the count in binary (2 -> 1 | log2 n)
  sequence      detect a random k-bit pattern (overlaps allowed); planted occurrences and near misses (1 -> 1)
  pulse_width   is the last completed pulse at least W samples wide (guard band g around W) (1 -> 1)
  frequency     which of two tones (half-periods h_lo, h_hi) is present, decided after one slow period;
                Thompson's 1 kHz / 10 kHz tone discriminator is the catalog entry `thompson_tone` (1 -> 1)
  integrator    saturating (optionally leaky) accumulator of a level stream with a reset line (2 -> 1)
  boolean       m random k-input Boolean functions of per-input delayed bits (k -> m)
  multirate     a pulse-density stream decoded over windows of r_a samples, combined with a level updated every
                r_b samples (sum, product, gate, absdiff) (2 -> 1)
  interval      time between an event on line A and the next on line B: compare to a threshold or convert to a
                level (2 -> 1)

Nonlinearity values for kinds that are not pure Boolean maps (documented constants; Boolean kinds are computed):
  fir, iir 0 (linear); integrator 0.25 (affine with saturation and reset gating); interval/measure 0.25;
  oscillator 0.5 (phase to square wave); multirate: sum 0, product / gate / absdiff 0.5;
  pulse_width, frequency, interval/compare 1.0 (a threshold decision on a measured duration).
"""
from __future__ import annotations

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import DemandTags, FamilyBase, InCode, IOSpec, OutCode, TimeModel, as_key, np_rng, task_uid, time_model

F32 = jnp.float32
I32 = jnp.int32

# ---- small helpers ---------------------------------------------------------------------------------------------


def _ri(rng, lo, hi):
    """Uniform integer in [lo, hi] (inclusive)."""
    return int(rng.integers(lo, hi + 1))


def _lu(rng, lo, hi):
    """Log-uniform float in [lo, hi]."""
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _quantize(v, Q):
    """Levels -1 + 2k / (Q - 1), k = 0..Q-1."""
    return jnp.round((jnp.clip(v, -1.0, 1.0) + 1.0) / 2.0 * (Q - 1)) / (Q - 1) * 2.0 - 1.0


def _levels(key, T, Q, n=1):
    """Broadband quantized level stream [T, n] in [-1, 1]: white noise plus three random sinusoids."""
    kw, kf, kp, ka = jax.random.split(key, 4)
    white = jax.random.uniform(kw, (T, n), F32, -1.0, 1.0)
    f = jnp.exp(jax.random.uniform(kf, (3, n), F32, math.log(0.003), math.log(0.25)))
    ph = jax.random.uniform(kp, (3, n), F32, 0.0, 2 * math.pi)
    amp = jax.random.uniform(ka, (3, n), F32, 0.0, 1.0)
    t = jnp.arange(T, dtype=F32)[:, None, None]
    sines = (amp * jnp.sin(2 * math.pi * f * t + ph)).sum(1) / 1.5
    return _quantize(0.5 * white + 0.7 * sines, Q)


def _segments(key, T, mean_len):
    """Piecewise segmentation with geometric lengths: (segment id [T], age since segment start [T])."""
    change = jax.random.bernoulli(key, 1.0 / mean_len, (T,)).at[0].set(True)
    seg = jnp.cumsum(change.astype(I32)) - 1
    t = jnp.arange(T, dtype=I32)
    start = jax.lax.cummax(jnp.where(change, t, 0), axis=0)
    return seg, t - start


def _pulse_stream(starts, widths, T):
    """bool [T]: t is inside some pulse [starts_j, starts_j + widths_j) (starts increasing)."""
    t = jnp.arange(T, dtype=I32)
    j = jnp.searchsorted(starts, t, side="right") - 1
    jj = jnp.clip(j, 0, starts.shape[0] - 1)
    return (j >= 0) & (t < starts[jj] + widths[jj])


def anf_degree(tt) -> int:
    """GF(2) algebraic degree of a truth table of length 2^n (index bit i = variable i). Constant 0 -> 0."""
    a = np.asarray(tt, np.uint8).copy()
    n = int(round(math.log2(len(a))))
    for i in range(n):
        s = 1 << i
        a = a.reshape(-1, 2, s)
        a[:, 1, :] ^= a[:, 0, :]
        a = a.reshape(-1)
    w = np.array([bin(i).count("1") for i in range(len(a))])
    return int(w[a == 1].max()) if a.any() else 0


def walsh_nl(tt) -> float:
    """Hamming distance to the nearest affine Boolean function, normalized by the bent-function bound."""
    f = 1 - 2 * np.asarray(tt, np.int64)
    n = int(round(math.log2(len(f))))
    h = 1
    while h < len(f):
        f = f.reshape(-1, 2, h)
        f = np.stack([f[:, 0] + f[:, 1], f[:, 0] - f[:, 1]], 1).reshape(-1)
        h *= 2
    nl = 2 ** (n - 1) - np.abs(f).max() / 2
    bound = 2 ** (n - 1) - 2 ** (n / 2 - 1)
    return float(nl / bound) if bound > 0 else 0.0


def bool_nonlinearity(tables) -> tuple[float, float]:
    """(mean normalized degree (deg - 1) / (n - 1), mean Walsh nonlinearity) over output truth tables."""
    degs, wnl = [], []
    for tt in tables:
        n = int(round(math.log2(len(tt))))
        d = anf_degree(tt)
        degs.append(0.0 if n <= 1 else max(d - 1, 0) / (n - 1))
        wnl.append(walsh_nl(tt))
    return float(np.mean(degs)), float(np.mean(wnl))


def _level_code(in_code, Q):
    if in_code == "thermometer":
        return InCode("thermometer", max(Q - 1, 1), -1.0, 1.0)
    if in_code == "binary":
        return InCode("binary", max(int(math.ceil(math.log2(Q))), 1), -1.0, 1.0)
    return InCode("sigma_delta", 1, -1.0, 1.0)


BIT_IN = InCode("bit", 1, 0.0, 1.0)
BIT_OUT = OutCode("duty", 1, 0.0, 1.0)
LEVEL_OUT = OutCode("duty", 1, -1.0, 1.0)
IN_CODES = ("thermometer", "binary", "sigma_delta")


def _mut_num(rng, v, lo, hi, kind):
    if kind == "int":
        return int(np.clip(v + rng.choice([-2, -1, 1, 2]), lo, hi))
    if kind == "log":
        return float(np.clip(v * math.exp(rng.normal(0, 0.3)), lo, hi))
    return float(np.clip(v + rng.normal(0, 0.15) * (hi - lo), lo, hi))


# ---- kinds -----------------------------------------------------------------------------------------------------


class Kind:
    """Base class. SPACE: param -> (lo, hi, "int" | "float" | "log") or a tuple of categories (mutation space,
    wider than the extrap box; static array sizes bound it)."""
    name = ""
    n_z = 0
    SPACE: dict = {}

    def draw(self, rng, split) -> dict:
        raise NotImplementedError

    def static(self, p) -> tuple:
        return (1, 1)

    def arrays(self, p) -> dict:
        raise NotImplementedError

    def gen(self, a, key, T, st):
        raise NotImplementedError

    def init(self, a, st):
        return jnp.zeros((), F32)

    def ref(self, a, st, ts, x, z):
        raise NotImplementedError

    def tags(self, p) -> dict:
        raise NotImplementedError

    def io(self, p):
        raise NotImplementedError

    def min_horizon(self, p) -> int:
        return 0

    def fix(self, p, rng) -> dict:
        return p

    def mutate(self, p, rng) -> dict:
        p = dict(p)
        names = list(self.SPACE)
        for name in rng.choice(names, size=min(len(names), 1 + int(rng.random() < 0.3)), replace=False):
            spec = self.SPACE[name]
            if isinstance(spec[-1], str) and spec[-1] in ("int", "float", "log"):
                p[name] = _mut_num(rng, p[name], spec[0], spec[1], spec[2])
            else:
                p[name] = spec[int(rng.integers(len(spec)))]
        return self.fix(p, rng)

    def episode_return(self, p, xs, zs, ys, ya, io) -> float:
        """The task's own criterion on a whole episode, in [0, 1] (default: 1 - mean normalized abs error)."""
        rng_ = io.act_hi - io.act_lo
        return float(1.0 - np.mean(np.abs(np.asarray(ya) - np.asarray(ys)) / rng_))


def _fir_taps(band, L, fc, bw):
    n = np.arange(L) - (L - 1) / 2
    win = np.hamming(L) if L > 2 else np.ones(L)

    def lp(f):
        h = 2 * f * np.sinc(2 * f * n) * win
        return h / h.sum()

    if band == "low":
        h = lp(fc)
    elif band == "high":
        h = lp(0.5 - fc) * np.cos(np.pi * np.arange(L))          # lowpass shifted to Nyquist
        h = h - win * (h.sum() / win.sum())                       # exact zero at DC
        h = h / max(abs(np.sum(h * np.cos(np.pi * np.arange(L)))), 1e-6)   # unit gain at Nyquist
    else:
        h = 2 * lp(bw / 2) * np.cos(2 * np.pi * fc * n)
        g = abs(np.sum(h * np.exp(-2j * np.pi * fc * np.arange(L))))
        h = h / max(g, 1e-6)
    return h


class FIR(Kind):
    name = "fir"
    MAXL = 32
    SPACE = dict(L=(2, 32, "int"), fc=(0.02, 0.48, "float"), bw=(0.02, 0.3, "float"), Q=(2, 32, "int"),
                 band=("low", "high", "band"), in_code=IN_CODES)

    def draw(self, rng, split):
        band = ("low", "high", "band")[rng.integers(3)]
        L = _ri(rng, 17, 32) if split == "extrap" else _ri(rng, 2, 16)
        fc = rng.uniform(0.1, 0.4) if band == "band" else rng.uniform(0.05, 0.4)
        return dict(band=band, L=L, fc=float(fc), bw=float(rng.uniform(0.05, 0.2)), Q=_ri(rng, 4, 32),
                    in_code=IN_CODES[rng.integers(3)])

    def arrays(self, p):
        h = np.zeros(self.MAXL, np.float32)
        h[:p["L"]] = _fir_taps(p["band"], p["L"], p["fc"], p["bw"])
        return dict(h=h, Q=np.float32(p["Q"]))

    def gen(self, a, key, T, st):
        return _levels(key, T, a["Q"]), jnp.zeros((T, 0), F32)

    def init(self, a, st):
        return jnp.zeros(self.MAXL, F32)

    def ref(self, a, st, buf, x, z):
        buf = jnp.concatenate([x[:1], buf[:-1]])
        return buf, jnp.clip(jnp.dot(a["h"], buf), -1.0, 1.0)[None]

    def tags(self, p):
        return dict(memory_depth=p["L"], timing_precision=1.0, integration=0, nonlinearity=0.0, multirate=1,
                    extra=dict(state_bits=(p["L"] - 1) * math.log2(p["Q"])))

    def io(self, p):
        return (_level_code(p["in_code"], p["Q"]),), (LEVEL_OUT,)


class IIR(Kind):
    name = "iir"
    SPACE = dict(f0=(0.002, 0.45, "log"), Qf=(0.3, 20.0, "log"), Q=(2, 32, "int"),
                 band=("low", "high", "band"), in_code=IN_CODES)

    MEM_SPLIT = 48       # train: impulse response shorter than this; extrap: longer (low f0 or sharp resonance)

    def draw(self, rng, split):
        band = ("low", "high", "band")[rng.integers(3)]
        p = dict(band=band, Q=_ri(rng, 4, 32), in_code=IN_CODES[rng.integers(3)])
        for _ in range(1000):
            if split == "extrap":
                if rng.random() < 0.5:
                    p.update(f0=_lu(rng, 0.003, 0.03), Qf=float(rng.uniform(0.5, 2.0)))
                else:
                    p.update(f0=_lu(rng, 0.02, 0.25), Qf=float(rng.uniform(3.0, 12.0)))
            else:
                p.update(f0=_lu(rng, 0.02, 0.25), Qf=float(rng.uniform(0.5, 2.0)))
            if (self.memory(p) > self.MEM_SPLIT) == (split == "extrap"):
                break
        return p

    def memory(self, p):
        _, a = self.coeffs(p)
        r = float(np.abs(np.roots([1.0, a[0], a[1]])).max())
        return max(int(math.ceil(math.log(0.01) / math.log(r))), 1) if r > 0 else 1

    @staticmethod
    def coeffs(p):
        w0 = 2 * math.pi * p["f0"]
        c, al = math.cos(w0), math.sin(w0) / (2 * p["Qf"])
        b = {"low": ((1 - c) / 2, 1 - c, (1 - c) / 2), "high": ((1 + c) / 2, -(1 + c), (1 + c) / 2),
             "band": (al, 0.0, -al)}[p["band"]]
        a0 = 1 + al
        return np.array(b) / a0, np.array([-2 * c, 1 - al]) / a0

    def arrays(self, p):
        b, a = self.coeffs(p)
        return dict(b=b.astype(np.float32), a=a.astype(np.float32), Q=np.float32(p["Q"]))

    def gen(self, a, key, T, st):
        return _levels(key, T, a["Q"]), jnp.zeros((T, 0), F32)

    def init(self, a, st):
        return jnp.zeros(4, F32)                       # x1, x2, y1, y2

    def ref(self, a, st, s, x, z):
        b, aa = a["b"], a["a"]
        y = b[0] * x[0] + b[1] * s[0] + b[2] * s[1] - aa[0] * s[2] - aa[1] * s[3]
        return jnp.stack([x[0], s[0], y, s[2]]), jnp.clip(y, -1.0, 1.0)[None]

    def tags(self, p):
        _, a = self.coeffs(p)
        r = float(np.abs(np.roots([1.0, a[0], a[1]])).max())
        return dict(memory_depth=self.memory(p), timing_precision=1.0, integration=1, nonlinearity=0.0, multirate=1,
                    extra=dict(pole_radius=r, state_bits=4 * math.log2(p["Q"])))

    def io(self, p):
        return (_level_code(p["in_code"], p["Q"]),), (LEVEL_OUT,)


class Oscillator(Kind):
    name = "oscillator"
    SPACE = dict(m=(1, 1024, "int"), R=(1, 2, 4, 8), tol=(0.01, 0.2, "float"))

    def draw(self, rng, split):
        R = (1, 2)[rng.integers(2)]
        m = _ri(rng, 2 * R, 16 * R)
        if split == "extrap":
            if rng.random() < 0.5:
                m = _ri(rng, 16 * R + 1, 64 * R)               # slower: longer count
            else:
                R = (4, 8)[rng.integers(2)]                    # finer: edges between samples
                m = 2 * _ri(rng, R, 8 * R) + 1                 # odd m: h = m / R is not a coarser fraction
        return dict(m=m, R=R, tol=float(rng.uniform(0.03, 0.1)))

    def fix(self, p, rng):
        p["m"] = int(np.clip(p["m"], p["R"], 128 * p["R"]))
        return p

    @staticmethod
    def half(p):
        return p["m"] / p["R"]

    def min_horizon(self, p):
        return int(8 * 8 * 2 * self.half(p))

    def arrays(self, p):
        h = self.half(p)
        return dict(h=np.float32(h), seg=np.float32(8 * 2 * h))

    def gen(self, a, key, T, st):
        k1, k2 = jax.random.split(key)
        seg, _ = _segments(k1, T, a["seg"])
        en = ((seg + jax.random.bernoulli(k2).astype(I32)) % 2).astype(F32)
        return en[:, None], jnp.zeros((T, 0), F32)

    def ref(self, a, st, ph, x, z):
        h = a["h"]
        P = 2 * h

        def H(v):          # integral of the square wave (high on [0, h) of each period) from 0 to v
            k = jnp.floor(v / P)
            return k * h + jnp.minimum(v - k * P, h)

        en = x[0] > 0.5
        y = jnp.where(en, H(ph + 1.0) - H(ph), 0.0)
        return jnp.where(en, jnp.mod(ph + 1.0, P), 0.0), y[None]

    def tags(self, p):
        h, R = self.half(p), p["R"]
        Rr = R // math.gcd(p["m"], R)
        return dict(memory_depth=2 * h, timing_precision=float(Rr), integration=1, nonlinearity=0.5, multirate=1,
                    extra=dict(period=2 * h, tol=p["tol"], timing_range=2 * h * Rr, state_bits=math.log2(2 * p["m"])))

    def io(self, p):
        return (BIT_IN,), (BIT_OUT,)

    def episode_return(self, p, xs, zs, ys, ya, io):
        """Fraction of enabled runs (>= 4 periods) whose measured output period is within tol of the target,
        ignoring phase (the task's own criterion; the teacher additionally fixes the phase)."""
        P = 2 * self.half(p)
        en = np.asarray(xs)[:, 0] > 0.5
        out = np.asarray(ya)[:, 0] > 0.5
        idx = np.flatnonzero(np.diff(np.concatenate([[0], en.astype(int), [0]])))
        ok, n = 0, 0
        for s, e in zip(idx[::2], idx[1::2]):
            if e - s < 4 * P:
                continue
            n += 1
            o = out[s + int(P):e]
            rise = np.flatnonzero(o[1:] & ~o[:-1])
            if len(rise) >= 2:
                Pm = (rise[-1] - rise[0]) / (len(rise) - 1)
                ok += abs(Pm / P - 1) <= p["tol"] + 1.0 / P
        return float(ok / n) if n else super().episode_return(p, xs, zs, ys, ya, io)


@functools.lru_cache(maxsize=None)
def _counter_tables(n, out):
    m = max(int(math.ceil(math.log2(n))), 1)
    tabs = []
    nb = 1 if out == "wrap" else m
    for j in range(nb):
        tt = np.zeros(2 ** (m + 2), np.uint8)
        for c in range(2 ** m):
            for pulse in (0, 1):
                for reset in (0, 1):
                    if c >= n:
                        continue
                    c1 = 0 if reset else (c + pulse) % n
                    v = (pulse and not reset and c1 == 0) if out == "wrap" else (c1 >> (m - 1 - j)) & 1
                    tt[c | (pulse << m) | (reset << (m + 1))] = v
        tabs.append(tt)
    return bool_nonlinearity(tabs)


class Counter(Kind):
    name = "counter"
    SPACE = dict(n=(2, 32, "int"), out=("wrap", "binary"), p_pulse=(0.05, 0.9, "float"),
                 p_reset=(0.0005, 0.05, "log"))

    def draw(self, rng, split):
        n = _ri(rng, 9, 32) if split == "extrap" else _ri(rng, 2, 8)
        return dict(n=n, out=("wrap", "binary")[rng.integers(2)], p_pulse=float(rng.uniform(0.1, 0.6)),
                    p_reset=_lu(rng, 0.002, 0.02))

    @staticmethod
    def m(p):
        return max(int(math.ceil(math.log2(p["n"]))), 1)

    def static(self, p):     # (n_in, n_out, state bits, binary output?) -- the output mode changes n_out
        m = self.m(p)
        return (2, 1 if p["out"] == "wrap" else m, m, p["out"] == "binary")

    def arrays(self, p):
        return dict(n=np.int32(p["n"]), pp=np.float32(p["p_pulse"]), pr=np.float32(p["p_reset"]))

    def gen(self, a, key, T, st):
        k1, k2 = jax.random.split(key)
        x = jnp.stack([jax.random.bernoulli(k1, a["pp"], (T,)), jax.random.bernoulli(k2, a["pr"], (T,))], -1)
        return x.astype(F32), jnp.zeros((T, 0), F32)

    def init(self, a, st):
        return jnp.zeros((), I32)

    def ref(self, a, st, c, x, z):
        pulse, reset = x[0] > 0.5, x[1] > 0.5
        c1 = jnp.where(reset, 0, (c + pulse.astype(I32)) % a["n"])
        m = st[2]
        wrap = (pulse & ~reset & (c1 == 0)).astype(F32)[None]
        bits = ((c1 >> jnp.arange(m - 1, -1, -1, dtype=I32)) & 1).astype(F32)
        return c1, (bits if st[3] else wrap)

    def tags(self, p):
        deg, wnl = _counter_tables(p["n"], p["out"])
        mem = min(p["n"] / p["p_pulse"], 1.0 / p["p_reset"])
        return dict(memory_depth=float(mem), timing_precision=1.0, integration=1, nonlinearity=deg, multirate=1,
                    extra=dict(state_bits=math.log2(p["n"]), walsh_nl=wnl))

    def io(self, p):
        return (BIT_IN, BIT_IN), (BIT_OUT,) * self.static(p)[1]


class Sequence(Kind):
    name = "sequence"
    MAXK = 16
    SPACE = dict(p_plant=(0.01, 0.3, "float"), p_near=(0.0, 0.8, "float"))

    def draw(self, rng, split):
        k = _ri(rng, 7, 12) if split == "extrap" else _ri(rng, 2, 6)
        return dict(pattern=rng.integers(0, 2, k).tolist(), p_plant=float(rng.uniform(0.03, 0.12)),
                    p_near=float(rng.uniform(0.2, 0.5)))

    def mutate(self, p, rng):
        p = dict(p, pattern=list(p["pattern"]))
        r = rng.random()
        if r < 0.4:
            i = int(rng.integers(len(p["pattern"])))
            p["pattern"][i] ^= 1
        elif r < 0.7:
            if (rng.random() < 0.5 and len(p["pattern"]) < self.MAXK) or len(p["pattern"]) <= 1:
                p["pattern"].append(int(rng.integers(2)))
            else:
                p["pattern"].pop()
        else:
            return super().mutate(p, rng)
        return p

    def arrays(self, p):
        pat = np.zeros(self.MAXK, np.int32)
        pat[:len(p["pattern"])] = p["pattern"]
        return dict(pat=pat, k=np.int32(len(p["pattern"])), pp=np.float32(p["p_plant"]), pn=np.float32(p["p_near"]))

    def gen(self, a, key, T, st):
        kb, ks, kn, kf = jax.random.split(key, 4)
        t = jnp.arange(T, dtype=I32)
        base = jax.random.bernoulli(kb, 0.5, (T,)).astype(I32)
        start = jax.random.bernoulli(ks, a["pp"], (T,))
        flip = jnp.where(jax.random.bernoulli(kn, a["pn"], (T,)), jax.random.randint(kf, (T,), 0, a["k"]), -1)
        s = jax.lax.cummax(jnp.where(start, t, -1), axis=0)
        d = t - s
        inpl = (s >= 0) & (d < a["k"])
        sc, dc = jnp.clip(s, 0, T - 1), jnp.clip(d, 0, self.MAXK - 1)
        bit = a["pat"][dc] ^ (flip[sc] == d).astype(I32)
        return jnp.where(inpl, bit, base).astype(F32)[:, None], jnp.zeros((T, 0), F32)

    def init(self, a, st):
        return jnp.zeros(self.MAXK, I32)

    def ref(self, a, st, reg, x, z):
        reg = jnp.concatenate([(x[:1] > 0.5).astype(I32), reg[:-1]])
        j = jnp.arange(self.MAXK)
        want = a["pat"][jnp.clip(a["k"] - 1 - j, 0, self.MAXK - 1)]
        hit = jnp.all((reg == want) | (j >= a["k"]))
        return reg, hit.astype(F32)[None]

    def tags(self, p):
        k = len(p["pattern"])
        tt = np.zeros(2 ** k, np.uint8)
        tt[sum(b << (k - 1 - i) for i, b in enumerate(p["pattern"]))] = 1
        deg, wnl = bool_nonlinearity([tt])
        return dict(memory_depth=k, timing_precision=1.0, integration=0, nonlinearity=deg, multirate=1,
                    extra=dict(k=k, state_bits=math.log2(k + 1), walsh_nl=wnl))

    def io(self, p):
        return (BIT_IN,), (BIT_OUT,)


class PulseWidth(Kind):
    name = "pulse_width"
    SPACE = dict(W=(2, 64, "int"), g=(0, 3, "int"))

    def draw(self, rng, split):
        W = _ri(rng, 13, 48) if split == "extrap" else _ri(rng, 2, 12)
        return self.fix(dict(W=W, g=_ri(rng, 0, 1)), rng)

    def fix(self, p, rng):
        p["g"] = int(min(p["g"], max(p["W"] - 2, 0)))
        return p

    def min_horizon(self, p):
        return 64 * p["W"]

    def arrays(self, p):
        return dict(W=np.int32(p["W"]), g=np.int32(p["g"]))

    def gen(self, a, key, T, st):
        kc, kw, kg = jax.random.split(key, 3)
        W, g = a["W"], a["g"]
        c = jax.random.bernoulli(kc, 0.5, (T,))
        w = jax.random.randint(kw, (T,), jnp.where(c, W + g, 1), jnp.where(c, 2 * W + g, W - g))
        gap = jax.random.randint(kg, (T,), 1, 2 * W + 1)
        starts = jnp.cumsum(gap + w) - w
        return _pulse_stream(starts, w, T).astype(F32)[:, None], jnp.zeros((T, 0), F32)

    def init(self, a, st):
        return jnp.zeros(3, I32)                          # run length, class, previous input

    def ref(self, a, st, s, x, z):
        on = (x[0] > 0.5).astype(I32)
        fell = (on == 0) & (s[2] == 1)
        c = jnp.where(fell, (s[0] >= a["W"]).astype(I32), s[1])
        r = jnp.where(on == 1, s[0] + 1, 0)
        return jnp.stack([r, c, on]), c.astype(F32)[None]

    def tags(self, p):
        W, g = p["W"], p["g"]
        return dict(memory_depth=2 * W + g, timing_precision=1.0 / (2 * g + 1), integration=1, nonlinearity=1.0,
                    multirate=1, extra=dict(timing_range=W / (2 * g + 1), state_bits=math.log2(2 * W + g + 1) + 1))

    def io(self, p):
        return (BIT_IN,), (BIT_OUT,)


class Frequency(Kind):
    name = "frequency"
    n_z = 2
    SPACE = dict(h_lo=(0.5, 64.0, "log"), h_hi=(0.6, 256.0, "log"))

    def draw(self, rng, split):
        """train: h_lo in {1, 1.5, ..., 6}, ratio in [2, 8], h_hi <= 24 (resolution h_hi - h_lo >= 1 sample).
        extrap: finer (h_hi - h_lo in [0.2, 0.9]: sub-sample resolution) or slower (h_hi in (24, 64])."""
        if split == "extrap":
            if rng.random() < 0.5:
                h_lo = _ri(rng, 2, 5) / 2
                return dict(h_lo=float(h_lo), h_hi=float(h_lo + rng.uniform(0.2, 0.9)))
            h_hi = float(rng.uniform(24.5, 64.0))
            return dict(h_lo=float(max(round(2 * h_hi / rng.uniform(2.0, 4.0)) / 2, 1.0)), h_hi=h_hi)
        h_lo = _ri(rng, 2, 12) / 2
        return dict(h_lo=float(h_lo), h_hi=float(h_lo * rng.uniform(2.0, min(8.0, 24.0 / h_lo))))

    def fix(self, p, rng):
        p["h_hi"] = float(max(p["h_hi"], 1.05 * p["h_lo"]))
        return p

    @staticmethod
    def latency(p):
        return int(math.ceil(2 * p["h_hi"])) + 1

    def min_horizon(self, p):
        return 8 * self._seg(p)

    def _seg(self, p):
        return max(8 * p["h_hi"], 2 * self.latency(p))

    def arrays(self, p):
        return dict(lo=np.float32(p["h_lo"]), hi=np.float32(p["h_hi"]), L=np.int32(self.latency(p)),
                    seg=np.float32(self._seg(p)))

    def gen(self, a, key, T, st):
        k1, k2 = jax.random.split(key)
        seg, age = _segments(k1, T, a["seg"])
        label = jax.random.bernoulli(k2, 0.5, (T,))[seg]
        h = jnp.where(label, a["hi"], a["lo"])
        x = (jnp.floor(age / h).astype(I32) % 2 == 0).astype(F32)
        return x[:, None], jnp.stack([label.astype(F32), age.astype(F32)], -1)

    def ref(self, a, st, c, x, z):
        c = jnp.where(z[1] >= a["L"], z[0], c)
        return c, c[None]

    def tags(self, p):
        res = p["h_hi"] - p["h_lo"]
        return dict(memory_depth=self.latency(p), timing_precision=1.0 / res, integration=1, nonlinearity=1.0,
                    multirate=2, extra=dict(timing_range=2 * p["h_hi"] / res, ratio=p["h_hi"] / p["h_lo"]))

    def io(self, p):
        return (BIT_IN,), (BIT_OUT,)


class Integrator(Kind):
    name = "integrator"
    SPACE = dict(g=(1 / 1024, 1.0, "log"), lam=(0.8, 1.0, "float"), p_reset=(1e-4, 0.2, "log"), Q=(2, 32, "int"),
                 in_code=IN_CODES)

    def draw(self, rng, split):
        if split == "extrap":
            g, pr = _lu(rng, 1 / 256, 1 / 32), _lu(rng, 1 / 1024, 1 / 128)
            lam = 1.0 if rng.random() < 0.5 else float(rng.uniform(0.993, 0.999))
        else:
            g, pr = _lu(rng, 1 / 32, 1 / 4), _lu(rng, 1 / 128, 1 / 16)
            lam = 1.0 if rng.random() < 0.5 else float(rng.uniform(0.9, 0.99))
        return dict(g=g, lam=lam, p_reset=pr, Q=_ri(rng, 4, 16), in_code=IN_CODES[rng.integers(3)])

    def min_horizon(self, p):
        return int(min(4 / p["p_reset"], 8192))

    def static(self, p):
        return (2, 1)

    def arrays(self, p):
        return dict(g=np.float32(p["g"]), lam=np.float32(p["lam"]), pr=np.float32(p["p_reset"]),
                    Q=np.float32(p["Q"]))

    def gen(self, a, key, T, st):
        kl, ks, kb, kr = jax.random.split(key, 4)
        seg, _ = _segments(ks, T, 64.0)
        bias = jax.random.uniform(kb, (T,), F32, -0.5, 0.5)[seg]
        x = _quantize(0.5 * _levels(kl, T, 64.0)[:, 0] + bias, a["Q"])
        r = jax.random.bernoulli(kr, a["pr"], (T,)).astype(F32)
        return jnp.stack([x, r], -1), jnp.zeros((T, 0), F32)

    def ref(self, a, st, acc, x, z):
        acc = jnp.where(x[1] > 0.5, 0.0, jnp.clip(a["lam"] * acc + a["g"] * x[0], -1.0, 1.0))
        return acc, acc[None]

    def tags(self, p):
        mem = 1.0 / p["p_reset"]
        if p["lam"] < 1:
            mem = min(mem, 1.0 / (1.0 - p["lam"]))
        return dict(memory_depth=float(mem), timing_precision=1.0, integration=1, nonlinearity=0.25, multirate=1,
                    extra=dict(gain=p["g"], leak=p["lam"]))

    def io(self, p):
        return (_level_code(p["in_code"], p["Q"]), BIT_IN), (LEVEL_OUT,)


class Boolean(Kind):
    name = "boolean"
    MAXD = 16
    SPACE = {}

    def draw(self, rng, split):
        k, m, dmax = _ri(rng, 1, 4), _ri(rng, 1, 2), _ri(rng, 0, 3)
        if split == "extrap":
            m = _ri(rng, 1, 3)
            if rng.random() < 0.5:
                k = _ri(rng, 5, 8)
            else:
                dmax = _ri(rng, 4, 15)
        delays = [_ri(rng, 0, dmax) for _ in range(k)]
        return dict(delays=delays, tables=[self._table(rng, k) for _ in range(m)])

    @staticmethod
    def _table(rng, k):
        while True:
            tt = rng.integers(0, 2, 2 ** k)
            if 0 < tt.sum() < len(tt):
                return tt.tolist()

    def mutate(self, p, rng):
        delays, tables = list(p["delays"]), [list(t) for t in p["tables"]]
        k, r = len(delays), rng.random()
        if r < 0.4:                                              # flip one truth-table entry
            j = int(rng.integers(len(tables)))
            i = int(rng.integers(2 ** k))
            tables[j][i] ^= 1
            if not 0 < sum(tables[j]) < len(tables[j]):
                tables[j][i] ^= 1
        elif r < 0.7:                                            # shift one input's delay
            i = int(rng.integers(k))
            delays[i] = int(np.clip(delays[i] + rng.choice([-1, 1]), 0, self.MAXD - 1))
        elif r < 0.85:                                           # add / drop an input
            if (k < 8 and rng.random() < 0.5) or k == 1:
                delays.append(_ri(rng, 0, max(delays)))
                tables = [t + t for t in tables]                 # new input ignored at first
                j = int(rng.integers(len(tables)))
                tables[j][int(rng.integers(len(tables[j])))] ^= 1
            else:
                delays.pop()
                tables = [t[:2 ** (k - 1)] for t in tables]
                tables = [t if 0 < sum(t) < len(t) else self._table(rng, k - 1) for t in tables]
        else:                                                    # add / drop an output
            if len(tables) < 3 and (rng.random() < 0.5 or len(tables) == 1):
                tables.append(self._table(rng, k))
            else:
                tables.pop()
        return dict(delays=delays, tables=tables)

    def static(self, p):
        return (len(p["delays"]), len(p["tables"]))

    def arrays(self, p):
        return dict(d=np.asarray(p["delays"], np.int32), tt=np.asarray(p["tables"], np.int32))

    def gen(self, a, key, T, st):
        return jax.random.bernoulli(key, 0.5, (T, st[0])).astype(F32), jnp.zeros((T, 0), F32)

    def init(self, a, st):
        return jnp.zeros((self.MAXD, st[0]), I32)

    def ref(self, a, st, buf, x, z):
        k = st[0]
        buf = jnp.concatenate([(x[None] > 0.5).astype(I32), buf[:-1]])
        vals = buf[a["d"], jnp.arange(k)]
        idx = jnp.sum(vals << jnp.arange(k, dtype=I32))
        return buf, a["tt"][:, idx].astype(F32)

    def tags(self, p):
        deg, wnl = bool_nonlinearity([np.asarray(t, np.uint8) for t in p["tables"]])
        return dict(memory_depth=max(p["delays"]) + 1, timing_precision=1.0, integration=0, nonlinearity=deg,
                    multirate=1, extra=dict(walsh_nl=wnl, state_bits=float(sum(p["delays"]))))

    def io(self, p):
        return (BIT_IN,) * len(p["delays"]), (BIT_OUT,) * len(p["tables"])


MR_OPS = ("sum", "product", "gate", "absdiff")
MR_NL = dict(sum=0.0, product=0.5, gate=0.5, absdiff=0.5)


class Multirate(Kind):
    name = "multirate"
    n_z = 2
    SPACE = dict(r_a=(2, 128, "int"), r_b=(2, 256, "int"), op=MR_OPS)

    def draw(self, rng, split):
        r_a = _ri(rng, 17, 64) if split == "extrap" else _ri(rng, 4, 16)
        return dict(r_a=r_a, r_b=_ri(rng, 8, 64), op=MR_OPS[rng.integers(4)])

    def min_horizon(self, p):
        return 16 * max(p["r_a"], p["r_b"])

    def static(self, p):
        return (2, 1)

    def arrays(self, p):
        return dict(ra=np.int32(p["r_a"]), rb=np.int32(p["r_b"]), op=np.int32(MR_OPS.index(p["op"])))

    def gen(self, a, key, T, st):
        ko, kc, kb, kq = jax.random.split(key, 4)
        oa, ob = jax.random.randint(ko, (2,), 0, jnp.stack([a["ra"], a["rb"]]))
        t = jnp.arange(T, dtype=I32)
        ra, rb = a["ra"], a["rb"]
        w, i = (t + oa) // ra, (t + oa) % ra
        cnt = jax.random.randint(kc, (T,), 0, ra + 1)
        c = cnt[w]
        bit = (((i + 1) * c * 2 + ra) // (2 * ra)) - ((i * c * 2 + ra) // (2 * ra))   # exact sigma-delta of c / ra
        full = w >= 1 + (oa > 0)                       # the previous window was seen whole
        av = jnp.where(full, 2.0 * cnt[jnp.maximum(w - 1, 0)] / ra - 1.0, 0.0)
        b = _quantize(jax.random.uniform(kb, (T,), F32, -1.0, 1.0)[(t + ob) // rb], 8.0)
        return jnp.stack([bit.astype(F32), b], -1), jnp.stack([av, b], -1)

    def ref(self, a, st, s, x, z):
        av, b = z[0], z[1]
        y = jnp.select([a["op"] == 0, a["op"] == 1, a["op"] == 2],
                       [(av + b) / 2, av * b, jnp.where(b > 0, av, 0.0)], jnp.abs(av - b) - 1.0)
        return s, y[None]

    def tags(self, p):
        return dict(memory_depth=p["r_a"], timing_precision=1.0, integration=1, nonlinearity=MR_NL[p["op"]],
                    multirate=2, extra=dict(r_a=p["r_a"], r_b=p["r_b"], op=p["op"]))

    def io(self, p):
        return (BIT_IN, InCode("thermometer", 7, -1.0, 1.0)), (LEVEL_OUT,)


class Interval(Kind):
    name = "interval"
    SPACE = dict(dmax=(3, 128, "int"), g=(0, 3, "int"), mode=("compare", "measure"))

    def draw(self, rng, split):
        dmax = _ri(rng, 17, 64) if split == "extrap" else _ri(rng, 4, 16)
        return self.fix(dict(mode=("compare", "measure")[rng.integers(2)], dmax=dmax, g=_ri(rng, 0, 1)), rng)

    def fix(self, p, rng):
        thr = p["dmax"] // 2
        p["g"] = int(max(0, min(p["g"], thr - 2, p["dmax"] - thr)))
        return p

    def min_horizon(self, p):
        return 64 * p["dmax"]

    def static(self, p):
        return (2, 1)

    def arrays(self, p):
        return dict(dmax=np.int32(p["dmax"]), thr=np.int32(p["dmax"] // 2), g=np.int32(p["g"]),
                    mode=np.int32(p["mode"] == "measure"))

    def gen(self, a, key, T, st):
        kc, kd, kg = jax.random.split(key, 3)
        J = T // 2 + 1
        c = jax.random.bernoulli(kc, 0.5, (J,))
        lo = jnp.where(a["mode"] == 1, 1, jnp.where(c, a["thr"] + a["g"], 1))
        hi = jnp.where(a["mode"] == 1, a["dmax"] + 1, jnp.where(c, a["dmax"] + 1, a["thr"] - a["g"]))
        d = jax.random.randint(kd, (J,), lo, hi)
        gap = jax.random.randint(kg, (J,), 1, a["dmax"] + 1)
        ta = jnp.cumsum(d + gap) - d
        xa = jnp.zeros(T, F32).at[ta].set(1.0, mode="drop")
        xb = jnp.zeros(T, F32).at[ta + d].set(1.0, mode="drop")
        return jnp.stack([xa, xb], -1), jnp.zeros((T, 0), F32)

    def init(self, a, st):
        return (jnp.zeros((), I32), jnp.zeros((), I32), jnp.zeros((), F32))

    def ref(self, a, st, s, x, z):
        cnt, armed, out = s
        A, B = x[0] > 0.5, x[1] > 0.5
        cnt = jnp.where(A, 0, jnp.where(armed == 1, cnt + 1, cnt))
        armed = jnp.where(A, 1, armed)
        on = B & (armed == 1)
        val = jnp.where(a["mode"] == 0, (cnt < a["thr"]).astype(F32),
                        jnp.clip(2.0 * cnt / a["dmax"] - 1.0, -1.0, 1.0))
        out = jnp.where(on, val, out)
        return (cnt, jnp.where(on, 0, armed), out), out[None]

    def tags(self, p):
        cmp = p["mode"] == "compare"
        tp = 1.0 / (2 * p["g"] + 1) if cmp else 1.0
        return dict(memory_depth=p["dmax"], timing_precision=tp, integration=1, nonlinearity=1.0 if cmp else 0.25,
                    multirate=1, extra=dict(timing_range=p["dmax"] * tp, state_bits=math.log2(p["dmax"] + 1)))

    def io(self, p):
        return (BIT_IN, BIT_IN), (BIT_OUT if p["mode"] == "compare" else LEVEL_OUT,)


KINDS: dict[str, Kind] = {k.name: k for k in (FIR(), IIR(), Oscillator(), Counter(), Sequence(), PulseWidth(),
                                              Frequency(), Integrator(), Boolean(), Multirate(), Interval())}

# ---- the task --------------------------------------------------------------------------------------------------


class SigState(NamedTuple):
    t: jax.Array          # int32 step
    xs: jax.Array         # [T, n_in] the episode's input stream (exogenous, drawn at reset)
    zs: jax.Array         # [T, n_z] latent generator variables the spec may read (e.g. the current tone)
    ts: object            # the reference's internal state before consuming x_t


def _pow2(n):
    return 1 << max(int(math.ceil(math.log2(max(n, 1)))), 0)


DEFAULT_DT = 1e-3          # seconds per sample when nothing else says (also the legacy default)
LEGACY_DT = 1e-3           # what every pre-time-model task used (except thompson_tone, 1e-5)
# Family sample rates (Hz): sensor and controller rates from 1 kHz to 1 MHz in 1-2-5 steps. A task draws one
# uniformly from those inside the family's dt range.
SAMPLE_RATES = (1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6)


class SignalTask:
    """A Task (see base.Task) whose teacher is an exact reference function. params = {"kind": ..., **kind params}.

    dt is the physical sample period (seconds); `time` the TimeModel (default: the physical one, 10 ps ticks), and
    K = time.K(dt) the derived substrate ticks per sample. K= is the legacy shorthand for time=TimeModel.legacy(K).

    open_loop: the input stream never depends on the actions, so a student's episode is fully determined by the
    reset key and DAgger relabelling reduces to labelling fresh streams (the search relies on this flag)."""
    family = "signal"
    open_loop = True

    def __init__(self, kind: str, params: dict, horizon: int = 1024, dt: float = DEFAULT_DT, K: int | None = None,
                 name: str | None = None, time: TimeModel | dict | None = None):
        self.kind = KINDS[kind]
        p = {k: v for k, v in params.items() if k != "kind"}
        self._p = p
        self.params = dict(kind=kind, **p)
        self.horizon = max(int(horizon), _pow2(self.kind.min_horizon(p)))
        self.dt = float(dt)
        self.time = time_model(time, K)
        self.K = self.time.K(self.dt)
        self._static = self.kind.static(p)
        self.obs_dim, self.act_dim = int(self._static[0]), int(self._static[1])
        self.uid = task_uid(dict(family=self.family, params=self.params, horizon=self.horizon), self.dt, self.time)
        self.name = name or f"{kind}-{self.uid[:8]}"
        self._cache = {}

    def __repr__(self):
        return f"SignalTask({self.name}, {self.params})"

    # contract ------------------------------------------------------------------------------------------------
    @property
    def tags(self) -> DemandTags:
        if "tags" not in self._cache:
            d = self.kind.tags(self._p)
            io = self.io_spec()
            tp = float(d["timing_precision"])
            extra = dict(kind=self.kind.name, in_wires=io.n_in, out_wires=io.n_out, dt=self.dt, rate_hz=1.0 / self.dt,
                         tau_res_s=self.dt / tp if tp > 0 else None, K=self.K, **io.strobe_extra(), **d.get("extra", {}))
            self._cache["tags"] = DemandTags(float(d["memory_depth"]), float(d["timing_precision"]),
                                             float(d["integration"]), float(d["nonlinearity"]), self.obs_dim,
                                             self.act_dim, int(d["multirate"]), self.family, extra)
        return self._cache["tags"]

    def io_spec(self) -> IOSpec:
        if "io" not in self._cache:
            ins, outs = self.kind.io(self._p)
            tm = self.time
            self._cache["io"] = IOSpec(ins, outs, self.K, self.dt, None if tm.is_legacy else tm.tick,
                                       tm.modulator_period(), **tm.io_kw())
        return self._cache["io"]

    @property
    def arrays(self) -> dict:
        if "a" not in self._cache:                   # numpy cache; converted per call so no tracer is ever stored
            self._cache["a"] = {k: np.asarray(v) for k, v in self.kind.arrays(self._p).items()}
        return {k: jnp.asarray(v) for k, v in self._cache["a"].items()}

    def reset(self, key) -> SigState:
        a = self.arrays
        xs, zs = self.kind.gen(a, as_key(key), self.horizon, self._static)
        return SigState(jnp.zeros((), I32), xs, zs, self.kind.init(a, self._static))

    def observe(self, s: SigState):
        return s.xs[jnp.minimum(s.t, self.horizon - 1)]

    def _ref(self, s: SigState):
        i = jnp.minimum(s.t, self.horizon - 1)
        return self.kind.ref(self.arrays, self._static, s.ts, s.xs[i], s.zs[i])

    def teacher(self, s: SigState):
        return self._ref(s)[1]

    def step(self, s: SigState, action):
        ts, y = self._ref(s)
        io = self.io_spec()
        err = jnp.mean(jnp.abs(jnp.asarray(action, F32) - y) / (io.act_hi - io.act_lo))
        s1 = SigState(s.t + 1, s.xs, s.zs, ts)
        return s1, self.observe(s1), -err, s1.t >= self.horizon

    # extras --------------------------------------------------------------------------------------------------
    def reference(self, key):
        """(xs [T, n_in], ys [T, n_out]) for one episode: the input stream and the teacher's outputs."""
        xs, zs, ys = _ref_fn(self.kind.name, self._static, self.horizon)(
            jax.tree.map(lambda v: v[None], self.arrays), as_key(key)[None])
        return np.asarray(xs[0]), np.asarray(ys[0])

    def apply(self, xs, zs=None) -> np.ndarray:
        """The reference run causally over a given input stream xs [T, n_in] (and latents zs [T, n_z] for kinds
        that read them: frequency, multirate). Returns ys [T, n_out]."""
        xs = jnp.asarray(xs, F32)
        zs = jnp.zeros((xs.shape[0], self.kind.n_z), F32) if zs is None else jnp.asarray(zs, F32)
        a = self.arrays

        def body(ts, xz):
            return self.kind.ref(a, self._static, ts, xz[0], xz[1])

        _, ys = jax.lax.scan(body, self.kind.init(a, self._static), (xs, zs))
        return np.asarray(ys)

    def stream_return(self, xs, ys, ya) -> float:
        """The task's own criterion in [0, 1] for student outputs ya [T, act_dim] on a recorded episode: inputs xs
        [T, obs_dim] and teacher outputs ys [T, act_dim] (e.g. one episode of a TeacherDataset). No latents are
        needed: no kind's criterion reads them."""
        return float(self.kind.episode_return(self._p, np.asarray(xs), None, np.asarray(ys), np.asarray(ya),
                                              self.io_spec()))

    def episode_return(self, key, ya) -> float:
        """The task's own criterion in [0, 1] for student outputs ya [T, n_out] on the episode drawn from key."""
        xs, zs, ys = _ref_fn(self.kind.name, self._static, self.horizon)(
            jax.tree.map(lambda v: v[None], self.arrays), as_key(key)[None])
        return self.kind.episode_return(self._p, xs[0], zs[0], ys[0], ya, self.io_spec())


@functools.lru_cache(maxsize=None)
def _ref_fn(kind_name, static, T):
    kind = KINDS[kind_name]

    def one(a, key):
        xs, zs = kind.gen(a, key, T, static)

        def body(ts, xz):
            return kind.ref(a, static, ts, xz[0], xz[1])

        _, ys = jax.lax.scan(body, kind.init(a, static), (xs, zs))
        return xs, zs, ys

    return jax.jit(jax.vmap(one))


def reference_batch(tasks, keys) -> list[tuple[np.ndarray, np.ndarray]]:
    """(xs, ys) for many signal tasks, one key each. Tasks sharing (kind, static shape, horizon) run as one
    vmapped, jit-compiled call, so thousands of tasks cost a handful of compilations."""
    keys = [as_key(k) for k in keys] if isinstance(keys, (list, tuple)) else list(keys)
    groups: dict = {}
    for i, t in enumerate(tasks):
        groups.setdefault((t.kind.name, t._static, t.horizon), []).append(i)
    out: list = [None] * len(tasks)
    for (kn, st, T), idx in groups.items():
        a = jax.tree.map(lambda *v: jnp.stack(v), *[tasks[i].arrays for i in idx])
        xs, _, ys = _ref_fn(kn, st, T)(a, jnp.stack([keys[i] for i in idx]))
        xs, ys = np.asarray(xs), np.asarray(ys)
        for j, i in enumerate(idx):
            out[i] = (xs[j], ys[j])
    return out


# ---- family and catalog ----------------------------------------------------------------------------------------


class SignalFamily(FamilyBase):
    """Generated signal tasks. Each task draws its kind uniformly from `kinds`, then its parameters from the kind's
    train box (train / interp) or outside it along one demand axis (extrap), then (physical time model) its sample
    rate.

    dt: None (physical: drawn per task from SAMPLE_RATES within [1 us, 1 ms]; legacy: LEGACY_DT), a number (every
    task), or a range [lo, hi] in seconds (drawn from SAMPLE_RATES inside it). The rate is drawn after the kind's
    parameters, so parameters do not depend on it. time / K: as SignalTask."""
    name = "signal"

    def __init__(self, kinds=None, horizon: int = 1024, dt=None, K: int | None = None, time=None):
        self.kinds = tuple(kinds or KINDS)
        self.horizon = horizon
        self.time = time_model(time, K)
        if dt is None:
            dt = LEGACY_DT if self.time.is_legacy else (1e-6, 1e-3)
        self.dt = tuple(float(v) for v in dt) if isinstance(dt, (list, tuple)) else float(dt)
        if isinstance(self.dt, tuple):
            lo, hi = self.dt
            self._rates = [r for r in SAMPLE_RATES if lo * (1 - 1e-9) <= 1.0 / r <= hi * (1 + 1e-9)]
            if not self._rates:
                raise ValueError(f"no sample rate in SAMPLE_RATES has dt within {self.dt}")

    @property
    def K(self) -> int | None:
        """The family's K when every task shares one (fixed dt or legacy), else None."""
        if self.time.is_legacy or not isinstance(self.dt, tuple):
            return self.time.K(self.dt if not isinstance(self.dt, tuple) else self.dt[0])
        return None

    def draw_dt(self, rng) -> float:
        if not isinstance(self.dt, tuple):
            return self.dt
        return 1.0 / self._rates[int(rng.integers(len(self._rates)))]

    def sample_one(self, rng, split):
        kind = self.kinds[int(rng.integers(len(self.kinds)))]
        p = KINDS[kind].draw(rng, split)
        return SignalTask(kind, p, self.horizon, self.draw_dt(rng), time=self.time)

    def from_params(self, params, **kw):
        kw = dict(dict(horizon=self.horizon), **kw)
        if "dt" not in kw:
            if isinstance(self.dt, tuple):
                raise ValueError("from_params on a family with a dt range needs dt=")
            kw["dt"] = self.dt
        if "K" not in kw and "time" not in kw:
            kw["time"] = self.time
        return SignalTask(params["kind"], params, **kw)

    def mutate(self, task, key):
        rng = np_rng(key, 4)
        p = task.kind.mutate(dict(task._p), rng)
        return SignalTask(task.kind.name, p, self.horizon, task.dt, time=task.time)


# name -> (kind, params, physical dt in seconds). Rates are what a sensor or controller doing this job would sample at;
# legacy mode uses LEGACY_DT for all of them except thompson_tone (its dt was always 1e-5).
CATALOG = {
    # Thompson (1996): 1 kHz vs 10 kHz square waves, output steady high / low. 100 kHz sampling: half-periods of 50
    # and 5 samples.
    "thompson_tone": ("frequency", dict(h_lo=5.0, h_hi=50.0), 1e-5),
    # filters on a 10 kHz sensor stream (fc = 0.1 of the rate: a 1 kHz corner)
    "lowpass_fir8": ("fir", dict(band="low", L=8, fc=0.1, bw=0.1, Q=16, in_code="thermometer"), 1e-4),
    "highpass_fir16": ("fir", dict(band="high", L=16, fc=0.25, bw=0.1, Q=16, in_code="binary"), 1e-4),
    "lowpass_biquad": ("iir", dict(band="low", f0=0.05, Qf=0.707, Q=16, in_code="thermometer"), 1e-4),
    "bandpass_biquad": ("iir", dict(band="band", f0=0.1, Qf=2.0, Q=16, in_code="thermometer"), 1e-4),
    # oscillators clocked at 1 MHz: 125 kHz and 76.9 kHz square waves (a PWM carrier, a chopper)
    "osc_period8": ("oscillator", dict(m=4, R=1, tol=0.05), 1e-6),
    "osc_period13": ("oscillator", dict(m=13, R=2, tol=0.05), 1e-6),          # half-period 6.5: sub-sample edges
    # pulse counting on a line sampled at 1 MHz (an encoder or tachometer)
    "divide_by_3": ("counter", dict(n=3, out="wrap", p_pulse=0.5, p_reset=0.005), 1e-6),
    "mod16_counter": ("counter", dict(n=16, out="binary", p_pulse=0.3, p_reset=0.002), 1e-6),
    # a 1 Mbit/s serial line
    "detect_1011": ("sequence", dict(pattern=[1, 0, 1, 1], p_plant=0.08, p_near=0.3), 1e-6),
    # pulse width at 100 kHz (80 us pulses)
    "pulse_width_8": ("pulse_width", dict(W=8, g=0), 1e-5),
    # integrators in a 1 kHz control loop
    "integrator_reset": ("integrator", dict(g=0.0625, lam=1.0, p_reset=0.02, Q=8, in_code="thermometer"), 1e-3),
    "leaky_integrator": ("integrator", dict(g=0.05, lam=0.95, p_reset=0.005, Q=8, in_code="thermometer"), 1e-3),
    # Boolean maps of 1 MHz bit streams
    "xor2": ("boolean", dict(delays=[0, 0], tables=[[0, 1, 1, 0]]), 1e-6),
    "and2_delay3": ("boolean", dict(delays=[3, 3], tables=[[0, 0, 0, 1]]), 1e-6),
    "majority3": ("boolean", dict(delays=[0, 0, 0], tables=[[0, 0, 0, 1, 0, 1, 1, 1]]), 1e-6),
    # a pulse-density stream decoded at 10 kHz
    "multirate_product": ("multirate", dict(r_a=8, r_b=32, op="product"), 1e-4),
    # time-to-digital at 1 us resolution
    "interval_compare_12": ("interval", dict(mode="compare", dmax=12, g=0), 1e-6),
    "interval_measure_16": ("interval", dict(mode="measure", dmax=16, g=0), 1e-6),
}
LEGACY_CATALOG_DT = {"thompson_tone": 1e-5}


def catalog(horizon: int = 1024, K: int | None = None, time=None, dt: float | None = None) -> dict[str, SignalTask]:
    """Named canonical instances for the paper, each at its own physical dt (CATALOG) under `time` (default: the
    physical time model). dt overrides every entry's rate. K= / a legacy time model: the pre-time-model tasks
    (LEGACY_DT, uids unchanged)."""
    tm = time_model(time, K)
    out = {}
    for name, (kind, p, dt_phys) in CATALOG.items():
        d = dt if dt is not None else (LEGACY_CATALOG_DT.get(name, LEGACY_DT) if tm.is_legacy else dt_phys)
        out[name] = SignalTask(kind, p, horizon=horizon, dt=d, name=name, time=tm)
    return out


def catalog_dt(name: str, time: TimeModel | None = None) -> float:
    """The dt a catalog entry gets under `time` (legacy: LEGACY_DT)."""
    if time is not None and time.is_legacy:
        return LEGACY_CATALOG_DT.get(name, LEGACY_DT)
    return CATALOG[name][2]
