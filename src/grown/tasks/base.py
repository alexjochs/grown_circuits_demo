"""The task/teacher contract: what every task family hands to the circuit search.

The search never optimizes task reward directly. Every task carries a frozen, deterministic teacher; circuits are
scored by dense per-step error against the teacher's action (`score`), task return is blended in later, and the
teacher relabels the states a student visits (DAgger, `TeacherDataset.relabel`). Tasks come from generated
families and carry `DemandTags`, so generalization claims can be made about held-out regions of demand space.

Pieces defined here (the circuit side consumes exactly these):

  DemandTags   computational-demand coordinates of a task (memory, timing, integration, nonlinearity, I/O, rates)
  TimeModel    the physical time base: the PDK's tick (seconds), the sigma-delta modulator rate, the event budget
               per control step. K = round(dt / tick) is derived from it and a task's dt, never chosen
               (docs/grown_time.md)
  InCode, OutCode, IOSpec
               the bit interface: how each observation dim becomes input wires, how output wires become actions,
               and K, the number of substrate ticks per control step (derived: K = round(dt / tick))
  Task         protocol: obs_dim, act_dim, dt, horizon, tags, reset, step, observe, teacher, io_spec
  Family       protocol: sample, splits, demand_coverage (+ mutate / from_params for the frontier sampler)
  FamilyBase   shared implementation of sample / splits / demand_coverage
  rollout      one episode of any Task under its teacher or a student policy (lax.scan, vmappable over keys)
  TeacherDataset, score
               teacher-labelled (obs, act) streams with npz + json sidecar storage, DAgger relabelling, and the
               per-step error the search uses

Randomness is counter-based: every random draw is a pure function of an explicit key (a jax PRNG key or an int
seed) and a path of integers (`np_rng(key, split, i)`), so task i of a family split is reproducible from
(family, seed) alone and `sample(key, n)[:m] == sample(key, m)`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

# ---- keys, hashing ---------------------------------------------------------------------------------------------


def key_words(key) -> tuple[int, ...]:
    """Non-negative ints identifying a key: an int seed, a legacy uint32[2] jax key, or a typed jax key."""
    if isinstance(key, (int, np.integer)):
        k = int(key)
        if k < 0:
            raise ValueError("integer seeds must be non-negative")
        return ((k >> 32) & 0xFFFFFFFF, k & 0xFFFFFFFF)          # same words as jax.random.PRNGKey(k)
    arr = key
    if hasattr(key, "dtype") and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        arr = jax.random.key_data(key)
    return tuple(int(x) for x in np.asarray(arr).ravel())


def as_key(key):
    """A jax PRNG key from an int seed or a key."""
    if isinstance(key, (int, np.integer)):
        return jax.random.PRNGKey(int(key))
    return key


def np_rng(key, *path: int) -> np.random.Generator:
    """numpy Generator that is a pure function of (key, *path). Cheap (no jax dispatch), for task generation."""
    return np.random.default_rng(np.random.SeedSequence([*key_words(key), *(int(p) for p in path)]))


def jsonable(x):
    """Plain-JSON view of params / tags (numpy and tuples become lists, numpy scalars become python numbers)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return jsonable(dataclasses.asdict(x))
    return x


def stable_hash(obj, n: int = 16) -> str:
    """Hex digest of the canonical JSON of obj. Used as task uid and teacher-cache key."""
    s = json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()[:n]


# ---- demand tags -----------------------------------------------------------------------------------------------

AXES = ("memory_depth", "timing_precision", "integration", "nonlinearity", "n_in", "n_out", "multirate")


@dataclass(frozen=True)
class DemandTags:
    """Where a task sits in computational-demand space. Every family computes these itself, analytically where it
    can (signal tasks) and from the model where it must (bodies). All are "bigger is harder".

    memory_depth      Control steps of input history the teacher's current output depends on (1 = the current
                      observation suffices; a Markov full-state body teacher is 1). Signal: the analytic window
                      (FIR length, pattern length, longest timed interval), or the effective impulse-response length
                      for recursive specs (steps until the response decays to 1%), or the expected interval between
                      resets for unbounded accumulators. Bodies: 1, +1 if velocities must be inferred from positions;
                      sensorimotor adds sensor delay and a noise-averaging window.
    timing_precision  Required time resolution relative to the control step: dt / tau_res, where tau_res is the
                      finest time difference that changes the correct output. 1 = the control-step grid is enough;
                      > 1 = the circuit must resolve time below one step, which only its K electrical steps can do;
                      < 1 = a coarser tolerance suffices. Signal: from the spec (fractional oscillator half-periods,
                      guard bands of discriminators). Bodies: dt / the fastest control-relevant timescale (actuator
                      authority sqrt(2 M_jj / gear_j), gravity sqrt(extent / g)).
    integration       Order of the integration (or differentiation) chain the circuit must realize internally:
                      0 none, 1 one accumulator / leaky recursion / velocity from positions, 2 two chained, ...
    nonlinearity      0..1. 0: the output is an affine function of inputs and (linear) state. For Boolean maps:
                      the normalized GF(2) algebraic degree (deg - 1) / (n_vars - 1) of the output as a function of
                      its relevant input and state bits (Walsh distance to affine is kept in extra['walsh_nl']).
                      For mixed level/bit specs a documented per-kind value (see signal.py). Bodies: 1 - R^2 of the
                      best linear one-step model of the dynamics under random actions (contact makes this large).
    n_in, n_out       Input / output channels (obs_dim, act_dim). Wire counts after encoding are in
                      extra['in_wires'] / extra['out_wires'].
    multirate         Number of distinct timescales the task mixes. Signal: distinct input / update rates. Bodies:
                      clusters of the model's timescales separated by more than a factor of 3.
    family            Family name ("signal", "bodies", "named_bodies", "sensorimotor", ...).
    extra             Free-form: kind, state_bits, timing_range (longest timed interval / tau_res), etc.
    """
    memory_depth: float
    timing_precision: float
    integration: float
    nonlinearity: float
    n_in: int
    n_out: int
    multirate: int
    family: str
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def vector(self) -> np.ndarray:
        """float64 [7] in AXES order."""
        return np.array([float(getattr(self, a)) for a in AXES])

    def to_dict(self) -> dict:
        return jsonable(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "DemandTags":
        return cls(**{k: d[k] for k in (*AXES, "family")}, extra=dict(d.get("extra", {})))

    def replace(self, **kw) -> "DemandTags":
        return dataclasses.replace(self, **kw)


# ---- the time model --------------------------------------------------------------------------------------------

BUDGET_MODES = ("fail", "penalize", "ignore")


@dataclass(frozen=True)
class TimeModel:
    """The physical time base shared by tasks and substrates (docs/grown_time.md). Lives here, in the task layer,
    because the task owns time: its control step `dt` (seconds) is physical, and the number of substrate ticks per
    step is derived from it, K = round(dt / tick), never chosen.

    tick            seconds per substrate tick, the PDK's time resolution (10 ps default for SKY130 gate delays of
                    30-120 ps; 1 ps allowed)
    modulator_hz    sigma-delta (pulse-density) input modulator rate, 1-100 MHz: one bit per modulator period,
                    each pulse one period wide (thousands of ticks), see IOSpec.encode_events
    event_budget    committed gate-net transitions allowed per control step per lane (the energy envelope), or None
    event_rate      alternatively, transitions per second: the budget of a step is event_rate * dt (a power envelope
                    that scales with the step); event_budget wins when both are set
    budget_mode     what the search does with a lane over budget: ignore (default since 2026-10-12: only logged, the
                    archive's events_per_step axis holds the energy frontier), penalize (fitness -= budget_penalty *
                    fraction of the circuit's lanes over budget), fail (fitness = worst, error = 1; a hard envelope,
                    too aggressive as a search constraint, kept for the E1-phys configs)
    budget_penalty  see budget_mode
    sim_event_cap   COMPUTE guard, not physics: gate transitions per control step per lane above which the substrate
                    stops simulating a lane (early stop), because simulating it further would be unaffordable. A capped
                    lane cannot be scored (its later steps were not simulated), so its circuit counts as failed; the
                    fraction of capped lanes is logged (behaviour `capped`). None: no cap
    sim_event_cap_rate  the same cap as transitions per second of simulated time (cap per step = rate * dt); with
                    both set the smaller per-step cap applies
    legacy_K        reproduce a pre-time-model run: every task gets exactly this K whatever its dt, sigma-delta inputs
                    run at the tick rate, and task uids are the old ones. Only for re-running old configs; such
                    configs are labelled legacy (configs/grown/exp/e0_*, e1_* at K <= 32)
    strobe          expose the sample clock: one extra input wire (the last) that pulses once per control step, rising
                    at the step boundary t = s*K (power-up value 1 at t = 0) and falling strobe_width later, as an
                    ADC's conversion strobe or a serial link's bit clock does. The circuit may use it or ignore it.
                    It changes the circuit interface (n_in + 1), so it is part of a task's identity (uid)
    strobe_width    seconds the strobe stays high (None: one modulator period, 100 ns at 10 MHz; legacy: 1 tick);
                    quantized to whole ticks and must be shorter than the step
    """
    tick: float = 1e-11
    modulator_hz: float = 1e7
    event_budget: float | None = None
    event_rate: float | None = None
    budget_mode: str = "ignore"
    budget_penalty: float = 1.0
    sim_event_cap: float | None = None
    sim_event_cap_rate: float | None = None
    legacy_K: int | None = None
    strobe: bool = False
    strobe_width: float | None = None

    def __post_init__(self):
        if not self.tick > 0 or not self.modulator_hz > 0:
            raise ValueError(f"tick and modulator_hz must be positive: {self}")
        if self.budget_mode not in BUDGET_MODES:
            raise ValueError(f"budget_mode {self.budget_mode!r} not in {BUDGET_MODES}")
        if self.legacy_K is not None and int(self.legacy_K) < 1:
            raise ValueError("legacy_K must be >= 1")
        if self.strobe_width is not None and not self.strobe_width > 0:
            raise ValueError(f"strobe_width must be positive seconds: {self.strobe_width}")
        for f in ("sim_event_cap", "sim_event_cap_rate"):
            if getattr(self, f) is not None and not getattr(self, f) > 0:
                raise ValueError(f"{f} must be positive: {getattr(self, f)}")

    @classmethod
    def legacy(cls, K: int) -> "TimeModel":
        """The pre-time-model convention: K ticks per step whatever dt (reproducing old runs and stand-in tests)."""
        return cls(legacy_K=int(K))

    @property
    def is_legacy(self) -> bool:
        return self.legacy_K is not None

    def K(self, dt: float) -> int:
        """Ticks per control step: round(dt / tick) (legacy_K when set). dt is thereby quantized to whole ticks."""
        if self.legacy_K is not None:
            return int(self.legacy_K)
        k = int(round(float(dt) / self.tick))
        if k < 1:
            raise ValueError(f"control step dt = {dt} s is shorter than one tick ({self.tick} s)")
        return k

    def modulator_period(self) -> int:
        """Ticks per sigma-delta modulator period (1 in legacy mode: the old per-tick pulse train)."""
        if self.legacy_K is not None:
            return 1
        return max(int(round(1.0 / (self.modulator_hz * self.tick))), 1)

    def strobe_ticks(self) -> int:
        """Ticks the strobe stays high: strobe_width / tick, else one modulator period (1 tick in legacy mode)."""
        if self.strobe_width is None:
            return self.modulator_period()
        return max(int(round(float(self.strobe_width) / self.tick)), 1)

    def io_kw(self) -> dict:
        """IOSpec keywords a task adds for this time model (the sample-clock strobe; empty when off)."""
        return dict(strobe=True, strobe_width=self.strobe_ticks()) if self.strobe else {}

    def budget(self, dt: float) -> float | None:
        """Events allowed per control step of a task with step dt (None: unlimited)."""
        if self.event_budget is not None:
            return float(self.event_budget)
        if self.event_rate is not None:
            return float(self.event_rate) * float(dt)
        return None

    def cap(self, dt: float) -> float | None:
        """The compute guard: events per control step above which a lane stops being simulated (None: no cap)."""
        c = [float(self.sim_event_cap)] if self.sim_event_cap is not None else []
        if self.sim_event_cap_rate is not None:
            c.append(float(self.sim_event_cap_rate) * float(dt))
        return min(c) if c else None

    def key(self) -> dict:
        """What of the time model is part of a task's identity (its uid): the time base and the interface (the strobe
        wire, only when on, so uids without it are unchanged), not the budget."""
        k = dict(tick=self.tick, modulator_hz=self.modulator_hz)
        if self.strobe:
            k.update(strobe=True, strobe_ticks=self.strobe_ticks())
        return k

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_config(cls, d: dict | None) -> "TimeModel":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise KeyError(f"unknown time model keys {sorted(unknown)} (known: tick, modulator_hz, event_budget, "
                           "event_rate, budget_mode, budget_penalty, sim_event_cap, sim_event_cap_rate, legacy_K, "
                           "strobe, strobe_width)")
        kw = {}
        for k, v in d.items():
            if v is None:
                kw[k] = None
            elif k == "budget_mode":
                kw[k] = str(v)
            elif k == "legacy_K":
                kw[k] = int(v)
            elif k == "strobe":
                if not isinstance(v, (bool, np.bool_)) and v not in (0, 1):
                    raise ValueError(f"time.strobe must be true or false, not {v!r}")
                kw[k] = bool(v)
            else:
                kw[k] = float(v)
        return cls(**kw)


def time_model(time=None, K: int | None = None) -> TimeModel:
    """The time model of a task constructor: `time` (a TimeModel or its config dict), or the legacy shorthand
    `K=` (= TimeModel.legacy(K), kept for the stand-in era API: baselines, old tests), or the default physical
    model. Giving both is an error."""
    if K is not None:
        if time is not None:
            raise ValueError("give either time= or the legacy K=, not both")
        return TimeModel.legacy(K)
    if time is None:
        return TimeModel()
    return time if isinstance(time, TimeModel) else TimeModel.from_config(time)


def task_uid(base: dict, dt: float, time: TimeModel) -> str:
    """Task uid: legacy tasks hash exactly what they hashed before the time model (base + dt + K), so old runs'
    seeds (derived from the uid) are reproduced; physical tasks also hash the time base."""
    K = time.K(dt)
    if time.is_legacy:
        if time.strobe:                         # a different interface; uids without a strobe are unchanged
            return stable_hash(dict(base, dt=dt, K=K, strobe=time.strobe_ticks()))
        return stable_hash(dict(base, dt=dt, K=K))
    return stable_hash(dict(base, dt=dt, K=K, time=time.key()))


# ---- the bit interface -----------------------------------------------------------------------------------------

IN_KINDS = ("bit", "thermometer", "binary", "sigma_delta")
OUT_KINDS = ("bit", "duty", "pwm", "popcount")


@dataclass(frozen=True)
class InCode:
    """How one observation dim becomes input wires. The value is first mapped to u = (v - lo) / (hi - lo) in [0, 1].

    bit          1 wire: u >= 0.5.
    thermometer  `bits` wires: wire k = u > (k + 0.5) / bits (exact for a level stream of bits + 1 levels).
    binary       `bits` wires: q = floor(u * 2^bits) clipped, unsigned, MSB first.
    sigma_delta  1 wire carrying a first-order sigma-delta pulse train of density u, one bit per modulator period
                 (TimeModel.modulator_hz; IOSpec.encode_events), each 1 bit a pulse one period wide. The legacy
                 dense path (`IOSpec.encode_ticks`) runs the modulator at the tick rate; `IOSpec.encode` gives its
                 first bit from a fresh modulator (= u >= 0.5).
    """
    kind: str = "thermometer"
    bits: int = 8
    lo: float = -1.0
    hi: float = 1.0

    def __post_init__(self):
        if self.kind not in IN_KINDS:
            raise ValueError(f"input code {self.kind!r} not in {IN_KINDS}")
        if self.kind in ("bit", "sigma_delta") and self.bits != 1:
            object.__setattr__(self, "bits", 1)
        if self.bits < 1 or not self.hi > self.lo:
            raise ValueError(f"bad input code {self}")


@dataclass(frozen=True)
class OutCode:
    """How output wires become one action dim. Wire values are bits or duty cycles in [0, 1] (fraction of the K
    electrical steps the wire was high); the action is lo + f * (hi - lo) with:

    bit       1 wire, f = [wire > 0.5].
    duty      1 wire, f = its duty cycle (continuous).
    pwm       1 wire, f = its duty cycle quantized to the K + 1 levels a K-step PWM period can express.
    popcount  `bits` wires, f = mean of the wires (popcount / bits for bits).
    """
    kind: str = "duty"
    bits: int = 1
    lo: float = -1.0
    hi: float = 1.0

    def __post_init__(self):
        if self.kind not in OUT_KINDS:
            raise ValueError(f"output code {self.kind!r} not in {OUT_KINDS}")
        if self.kind != "popcount" and self.bits != 1:
            object.__setattr__(self, "bits", 1)
        if self.bits < 1 or not self.hi > self.lo:
            raise ValueError(f"bad output code {self}")


class _Group(NamedTuple):
    kind: str
    bits: int
    dims: np.ndarray      # which channels
    lo: np.ndarray
    hi: np.ndarray


def _groups(codes) -> tuple[list[_Group], np.ndarray, np.ndarray]:
    """Group channels by (kind, bits) so encode/decode is one vectorized op per group.
    Returns groups, the inverse permutation from group-concatenated wires to wire order, and wire offsets."""
    offsets = np.concatenate([[0], np.cumsum([c.bits for c in codes])]).astype(np.int64)
    by = {}
    for d, c in enumerate(codes):
        by.setdefault((c.kind, c.bits), []).append(d)
    groups, pos = [], []
    for (kind, bits), dims in by.items():
        dims = np.asarray(dims, np.int64)
        groups.append(_Group(kind, bits, dims, np.array([codes[d].lo for d in dims], np.float32),
                             np.array([codes[d].hi for d in dims], np.float32)))
        pos.extend((offsets[dims][:, None] + np.arange(bits)[None, :]).ravel().tolist())
    return groups, np.argsort(np.asarray(pos, np.int64)), offsets


@dataclass(frozen=True)
class IOSpec:
    """The circuit-facing interface of a task. Wires of channel d are contiguous, channels in order.

    K           substrate ticks per control step. Tasks fill it from their TimeModel at io_spec() time,
                K = round(dt / tick) (derived, never chosen; TimeModel.legacy_K for old runs). dt_circuit = dt / K.
    dt          seconds per control step (the task's)
    tick        seconds per tick (None: unspecified, e.g. a hand-built legacy spec)
    mod_period  ticks per sigma-delta modulator period (1: the legacy per-tick pulse train)
    strobe      one more input wire after all channels (the last, `strobe_wire`): the sample clock, high for the
                first `strobe_width` ticks of every control step (TimeModel.strobe). n_in counts it; no observation
                dim maps to it, and encode / encode_ticks / encode_events all emit it
    strobe_width  ticks the strobe is high in each step, 1 <= strobe_width < K

    Inputs reach a substrate either as dense per-tick bits (`encode_episode`, [.., T, K, n_in], only for small K: the
    stand-in and tests) or as sparse event streams (`encode_events`, any K). Both describe the same waveform when
    mod_period = 1 (tested).
    """
    inputs: tuple
    outputs: tuple
    K: int
    dt: float
    tick: float | None = None
    mod_period: int = 1
    strobe: bool = False
    strobe_width: int = 0

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        if self.K < 1:
            raise ValueError("K must be >= 1")
        if self.mod_period < 1:
            raise ValueError("mod_period must be >= 1")
        object.__setattr__(self, "strobe", bool(self.strobe))
        if self.strobe:
            w = int(self.strobe_width) if self.strobe_width else int(self.mod_period)
            if not 1 <= w < self.K:
                raise ValueError(f"strobe width {w} ticks must be at least 1 and shorter than the step (K = {self.K})")
            object.__setattr__(self, "strobe_width", w)
        else:
            object.__setattr__(self, "strobe_width", 0)
        gi, ii, oi = _groups(self.inputs)
        go, io_, oo = _groups(self.outputs)
        object.__setattr__(self, "_gin", (gi, ii, oi))
        object.__setattr__(self, "_gout", (go, io_, oo))

    # sizes
    @property
    def n_in(self) -> int:
        return int(sum(c.bits for c in self.inputs)) + int(self.strobe)

    @property
    def strobe_wire(self) -> int | None:
        """Index of the sample-clock wire (the last input), or None without a strobe."""
        return self.n_in - 1 if self.strobe else None

    def strobe_extra(self) -> dict:
        """What a task's DemandTags.extra records about the strobe (empty without one)."""
        if not self.strobe:
            return {}
        return dict(strobe=True, strobe_width_ticks=self.strobe_width,
                    strobe_width_s=None if self.tick is None else self.strobe_width * self.tick)

    @property
    def n_out(self) -> int:
        return int(sum(c.bits for c in self.outputs))

    @property
    def dt_circuit(self) -> float:
        return self.dt / self.K

    @property
    def act_lo(self) -> np.ndarray:
        return np.array([c.lo for c in self.outputs], np.float32)

    @property
    def act_hi(self) -> np.ndarray:
        return np.array([c.hi for c in self.outputs], np.float32)

    def in_wires(self, d: int) -> range:
        o = self._gin[2]
        return range(int(o[d]), int(o[d + 1]))

    def out_wires(self, d: int) -> range:
        o = self._gout[2]
        return range(int(o[d]), int(o[d + 1]))

    # encoding
    def encode(self, obs) -> jax.Array:
        """obs float [..., obs_dim] -> bool [..., n_in]: the wire values at the start of the control step."""
        obs = jnp.asarray(obs, jnp.float32)
        groups, inv, _ = self._gin
        parts = []
        for g in groups:
            u = jnp.clip((obs[..., g.dims] - g.lo) / (g.hi - g.lo), 0.0, 1.0)
            if g.kind in ("bit", "sigma_delta"):
                b = (u >= 0.5)[..., None]
            elif g.kind == "thermometer":
                b = u[..., None] > (jnp.arange(g.bits, dtype=jnp.float32) + 0.5) / g.bits
            else:   # binary, MSB first
                q = jnp.clip(jnp.floor(u * 2.0 ** g.bits), 0, 2 ** g.bits - 1).astype(jnp.int32)
                b = ((q[..., None] >> jnp.arange(g.bits - 1, -1, -1, dtype=jnp.int32)) & 1).astype(bool)
            parts.append(b.reshape(*obs.shape[:-1], -1))
        bits = jnp.concatenate(parts, -1)[..., inv]
        if self.strobe:                                   # the sample clock is high at the start of every step
            bits = jnp.concatenate([bits, jnp.ones((*obs.shape[:-1], 1), bool)], -1)
        return bits

    def init_carry(self, batch_shape=()) -> jax.Array:
        """Sigma-delta modulator state per channel (0.5 = fresh)."""
        return jnp.full((*batch_shape, len(self.inputs)), 0.5, jnp.float32)

    def encode_ticks(self, obs, carry=None) -> tuple[jax.Array, jax.Array]:
        """obs [..., obs_dim] -> (bool [..., K, n_in], carry [..., obs_dim]). Static codes hold their value for all K
        ticks; sigma-delta channels emit b_k = floor(c + (k+1) u) - floor(c + k u), carrying frac(c + K u) to the
        next control step so the pulse density is unbiased over time."""
        obs = jnp.asarray(obs, jnp.float32)
        lead = obs.shape[:-1]
        if carry is None:
            carry = self.init_carry(lead)
        static = jnp.broadcast_to(self.encode(obs)[..., None, :], (*lead, self.K, self.n_in))
        if self.strobe:                                   # high for the first strobe_width ticks of the step
            pulse = jnp.arange(self.K) < self.strobe_width
            static = static.at[..., self.strobe_wire].set(jnp.broadcast_to(pulse, (*lead, self.K)))
        groups, inv, offsets = self._gin
        sd = [g for g in groups if g.kind == "sigma_delta"]
        if not sd:
            return static, carry
        dims = np.concatenate([g.dims for g in sd])
        lo = np.concatenate([g.lo for g in sd])
        hi = np.concatenate([g.hi for g in sd])
        u = jnp.clip((obs[..., dims] - lo) / (hi - lo), 0.0, 1.0)              # [..., S]
        c = carry[..., dims]
        k = jnp.arange(self.K, dtype=jnp.float32)[:, None]                     # [K, 1]
        pulses = (jnp.floor(c[..., None, :] + (k + 1) * u[..., None, :])
                  - jnp.floor(c[..., None, :] + k * u[..., None, :])) > 0.5     # [..., K, S]
        wires = offsets[dims]                                                  # sigma-delta channels are 1 wire
        bits = static.at[..., wires].set(pulses)
        acc = c + self.K * u
        carry = carry.at[..., dims].set(acc - jnp.floor(acc))
        return bits, carry

    def encode_events(self, obs, time: "TimeModel | None" = None):
        """Sparse input events for whole open-loop episodes: obs [E, T, obs_dim] -> sim.stepper.EventStreams with one
        stream per episode (t in ticks from the episode start, sorted by (t, wire); events at t = 0 are the power-up
        input values, wires without one start at 0; every event is a change of its wire).

        Held codes (bit, thermometer, binary) are a sample-and-hold: wire values change only at step boundaries
        t = s*K, one event per wire whose value differs from step s - 1.
        sigma_delta runs a first-order modulator with period P = mod_period ticks (`time.modulator_period()` when a
        time model is given), phase-locked to the episode start: modulator bit j is emitted at t = j*P and held for
        one period, so a 1 bit is a pulse of width P and consecutive 1 bits merge into one longer pulse. Bit j samples
        the held value u_s of the step s with s*K <= j*P < (s+1)*K; with i the bit's index within its step and c_s the
        carry at the step's first bit, b_j = floor(c_s + (i+1) u_s) - floor(c_s + i u_s), and the carry passed on is
        frac(c_s + n_s u_s) for the step's n_s bits (c_0 = 0.5, as encode_ticks: with P = 1 the two agree). Pulses
        may straddle a step boundary when P does not divide K. Arithmetic is float64 (encode_ticks: float32).
        The strobe wire (when on) rises at every step boundary t = s*K (at t = 0 that is its power-up value) and falls
        at s*K + strobe_width: 2 events per step."""
        from ..sim.stepper import EventStreams
        obs = np.asarray(obs, np.float32)
        if obs.ndim == 2:
            obs = obs[None]
        E, T, _ = obs.shape
        K = int(self.K)
        P = int(time.modulator_period()) if time is not None else int(self.mod_period)
        static = np.asarray(self.encode(obs), np.uint8)                          # [E, T, n_in]
        groups, inv, offsets = self._gin
        sd = [g for g in groups if g.kind == "sigma_delta"]
        sd_wires = np.concatenate([offsets[g.dims] for g in sd]).astype(np.int64) if sd else np.zeros(0, np.int64)
        held = np.setdiff1d(np.arange(self.n_in - int(self.strobe)), sd_wires)
        parts_e, parts_t, parts_w, parts_v = [], [], [], []
        if len(held):
            x = static[:, :, held]
            prev = np.concatenate([np.zeros_like(x[:, :1]), x[:, :-1]], 1)
            e, s, i = np.nonzero(x != prev)
            parts_e.append(e)
            parts_t.append(s.astype(np.int64) * K)
            parts_w.append(held[i])
            parts_v.append(x[e, s, i])
        if sd:
            dims = np.concatenate([g.dims for g in sd])
            lo = np.concatenate([g.lo for g in sd]).astype(np.float64)
            hi = np.concatenate([g.hi for g in sd]).astype(np.float64)
            u = np.clip((obs[..., dims].astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)     # [E, T, C]
            j0 = -(-np.arange(T + 1, dtype=np.int64) * K // P)                              # first bit of each step
            n = np.diff(j0)                                                                 # bits per step [T]
            c = np.empty_like(u)
            acc = np.full(u.shape[::2], 0.5)                                                # [E, C]
            for s in range(T):
                c[:, s] = acc
                a = acc + n[s] * u[:, s]
                acc = a - np.floor(a)
            J = int(j0[-1])
            j = np.arange(J, dtype=np.int64)
            sj = np.repeat(np.arange(T), n)                                                 # step of bit j
            i = (j - j0[sj]).astype(np.float64)[None, :, None]
            cs, us = c[:, sj], u[:, sj]                                                     # [E, J, C]
            bits = (np.floor(cs + (i + 1) * us) - np.floor(cs + i * us) > 0.5).astype(np.uint8)
            prev = np.concatenate([np.zeros_like(bits[:, :1]), bits[:, :-1]], 1)
            e, jj, ci = np.nonzero(bits != prev)
            parts_e.append(e)
            parts_t.append(jj * P)
            parts_w.append(sd_wires[ci])
            parts_v.append(bits[e, jj, ci])
        if self.strobe:
            e = np.repeat(np.arange(E, dtype=np.int64), 2 * T)
            s0 = np.tile(np.repeat(np.arange(T, dtype=np.int64) * K, 2), E)
            up = np.tile(np.array([1, 0], np.uint8), E * T)
            parts_e.append(e)
            parts_t.append(s0 + (1 - up.astype(np.int64)) * int(self.strobe_width))
            parts_w.append(np.full(len(e), self.strobe_wire, np.int64))
            parts_v.append(up)
        if parts_e:
            e = np.concatenate(parts_e).astype(np.int64)
            t = np.concatenate(parts_t).astype(np.int64)
            w = np.concatenate(parts_w).astype(np.int32)
            v = np.concatenate(parts_v).astype(np.uint8)
        else:
            e, t, w, v = (np.zeros(0, d) for d in (np.int64, np.int64, np.int32, np.uint8))
        o = np.lexsort((w, t, e))
        off = np.zeros(E + 1, np.int64)
        off[1:] = np.cumsum(np.bincount(e, minlength=E))
        return EventStreams(off, t[o], w[o], v[o])

    def encode_episode(self, obs) -> np.ndarray:
        """Open-loop encoder for whole episodes: obs [..., T, obs_dim] -> uint8 [..., T, K, n_in], the wire value at
        every circuit tick. The sigma-delta carry starts fresh (init_carry) at step 0 of each episode and is carried
        from step to step here, so the task layer owns it; closed-loop students carry it in their policy carry."""
        obs = jnp.asarray(obs, jnp.float32)
        lead = obs.shape[:-2]

        def body(c, o):
            b, c = self.encode_ticks(o, c)
            return c, b

        _, bits = jax.lax.scan(body, self.init_carry(lead), jnp.moveaxis(obs, -2, 0))
        return np.asarray(jnp.moveaxis(bits, 0, -3), np.uint8)

    # decoding
    def decode(self, y) -> jax.Array:
        """y [..., n_out] (bits or duty cycles in [0, 1]) -> action [..., act_dim]. A numpy input is decoded in numpy
        and returns numpy (the search decodes large batches outside jit); anything else goes through jax."""
        xp = np if isinstance(y, np.ndarray) else jnp
        y = xp.asarray(y, xp.float32)
        groups, inv, offsets = self._gout
        parts = []
        for g in groups:
            w = y[..., (offsets[g.dims][:, None] + np.arange(g.bits)[None, :]).ravel()]
            w = xp.clip(w.reshape(*y.shape[:-1], len(g.dims), g.bits), 0.0, 1.0)
            if g.kind == "bit":
                f = (w[..., 0] > 0.5).astype(xp.float32)
            elif g.kind == "duty":
                f = w[..., 0]
            elif g.kind == "pwm":
                f = xp.round(w[..., 0] * self.K) / self.K
            else:
                f = w.mean(-1)
            parts.append(g.lo + f * (g.hi - g.lo))
        order = np.argsort(np.concatenate([g.dims for g in groups]))
        return xp.concatenate(parts, -1)[..., order]

    def decode_ticks(self, bits) -> jax.Array:
        """bits [..., K, n_out] -> action: decode of the per-wire duty cycles."""
        return self.decode(jnp.asarray(bits, jnp.float32).mean(-2))

    def target_wires(self, action) -> jax.Array:
        """Per-wire target duty [..., n_out] that decodes to `action` (popcount: the same expected duty on every
        wire of the group, since the code is permutation invariant). For wire-level dense losses."""
        a = jnp.asarray(action, jnp.float32)
        lo, hi = self.act_lo, self.act_hi
        u = jnp.clip((a - lo) / (hi - lo), 0.0, 1.0)
        reps = np.repeat(np.arange(len(self.outputs)), [c.bits for c in self.outputs])
        kinds = np.repeat(np.array([c.kind == "bit" for c in self.outputs]), [c.bits for c in self.outputs])
        w = u[..., reps]
        return jnp.where(kinds, (w >= 0.5).astype(jnp.float32), w)

    # serialization
    def to_dict(self) -> dict:
        return dict(inputs=[dataclasses.asdict(c) for c in self.inputs],
                    outputs=[dataclasses.asdict(c) for c in self.outputs], K=self.K, dt=self.dt,
                    tick=self.tick, mod_period=self.mod_period, n_in=self.n_in, n_out=self.n_out,
                    strobe=self.strobe, strobe_width=self.strobe_width)

    @classmethod
    def from_dict(cls, d: dict) -> "IOSpec":
        return cls(tuple(InCode(**c) for c in d["inputs"]), tuple(OutCode(**c) for c in d["outputs"]),
                   int(d["K"]), float(d["dt"]), d.get("tick"), int(d.get("mod_period", 1)), bool(d.get("strobe", False)),
                   int(d.get("strobe_width", 0)))


# ---- protocols -------------------------------------------------------------------------------------------------


@runtime_checkable
class Task(Protocol):
    """One control or signal-processing problem with a frozen teacher.

    All methods are pure functions of their arguments (jittable; vmappable over keys / states).
      reset(key) -> state
      step(state, action) -> (state, obs, reward, done)     reward is the task's own per-step reward
      observe(state) -> obs float32 [obs_dim]
      teacher(state) -> action float32 [act_dim]            frozen and deterministic. It reads the full state, so
                                                            it may be privileged (true physics state) and stateful
                                                            (signal specs keep their reference state in `state`).
      io_spec() -> IOSpec
    Attributes: name, uid (stable hash of family + params), family, params (JSON-able, enough for
    Family.from_params), obs_dim, act_dim, dt (seconds per control step), horizon (steps), tags.
    """
    name: str
    uid: str
    family: str
    params: dict
    obs_dim: int
    act_dim: int
    dt: float
    horizon: int

    @property
    def tags(self) -> DemandTags: ...

    def reset(self, key): ...

    def step(self, state, action): ...

    def observe(self, state): ...

    def teacher(self, state): ...

    def io_spec(self) -> IOSpec: ...


class Splits(NamedTuple):
    train: list
    interp: list
    extrap: list


@runtime_checkable
class Family(Protocol):
    """A generator of tasks over a parameter box.

    sample(key, n, split="train") -> list[Task]    task i is a pure function of (key, split, i)
    splits(key, ...) -> Splits(train, interp, extrap)
                                                  train / interp: same box, disjoint draws (no shared uid);
                                                  extrap: outside the box along at least one demand axis
    demand_coverage(tasks) -> dict                 tag distribution summary
    mutate(task, key) -> Task                      (frontier) a nearby task, possibly outside the box
    from_params(params) -> Task                    rebuild a task from its params
    """
    name: str

    def sample(self, key, n: int, split: str = "train") -> list: ...

    def splits(self, key, n_train: int = 64, n_interp: int = 16, n_extrap: int = 16) -> Splits: ...

    def demand_coverage(self, tasks) -> dict: ...


SPLIT_IDS = {"train": 0, "interp": 1, "extrap": 2, "nominal": 3, "mutate": 4}


class FamilyBase:
    """Shared Family machinery. Subclasses implement `sample_one(rng, split) -> Task` (and mutate / from_params)."""
    name = "base"

    def sample_one(self, rng: np.random.Generator, split: str):
        raise NotImplementedError

    def sample(self, key, n: int, split: str = "train") -> list:
        sid = SPLIT_IDS[split]
        return [self.sample_one(np_rng(key, sid, i), split) for i in range(n)]

    def splits(self, key, n_train: int = 64, n_interp: int = 16, n_extrap: int = 16) -> Splits:
        train = self.sample(key, n_train, "train")
        seen = {t.uid for t in train}
        interp, i = [], 0
        while len(interp) < n_interp and i < 50 * n_interp + 100:
            t = self.sample_one(np_rng(key, SPLIT_IDS["interp"], i), "interp")
            i += 1
            if t.uid not in seen:
                seen.add(t.uid)
                interp.append(t)
        extrap = [t for t in self.sample(key, n_extrap, "extrap") if t.uid not in seen]
        return Splits(train, interp, extrap)

    def demand_coverage(self, tasks) -> dict:
        return demand_coverage(tasks)

    def mutate(self, task, key):
        raise NotImplementedError(f"{type(self).__name__} has no mutation operator")

    def from_params(self, params: dict):
        raise NotImplementedError


_LOG_AXES = ("memory_depth", "timing_precision", "n_in", "n_out")


def demand_coverage(tasks, bins: int = 8) -> dict:
    """Summary of the tag distribution: per-axis range, quantiles and histogram (log2 bins for the scale axes),
    counts per family and kind, and the number of occupied cells of a coarse grid over all axes (a coverage
    volume proxy: one cell per log2 octave of the scale axes, quarter of nonlinearity, integer integration and
    multirate)."""
    tags = [t.tags if not isinstance(t, DemandTags) else t for t in tasks]
    out: dict[str, Any] = dict(n=len(tags))
    if not tags:
        return out
    V = np.stack([t.vector() for t in tags])
    out["families"] = _count(t.family for t in tags)
    out["kinds"] = _count(str(t.extra.get("kind", "?")) for t in tags)
    axes = {}
    for j, a in enumerate(AXES):
        v = V[:, j]
        tv = np.log2(np.maximum(v, 1e-6)) if a in _LOG_AXES else v
        lo, hi = float(tv.min()), float(tv.max())
        edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
        cnt, _ = np.histogram(tv, edges)
        axes[a] = dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
                       q05=float(np.quantile(v, 0.05)), q50=float(np.quantile(v, 0.5)),
                       q95=float(np.quantile(v, 0.95)), hist=cnt.tolist(),
                       edges=(2.0 ** edges).tolist() if a in _LOG_AXES else edges.tolist(),
                       log2=a in _LOG_AXES)
    out["axes"] = axes
    out["cells"] = len({cell(t) for t in tags})
    return out


def demand_holdout(tasks, box: dict) -> tuple[list, list]:
    """Partition tasks by a box in demand space: box = {axis: (lo, hi)} over AXES (or extra keys). Returns
    (inside, held_out); a task is held out if any listed tag falls outside its interval. This is how
    generalization claims about unseen demand regions are set up, independently of any family's parameter box."""
    inside, out = [], []
    for t in tasks:
        g = t.tags
        ok = True
        for a, (lo, hi) in box.items():
            v = getattr(g, a) if a in AXES else g.extra.get(a)
            if v is None or not (lo <= float(v) <= hi):
                ok = False
                break
        (inside if ok else out).append(t)
    return inside, out


def cell(tags: DemandTags) -> tuple:
    """Coarse grid cell of a tag vector (see demand_coverage)."""
    v = tags.vector()
    lg = lambda x: int(math.floor(math.log2(max(x, 1e-6))))
    return (lg(v[0]), lg(v[1]), int(round(v[2])), int(min(v[3], 0.999) * 4), lg(v[4]), lg(v[5]), int(v[6]))


def _count(it) -> dict:
    d: dict = {}
    for x in it:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items()))


# ---- rollouts --------------------------------------------------------------------------------------------------


def tree_where(c, new, old):
    """Select new where c (a scalar bool) else old, leafwise."""
    return jax.tree.map(lambda n, o: jnp.where(c, n, o), new, old)


def rollout(task, key, policy=None, carry0=None, beta: float = 0.0, return_states: bool = False,
            horizon: int | None = None) -> dict:
    """One episode, lax.scan over the horizon. Pure and vmappable over `key`.

    policy: None (execute the teacher) or a function (obs, carry) -> (action, carry); carry0 is its initial carry.
    beta:   DAgger mixing, the probability of executing the teacher's action instead of the policy's at each step.
    Returns arrays over time: obs [T, D], act [T, A] (teacher label at the visited state), exec [T, A] (executed),
    reward [T], alive [T] (False after the episode ended; those steps hold the frozen final state and are not
    data), and state (the pre-step state pytree) if return_states.
    """
    T = horizon or task.horizon
    k_reset, k_mix = jax.random.split(as_key(key))
    s0 = task.reset(k_reset)

    def body(c, t):
        s, pc, alive = c
        o = task.observe(s)
        y = task.teacher(s)
        if policy is None:
            a, pc2 = y, pc
        else:
            a, pc2 = policy(o, pc)
            a = jnp.asarray(a, jnp.float32)
            if beta > 0:
                a = jnp.where(jax.random.bernoulli(jax.random.fold_in(k_mix, t), beta), y, a)
        s2, _, r, done = task.step(s, a)
        rec = dict(obs=o, act=y, exec=a, reward=jnp.where(alive, r, 0.0), alive=alive)
        if return_states:
            rec["state"] = s
        return (tree_where(alive, s2, s), pc2, alive & ~done), rec

    _, traj = jax.lax.scan(body, (s0, carry0, jnp.bool_(True)), jnp.arange(T))
    return traj


# ---- teacher datasets ------------------------------------------------------------------------------------------


def task_meta(task) -> dict:
    """JSON sidecar describing a task: enough to rebuild it with Family.from_params and to read its data."""
    io = task.io_spec()
    return jsonable(dict(name=task.name, uid=task.uid, family=task.family, params=task.params,
                         tags=task.tags.to_dict(), io_spec=io.to_dict(), dt=task.dt, horizon=task.horizon,
                         obs_dim=task.obs_dim, act_dim=task.act_dim, act_lo=io.act_lo, act_hi=io.act_hi))


def _keys(keys, n_episodes):
    if isinstance(keys, (int, np.integer)):
        return jax.random.split(jax.random.PRNGKey(int(keys)), n_episodes or 1)
    keys = jnp.asarray(keys) if not hasattr(keys, "dtype") else keys
    if keys.ndim == 1 and not jax.dtypes.issubdtype(keys.dtype, jax.dtypes.prng_key):
        return jax.random.split(keys, n_episodes or 1)       # a single legacy key
    return keys


@dataclass
class TeacherDataset:
    """Teacher-labelled streams, flattened over episodes: obs [N, obs_dim], act [N, act_dim] (teacher action at the
    visited state), episode_id [N], step [N]. Optional: exec_act [N, act_dim] (action executed, = act unless a
    student drove), reward [N]. meta is the task sidecar (task_meta) plus provenance.

    Stored as <path>.npz (arrays) + <path>.json (meta)."""
    obs: np.ndarray
    act: np.ndarray
    episode_id: np.ndarray
    step: np.ndarray
    meta: dict = field(default_factory=dict)
    exec_act: np.ndarray | None = None
    reward: np.ndarray | None = None

    def __len__(self):
        return int(self.obs.shape[0])

    @property
    def n_episodes(self) -> int:
        return int(len(np.unique(self.episode_id))) if len(self) else 0

    @classmethod
    def from_rollouts(cls, task, policy=None, keys=0, n_episodes: int | None = None, carry0=None,
                      beta: float = 0.0, return_states: bool = False, horizon: int | None = None):
        """Roll out `policy` (default: the teacher) on `task` for each key and label every visited state with the
        teacher. keys: int seed (with n_episodes), a single key (split into n_episodes), or a key array [E].
        With a student policy this is one DAgger round (beta mixes in teacher actions). Returns the dataset, or
        (dataset, states) with the visited states (pytree, leading dim N) if return_states."""
        ks = _keys(keys, n_episodes)
        fn = jax.jit(jax.vmap(lambda k: rollout(task, k, policy, carry0, beta, return_states, horizon)))
        traj = fn(ks)
        alive = np.asarray(traj["alive"])                     # [E, T]
        E, T = alive.shape
        m = alive.ravel()
        flat = lambda x: np.asarray(x).reshape(E * T, *np.asarray(x).shape[2:])[m]
        ep = np.broadcast_to(np.arange(E, dtype=np.int32)[:, None], (E, T)).ravel()[m]
        st = np.broadcast_to(np.arange(T, dtype=np.int32)[None, :], (E, T)).ravel()[m]
        meta = task_meta(task)
        meta["provenance"] = dict(policy="teacher" if policy is None else getattr(policy, "__name__", "student"),
                                  beta=beta, episodes=E, keys=np.asarray(jax.random.key_data(ks)
                                  if jax.dtypes.issubdtype(ks.dtype, jax.dtypes.prng_key) else ks).tolist())
        ds = cls(flat(traj["obs"]).astype(np.float32), flat(traj["act"]).astype(np.float32), ep, st, meta,
                 exec_act=flat(traj["exec"]).astype(np.float32), reward=flat(traj["reward"]).astype(np.float32))
        if return_states:
            return ds, jax.tree.map(flat, traj["state"])
        return ds

    @staticmethod
    def relabel(task, states) -> np.ndarray:
        """Teacher actions [N, act_dim] for a batch of states (pytree with leading dim N): DAgger relabelling."""
        fn = _relabel_fn(task)
        return np.asarray(fn(states), np.float32)

    def concat(self, other: "TeacherDataset") -> "TeacherDataset":
        """Append other (episode ids are offset so they stay distinct)."""
        off = int(self.episode_id.max()) + 1 if len(self) else 0
        cat = lambda a, b: None if a is None or b is None else np.concatenate([a, b])
        return TeacherDataset(np.concatenate([self.obs, other.obs]), np.concatenate([self.act, other.act]),
                              np.concatenate([self.episode_id, other.episode_id + off]),
                              np.concatenate([self.step, other.step]), dict(self.meta),
                              cat(self.exec_act, other.exec_act), cat(self.reward, other.reward))

    def save(self, path: str) -> str:
        stem = path[:-4] if path.endswith(".npz") else path
        os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
        arrays = dict(obs=self.obs, act=self.act, episode_id=self.episode_id, step=self.step)
        if self.exec_act is not None:
            arrays["exec_act"] = self.exec_act
        if self.reward is not None:
            arrays["reward"] = self.reward
        np.savez_compressed(stem + ".npz", **arrays)
        with open(stem + ".json", "w") as f:
            json.dump(jsonable(self.meta), f, indent=1)
        return stem

    @classmethod
    def load(cls, path: str) -> "TeacherDataset":
        stem = path[:-4] if path.endswith((".npz", ".json")) else path
        z = np.load(stem + ".npz")
        meta = json.load(open(stem + ".json")) if os.path.exists(stem + ".json") else {}
        return cls(z["obs"], z["act"], z["episode_id"], z["step"], meta,
                   z["exec_act"] if "exec_act" in z else None, z["reward"] if "reward" in z else None)


_RELABEL_CACHE: dict = {}


def _relabel_fn(task):
    k = id(task)
    hit = _RELABEL_CACHE.get(k)
    if hit is None or hit[0] is not task:
        hit = (task, jax.jit(jax.vmap(task.teacher)))
        _RELABEL_CACHE[k] = hit
    return hit[1]


def score(student_actions, dataset: TeacherDataset, metric: str = "abs") -> np.ndarray:
    """Per-step error [N] of student actions [N, act_dim] against the dataset's teacher actions, each action dim
    normalized by its range (act_hi - act_lo from the io spec) and averaged over dims. Leading batch dims are
    allowed: actions [..., N, act_dim] with more dims than the labels give errors [..., N] (one row per student, e.g.
    a batch of circuits); anything else is reshaped to [N, act_dim] as before.

    metric: "abs" |a - y| / range, "sq" squared, "bit" mismatch after thresholding at mid-range (for bit outputs).
    """
    y = dataset.act
    a = np.asarray(student_actions, np.float32)
    a = a.reshape(*a.shape[:-2], *y.shape) if a.ndim > y.ndim else a.reshape(y.shape)
    lo = np.asarray(dataset.meta.get("act_lo", np.full(y.shape[1], -1.0)), np.float32)
    hi = np.asarray(dataset.meta.get("act_hi", np.full(y.shape[1], 1.0)), np.float32)
    rng = np.where(hi > lo, hi - lo, 1.0)
    if metric == "abs":
        e = np.abs(a - y) / rng
    elif metric == "sq":
        e = ((a - y) / rng) ** 2
    elif metric == "bit":
        mid = (lo + hi) / 2
        e = ((a > mid) != (y > mid)).astype(np.float32)
    else:
        raise ValueError(metric)
    return e.mean(-1)


def score_by_episode(errors: np.ndarray, dataset: TeacherDataset) -> np.ndarray:
    """Mean per-step error per episode [E]."""
    ids, inv = np.unique(dataset.episode_id, return_inverse=True)
    return np.bincount(inv, weights=errors) / np.bincount(inv)


def stateless(f: Callable) -> Callable:
    """Wrap a memoryless policy obs -> action into the (obs, carry) -> (action, carry) form rollout expects."""
    def g(o, c):
        return f(o), c
    g.__name__ = getattr(f, "__name__", "policy")
    return g
