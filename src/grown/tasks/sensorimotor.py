"""Substrate-coupled sensorimotor tasks: a wrapper that turns a body task into one wired like a real controller.

Observations become raw sensor channels and actions become PWM duty cycles:
  pos / quat channels  absolute encoders: the value quantized to `enc_bits` (mid-rise), presented as the binary code
                       (InCode "binary", exact)
  vel channels         vel_mode "hidden": not sensed (the circuit must difference encoder readings);
                       "rate": a noiseless pulse train whose density encodes the value (InCode "sigma_delta");
                       "imu": the same with Gaussian noise of std `noise` (fraction of the range)
  every channel        a per-channel delay of 0..max_delay control steps (a ring buffer in the state)
  actions              one PWM wire per actuator; the duty cycle is quantized to the K + 1 levels a K-tick PWM
                       period can express and mapped onto the inner task's action range
  time                 inherited from the inner task: the same dt and TimeModel, so K = round(dt / tick) (sigma-delta
                       channels run at the time model's modulator rate). sensors["K"] is the legacy override (a fixed
                       K, part of the uid as before); new tasks do not set it

The teacher is the inner task's (privileged, on the true state), mapped to duty cycles. Tags inherit the inner
task's, adding delay and noise-averaging to memory_depth, one integration order when velocities are hidden, and a
within-step integration (pulse counting) whenever a sigma-delta channel is present.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import DemandTags, FamilyBase, InCode, IOSpec, OutCode, TimeModel, as_key, jsonable, np_rng, stable_hash

OBS_RANGE = dict(pos=(-3.2, 3.2), quat=(-1.0, 1.0), vel=(-10.0, 10.0))
VEL_MODES = ("hidden", "rate", "imu")
TRAIN = dict(enc_bits=(8, 12), max_delay=(0, 2), noise=(0.0, 0.03))
EXTRAP = dict(enc_bits=(4, 7), max_delay=(3, 6), noise=(0.05, 0.15))


class SMState(NamedTuple):
    inner: object       # inner task state
    buf: jax.Array      # [max_delay + 1, inner obs_dim] past inner observations, newest first
    key: jax.Array      # noise key (noise at step t = f(key, t))
    t: jax.Array


class SensorimotorTask:
    """Task wrapper (see module doc). sensors: dict(enc_bits, vel_mode, noise, delays (per inner obs dim), and
    optionally the legacy K)."""
    family = "sensorimotor"

    def __init__(self, inner, sensors: dict):
        self.inner = inner
        D = inner.obs_dim
        s = dict(sensors)
        s.setdefault("delays", [0] * D)
        if len(s["delays"]) != D:
            raise ValueError("one delay per inner observation dim")
        if s["vel_mode"] not in VEL_MODES:
            raise ValueError(s["vel_mode"])
        if s["vel_mode"] != "imu":
            s["noise"] = 0.0
        self.sensors = jsonable(s)
        self.params = dict(inner=inner.params, inner_family=inner.family, sensors=self.sensors)
        self.uid = stable_hash(dict(family=self.family, inner=inner.uid, sensors=self.sensors))
        self.name = f"sm-{inner.name}-{self.uid[:6]}"
        self.dt, self.horizon, self.act_dim = inner.dt, inner.horizon, inner.act_dim
        if s.get("K") is not None:
            self.time = TimeModel.legacy(int(s["K"]))
        else:
            self.time = getattr(inner, "time", None) or TimeModel()
        self.K = self.time.K(self.dt)
        kinds = getattr(inner, "obs_kinds", ("pos",) * D)
        self.inner_kinds = tuple(kinds)
        self.keep = np.array([i for i, k in enumerate(kinds) if not (k == "vel" and s["vel_mode"] == "hidden")], np.int64)
        self.obs_dim = len(self.keep)
        self.obs_kinds = tuple(kinds[i] for i in self.keep)
        self.max_delay = int(max(s["delays"]))
        io = inner.io_spec()
        self.inner_lo, self.inner_hi = io.act_lo, io.act_hi
        self._c: dict = {}

    def __repr__(self):
        return f"SensorimotorTask({self.name})"

    @property
    def physics_id(self):
        return self.inner.physics_id

    # contract ------------------------------------------------------------------------------------------------
    def io_spec(self) -> IOSpec:
        if "io" not in self._c:
            ins = tuple(InCode("binary", self.sensors["enc_bits"], 0.0, 1.0) if k != "vel" else
                        InCode("sigma_delta", 1, 0.0, 1.0) for k in self.obs_kinds)
            outs = tuple(OutCode("pwm", 1, 0.0, 1.0) for _ in range(self.act_dim))
            tm = self.time
            self._c["io"] = IOSpec(ins, outs, self.K, self.dt, None if tm.is_legacy else tm.tick,
                                   tm.modulator_period(), **tm.io_kw())
        return self._c["io"]

    @property
    def tags(self) -> DemandTags:
        if "tags" not in self._c:
            g = self.inner.tags
            s = self.sensors
            hidden = s["vel_mode"] == "hidden" and "vel" in self.inner_kinds
            sd = any(k == "vel" for k in self.obs_kinds)
            window = min(int(math.ceil((s["noise"] / 0.02) ** 2)), 64) if s["noise"] > 0 else 0
            io = self.io_spec()
            extra = dict(g.extra, kind=f"sm_{g.extra.get('kind', '?')}", in_wires=io.n_in, out_wires=io.n_out,
                         enc_bits=s["enc_bits"], vel_mode=s["vel_mode"], noise=s["noise"], max_delay=self.max_delay,
                         inner_uid=self.inner.uid, **io.strobe_extra())
            self._c["tags"] = DemandTags(
                memory_depth=g.memory_depth + self.max_delay + window + int(hidden),
                timing_precision=g.timing_precision, integration=max(g.integration + int(hidden), float(sd)),
                nonlinearity=g.nonlinearity, n_in=self.obs_dim, n_out=self.act_dim, multirate=g.multirate,
                family=self.family, extra=extra)
        return self._c["tags"]

    def _consts(self):
        if "k" not in self._c:
            with jax.ensure_compile_time_eval():
                lo = np.array([OBS_RANGE[k][0] for k in self.inner_kinds], np.float32)
                hi = np.array([OBS_RANGE[k][1] for k in self.inner_kinds], np.float32)
                vel = np.array([k == "vel" for k in self.inner_kinds])
                self._c["k"] = dict(lo=lo, hi=hi, vel=vel, keep=self.keep, delays=np.asarray(self.sensors["delays"]),
                                    scale=float(2 ** self.sensors["enc_bits"]))
        return self._c["k"]

    def reset(self, key) -> SMState:
        k1, k2 = jax.random.split(as_key(key))
        si = self.inner.reset(k1)
        o = self.inner.observe(si)
        buf = jnp.broadcast_to(o, (self.max_delay + 1, o.shape[0]))
        return SMState(si, buf, k2, jnp.zeros((), jnp.int32))

    def observe(self, s: SMState):
        c = self._consts()
        D = len(self.inner_kinds)
        v = s.buf[jnp.asarray(c["delays"]), jnp.arange(D)]
        n = jax.random.normal(jax.random.fold_in(s.key, s.t), (D,)) * self.sensors["noise"]
        u = jnp.clip((v - c["lo"]) / (c["hi"] - c["lo"]) + jnp.where(c["vel"], n, 0.0), 0.0, 1.0)
        q = jnp.minimum(jnp.floor(u * c["scale"]), c["scale"] - 1) / c["scale"] + 0.5 / c["scale"]
        return jnp.where(c["vel"], u, q)[jnp.asarray(c["keep"])]

    def _to_inner(self, duty):
        K = self.K
        d = jnp.round(jnp.clip(jnp.asarray(duty, jnp.float32), 0.0, 1.0) * K) / K
        return self.inner_lo + d * (self.inner_hi - self.inner_lo)

    def step(self, s: SMState, duty):
        si, _, r, done = self.inner.step(s.inner, self._to_inner(duty))
        buf = jnp.concatenate([self.inner.observe(si)[None], s.buf[:-1]])
        s1 = SMState(si, buf, s.key, s.t + 1)
        return s1, self.observe(s1), r, done

    def teacher(self, s: SMState):
        a = self.inner.teacher(s.inner)
        return jnp.clip((a - self.inner_lo) / (self.inner_hi - self.inner_lo), 0.0, 1.0)

    @property
    def has_teacher(self):
        return getattr(self.inner, "has_teacher", True)

    def attach_teacher(self, policy):
        self.inner.attach_teacher(policy)
        return self


def draw_sensors(rng, D, split="train") -> dict:
    ext = set()
    if split == "extrap":
        ext = set(rng.choice(list(EXTRAP), size=1 + int(rng.random() < 0.3), replace=False).tolist())
    box = lambda a: EXTRAP[a] if a in ext else TRAIN[a]
    mode = "imu" if "noise" in ext else VEL_MODES[int(rng.integers(3))]
    dmax = int(rng.integers(box("max_delay")[0], box("max_delay")[1] + 1))
    delays = rng.integers(0, dmax + 1, D)
    if dmax > 0:
        delays[int(rng.integers(D))] = dmax         # max_delay is attained, so the split box is meaningful
    return dict(enc_bits=int(rng.integers(box("enc_bits")[0], box("enc_bits")[1] + 1)), vel_mode=mode,
                noise=float(rng.uniform(*box("noise"))), delays=delays.tolist(), extrap_axes=sorted(ext))


class SensorimotorFamily(FamilyBase):
    """Body tasks from `inner` (default: generated bodies with visible velocities) wrapped with random sensors.
    extrap draws the body from its extrap split and pushes one or two sensor axes (encoder bits, delay, noise) out of
    the train box."""
    name = "sensorimotor"

    def __init__(self, inner=None, K: int | None = None, time=None):
        """K: legacy fixed K (stored in every task's sensors); otherwise tasks inherit the inner family's time
        model (time= builds the default inner family with it)."""
        if inner is None:
            from .bodies import BodyFamily
            inner = BodyFamily(p_hide_vel=0.0, time=time)
        self.inner, self.K = inner, K

    def sample_one(self, rng, split):
        t = self.inner.sample_one(rng, split)
        s = draw_sensors(rng, t.obs_dim, split)
        if self.K is not None:
            s["K"] = int(self.K)
        return SensorimotorTask(t, s)

    def from_params(self, params):
        from . import get_family
        inner = get_family(params["inner_family"]).from_params(params["inner"])
        return SensorimotorTask(inner, params["sensors"])

    def mutate(self, task, key):
        rng = np_rng(key, 4)
        s = dict(task.sensors)
        r = rng.random()
        if r < 0.5 and hasattr(self.inner, "mutate"):
            inner = self.inner.mutate(task.inner, key)
            if inner.obs_dim != task.inner.obs_dim:
                s["delays"] = rng.integers(0, max(s["delays"]) + 1, inner.obs_dim).tolist()
            return SensorimotorTask(inner, s)
        if r < 0.7:
            s["enc_bits"] = int(np.clip(s["enc_bits"] + rng.choice([-1, 1]), 2, 16))
        elif r < 0.85:
            d = list(s["delays"])
            i = int(rng.integers(len(d)))
            d[i] = int(np.clip(d[i] + rng.choice([-1, 1]), 0, 16))
            s["delays"] = d
        else:
            s["vel_mode"] = "imu"
            s["noise"] = float(np.clip(s.get("noise", 0.0) * math.exp(rng.normal(0, 0.3)) + 0.005, 0.0, 0.5))
        return SensorimotorTask(task.inner, s)
