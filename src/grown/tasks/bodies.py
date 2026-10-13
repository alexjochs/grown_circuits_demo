"""Generated bodies: procedurally built articulated linkages in MuJoCo MJX, with scrambled interfaces.

A body is a torso plus a random tree of 2-8 capsule links, planar (hinges about y, root slides in x-z) or 3D (free
root, one or two hinges per link about random axes perpendicular to the link), or welded to the world (no floor, no
contact: arms that swing up against gravity). Masses, lengths, joint ranges, damping, actuator strength and sign,
friction, gravity and the control step are drawn per body; the MJCF is built as a string from these parameters.

Objectives: free bodies move forward (x velocity of the root minus a control cost); welded bodies raise their tips
(mean tip height above the torso over the reach). Named points (Hopper, Humanoid, the gymnasium locomotion bodies, DM
Control and MuJoCo Menagerie bodies when those are installed) go through the same BodyTask with their own XML.

Interface scrambling (as evocirc.families' hopper_scr): interface obs dim i reads physical obs dim obs_perm[i];
interface action i drives physical actuator act_perm[i] with signed gain act_mul[i]. The nominal interface is the
identity. Velocities can be hidden (vel_vis = 0: zeroed), so the circuit must infer them from positions.

Teachers are privileged: a linear ARS policy on the physical, full-velocity observation of the same body, trained
once per physics hash and cached on disk (teachers.py). The scrambled task maps it through the interface.

Tags are estimated from the model with CPU MuJoCo (cheap, no MJX): timescales from the mass matrix, actuator gear,
damping and gravity; nonlinearity as 1 - R^2 of a linear one-step model under random actions; contact presence
from the same rollout. See DemandTags for definitions.

mujoco is imported lazily so `grown.tasks` works without it for signal tasks.
"""
from __future__ import annotations

import functools
import importlib.util
import math
import os
from typing import NamedTuple

import numpy as np

from .base import DemandTags, FamilyBase, InCode, IOSpec, OutCode, jsonable, np_rng, stable_hash, time_model

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
TIMESTEP = 0.005
OBS_RANGE = dict(pos=(-3.2, 3.2), quat=(-1.0, 1.0), vel=(-10.0, 10.0))


def _mj():
    import mujoco
    return mujoco


def _mjx():
    from mujoco import mjx
    return mjx


# ---- parameter boxes -------------------------------------------------------------------------------------------

TRAIN = dict(n_links=(2, 6), strength=(0.7, 2.0), friction=(0.4, 1.2), gravity=(7.0, 12.0))
EXTRAP = dict(n_links=((7, 8),), strength=((0.3, 0.7), (2.0, 4.0)), friction=((0.1, 0.4), (1.2, 2.0)),
              gravity=((3.0, 7.0), (12.0, 18.0)))
GAIN_TRAIN, GAIN_EXTRAP = (0.8, 1.2), ((0.5, 0.8), (1.2, 1.5))


def _side(rng, sides):
    lo, hi = sides[int(rng.integers(len(sides)))]
    return float(rng.uniform(lo, hi))


def _unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def _perp(rng, u):
    v = rng.normal(size=3)
    return _unit(v - v.dot(u) * u)


def draw_body(rng, split="train") -> dict:
    """Physical body parameters (JSON-able). extrap: one or two of n_links, strength, friction, gravity are drawn
    outside the train box (recorded in body["extrap_axes"])."""
    ext = []
    if split == "extrap":
        axes = list(EXTRAP)
        ext = sorted(rng.choice(axes, size=1 + int(rng.random() < 0.3), replace=False).tolist())
    val = {}
    for a in ("strength", "friction", "gravity"):
        val[a] = _side(rng, EXTRAP[a]) if a in ext else float(rng.uniform(*TRAIN[a]))
    lo, hi = EXTRAP["n_links"][0] if "n_links" in ext else TRAIN["n_links"]
    n_links = int(rng.integers(lo, hi + 1))
    r = rng.random()
    root, dim = ("free", 2) if r < 0.4 else ("free", 3) if r < 0.7 else ("fixed", int(rng.integers(2, 4)))
    tdir = [1.0, 0.0, 0.0] if dim == 2 else _unit([1.0, rng.normal(0, 0.3), 0.0]).tolist()
    torso = dict(length=float(rng.uniform(0.2, 0.6)), radius=float(rng.uniform(0.04, 0.08)),
                 mass=float(math.exp(rng.uniform(math.log(1.0), math.log(8.0)))), dir=tdir)
    links = []
    for i in range(n_links):
        # parent: the torso, or a random earlier link (biased to the previous one: chains)
        parent = -1 if i == 0 or rng.random() < 0.3 else (i - 1 if rng.random() < 0.6 else int(rng.integers(0, i)))
        if root == "fixed":
            th = rng.uniform(0, math.pi)             # anywhere in the lower half: arms hang
        else:
            th = rng.uniform(0, 0.6 * math.pi)       # mostly downward: legs
        if dim == 2:
            d = [math.sin(th) * rng.choice([-1, 1]), 0.0, -math.cos(th)]
            axes = [[0.0, 1.0, 0.0]]
        else:
            ph = rng.uniform(0, 2 * math.pi)
            d = [math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), -math.cos(th)]
            u = _unit(d)
            a1 = _perp(rng, u)
            axes = [a1.tolist()] + ([_unit(np.cross(u, a1)).tolist()] if rng.random() < 0.3 else [])
        links.append(dict(parent=int(parent), attach=float(rng.uniform(-1, 1)), length=float(rng.uniform(0.15, 0.5)),
                          radius=float(rng.uniform(0.025, 0.05)),
                          mass=float(math.exp(rng.uniform(math.log(0.3), math.log(3.0)))),
                          dir=[float(x) for x in _unit(d)], axes=axes,
                          range=[-float(rng.uniform(math.pi / 8, math.pi / 2)), float(rng.uniform(math.pi / 8, math.pi / 2))],
                          damping=float(rng.uniform(0.02, 0.2)), sign=int(rng.choice([-1, 1]))))
    return dict(root=root, dim=dim, torso=torso, links=links, frame_skip=int(rng.integers(2, 7)), extrap_axes=ext,
                **val)


def _subtree(links):
    """Per link: (mass of its subtree, longest path length from its joint to a leaf tip)."""
    n = len(links)
    kids = [[j for j in range(n) if links[j]["parent"] == i] for i in range(n)]

    @functools.lru_cache(None)
    def f(i):
        m, L = links[i]["mass"], links[i]["length"]
        best = 0.0
        for k in kids[i]:
            mk, Lk = f(k)
            m += mk
            best = max(best, Lk)
        return m, L + best

    return [f(i) for i in range(n)], kids


def _points(body):
    """World positions at qpos0 (torso centre at the origin): torso ends, link joints and link tips."""
    t = body["torso"]
    td = np.asarray(t["dir"])
    base, tip = [], []
    for i, l in enumerate(body["links"]):
        p = (l["attach"] * t["length"] / 2) * td if l["parent"] < 0 else tip[l["parent"]]
        base.append(p)
        tip.append(p + l["length"] * np.asarray(l["dir"]))
    return td * t["length"] / 2, base, tip


def build_mjcf(body: dict) -> str:
    """MJCF string for a generated body (see draw_body)."""
    t, links = body["torso"], body["links"]
    sub, kids = _subtree(links)
    n_root = max(sum(1 for l in links if l["parent"] < 0), 1)
    tend, base, tip = _points(body)
    fixed = body["root"] == "fixed"
    zmin = min([-abs(tend[2]) - t["radius"]] + [min(b[2], e[2]) - l["radius"] for b, e, l in zip(base, tip, links)])
    reach = max(np.linalg.norm(e) for e in tip)
    z0 = (reach + 0.5) if fixed else (-zmin + 0.02)
    f3 = lambda v: " ".join(f"{x:.6g}" for x in v)

    def gear(i):
        return body["strength"] * 9.81 * sub[i][1] * (sub[i][0] + (0.0 if fixed else t["mass"] / n_root))

    def link_xml(i, depth):
        l = links[i]
        # armature (reflected rotor inertia) grows with the actuator: full torque from rest moves a joint at most
        # ~0.02 rad per physics step, which keeps explicit Euler stable for light links on strong motors
        arm = max(0.01, gear(i) * TIMESTEP ** 2 / 0.02)
        pos = (l["attach"] * t["length"] / 2) * np.asarray(t["dir"]) if l["parent"] < 0 else \
            links[l["parent"]]["length"] * np.asarray(links[l["parent"]]["dir"])
        end = l["length"] * np.asarray(l["dir"])
        ind = "  " * depth
        s = [f'{ind}<body name="l{i}" pos="{f3(pos)}">']
        for j, ax in enumerate(l["axes"]):
            s.append(f'{ind}  <joint name="j{i}_{j}" type="hinge" axis="{f3(ax)}" range="{f3(l["range"])}" '
                     f'damping="{l["damping"] * sub[i][0]:.6g}" armature="{arm:.6g}"/>')
        s.append(f'{ind}  <geom name="g{i}" type="capsule" fromto="0 0 0 {f3(end)}" size="{l["radius"]:.6g}" '
                 f'mass="{l["mass"]:.6g}"/>')
        if not kids[i]:
            s.append(f'{ind}  <site name="tip{i}" pos="{f3(end)}" size="0.01"/>')
        for k in kids[i]:
            s.extend(link_xml(k, depth + 1))
        s.append(f"{ind}</body>")
        return s

    if fixed:
        root_j = []
    elif body["dim"] == 2:
        root_j = ['      <joint name="rootx" type="slide" axis="1 0 0" limited="false" damping="0"/>',
                  '      <joint name="rootz" type="slide" axis="0 0 1" limited="false" damping="0"/>',
                  '      <joint name="rooty" type="hinge" axis="0 1 0" limited="false" damping="0"/>']
    else:
        root_j = ['      <freejoint name="root"/>']
    acts = []
    for i, l in enumerate(links):
        for j in range(len(l["axes"])):
            acts.append(f'    <motor name="m{i}_{j}" joint="j{i}_{j}" gear="{l["sign"] * gear(i):.6g}" '
                        f'ctrllimited="true" ctrlrange="-1 1"/>')
    floor = [] if fixed else ['    <geom name="floor" type="plane" size="0 0 0.1" contype="1" conaffinity="0" '
                              'rgba="0.8 0.9 0.8 1"/>']
    xml = [f'<mujoco model="gen">',
           '  <compiler angle="radian" inertiafromgeom="true"/>',
           f'  <option timestep="{TIMESTEP}" gravity="0 0 {-body["gravity"]:.6g}" integrator="Euler"/>',
           '  <default>',
           '    <joint limited="true" armature="0.01"/>',
           f'    <geom contype="0" conaffinity="1" condim="3" friction="{body["friction"]:.6g} 0.1 0.1" '
           'rgba="0.7 0.5 0.3 1"/>',
           '  </default>',
           '  <worldbody>',
           *floor,
           f'    <body name="torso" pos="0 0 {z0:.6g}">',
           *root_j,
           f'      <geom name="torso" type="capsule" fromto="{f3(-tend)} {f3(tend)}" size="{t["radius"]:.6g}" '
           f'mass="{t["mass"]:.6g}"/>',
           *[s for i, l in enumerate(links) if l["parent"] < 0 for s in link_xml(i, 3)],
           '    </body>',
           '  </worldbody>',
           '  <actuator>', *acts, '  </actuator>',
           '</mujoco>']
    return "\n".join(xml)


# ---- named points ----------------------------------------------------------------------------------------------


def _gym_assets():
    spec = importlib.util.find_spec("gymnasium")
    if spec is None or not spec.submodule_search_locations:
        return None
    d = os.path.join(list(spec.submodule_search_locations)[0], "envs", "mujoco", "assets")
    return d if os.path.isdir(d) else None


def _dmc(domain, *args):
    def load():
        mod = __import__(f"dm_control.suite.{domain}", fromlist=["get_model_and_assets"])
        xml, assets = mod.get_model_and_assets(*args)
        return xml, assets
    return load


NAMED = {   # name -> source (path | callable returning (xml, assets)), objective, frame_skip, horizon
    "hopper": dict(path=os.path.join(ASSETS, "hopper.xml"), objective="gym_hopper", frame_skip=4, horizon=1000),
    "humanoid": dict(path=os.path.join(ASSETS, "humanoid.xml"), objective="gym_humanoid", frame_skip=5, horizon=1000),
}
for _n, _fs in (("walker2d", 4), ("half_cheetah", 5), ("swimmer", 4), ("ant", 5)):
    NAMED[_n] = dict(gym=f"{_n}.xml", objective="forward", frame_skip=_fs, horizon=1000)
for _n, _args in (("cheetah", ()), ("walker", ()), ("hopper", ()), ("swimmer", (6,)), ("humanoid", ())):
    NAMED[f"dmc_{_n}"] = dict(dmc=(_n, _args), objective="forward", frame_skip=4, horizon=1000)
for _n, _rel in (("go2", "unitree_go2/scene_mjx.xml"), ("h1", "unitree_h1/scene.xml"),
                 ("op3", "robotis_op3/scene.xml")):
    NAMED[f"menagerie_{_n}"] = dict(menagerie=_rel, objective="forward", frame_skip=4, horizon=1000)


def _named_source(name):
    e = NAMED[name]
    if "path" in e:
        return e["path"] if os.path.exists(e["path"]) else None
    if "gym" in e:
        d = _gym_assets()
        p = os.path.join(d, e["gym"]) if d else None
        return p if p and os.path.exists(p) else None
    if "dmc" in e:
        return _dmc(*e["dmc"]) if importlib.util.find_spec("dm_control") else None
    root = os.environ.get("MUJOCO_MENAGERIE_PATH")
    p = os.path.join(root, e["menagerie"]) if root else None
    return p if p and os.path.exists(p) else None


def available_named() -> list[str]:
    """Named bodies whose model source is present on this machine."""
    return [n for n in NAMED if _named_source(n) is not None]


def _load_model(params):
    mj = _mj()
    if params.get("body") is not None:
        mm = mj.MjModel.from_xml_string(build_mjcf(params["body"]))
    else:
        src = _named_source(params["named"])
        if src is None:
            raise FileNotFoundError(f"named body {params['named']!r} is not available here")
        if callable(src):
            xml, assets = src()
            mm = mj.MjModel.from_xml_string(xml, assets)
        else:
            mm = mj.MjModel.from_xml_path(src)
        if mm.opt.solver == mj.mjtSolver.mjSOL_PGS:          # MJX implements Newton and CG only
            mm.opt.solver = mj.mjtSolver.mjSOL_NEWTON
    return mm


class ModelInfo(NamedTuple):
    root: str               # "free3d" | "planar" | "fixed"
    obs_qpos: np.ndarray    # qpos indices observed (root x / y translation dropped)
    obs_kinds: tuple        # per physical obs dim: "pos" | "quat" | "vel"
    root_x: int             # qpos index of the root x coordinate (-1 if fixed)
    qpos_noise: np.ndarray  # 1 where reset noise is added to qpos (not quaternions)
    tips: np.ndarray        # site ids of link tips (generated bodies)
    ctrl_lo: np.ndarray
    ctrl_hi: np.ndarray
    floor: bool


def model_info(mm) -> ModelInfo:
    mj = _mj()
    nq = mm.nq
    drop, quat, root_x = set(), set(), -1
    root = "fixed"
    for j in range(mm.njnt):
        b = mm.jnt_bodyid[j]
        if mm.body_parentid[b] != 0:
            continue
        adr, typ = mm.jnt_qposadr[j], mm.jnt_type[j]
        if typ == mj.mjtJoint.mjJNT_FREE:
            root, root_x = "free3d", adr
            drop |= {adr, adr + 1}
            quat |= {adr + 3, adr + 4, adr + 5, adr + 6}
        elif typ == mj.mjtJoint.mjJNT_SLIDE:
            root = "planar"
            ax = np.abs(mm.jnt_axis[j])
            if ax[0] > 0.99:
                drop.add(adr)
                root_x = adr
            elif ax[1] > 0.99:
                drop.add(adr)
        elif typ == mj.mjtJoint.mjJNT_BALL:
            quat |= {adr, adr + 1, adr + 2, adr + 3}
    obs_q = np.array([i for i in range(nq) if i not in drop], np.int64)
    kinds = tuple("quat" if i in quat else "pos" for i in obs_q) + ("vel",) * mm.nv
    noise = np.array([0.0 if i in quat else 1.0 for i in range(nq)], np.float32)
    tips = np.array([s for s in range(mm.nsite) if (mj.mj_id2name(mm, mj.mjtObj.mjOBJ_SITE, s) or "").startswith("tip")],
                    np.int64)
    lim = mm.actuator_ctrllimited.astype(bool)
    lo = np.where(lim, mm.actuator_ctrlrange[:, 0], -1.0).astype(np.float32)
    hi = np.where(lim, mm.actuator_ctrlrange[:, 1], 1.0).astype(np.float32)
    floor = bool(np.any(mm.geom_type == mj.mjtGeom.mjGEOM_PLANE))
    return ModelInfo(root, obs_q, kinds, int(root_x), noise, tips, lo, hi, floor)


# ---- tag estimation (CPU MuJoCo) -------------------------------------------------------------------------------


def _clusters(taus, factor=3.0) -> int:
    t = np.sort(np.log(np.asarray([x for x in taus if np.isfinite(x) and x > 0])))
    return int(1 + np.sum(np.diff(t) > math.log(factor))) if len(t) else 1


def estimate_physics(mm, frame_skip, seed=0, steps=300) -> dict:
    """Model-derived demand quantities, independent of the interface."""
    mj = _mj()
    info = model_info(mm)
    d = mj.MjData(mm)
    mj.mj_forward(mm, d)
    M = np.zeros((mm.nv, mm.nv))
    mj.mj_fullM(mm, d, M)
    dof = mm.jnt_dofadr[mm.actuator_trnid[:, 0]]
    Mjj = np.maximum(M[dof, dof], 1e-9)
    gear = np.abs(mm.actuator_gear[:, 0]) * np.maximum(np.abs(info.ctrl_lo), np.abs(info.ctrl_hi))
    tau_act = np.sqrt(2 * Mjj / np.maximum(gear, 1e-9))
    g = float(abs(mm.opt.gravity[2]))
    tau_grav = math.sqrt(mm.stat.extent / g) if g > 0 else math.inf
    damp = mm.dof_damping[dof]
    tau_damp = np.where(damp > 0, Mjj / np.maximum(damp, 1e-12), np.inf)
    dt = mm.opt.timestep * frame_skip
    fast = min(float(tau_act.min()) if len(tau_act) else math.inf, tau_grav)
    # random-action rollout: contact presence and linear predictability of the dynamics
    rng = np.random.default_rng(seed)
    X, Y, ncon = [], [], 0
    mj.mj_resetData(mm, d)
    mj.mj_forward(mm, d)
    a = np.zeros(mm.nu)
    lo, hi = info.ctrl_lo, info.ctrl_hi
    for t in range(steps):
        if t % 4 == 0:
            a = lo + (hi - lo) * rng.random(mm.nu)
        s = np.concatenate([d.qpos[info.obs_qpos], d.qvel])
        d.ctrl[:] = a
        for _ in range(frame_skip):
            mj.mj_step(mm, d)
            ncon = max(ncon, d.ncon)
        s1 = np.concatenate([d.qpos[info.obs_qpos], d.qvel])
        if not np.all(np.isfinite(s1)) or np.abs(s1).max() > 1e4:
            mj.mj_resetData(mm, d)
            mj.mj_forward(mm, d)
            continue
        X.append(np.concatenate([s, (a - lo) / (hi - lo) * 2 - 1, [1.0]]))
        Y.append(s1 - s)
    X, Y = np.asarray(X), np.asarray(Y)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ coef
    var = Y.var(0)
    ok = var > 1e-10
    r2 = 1 - res.var(0)[ok] / var[ok] if ok.any() else np.ones(1)
    return dict(dof=int(mm.nv), nu=int(mm.nu), dt=dt, contact=bool(ncon > 0), root=info.root,
                tau_act_min=float(tau_act.min()) if len(tau_act) else math.inf, tau_grav=tau_grav,
                timing_precision=dt / fast if math.isfinite(fast) else 1.0,
                multirate=_clusters(list(tau_act) + [tau_grav] + list(tau_damp)),
                nonlinearity=float(np.clip(1 - np.mean(r2), 0.0, 1.0)))


_PHYS_CACHE: dict = {}

# ---- the task --------------------------------------------------------------------------------------------------


class BodyState(NamedTuple):
    data: object           # mjx.Data
    t: object              # int32 step


def identity_iface(obs_dim, act_dim) -> dict:
    return dict(obs_perm=list(range(obs_dim)), act_perm=list(range(act_dim)), act_mul=[1.0] * act_dim)


def random_iface(rng, obs_dim, act_dim, split="train") -> dict:
    gain = rng.uniform(*GAIN_TRAIN, act_dim) if split != "extrap" else np.array(
        [_side(rng, GAIN_EXTRAP) for _ in range(act_dim)])
    sign = rng.choice([-1.0, 1.0], act_dim)
    return dict(obs_perm=rng.permutation(obs_dim).tolist(), act_perm=rng.permutation(act_dim).tolist(),
                act_mul=(sign * gain).round(6).tolist())


class BodyTask:
    """A Task (see base.Task) on a MuJoCo body, stepped in MJX.

    params: body (generated body dict) or named (NAMED key); objective; frame_skip; horizon; vel_vis (0/1);
    iface (obs_perm, act_perm, act_mul); ctrl_w (control cost weight).
    State: BodyState(mjx.Data, t). Actions: [-1, 1]^act_dim in interface order. Obs: interface order.
    """

    def __init__(self, params: dict, family: str | None = None, K: int | None = None, obs_bits: int = 8,
                 impl: str = "jax", time=None):
        """time: the TimeModel (default physical); K is derived from the MJX control dt (timestep * frame_skip,
        typically 1-20 ms: K = 10^8 to 2 * 10^9 at 10 ps). K= is the legacy shorthand (TimeModel.legacy(K))."""
        self.params = jsonable(params)
        self.family = family or ("bodies" if params.get("body") is not None else "named_bodies")
        self.horizon = int(params["horizon"])
        self.frame_skip = int(params["frame_skip"])
        self.time = time_model(time, K)
        self.obs_bits, self.impl = obs_bits, impl
        base = dict(family=self.family, params=self.params)
        self.uid = stable_hash(base if self.time.is_legacy else dict(base, time=self.time.key()))
        nm = params.get("named") or "gen"
        self.name = f"{nm}-{self.uid[:8]}"
        self._c: dict = {}
        self._teacher = None

    def __repr__(self):
        return f"BodyTask({self.name})"

    # model -----------------------------------------------------------------------------------------------------
    @property
    def mj_model(self):
        if "mm" not in self._c:
            self._c["mm"] = _load_model(self.params)
        return self._c["mm"]

    @property
    def xml(self) -> str | None:
        return build_mjcf(self.params["body"]) if self.params.get("body") is not None else None

    @property
    def info(self) -> ModelInfo:
        if "info" not in self._c:
            self._c["info"] = model_info(self.mj_model)
        return self._c["info"]

    @property
    def physics_id(self) -> str:
        """Hash of everything physical (not the interface): keys the teacher cache."""
        p = self.params
        src = dict(body=p["body"]) if p.get("body") is not None else dict(named=p["named"])
        return stable_hash(dict(src=src, objective=p["objective"], frame_skip=self.frame_skip, horizon=self.horizon,
                                ctrl_w=p.get("ctrl_w", 1e-2)))

    @property
    def obs_dim(self) -> int:
        return len(self.info.obs_qpos) + self.mj_model.nv

    @property
    def act_dim(self) -> int:
        return int(self.mj_model.nu)

    @property
    def dt(self) -> float:
        return float(self.mj_model.opt.timestep * self.frame_skip)

    @property
    def K(self) -> int:
        """Substrate ticks per control step, derived from dt and the time model."""
        return self.time.K(self.dt)

    @property
    def obs_kinds(self) -> tuple:
        """Kind of each interface obs dim ("pos", "quat", "vel")."""
        k = self.info.obs_kinds
        return tuple(k[i] for i in self.params["iface"]["obs_perm"])

    def physical(self) -> "BodyTask":
        """The same body with the identity interface and visible velocities (what the teacher is trained on)."""
        p = dict(self.params, iface=identity_iface(self.obs_dim, self.act_dim), vel_vis=1)
        t = BodyTask(p, self.family, obs_bits=self.obs_bits, impl=self.impl, time=self.time)
        t._c["mm"], t._c["info"] = self.mj_model, self.info
        t._teacher = self._teacher
        return t

    # tags / io -------------------------------------------------------------------------------------------------
    @property
    def tags(self) -> DemandTags:
        if "tags" not in self._c:
            pid = self.physics_id
            if pid not in _PHYS_CACHE:
                _PHYS_CACHE[pid] = estimate_physics(self.mj_model, self.frame_skip)
            ph = _PHYS_CACHE[pid]
            vis = int(self.params.get("vel_vis", 1))
            body = self.params.get("body")
            io = self.io_spec()
            kind = (f"gen_{ph['root']}" if body is not None else f"named_{self.params['named']}")
            extra = dict(kind=kind, in_wires=io.n_in, out_wires=io.n_out, contact=ph["contact"], dof=ph["dof"],
                         tau_act_min=ph["tau_act_min"], tau_grav=ph["tau_grav"], dt=ph["dt"], vel_vis=vis,
                         n_links=len(body["links"]) if body else None,
                         extrap_axes=body.get("extrap_axes", []) if body else [], **io.strobe_extra())
            self._c["tags"] = DemandTags(memory_depth=1.0 + (1 - vis), timing_precision=ph["timing_precision"],
                                         integration=float(1 - vis), nonlinearity=ph["nonlinearity"],
                                         n_in=self.obs_dim, n_out=self.act_dim, multirate=ph["multirate"],
                                         family=self.family, extra=extra)
        return self._c["tags"]

    def io_spec(self) -> IOSpec:
        if "io" not in self._c:
            ins = tuple(InCode("binary", self.obs_bits, *OBS_RANGE[k]) for k in self.obs_kinds)
            outs = tuple(OutCode("duty", 1, -1.0, 1.0) for _ in range(self.act_dim))
            tm = self.time
            self._c["io"] = IOSpec(ins, outs, self.K, self.dt, None if tm.is_legacy else tm.tick,
                                   tm.modulator_period(), **tm.io_kw())
        return self._c["io"]

    # dynamics --------------------------------------------------------------------------------------------------
    def _fns(self):
        if "fns" in self._c:
            return self._c["fns"]
        import jax
        import jax.numpy as jnp
        mjx = _mjx()
        mm, info, p = self.mj_model, self.info, self.params
        with jax.ensure_compile_time_eval():     # constants stay concrete even if first built inside a jit
            m = mjx.put_model(mm, impl=self.impl)
            obs_q = jnp.asarray(info.obs_qpos)
            n_q = len(info.obs_qpos)
            vis = jnp.concatenate([jnp.ones(n_q), jnp.full(mm.nv, float(p.get("vel_vis", 1)))]).astype(jnp.float32)
            operm = jnp.asarray(p["iface"]["obs_perm"], jnp.int32)
            aperm = jnp.asarray(p["iface"]["act_perm"], jnp.int32)
            amul = jnp.asarray(p["iface"]["act_mul"], jnp.float32)
            lo, hi = jnp.asarray(info.ctrl_lo), jnp.asarray(info.ctrl_hi)
            qpos0, noise_mask = jnp.asarray(mm.qpos0, jnp.float32), jnp.asarray(info.qpos_noise)
            noise = 5e-3 if p["objective"] == "gym_hopper" else 1e-2
            fs, dt, rx, ctrl_w = self.frame_skip, self.dt, info.root_x, float(p.get("ctrl_w", 1e-2))
            tips = jnp.asarray(info.tips)
        reach = float(max(mm.stat.extent, 1e-3))
        obj, H = p["objective"], self.horizon

        def phys_obs(d):
            return jnp.concatenate([d.qpos[obs_q], jnp.clip(d.qvel, -10.0, 10.0)]).astype(jnp.float32)

        def reset(key):
            k1, k2 = jax.random.split(key)
            d = mjx.make_data(mm, impl=self.impl)
            qpos = qpos0 + noise_mask * jax.random.uniform(k1, (mm.nq,), minval=-noise, maxval=noise)
            qvel = jax.random.uniform(k2, (mm.nv,), minval=-noise, maxval=noise)
            return BodyState(mjx.forward(m, d.replace(qpos=qpos, qvel=qvel)), jnp.zeros((), jnp.int32))

        def observe(s):
            return (phys_obs(s.data) * vis)[operm]

        def to_ctrl(c):                          # physical normalized [-1, 1] -> ctrl range
            return lo + (jnp.clip(c, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo)

        def step(s, action):
            a = jnp.clip(jnp.asarray(action, jnp.float32), -1.0, 1.0)
            c = jnp.zeros(mm.nu, jnp.float32).at[aperm].set(jnp.clip(amul * a, -1.0, 1.0))
            d0 = s.data
            d = jax.lax.fori_loop(0, fs, lambda _, d: mjx.step(m, d), d0.replace(ctrl=to_ctrl(c)))
            xv = (d.qpos[rx] - d0.qpos[rx]) / dt if rx >= 0 else 0.0
            cost = jnp.sum(c ** 2)
            if obj == "forward":
                r, bad = xv - ctrl_w * cost, False
            elif obj == "height":
                z = d.site_xpos[tips, 2].mean() if len(info.tips) else 0.0
                r, bad = (z - d.xpos[1, 2]) / reach - ctrl_w * cost, False
            elif obj == "gym_hopper":
                st = jnp.concatenate([d.qpos, d.qvel])[2:]
                healthy = jnp.all(jnp.abs(st) < 100.0) & (d.qpos[1] > 0.7) & (jnp.abs(d.qpos[2]) < 0.2)
                r, bad = xv + healthy - 1e-3 * jnp.sum(d.ctrl ** 2), ~healthy
            elif obj == "gym_humanoid":
                healthy = (d.qpos[2] > 1.0) & (d.qpos[2] < 2.0)
                r, bad = 1.25 * xv + 5.0 * healthy - 0.1 * jnp.sum(d.ctrl ** 2), ~healthy
            else:
                raise ValueError(obj)
            finite = jnp.all(jnp.isfinite(d.qpos)) & jnp.all(jnp.isfinite(d.qvel))
            s1 = BodyState(d, s.t + 1)
            done = bad | ~finite | (s1.t >= H)
            return s1, observe(s1), jnp.where(finite, r, 0.0).astype(jnp.float32), done

        def teacher_fn(policy):
            def f(s):
                c = policy(phys_obs(s.data))
                return jnp.clip(c[aperm] / amul, -1.0, 1.0)
            return f

        self._c["fns"] = dict(reset=reset, observe=observe, step=step, phys_obs=phys_obs, teacher=teacher_fn)
        return self._c["fns"]

    def reset(self, key):
        from .base import as_key
        return self._fns()["reset"](as_key(key))

    def observe(self, s):
        return self._fns()["observe"](s)

    def step(self, s, action):
        return self._fns()["step"](s, action)

    def physical_obs(self, s):
        return self._fns()["phys_obs"](s.data)

    # teacher ---------------------------------------------------------------------------------------------------
    def attach_teacher(self, policy) -> "BodyTask":
        """policy: physical full obs -> physical action in [-1, 1] (e.g. teachers.LinearPolicy)."""
        self._teacher = policy
        self._c.pop("teacher_fn", None)
        return self

    @property
    def has_teacher(self) -> bool:
        return self._teacher is not None

    def teacher(self, s):
        if self._teacher is None:
            raise RuntimeError(f"{self.name} has no teacher; call grown.tasks.teachers.ensure_teacher(task)")
        if "teacher_fn" not in self._c:
            self._c["teacher_fn"] = self._fns()["teacher"](self._teacher)
        return self._c["teacher_fn"](s)

    # interface maps (numpy; for tests and analysis) -------------------------------------------------------------
    def iface_to_physical_action(self, a):
        it = self.params["iface"]
        c = np.zeros(self.act_dim, np.float32)
        c[np.asarray(it["act_perm"])] = np.asarray(it["act_mul"]) * np.asarray(a)
        return c

    def physical_to_iface_action(self, c):
        it = self.params["iface"]
        return np.asarray(c)[np.asarray(it["act_perm"])] / np.asarray(it["act_mul"])

    def physical_to_iface_obs(self, o):
        return np.asarray(o)[np.asarray(self.params["iface"]["obs_perm"])]

    def iface_to_physical_obs(self, o):
        out = np.zeros(self.obs_dim, np.float32)
        out[np.asarray(self.params["iface"]["obs_perm"])] = np.asarray(o)
        return out


# ---- families --------------------------------------------------------------------------------------------------


def _gen_dims(body):
    n_h = sum(len(l["axes"]) for l in body["links"])
    if body["root"] == "fixed":
        return 2 * n_h, n_h
    if body["dim"] == 2:
        return (3 + n_h - 1) + (3 + n_h), n_h
    return (7 + n_h - 2) + (6 + n_h), n_h


class BodyFamily(FamilyBase):
    """Generated bodies. sample(key, n, split): train / interp draw from the train box (TRAIN, GAIN_TRAIN); extrap
    pushes one or two physics axes outside it (EXTRAP) and the interface gains outside GAIN_TRAIN.

    horizon: steps per episode. scramble: draw random interfaces (else identity). p_hide_vel: probability that
    velocities are hidden."""
    name = "bodies"

    def __init__(self, horizon: int = 500, scramble: bool = True, p_hide_vel: float = 0.25, K: int | None = None,
                 time=None):
        self.horizon, self.scramble, self.p_hide_vel = horizon, scramble, p_hide_vel
        self.time = time_model(time, K)

    def make(self, body, rng=None, split="train", vel_vis=1) -> BodyTask:
        D, A = _gen_dims(body)
        iface = random_iface(rng, D, A, split) if (self.scramble and rng is not None) else identity_iface(D, A)
        p = dict(body=body, objective="height" if body["root"] == "fixed" else "forward",
                 frame_skip=body["frame_skip"], horizon=self.horizon, vel_vis=int(vel_vis), iface=iface, ctrl_w=1e-2)
        return BodyTask(p, "bodies", time=self.time)

    def sample_one(self, rng, split):
        body = draw_body(rng, split)
        vis = 0 if rng.random() < self.p_hide_vel else 1
        return self.make(body, rng, split, vis)

    def from_params(self, params):
        return BodyTask(params, "bodies", time=self.time)

    def mutate(self, task, key):
        """Perturb physics multiplicatively; occasionally add or drop a leaf link (new interface if dims change)."""
        rng = np_rng(key, 4)
        body = jsonable(task.params["body"])
        r = rng.random()
        if r < 0.15 and len(body["links"]) < 8:
            parent = int(rng.integers(-1, len(body["links"])))
            src = body["links"][int(rng.integers(len(body["links"])))]
            body["links"].append(dict(src, parent=parent, length=float(rng.uniform(0.15, 0.5)), sign=int(rng.choice([-1, 1]))))
        elif r < 0.25 and len(body["links"]) > 2:
            kids = {l["parent"] for l in body["links"]}
            leaves = [i for i in range(len(body["links"])) if i not in kids]
            drop = leaves[int(rng.integers(len(leaves)))]
            body["links"] = [dict(l, parent=l["parent"] - (l["parent"] > drop)) for i, l in enumerate(body["links"])
                             if i != drop]
        else:
            for a in rng.choice(["strength", "friction", "gravity", "mass", "length"], size=2, replace=False):
                f = math.exp(rng.normal(0, 0.2))
                if a in ("mass", "length"):
                    i = int(rng.integers(len(body["links"])))
                    body["links"][i][a] = float(np.clip(body["links"][i][a] * f, 0.05, 10.0))
                else:
                    body[a] = float(np.clip(body[a] * f, 0.05, 30.0))
        dims = _gen_dims(body)
        if dims == (task.obs_dim, task.act_dim):
            iface = task.params["iface"]
        else:
            iface = random_iface(rng, *dims) if self.scramble else identity_iface(*dims)
        return BodyTask(dict(task.params, body=body, iface=iface), "bodies", time=task.time)


class NamedBodyFamily(FamilyBase):
    """Named bodies (Hopper, Humanoid, ...) as points of the body distribution. Task i of a sample uses
    names[i % len(names)]. Splits: nominal (identity interface), train / interp (random interface, gains in
    GAIN_TRAIN), extrap (gains outside it)."""
    name = "named_bodies"

    def __init__(self, names=None, horizon: int | None = None, K: int | None = None, time=None):
        self.names = tuple(names or [n for n in ("hopper", "humanoid") if n in available_named()])
        self.horizon = horizon
        self.time = time_model(time, K)

    def make(self, name, rng=None, split="nominal") -> BodyTask:
        e = NAMED[name]
        p = dict(body=None, named=name, objective=e["objective"], frame_skip=e["frame_skip"],
                 horizon=self.horizon or e["horizon"], vel_vis=1, iface=None)
        probe = BodyTask(dict(p, iface=identity_iface(1, 1)), "named_bodies", time=self.time)
        D, A = probe.obs_dim, probe.act_dim
        p["iface"] = identity_iface(D, A) if split == "nominal" or rng is None else random_iface(rng, D, A, split)
        t = BodyTask(p, "named_bodies", time=self.time)
        t._c["mm"], t._c["info"] = probe.mj_model, probe.info
        return t

    def sample(self, key, n, split="train"):
        from .base import SPLIT_IDS
        sid = SPLIT_IDS[split]
        return [self.make(self.names[i % len(self.names)], np_rng(key, sid, i), split) for i in range(n)]

    def sample_one(self, rng, split):
        return self.make(self.names[int(rng.integers(len(self.names)))], rng, split)

    def nominal(self) -> list[BodyTask]:
        return [self.make(n) for n in self.names]

    def from_params(self, params):
        return BodyTask(params, "named_bodies", time=self.time)
