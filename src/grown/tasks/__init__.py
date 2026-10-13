"""Tasks and teachers for the circuit search.

Families (all share the contract in `base`):
  signal        specs with exact reference functions (no physics; numpy + JAX only)
  bodies        procedurally generated MuJoCo/MJX bodies with ARS teachers; named points (Hopper, Humanoid, ...)
  sensorimotor  wrapper: raw sensor bit streams in, PWM out, around any body task
  frontier      POET-style proposal of new tasks at the edge of solvability

Only `base` and `signal` are imported eagerly; `bodies` / `sensorimotor` import mujoco when first used.
"""
from .base import (AXES, DemandTags, Family, FamilyBase, InCode, IOSpec, OutCode, Splits, Task, TeacherDataset,
                   as_key, demand_coverage, demand_holdout, np_rng, rollout, score, score_by_episode, stable_hash, stateless,
                   task_meta)

__all__ = ["AXES", "DemandTags", "Family", "FamilyBase", "InCode", "IOSpec", "OutCode", "Splits", "Task",
           "TeacherDataset", "as_key", "demand_coverage", "demand_holdout", "np_rng", "rollout", "score", "score_by_episode",
           "stable_hash", "stateless", "task_meta", "get_family"]


def get_family(name: str, **kw):
    """Family by name: signal, bodies, named_bodies, sensorimotor (lazy imports)."""
    if name == "signal":
        from .signal import SignalFamily
        return SignalFamily(**kw)
    if name == "bodies":
        from .bodies import BodyFamily
        return BodyFamily(**kw)
    if name == "named_bodies":
        from .bodies import NamedBodyFamily
        return NamedBodyFamily(**kw)
    if name == "sensorimotor":
        from .sensorimotor import SensorimotorFamily
        return SensorimotorFamily(**kw)
    raise KeyError(name)
