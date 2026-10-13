"""Run configuration: a nested dict read from YAML (PyYAML when installed, else a small built-in reader for the
subset the configs use) or JSON, deep-merged over `DEFAULTS`.

The built-in reader handles block mappings, block sequences of scalars or flow collections, flow collections
(`[a, b]`, `{k: v}`), comments, quoted strings, ints, floats, booleans and null. That is enough for
configs/grown/*.yaml; anything fancier needs PyYAML (`pip install pyyaml`).

Schema (every key optional; see DEFAULTS for the values used when missing):

  base                      another config file (relative path) this one overrides
  name                      run name (informational)
  seed                      search seed: per-task generators, initial circuits, data keys derive from (seed, task)
  workers                   processes; tasks are partitioned over them (results do not depend on it)
  substrate                 {name: sync | freerun | freerun_events, backend: cpu | cuda, threads: 0, home: null,
                            libdir: null} (freerun: home / libdir override $FREERUN_HOME / $FREERUN_LIBDIR; stimulus:
                            multistep (default: one engine call per episode) | step (step loop) | packed | api, see
                            FreerunSubstrate; physical: device
                            population and idealized twin per lane, grown.sim.devices.PhysicalSpec fields {select:
                            device | twin, seed, preset, boxes, scale_sigma, mismatch_sigma, jitter_rel, tick,
                            energy: approx | exact | off, run_devices}; sim_kw: freerun Simulator keywords;
                            freerun_events: the reference event-stream wrapper, grown.sim.freerun_events)
  time                      the physical time model (grown.tasks.TimeModel, docs/grown_time.md):
    tick                    seconds per substrate tick (1e-11: 10 ps, the SKY130 default; 1e-12 allowed)
    modulator_hz            sigma-delta input modulator rate (1e7; 1-100 MHz)
    event_budget            gate transitions allowed per control step per lane (null: unlimited)
    event_rate              or transitions per second (budget per step = event_rate * dt)
    budget_mode             ignore (default: only logged) | penalize | fail (fitness worst, error 1)
    budget_penalty          penalize: fitness -= budget_penalty * fraction of lanes over budget
    sim_event_cap           compute guard, not physics: events per step per lane above which a lane stops being
                            simulated; a capped lane's circuit counts as failed (unscorable), logged as `capped`
    sim_event_cap_rate      the same as transitions per second (cap per step = rate * dt); the smaller applies
    legacy_K                reproduce a pre-time-model run: every task gets this K, old dt and uids. Only for old
                            configs (labelled legacy); K is otherwise derived, K = round(dt / tick)
    strobe                  true: every task gets a sample-clock input wire (the last), one pulse per control step
                            rising at the step boundary (an ADC's strobe); n_in + 1, part of the task uid
    strobe_width            seconds the strobe is high (null: one modulator period, 100 ns at 10 MHz)
  tasks.source              catalog | family | demand_holdout
  tasks.names               catalog: training task names
  tasks.heldout             catalog: held-out task names (evaluation only)
  tasks.extra               catalog: more named signal tasks [{name, kind, params, dt}], usable in names / heldout
                            (dt: the task's sample period; else tasks.dt; required in physical mode)
  tasks.horizon             task horizon (steps)
  tasks.dt                  seconds per control step: null = each source's physical default (catalog: the entry's
                            own rate; family: drawn from 1 kHz-1 MHz), a number (every task), or [lo, hi] (family:
                            drawn inside). tasks.K is REMOVED: K = round(dt / time.tick) is derived; setting it raises
                            (use time.legacy_K to re-run an old config)
  tasks.family, kinds       family / demand_holdout: family name ("signal") and kinds to draw from (null: all)
  tasks.family_seed         family sampling key
  tasks.n_train, n_interp, n_extrap
                            family: Family.splits sizes (train on train, hold out interp and extrap)
  tasks.n_pool, box, max_train, max_heldout
                            demand_holdout: draw n_pool train-split tasks, train on the ones inside `box`
                            ({axis: [lo, hi]}, grown.tasks.demand_holdout) up to max_train, hold out the complement
                            up to max_heldout
  data.episodes, horizon    teacher episodes per task and steps per episode for the training dataset
  data.metric               per-step error: abs | sq | bit (grown.tasks.score)
  data.eval_episodes, eval_horizon
                            fresh episodes for grown_eval
  search.generations, batch, sampling (uniform | curiosity | lineage), eps, n_init, init_nodes, init_arity,
  search.init_feedback      random initial circuits may contain feedback
  search.operator           OperatorConfig fields (max_nodes, arity_max, delay_range, ...)
  archive.axes              list of Axis dicts {name, lo, hi, bins, source: structural | behavior, log}; behavior
                            names come from grown.search.SIG_NAMES, or grown.sim.base.EXTRA_BEHAVIOR when the
                            substrate supplies them (freerun with substrate.physical: gap_duty, gap_disagree,
                            energy, log10_energy)
  archive.save_every        save archives and history every M generations (0: only at the end)
  archive.rescore           inplace | rebin: how elites are re-scored when the objective or data change
  objective.schedule        imitation | linear | threshold (grown.search.Blend)
  objective.w_max, start, end, steps, threshold
  dagger.every              DAgger round every N generations (0: off)
  dagger.episodes, elites, max_episodes
  solved                    imitation error at or below which a task counts as solved
  status_every              print a status line every N generations
"""
from __future__ import annotations

import copy
import json
import os
import re

DEFAULTS = {
    "name": "run",
    "seed": 0,
    "workers": 1,
    "substrate": {"name": "sync", "backend": "cpu", "threads": 0},
    "time": {"tick": 1e-11, "modulator_hz": 1e7, "event_budget": None, "event_rate": None, "budget_mode": "ignore",
             "budget_penalty": 1.0, "sim_event_cap": None, "sim_event_cap_rate": None, "legacy_K": None,
             "strobe": False, "strobe_width": None},
    "tasks": {"source": "catalog", "names": ["xor2"], "heldout": [], "horizon": 1024, "dt": None,
              "family": "signal", "kinds": None, "family_seed": 0, "n_train": 32, "n_interp": 8, "n_extrap": 8,
              "n_pool": 64, "box": {}, "max_train": 16, "max_heldout": 16},
    "data": {"episodes": 2, "horizon": 256, "metric": "abs", "eval_episodes": 4, "eval_horizon": 256},
    "search": {"generations": 100, "batch": 64, "sampling": "curiosity", "eps": 0.1, "n_init": 64, "init_nodes": 8,
               "init_arity": [1, 3], "init_feedback": True, "operator": {"max_nodes": 64}},
    "archive": {"axes": [
        {"name": "n_gates", "lo": 0, "hi": 128, "bins": 8, "log": True},
        {"name": "cycle_frac", "lo": 0.0, "hi": 1.0, "bins": 4},
        {"name": "toggle_rate", "lo": 0.0, "hi": 0.5, "bins": 8, "source": "behavior"},
        {"name": "duty_mean", "lo": 0.0, "hi": 1.0, "bins": 4, "source": "behavior"}],
        "save_every": 50, "rescore": "inplace"},
    "objective": {"schedule": "imitation", "w_max": 0.5, "start": 0, "end": 0, "steps": 4, "threshold": 0.05},
    "dagger": {"every": 0, "episodes": 2, "elites": 4, "max_episodes": 0},
    "solved": 0.01,
    "status_every": 1,
}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "box":
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def read(path: str) -> dict:
    """The raw dict of one file; a top-level `base: other.yaml` (relative to this file) is read first and this
    file's keys are merged over it."""
    with open(path) as f:
        text = f.read()
    if path.endswith(".json"):
        raw = json.loads(text)
    else:
        try:
            import yaml
            raw = yaml.safe_load(text)
        except ImportError:
            raw = parse_yaml(text)
    raw = raw or {}
    base = raw.pop("base", None)
    if base:
        raw = merge(read(os.path.join(os.path.dirname(os.path.abspath(path)), base)), raw)
    return raw


class DeprecatedConfig(ValueError):
    """A config uses a key the time model removed."""


def check(cfg: dict) -> dict:
    """Refuse removed keys loudly (no silent fallback). Returns cfg."""
    if "K" in (cfg.get("tasks") or {}):
        raise DeprecatedConfig(
            f"tasks.K = {cfg['tasks']['K']!r} is no longer a config knob: the substrate ticks per control step are "
            "derived from the task's physical step, K = round(dt / time.tick) (docs/grown_time.md). Set tasks.dt "
            "(and time.tick) instead. To re-run a pre-time-model config exactly, set time.legacy_K to the old K "
            "and remove tasks.K.")
    return cfg


def load(path: str, overrides: dict | None = None) -> dict:
    cfg = merge(DEFAULTS, read(path))
    if overrides:
        cfg = merge(cfg, overrides)
    cfg.setdefault("config_path", os.path.abspath(path))
    return check(cfg)


# ---- a small YAML-subset reader ----------------------------------------------------------------------------------

_INT = re.compile(r"^[-+]?\d+$")
_FLOAT = re.compile(r"^[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$")


def _scalar(s: str):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    low = s.lower()
    if low in ("null", "~", ""):
        return None
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if _INT.match(s):
        return int(s)
    if _FLOAT.match(s) or low in (".inf", "-.inf"):
        return float(s.replace(".inf", "inf"))
    return s


def _strip_comment(line: str) -> str:
    q = None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def _flow(s: str):
    """Parse a flow collection or scalar."""
    pos = 0

    def ws():
        nonlocal pos
        while pos < len(s) and s[pos] in " \t":
            pos += 1

    def value():
        nonlocal pos
        ws()
        if s[pos] == "[":
            pos += 1
            out = []
            ws()
            if s[pos] == "]":
                pos += 1
                return out
            while True:
                out.append(value())
                ws()
                if s[pos] == ",":
                    pos += 1
                    continue
                if s[pos] == "]":
                    pos += 1
                    return out
                raise ValueError(f"bad flow sequence: {s!r}")
        if s[pos] == "{":
            pos += 1
            out = {}
            ws()
            if s[pos] == "}":
                pos += 1
                return out
            while True:
                ws()
                j = s.index(":", pos)
                key = _scalar(s[pos:j])
                pos = j + 1
                out[key] = value()
                ws()
                if s[pos] == ",":
                    pos += 1
                    continue
                if s[pos] == "}":
                    pos += 1
                    return out
                raise ValueError(f"bad flow mapping: {s!r}")
        start = pos
        q = None
        while pos < len(s):
            ch = s[pos]
            if q:
                if ch == q:
                    q = None
            elif ch in "'\"":
                q = ch
            elif ch in ",]}":
                break
            pos += 1
        return _scalar(s[start:pos])

    v = value()
    ws()
    if pos != len(s):
        raise ValueError(f"trailing text in {s!r}")
    return v


def parse_yaml(text: str):
    lines = []
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if line.strip():
            lines.append((len(line) - len(line.lstrip(" ")), line.strip()))

    def block(i, indent):
        if lines[i][1].startswith("- ") or lines[i][1] == "-":
            out = []
            while i < len(lines) and lines[i][0] == indent and (lines[i][1].startswith("- ") or lines[i][1] == "-"):
                item = lines[i][1][1:].strip()
                if item:
                    out.append(_flow(item))
                    i += 1
                else:
                    v, i = block(i + 1, lines[i + 1][0])
                    out.append(v)
            return out, i
        out = {}
        while i < len(lines) and lines[i][0] == indent:
            body = lines[i][1]
            if ":" not in body:
                raise ValueError(f"expected 'key: value', got {body!r}")
            k, v = body.split(":", 1)
            k, v = _scalar(k), v.strip()
            if v:
                out[k] = _flow(v)
                i += 1
            elif i + 1 < len(lines) and lines[i + 1][0] > indent:
                out[k], i = block(i + 1, lines[i + 1][0])
            elif i + 1 < len(lines) and lines[i + 1][0] == indent and lines[i + 1][1].startswith("- "):
                out[k], i = block(i + 1, indent)
            else:
                out[k] = None
                i += 1
        return out, i

    if not lines:
        return {}
    v, i = block(0, lines[0][0])
    if i != len(lines):
        raise ValueError(f"could not parse line {i}: {lines[i][1]!r}")
    return v
