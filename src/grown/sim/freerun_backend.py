"""freerun adapter and backend (event-driven clockless simulation; see ~/freerun/docs/SPEC.md).

`to_freerun(netlist)` / `from_freerun(circuit, outputs)` convert between our flattened netlists and
`freerun.Circuit` (the same arrays: nets 0..I-1 inputs, net I+g gate g, pin 0 = LSB of the truth-table row,
per-arc rise/fall delays, init per net). The conversion is lossless both ways except for primary-input init
values, which our circuits do not carry (the backend sets them from the step-0 inputs).

`FreerunSubstrate` evaluates batches in one of four ways (`stimulus`, config key substrate.stimulus):
  multistep (default) grown.sim.stepper with freerun's multi-step runs (Simulator.run_steps, freerun SPEC §13): the
          whole episode of every lane of the batch in one engine call, input changes as stimulus triples, per-step
          outputs (value, high time, gate events) written by the engine at the step boundaries, the event budget
          applied inside the engine. Bit-identical to "step" for every lane under budget. Used whenever the freerun
          library has run_steps; with an older library the substrate falls back to "step" by itself.
  step    freerun's closed-loop stepping API through grown.sim.stepper: one Simulator for the whole batch
          (state device-resident on CUDA), dense input rows per control step, dense outputs (value, high time) per
          step, no traces: two engine calls per control step (kept for differential tests and old libraries). `evaluate_many(jobs)` takes lanes of several tasks at once (different K, widths, episode
          lengths, input streams) in ONE batch: a whole generation is one Simulator. `evaluate_batch(nb, lane_circ,
          ...)` is the NetBatch entry point of grown.search's generation batching.
  packed  the pre-stepping path: input changes as stimulus triples (packed in numpy), one `run(T)`, the output
          nets' traces turned into per-step samples and high fractions.
  api     the same through Simulator.add_stimulus (validated in Python; slow; kept for differential tests).
All four give bit-identical results (tests); "multistep" is the fast one, on CUDA above all. With `physical`
(grown.sim.devices) the multistep and step paths also draw a device per lane and run the idealized twin next to it
(E2).

freerun is found as an installed package, or under $FREERUN_HOME/python (default: the vendored freerun/ of this
repository, else ~/freerun); its compiled backends are loaded from $FREERUN_LIBDIR (default: the package's _lib). Both can also be given per substrate
(`home`, `libdir`; config keys substrate.home / substrate.libdir), which set the same variables before freerun is
imported. On the cluster the sbatch scripts set them (hpc/grown/env.sh).

`FreerunSubstrate.timing` accumulates wall seconds per phase of every call (convert: netlists to freerun
circuits / lane arrays; create: Simulator construction, batch packing and device allocation; stimulus: input changes
to stimulus triples or dense rows; kernel: the backend's run / run_steps call, or all step calls; extract: copying the trace and
counters back (packed / api); post: per-step samples and high fractions; devices: device draws (physical)) plus
calls, steps (control-step cuts: the step loop's engine calls), launches (engine calls made), lanes, simulated lane-ticks and events (committed net transitions, primary inputs
included), and for the step path engine_kernel (the engine's own kernel time, freerun Simulator.profile) and
engine_windows, so a profile can say how much of an evaluation is the simulator and how much is Python around it.

Lane placement (`placement`, config key substrate.placement, multistep path): a GPU call lasts as long as its busiest
lane (a lane's events are processed in time order, ~3e5 events/s per lane on an H200) while a CPU thread does ~4.5e7
events/s on one lane, so lanes predicted heavy run on the CPU backend's threads and the rest on the GPU, both at
once (the CPU call in a thread; engine calls release the GIL). mode: none (default: every lane on `backend`), probe
(each lane is first simulated for its first probe_steps steps on the CPU; its events per step x its step count is
the prediction), cpu (every lane on the CPU threads, heaviest predicted first when probe_steps > 0: the threads
claim lanes in order, so the longest start first). A GPU lane that makes more than gpu_guard x the predicted call
time's worth of events is stopped in the engine and run again on the CPU (a lane that starts oscillating after the
probed steps would otherwise hold the whole GPU call). The split minimizes max(GPU time = busiest GPU lane / gpu_lane_rate,
CPU time = max(busiest CPU lane, CPU events / threads) / cpu_thread_rate) over "the c heaviest lanes go to the CPU".
Lanes are independent and the backends bit-identical, so results do not depend on the placement. Every placed call
appends a record to `placement_log` (take_placement_log()): lanes and events on each side, predicted and measured,
busiest lane, seconds.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..circuit.flatten import Netlist, flatten, from_netlist
from ..circuit.graph import Circuit
from .base import EvalResult, tick_inputs
from .stepper import Job, LaneResult, LaneSet, NetArrays, run_lanes

STIMULUS = ("multistep", "step", "packed", "api")
STEPPING = ("multistep", "step")         # the grown.sim.stepper paths
PLACEMENT_MODES = ("none", "probe", "cpu")


@dataclasses.dataclass
class Placement:
    """Lane placement between the CPU backend's threads and `backend` (module docstring)."""
    mode: str = "none"
    probe_steps: int = 2            # steps simulated per lane for the prediction (probe; cpu: the lane order)
    cpu_threads: int = 0            # CPU backend threads; 0: the job's CPUs (SLURM_CPUS_PER_TASK, else all cores)
    gpu_lane_rate: float = 3.3e5    # events/s of one lane on the GPU (measured, H200)
    cpu_thread_rate: float = 4.5e7  # events/s of one CPU thread (measured)
    gpu_guard: float = 1.0          # a GPU lane may make at most this x its share of the predicted call time in
    #                                 events per step (an engine early stop); lanes the guard stops are run again on
    #                                 the CPU, so a lane the probe misjudged cannot hold the GPU call (0: no guard)
    gpu_min_events: float = 1e5     # the guard never allows fewer events per lane than this

    def __post_init__(self):
        if self.mode not in PLACEMENT_MODES:
            raise ValueError(f"placement mode {self.mode!r} not in {PLACEMENT_MODES}")
        if int(self.probe_steps) < 0 or (self.mode == "probe" and int(self.probe_steps) < 1):
            raise ValueError("placement.probe_steps must be >= 1 (mode cpu: >= 0, 0 = no probe, lanes in order)")

    @classmethod
    def from_config(cls, d) -> "Placement":
        if isinstance(d, Placement):
            return d
        if isinstance(d, str):
            return cls(mode=d)
        d = dict(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        bad = sorted(set(d) - known)
        if bad:
            raise KeyError(f"unknown placement keys {bad} (known: {sorted(known)})")
        return cls(**d)

    def threads(self) -> int:
        if self.cpu_threads:
            return int(self.cpu_threads)
        n = os.environ.get("SLURM_CPUS_PER_TASK")
        if n:
            return max(int(n), 1)
        try:
            return max(len(os.sched_getaffinity(0)), 1)
        except AttributeError:  # pragma: no cover
            return os.cpu_count() or 1

    def split(self, est: np.ndarray, threads: int) -> int:
        """How many of the heaviest lanes (est sorted descending) go to the CPU."""
        L = len(est)
        if L == 0:
            return 0
        cum = np.concatenate([[0.0], np.cumsum(est)])
        c = np.arange(L + 1)
        cpu, gpu = self._times(est, threads)
        return int(np.lexsort((cpu + gpu, np.maximum(cpu, gpu)))[0])   # the shortest call; ties: less total work

    def _times(self, est: np.ndarray, threads: int):
        """Predicted CPU and GPU seconds when the c heaviest lanes (est descending) go to the CPU, c = 0 .. L."""
        L = len(est)
        cum = np.concatenate([[0.0], np.cumsum(est)])
        c = np.arange(L + 1)
        cpu = np.where(c > 0, np.maximum(est[0] if L else 0.0, cum / threads), 0.0) / self.cpu_thread_rate
        gpu = np.append(est, 0.0)[c] / self.gpu_lane_rate
        return cpu, gpu


def _take(r: LaneResult, m: np.ndarray) -> LaneResult:
    """The lanes m (bool) of a LaneResult."""
    return LaneResult(r.outputs[m], r.high[m], r.toggles[m], None, r.calls,
                      None if r.events is None else r.events[m], None if r.over_budget is None else r.over_budget[m])


def _merge(lanes: LaneSet, parts) -> LaneResult:
    """One LaneResult for `lanes` from (lane indices, LaneResult of lanes.select(indices)) parts."""
    L, Smax, W = len(lanes), int(lanes.S.max(initial=0)), lanes.outs.shape[1]
    out = np.zeros((L, Smax, W), np.uint8)
    high = np.zeros((L, Smax, W), np.float32)
    tog = np.zeros(L, np.float64)
    events = None if all(r.events is None for _, r in parts) else np.zeros((L, Smax), np.int64)
    over = None if all(r.over_budget is None for _, r in parts) else np.zeros(L, bool)
    calls = 0
    for idx, r in parts:
        s = r.outputs.shape[1]
        out[idx, :s] = r.outputs
        high[idx, :s] = r.high
        tog[idx] = r.toggles
        if r.events is not None:
            events[idx, :r.events.shape[1]] = r.events
        if r.over_budget is not None:
            over[idx] = r.over_budget
        calls += r.calls
    return LaneResult(out, high, tog, None, calls, events, over)


def default_freerun_home() -> str:
    """$FREERUN_HOME, else the copy of freerun vendored next to src/ in this repository, else ~/freerun."""
    if os.environ.get("FREERUN_HOME"):
        return os.environ["FREERUN_HOME"]
    vendored = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "freerun"))
    return vendored if os.path.isdir(vendored) else os.path.expanduser("~/freerun")


def import_freerun(home: str | None = None, libdir: str | None = None):
    """The freerun package. `home` (else default_freerun_home()) is searched when freerun is not installed;
    `libdir` (else $FREERUN_LIBDIR, else the package's _lib) is where its compiled backends are loaded from."""
    if home:
        os.environ["FREERUN_HOME"] = os.path.abspath(os.path.expanduser(home))
    if libdir:
        os.environ["FREERUN_LIBDIR"] = os.path.abspath(os.path.expanduser(libdir))
    try:
        import freerun  # noqa: F401
    except ImportError:
        home = default_freerun_home()
        path = os.path.join(home, "python")
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)
        import freerun  # noqa: F401
    import freerun.netlist
    import freerun.sim
    if libdir:                                      # freerun.sim reads FREERUN_LIBDIR once, at import
        from pathlib import Path
        want = Path(os.environ["FREERUN_LIBDIR"])
        if freerun.sim._LIBDIR != want:
            freerun.sim._LIBDIR = want
            freerun.sim._libs.clear()
    return freerun


def to_freerun(n: Netlist, input_init=None):
    fr = import_freerun()
    init = n.init.copy()
    if input_init is not None:
        init[:n.n_inputs] = np.asarray(input_init, np.uint8)
    return fr.netlist.Circuit(n.n_inputs, n.fanin_ptr, n.fanin, n.tt, n.dr, n.df, init)


def from_freerun(fc, outputs) -> Circuit:
    """freerun Circuit plus output net indices -> our Circuit of GATE tokens."""
    net = Netlist(fc.n_inputs, fc.fanin_ptr.astype(np.int64), fc.fanin.astype(np.int32), fc.tt.astype(np.uint64),
                  fc.dr.astype(np.int32), fc.df.astype(np.int32), fc.init.astype(np.uint8),
                  np.asarray(outputs, np.int64), [(g,) for g in range(fc.n_gates)])
    return from_netlist(net)


def _dense_outputs(traces, outs, init, I, x4, S, K):
    """Per-step samples and high fractions of every output from the traces, all lanes at once.

    traces: per lane, rows (t, net, v) of watched-net changes; outs [B, n_out] output nets; init [B, n_out] their
    initial values; x4 [B, S, Kx, I] inputs (outputs wired to a primary input read them directly). The value of an
    output at tick u (after phase A at u) is the last change at or before u; out[:, s] is its value at (s+1)K and
    high[:, s] the mean over ticks s*K+1 .. (s+1)K (base.py)."""
    B, n_out = outs.shape
    T = S * K
    out = np.zeros((B, S, n_out), np.uint8)
    high = np.zeros((B, S, n_out), np.float32)
    lens = np.array([len(t) for t in traces], np.int64)
    tr = np.concatenate(traces) if lens.sum() else np.zeros((0, 3), np.int64)
    lane = np.repeat(np.arange(B), lens)
    for j in range(n_out):
        o = outs[:, j]
        wired = o < I
        if wired.any():
            w = np.flatnonzero(wired)
            out[w, :, j] = x4[w, :, -1, o[w]]
            high[w, :, j] = x4[w, :, :, o[w]].mean(-1)
        g = np.flatnonzero(~wired)
        if not len(g):
            continue
        m = tr[:, 1] == o[lane]
        rows = np.full(B, -1, np.int64)
        rows[g] = np.arange(len(g))
        q, t, v = rows[lane[m]], tr[m, 0], tr[m, 2]
        k = np.lexsort((t, q))                                    # chronological per output, whatever the trace order
        q, t, v = q[k], t[k], v[k]
        last = np.full((len(g), T + 1), -1, np.int64)            # index of the change in effect at each tick
        last[q, t] = np.arange(len(t))
        np.maximum.accumulate(last, axis=1, out=last)
        v = np.append(v, 0)                                       # never read; keeps the gather valid without changes
        val = np.where(last >= 0, v[np.maximum(last, 0)], init[g, j][:, None]).astype(np.uint8)
        out[g, :, j] = val[:, K::K]
        high[g, :, j] = val[:, 1:].reshape(len(g), S, K).mean(-1)
    return out, high


class FreerunSubstrate:
    name = "freerun"
    threadsafe = True       # one evaluation at a time may run in another thread (search.SearchGroup overlap): the
    #                         engine calls release the GIL; the substrate is never used by two threads at once

    def __init__(self, backend: str = "cpu", threads: int = 0, home: str | None = None, libdir: str | None = None,
                 stimulus: str = "multistep", physical=None, sim_kw: dict | None = None, placement=None):
        """backend: freerun backend ("cpu", "cuda" or "ref"); threads: CPU backend threads (0: all cores).
        stimulus: "multistep" (one engine call per episode, module docstring); "step" (the stepping API's step loop);
        "packed" builds the engine's stimulus arrays for the whole
        batch in numpy and makes one run() call; "api" goes through Simulator.add_stimulus (10-100x slower; kept for
        differential tests). physical: None, or a grown.sim.devices.PhysicalSpec (or its config dict): every lane
        draws a device and runs next to its idealized twin (step path only). sim_kw: extra freerun Simulator
        keywords for the step path (stepped, smem, block_threshold, ...: kernel variants for tests). placement:
        lane placement between CPU threads and `backend` (Placement or its config dict / mode name; module
        docstring)."""
        if backend not in ("cpu", "cuda", "ref"):
            raise ValueError(f"unknown freerun backend {backend!r}")
        if stimulus not in STIMULUS:
            raise ValueError(f"unknown stimulus mode {stimulus!r} (one of {STIMULUS})")
        self.backend = backend
        self.threads = threads
        self.stimulus = stimulus
        self.home, self.libdir = home, libdir
        self.sim_kw = dict(sim_kw or {})
        self.timing: dict = defaultdict(float)
        if home or libdir:
            import_freerun(home, libdir)
        self.physical = None
        if physical:
            from .devices import PhysicalSpec
            self.physical = physical if isinstance(physical, PhysicalSpec) else PhysicalSpec.from_config(physical)
            if stimulus not in STEPPING:
                raise ValueError("substrate.physical needs a stepping path (stimulus: multistep or step)")
        self.placement = Placement.from_config(placement)
        self.placement_log: list = []
        if self.placement.mode != "none" and (stimulus != "multistep" or self.physical is not None):
            raise ValueError("substrate.placement needs stimulus: multistep and no physical device lanes")

    @classmethod
    def from_config(cls, d: dict) -> "FreerunSubstrate":
        """From a config's substrate section: backend, threads, home, libdir, stimulus (default multistep), physical
        (grown.sim.devices.PhysicalSpec fields), sim_kw."""
        d = dict(d or {})
        known = ("name", "backend", "threads", "home", "libdir", "stimulus", "physical", "sim_kw", "placement")
        bad = [k for k in d if k not in known]
        if bad:
            raise KeyError(f"unknown freerun substrate keys {bad} (known: {known})")
        return cls(d.get("backend", "cpu"), int(d.get("threads", 0) or 0), d.get("home"), d.get("libdir"),
                   d.get("stimulus", "multistep"), d.get("physical"), d.get("sim_kw"), d.get("placement"))

    def evaluate(self, circuits, inputs, K: int, library=None, per_tick: bool = False) -> EvalResult:
        return self.evaluate_netlists([flatten(c, library) for c in circuits], inputs, K, per_tick=per_tick)

    def evaluate_netlists(self, nets, inputs, K: int, per_tick: bool = False) -> EvalResult:
        if self.stimulus in STEPPING:
            return self.evaluate_many([Job(nets, None, inputs, K, per_tick)])[0]
        return self._evaluate_run(nets, inputs, K, per_tick)

    def evaluate_batch(self, nb, lane_circ, inputs, K: int, per_tick: bool = False) -> EvalResult:
        """A NetBatch (circuit.netbatch, or stepper.NetArrays): lane b runs circuit lane_circ[b] (None: lane b runs
        circuit b) on inputs[b]."""
        if self.stimulus in STEPPING:
            return self.evaluate_many([Job(nb, lane_circ, inputs, K, per_tick)])[0]
        nets = [nb.netlist(b) for b in range(len(nb))] if hasattr(nb, "netlist") else _netlists(nb)
        lc = np.arange(len(nets)) if lane_circ is None else np.asarray(lane_circ)
        return self._evaluate_run([nets[c] for c in lc], inputs, K, per_tick)

    def evaluate_requests(self, requests) -> list:
        """grown.sim.base.LaneBatch requests (distinct streams + a stream index per lane), all in one engine batch
        on the step path (the search's one call per generation)."""
        from .stepper import EventStreams
        jobs = []
        for r in requests:
            if isinstance(r.streams, EventStreams):              # base.EventLaneBatch: sparse inputs, explicit S
                jobs.append(Job(r.nets, r.lane_circ, r.streams, r.K, False, r.lane_stream, int(r.S),
                                getattr(r, "budget", None)))
                continue
            st = np.asarray(r.streams, np.uint8)
            per_tick = st.ndim == 4 and st.shape[2] > 1
            jobs.append(Job(r.nets, r.lane_circ, st, r.K, per_tick, r.lane_stream))
        return self.evaluate_many(jobs)

    def evaluate_many(self, jobs) -> list:
        """Several batches in one engine batch (step path): jobs are stepper.Job(nets, lane_circ, x, K, per_tick) or
        tuples of those fields; they may differ in K, episode length, input and output widths. -> one EvalResult per
        job (lanes in job order). The other paths evaluate the jobs one after another."""
        jobs = [j if isinstance(j, Job) else Job(*j) for j in jobs]
        if not jobs:
            return []
        if self.stimulus not in STEPPING:
            out = []
            for j in jobs:
                if not isinstance(j.x, np.ndarray) and hasattr(j.x, "off"):     # event streams: per tick (small K)
                    from .base import EventLaneBatch
                    lc = np.arange(len(NetArrays.of(j.nets))) if j.lane_circ is None else j.lane_circ
                    rows = np.arange(len(lc)) if j.rows is None else j.rows
                    j = Job(j.nets, lc, EventLaneBatch(NetArrays.of(j.nets), lc, j.x, rows, j.K, j.S).inputs(), j.K,
                            True)
                x = j.x if j.rows is None else np.asarray(j.x)[np.asarray(j.rows)]
                if x.ndim == 4 and x.shape[2] == 1 and int(j.K) != 1:
                    x, pt = x[:, :, 0], False
                else:
                    pt = j.per_tick or (x.ndim == 4)
                out.append(self.evaluate_batch(NetArrays.of(j.nets), j.lane_circ, x, j.K, pt))
            return out
        tm = self.timing
        t0 = time.perf_counter()
        lanes = LaneSet.build(jobs)
        tm["convert"] += time.perf_counter() - t0
        if self.physical is not None:
            from .devices import run_physical
            res, extra = run_physical(lanes, self.physical, self.backend, self.threads, tm, self.sim_kw,
                                      multistep=self._multistep())
        elif self.placement.mode != "none":
            res, extra = self._run_placed(lanes), {}
        else:
            res, extra = run_lanes(lanes, self.backend, self.threads, timing=tm, sim_kw=self.sim_kw,
                                   multistep=self._multistep()), {}
        from .slow import lane_slow_levels, table as slow_table
        lev = lane_slow_levels(lanes)
        if (lev >= 0).any():                 # slow delay elements: their static power and count per lane
            gl = lanes.gate_lane()
            sg = np.flatnonzero(lev >= 0)
            extra = dict(extra, slow_static_w=np.bincount(gl[sg], slow_table().p_static_w[lev[sg]],
                                                          minlength=len(lanes)),
                         slow_elements=np.bincount(gl[sg], minlength=len(lanes)).astype(np.float64))
        out = []
        G = lanes.n_gates
        for j, jb in enumerate(jobs):
            m = np.flatnonzero(lanes.job == j)
            S, W = int(lanes.S[m[0]]), int(lanes.n_out[m[0]])
            r = EvalResult(res.outputs[m, :S, :W], res.high[m, :S, :W], res.toggles[m], G[m].astype(np.int64),
                           int(jb.K), events=None if res.events is None else res.events[m, :S],
                           over_budget=None if res.over_budget is None else res.over_budget[m])
            r.extra = {k: v[m] for k, v in extra.items()}
            out.append(r)
        return out

    def take_placement_log(self) -> list:
        log, self.placement_log = self.placement_log, []
        return log

    def _run_placed(self, lanes: LaneSet) -> LaneResult:
        """run_lanes with the lanes split between the CPU backend's threads and `backend` (module docstring)."""
        pl, tm = self.placement, self.timing
        L = len(lanes)
        nthr = pl.threads()
        t0 = time.perf_counter()
        est = None
        if pl.mode == "cpu" and int(pl.probe_steps) == 0:
            n_cpu, order = L, np.arange(L)
        else:
            P = np.minimum(lanes.S, int(pl.probe_steps))
            pr = run_lanes(dataclasses.replace(lanes, S=P), "cpu", nthr, multistep=True)
            ev = np.asarray(pr.events, np.float64) if pr.events is not None else np.zeros((L, 1))
            est = ev.sum(1) / np.maximum(P, 1) * lanes.S
            order = np.argsort(-est, kind="stable")
            n_cpu = L if pl.mode == "cpu" else pl.split(est[order], nthr)
        t1 = time.perf_counter()
        # CPU lanes heaviest first: the engine's threads claim lanes in order, so the longest start first (LPT)
        cpu_idx, gpu_idx = order[:n_cpu], np.sort(order[n_cpu:])
        side = {}
        real = np.asarray(lanes.extra.get("budget", np.full(L, np.inf)), np.float64)
        guard = None
        if len(gpu_idx) and pl.gpu_guard > 0 and est is not None:
            cpu_t, gpu_t = pl._times(est[order], nthr)
            allow = max(pl.gpu_guard * max(cpu_t[n_cpu], gpu_t[n_cpu]) * pl.gpu_lane_rate, pl.gpu_min_events)
            guard = np.ceil(allow / np.maximum(lanes.S[gpu_idx], 1))        # events per step per GPU lane

        def run(name, idx, backend, threads, sim_kw, budget=None):
            ts = time.perf_counter()
            t = defaultdict(float)
            sub = lanes.select(idx)
            if budget is not None:
                sub = dataclasses.replace(sub, extra=dict(sub.extra, budget=budget))
            r = run_lanes(sub, backend, threads, timing=t, sim_kw=sim_kw, multistep=self._multistep())
            side[name] = (time.perf_counter() - ts, t)
            return r
        parts = []
        rerun = np.zeros(0, np.int64)
        with ThreadPoolExecutor(1) as ex:
            fut = ex.submit(run, "cpu", cpu_idx, "cpu", nthr, None) if len(cpu_idx) else None
            if len(gpu_idx):
                gb = None if guard is None else np.minimum(real[gpu_idx], guard)
                rg = run("gpu", gpu_idx, self.backend, self.threads, self.sim_kw, gb)
                if guard is not None and rg.over_budget is not None:
                    hit = np.asarray(rg.over_budget, bool) & (guard < real[gpu_idx])  # stopped by the guard
                    rerun = gpu_idx[hit]
                    keep = ~hit
                    if not np.isfinite(real[gpu_idx]).any():      # no real budget: no stop flags in the result
                        rg.over_budget = None
                    parts.append((gpu_idx[keep], _take(rg, keep)))
                else:
                    parts.append((gpu_idx, rg))
            if fut is not None:
                parts.append((cpu_idx, fut.result()))
        if len(rerun):                                  # misjudged lanes: the whole episode again on the CPU
            parts.append((rerun, run("rerun", rerun, "cpu", nthr, None)))
        res = _merge(lanes, parts)
        t2 = time.perf_counter()
        for _, t in side.values():
            for k, v in t.items():
                tm[k] += v
        tm["placement_probe"] += t1 - t0
        lane_ev = res.events.sum(1).astype(np.float64) if res.events is not None else res.toggles
        busiest = int(np.argmax(lane_ev)) if L else -1
        on_cpu = np.zeros(L, bool)
        on_cpu[cpu_idx] = True
        self.placement_log.append(dict(
            mode=pl.mode, lanes=L, threads=nthr, cpu_lanes=int(len(cpu_idx)), gpu_lanes=int(len(gpu_idx)),
            events=float(lane_ev.sum()), events_cpu=float(lane_ev[on_cpu].sum()),
            events_gpu=float(lane_ev[~on_cpu].sum()),
            est_events=None if est is None else float(est.sum()),
            busiest_lane=float(lane_ev[busiest]) if L else 0.0,
            busiest_on="cpu" if L and on_cpu[busiest] else "gpu",
            busiest_K=int(lanes.K[busiest]) if L else 0,
            max_gpu_lane=float(lane_ev[~on_cpu].max(initial=0.0)),
            gpu_guard_events=None if guard is None else float(guard.max() * lanes.S[gpu_idx].max()),
            rerun_lanes=int(len(rerun)), rerun_s=side.get("rerun", (0.0,))[0],
            probe_s=t1 - t0, cpu_s=side.get("cpu", (0.0,))[0], gpu_s=side.get("gpu", (0.0,))[0], wall_s=t2 - t0))
        return res

    def _multistep(self):
        """run_lanes' multistep argument: None (use run_steps when the library has it) or False (the step loop)."""
        return None if self.stimulus == "multistep" else False

    def _evaluate_run(self, nets, inputs, K: int, per_tick: bool = False) -> EvalResult:
        """The pre-stepping path (stimulus packed | api): one run() over the whole episode, traces of the outputs."""
        B = len(nets)
        x4 = tick_inputs(inputs, B, K, per_tick)                  # [B, S, Kx, I]
        S, Kx = x4.shape[1], x4.shape[2]
        x = x4.reshape(B, S * Kx, x4.shape[3])                   # one row per input change opportunity
        dt_in = K // Kx                                           # ticks between rows (K held, 1 per tick)
        I = nets[0].n_inputs
        n_out = len(nets[0].outputs)
        if any(n.n_inputs != I or len(n.outputs) != n_out for n in nets):
            raise ValueError("all circuits in a batch need the same number of inputs and outputs")
        tm = self.timing
        t0 = time.perf_counter()
        circuits = [to_freerun(n, x[b, 0]) for b, n in enumerate(nets)]
        watch = [sorted(set(int(o) for o in n.outputs if o >= I)) for n in nets]
        tm["convert"] += time.perf_counter() - t0
        traces, toggles = self._drive(circuits, watch, x, dt_in, S * K)
        t0 = time.perf_counter()
        outs = np.stack([n.outputs for n in nets]).astype(np.int64)          # [B, n_out]
        init = np.stack([n.init[np.maximum(n.outputs, 0)] for n in nets]).astype(np.uint8)
        out, high = _dense_outputs(traces, outs, init, I, x4, S, K)
        tm["post"] += time.perf_counter() - t0
        tm["calls"] += 1
        tm["lanes"] += B
        tm["ticks"] += B * (S * K + 1)
        return EvalResult(out, high, toggles, np.array([n.n_gates for n in nets], np.int64), K)

    def _drive(self, circuits, watch, x, dt_in, T):
        """Current freerun API: stimulus triples and one run. x [B, R, I]: row r is applied at time r * dt_in.
        -> (per-circuit trace rows (t, net, v), toggles [B])."""
        fr = import_freerun()
        tm = self.timing
        B, R, I = x.shape
        t0 = time.perf_counter()
        # every watched net changes at most once per tick (inertial delays >= 1), so this capacity cannot overflow
        cap = max(1 << 10, max((len(w) for w in watch), default=0) * (T + 2))
        sim = fr.sim.Simulator(circuits, backend=self.backend, watch=watch, threads=self.threads,
                               trace_capacity=cap)
        t1 = time.perf_counter()
        tm["create"] += t1 - t0
        ext = [0.0, 0.0]                                 # python inside run(): trace extraction, stimulus packing

        def timed(fn, slot):
            def wrapped(*a, **kw):
                t = time.perf_counter()
                try:
                    return fn(*a, **kw)
                finally:
                    ext[slot] += time.perf_counter() - t
            return wrapped
        packed = self.stimulus == "packed" and self.backend != "ref" and hasattr(sim, "_take_stimulus")
        if packed:
            # Simulator._take_stimulus(T) -> (per-circuit offsets, t, net, v), each circuit's events sorted by (t, net):
            # np.nonzero over [B, R-1, I] yields exactly that order. Every event is a valid primary-input change at
            # 0 < t < T, which is what add_stimulus would check one event at a time.
            if R > 1:
                b, r, i = np.nonzero(x[:, 1:] != x[:, :-1])
                r = r + 1
                off = np.zeros(B + 1, np.int64)
                off[1:] = np.cumsum(np.bincount(b, minlength=B))
                arrays = (off, (r * dt_in).astype(np.int64), i.astype(np.int32), x[b, r, i].astype(np.uint8))
            else:
                arrays = (np.zeros(B + 1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int32),
                          np.zeros(0, np.uint8))
            sim._take_stimulus = lambda *a: arrays       # (T) before freerun 0eb0067, (T, dense_rows) since
        wname = "windows" if hasattr(type(sim), "windows") else "_windows"   # public since freerun 0eb0067
        setattr(sim, wname, timed(getattr(sim, wname), 0))   # split run() into the backend call and the python around it
        sim._take_stimulus = timed(sim._take_stimulus, 1)
        try:
            if not packed:
                for b in range(B):
                    if R > 1:
                        r, i = np.nonzero(x[b, 1:] != x[b, :-1])
                        r = r + 1
                        sim.add_stimulus(b, zip((r * dt_in).tolist(), i.tolist(), x[b, r, i].tolist()))
            t2 = time.perf_counter()
            tm["stimulus"] += t2 - t1
            wins = sim.run(T + 1)
            t3 = time.perf_counter()
        finally:
            t4 = time.perf_counter()
            sim.close()
            tm["destroy"] += time.perf_counter() - t4
        tm["kernel"] += (t3 - t2) - ext[0] - ext[1]
        tm["extract"] += ext[0]
        tm["stimulus"] += ext[1]
        t4 = time.perf_counter()
        toggles = np.array([w.toggles[I:].sum() for w in wins], np.float64)
        tm["events"] += float(sum(int(w.toggles.sum()) for w in wins))   # committed transitions, inputs included
        tm["extract"] += time.perf_counter() - t4
        return [w.trace for w in wins], toggles


def _netlists(na) -> list:
    """Netlists of a NetArrays (stepper layout)."""
    out = []
    I = na.n_inputs
    for b in range(len(na)):
        g0, g1 = int(na.goff[b]), int(na.goff[b + 1])
        p0, p1 = int(na.poff[g0]), int(na.poff[g1])
        init = np.zeros(I + g1 - g0, np.uint8)
        init[I:] = na.init[g0:g1]
        out.append(Netlist(I, np.asarray(na.poff[g0:g1 + 1], np.int64) - p0, np.asarray(na.fanin[p0:p1], np.int32),
                           np.asarray(na.tt[g0:g1], np.uint64), np.asarray(na.dr[p0:p1], np.int32),
                           np.asarray(na.df[p0:p1], np.int32), init, np.asarray(na.outs[b], np.int64),
                           [()] * (g1 - g0)))
    return out
