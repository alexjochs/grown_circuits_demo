"""Simulator: one API over all backends.

    sim = Simulator(circuits, backend="cpu")
    sim.add_stimulus(b, [(t, net, v), ...])     # circuit b, local primary-input net
    wins = sim.run(T)                            # list of per-circuit Window objects for [previous T, T)

Closed-loop stepping (state stays in the engine, on the device for CUDA):

    sim = Simulator(circuits, backend="cuda", watch=None, net_observables=False)
    sim.set_io(n_inputs=4, outputs=[[n0, n1], ...], high=True)   # dense input width, output nets per circuit
    start = sim.snapshot()
    for step in range(1000):
        r = sim.step(sim.t + dt, inputs)         # inputs uint8 [B, 4] applied at the current time
        r.values, r.high                         # [B, n_out] at the end of the step / over the step
    sim.reset()                                  # power-up state (or sim.restore(start), optionally masked)

Backends: "ref" (python/freerun/ref.py, the spec), "cpu" (C++), "cuda" (GPU, when built). All must agree bit-exactly.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .batch import Batch
from .netlist import Circuit
from .ref import RefSim, Window

NEVER = np.iinfo(np.int64).max
NSTATS = 5
NPROF = 3
IO_HIGH, IO_TOGGLES = 1, 2
SNAPSHOT_POWERUP = 0
_LIBDIR = Path(os.environ.get("FREERUN_LIBDIR", Path(__file__).resolve().parent / "_lib"))


_DESC_ARRAYS = ("net_off", "gate_off", "edge_off", "n_inputs", "fanin_ptr", "fanin", "dr", "df", "tt", "init", "fo_ptr",
                "fo_edge", "edge_gate")
_JIT_ARRAYS = ("jit_seed", "jit_cid", "jit_dmin", "jit_amp")  # NULL when no circuit in the batch has jitter


class _BatchDesc(ctypes.Structure):
    _fields_ = [("n_circuits", ctypes.c_int64)] + [(name, ctypes.c_void_p) for name in _DESC_ARRAYS + _JIT_ARRAYS]


class _Stim(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in ("stim_off", "t", "net", "v")]


class _Steps(ctypes.Structure):
    _fields_ = [("K", ctypes.c_void_p), ("S", ctypes.c_void_p), ("budget", ctypes.c_void_p),
                ("offset", ctypes.c_int64), ("s_max", ctypes.c_int64), ("flags", ctypes.c_int)]


class _StepsOut(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in ("val", "high", "tog", "events", "stop")]


STEPS_HOLD_AT_END = 1


def _ptr(a):
    return None if a is None else a.ctypes.data_as(ctypes.c_void_p)


_libs: dict[str, ctypes.CDLL] = {}


def load_library(backend: str) -> ctypes.CDLL:
    if backend not in _libs:
        path = _LIBDIR / f"libfreerun_{backend}.so"
        if not path.exists():
            raise RuntimeError(f"{path} not found; build with scripts/build.sh")
        lib = ctypes.CDLL(str(path))
        vp, i64, i32, cint = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32, ctypes.c_int

        def sig(name, args, res=None, optional=False):
            f = getattr(lib, f"fr_{backend}_{name}", None)
            if f is None:
                if optional:  # a library built before this entry point existed
                    return
                raise AttributeError(f"fr_{backend}_{name} not found in {path}")
            f.argtypes, f.restype = args, res

        sig("create", [vp, cint, cint], vp)
        sig("destroy", [vp])
        sig("set_watch", [vp, vp])
        sig("set_trace_capacity", [vp, i64])
        sig("set_net_observables", [vp, cint])
        sig("run", [vp, i64, vp], cint)
        sig("set_io", [vp, i32, i32, vp, cint], cint)
        sig("step", [vp, i64, vp, vp, vp, vp, vp], cint)
        sig("run_steps", [vp, vp, vp, vp], cint, optional=True)
        sig("snapshot", [vp], i64)
        sig("restore", [vp, i64, vp], cint)
        sig("snapshot_free", [vp, i64])
        sig("set_state", [vp, i64, vp, vp, vp, vp], cint)
        sig("time", [vp], i64)
        for name in ("values", "pending", "toggles", "high", "stats", "trace_counts", "counters", "profile"):
            sig(name, [vp, vp])
        sig("trace", [vp, vp, vp, vp])
        sig("error", [vp], ctypes.c_char_p)
        if backend == "cuda":
            sig("set_block_threshold", [vp, i64, cint])
            sig("set_reorder", [vp, cint])
            sig("step_device", [vp, i64, vp, vp, vp, vp, vp, ctypes.c_size_t], cint)
            sig("run_steps_device", [vp, vp, vp, vp, ctypes.c_size_t], cint, optional=True)
        _libs[backend] = lib
    return _libs[backend]


def supports_run_steps(backend: str) -> bool:
    """Whether Simulator.run_steps works on this backend: the reference always, a compiled backend if its library was
    built with the entry point."""
    if backend == "ref":
        return True
    try:
        return hasattr(load_library(backend), f"fr_{backend}_run_steps")
    except (RuntimeError, OSError):
        return False


class StepsResult(NamedTuple):
    """Per-step outputs of Simulator.run_steps (SPEC §13): rows per circuit, then steps (s_max = max S; entries of
    steps >= S_b are 0), then output slots (set_io)."""
    values: object          # uint8 [B, s_max, n_out]: value of each output at the end of the step's window
    high: object = None     # int64 [B, s_max, n_out]: ticks at 1 in the window (if requested in set_io)
    toggles: object = None  # int64 [B, s_max, n_out]: changes in the window (if requested in set_io)
    events: object = None   # int64 [B, s_max]: committed gate-net transitions in the window
    stop: object = None     # int64 [B]: step at which the circuit was frozen over its budget, -1 = none


class StepResult(NamedTuple):
    """Dense outputs of one step, rows per circuit, columns per output slot (see Simulator.set_io)."""
    values: object         # uint8 [B, n_out]: output values at the end of the step
    high: object = None    # int64 [B, n_out]: ticks at 1 during the step (if requested in set_io)
    toggles: object = None  # int64 [B, n_out]: changes during the step (if requested in set_io)


class Snapshot:
    """A saved state of every circuit (device-resident on CUDA). Restore it with Simulator.restore."""

    def __init__(self, sim: "Simulator", key, t: int):
        self._sim, self._key, self.t = sim, key, t

    def free(self):
        if self._key is not None and self._sim is not None:
            self._sim._free_snapshot(self._key)
        self._key = None

    def __del__(self):
        try:
            self.free()
        except Exception:  # interpreter shutdown
            pass


def _cuda_array(a, shape, typestr, what):
    """(device pointer, stream) of an object exposing __cuda_array_interface__, after checking its layout."""
    ai = a.__cuda_array_interface__
    if tuple(ai["shape"]) != tuple(shape) or np.dtype(ai["typestr"]) != np.dtype(typestr):
        raise ValueError(f"{what}: expected a {np.dtype(typestr)} CUDA array of shape {tuple(shape)}, got "
                         f"{np.dtype(ai['typestr'])} {tuple(ai['shape'])}")
    strides = ai.get("strides")
    if strides is not None and tuple(strides) != tuple(np.empty(shape, typestr).strides):
        raise ValueError(f"{what}: CUDA array must be C-contiguous")
    if ai.get("mask") is not None:
        raise ValueError(f"{what}: masked arrays are not supported")
    stream = ai.get("stream")
    return int(ai["data"][0]), (0 if stream in (None, 1, 2) else int(stream))


class Simulator:
    def __init__(self, circuits, backend: str = "cpu", watch="all", stats: bool = False, threads: int = 0,
                 trace_capacity: int = 1 << 16, stepped: bool = False, block_threshold: int | None = None,
                 block_threads: int = 0, smem: bool = True, packed: bool = True, net_observables: bool = True,
                 reorder_every: int | None = None):
        """circuits: a Batch, a Circuit, or a list of Circuits.
        watch: "all", None (no trace), or a list (one entry per circuit) of local net index lists.
        stats: also count ties / cancels / advances (diagnostics).
        trace_capacity: per-circuit trace rows per run window on fixed-buffer backends (cuda).
        stepped: cuda only, use the per-timestamp kernel instead of lookahead windows (testing/comparison).
        block_threshold / block_threads: cuda only, circuits with more gates than the threshold run one per thread
        block of block_threads threads, the others one per warp (default: chosen from batch size and GPU).
        smem: cuda only, allow keeping circuits' working state in shared memory (testing/comparison).
        packed: cuda only, allow the 16-bit packed netlist layout for small circuits (testing/comparison).
        reorder_every: cuda only, reorder circuits by the cost of their last run every this many runs, so that busy
        circuits start first (default 16; 0 = keep the size order).
        net_observables: maintain per-net toggle counts and high time (Window.toggles / Window.high). Closed-loop
        clients that only read their outputs (set_io) turn this off; the Window fields are then zero."""
        if isinstance(circuits, Circuit):
            circuits = [circuits]
        self.batch = circuits if isinstance(circuits, Batch) else Batch(circuits)
        self.backend = backend
        self.stats = stats
        self.net_observables = net_observables
        self.t = 0
        self._t_last = 0  # start of the last run / step
        B = self.batch.B
        self._n_in, self._n_out, self._io_flags = 0, 0, 0
        self._out_net = np.zeros((B, 0), dtype=np.int32)
        self._q: list[tuple[np.ndarray, ...]] = []  # queued stimulus chunks (b, t, net, v), validated
        if watch == "all":
            self._watch = [None] * B
        elif watch is None:
            self._watch = [[] for _ in range(B)]
        else:
            self._watch = [list(w) for w in watch]
        if backend == "ref":
            self._refs = [RefSim(c, w) for c, w in zip(self.batch.circuits, self._watch)]
            self._ref_snaps: dict[int, tuple] = {}
            self._ref_snaps[SNAPSHOT_POWERUP] = self._ref_state()
            self._ref_last: list[Window] = []
            return
        self._lib = load_library(backend)
        b = self.batch
        self._keep = b  # arrays must outlive the call (the engine copies them)
        d = _BatchDesc(B, *[_ptr(getattr(b, n)) for n in _DESC_ARRAYS[:10]],
                       None, None, None,  # fanout: derived by the engine
                       *[_ptr(getattr(b, n)) if b.has_jitter else None for n in _JIT_ARRAYS])
        flags = (1 if stats else 0) | (2 if stepped else 0) | (0 if smem else 4) | (0 if packed else 8)
        self._h = self._fn("create")(ctypes.byref(d), int(threads), flags)
        if not self._h:
            raise RuntimeError(f"{backend} backend failed to initialize")
        self._check(0 if not self._fn("error")(self._h) else 1)
        if backend == "cuda" and (block_threshold is not None or block_threads):
            self._lib.fr_cuda_set_block_threshold(self._h, -1 if block_threshold is None else int(block_threshold),
                                                  int(block_threads))
        if backend == "cuda" and reorder_every is not None:
            self._lib.fr_cuda_set_reorder(self._h, int(reorder_every))
        if watch is not None:
            mask = np.zeros(b.n_nets, dtype=np.uint8)
            for i, w in enumerate(self._watch):
                if w is None:
                    mask[b.nets(i)] = 1
                else:
                    mask[b.net_off[i] + np.asarray(w, dtype=np.int64)] = 1
            self._fn("set_trace_capacity")(self._h, int(trace_capacity))
            self._fn("set_watch")(self._h, _ptr(mask))
        if not net_observables:
            self._fn("set_net_observables")(self._h, 0)

    def _fn(self, name):
        return getattr(self._lib, f"fr_{self.backend}_{name}")

    def _check(self, rc):
        if rc != 0:
            raise RuntimeError(self._fn("error")(self._h).decode())

    def close(self):
        if getattr(self, "_h", None):
            self._fn("destroy")(self._h)
            self._h = None

    def __del__(self):
        self.close()

    # --- stimulus (validated here for every backend; SPEC §4) ---------------------------------------------------
    def add_stimulus(self, b: int, events) -> None:
        """Queue stimulus triples (t, input net, value) for circuit b. Times must be >= the current time."""
        if self.backend == "ref":
            self._refs[b].add_stimulus(events)
            return
        ev = np.asarray(list(events) if not isinstance(events, np.ndarray) else events, dtype=np.int64).reshape(-1, 3)
        self.add_stimulus_arrays(np.full(len(ev), b, dtype=np.int64), ev[:, 0], ev[:, 1], ev[:, 2])

    def add_stimulus_arrays(self, b, t, net, v) -> None:
        """Queue many triples at once: equal-length arrays of circuit, time, local input net and value."""
        b, t, net, v = (np.asarray(x, dtype=np.int64).ravel() for x in (b, t, net, v))
        if not len(b) == len(t) == len(net) == len(v):
            raise ValueError("stimulus arrays must have equal lengths")
        if len(b) == 0:
            return
        if self.backend == "ref":
            for i in range(len(b)):
                self._refs[int(b[i])].add_stimulus([(int(t[i]), int(net[i]), int(v[i]))])
            return
        if b.min() < 0 or b.max() >= self.batch.B:
            raise ValueError("stimulus for a circuit outside the batch")
        bad = (net < 0) | (net >= self.batch.n_inputs[b])
        if bad.any():
            raise ValueError(f"net {int(net[bad][0])} is not a primary input of circuit {int(b[bad][0])}")
        if ((v != 0) & (v != 1)).any():
            raise ValueError("stimulus values must be 0 or 1")
        if t.min() < self.t:
            raise ValueError(f"stimulus at {int(t.min())} is before the current time {self.t}")
        self._q.append((b, t, net, v))

    def _take_stimulus(self, T: int, dense_rows: int):
        """Stimulus with t < T as C arrays sorted by (circuit, t, net); None if there is none. Checks SPEC §4's one
        triple per (net, t), and that no triple drives a dense input at the current time."""
        if not self._q:
            return None
        b, t, n, v = (np.concatenate(x) for x in zip(*self._q))
        o = np.lexsort((n, t, b))
        b, t, n, v = b[o], t[o], n[o], v[o]
        dup = (b[1:] == b[:-1]) & (t[1:] == t[:-1]) & (n[1:] == n[:-1])
        if dup.any():
            i = int(np.nonzero(dup)[0][0])
            raise ValueError(f"two stimulus events for net {int(n[i])} at time {int(t[i])} (circuit {int(b[i])})")
        if dense_rows:
            clash = (t == self.t) & (n < np.minimum(dense_rows, self.batch.n_inputs[b]))
            if clash.any():
                i = int(np.nonzero(clash)[0][0])
                raise ValueError(f"stimulus triple and dense input both drive net {int(n[i])} of circuit {int(b[i])} "
                                 f"at time {self.t}")
        now = t < T
        rest = ~now
        self._q = [(b[rest], t[rest], n[rest], v[rest])] if rest.any() else []
        b, t, n, v = b[now], t[now], n[now], v[now]
        if len(b) == 0:
            return None
        off = np.searchsorted(b, np.arange(self.batch.B + 1)).astype(np.int64)
        return off, t, n.astype(np.int32), v.astype(np.uint8)

    # --- running ------------------------------------------------------------------------------------------------
    def run(self, T: int) -> list[Window]:
        """Process every time < T (SPEC §6); returns one Window of observables per circuit for [previous T, T)."""
        self._advance(T, None, None)
        return self.windows()

    def set_io(self, n_inputs: int | None = None, outputs=None, high: bool = False, toggles: bool = False) -> None:
        """Configure dense closed-loop I/O for step().
        n_inputs: width of the input rows (default: the largest number of primary inputs in the batch). Row b drives
        inputs 0 .. min(n_inputs, circuit b's inputs) - 1.
        outputs: per circuit, a list of local net indices (ragged lists are padded with -1 = unused slot), or an int
        array [B, n_out].
        high / toggles: also report each output's high time / number of changes over the step."""
        B = self.batch.B
        n_in = int(self.batch.n_inputs.max()) if n_inputs is None else int(n_inputs)
        if outputs is None:
            out = np.zeros((B, 0), dtype=np.int32)
        elif isinstance(outputs, np.ndarray):
            out = np.ascontiguousarray(outputs, dtype=np.int32).reshape(B, -1)
        else:
            outputs = list(outputs)
            if len(outputs) != B:
                raise ValueError("outputs needs one list per circuit")
            w = max((len(o) for o in outputs), default=0)
            out = np.full((B, w), -1, dtype=np.int32)
            for i, o in enumerate(outputs):
                out[i, :len(o)] = o
        n_nets = np.diff(self.batch.net_off)[:, None]
        if n_in < 0 or ((out < -1) | (out >= n_nets)).any():
            raise ValueError("set_io: bad input width or output net")
        flags = (IO_HIGH if high else 0) | (IO_TOGGLES if toggles else 0)
        if self.backend != "ref":
            self._check(self._fn("set_io")(self._h, n_in, out.shape[1], _ptr(out), flags))
        self._n_in, self._n_out, self._io_flags, self._out_net = n_in, out.shape[1], flags, out

    def step(self, T: int, inputs=None, out: StepResult | None = None) -> StepResult:
        """Run to T after applying a dense input row per circuit at the current time (see set_io); return the dense
        outputs. Records the same per-run observables as run() (fetch them with windows()).

        inputs: uint8 [B, n_inputs] (a NumPy array, or any CUDA array exposing __cuda_array_interface__ on the cuda
        backend), or None to leave the inputs unchanged.
        out: optional preallocated StepResult; with CUDA arrays, the outputs are written on the device (zero-copy)."""
        B = self.batch.B
        device = self.backend == "cuda" and (hasattr(inputs, "__cuda_array_interface__") or (
            out is not None and hasattr(out.values, "__cuda_array_interface__")))
        if device:
            return self._step_device(int(T), inputs, out)
        if inputs is not None:
            inputs = np.ascontiguousarray(inputs, dtype=np.uint8)
            if inputs.shape != (B, self._n_in):
                raise ValueError(f"inputs must have shape {(B, self._n_in)}, got {inputs.shape}")
            if (inputs > 1).any():
                raise ValueError("input values must be 0 or 1")
        shape = (B, self._n_out)
        if out is None:
            out = StepResult(np.empty(shape, np.uint8), np.empty(shape, np.int64) if self._io_flags & IO_HIGH else None,
                             np.empty(shape, np.int64) if self._io_flags & IO_TOGGLES else None)
        for a, dt in zip(out, (np.uint8, np.int64, np.int64)):
            if a is not None and (a.shape != shape or a.dtype != dt or not a.flags.c_contiguous):
                raise ValueError(f"out arrays must be C-contiguous {np.dtype(dt)} of shape {shape}")
        if out.high is not None and not self._io_flags & IO_HIGH or out.toggles is not None and \
                not self._io_flags & IO_TOGGLES:
            raise ValueError("high / toggles outputs were not requested in set_io")
        self._advance(int(T), inputs, out)
        return out

    def run_steps(self, K, S, stimulus=None, budget=None, offset: int = 0, hold_at_end: bool = False,
                  out: StepsResult | None = None) -> StepsResult:
        """A whole open-loop episode in one engine call (SPEC §13). Circuit b runs S[b] steps of K[b] ticks; step s
        observes the window [t + offset + s K, t + offset + (s+1) K), t = the current time. Every time before
        T_end = t + offset + max(S K) is processed exactly as by run(T_end), or by step() calls cut at the boundaries,
        and the per-step outputs of set_io's output slots come back at once.

        K, S: ints or int arrays [B]. stimulus: None or (off [B+1], t, net, v): triples per circuit (circuit b's are
        off[b] .. off[b+1]-1) with times in [t, T_end); triples queued with add_stimulus before T_end are included.
        budget: None, or a number or array [B] (negative, inf or nan = none): a circuit whose committed gate-net
        transitions in one step exceed floor(budget) is frozen right after the time at which that happened (pending
        transitions dropped, its remaining stimulus of this call discarded); that step reports the counts up to the
        freeze and the held state, later steps report 0, and StepsResult.stop says which step it was.
        hold_at_end: freeze every circuit the same way at the end of its own last step (no flag); saves the work of
        circuits whose episode is shorter than the longest one.
        out: optional StepsResult of preallocated arrays (CUDA arrays on the cuda backend: written on the device, no
        host copies). Afterwards the current time is T_end and windows() covers [t, T_end)."""
        B = self.batch.B
        K = np.ascontiguousarray(np.broadcast_to(np.asarray(K, dtype=np.int64), (B,)))
        S = np.ascontiguousarray(np.broadcast_to(np.asarray(S, dtype=np.int64), (B,)))
        offset = int(offset)
        if (K < 1).any() or (S < 0).any() or offset < 0:
            raise ValueError("run_steps needs K >= 1, S >= 0 and offset >= 0")
        bud = None
        if budget is not None:
            bf = np.broadcast_to(np.asarray(budget, dtype=np.float64), (B,))
            ok = np.isfinite(bf) & (bf >= 0)
            bud = np.ascontiguousarray(np.where(ok, np.floor(np.minimum(np.where(ok, bf, 0), 2.0**62)), -1)
                                       .astype(np.int64))
        s_max = int(S.max(initial=0))
        T_end = self.t + offset + int((S * K).max(initial=0))
        if stimulus is not None and len(stimulus[1]) and int(np.max(stimulus[1])) >= T_end:
            raise ValueError(f"run_steps: stimulus at {int(np.max(stimulus[1]))} is at or after the run's end {T_end}")
        shape3, shape2 = (B, s_max, self._n_out), (B, s_max)
        if self.backend == "ref":
            if stimulus is not None:
                self._queue_stimulus_arrays(*stimulus)
            return self._ref_run_steps(K, S, bud, offset, hold_at_end, T_end, out)
        if not hasattr(self._lib, f"fr_{self.backend}_run_steps"):
            raise RuntimeError(f"the {self.backend} library was built without run_steps; rebuild (scripts/build.sh)")
        st = self._sorted_stimulus(stimulus, T_end) if stimulus is not None and not self._q else None
        if st is None:
            if stimulus is not None:
                self._queue_stimulus_arrays(*stimulus)
            st = self._take_stimulus(T_end, 0)
        elif st is False:
            st = None
        stim = _Stim(*[_ptr(a) for a in st]) if st else None
        desc = _Steps(_ptr(K), _ptr(S), _ptr(bud), offset, s_max, STEPS_HOLD_AT_END if hold_at_end else 0)
        shapes = (shape3, shape3, shape3, shape2, (B,))
        if self.backend == "cuda" and out is not None and hasattr(out.values, "__cuda_array_interface__"):
            ptrs, stream = [], 0
            for a, shape, ts, name in zip(out, shapes, ("u1", "<i8", "<i8", "<i8", "<i8"), StepsResult._fields):
                if a is None:
                    ptrs.append(None)
                    continue
                p, stream = _cuda_array(a, shape, ts, f"out.{name}")
                ptrs.append(p)
            o = _StepsOut(*ptrs)
            rc = self._lib.fr_cuda_run_steps_device(self._h, ctypes.byref(stim) if stim else None, ctypes.byref(desc),
                                                    ctypes.byref(o), stream)
            self._check(rc)
            self._t_last, self.t = self.t, T_end
            return out
        if out is None:
            out = StepsResult(np.zeros(shape3, np.uint8),
                              np.zeros(shape3, np.int64) if self._io_flags & IO_HIGH else None,
                              np.zeros(shape3, np.int64) if self._io_flags & IO_TOGGLES else None,
                              np.zeros(shape2, np.int64), np.zeros(B, np.int64))
        for a, shape, dt in zip(out, shapes, (np.uint8, np.int64, np.int64, np.int64, np.int64)):
            if a is not None and (a.shape != shape or a.dtype != dt or not a.flags.c_contiguous):
                raise ValueError(f"out arrays must be C-contiguous {np.dtype(dt)} of shape {shape}")
        o = _StepsOut(*[_ptr(a) for a in out])
        rc = self._fn("run_steps")(self._h, ctypes.byref(stim) if stim else None, ctypes.byref(desc), ctypes.byref(o))
        self._check(rc)
        self._t_last, self.t = self.t, T_end
        return out

    def _ref_run_steps(self, K, S, bud, offset, hold, T_end, out):
        B = self.batch.B
        s_max, n_out = int(S.max(initial=0)), self._n_out
        res = StepsResult(np.zeros((B, s_max, n_out), np.uint8), np.zeros((B, s_max, n_out), np.int64),
                          np.zeros((B, s_max, n_out), np.int64), np.zeros((B, s_max), np.int64),
                          np.zeros(B, np.int64))
        wins = []
        for b, r in enumerate(self._refs):
            w, v, h, tg, ev, stop = r.run_steps(T_end, self.t + offset, int(K[b]), int(S[b]), self._out_net[b],
                                                -1 if bud is None else int(bud[b]), hold)
            Sb = int(S[b])
            res.values[b, :Sb], res.high[b, :Sb], res.toggles[b, :Sb], res.events[b, :Sb] = v, h, tg, ev
            res.stop[b] = stop
            wins.append(w)
        if not self.net_observables:
            for w in wins:
                w.toggles[:] = 0
                w.high[:] = 0
        self._ref_last = wins
        self._t_last, self.t = self.t, T_end
        res = res._replace(high=res.high if self._io_flags & IO_HIGH else None,
                           toggles=res.toggles if self._io_flags & IO_TOGGLES else None)
        if out is None:
            return res
        for a, r in zip(out, res):
            if a is not None and r is not None:
                a[...] = r
        return out

    def _sorted_stimulus(self, stimulus, T_end):
        """The given per-circuit triples as engine arrays without sorting them, if they are already sorted by
        (circuit, t, net) with one triple per (net, t) and valid (SPEC §4) with times in [t, T_end): else None (the
        caller then queues and sorts them). False: valid and empty."""
        off, t, net, v = stimulus
        off = np.ascontiguousarray(off, dtype=np.int64)
        t = np.ascontiguousarray(t, dtype=np.int64)
        net = np.ascontiguousarray(net, dtype=np.int32)
        v = np.ascontiguousarray(v, dtype=np.uint8)
        B = self.batch.B
        if off.shape != (B + 1,) or off[0] != 0 or off[-1] != len(t) or not len(t) == len(net) == len(v) or \
                (np.diff(off) < 0).any():
            return None
        if len(t) == 0:
            return False
        b = np.repeat(np.arange(B, dtype=np.int64), np.diff(off))
        if t.min() < self.t or t.max() >= T_end or net.min() < 0 or (net >= self.batch.n_inputs[b]).any() or \
                v.max() > 1:
            return None
        same = b[1:] == b[:-1]
        if (same & ((t[1:] < t[:-1]) | ((t[1:] == t[:-1]) & (net[1:] <= net[:-1])))).any():
            return None
        return off, t, net, v

    def _queue_stimulus_arrays(self, off, t, net, v) -> None:
        """Queue triples given per circuit (off [B+1] into t, net, v), validated like add_stimulus_arrays."""
        off = np.asarray(off, dtype=np.int64)
        if off.shape != (self.batch.B + 1,) or off[0] != 0 or (np.diff(off) < 0).any() or off[-1] != len(t):
            raise ValueError("stimulus offsets must be [B+1], start at 0, be non-decreasing and end at len(t)")
        b = np.repeat(np.arange(self.batch.B, dtype=np.int64), np.diff(off))
        self.add_stimulus_arrays(b, t, net, v)

    def _step_device(self, T, inputs, out):
        B = self.batch.B
        if out is None:
            raise ValueError("a device step needs preallocated CUDA output arrays (out=StepResult(...))")
        shape = (B, self._n_out)
        in_p, stream = (0, 0) if inputs is None else _cuda_array(inputs, (B, self._n_in), "u1", "inputs")
        ptrs = []
        for a, ts, name in zip(out, ("u1", "<i8", "<i8"), ("values", "high", "toggles")):
            ptrs.append(0 if a is None else _cuda_array(a, shape, ts, f"out.{name}")[0])
        if T < self.t:
            raise ValueError(f"cannot run backwards from {self.t} to {T}")
        st = self._take_stimulus(T, self._n_in if inputs is not None else 0)
        s = _Stim(*[_ptr(a) for a in st]) if st else None
        rc = self._lib.fr_cuda_step_device(self._h, T, ctypes.byref(s) if s else None, in_p or None,
                                           ptrs[0] or None, ptrs[1] or None, ptrs[2] or None, stream)
        self._check(rc)
        self._t_last, self.t = self.t, T
        return out

    def _advance(self, T: int, inputs, out):
        T = int(T)
        if T < self.t:
            raise ValueError(f"cannot run backwards from {self.t} to {T}")
        if self.backend == "ref":
            self._ref_advance(T, inputs, out)
            return
        st = self._take_stimulus(T, self._n_in if inputs is not None else 0)
        s = _Stim(*[_ptr(a) for a in st]) if st else None
        o = out if out is not None else StepResult(None)
        rc = self._fn("step")(self._h, T, ctypes.byref(s) if s else None, _ptr(inputs), _ptr(o.values), _ptr(o.high),
                              _ptr(o.toggles))
        self._check(rc)
        self._t_last, self.t = self.t, T

    def _ref_advance(self, T, inputs, out):
        if inputs is not None:
            for b, r in enumerate(self._refs):
                k = min(self._n_in, r.c.n_inputs)
                r.add_stimulus([(self.t, i, int(inputs[b, i])) for i in range(k)])
        wins = [r.run(T) for r in self._refs]
        if out is not None:
            for b, (r, w) in enumerate(zip(self._refs, wins)):
                for j, n in enumerate(self._out_net[b]):
                    ok = n >= 0
                    if out.values is not None:
                        out.values[b, j] = r.val[n] if ok else 0
                    if out.high is not None:
                        out.high[b, j] = w.high[n] if ok else 0
                    if out.toggles is not None:
                        out.toggles[b, j] = w.toggles[n] if ok else 0
        if not self.net_observables:
            for w in wins:
                w.toggles[:] = 0
                w.high[:] = 0
        self._ref_last = wins
        self._t_last, self.t = self.t, T

    def _get(self, name, n, dtype):
        out = np.empty(n, dtype=dtype)
        self._fn(name)(self._h, _ptr(out))
        return out

    def windows(self) -> list[Window]:
        """Observables of the last run / step, one Window per circuit."""
        if self.backend == "ref":
            return self._ref_last
        b = self.batch
        t0, T = self._t_last, self.t
        toggles = self._get("toggles", b.n_nets, np.int64)
        high = self._get("high", b.n_nets, np.int64)
        stats = self._get("stats", b.B * NSTATS, np.int64).reshape(b.B, NSTATS)
        values = self._get("values", b.n_nets, np.uint8)
        counts = self._get("trace_counts", b.B, np.int64)
        M = int(counts.sum())
        tt, tn, tv = np.empty(M, np.int64), np.empty(M, np.int32), np.empty(M, np.uint8)
        self._fn("trace")(self._h, _ptr(tt), _ptr(tn), _ptr(tv))
        rows = np.stack([tt, tn.astype(np.int64), tv.astype(np.int64)], axis=1)
        ends = np.cumsum(counts).tolist()
        no = b.net_off.tolist()
        st = stats.tolist()
        wins = []
        for i in range(b.B):  # views into the freshly fetched arrays
            ev, ties, first_tie, cancels, advances = st[i]
            wins.append(Window(t0, T, rows[ends[i] - int(counts[i]):ends[i]], toggles[no[i]:no[i + 1]],
                               high[no[i]:no[i + 1]], ties, first_tie, ev, cancels, advances, values[no[i]:no[i + 1]]))
        return wins

    def events(self) -> np.ndarray:
        """Net changes per circuit in the last run / step (int64 [B])."""
        if self.backend == "ref":
            return np.array([w.events for w in self._ref_last], dtype=np.int64)
        return self._get("stats", self.batch.B * NSTATS, np.int64).reshape(-1, NSTATS)[:, 0].copy()

    def profile(self) -> dict:
        """Timing of the last run / step inside the engine: call_s (wall), kernel_s (simulation), windows."""
        if self.backend == "ref":
            return dict(call_s=float("nan"), kernel_s=float("nan"), windows=0)
        p = self._get("profile", NPROF, np.float64)
        return dict(call_s=float(p[0]), kernel_s=float(p[1]), windows=int(p[2]))

    # --- state --------------------------------------------------------------------------------------------------
    def values(self, b: int) -> np.ndarray:
        if self.backend == "ref":
            return self._refs[b].values()
        return self._get("values", self.batch.n_nets, np.uint8)[self.batch.nets(b)]

    def pending(self, b: int) -> np.ndarray:
        """Pending transition time per gate of circuit b, -1 for none."""
        if self.backend == "ref":
            return self._refs[b].pending()
        p = self._get("pending", self.batch.n_gates, np.int64)[self.batch.gates(b)]
        return np.where(p == NEVER, -1, p)

    def counters(self, b: int) -> np.ndarray:
        """Per-gate transition counters n_g of circuit b (SPEC §10.2); all 0 for a circuit without jitter."""
        if self.backend == "ref":
            return self._refs[b].counters()
        return self._get("counters", self.batch.n_gates, np.uint64)[self.batch.gates(b)]

    def _ref_state(self):
        return (self.t, [r.values() for r in self._refs], [r.pending() for r in self._refs],
                [r.counters() for r in self._refs])

    def state(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """(time, values [all nets], pending [all gates], -1 = none, transition counters [all gates]) of the whole
        batch, as host arrays."""
        if self.backend == "ref":
            t, v, p, n = self._ref_state()
            return t, np.concatenate(v), np.concatenate(p), np.concatenate(n)
        p = self._get("pending", self.batch.n_gates, np.int64)
        return (self.t, self._get("values", self.batch.n_nets, np.uint8), np.where(p == NEVER, -1, p),
                self._get("counters", self.batch.n_gates, np.uint64))

    def _mask(self, mask):
        if mask is None:
            return None
        m = np.asarray(mask)
        if m.dtype != bool:  # circuit indices
            sel = np.zeros(self.batch.B, dtype=bool)
            sel[m.astype(np.int64)] = True
            m = sel
        if m.shape != (self.batch.B,):
            raise ValueError("mask must be a boolean array [B] or a list of circuit indices")
        return np.ascontiguousarray(m, dtype=np.uint8)

    def load_state(self, state, mask=None) -> None:
        """Load a state from state() (possibly of another Simulator or backend over the same batch) into the selected
        circuits (mask: bool [B] or circuit indices; None = all), translated to the current time (SPEC §6). The
        transition counters are optional (default 0)."""
        t_ref, val, pend = state[:3]
        cnt = state[3] if len(state) > 3 else None
        val = np.ascontiguousarray(val, dtype=np.uint8)
        pend = np.asarray(pend, dtype=np.int64)
        cnt = None if cnt is None else np.ascontiguousarray(cnt, dtype=np.uint64)
        if val.shape != (self.batch.n_nets,) or pend.shape != (self.batch.n_gates,) or \
                (cnt is not None and cnt.shape != (self.batch.n_gates,)):
            raise ValueError("state arrays do not match the batch")
        m = self._mask(mask)
        if self.backend == "ref":
            for b, r in enumerate(self._refs):
                if m is None or m[b]:
                    g = self.batch.gates(b)
                    r.load_state(t_ref, val[self.batch.nets(b)], pend[g], None if cnt is None else cnt[g])
            return
        pend = np.ascontiguousarray(np.where(pend < 0, NEVER, pend), dtype=np.int64)
        self._check(self._fn("set_state")(self._h, int(t_ref), _ptr(val), _ptr(pend), _ptr(cnt), _ptr(m)))

    def snapshot(self) -> Snapshot:
        """Save every circuit's state (on the device for CUDA)."""
        if self.backend == "ref":
            key = max(self._ref_snaps) + 1
            self._ref_snaps[key] = self._ref_state()
            return Snapshot(self, key, self.t)
        key = int(self._fn("snapshot")(self._h))
        if key < 0:
            self._check(1)
        return Snapshot(self, key, self.t)

    def restore(self, snap: Snapshot | int, mask=None) -> None:
        """Restore a snapshot into the selected circuits (mask: bool [B] or circuit indices; None = all). The saved
        state is translated to the current time; time does not move backwards (SPEC §6)."""
        key = snap if isinstance(snap, int) else snap._key
        if key is None:
            raise ValueError("snapshot was freed")
        m = self._mask(mask)
        if self.backend == "ref":
            t_ref, vals, pends, cnts = self._ref_snaps[key]
            for b, r in enumerate(self._refs):
                if m is None or m[b]:
                    r.load_state(t_ref, vals[b], pends[b], cnts[b])
            return
        self._check(self._fn("restore")(self._h, key, _ptr(m)))

    def reset(self, mask=None) -> None:
        """Return the selected circuits (default: all) to their power-up state, at the current time."""
        self.restore(SNAPSHOT_POWERUP, mask)

    def _free_snapshot(self, key):
        if self.backend == "ref":
            self._ref_snaps.pop(key, None)
        elif getattr(self, "_h", None):
            self._fn("snapshot_free")(self._h, key)
