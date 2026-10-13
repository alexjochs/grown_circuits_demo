#!/usr/bin/env python
"""Small CUDA run used under compute-sanitizer / ncu: a few fuzz circuits through every CUDA kernel, the stepping API
(dense I/O, snapshot, restore, reset) and a multi-step run with budget stops (run_steps), compared to CPU."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from freerun import gen  # noqa: E402
from freerun.sim import Simulator  # noqa: E402

cases = [gen.fuzz_case(i, max_gates=200) for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 6)]
circuits = [c for c, *_ in cases]
if "jitter" in sys.argv[2:]:  # per-transition jitter (SPEC §10) on every other circuit
    from freerun.jitter import sigma_relative  # noqa: E402
    from freerun.netlist import Jitter  # noqa: E402
    circuits = [c.with_jitter(Jitter.from_sigma(sigma_relative(c, 0.2), seed=i, cid=i)) if i % 2 == 0 else c
                for i, c in enumerate(circuits)]
B = len(circuits)
T = max(t for *_, t, _ in cases)
rng = np.random.default_rng(0)
xs = [rng.integers(0, 2, size=(B, 3), dtype=np.uint8) for _ in range(6)]
outs = [[0, c.n_nets - 1, -1] for c in circuits]
results = {}
CONFIGS = [("cpu", dict(backend="cpu")), ("warp", dict(backend="cuda")),
           ("block-smem", dict(backend="cuda", block_threshold=0, block_threads=64)),
           ("block-global", dict(backend="cuda", block_threshold=0, block_threads=64, smem=False)),
           ("generic-global", dict(backend="cuda", packed=False, smem=False)),
           ("generic-block", dict(backend="cuda", packed=False, block_threshold=0, block_threads=64)),
           ("stepped", dict(backend="cuda", stepped=True))]
if "cpu-only" in sys.argv[2:]:  # check the script itself without a GPU
    CONFIGS = CONFIGS[:1]
for name, kw in CONFIGS:
    if name != "cpu":
        kw["trace_capacity"] = 1 << 21
    s = Simulator(circuits, stats=True, **kw)
    for b, (_, st, _, _) in enumerate(cases):
        s.add_stimulus(b, st)
    ws = s.run(T // 2) + s.run(T)
    res = [(w.trace.tobytes(), w.events, w.ties, w.cancels) for w in ws] + \
        [s.counters(b).tobytes() for b in range(len(circuits))]
    s.set_io(3, outs, high=True, toggles=True)
    dt = max(1, T // 20)
    snap = s.snapshot()
    for k, x in enumerate(xs):
        if k == 3:
            s.restore(snap, mask=np.arange(B) % 2 == 0)
        if k == 4:
            s.reset([0])
        r = s.step(s.t + dt, x)
        res.append((r.values.tobytes(), r.high.tobytes(), r.toggles.tobytes()))
    # a multi-step run (SPEC §13) from here: per-circuit step lengths, a budget that stops some circuits, hold at end
    rng2 = np.random.default_rng(1)
    K = np.array([max(1, T // (40 + 7 * b)) for b in range(B)], np.int64)
    S = np.array([12 - (b % 3) for b in range(B)], np.int64)
    budget = np.array([-1 if b % 2 else 3 + b for b in range(B)], np.int64)
    span = int((S * K).max()) + 1
    trip = [sorted({(s.t + int(u), int(rng2.integers(0, c.n_inputs))): int(v)
                    for u, v in zip(rng2.integers(0, span, 8), rng2.integers(0, 2, 8))}.items()) for c in circuits]
    off = np.concatenate([[0], np.cumsum([len(x) for x in trip])]).astype(np.int64)
    rows = np.array([(t, n, v) for x in trip for (t, n), v in x], np.int64).reshape(-1, 3)
    rs = s.run_steps(K, S, (off, rows[:, 0], rows[:, 1], rows[:, 2]), budget=budget, offset=1, hold_at_end=True)
    res.append(tuple(a.tobytes() for a in rs))
    _, v_, p_, n_ = s.state()
    res.append(v_.tobytes() + p_.tobytes() + n_.tobytes())
    results[name] = res + [s.counters(b).tobytes() for b in range(len(circuits))]
ok = all(results[k] == results["cpu"] for k in results)
print("small_cuda_run:", "all kernels agree with cpu" if ok else "MISMATCH", flush=True)
sys.exit(0 if ok else 1)
