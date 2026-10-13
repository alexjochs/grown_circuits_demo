# freerun

Batched, event-driven simulation of **free-running (clockless, cyclic) gate-level circuits**, on GPU and CPU.

- **Arbitrary feedback.** Ring oscillators, cross-coupled latches and combinational loops. There is no clock and
  no levelization.
- **Exact, specified semantics.** The specification defines two-state values, integer time, and
  per-arc rise/fall inertial delays with Verilog-style pulse rejection. Every backend reproduces the reference
  bit-exactly. The reference itself is cross-checked against Icarus Verilog, GHDL and nvc.
- **Many circuits at once.** The batch is the parallel axis: thousands of *different* netlists per GPU, each from a
  few gates to (eventually) millions.
- **Real cell libraries.** Liberty files (SKY130 `sky130_fd_sc_hd` first) provide truth tables and per-instance
  NLDM delays from load and slew. Gate-level Verilog netlists can be read directly.
- **Device populations.** One batch can span process corners (ss/tt/ff, with temperature and voltage
  interpolated between characterized libraries), die-to-die scale, per-cell mismatch, and per-transition jitter.
  Everything is reproducible from `(seed, circuit id)`, and the jitter is bit-exact on every backend.
- **Idealized twins.** Each device has a twin with the same topology and nominal or uniform delays, for measuring
  how much a behavior depends on physical detail.
- **Energy.** Switching, internal and leakage energy per circuit from Liberty data, validated against ngspice.

Status: early development (0.2.0).

This copy is vendored in the Grown Circuits demo repository without freerun's test suite, reference tools and
design documents.

## Quick start

```bash
pip install -e .
scripts/build.sh                # CPU backend; add -DFREERUN_CUDA=ON on a machine with CUDA
scripts/build_tools.sh          # optional: reference tools into tools/ (Icarus, GHDL, nvc, ngspice, SKY130 PDK)
```

```python
from freerun import Builder, gen
from freerun.netlist import NAND2
from freerun.sim import Simulator

# A cross-coupled NAND latch plus 1000 random cyclic circuits, simulated together
b = Builder()
s_n, r_n = b.input(1), b.input(1)
q, qn = b.gate(NAND2, k=2), b.gate(NAND2, k=2, init=1)
b.connect(q, [s_n, qn], rise=40, fall=30)
b.connect(qn, [r_n, q], rise=40, fall=30)
circuits = [b.build()] + [gen.fuzz_case(i)[0] for i in range(1000)]

sim = Simulator(circuits, backend="cpu")       # or "cuda", or "ref" (the executable spec)
sim.add_stimulus(0, [(100, 0, 0), (200, 0, 1)])  # circuit 0: pulse S_n low at t = 100..200
win = sim.run(1000)                             # one Window per circuit for [0, 1000)
print(win[0].trace)                             # rows (t, net, value)
```

Closed-loop stepping: the batch state stays in the engine (on the GPU for `cuda`). Each step uploads only the new
input rows and downloads only the chosen outputs.

```python
import numpy as np

sim = Simulator(circuits, backend="cuda", watch=None, net_observables=False)
sim.set_io(n_inputs=2, outputs=[[c.n_nets - 1] for c in circuits], high=True)  # one output net per circuit
warm = sim.snapshot()                           # device-resident copy of every circuit's state
dt = 1_000_000                                  # 1 us per step at 1 tick = 1 fs
for k in range(1000):
    x = np.random.randint(0, 2, size=(len(circuits), 2), dtype=np.uint8)
    r = sim.step(sim.t + dt, x)                 # inputs applied now, run to now + dt
    r.values, r.high                            # [B, 1]: output at the end of the step, ticks at 1 during it
sim.reset(mask=[0, 5])                          # circuits 0 and 5 back to power-up, at the current time
sim.restore(warm)                               # every circuit back to the snapshot, translated in time
```

`step` also takes CUDA arrays (anything with `__cuda_array_interface__`: CuPy, PyTorch, Numba) for the inputs, and
a preallocated `StepResult` of CUDA arrays for the outputs, with no host copies. Restore and reset translate the
saved state to the current time (SPEC §6); `state()` / `load_state()` move a state between simulators and backends.
`bench/closed_loop.py` measures this loop.

Open loop: when every input change of an episode is known up front, `run_steps` runs all of its steps in one engine
call (one kernel launch per segment on CUDA) and returns every step's outputs, instead of one `step` call per step
(SPEC §13). A step is a window of K ticks; inputs are stimulus triples; an optional event budget stops circuits whose
gate transitions in one step exceed it.

```python
sim = Simulator(circuits, backend="cuda", watch=None, net_observables=False)
sim.set_io(outputs=[[c.n_nets - 1] for c in circuits], high=True)
off = np.array([0, 2] + [2] * (len(circuits) - 1))           # circuit 0: input 0 high at t = 5e5, low at 2e6
r = sim.run_steps(K=1_000_000, S=64, stimulus=(off, [500_000, 2_000_000], [0, 0], [1, 0]), budget=10_000)
r.values, r.high, r.events, r.stop  # [B, 64, 1] values at each step's end, ticks at 1; [B, 64] gate events; [B]
```

Standard cells:

```python
from freerun import liberty, verilog, cells
lib = liberty.load("tools/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib")
ann = cells.to_circuit(verilog.read(open("netlist.v").read()), lib, tick=1e-15)  # 1 tick = 1 fs
ann.circuit, ann.net_index["out"]
```

A population of devices, their idealized twin, energy and a waveform dump:

```python
from freerun import energy, vcd
from freerun.variation import Box, Design, LibraryFamily, ideal_twin, population

fam = LibraryFamily("tools/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib")      # all ss/tt/ff corner libraries
design = Design(verilog.read(open("netlist.v").read()), fam)
devs = population(design, 1000, seed=1,
                  corners=[Box("ss", (-40, 100), (1.40, 1.60)), Box("tt", (25, 100), 1.80),
                           Box("ff", (-40, 100), (1.65, 1.95))],
                  scale_sigma=0.03, mismatch_sigma=0.05, jitter_rel=0.01)  # all reproducible from (seed, cid)
sim = Simulator([d.circuit for d in devs] + [ideal_twin(devs[0])], backend="cpu")
wins = sim.run(10_000_000)
print(energy.energy(wins[0], devs[0].energy).total)                         # joules
vcd.write("device0.vcd", wins[0], names=design.nominal.net_names)
```

`examples/idealization_gap.py` runs this for a ring oscillator and a latch and prints how far the devices are from
their twins.

## Layout

| path | contents |
|---|---|
| `python/freerun/ref.py` | reference simulator, the executable spec |
| `python/freerun/windowed.py` | lookahead-window formulation used by the parallel backends, proven exact |
| `csrc/` | C ABI (`freerun.h`), CPU engine, CUDA engine (windowed and per-timestamp kernels), jitter |
| `python/freerun/{liberty,cells,verilog}.py` | cell libraries, delay and energy annotation, netlist reader |
| `python/freerun/{jitter,variation,energy}.py` | per-transition jitter, corners and device variation with idealized twins, energy |
| `python/freerun/{hdl,oracle,spice,vcd}.py` | Verilog and VHDL export, HDL cross-check runners, ngspice harness, VCD export |
| `scripts/` | oracle fuzzing campaigns, energy vs ngspice, corner-interpolation hold-out |
| `examples/` | the idealization gap of a ring oscillator and a latch |
| `bench/bench.py`, `bench/closed_loop.py` | throughput and closed-loop stepping benchmarks |
| `hpc/` | cluster scripts (OSU HPC, single H200) |
