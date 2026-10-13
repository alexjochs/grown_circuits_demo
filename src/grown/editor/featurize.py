"""Circuit -> padded numpy arrays for the editor model, and Move <-> factorized action indices.

Tokens. Index 0 is a global CLS token (it carries the behaviour signature and task), indices 1..n_in are the
primary inputs, and 1 + n_in + i is node i. Per token: a float feature vector (op kind, arity, truth-table bits,
init, drive, delay statistics, output and liveness marks, module statistics) and two integer ids (input index,
module hash bucket). Wiring is a sparse edge list (dst token, src token, pin) that the model turns into per-pin
message passing and an attention bias.

Pins and ports. Pin slot 0 is the pseudo-pin NONE (a dangling insertion, or no redirect); slots 1..P are
`circuit.pins()` in order. Port slot 0 is the pseudo-port NEW (port 0 of the token being added); slots 1..Q are
`circuit.ports()` in order. Putting the pseudo-entries at index 0 keeps every index stable under end padding.

The factorization of a Move (every MoveType of moves.py; see model.py for the heads):

  type                                  categorical over the applicable types (the operator's `applicable`)
  TT_REPLACE   node ptr -> 2^k Bernoulli bits of the new table
  TT_FLIP      node ptr -> row in [0, 2^k)
  REWIRE       pin ptr  -> source port ptr (old source masked)
  ADD_NODE     splice ptr (NONE or pin) -> kind {gate, delay, wire} -> arity (gate) -> 2^k tt bits (gate) ->
               source port per free pin (port NEW = the new node itself) -> rise/fall value per arc -> init
  DELETE_NODE  node ptr -> fallback port ptr (ports of the node masked)
  DUPLICATE    anchor ptr (= min of the set) -> inclusion Bernoulli for each later node, truncated to the size
               budget (exactly normalized) -> redirect ptr over NONE and the external pins reading the set
  INSERT_MODULE module ptr (library.active()) -> out port -> splice ptr -> source port per free module pin
  ADJUST_PARAM node ptr -> option (rise pin p, fall pin p, init, drive, slot s) -> value (init toggles; delays
               and drive: value bin then uniform inside the bin)
  REGENERATE   anchor ptr -> inclusion Bernoulli (the region), as DUPLICATE without redirect
  INLINE_MODULE node ptr (module instances)

Values (delays, drive) use one shared set of bins: 1..8 exactly, then geometric with ratio sqrt(2) up to
MAX_DELAY. A value's log-probability is log p(bin) - log |bin ∩ allowed range|.

Representation limits (moves beyond them are not representable and are skipped in datasets; the model never
samples them): module instances with more than MODPIN inputs are not offered for insertion, out ports beyond
OUTPORT, parameter slots beyond SLOTMAX, and new nodes always get drive 1 (as the uniform operator does).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..circuit import codec
from ..circuit.graph import DELAY, GATE, INPUT, MAX_DELAY, MODULE, OUT, WIRE, Circuit, live_nodes
from ..circuit.moves import DRIVE, FALL, INIT, NONE, RISE, SLOT, MoveType, Move, N_MOVES

CLS_K, INPUT_K, GATE_K, DELAY_K, WIRE_K, MODULE_K = range(6)
N_KIND = 6
PINMAX = 8           # pin / port position embeddings (higher positions share the last)
OUTMAX = 8           # circuit output index embeddings
NEWPIN = 6           # ADD_NODE pins (MAX_ARITY)
ARITY = 7            # ADD_NODE gate arity classes 0..6
MODPIN = 16          # INSERT_MODULE pins modelled
OUTPORT = 8          # INSERT_MODULE out-port classes
SLOTMAX = 16
NOPT = 8 + 8 + 1 + 1 + SLOTMAX     # rise pins, fall pins, init, drive, slots
OPT_INIT, OPT_DRIVE, OPT_SLOT = 16, 17, 18
TTB = 64
N_INIDX = 16
N_MODBUCKET = 64
N_FEAT = N_KIND + 8 + TTB + 2 + 4 + 4 + 3 + OUTMAX
N_MODFEAT = 5
DUPMAX = 8           # static bound on DUPLICATE / REGENERATE region size

# pointer slots over tokens (one learned query each)
PTR = {MoveType.TT_REPLACE: 0, MoveType.TT_FLIP: 1, MoveType.DELETE_NODE: 2, MoveType.DUPLICATE: 3,
       MoveType.ADJUST_PARAM: 4, MoveType.REGENERATE: 5, MoveType.INLINE_MODULE: 6}
N_PTR = 7
VK_ADJ_DELAY, VK_DRIVE, VK_NEW_DELAY = 0, 1, 2   # value kinds (bin masks / widths); toggles have no value factor
VK_TOGGLE = 3


def _value_bins():
    lo = list(range(1, 9))
    i = 1
    while True:
        e = int(math.floor(8 * 2 ** (i / 2)))
        if e > MAX_DELAY:
            break
        if e > lo[-1]:
            lo.append(e)
        i += 1
    lo = np.array(lo, np.int64)
    hi = np.concatenate([lo[1:] - 1, [MAX_DELAY]])
    return lo, hi


BIN_LO, BIN_HI = _value_bins()
VB = len(BIN_LO)


def value_bin(v: int) -> int:
    return int(np.searchsorted(BIN_LO, int(v), side="right") - 1)


def bin_tables(op_cfg) -> tuple:
    """-> (mask [3, VB] bool, log width [3, VB], lo [3, VB], hi [3, VB]) for the value kinds."""
    ranges = [(1, MAX_DELAY), (1, op_cfg.drive_max), tuple(op_cfg.delay_range)]
    mask = np.zeros((3, VB), bool)
    lw = np.zeros((3, VB), np.float32)
    lo = np.zeros((3, VB), np.int64)
    hi = np.zeros((3, VB), np.int64)
    for k, (a, b) in enumerate(ranges):
        l, h = np.maximum(BIN_LO, a), np.minimum(BIN_HI, b)
        mask[k] = l <= h
        lo[k], hi[k] = l, h
        lw[k] = np.where(mask[k], np.log(np.maximum(h - l + 1, 1)), 0.0)
    return mask, lw, lo, hi


def _log2(x):
    return math.log2(max(x, 1)) / 8.0


class Unrepresentable(ValueError):
    pass


@dataclass
class Ctx:
    """Featurized circuit: numpy arrays (unpadded) plus the python lookups needed to decode samples."""
    circuit: Circuit
    arrays: dict
    pins: list               # slot s >= 1 -> circuit.pins()[s - 1]
    ports: list              # slot q >= 1 -> circuit.ports()[q - 1]
    pin_ix: dict
    port_ix: dict
    modules: list            # library.active() hashes (module pointer order)
    params: list             # per node: list of (field, pin) adjustable (operator._params)
    library: object = None

    @property
    def T(self):
        return len(self.arrays["tok_f"])

    def tok(self, node: int) -> int:
        return 1 + self.circuit.n_in + node

    def node(self, tok: int) -> int:
        return tok - 1 - self.circuit.n_in


def opt_index(field: int, pin: int) -> int:
    if field == RISE:
        return pin if pin < 8 else -1
    if field == FALL:
        return 8 + pin if pin < 8 else -1
    if field == INIT:
        return OPT_INIT
    if field == DRIVE:
        return OPT_DRIVE
    return OPT_SLOT + pin if pin < SLOTMAX else -1


def opt_field(o: int) -> tuple:
    if o < 8:
        return RISE, o
    if o < 16:
        return FALL, o - 8
    if o == OPT_INIT:
        return INIT, 0
    if o == OPT_DRIVE:
        return DRIVE, 0
    return SLOT, o - OPT_SLOT


_SLOT_VK = {"rise": VK_ADJ_DELAY, "fall": VK_ADJ_DELAY, "drive": VK_DRIVE, "init": VK_TOGGLE}


def _old_value(nd, field, pin, library):
    if field == SLOT:
        d = library.get(nd.module)
        return (nd.params or d.defaults())[pin], _SLOT_VK[library.slot_field(nd.module, pin)]
    if field == RISE:
        return nd.rise[pin], VK_ADJ_DELAY
    if field == FALL:
        return nd.fall[pin], VK_ADJ_DELAY
    if field == INIT:
        return nd.init, VK_TOGGLE
    return nd.drive, VK_DRIVE


def featurize(c: Circuit, operator, sig=None, task: int = 0) -> Ctx:
    """`operator`: the UniformOperator whose move space the editor shares (config, library, applicability)."""
    lib = operator.library
    cfg = operator.cfg
    I, N = c.n_in, len(c.nodes)
    T = 1 + I + N
    tf = np.zeros((T, N_FEAT), np.float32)
    inidx = np.full(T, N_INIDX, np.int32)
    modb = np.full(T, N_MODBUCKET, np.int32)
    tf[0, CLS_K] = 1
    for i in range(I):
        tf[1 + i, INPUT_K] = 1
        inidx[1 + i] = min(i, N_INIDX - 1)
    cons = c.consumers()
    live = live_nodes(c)
    o_arity = N_KIND
    o_tt = o_arity + 8
    o_sc = o_tt + TTB            # init, drive
    o_dl = o_sc + 2              # delay stats (4)
    o_mk = o_dl + 4              # n_out, n_consumers, out count, live (4)
    o_md = o_mk + 4              # module n_prims, n_in, selfloop (3)
    o_out = o_md + 3
    assert o_out + OUTMAX == N_FEAT
    for j, r in enumerate(c.outputs):
        t = 1 + r[1] if r[0] == INPUT else 1 + I + r[0]
        tf[t, o_out + min(j, OUTMAX - 1)] = 1
        tf[t, o_mk + 2] += 1
    for i in range(I):
        tf[1 + i, o_mk + 1] = math.log1p(len(cons.get((INPUT, i), ())))
    kmap = {GATE: GATE_K, DELAY: DELAY_K, WIRE: WIRE_K, MODULE: MODULE_K}
    e_dst, e_src, e_pin = [], [], []
    for i, nd in enumerate(c.nodes):
        t = 1 + I + i
        tf[t, kmap[nd.op]] = 1
        k = nd.k
        tf[t, o_arity + min(k, 7)] = 1
        if nd.op == GATE:
            nb = 1 << k
            tt = nd.tt
            tf[t, o_tt:o_tt + nb] = [(tt >> b) & 1 for b in range(nb)]
        if nd.op in (GATE, DELAY):
            tf[t, o_sc] = nd.init
            tf[t, o_sc + 1] = nd.drive / 8.0
            if nd.rise:
                d = nd.rise + nd.fall
                tf[t, o_dl:o_dl + 4] = [_log2(np.mean(nd.rise)), _log2(np.mean(nd.fall)), _log2(min(d)), _log2(max(d))]
        tf[t, o_mk] = math.log1p(nd.n_out)
        tf[t, o_mk + 1] = math.log1p(sum(len(cons.get((i, p), ())) for p in range(nd.n_out)))
        tf[t, o_mk + 3] = float(live[i])
        if nd.op == MODULE:
            d = lib.get(nd.module) if lib is not None else None
            tf[t, o_md] = math.log1p(d.n_prims) if d is not None else 0.0
            tf[t, o_md + 1] = math.log1p(k)
            modb[t] = nd.module[0] % N_MODBUCKET
        for p, (s, q) in enumerate(nd.ins):
            e_dst.append(t)
            e_src.append(1 + q if s == INPUT else 1 + I + s)
            e_pin.append(min(p, PINMAX - 1))
            if s == i:
                tf[t, o_md + 2] = 1
    pins = c.pins()
    ports = c.ports()
    pin_ix = {p: s + 1 for s, p in enumerate(pins)}
    port_ix = {p: s + 1 for s, p in enumerate(ports)}
    P, Q = len(pins) + 1, len(ports) + 1
    pin_cons = np.zeros(P, np.int32)
    pin_pos = np.zeros(P, np.int32)
    pin_src = np.zeros(P, np.int32)
    pin_srcport = np.zeros(P, np.int32)
    pin_out = np.full(P, -1, np.int32)
    pin_valid = np.zeros(P, bool)
    pin_valid[0] = True
    pin_src_slot = np.zeros(P, np.int32)       # port slot the pin currently reads
    for s, (ci, p) in enumerate(pins, 1):
        r = c.ref_at(ci, p)
        if ci == OUT:
            pin_cons[s], pin_out[s] = 0, min(p, OUTMAX - 1)
        else:
            pin_cons[s], pin_pos[s] = 1 + I + ci, min(p, PINMAX - 1)
        pin_src[s] = 1 + r[1] if r[0] == INPUT else 1 + I + r[0]
        pin_srcport[s] = 0 if r[0] == INPUT else min(r[1], PINMAX - 1)
        pin_src_slot[s] = port_ix[tuple(r)]
        pin_valid[s] = True
    port_tok = np.zeros(Q, np.int32)
    port_pos = np.zeros(Q, np.int32)
    port_node = np.full(Q, -1, np.int32)        # token of the node owning the port (inputs: -1)
    port_valid = np.ones(Q, bool)
    for s, (n, p) in enumerate(ports, 1):
        port_tok[s] = 1 + p if n == INPUT else 1 + I + n
        port_pos[s] = 0 if n == INPUT else min(p, PINMAX - 1)
        port_node[s] = -1 if n == INPUT else 1 + I + n

    # applicability, pointer masks, adjust options
    types = set(operator.applicable(c))
    type_mask = np.zeros(N_MOVES, bool)
    ptr_mask = np.zeros((N_PTR, T), bool)
    is_gate = np.array([nd.op == GATE for nd in c.nodes], bool)
    is_mod = np.array([nd.op == MODULE for nd in c.nodes], bool)
    node_sl = slice(1 + I, T)
    ptr_mask[PTR[MoveType.TT_REPLACE], node_sl] = is_gate
    ptr_mask[PTR[MoveType.TT_FLIP], node_sl] = is_gate
    n_ports_of = np.array([nd.n_out for nd in c.nodes], np.int64)
    ptr_mask[PTR[MoveType.DELETE_NODE], node_sl] = (Q - 1 - n_ports_of) > 0
    ptr_mask[PTR[MoveType.DUPLICATE], node_sl] = True
    ptr_mask[PTR[MoveType.REGENERATE], node_sl] = True
    ptr_mask[PTR[MoveType.INLINE_MODULE], node_sl] = is_mod
    params = [operator._params(nd) for nd in c.nodes]
    opt_mask = np.zeros((T, NOPT), bool)
    old_bin = np.zeros((T, NOPT), np.int32)
    vkind = np.full((T, NOPT), VK_TOGGLE, np.int32)
    for i, (nd, opts) in enumerate(zip(c.nodes, params)):
        t = 1 + I + i
        for f, p in opts:
            o = opt_index(f, p)
            if o < 0:
                continue
            v, vk = _old_value(nd, f, p, lib)
            opt_mask[t, o] = True
            old_bin[t, o] = value_bin(max(int(v), 1))
            vkind[t, o] = vk
    ptr_mask[PTR[MoveType.ADJUST_PARAM]] = opt_mask.any(1)
    modules = lib.active() if lib is not None else []
    M = max(len(modules), 1)
    mod_f = np.zeros((M, N_MODFEAT), np.float32)
    mod_bucket = np.full(M, N_MODBUCKET, np.int32)
    mod_valid = np.zeros(M, bool)
    mod_nin = np.zeros(M, np.int32)
    mod_nout = np.zeros(M, np.int32)
    for m, h in enumerate(modules):
        d = lib.get(h)
        mod_f[m] = [math.log1p(d.n_in), math.log1p(d.n_out), math.log1p(d.n_prims), len(d.slots) / 8.0,
                    math.log1p(lib.uses.get(h, 0))]
        mod_bucket[m] = h[0] % N_MODBUCKET
        mod_valid[m] = d.n_in <= MODPIN
        mod_nin[m], mod_nout[m] = d.n_in, d.n_out
    for t in types:
        if t in PTR and not ptr_mask[PTR[t]].any():
            continue
        if t == MoveType.INSERT_MODULE and not mod_valid.any():
            continue
        type_mask[t] = True
    dup_budget = min(cfg.dup_max, cfg.max_nodes - N, DUPMAX)
    regen_budget = min(cfg.dup_max, DUPMAX)
    bmask, blw, blo, bhi = bin_tables(cfg)
    sig_dim = 0 if sig is None else len(sig)
    s = np.zeros(sig_dim, np.float32) if sig is None else np.asarray(sig, np.float32).reshape(-1)
    arrays = dict(
        tok_f=tf, tok_inidx=inidx, tok_mod=modb, tok_valid=np.ones(T, bool),
        e_dst=np.array(e_dst, np.int32), e_src=np.array(e_src, np.int32), e_pin=np.array(e_pin, np.int32),
        e_valid=np.ones(len(e_dst), bool),
        pin_cons=pin_cons, pin_pos=pin_pos, pin_src=pin_src, pin_srcport=pin_srcport, pin_out=pin_out,
        pin_valid=pin_valid, pin_src_slot=pin_src_slot,
        port_tok=port_tok, port_pos=port_pos, port_node=port_node, port_valid=port_valid,
        type_mask=type_mask, ptr_mask=ptr_mask, opt_mask=opt_mask, old_bin=old_bin, vkind=vkind,
        node_k=np.array([0] * (1 + I) + [nd.k if nd.op == GATE else 0 for nd in c.nodes], np.int32),
        mod_f=mod_f, mod_bucket=mod_bucket, mod_valid=mod_valid, mod_nin=mod_nin, mod_nout=mod_nout,
        dup_budget=np.int32(max(dup_budget, 1)), regen_budget=np.int32(max(regen_budget, 1)),
        bin_mask=bmask, bin_lw=blw, bin_vlo=blo.astype(np.int32), bin_vhi=bhi.astype(np.int32),
        type_w=np.array([cfg.weights.get(t, 1.0) for t in MoveType], np.float32), p_splice=np.float32(cfg.p_splice), arity_max=np.int32(min(cfg.arity_max, NEWPIN)),
        sig=np.nan_to_num(s), sig_on=np.float32(0.0 if sig is None or np.isnan(s).all() else 1.0),
        task=np.int32(task),
    )
    return Ctx(c, arrays, pins, ports, pin_ix, port_ix, modules, params, lib)


# --- actions ------------------------------------------------------------------------------------------------------

def empty_action(T: int, P: int) -> dict:
    z = lambda *s: np.zeros(s, np.int32)  # noqa: E731
    return dict(
        type=np.int32(0), node=np.int32(0), rw_pin=np.int32(0), rw_src=np.int32(0), del_fb=np.int32(0),
        tt_bits=z(TTB), flip_row=np.int32(0),
        region=np.zeros(T, bool), dup_redir=np.int32(0), redir_mask=np.zeros(P, bool),
        add_splice=np.int32(0), add_kind=np.int32(0), add_k=np.int32(0), add_tt=z(TTB),
        add_src=z(NEWPIN), add_src_on=np.zeros(NEWPIN, bool), add_rise=z(NEWPIN), add_fall=z(NEWPIN),
        add_arc_on=np.zeros(NEWPIN, bool), add_init=np.int32(0), add_init_on=np.bool_(False),
        ins_mod=np.int32(0), ins_out=np.int32(0), ins_splice=np.int32(0), ins_src=z(MODPIN),
        ins_src_on=np.zeros(MODPIN, bool),
        adj_opt=np.int32(0), adj_bin=np.int32(0), adj_on=np.bool_(False),
    )


def _bits(x: int, n: int) -> np.ndarray:
    out = np.zeros(TTB, np.int32)
    out[:n] = [(x >> b) & 1 for b in range(n)]
    return out


def redirect_mask(ctx: Ctx, S) -> np.ndarray:
    """Pins allowed as the DUPLICATE redirect for set S (node indices): NONE and every external pin reading S."""
    c = ctx.circuit
    Sset = set(S)
    m = np.zeros(len(ctx.pins) + 1, bool)
    m[0] = True
    for s, (ci, p) in enumerate(ctx.pins, 1):
        if ci not in Sset and c.ref_at(ci, p)[0] in Sset:
            m[s] = True
    return m


def encode_move(ctx: Ctx, mv: Move) -> dict:
    """Move -> action indices. Raises Unrepresentable when the move is outside the model's support."""
    c = ctx.circuit
    A = ctx.arrays
    act = empty_action(ctx.T, len(ctx.pins) + 1)
    t = MoveType(mv.type)
    if not A["type_mask"][t]:
        raise Unrepresentable(f"{t.name} not applicable")
    act["type"] = np.int32(t)
    port = lambda r: ctx.port_ix[tuple(r)] if tuple(r) in ctx.port_ix else (  # noqa: E731
        0 if tuple(r) == (len(c.nodes), 0) else _raise(f"port {r}"))
    pin = lambda a, b: 0 if a == NONE else ctx.pin_ix[(a, b)]  # noqa: E731
    if t in PTR and t not in (MoveType.DUPLICATE, MoveType.REGENERATE):
        act["node"] = np.int32(ctx.tok(mv.node))
        if not A["ptr_mask"][PTR[t], act["node"]]:
            raise Unrepresentable("node outside pointer mask")
    if t == MoveType.TT_REPLACE:
        k = c.nodes[mv.node].k
        act["tt_bits"] = _bits(codec.u64(mv.a), 1 << k)
    elif t == MoveType.TT_FLIP:
        act["flip_row"] = np.int32(mv.a)
    elif t == MoveType.REWIRE:
        act["rw_pin"] = np.int32(ctx.pin_ix[(mv.node, mv.pin)])
        act["rw_src"] = np.int32(ctx.port_ix[(mv.a, mv.b)])
        if act["rw_src"] == A["pin_src_slot"][act["rw_pin"]]:
            raise Unrepresentable("rewire to the same source")
    elif t == MoveType.DELETE_NODE:
        if mv.a == mv.node:
            raise Unrepresentable("fallback on the deleted node")
        act["del_fb"] = np.int32(ctx.port_ix[(mv.a, mv.b)])
    elif t in (MoveType.DUPLICATE, MoveType.REGENERATE):
        S = sorted(set(int(x) for x in mv.data))
        budget = int(A["dup_budget"] if t == MoveType.DUPLICATE else A["regen_budget"])
        if not S or len(S) > budget:
            raise Unrepresentable("region size")
        act["node"] = np.int32(ctx.tok(S[0]))
        reg = np.zeros(ctx.T, bool)
        reg[[ctx.tok(s) for s in S[1:]]] = True
        act["region"] = reg
        if t == MoveType.DUPLICATE:
            act["redir_mask"] = redirect_mask(ctx, S)
            act["dup_redir"] = np.int32(pin(mv.a, mv.b))
            if not act["redir_mask"][act["dup_redir"]]:
                raise Unrepresentable("redirect pin")
    elif t == MoveType.ADD_NODE:
        nd, _ = codec.decode_node(mv.data)
        if mv.node != len(c.nodes) or nd.drive != 1:
            raise Unrepresentable("ADD_NODE index or drive")
        act["add_splice"] = np.int32(pin(mv.a, mv.b))
        kind = {GATE: 0, DELAY: 1, WIRE: 2}.get(nd.op)
        if kind is None:
            raise Unrepresentable("ADD_NODE of a module token")
        act["add_kind"] = np.int32(kind)
        k = nd.k
        if kind == 0:
            if k > int(A["arity_max"]):
                raise Unrepresentable("arity")
            act["add_k"] = np.int32(k)
            act["add_tt"] = _bits(nd.tt, 1 << k)
        spliced = mv.a != NONE
        for i, r in enumerate(nd.ins):
            if i == 0 and spliced:
                if tuple(r) != c.ref_at(mv.a, mv.b):
                    raise Unrepresentable("spliced node's pin 0 must read the spliced pin's source")
                continue
            act["add_src"][i] = port(r)
            act["add_src_on"][i] = True
        if kind in (0, 1):
            vk = VK_NEW_DELAY
            for i in range(len(nd.rise)):
                for name, v in (("add_rise", nd.rise[i]), ("add_fall", nd.fall[i])):
                    b = value_bin(v)
                    if not _in_range(A, vk, b, v):
                        raise Unrepresentable("delay outside range")
                    act[name][i] = b
                act["add_arc_on"][i] = True
            if not (kind == 0 and k == 0):
                act["add_init"] = np.int32(nd.init)
                act["add_init_on"] = np.bool_(True)
    elif t == MoveType.INSERT_MODULE:
        lo, hi, out_port, n_in = mv.data[:4]
        h = codec.ints_to_hash(lo, hi)
        if h not in ctx.modules:
            raise Unrepresentable("module not active")
        m = ctx.modules.index(h)
        if not A["mod_valid"][m] or out_port >= OUTPORT or mv.node != len(c.nodes):
            raise Unrepresentable("module pins / out port")
        act["ins_mod"], act["ins_out"] = np.int32(m), np.int32(out_port)
        act["ins_splice"] = np.int32(pin(mv.a, mv.b))
        spliced = mv.a != NONE
        src = mv.data[4:]
        for j in range(n_in):
            r = (src[2 * j], src[2 * j + 1])
            if j == 0 and spliced:
                if r != c.ref_at(mv.a, mv.b):
                    raise Unrepresentable("spliced module's pin 0 must read the spliced pin's source")
                continue
            act["ins_src"][j] = port(r)
            act["ins_src_on"][j] = True
    elif t == MoveType.ADJUST_PARAM:
        o = opt_index(mv.a, mv.pin)
        if o < 0 or not A["opt_mask"][act["node"], o]:
            raise Unrepresentable("option")
        act["adj_opt"] = np.int32(o)
        vk = int(A["vkind"][act["node"], o])
        if vk == VK_TOGGLE:
            old, _ = _old_value(c.nodes[mv.node], mv.a, mv.pin, ctx.library)
            if mv.b != 1 - old:
                raise Unrepresentable("init must toggle")
        else:
            b = value_bin(mv.b) if mv.b >= 1 else -1
            if b < 0 or not _in_range(A, vk, b, mv.b):
                raise Unrepresentable("value outside range")
            act["adj_bin"], act["adj_on"] = np.int32(b), np.bool_(True)
    return act


def _in_range(A, vk, b, v) -> bool:
    return bool(A["bin_mask"][vk, b]) and A["bin_vlo"][vk, b] <= v <= A["bin_vhi"][vk, b]


def _raise(msg):
    raise Unrepresentable(msg)


# --- decoding (sampled indices -> Move) -----------------------------------------------------------------------------

def decode_port(ctx: Ctx, q: int) -> tuple:
    return (len(ctx.circuit.nodes), 0) if q == 0 else ctx.ports[q - 1]


def decode_pin(ctx: Ctx, s: int) -> tuple:
    return (NONE, 0) if s == 0 else ctx.pins[s - 1]


def bits_to_int(bits) -> int:
    return int(sum(int(b) << i for i, b in enumerate(bits)))


# --- batching -----------------------------------------------------------------------------------------------------

# array name -> (axis, dim) padded; dims: T tokens, E edges, P pin slots, Q port slots, M modules
PAD = {
    "tok_f": (0, "T"), "tok_inidx": (0, "T"), "tok_mod": (0, "T"), "tok_valid": (0, "T"),
    "e_dst": (0, "E"), "e_src": (0, "E"), "e_pin": (0, "E"), "e_valid": (0, "E"),
    "pin_cons": (0, "P"), "pin_pos": (0, "P"), "pin_src": (0, "P"), "pin_srcport": (0, "P"), "pin_out": (0, "P"),
    "pin_valid": (0, "P"), "pin_src_slot": (0, "P"),
    "port_tok": (0, "Q"), "port_pos": (0, "Q"), "port_node": (0, "Q"), "port_valid": (0, "Q"),
    "ptr_mask": (1, "T"), "opt_mask": (0, "T"), "old_bin": (0, "T"), "vkind": (0, "T"), "node_k": (0, "T"),
    "mod_f": (0, "M"), "mod_bucket": (0, "M"), "mod_valid": (0, "M"), "mod_nin": (0, "M"), "mod_nout": (0, "M"),
    # actions
    "region": (0, "T"), "redir_mask": (0, "P"),
}
_FILL = {"pin_out": -1, "port_node": -1, "vkind": VK_TOGGLE, "tok_inidx": N_INIDX, "tok_mod": N_MODBUCKET,
         "mod_bucket": N_MODBUCKET}
BUCKETS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192)


def bucket(n: int) -> int:
    for b in BUCKETS:
        if n <= b:
            return b
    return int(2 ** math.ceil(math.log2(n)))


def dims_of(arrays: dict) -> dict:
    return {"T": len(arrays["tok_f"]), "E": len(arrays["e_dst"]), "P": len(arrays["pin_valid"]),
            "Q": len(arrays["port_valid"]), "M": len(arrays["mod_valid"])}


def padded_dims(arrays_list, bucketed: bool = True) -> dict:
    d = {k: max(dims_of(a)[k] for a in arrays_list) for k in "TEPQM"}
    d["E"] = max(d["E"], 1)
    return {k: bucket(v) for k, v in d.items()} if bucketed else d


def pad_one(name: str, x: np.ndarray, dims: dict) -> np.ndarray:
    if name not in PAD:
        return x
    ax, dim = PAD[name]
    n = dims[dim] - x.shape[ax]
    if n < 0:
        raise ValueError(f"{name}: {x.shape[ax]} > padded {dims[dim]}")
    if n == 0:
        return x
    w = [(0, 0)] * x.ndim
    w[ax] = (0, n)
    return np.pad(x, w, constant_values=_FILL.get(name, 0))


def collate(items, dims: dict, sig_dim: int) -> dict:
    """Stack a list of dicts (featurized arrays, optionally merged with action arrays), padding to `dims`."""
    out = {}
    for k in items[0]:
        if k == "sig":
            s = np.zeros((len(items), sig_dim), np.float32)
            for i, it in enumerate(items):
                v = it["sig"][:sig_dim]
                s[i, :len(v)] = v
            out[k] = s
            continue
        out[k] = np.stack([pad_one(k, it[k], dims) for it in items])
    return out
