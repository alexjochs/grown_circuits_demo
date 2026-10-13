"""Append-only, chunked, compressed log of every edit and its fate.

Layout: a directory with `index.json` and chunk files `chunk_000000.parquet` (pyarrow, zstd) or
`chunk_000000.npz` (numpy, zlib) when pyarrow is missing or `format="npz"`. Rows are buffered in memory and
written as one chunk every `chunk_rows` rows (and on flush/close); the index is rewritten atomically after each
chunk, so a crash loses at most the unflushed buffer.

Columns (one row per edit):
  edit_id int64 (sequential), generation int32, task int32,
  move int8 (MoveType), source int8 (Source), fate int8 (Fate),
  node int32, pin int32, a int64, b int64, payload list<int64>      -- the Move, replayable with apply_move
  parent_h0/h1, child_h0/h1, displaced_h0/h1 uint64                 -- 16-byte hashes as little-endian halves
  cell int64, parent_fitness float32, child_fitness float32,
  parent_sig, child_sig float32[sig_dim]                             -- behaviour signatures

`scan(filter, columns)` yields batches (dicts of numpy arrays; the payload as `payload_values` plus
`payload_offsets`). Filters: {column: value | (lo, hi) inclusive | list/set/array of values | callable(batch)->mask}.
Chunks are skipped using per-chunk ranges of edit_id, generation and task kept in the index.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from ..circuit.moves import Edit, Fate, MoveType, Source

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = pq = None

SCALARS = {
    "edit_id": np.int64, "generation": np.int32, "task": np.int32, "move": np.int8, "source": np.int8,
    "fate": np.int8, "node": np.int32, "pin": np.int32, "a": np.int64, "b": np.int64,
    "parent_h0": np.uint64, "parent_h1": np.uint64, "child_h0": np.uint64, "child_h1": np.uint64,
    "displaced_h0": np.uint64, "displaced_h1": np.uint64, "cell": np.int64,
    "parent_fitness": np.float32, "child_fitness": np.float32,
}
SIGS = ("parent_sig", "child_sig")
INDEX_VERSION = 1
_RANGED = ("edit_id", "generation", "task")


@dataclass
class _Chunk:
    file: str
    rows: int
    ranges: dict


class EditLog:
    def __init__(self, path, sig_dim: int = 0, chunk_rows: int = 1 << 16, format: str = "auto"):
        self.path = str(path)
        os.makedirs(self.path, exist_ok=True)
        idx = os.path.join(self.path, "index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                meta = json.load(f)
            if meta["sig_dim"] != sig_dim and sig_dim != 0:
                raise ValueError(f"log has sig_dim {meta['sig_dim']}, opened with {sig_dim}")
            self.sig_dim = meta["sig_dim"]
            self.format = meta["format"]
            self.chunks = [_Chunk(**c) for c in meta["chunks"]]
            self.next_id = meta["rows"]
        else:
            self.sig_dim = sig_dim
            self.format = ("parquet" if pq is not None else "npz") if format == "auto" else format
            self.chunks = []
            self.next_id = 0
        if self.format == "parquet" and pq is None:
            raise RuntimeError("parquet log requires pyarrow")
        self.chunk_rows = chunk_rows
        self._segments: list = []     # list of column dicts (numpy), in order
        self._seg_rows = 0
        self._rows = {k: [] for k in SCALARS if k != "edit_id" and not k.endswith(("_h0", "_h1"))}
        self._rows_hash = ([], [], [])
        self._rows_payload: list = []
        self._rows_sig = {k: [] for k in SIGS}
        self._n_rows = 0
        self._write_index()

    # --- appending --------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return self.next_id

    @property
    def buffered(self) -> int:
        return self._seg_rows + self._n_rows

    def append(self, e: Edit) -> int:
        """Append one edit. Returns its edit_id."""
        r = self._rows
        mv = e.move
        r["generation"].append(e.generation)
        r["task"].append(e.task)
        r["move"].append(mv.type)
        r["source"].append(e.source)
        r["fate"].append(e.fate)
        r["node"].append(mv.node)
        r["pin"].append(mv.pin)
        r["a"].append(mv.a)
        r["b"].append(mv.b)
        hs = self._rows_hash
        hs[0].append(e.parent)
        hs[1].append(e.child)
        hs[2].append(e.displaced)
        r["cell"].append(e.cell)
        r["parent_fitness"].append(e.parent_fitness)
        r["child_fitness"].append(e.child_fitness)
        self._rows_payload.append(mv.data)
        if self.sig_dim:
            self._rows_sig["parent_sig"].append(e.parent_sig if e.parent_sig is not None else np.full(self.sig_dim, np.nan))
            self._rows_sig["child_sig"].append(e.child_sig if e.child_sig is not None else np.full(self.sig_dim, np.nan))
        self._n_rows += 1
        eid = self.next_id + self.buffered - 1
        if self.buffered >= self.chunk_rows:
            self.flush()
        return eid

    def extend(self, edits) -> None:
        for e in edits:
            self.append(e)

    def append_batch(self, payload=None, **cols) -> None:
        """Columnar append. `cols` holds every scalar column except edit_id (missing ones default to 0/-1/NaN),
        as arrays of one length B; hashes may be given as `parent_hash`/`child_hash`/`displaced_hash` uint64 [B, 2];
        `parent_sig`/`child_sig` [B, sig_dim]; payload as a list of int tuples or (values, offsets)."""
        for k in ("parent", "child", "displaced"):
            h = cols.pop(f"{k}_hash", None)
            if h is not None:
                h = np.asarray(h, np.uint64).reshape(-1, 2)
                cols[f"{k}_h0"], cols[f"{k}_h1"] = h[:, 0], h[:, 1]
        lens = [len(v) for k, v in cols.items() if k in SCALARS and np.ndim(v) > 0]
        if not lens:
            raise ValueError("append_batch needs at least one array column")
        B = lens[0]
        self._materialize_rows()
        seg = {}
        for k, dt in SCALARS.items():
            if k == "edit_id":
                continue
            if k in cols:
                v = np.asarray(cols.pop(k))
                seg[k] = np.broadcast_to(v, (B,)).astype(dt) if v.ndim == 0 else v.astype(dt, copy=False)
            else:
                fill = np.nan if dt == np.float32 else (-1 if k in ("cell", "node", "pin") else 0)
                seg[k] = np.full(B, fill, dt)
        for k in SIGS:
            v = cols.pop(k, None)
            seg[k] = (np.full((B, self.sig_dim), np.nan, np.float32) if v is None
                      else np.asarray(v, np.float32).reshape(B, self.sig_dim))
        if cols:
            raise KeyError(f"unknown columns {sorted(cols)}")
        if payload is None:
            seg["payload_values"], seg["payload_offsets"] = np.zeros(0, np.int64), np.zeros(B + 1, np.int64)
        elif isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[1], np.ndarray):
            seg["payload_values"], seg["payload_offsets"] = np.asarray(payload[0], np.int64), np.asarray(payload[1], np.int64)
        else:
            seg["payload_values"], seg["payload_offsets"] = _pack_payload(payload)
        self._segments.append(seg)
        self._seg_rows += B
        if self.buffered >= self.chunk_rows:
            self.flush()

    def _materialize_rows(self) -> None:
        if not self._n_rows:
            return
        seg = {k: np.array(v, dtype=SCALARS[k]) for k, v in self._rows.items()}
        for name, hs in zip(("parent", "child", "displaced"), self._rows_hash):
            h = np.frombuffer(b"".join(hs), dtype="<u8").reshape(-1, 2)
            seg[f"{name}_h0"], seg[f"{name}_h1"] = h[:, 0].copy(), h[:, 1].copy()
        for k in SIGS:
            seg[k] = (np.asarray(self._rows_sig[k], np.float32).reshape(self._n_rows, self.sig_dim)
                      if self.sig_dim else np.zeros((self._n_rows, 0), np.float32))
        seg["payload_values"], seg["payload_offsets"] = _pack_payload(self._rows_payload)
        self._segments.append(seg)
        self._seg_rows += self._n_rows
        self._rows = {k: [] for k in self._rows}
        self._rows_hash = ([], [], [])
        self._rows_payload = []
        self._rows_sig = {k: [] for k in SIGS}
        self._n_rows = 0

    def flush(self) -> None:
        self._materialize_rows()
        if not self._segments:
            return
        cols = _concat(self._segments)
        n = len(cols["generation"])
        cols["edit_id"] = np.arange(self.next_id, self.next_id + n, dtype=np.int64)
        name = f"chunk_{len(self.chunks):06d}.{self.format}"
        fpath = os.path.join(self.path, name)
        if self.format == "parquet":
            _write_parquet(fpath, cols, self.sig_dim)
        else:
            np.savez_compressed(fpath, **cols)
        ranges = {k: [int(cols[k].min()), int(cols[k].max())] for k in _RANGED}
        self.chunks.append(_Chunk(name, n, ranges))
        self.next_id += n
        self._segments, self._seg_rows = [], 0
        self._write_index()

    def close(self) -> None:
        self.flush()

    def truncate(self, n_chunks: int, rows: int) -> None:
        """Back to the log as it was with `n_chunks` chunks and `rows` rows (a checkpoint's position): later chunks
        and anything buffered are dropped (a search resumed from a checkpoint logs those edits again)."""
        if n_chunks > len(self.chunks) or sum(c.rows for c in self.chunks[:n_chunks]) != rows:
            raise ValueError(f"edit log {self.path}: no chunk boundary at {n_chunks} chunks / {rows} rows")
        for c in self.chunks[n_chunks:]:
            try:
                os.remove(os.path.join(self.path, c.file))
            except FileNotFoundError:
                pass
        self.chunks = self.chunks[:n_chunks]
        self.next_id = rows
        self._segments, self._seg_rows = [], 0
        self._rows = {k: [] for k in self._rows}
        self._rows_hash = ([], [], [])
        self._rows_payload = []
        self._rows_sig = {k: [] for k in SIGS}
        self._n_rows = 0
        self._write_index()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _write_index(self) -> None:
        meta = {"version": INDEX_VERSION, "format": self.format, "sig_dim": self.sig_dim, "rows": self.next_id,
                "chunks": [c.__dict__ for c in self.chunks],
                "enums": {"move": {m.name: int(m) for m in MoveType}, "fate": {f.name: int(f) for f in Fate},
                          "source": {s.name: int(s) for s in Source}}}
        tmp = os.path.join(self.path, "index.json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, os.path.join(self.path, "index.json"))

    # --- reading ----------------------------------------------------------------------------------------------
    def scan(self, filter: dict | None = None, columns=None):
        """Yield one batch per chunk (after flushing the buffer), filtered."""
        self.flush()
        filter = filter or {}
        want = None if columns is None else set(columns) | {k for k in filter if isinstance(k, str)}
        for ch in self.chunks:
            if not _chunk_may_match(ch, filter):
                continue
            batch = self._read(ch, want)
            mask = np.ones(ch.rows, bool)
            for k, cond in filter.items():
                mask &= _mask(batch, k, cond)
            if not mask.any():
                continue
            if not mask.all():
                batch = _take(batch, mask)
            if columns is not None:
                keep = set(columns) | ({"payload_values", "payload_offsets"} if "payload" in columns else set())
                batch = {k: v for k, v in batch.items() if k in keep}
            yield batch

    def _read(self, ch: _Chunk, want) -> dict:
        fpath = os.path.join(self.path, ch.file)
        if ch.file.endswith(".parquet"):
            return _read_parquet(fpath, want, self.sig_dim)
        z = np.load(fpath)
        keys = z.files if want is None else [k for k in z.files if k in want or
                                             (k.startswith("payload") and "payload" in want)]
        return {k: z[k] for k in keys}

    def summary(self) -> dict:
        """Counts by move type and fate, per source, tasks, generations, disk usage."""
        M, F = len(MoveType), len(Fate)
        mf = np.zeros((M, F), np.int64)
        src = np.zeros(len(Source), np.int64)
        tasks: dict = {}
        gen = [None, None]
        rows = 0
        for b in self.scan(columns=["move", "fate", "source", "task", "generation"]):
            np.add.at(mf, (b["move"].astype(np.int64), b["fate"].astype(np.int64)), 1)
            src += np.bincount(b["source"].astype(np.int64), minlength=len(Source))[:len(Source)]
            for t, n in zip(*np.unique(b["task"], return_counts=True)):
                tasks[int(t)] = tasks.get(int(t), 0) + int(n)
            gen = [min(x for x in (gen[0], int(b["generation"].min())) if x is not None),
                   max(x for x in (gen[1], int(b["generation"].max())) if x is not None)]
            rows += len(b["move"])
        inserted = mf[:, Fate.ELITE_NEW] + mf[:, Fate.ELITE_REPLACE]
        total = mf.sum(1)
        return {
            "rows": rows, "chunks": len(self.chunks), "format": self.format,
            "bytes": sum(os.path.getsize(os.path.join(self.path, c.file)) for c in self.chunks),
            "by_move": {MoveType(m).name: {Fate(f).name: int(mf[m, f]) for f in range(F) if mf[m, f]}
                        for m in range(M) if total[m]},
            "accept_rate": {MoveType(m).name: float(inserted[m] / total[m]) for m in range(M) if total[m]},
            "by_fate": {Fate(f).name: int(mf[:, f].sum()) for f in range(F) if mf[:, f].sum()},
            "by_source": {Source(s).name: int(src[s]) for s in range(len(Source)) if src[s]},
            "tasks": tasks, "generations": gen,
        }


# --- helpers ----------------------------------------------------------------------------------------------------

def _pack_payload(payloads) -> tuple:
    lens = np.fromiter((len(p) for p in payloads), np.int64, len(payloads))
    off = np.zeros(len(payloads) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    vals = np.fromiter((x for p in payloads for x in p), np.int64, int(off[-1]))
    return vals, off


def _concat(segs) -> dict:
    out = {}
    for k in segs[0]:
        if k == "payload_offsets":
            parts, base = [np.zeros(1, np.int64)], 0
            for s in segs:
                parts.append(s[k][1:] + base)
                base += s[k][-1]
            out[k] = np.concatenate(parts)
        else:
            out[k] = np.concatenate([s[k] for s in segs])
    return out


def _take(batch: dict, mask: np.ndarray) -> dict:
    out = {}
    for k, v in batch.items():
        if k == "payload_values":
            continue
        if k == "payload_offsets":
            off = v
            idx = np.nonzero(mask)[0]
            lens = off[idx + 1] - off[idx]
            new = np.zeros(len(idx) + 1, np.int64)
            np.cumsum(lens, out=new[1:])
            vals = batch["payload_values"]
            sel = np.repeat(off[idx], lens) + (np.arange(new[-1]) - np.repeat(new[:-1], lens))
            out["payload_values"], out["payload_offsets"] = vals[sel], new
        else:
            out[k] = v[mask]
    return out


def _mask(batch, k, cond) -> np.ndarray:
    if callable(cond):
        return np.asarray(cond(batch), bool)
    col = batch[k]
    if isinstance(cond, tuple):
        lo, hi = cond
        return (col >= lo) & (col <= hi)
    if isinstance(cond, (list, set, frozenset, np.ndarray)):
        return np.isin(col, np.asarray(list(cond) if isinstance(cond, (set, frozenset)) else cond))
    return col == cond


def _chunk_may_match(ch: _Chunk, filter: dict) -> bool:
    for k, cond in filter.items():
        if k not in ch.ranges or callable(cond):
            continue
        lo, hi = ch.ranges[k]
        if isinstance(cond, tuple):
            if cond[1] < lo or cond[0] > hi:
                return False
        elif isinstance(cond, (list, set, frozenset, np.ndarray)):
            v = np.asarray(list(cond) if isinstance(cond, (set, frozenset)) else cond)
            if not ((v >= lo) & (v <= hi)).any():
                return False
        elif not lo <= cond <= hi:
            return False
    return True


def _write_parquet(fpath, cols, sig_dim) -> None:
    arrays, names = [], []
    for k in SCALARS:
        arrays.append(pa.array(cols[k]))
        names.append(k)
    for k in SIGS:
        flat = pa.array(np.ascontiguousarray(cols[k], np.float32).reshape(-1))
        arrays.append(pa.FixedSizeListArray.from_arrays(flat, sig_dim) if sig_dim else pa.nulls(len(cols["edit_id"])))
        names.append(k)
    arrays.append(pa.ListArray.from_arrays(pa.array(cols["payload_offsets"].astype(np.int32)),
                                           pa.array(cols["payload_values"], pa.int64())))
    names.append("payload")
    pq.write_table(pa.Table.from_arrays(arrays, names), fpath, compression="zstd")


def _read_parquet(fpath, want, sig_dim) -> dict:
    cols = None if want is None else [k for k in list(SCALARS) + list(SIGS) + ["payload"] if k in want]
    t = pq.read_table(fpath, columns=cols)
    out = {}
    for k in t.column_names:
        col = t.column(k).combine_chunks()
        if k == "payload":
            out["payload_offsets"] = col.offsets.to_numpy().astype(np.int64)
            out["payload_values"] = col.values.to_numpy(zero_copy_only=False).astype(np.int64)
            base = out["payload_offsets"][0]
            out["payload_offsets"] = out["payload_offsets"] - base
            out["payload_values"] = out["payload_values"][base:base + out["payload_offsets"][-1]]
        elif k in SIGS:
            n = len(col)
            out[k] = (col.values.to_numpy(zero_copy_only=False).reshape(n, sig_dim) if sig_dim
                      else np.zeros((n, 0), np.float32))
        else:
            out[k] = col.to_numpy(zero_copy_only=False)
    return out


def decode_hashes(batch: dict, which: str = "child") -> list:
    """16-byte hashes of a scanned batch's `which` column pair."""
    h = np.stack([batch[f"{which}_h0"], batch[f"{which}_h1"]], 1).astype("<u8")
    return [bytes(r) for r in h.view(np.uint8).reshape(-1, 16)]
