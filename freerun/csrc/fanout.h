// Fanout CSR of a batch, derived from the fanin (shared by the CPU and CUDA engines). Layout as in fr_batch_desc:
// per circuit b, fo_ptr holds N_b + 1 local offsets at net_off[b] + b; fo_edge holds local edge indices at
// edge_off[b]; a net's entries are sorted by (gate, pin); edge_gate is each edge's local gate.
#pragma once

#include <cstdint>
#include <vector>

#include "freerun.h"

struct Fanout {
  std::vector<int32_t> fo_ptr, fo_edge, edge_gate;
};

// Uses the descriptor's arrays when given, and builds them (a counting sort per circuit, in edge order, which is
// (gate, pin) order) when they are NULL.
inline Fanout batch_fanout(const fr_batch_desc* d) {
  const int64_t B = d->n_circuits, N = d->net_off[B], E = d->edge_off[B];
  Fanout f;
  if (d->fo_ptr && d->fo_edge && d->edge_gate) {
    f.fo_ptr.assign(d->fo_ptr, d->fo_ptr + N + B);
    f.fo_edge.assign(d->fo_edge, d->fo_edge + E);
    f.edge_gate.assign(d->edge_gate, d->edge_gate + E);
    return f;
  }
  f.fo_ptr.assign(size_t(N + B), 0);
  f.fo_edge.resize(size_t(E));
  f.edge_gate.resize(size_t(E));
  std::vector<int32_t> fill;
  for (int64_t b = 0; b < B; ++b) {
    const int64_t no = d->net_off[b], eo = d->edge_off[b], go = d->gate_off[b];
    const int64_t Nb = d->net_off[b + 1] - no, Gb = d->gate_off[b + 1] - go, Eb = d->edge_off[b + 1] - eo;
    const int32_t* fptr = d->fanin_ptr + go + b;
    const int32_t* fanin = d->fanin + eo;
    int32_t* ptr = &f.fo_ptr[no + b];
    for (int64_t g = 0; g < Gb; ++g)
      for (int32_t e = fptr[g]; e < fptr[g + 1]; ++e) f.edge_gate[eo + e] = int32_t(g);
    for (int64_t e = 0; e < Eb; ++e) ++ptr[fanin[e] + 1];
    for (int64_t n = 0; n < Nb; ++n) ptr[n + 1] += ptr[n];
    fill.assign(ptr, ptr + Nb);
    for (int64_t e = 0; e < Eb; ++e) f.fo_edge[eo + fill[fanin[e]]++] = int32_t(e);
  }
  return f;
}
