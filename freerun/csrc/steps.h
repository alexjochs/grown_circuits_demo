// Host-side checks of a multi-step run (freerun.h fr_steps, docs/SPEC.md §13), shared by the CPU and CUDA engines.
#pragma once

#include <stdint.h>

#include <string>

#include "freerun.h"

// The end of a multi-step run started at t0: t0 + offset + max_b S_b K_b (check_steps guarantees no overflow).
inline int64_t steps_end(int64_t B, int64_t t0, const fr_steps* p) {
  int64_t span = 0;
  for (int64_t b = 0; b < B; ++b) span = p->S[b] * p->K[b] > span ? p->S[b] * p->K[b] : span;
  return t0 + p->offset + span;
}

// An error message, or empty if the run is well formed.
inline std::string check_steps(int64_t B, int64_t t0, const fr_steps* p, const fr_steps_out* o, int io_flags) {
  if (!p || (B > 0 && (!p->K || !p->S))) return "missing step lengths or counts";
  if (p->offset < 0 || p->offset > INT64_MAX - t0) return "offset must be >= 0 (and the run's end must fit int64)";
  const int64_t room = INT64_MAX - t0 - p->offset;
  for (int64_t b = 0; b < B; ++b) {
    auto at = [b] { return " (circuit " + std::to_string(b) + ")"; };
    if (p->K[b] < 1) return "step length K must be >= 1" + at();
    if (p->S[b] < 0) return "step count S must be >= 0" + at();
    if (p->S[b] > p->s_max) return "S exceeds the output stride s_max" + at();
    if (p->S[b] > 0 && p->K[b] > room / p->S[b]) return "the run's end time overflows" + at();
    if (p->budget && p->budget[b] < -1) return "budget must be >= 0, or -1 for none" + at();
  }
  if (o && ((o->high && !(io_flags & FR_IO_HIGH)) || (o->tog && !(io_flags & FR_IO_TOGGLES))))
    return "output high time / toggles were not requested in set_io";
  return {};
}
