/* Per-transition delay jitter (docs/SPEC.md §10), shared by the CPU and CUDA backends.
 *
 * Everything here is integer arithmetic and must match python/freerun/jitter.py bit for bit. Each backend calls
 * fr_arc_delay() at the single place where it computes the delay of a scheduled transition; with jitter off for a
 * circuit (dmin == 0) it returns the base delay and touches nothing.
 */
#ifndef FREERUN_JITTER_H
#define FREERUN_JITTER_H

#include <stdint.h>

#if defined(__CUDACC__)
#define FR_HD __host__ __device__ __forceinline__
#else
#define FR_HD static inline
#endif

#define FR_MAX_DELAY 2147483647LL
#define FR_JIT_CMAX ((1LL << 18) - 4) /* |c| <= FR_JIT_CMAX */

/* Philox4x32-10 (Salmon et al., SC'11). c: 4-word counter, transformed in place. */
FR_HD void fr_philox4x32_10(uint32_t c[4], uint32_t k0, uint32_t k1) {
  for (int r = 0; r < 10; ++r) {
    if (r) {
      k0 += 0x9E3779B9u;
      k1 += 0xBB67AE85u;
    }
    const uint64_t p0 = (uint64_t)0xD2511F53u * c[0];
    const uint64_t p1 = (uint64_t)0xCD9E8D57u * c[2];
    const uint32_t n0 = (uint32_t)(p1 >> 32) ^ c[1] ^ k0;
    const uint32_t n2 = (uint32_t)(p0 >> 32) ^ c[3] ^ k1;
    c[0] = n0;
    c[1] = (uint32_t)p1;
    c[2] = n2;
    c[3] = (uint32_t)p0;
  }
}

/* Offset j_g(k) in ticks (SPEC §10.3). */
FR_HD int64_t fr_jitter_offset(int64_t amp, uint64_t seed, uint32_t cid, int32_t gate, uint64_t k) {
  if (amp == 0) return 0;
  uint32_t c[4] = {(uint32_t)k, (uint32_t)(k >> 32), (uint32_t)gate, cid};
  fr_philox4x32_10(c, (uint32_t)seed, (uint32_t)(seed >> 32));
  int64_t s = 0;
  for (int i = 0; i < 4; ++i) s += 2 * (int64_t)(c[i] >> 16) + 1;
  s -= (int64_t)1 << 18;
  return (amp * s + ((int64_t)1 << 31)) >> 32; /* arithmetic shift = floor division by 2^32 */
}

/* Smallest possible offset for amplitude amp (SPEC §10.6). */
FR_HD int64_t fr_jitter_min_offset(int64_t amp) { return (-amp * FR_JIT_CMAX + ((int64_t)1 << 31)) >> 32; }

FR_HD int64_t fr_clamp_delay(int64_t d, int64_t dmin) {
  return d < dmin ? dmin : (d > FR_MAX_DELAY ? FR_MAX_DELAY : d);
}

/* One circuit's jitter parameters. dmin == 0: no jitter for this circuit. amp: the circuit's slice [G]. */
typedef struct fr_jitter_view {
  uint64_t seed;
  uint32_t cid;
  int32_t dmin;
  const int64_t* amp;
} fr_jitter_view;

/* Effective delay of a transition scheduled by gate g with base (arc) delay `base` (SPEC §10.4).
 * create: the gate had no pending transition, so this creates transition n (and n is incremented); otherwise the
 * pending transition, index n - 1, is re-evaluated. n is the gate's transition counter. */
FR_HD int64_t fr_arc_delay_amp(const fr_jitter_view& j, int64_t amp, int32_t g, int64_t base, bool create,
                               uint64_t& n) {
  if (j.dmin == 0) return base;
  const uint64_t k = create ? n++ : n - 1;
  return fr_clamp_delay(base + fr_jitter_offset(amp, j.seed, j.cid, g, k), j.dmin);
}
FR_HD int64_t fr_arc_delay(const fr_jitter_view& j, int32_t g, int64_t base, bool create, uint64_t& n) {
  return j.dmin == 0 ? base : fr_arc_delay_amp(j, j.amp[g], g, base, create, n);
}
/* fr_arc_delay_amp is the same with the gate's amplitude j.amp[g] passed in (callers that load it early). */

/* Lower bound on every effective delay of an arc with base delay `base` of a gate with amplitude amp. */
FR_HD int64_t fr_min_effective_delay(int64_t base, int64_t amp, int32_t dmin) {
  if (dmin == 0) return base;
  return fr_clamp_delay(base + fr_jitter_min_offset(amp), dmin);
}

#endif
