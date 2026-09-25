#pragma once

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDACC__
#define GGML_RHT_HOST_DEVICE __host__ __device__
#else
#define GGML_RHT_HOST_DEVICE
#endif

#define GGML_RHT_K_MAX 256
#define GGML_RHT_GAMMA 0x9E3779B97F4A7C15ull

static inline GGML_RHT_HOST_DEVICE uint64_t ggml_rht_splitmix64_mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// the k-th splitmix64 output from state seed ^ (gamma*n); s_i = -1 iff bit i%64 of word i/64 is set
static inline GGML_RHT_HOST_DEVICE uint64_t ggml_rht_sign_word_impl(uint64_t seed, int64_t n, int64_t k) {
    const uint64_t state0 = seed ^ (GGML_RHT_GAMMA * (uint64_t) n);
    return ggml_rht_splitmix64_mix(state0 + (uint64_t) (k + 1) * GGML_RHT_GAMMA);
}

// unnormalized row-major H_K as floats, cached for the process lifetime; NULL for K not in the table
GGML_API const float * ggml_rht_matrix_f32(int K);

// signs, then H_P on each of the chunks [c0, c1) of one row, in place
GGML_API void ggml_rht_stage_chunks(float * x, int64_t n, uint64_t seed, int64_t P, int64_t c0, int64_t c1);

// H_K mix across chunks and the 1/sqrt(n) scale, for the positions [j0, j1) within a chunk, in place
GGML_API void ggml_rht_stage_mix(float * x, int64_t n, int K, int64_t P, const float * H, int64_t j0, int64_t j1);

// the same mix from u into y, for the positions [j0, j1) and the output chunks [co0, co1)
GGML_API void ggml_rht_stage_mix_out(const float * u, float * y, int64_t n, int K, int64_t P, const float * H,
                                     int64_t j0, int64_t j1, int co0, int co1);

// K = 1 split as H_n = H_P1 (x) H_P2: after ggml_rht_stage_chunks with chunk size P2, this applies the
// H_P1 levels across the chunks and the scale, from u into y, for the positions [j0, j1) of P2.
// The butterflies are the ones of the unsplit transform, so the result is bitwise the same.
GGML_API void ggml_rht_stage_fwht_outer(const float * u, float * y, int64_t n, int64_t P2, int64_t j0, int64_t j1);

#ifdef __cplusplus
}
#endif
