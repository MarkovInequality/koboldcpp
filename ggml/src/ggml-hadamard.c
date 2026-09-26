#include "ggml-hadamard.h"
#include "ggml-impl.h"
#include "ggml-threading.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GGML_HAD_DECL(K) static const char had_##K[]
#include "ggml-hadamard-tables.h"
#undef GGML_HAD_DECL

static const struct {
    int          K;
    const char * rows;
} ggml_had_table[] = {
#define X(K) { K, had_##K },
    GGML_HAD_ORDERS(X)
#undef X
};

#define GGML_HAD_N_ORDERS ((int) (sizeof(ggml_had_table)/sizeof(ggml_had_table[0])))

static float * ggml_had_cache[GGML_HAD_N_ORDERS];

static int ggml_had_index(int K) {
    for (int i = 0; i < GGML_HAD_N_ORDERS; ++i) {
        if (ggml_had_table[i].K == K) {
            return i;
        }
    }
    return -1;
}

bool ggml_rht_plan(int64_t n, int * K, int64_t * P) {
    if (n <= 0) {
        return false;
    }

    int64_t m = n;
    int     t = 0;
    while ((m & 1) == 0) {
        m >>= 1;
        t++;
    }

    int k;
    if (m == 1) {
        k = 1;
    } else if (t < 2 || 4*m > GGML_RHT_K_MAX) {
        return false;
    } else {
        k = (int) (4*m);
    }

    if (K) {
        *K = k;
    }
    if (P) {
        *P = n / k;
    }
    return true;
}

bool ggml_hadamard_matrix(int K, float * out) {
    const int idx = ggml_had_index(K);
    if (idx < 0) {
        return false;
    }

    const char * rows = ggml_had_table[idx].rows;
    for (int i = 0; i < K*K; ++i) {
        out[i] = rows[i] == '+' ? 1.0f : -1.0f;
    }
    return true;
}

const float * ggml_rht_matrix_f32(int K) {
    const int idx = ggml_had_index(K);
    if (idx < 0) {
        return NULL;
    }

    ggml_critical_section_start();
    if (!ggml_had_cache[idx]) {
        float * H = (float *) malloc((size_t) K*K*sizeof(float));
        GGML_ASSERT(H);
        ggml_hadamard_matrix(K, H);
        ggml_had_cache[idx] = H;
    }
    float * H = ggml_had_cache[idx];
    ggml_critical_section_end();

    return H;
}

uint64_t ggml_rht_sign_word(uint64_t seed, int64_t n, int64_t k) {
    return ggml_rht_sign_word_impl(seed, n, k);
}

void ggml_rht_stage_chunks(float * x, int64_t n, uint64_t seed, int64_t P, int64_t c0, int64_t c1) {
    for (int64_t c = c0; c < c1; ++c) {
        float * u = x + c*P;

        for (int64_t j = 0; j < P; ) {
            const int64_t  i   = c*P + j;
            const uint64_t w   = ggml_rht_sign_word_impl(seed, n, i >> 6);
            const int      b0  = (int) (i & 63);
            const int64_t  len = MIN(64 - b0, P - j);
            for (int64_t b = 0; b < len; ++b) {
                u[j + b] *= 1.0f - 2.0f*(float) ((w >> (b0 + b)) & 1);
            }
            j += len;
        }

        for (int64_t h = 1; h < P; h <<= 1) {
            for (int64_t i = 0; i < P; i += 2*h) {
                for (int64_t k = i; k < i + h; ++k) {
                    const float a = u[k];
                    const float b = u[k + h];
                    u[k]     = a + b;
                    u[k + h] = a - b;
                }
            }
        }
    }
}

#define GGML_RHT_MIX_TILE 32

static float ggml_rht_scale(int64_t n) {
    return (float) (1.0 / sqrt((double) n));
}

// y[co*ys + k] = scale * sum_c H[co][c] * u[c*us + k] for k < w; every output sums over c in the
// same order whatever the split, so all splits give the same bytes
#if defined(__GNUC__) || defined(__clang__)
// a 32-float accumulator array stays in memory across the c loop; vector variables stay in registers
typedef float ggml_rht_v8 __attribute__((vector_size(32)));

static inline ggml_rht_v8 ggml_rht_load8(const float * p) {
    ggml_rht_v8 v;
    memcpy(&v, p, sizeof(v));
    return v;
}
#endif

static void ggml_rht_mix_tile(const float * u, int64_t us, float * y, int64_t ys, int K, const float * H,
                              int64_t w, int co0, int co1, float scale) {
    for (int co = co0; co < co1; ++co) {
        const float * h  = H + (size_t) co*K;
        float       * yo = y + co*ys;

#if defined(__GNUC__) || defined(__clang__)
        if (w == GGML_RHT_MIX_TILE) {
            ggml_rht_v8 a0 = { 0 }, a1 = { 0 }, a2 = { 0 }, a3 = { 0 };
            for (int c = 0; c < K; ++c) {
                const float   hc = h[c];
                const float * t  = u + c*us;
                a0 += hc*ggml_rht_load8(t +  0);
                a1 += hc*ggml_rht_load8(t +  8);
                a2 += hc*ggml_rht_load8(t + 16);
                a3 += hc*ggml_rht_load8(t + 24);
            }
            a0 *= scale; a1 *= scale; a2 *= scale; a3 *= scale;
            memcpy(yo +  0, &a0, sizeof(a0));
            memcpy(yo +  8, &a1, sizeof(a1));
            memcpy(yo + 16, &a2, sizeof(a2));
            memcpy(yo + 24, &a3, sizeof(a3));
            continue;
        }
#endif

        float acc[GGML_RHT_MIX_TILE] = { 0.0f };
        for (int c = 0; c < K; ++c) {
            const float   hc = h[c];
            const float * t  = u + c*us;
            for (int64_t k = 0; k < w; ++k) {
                acc[k] += hc*t[k];
            }
        }
        for (int64_t k = 0; k < w; ++k) {
            yo[k] = acc[k]*scale;
        }
    }
}

void ggml_rht_stage_mix(float * x, int64_t n, int K, int64_t P, const float * H, int64_t j0, int64_t j1) {
    const float scale = ggml_rht_scale(n);

    if (K == 1) {
        for (int64_t j = j0; j < j1; ++j) {
            x[j] *= scale;
        }
        return;
    }

    float tile[GGML_RHT_K_MAX*GGML_RHT_MIX_TILE];

    for (int64_t j = j0; j < j1; j += GGML_RHT_MIX_TILE) {
        const int64_t w = MIN(GGML_RHT_MIX_TILE, j1 - j);

        for (int c = 0; c < K; ++c) {
            memcpy(tile + c*GGML_RHT_MIX_TILE, x + c*P + j, w*sizeof(float));
        }
        ggml_rht_mix_tile(tile, GGML_RHT_MIX_TILE, x + j, P, K, H, w, 0, K, scale);
    }
}

void ggml_rht_stage_mix_out(const float * u, float * y, int64_t n, int K, int64_t P, const float * H,
                            int64_t j0, int64_t j1, int co0, int co1) {
    const float scale = ggml_rht_scale(n);

    if (K == 1) {
        for (int64_t j = j0; j < j1; ++j) {
            y[j] = u[j]*scale;
        }
        return;
    }

    for (int64_t j = j0; j < j1; j += GGML_RHT_MIX_TILE) {
        ggml_rht_mix_tile(u + j, P, y + j, P, K, H, MIN(GGML_RHT_MIX_TILE, j1 - j), co0, co1, scale);
    }
}

void ggml_rht_stage_fwht_outer(const float * u, float * y, int64_t n, int64_t P2, int64_t j0, int64_t j1) {
    const float   scale = ggml_rht_scale(n);
    const int64_t P1    = n / P2;
    GGML_ASSERT(P1 <= GGML_RHT_K_MAX);

    float tile[GGML_RHT_K_MAX*GGML_RHT_MIX_TILE];

    for (int64_t j = j0; j < j1; j += GGML_RHT_MIX_TILE) {
        const int64_t w = MIN(GGML_RHT_MIX_TILE, j1 - j);

        for (int64_t c = 0; c < P1; ++c) {
            memcpy(tile + c*GGML_RHT_MIX_TILE, u + c*P2 + j, w*sizeof(float));
        }
        for (int64_t h = 1; h < P1; h <<= 1) {
            for (int64_t i = 0; i < P1; i += 2*h) {
                for (int64_t c = i; c < i + h; ++c) {
                    float * a = tile + c*GGML_RHT_MIX_TILE;
                    float * b = tile + (c + h)*GGML_RHT_MIX_TILE;
                    for (int64_t k = 0; k < w; ++k) {
                        const float va = a[k];
                        const float vb = b[k];
                        a[k] = va + vb;
                        b[k] = va - vb;
                    }
                }
            }
        }
        for (int64_t c = 0; c < P1; ++c) {
            for (int64_t k = 0; k < w; ++k) {
                y[c*P2 + j + k] = tile[c*GGML_RHT_MIX_TILE + k]*scale;
            }
        }
    }
}

void ggml_rht_ref(float * x, int64_t n, uint64_t seed) {
    int     K;
    int64_t P;
    GGML_ASSERT(ggml_rht_plan(n, &K, &P));

    ggml_rht_stage_chunks(x, n, seed, P, 0, K);
    ggml_rht_stage_mix(x, n, K, P, ggml_rht_matrix_f32(K), 0, P);
}

// ====================== fp64: the same R, and R^T

static void ggml_rht_signs_f64(double * x, int64_t n, uint64_t seed) {
    for (int64_t i = 0; i < n; i += 64) {
        const uint64_t w = ggml_rht_sign_word_impl(seed, n, i >> 6);
        for (int64_t b = 0; b < MIN(64, n - i); ++b) {
            if ((w >> b) & 1) {
                x[i + b] = -x[i + b];
            }
        }
    }
}

static void ggml_rht_fwht_f64(double * u, int64_t P) {
    for (int64_t h = 1; h < P; h <<= 1) {
        for (int64_t i = 0; i < P; i += 2*h) {
            for (int64_t k = i; k < i + h; ++k) {
                const double a = u[k];
                const double b = u[k + h];
                u[k]     = a + b;
                u[k + h] = a - b;
            }
        }
    }
}

#define GGML_RHT_MIX_TILE_F64 16

// x[co*P + j] = scale * sum_c H[co][c] * x[c*P + j], or with H^T
static void ggml_rht_mix_f64(double * x, int64_t n, int K, int64_t P, bool transpose) {
    const double scale = 1.0/sqrt((double) n);

    if (K == 1) {
        for (int64_t j = 0; j < n; ++j) {
            x[j] *= scale;
        }
        return;
    }

    const float * H = ggml_rht_matrix_f32(K);
    double tile[GGML_RHT_K_MAX*GGML_RHT_MIX_TILE_F64];

    for (int64_t j = 0; j < P; j += GGML_RHT_MIX_TILE_F64) {
        const int64_t w = MIN(GGML_RHT_MIX_TILE_F64, P - j);
        for (int c = 0; c < K; ++c) {
            memcpy(tile + c*GGML_RHT_MIX_TILE_F64, x + c*P + j, w*sizeof(double));
        }
        for (int co = 0; co < K; ++co) {
            double acc[GGML_RHT_MIX_TILE_F64] = { 0.0 };
            for (int c = 0; c < K; ++c) {
                const double   hc = transpose ? H[(size_t) c*K + co] : H[(size_t) co*K + c];
                const double * t  = tile + c*GGML_RHT_MIX_TILE_F64;
                for (int64_t k = 0; k < w; ++k) {
                    acc[k] += hc*t[k];
                }
            }
            for (int64_t k = 0; k < w; ++k) {
                x[co*P + j + k] = acc[k]*scale;
            }
        }
    }
}

void ggml_rht_ref_f64(double * x, int64_t n, uint64_t seed) {
    int     K;
    int64_t P;
    GGML_ASSERT(ggml_rht_plan(n, &K, &P));

    ggml_rht_signs_f64(x, n, seed);
    for (int c = 0; c < K; ++c) {
        ggml_rht_fwht_f64(x + c*P, P);
    }
    ggml_rht_mix_f64(x, n, K, P, false);
}

void ggml_rht_inv_f64(double * x, int64_t n, uint64_t seed) {
    int     K;
    int64_t P;
    GGML_ASSERT(ggml_rht_plan(n, &K, &P));

    ggml_rht_mix_f64(x, n, K, P, true);
    for (int c = 0; c < K; ++c) {
        ggml_rht_fwht_f64(x + c*P, P);
    }
    ggml_rht_signs_f64(x, n, seed);
}
