#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-threading.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static inline int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static const float hq_ones[32] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};

// the base searches with uniform weights (n <= 32)
static inline float make_qkx3_quants(int n, int nmax, const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L,
        float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux, float rmin, float rdelta, int nstep) {
    return ggml_make_qkx3_quants(n, nmax, x, hq_ones, L, the_min, Laux, rmin, rdelta, nstep, false);
}

static inline float make_qp_quants(int n, int nmax, const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L) {
    return ggml_make_qp_quants(n, nmax, x, L, hq_ones);
}

// make_qx_quants with uniform weights: the codes l + nmax in L, the least-squares scale returned
static float hq_make_qx_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) {
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        sumlx += x[i]*l;
        suml2 += l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            sumlx += x[i]*l;
            suml2 += l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static void hq_set_codes(enum ggml_type type, void * GGML_RESTRICT y, const uint8_t * GGML_RESTRICT L, int unit);

// ====================== HQ4_K / HQ5_K

// the final code of an affine value (d*l - dm), at a scale d != 0
static inline int hq_k_code(float x, float d, float dm, int nmax) {
    const float v = (x + dm)/d;
    const int l = nearest_int(fminf(fmaxf(v, -4194303.f), 4194303.f));
    return MAX(0, MIN(nmax, l));
}

// the final level of a symmetric value (d*l), at a scale d != 0
static inline int hq_s_code(float x, float d, int lmin, int lmax) {
    const float v = x/d;
    const int l = nearest_int(fminf(fmaxf(v, -4194303.f), 4194303.f));
    return MAX(lmin, MIN(lmax, l));
}

static void quantize_row_hq4_K_impl(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t n_per_row) {
    assert(n_per_row % QK_K == 0);
    const int64_t nb = n_per_row / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K/32];
    uint8_t Lm[QK_K/32];
    float   mins[QK_K/32];
    float   scales[QK_K/32];

    for (int i = 0; i < nb; i++) {

        for (int j = 0; j < QK_K/32; ++j) {
            scales[j] = make_qkx3_quants(32, 15, x + 32*j, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36);
        }

        float d_block = make_qp_quants(QK_K/32, 63, scales, Ls);
        float m_block = make_qp_quants(QK_K/32, 63, mins,   Lm);
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = Ls[j];
            uint8_t lm = Lm[j];
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(d_block);
        y[i].dmin = GGML_FP32_TO_FP16(m_block);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                L[32*j + ii] = hq_k_code(x[32*j + ii], d, dm, 15);
            }
        }
        hq_set_codes(GGML_TYPE_HQ4_K, y + i, L, QK_K);

        x += QK_K;
    }
}

static void quantize_row_hq5_K_impl(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t n_per_row) {
    assert(n_per_row % QK_K == 0);
    const int64_t nb = n_per_row / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K/32];
    uint8_t Lm[QK_K/32];
    float   mins[QK_K/32];
    float   scales[QK_K/32];

    for (int i = 0; i < nb; i++) {

        for (int j = 0; j < QK_K/32; ++j) {
            scales[j] = make_qkx3_quants(32, 31, x + 32*j, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36);
        }

        float d_block = make_qp_quants(QK_K/32, 63, scales, Ls);
        float m_block = make_qp_quants(QK_K/32, 63, mins,   Lm);

        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = Ls[j];
            uint8_t lm = Lm[j];
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(d_block);
        y[i].dmin = GGML_FP32_TO_FP16(m_block);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                L[32*j + ii] = hq_k_code(x[32*j + ii], d, dm, 31);
            }
        }
        hq_set_codes(GGML_TYPE_HQ5_K, y + i, L, QK_K);

        x += QK_K;
    }
}

// ====================== HQ2 (IQ2 grids)

// how a codebook sub-block's signs were set: by hq_signs8, by its complement (the search's scale came out
// negative), or not at all (the sub-block was skipped, and its stored signs are zero)
enum { HQ_SIGNS_RULE, HQ_SIGNS_INVERT, HQ_SIGNS_KEEP };

// the sign bits of 8 values (bit i set: x[i] < 0), and |x| in xval; with parity the smallest |x| is
// flipped too if needed, so that the count of set bits is even (that value's xval becomes negative)
static inline uint8_t hq_signs8(const float * GGML_RESTRICT x, float * GGML_RESTRICT xval, bool parity) {
    int nflip = 0;
    uint8_t s = 0;
    for (int i = 0; i < 8; ++i) {
        if (x[i] >= 0) xval[i] = x[i];
        else {
            xval[i] = -x[i]; ++nflip; s |= (1 << i);
        }
    }
    if (parity && nflip%2) {
        int imin = 0; float min = xval[imin];
        for (int i = 1; i < 8; ++i) {
            if (xval[i] < min) {
                min = xval[i]; imin = i;
            }
        }
        xval[imin] = -xval[imin];
        s ^= (1 << imin);
    }
    return s;
}

static int hq2_find_best_neighbour(const uint16_t * GGML_RESTRICT neighbours, const uint64_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, float scale, int8_t * GGML_RESTRICT L) {
    int num_neighbors = neighbours[0];
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

// smode: each sub-block's HQ_SIGNS_*
static void quantize_row_hq2_xxs_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, uint8_t * smode) {

    const uint64_t * kgrid_q2xs;
    const int      * kmap_q2xs;
    const uint16_t * kneighbors_q2xs;
    ggml_iq2_entry(GGML_TYPE_IQ2_XXS, &kgrid_q2xs, &kmap_q2xs, &kneighbors_q2xs);

    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n/QK_K;

    block_iq2_xxs * y = vy;

    float scales[QK_K/32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    uint8_t block_signs[4];
    uint32_t q2[2*(QK_K/32)];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        memset(q2, 0, QK_K/4);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float * xb = xbl + 32*ib;
            for (int k = 0; k < 4; ++k) {
                block_signs[k] = hq_signs8(xb + 8*k, xval + 8*k, true) & 127;
            }
            smode[ibl*(QK_K/32) + ib] = HQ_SIGNS_RULE;
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                memset(L, 0, 32);
                smode[ibl*(QK_K/32) + ib] = HQ_SIGNS_KEEP;
                continue;
            }
            float scale = make_qp_quants(32, kMaxQ+1, xval, (uint8_t*)L);
            float eff_max = scale*kMaxQ;
            if (eff_max <= 0) {
                scales[ib] = 0;
                memset(L, 0, 32);
                smode[ibl*(QK_K/32) + ib] = HQ_SIGNS_KEEP;
                continue;
            }
            float best = 0;
            for (int is = -6; is <= 6; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/eff_max;
                float this_scale = 1/id;
                for (int k = 0; k < 4; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = hq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, this_scale, Laux + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float q = 2*Laux[i] + 1;
                    sumqx += xval[i]*q;
                    sumq2 += q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    memcpy(L, Laux, 32);
                }
            }
            if (scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 4; ++k) {
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 2*i);
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = hq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, scale, L + 8*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q2xs + grid_index);
                    for (int i = 0; i < 8; ++i) L[8*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float q = 2*L[i] + 1;
                    sumqx += xval[i]*q;
                    sumq2 += q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                smode[ibl*(QK_K/32) + ib] = HQ_SIGNS_INVERT;
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                q2[2*ib+0] |= ((uint32_t) grid_index << 8*k);
                q2[2*ib+1] |= (block_signs[k] << 7*k);
            }
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            memset(smode + ibl*(QK_K/32), HQ_SIGNS_KEEP, QK_K/32);
            memset(y[ibl].qs, 0, QK_K/4);
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            q2[2*ib+1] |= ((uint32_t)l << 28);
        }
        memcpy(y[ibl].qs, q2, QK_K/4);
    }
}

// the IQ2_XS / IQ2_S search over one 16-value sub-block of magnitudes xval: lattice coordinates in L (left as they
// are where no scale improves on them) and the least-squares scale, which can come out negative
static float hq2_search16(const float * GGML_RESTRICT xval, float max, const uint64_t * kgrid_q2xs, const int * kmap_q2xs,
                          const uint16_t * kneighbors_q2xs, int8_t * GGML_RESTRICT L) {
    const int kMaxQ = 3;
    int8_t Laux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    float best = 0;
    float scale = max/(2*kMaxQ-1);
    is_on_grid[0] = is_on_grid[1] = true;
    for (int is = -9; is <= 9; ++is) {
        float id = (2*kMaxQ-1+is*0.1f)/max;
        float this_scale = 1/id;
        for (int k = 0; k < 2; ++k) {
            for (int i = 0; i < 8; ++i) {
                int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
            }
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
            int grid_index = kmap_q2xs[u];
            is_on_grid_aux[k] = true;
            if (grid_index < 0) {
                is_on_grid_aux[k] = false;
                const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                grid_index = hq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, this_scale, Laux + 8*k);
            }
        }
        float sumqx = 0, sumq2 = 0;
        for (int i = 0; i < 16; ++i) {
            float q = 2*Laux[i] + 1;
            sumqx += xval[i]*q;
            sumq2 += q*q;
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            scale = sumqx/sumq2; best = scale*sumqx;
            for (int i = 0; i < 16; ++i) L[i] = Laux[i];
            for (int k = 0; k <  2; ++k) is_on_grid[k] = is_on_grid_aux[k];
        }
    }
    int n_not_ongrid = 0;
    for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
    if (n_not_ongrid > 0 && scale > 0) {
        float id = 1/scale;
        for (int k = 0; k < 2; ++k) {
            if (is_on_grid[k]) continue;
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) {
                int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                l = MAX(0, MIN(kMaxQ-1, l));
                u |= (l << 2*i);
                L[8*k + i] = l;
            }
            int grid_index = kmap_q2xs[u];
            if (grid_index < 0) {
                const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                grid_index = hq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, scale, L + 8*k);
            }
        }
        float sumqx = 0, sumq2 = 0;
        for (int i = 0; i < 16; ++i) {
            float q = 2*L[i] + 1;
            sumqx += xval[i]*q;
            sumq2 += q*q;
        }
        if (sumq2 > 0) scale = sumqx/sumq2;
    }
    return scale;
}

static void quantize_row_hq2_xs_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, uint8_t * smode) {

    const uint64_t * kgrid_q2xs;
    const int      * kmap_q2xs;
    const uint16_t * kneighbors_q2xs;
    ggml_iq2_entry(GGML_TYPE_IQ2_XS, &kgrid_q2xs, &kmap_q2xs, &kneighbors_q2xs);

    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int64_t nbl = n/QK_K;

    block_iq2_xs * y = vy;

    float scales[QK_K/16];
    float xval[16];
    int8_t L[16];
    uint8_t block_signs[2];
    uint16_t q2[2*(QK_K/16)];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        memset(q2, 0, QK_K/4);
        memset(y[ibl].scales, 0, QK_K/32);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            for (int k = 0; k < 2; ++k) {
                block_signs[k] = hq_signs8(xb + 8*k, xval + 8*k, true) & 127;
            }
            smode[ibl*(QK_K/16) + ib] = HQ_SIGNS_RULE;
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 16);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                smode[ibl*(QK_K/16) + ib] = HQ_SIGNS_KEEP;
                continue;
            }
            float scale = hq2_search16(xval, max, kgrid_q2xs, kmap_q2xs, kneighbors_q2xs, L);
            if (scale < 0) {
                smode[ibl*(QK_K/16) + ib] = HQ_SIGNS_INVERT;
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                q2[2*ib+k] = grid_index | (block_signs[k] << 9);
            }
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            memset(smode + ibl*(QK_K/16), HQ_SIGNS_KEEP, QK_K/16);
            memset(y[ibl].qs, 0, QK_K/4);
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            if (ib%2 == 0) y[ibl].scales[ib/2] = l;
            else y[ibl].scales[ib/2] |= (l << 4);
        }
        memcpy(y[ibl].qs, q2, QK_K/4);
    }
}

static void quantize_row_hq2_s_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, uint8_t * smode) {

    const uint64_t * kgrid_q2xs;
    const int      * kmap_q2xs;
    const uint16_t * kneighbors_q2xs;
    ggml_iq2_entry(GGML_TYPE_IQ2_S, &kgrid_q2xs, &kmap_q2xs, &kneighbors_q2xs);

    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int64_t nbl = n/QK_K;

    block_iq2_s * y = vy;

    float scales[QK_K/16];
    float xval[16];
    int8_t L[16];
    uint8_t block_signs[2];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_s));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            for (int k = 0; k < 2; ++k) {
                block_signs[k] = hq_signs8(xb + 8*k, xval + 8*k, false);
            }
            smode[ibl*(QK_K/16) + ib] = HQ_SIGNS_RULE;
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 16);
            if (max < GROUP_MAX_EPS_IQ2_S) {
                scales[ib] = 0;
                smode[ibl*(QK_K/16) + ib] = HQ_SIGNS_KEEP;
                continue;
            }
            float scale = hq2_search16(xval, max, kgrid_q2xs, kmap_q2xs, kneighbors_q2xs, L);
            if (scale < 0) {
                smode[ibl*(QK_K/16) + ib] = HQ_SIGNS_INVERT;
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = ~block_signs[k];
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                const int i8 = 2*ib + k;
                y[ibl].qs[i8] = grid_index & 255;
                y[ibl].qh[i8/4] |= ((grid_index >> 8) << 2*(i8%4));
                y[ibl].qs[QK_K/8 + i8] = block_signs[k];
            }
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d * 0.9875f);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            if (ib%2 == 0) y[ibl].scales[ib/2] = l;
            else y[ibl].scales[ib/2] |= (l << 4);
        }
    }
}

// ====================== HQ3 (IQ3 grids)

static int hq3_find_best_neighbour(const uint16_t * GGML_RESTRICT neighbours, const uint32_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, float scale, int8_t * GGML_RESTRICT L) {
    int num_neighbors = neighbours[0];
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 4; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

static void quantize_row_hq3_xxs_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, uint8_t * smode) {

    const uint32_t * kgrid_q3xs;
    const int      * kmap_q3xs;
    const uint16_t * kneighbors_q3xs;
    ggml_iq3_entry(256, &kgrid_q3xs, &kmap_q3xs, &kneighbors_q3xs);

    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 8;

    const int64_t nbl = n/QK_K;

    block_iq3_xxs * y = vy;

    float scales[QK_K/32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    bool   is_on_grid[8];
    bool   is_on_grid_aux[8];
    uint8_t block_signs[8];
    uint32_t q3[3*QK_K/32];
    uint8_t  * q3_grid = (uint8_t *) q3;
    uint32_t * scales_and_signs = q3 + QK_K/16;

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        memset(q3, 0, sizeof(q3));

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float * xb = xbl + 32*ib;
            for (int k = 0; k < 4; ++k) {
                block_signs[k] = hq_signs8(xb + 8*k, xval + 8*k, true) & 127;
            }
            smode[ibl*(QK_K/32) + ib] = HQ_SIGNS_RULE;
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 32);
            if (max < GROUP_MAX_EPS_IQ3_XXS) {
                scales[ib] = 0;
                smode[ibl*(QK_K/32) + ib] = HQ_SIGNS_KEEP;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            for (int k = 0; k < 8; ++k) is_on_grid[k] = true;
            for (int is = -15; is <= 15; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < 8; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = hq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, this_scale, Laux + 4*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float q = 2*Laux[i] + 1;
                    sumqx += xval[i]*q;
                    sumq2 += q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < 32; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  8; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 8; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 8; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = hq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, scale, L + 4*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float q = 2*L[i] + 1;
                    sumqx += xval[i]*q;
                    sumq2 += q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                smode[ibl*(QK_K/32) + ib] = HQ_SIGNS_INVERT;
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 8; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                q3_grid[8*ib+k] = grid_index;
            }
            scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            memset(smode + ibl*(QK_K/32), HQ_SIGNS_KEEP, QK_K/32);
            memset(y[ibl].qs, 0, sizeof(y[ibl].qs));
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d * 1.0125f);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            scales_and_signs[ib] |= ((uint32_t)l << 28);
        }
        memcpy(y[ibl].qs, q3, sizeof(y[ibl].qs));
    }
}

static void quantize_row_hq3_s_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, uint8_t * smode) {

    const uint32_t * kgrid_q3xs;
    const int      * kmap_q3xs;
    const uint16_t * kneighbors_q3xs;
    ggml_iq3_entry(512, &kgrid_q3xs, &kmap_q3xs, &kneighbors_q3xs);

    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    enum { block_size = 32, bs4 = block_size/4, bs8 = block_size/8 };

    const int kMaxQ = 8;

    const int64_t nbl = n/QK_K;

    block_iq3_s * y = vy;

    float scales[QK_K/block_size];
    float xval[block_size];
    int8_t L[block_size];
    int8_t Laux[block_size];
    bool   is_on_grid[bs4];
    bool   is_on_grid_aux[bs4];
    uint8_t block_signs[bs8];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_s));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        uint8_t * qs = y[ibl].qs;
        uint8_t * qh = y[ibl].qh;
        uint8_t * signs = y[ibl].signs;

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;
            for (int k = 0; k < bs8; ++k) {
                block_signs[k] = hq_signs8(xb + 8*k, xval + 8*k, false);
            }
            smode[ibl*(QK_K/32) + ib] = HQ_SIGNS_RULE;
            float max = xval[0];
            for (int i = 1; i < block_size; ++i) max = MAX(max, xval[i]);
            memset(L, 0, block_size);
            if (!max) {
                scales[ib] = 0;
                smode[ibl*(QK_K/32) + ib] = HQ_SIGNS_KEEP;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            for (int k = 0; k < bs4; ++k) is_on_grid[k] = false;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < bs4; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = hq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, this_scale, Laux + 4*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float q = 2*Laux[i] + 1;
                    sumqx += xval[i]*q;
                    sumq2 += q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < block_size; ++i) L[i] = Laux[i];
                    for (int k = 0; k < bs4; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < bs4; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < bs4; ++k) {
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = hq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, scale, L + 4*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float q = 2*L[i] + 1;
                    sumqx += xval[i]*q;
                    sumq2 += q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                smode[ibl*(QK_K/32) + ib] = HQ_SIGNS_INVERT;
                scale = -scale;
                for (int k = 0; k < bs8; ++k) block_signs[k] = ~block_signs[k];
            }
            for (int k = 0; k < bs4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                qs[k] = grid_index & 255;
                qh[(ib*bs4+k)/8] |= ((grid_index >> 8) << ((ib*bs4+k)%8));
            }
            qs += bs4;
            for (int k = 0; k < bs8; ++k) signs[k] = block_signs[k];
            signs += bs8;
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d * 1.033f);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/block_size; ib += 2) {
            int l1 = nearest_int(0.5f*(id*scales[ib+0]-1));
            l1 = MAX(0, MIN(15, l1));
            int l2 = nearest_int(0.5f*(id*scales[ib+1]-1));
            l2 = MAX(0, MIN(15, l2));
            y[ibl].scales[ib/2] = l1 | (l2 << 4);
        }
    }
}

// ====================== HQ2 / HQ3 code re-pick
//
// The searches above pick codes on the quantizer's lattice at the unrounded sub-block scale, but
// a block decodes with the re-rounded (and fudged) scale and a decode grid that is not exactly
// that lattice. So once a block is final, each code is re-picked as the grid point nearest to the
// real decode at the stored scale, reading the block the way dequantize_row_iq* does. Signs stay
// as stored, so every sign/parity constraint still holds.

typedef struct {
    int             gs;
    int             nlev;
    int             bits;       // bits per coordinate in a map index
    float           lev[8];     // decode magnitudes, ascending: lattice coordinate l decodes to lev[l]
    const uint8_t * grid;       // decode grid, gs magnitudes per point
    const int     * map;        // lattice index -> grid index, < 0 if not in the grid
} hq_dec_grid;

static void hq_dec_grid_init(hq_dec_grid * t, const uint8_t * grid, int ngrid, int gs, const int * map) {
    GGML_ASSERT(map && "forgot to call ggml_quantize_init()?");
    t->gs   = gs;
    t->bits = gs == 8 ? 2 : 3;
    t->nlev = 0;
    t->grid = grid;
    t->map  = map;
    for (int k = 0; k < gs*ngrid; ++k) {
        int pos = 0;
        while (pos < t->nlev && t->lev[pos] < grid[k]) ++pos;
        if (pos < t->nlev && t->lev[pos] == grid[k]) continue;
        GGML_ASSERT(t->nlev < (1 << t->bits));
        memmove(t->lev + pos + 1, t->lev + pos, (t->nlev - pos)*sizeof(float));
        t->lev[pos] = grid[k];
        t->nlev++;
    }
    for (int k = 0; k < ngrid; ++k) {
        int u = 0;
        for (int i = 0; i < gs; ++i) {
            int l = 0;
            while (t->lev[l] != grid[gs*k + i]) ++l;
            u |= l << t->bits*i;
        }
        GGML_ASSERT(map[u] == k);
    }
}

// the levels are built once per type; the map is looked up on every call, as ggml_quantize_free can reallocate it
static void hq_dec_grid_for(hq_dec_grid * t, enum ggml_type type) {
    static hq_dec_grid cache[GGML_TYPE_COUNT];
    static bool        ready[GGML_TYPE_COUNT];

    const uint64_t * grid2;
    const uint32_t * grid3;
    const int      * map = NULL;
    const uint16_t * neighbours;
    const uint8_t  * grid = NULL;
    int ngrid = 0, gs = 0;
    switch (type) {
        case GGML_TYPE_HQ2_XXS: ggml_iq2_entry(GGML_TYPE_IQ2_XXS, &grid2, &map, &neighbours); grid = (const uint8_t *) iq2xxs_grid; ngrid =  256; gs = 8; break;
        case GGML_TYPE_HQ2_XS:  ggml_iq2_entry(GGML_TYPE_IQ2_XS,  &grid2, &map, &neighbours); grid = (const uint8_t *) iq2xs_grid;  ngrid =  512; gs = 8; break;
        case GGML_TYPE_HQ2_S:   ggml_iq2_entry(GGML_TYPE_IQ2_S,   &grid2, &map, &neighbours); grid = (const uint8_t *) iq2s_grid;   ngrid = 1024; gs = 8; break;
        case GGML_TYPE_HQ3_XXS: ggml_iq3_entry(256, &grid3, &map, &neighbours);               grid = (const uint8_t *) iq3xxs_grid; ngrid =  256; gs = 4; break;
        case GGML_TYPE_HQ3_S:   ggml_iq3_entry(512, &grid3, &map, &neighbours);               grid = (const uint8_t *) iq3s_grid;   ngrid =  512; gs = 4; break;
        default: return;
    }

    ggml_critical_section_start();
    if (!ready[type]) {
        hq_dec_grid_init(&cache[type], grid, ngrid, gs, map);
        ready[type] = true;
    }
    *t = cache[type];
    ggml_critical_section_end();
    t->map = map;
}

typedef struct {
    const hq_dec_grid * t;
    float cost[8][8];   // cost[i][l]: squared error of coordinate i decoded at level l
    int   order[8][8];  // levels of coordinate i by ascending cost
    float rest[9];      // rest[i]: sum of the smallest costs of coordinates i..gs-1
    float best;
    int   best_u;
} hq_repick_state;

static void hq_repick_search(hq_repick_state * st, int i, float partial, int u) {
    const hq_dec_grid * t = st->t;
    if (i == t->gs) {
        if (partial < st->best && t->map[u] >= 0) {
            st->best   = partial;
            st->best_u = u;
        }
        return;
    }
    for (int r = 0; r < t->nlev; ++r) {
        const int l = st->order[i][r];
        const float p = partial + st->cost[i][l];
        if (p + st->rest[i + 1] >= st->best) {
            break;
        }
        hq_repick_search(st, i + 1, p, u | (l << t->bits*i));
    }
}

// the grid point nearest to xs at scale db > 0, found exactly by branch and bound over the
// per-coordinate decode levels; cur unless another point is strictly nearer
static int hq_repick(const hq_dec_grid * t, const float * GGML_RESTRICT xs, float db, int cur) {
    hq_repick_state st;
    st.t = t;
    st.rest[t->gs] = 0;
    for (int i = t->gs - 1; i >= 0; --i) {
        int lmin = 0;
        for (int l = 0; l < t->nlev; ++l) {
            const float diff = db*t->lev[l] - xs[i];
            st.cost[i][l] = diff*diff;
            if (st.cost[i][l] < st.cost[i][lmin]) lmin = l;
        }
        int lo = lmin - 1, hi = lmin + 1;
        st.order[i][0] = lmin;
        for (int r = 1; r < t->nlev; ++r) {
            if (hi >= t->nlev || (lo >= 0 && st.cost[i][lo] <= st.cost[i][hi])) {
                st.order[i][r] = lo--;
            } else {
                st.order[i][r] = hi++;
            }
        }
        st.rest[i] = st.rest[i + 1] + st.cost[i][lmin];
    }
    st.best = 0;
    for (int i = 0; i < t->gs; ++i) {
        const float diff = db*t->grid[t->gs*cur + i] - xs[i];
        st.best += diff*diff;
    }
    st.best_u = -1;
    hq_repick_search(&st, 0, 0.0f, 0);
    return st.best_u < 0 ? cur : t->map[st.best_u];
}

static void hq2_xxs_repick(const float * GGML_RESTRICT x, block_iq2_xxs * GGML_RESTRICT y, int64_t n, const hq_dec_grid * t) {
    float xs[8];
    for (int64_t ibl = 0; ibl < n/QK_K; ++ibl) {
        const float d = GGML_FP16_TO_FP32(y[ibl].d);
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            uint32_t aux32[2];
            uint8_t * aux8 = (uint8_t *) aux32;
            memcpy(aux32, y[ibl].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            if (!(db > 0)) continue;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                const float * xg = x + QK_K*ibl + 32*ib32 + 8*l;
                for (int j = 0; j < 8; ++j) xs[j] = signs & kmask_iq2xs[j] ? -xg[j] : xg[j];
                aux8[l] = hq_repick(t, xs, db, aux8[l]);
            }
            memcpy(y[ibl].qs + 4*ib32, aux32, 2*sizeof(uint32_t));
        }
    }
}

static void hq2_xs_repick(const float * GGML_RESTRICT x, block_iq2_xs * GGML_RESTRICT y, int64_t n, const hq_dec_grid * t) {
    float xs[8];
    for (int64_t ibl = 0; ibl < n/QK_K; ++ibl) {
        const float d = GGML_FP16_TO_FP32(y[ibl].d);
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const float db[2] = {
                d * (0.5f + (y[ibl].scales[ib32] & 0xf)) * 0.25f,
                d * (0.5f + (y[ibl].scales[ib32] >>  4)) * 0.25f,
            };
            for (int l = 0; l < 4; ++l) {
                if (!(db[l/2] > 0)) continue;
                uint16_t * q = y[ibl].qs + 4*ib32 + l;
                const uint8_t signs = ksigns_iq2xs[*q >> 9];
                const float * xg = x + QK_K*ibl + 32*ib32 + 8*l;
                for (int j = 0; j < 8; ++j) xs[j] = signs & kmask_iq2xs[j] ? -xg[j] : xg[j];
                *q = (*q & ~511) | hq_repick(t, xs, db[l/2], *q & 511);
            }
        }
    }
}

static void hq2_s_repick(const float * GGML_RESTRICT x, block_iq2_s * GGML_RESTRICT y, int64_t n, const hq_dec_grid * t) {
    float xs[8];
    for (int64_t ibl = 0; ibl < n/QK_K; ++ibl) {
        const float d = GGML_FP16_TO_FP32(y[ibl].d);
        uint8_t * qs = y[ibl].qs;
        uint8_t * qh = y[ibl].qh;
        const uint8_t * signs = y[ibl].qs + QK_K/8;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const float db[2] = {
                d * (0.5f + (y[ibl].scales[ib32] & 0xf)) * 0.25f,
                d * (0.5f + (y[ibl].scales[ib32] >>  4)) * 0.25f,
            };
            for (int l = 0; l < 4; ++l) {
                if (!(db[l/2] > 0)) continue;
                const float * xg = x + QK_K*ibl + 32*ib32 + 8*l;
                for (int j = 0; j < 8; ++j) xs[j] = signs[4*ib32 + l] & kmask_iq2xs[j] ? -xg[j] : xg[j];
                const int cur = qs[4*ib32 + l] | ((qh[ib32] << (8 - 2*l)) & 0x300);
                const int idx = hq_repick(t, xs, db[l/2], cur);
                qs[4*ib32 + l] = idx & 255;
                qh[ib32] = (qh[ib32] & ~(3 << 2*l)) | ((idx >> 8) << 2*l);
            }
        }
    }
}

static void hq3_xxs_repick(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t n, const hq_dec_grid * t) {
    float xs[4];
    for (int64_t ibl = 0; ibl < n/QK_K; ++ibl) {
        const float d = GGML_FP16_TO_FP32(y[ibl].d);
        uint8_t * qs = y[ibl].qs;
        const uint8_t * scales_and_signs = y[ibl].qs + QK_K/4;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            if (!(db > 0)) continue;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                for (int m = 0; m < 2; ++m) {
                    const float * xg = x + QK_K*ibl + 32*ib32 + 8*l + 4*m;
                    for (int j = 0; j < 4; ++j) xs[j] = signs & kmask_iq2xs[4*m + j] ? -xg[j] : xg[j];
                    qs[8*ib32 + 2*l + m] = hq_repick(t, xs, db, qs[8*ib32 + 2*l + m]);
                }
            }
        }
    }
}

static void hq3_s_repick(const float * GGML_RESTRICT x, block_iq3_s * GGML_RESTRICT y, int64_t n, const hq_dec_grid * t) {
    float xs[4];
    for (int64_t ibl = 0; ibl < n/QK_K; ++ibl) {
        const float d = GGML_FP16_TO_FP32(y[ibl].d);
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const float db = d * (1 + 2*((y[ibl].scales[ib32/2] >> 4*(ib32%2)) & 0xf));
            if (!(db > 0)) continue;
            uint8_t * qs = y[ibl].qs + 8*ib32;
            uint8_t * qh = y[ibl].qh + ib32;
            const uint8_t * signs = y[ibl].signs + 4*ib32;
            for (int l = 0; l < 4; ++l) {
                for (int m = 0; m < 2; ++m) {
                    const int k = 2*l + m;
                    const float * xg = x + QK_K*ibl + 32*ib32 + 8*l + 4*m;
                    for (int j = 0; j < 4; ++j) xs[j] = signs[l] & kmask_iq2xs[4*m + j] ? -xg[j] : xg[j];
                    const int idx = hq_repick(t, xs, db, qs[k] | (((*qh >> k) & 1) << 8));
                    qs[k] = idx & 255;
                    *qh = (*qh & ~(1 << k)) | ((idx >> 8) << k);
                }
            }
        }
    }
}

// ====================== HQ4_NL / HQ4_XS

// idl[ib]: the inverse scale block ib's codes were chosen with
static void quantize_row_hq4_nl_impl(const int super_block_size, const int block_size, const float * GGML_RESTRICT x,
        ggml_fp16_t * dh, uint8_t * q4, uint16_t * scales_h, uint8_t * scales_l,
        float * scales, uint8_t * L, float * idl_out) {

    const int8_t * values = kvalues_iq4nl;
    const int ntry = 7;

    memset(q4, 0, super_block_size/2);
    dh[0] = GGML_FP32_TO_FP16(0.f);

    float max_scale = 0, amax_scale = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        uint8_t * Lb = L + ib*block_size;
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        if (amax < GROUP_MAX_EPS) {
            scales[ib] = 0;
            continue;
        }
        float d = -max/values[0];
        float id = 1/d;
        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < block_size; ++j) {
            float al = id*xb[j];
            int l = best_index_int8(16, values, al);
            Lb[j] = l;
            float q = values[l];
            sumqx += q*xb[j];
            sumq2 += q*q;
        }
        d = sumq2 > 0 ? sumqx/sumq2 : 0.f;
        float best = d*sumqx;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0])/max;
            sumqx = sumq2 = 0;
            for (int j = 0; j < block_size; ++j) {
                float al = id*xb[j];
                int l = best_index_int8(16, values, al);
                float q = values[l];
                sumqx += q*xb[j];
                sumq2 += q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                d = sumqx/sumq2; best = d * sumqx;
            }
        }
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    if (super_block_size/block_size > 1) {
        int nb = super_block_size/block_size;
        memset(scales_h, 0, ((nb+7)/8)*sizeof(uint16_t));
        float d = -max_scale/32;
        dh[0] = GGML_FP32_TO_FP16(d);
        float id = d ? 1/d : 0.f;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            int l = nearest_int(id*scales[ib]);
            l = MAX(-32, MIN(31, l));
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            idl_out[ib] = idl;
            uint8_t * Lb = L + ib*block_size;
            const float * xb = x + ib*block_size;
            for (int j = 0; j < block_size; ++j) {
                Lb[j] = best_index_int8(16, values, idl*xb[j]);
            }
            l += 32;
            uint8_t l_l = l & 0xf;
            uint8_t l_h = l >>  4;
            if (ib%2 == 0) scales_l[ib/2] = l_l;
            else scales_l[ib/2] |= (l_l << 4);
            scales_h[ib/8] |= (l_h << 2*(ib%8));
        }
    } else {
        dh[0] = GGML_FP32_TO_FP16(scales[0]);
        float id = scales[0] ? 1/scales[0] : 0;
        idl_out[0] = id;
        for (int j = 0; j < super_block_size; ++j) {
            L[j] = best_index_int8(16, values, id*x[j]);
        }
    }

    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

// ====================== code layouts of the element-wise types

// the codes of one unit (the type's block), one per value: HQ4_K/HQ5_K/HQ2_K/HQ3_K/HQ6_K as 0..nmax,
// the IQ4 types as table indices, HQ4_0/HQ5_0/HQ8_0 as level + 8/16/128, HQ4_1/HQ5_1 as 0..nmax
static void hq_get_codes(enum ggml_type type, const void * GGML_RESTRICT y, uint8_t * GGML_RESTRICT L, int unit) {
    switch (type) {
        case GGML_TYPE_HQ4_K:
        case GGML_TYPE_HQ5_K:
            {
                const uint8_t * qs = type == GGML_TYPE_HQ4_K ? ((const block_q4_K *) y)->qs : ((const block_q5_K *) y)->qs;
                for (int j = 0; j < QK_K; ++j) {
                    const uint8_t q = qs[(j/64)*32 + j%32];
                    L[j] = j%64 < 32 ? q & 0xF : q >> 4;
                }
                if (type == GGML_TYPE_HQ5_K) {
                    const uint8_t * qh = ((const block_q5_K *) y)->qh;
                    for (int j = 0; j < QK_K; ++j) {
                        L[j] |= ((qh[j%32] >> (2*(j/64) + (j%64)/32)) & 1) << 4;
                    }
                }
            } break;
        case GGML_TYPE_HQ4_XS:
        case GGML_TYPE_HQ4_NL:
        case GGML_TYPE_HQ4_0:
        case GGML_TYPE_HQ4_1:
        case GGML_TYPE_HQ5_0:
        case GGML_TYPE_HQ5_1:
            {
                const uint8_t * qs =
                    type == GGML_TYPE_HQ4_XS ? ((const block_iq4_xs *) y)->qs :
                    type == GGML_TYPE_HQ4_NL ? ((const block_iq4_nl *) y)->qs :
                    type == GGML_TYPE_HQ4_0  ? ((const block_q4_0 *) y)->qs :
                    type == GGML_TYPE_HQ4_1  ? ((const block_q4_1 *) y)->qs :
                    type == GGML_TYPE_HQ5_0  ? ((const block_q5_0 *) y)->qs : ((const block_q5_1 *) y)->qs;
                for (int j = 0; j < unit; ++j) {
                    const uint8_t q = qs[16*(j/32) + j%16];
                    L[j] = j%32 < 16 ? q & 0xF : q >> 4;
                }
                if (type == GGML_TYPE_HQ5_0 || type == GGML_TYPE_HQ5_1) {
                    uint32_t qh;
                    memcpy(&qh, type == GGML_TYPE_HQ5_0 ? ((const block_q5_0 *) y)->qh : ((const block_q5_1 *) y)->qh, sizeof(qh));
                    for (int j = 0; j < QK5_0; ++j) {
                        L[j] |= ((qh >> j) & 1) << 4;
                    }
                }
            } break;
        case GGML_TYPE_HQ8_0:
            for (int j = 0; j < QK8_0; ++j) {
                L[j] = (uint8_t) (((const block_q8_0 *) y)->qs[j] + 128);
            }
            break;
        case GGML_TYPE_HQ2_K:
        case GGML_TYPE_HQ3_K:
            {
                const uint8_t * qs = type == GGML_TYPE_HQ2_K ? ((const block_q2_K *) y)->qs : ((const block_q3_K *) y)->qs;
                for (int j = 0; j < QK_K; ++j) {
                    L[j] = (qs[32*(j/128) + j%32] >> 2*((j%128)/32)) & 3;
                }
                if (type == GGML_TYPE_HQ3_K) {
                    const uint8_t * hmask = ((const block_q3_K *) y)->hmask;
                    for (int j = 0; j < QK_K; ++j) {
                        L[j] |= ((hmask[j%32] >> (j/32)) & 1) << 2;
                    }
                }
            } break;
        case GGML_TYPE_HQ6_K:
            {
                const block_q6_K * b = y;
                for (int j = 0; j < QK_K; ++j) {
                    const int c = j/128, l = j%32, q = (j%128)/32;
                    const int lo = (b->ql[64*c + l + 32*(q%2)] >> 4*(q/2)) & 0xF;
                    const int hi = (b->qh[32*c + l] >> 2*q) & 3;
                    L[j] = (uint8_t) (lo | (hi << 4));
                }
            } break;
        default: GGML_ABORT("fatal error");
    }
}

static void hq_set_codes(enum ggml_type type, void * GGML_RESTRICT y, const uint8_t * GGML_RESTRICT L, int unit) {
    switch (type) {
        case GGML_TYPE_HQ4_K:
        case GGML_TYPE_HQ5_K:
            {
                uint8_t * qs = type == GGML_TYPE_HQ4_K ? ((block_q4_K *) y)->qs : ((block_q5_K *) y)->qs;
                for (int j = 0; j < QK_K; j += 64) {
                    for (int l = 0; l < 32; ++l) {
                        qs[j/2 + l] = (L[j + l] & 0xF) | ((L[j + l + 32] & 0xF) << 4);
                    }
                }
                if (type == GGML_TYPE_HQ5_K) {
                    uint8_t * qh = ((block_q5_K *) y)->qh;
                    memset(qh, 0, QK_K/8);
                    for (int j = 0; j < QK_K; ++j) {
                        qh[j%32] |= (L[j] >> 4) << (2*(j/64) + (j%64)/32);
                    }
                }
            } break;
        case GGML_TYPE_HQ4_XS:
        case GGML_TYPE_HQ4_NL:
        case GGML_TYPE_HQ4_0:
        case GGML_TYPE_HQ4_1:
        case GGML_TYPE_HQ5_0:
        case GGML_TYPE_HQ5_1:
            {
                uint8_t * qs =
                    type == GGML_TYPE_HQ4_XS ? ((block_iq4_xs *) y)->qs :
                    type == GGML_TYPE_HQ4_NL ? ((block_iq4_nl *) y)->qs :
                    type == GGML_TYPE_HQ4_0  ? ((block_q4_0 *) y)->qs :
                    type == GGML_TYPE_HQ4_1  ? ((block_q4_1 *) y)->qs :
                    type == GGML_TYPE_HQ5_0  ? ((block_q5_0 *) y)->qs : ((block_q5_1 *) y)->qs;
                for (int i = 0; i < unit/32; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        qs[16*i + j] = (L[32*i + j] & 0xF) | ((L[32*i + 16 + j] & 0xF) << 4);
                    }
                }
                if (type == GGML_TYPE_HQ5_0 || type == GGML_TYPE_HQ5_1) {
                    uint32_t qh = 0;
                    for (int j = 0; j < QK5_0; ++j) {
                        qh |= (uint32_t) ((L[j] >> 4) & 1) << j;
                    }
                    memcpy(type == GGML_TYPE_HQ5_0 ? ((block_q5_0 *) y)->qh : ((block_q5_1 *) y)->qh, &qh, sizeof(qh));
                }
            } break;
        case GGML_TYPE_HQ8_0:
            for (int j = 0; j < QK8_0; ++j) {
                ((block_q8_0 *) y)->qs[j] = (int8_t) (L[j] - 128);
            }
            break;
        case GGML_TYPE_HQ2_K:
        case GGML_TYPE_HQ3_K:
            {
                uint8_t * qs = type == GGML_TYPE_HQ2_K ? ((block_q2_K *) y)->qs : ((block_q3_K *) y)->qs;
                for (int j = 0; j < QK_K; j += 128) {
                    for (int l = 0; l < 32; ++l) {
                        qs[j/4 + l] = (L[j + l] & 3) | ((L[j + l + 32] & 3) << 2) | ((L[j + l + 64] & 3) << 4) | ((L[j + l + 96] & 3) << 6);
                    }
                }
                if (type == GGML_TYPE_HQ3_K) {
                    uint8_t * hmask = ((block_q3_K *) y)->hmask;
                    memset(hmask, 0, QK_K/8);
                    for (int j = 0; j < QK_K; ++j) {
                        hmask[j%32] |= ((L[j] >> 2) & 1) << (j/32);
                    }
                }
            } break;
        case GGML_TYPE_HQ6_K:
            {
                uint8_t * ql = ((block_q6_K *) y)->ql;
                uint8_t * qh = ((block_q6_K *) y)->qh;
                for (int j = 0; j < QK_K; j += 128) {
                    for (int l = 0; l < 32; ++l) {
                        ql[l +  0] = (L[j + l +  0] & 0xF) | ((L[j + l + 64] & 0xF) << 4);
                        ql[l + 32] = (L[j + l + 32] & 0xF) | ((L[j + l + 96] & 0xF) << 4);
                        qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
                    }
                    ql += 64;
                    qh += 32;
                }
            } break;
        default: GGML_ABORT("fatal error");
    }
}

// ====================== HQ4_0 / HQ4_1 / HQ5_0 / HQ5_1 / HQ8_0
//
// The base _impls' scale searches with uniform weights (Q8_0's scale is absmax/127, as in its base).
// Unlike the base quantizers, every code is then re-picked at the stored fp16 scale, as the K-quant
// _impls do, so a code never decodes worse than the one the search chose.

static_assert(offsetof(block_q4_0, d) == 0 && offsetof(block_q5_0, d) == 0 && offsetof(block_q4_1, m) == sizeof(ggml_half) &&
              offsetof(block_q5_1, m) == sizeof(ggml_half), "every legacy block starts with its fp16 d (and m)");

// nmax: 8 for HQ4_0, 16 for HQ5_0
static void quantize_row_hq_0_impl(enum ggml_type type, int nmax, const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n) {
    int8_t  Lq[32];
    uint8_t L[32];
    const size_t bs = ggml_type_size(type);
    for (int64_t ib = 0; ib < n/32; ++ib) {
        const float * xb = x + 32*ib;
        char      * yb = (char *) vy + ib*bs;
        ggml_half * dh = (ggml_half *) yb;
        dh[0] = GGML_FP32_TO_FP16(hq_make_qx_quants(32, nmax, xb, Lq));
        const float d = GGML_FP16_TO_FP32(dh[0]);
        for (int j = 0; j < 32; ++j) {
            L[j] = d ? hq_s_code(xb[j], d, -nmax, nmax - 1) + nmax : Lq[j];
        }
        hq_set_codes(type, yb, L, 32);
    }
}

static void quantize_row_hq8_0_impl(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t n) {
    uint8_t L[QK8_0];
    for (int64_t ib = 0; ib < n/QK8_0; ++ib) {
        const float * xb = x + QK8_0*ib;
        float amax = 0;
        for (int j = 0; j < QK8_0; ++j) {
            amax = MAX(amax, fabsf(xb[j]));
        }
        y[ib].d = GGML_FP32_TO_FP16(amax/127);
        const float d = GGML_FP16_TO_FP32(y[ib].d);
        for (int j = 0; j < QK8_0; ++j) {
            L[j] = d ? hq_s_code(xb[j], d, -128, 127) + 128 : 128;
        }
        hq_set_codes(GGML_TYPE_HQ8_0, y + ib, L, QK8_0);
    }
}

// nmax: 15 for HQ4_1, 31 for HQ5_1
static void quantize_row_hq_1_impl(enum ggml_type type, int nmax, const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n) {
    uint8_t L[32], Laux[32];
    const size_t bs = ggml_type_size(type);
    for (int64_t ib = 0; ib < n/32; ++ib) {
        const float * xb = x + 32*ib;
        char      * yb = (char *) vy + ib*bs;
        ggml_half * dm = (ggml_half *) yb;
        float min;
        const float scale = make_qkx3_quants(32, nmax, xb, L, &min, Laux, -0.9f, 0.05f, 36);
        dm[0] = GGML_FP32_TO_FP16(scale);
        dm[1] = GGML_FP32_TO_FP16(-min);
        const float d = GGML_FP16_TO_FP32(dm[0]);
        const float m = GGML_FP16_TO_FP32(dm[1]);
        if (d) {
            for (int j = 0; j < 32; ++j) {
                L[j] = hq_k_code(xb[j], d, -m, nmax);
            }
        }
        hq_set_codes(type, yb, L, 32);
    }
}

// ====================== HQ2_K / HQ3_K / HQ6_K
//
// The base _impls with uniform weights: the same searches, and the same final pass that re-picks every
// code at the stored sub-block scales.

static void quantize_row_hq2_K_impl(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t n) {
    uint8_t L[QK_K];
    uint8_t Laux[16];
    uint8_t Ls[QK_K/16], Lm[QK_K/16];
    float   mins[QK_K/16];
    float   scales[QK_K/16];

    for (int64_t i = 0; i < n/QK_K; ++i) {
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = make_qkx3_quants(16, 3, x + 16*j, L + 16*j, &mins[j], Laux, -0.9f, 0.05f, 36);
        }
        y[i].d    = GGML_FP32_TO_FP16(make_qp_quants(QK_K/16, 15, scales, Ls));
        y[i].dmin = GGML_FP32_TO_FP16(make_qp_quants(QK_K/16, 15, mins,   Lm));
        for (int j = 0; j < QK_K/16; ++j) {
            y[i].scales[j] = Ls[j] | (Lm[j] << 4);
        }
        const float dd   = GGML_FP16_TO_FP32(y[i].d);
        const float dmin = GGML_FP16_TO_FP32(y[i].dmin);
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = dd * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float m = dmin * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                L[16*j + ii] = hq_k_code(x[16*j + ii], d, m, 3);
            }
        }
        hq_set_codes(GGML_TYPE_HQ2_K, y + i, L, QK_K);
        x += QK_K;
    }
}

// the signed 6-bit scale of sub-block j
static inline int hq_q3_K_scale(const uint8_t * GGML_RESTRICT scales, int j) {
    const int sc = j < 8 ? scales[j] & 0xF : scales[j-8] >> 4;
    return (sc | (((scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32;
}

static void quantize_row_hq3_K_impl(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t n) {
    int8_t  Lq[QK_K];
    uint8_t L[QK_K];
    float   scales[QK_K/16];
    int8_t  Ls[QK_K/16];

    for (int64_t i = 0; i < n/QK_K; ++i) {
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = hq_make_qx_quants(16, 4, x + 16*j, Lq + 16*j);
        }
        memset(y[i].scales, 0, 12);
        const float d_block = hq_make_qx_quants(QK_K/16, 32, scales, Ls);
        for (int j = 0; j < QK_K/16; ++j) {
            int l = Ls[j];
            if (j < 8) {
                y[i].scales[j] = l & 0xF;
            } else {
                y[i].scales[j-8] |= ((l & 0xF) << 4);
            }
            l >>= 4;
            y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
        }
        y[i].d = GGML_FP32_TO_FP16(d_block);
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = GGML_FP16_TO_FP32(y[i].d) * hq_q3_K_scale(y[i].scales, j);
            for (int ii = 0; ii < 16; ++ii) {
                L[16*j + ii] = d ? hq_s_code(x[16*j + ii], d, -4, 3) + 4 : Lq[16*j + ii];
            }
        }
        hq_set_codes(GGML_TYPE_HQ3_K, y + i, L, QK_K);
        x += QK_K;
    }
}

static void quantize_row_hq6_K_impl(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t n) {
    int8_t  Lq[QK_K];
    uint8_t L[QK_K];
    float   scales[QK_K/16];

    for (int64_t i = 0; i < n/QK_K; ++i) {
        float max_scale = 0;
        float max_abs_scale = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float scale = hq_make_qx_quants(16, 32, x + 16*ib, Lq + 16*ib);
            scales[ib] = scale;
            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }
        if (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            x += QK_K;
            continue;
        }
        const float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            for (int ii = 0; ii < 16; ++ii) {
                L[16*j + ii] = d ? hq_s_code(x[16*j + ii], d, -32, 31) + 32 : Lq[16*j + ii];
            }
        }
        hq_set_codes(GGML_TYPE_HQ6_K, y + i, L, QK_K);
        x += QK_K;
    }
}

// ====================== the HQ driver
//
// Rows are encoded one unit at a time, the type's block (the 256-value super-block; the 32-value block for
// HQ4_NL and the legacy types), by the _impls above. Without a factor that's all, apart from the codebook re-pick. With the packed
// upper Cholesky factor U of H^-1, GPTQ error feedback goes in between: each element is re-picked in
// order at its unit's scales, and its error is fed at once into the rest of its lazy block of 256
// columns, and into the columns after the block once the block is done.

#define HQ_GPTQ_ROWS 32

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && !defined(__AVX2__)
#define HQ_AVX2_CLONE
#endif

#if defined(__GNUC__) || defined(__clang__)
#define HQ_INLINE inline __attribute__((always_inline))
#else
#define HQ_INLINE inline
#endif

typedef struct {
    float   idl[QK_K/32];   // IQ4: the inverse scale each 32-block's codes were chosen with
    uint8_t smode[QK_K/16]; // codebook types: how each sub-block's signs were set (HQ_SIGNS_*)
} hq_unit_info;

static void hq_encode_unit(enum ggml_type type, const float * GGML_RESTRICT x, void * GGML_RESTRICT y,
                           hq_unit_info * info, const hq_dec_grid * dg, bool repick) {
    uint8_t L[QK_K];
    float   scales[QK_K/32];
    switch (type) {
        case GGML_TYPE_HQ4_K: quantize_row_hq4_K_impl(x, y, QK_K); break;
        case GGML_TYPE_HQ5_K: quantize_row_hq5_K_impl(x, y, QK_K); break;
        case GGML_TYPE_HQ4_XS:
            {
                block_iq4_xs * b = y;
                quantize_row_hq4_nl_impl(QK_K, 32, x, &b->d, b->qs, &b->scales_h, b->scales_l, scales, L, info->idl);
            } break;
        case GGML_TYPE_HQ4_NL:
            {
                block_iq4_nl * b = y;
                uint16_t unused_h;
                quantize_row_hq4_nl_impl(QK4_NL, 32, x, &b->d, b->qs, &unused_h, NULL, scales, L, info->idl);
            } break;
        case GGML_TYPE_HQ4_0: quantize_row_hq_0_impl(type,  8, x, y, QK4_0); break;
        case GGML_TYPE_HQ4_1: quantize_row_hq_1_impl(type, 15, x, y, QK4_1); break;
        case GGML_TYPE_HQ5_0: quantize_row_hq_0_impl(type, 16, x, y, QK5_0); break;
        case GGML_TYPE_HQ5_1: quantize_row_hq_1_impl(type, 31, x, y, QK5_1); break;
        case GGML_TYPE_HQ8_0: quantize_row_hq8_0_impl(x, y, QK8_0); break;
        case GGML_TYPE_HQ2_K: quantize_row_hq2_K_impl(x, y, QK_K); break;
        case GGML_TYPE_HQ3_K: quantize_row_hq3_K_impl(x, y, QK_K); break;
        case GGML_TYPE_HQ6_K: quantize_row_hq6_K_impl(x, y, QK_K); break;
        case GGML_TYPE_HQ2_XXS:
            quantize_row_hq2_xxs_impl(x, y, QK_K, info->smode);
            if (repick) hq2_xxs_repick(x, y, QK_K, dg);
            break;
        case GGML_TYPE_HQ2_XS:
            quantize_row_hq2_xs_impl(x, y, QK_K, info->smode);
            if (repick) hq2_xs_repick(x, y, QK_K, dg);
            break;
        case GGML_TYPE_HQ2_S:
            quantize_row_hq2_s_impl(x, y, QK_K, info->smode);
            if (repick) hq2_s_repick(x, y, QK_K, dg);
            break;
        case GGML_TYPE_HQ3_XXS:
            quantize_row_hq3_xxs_impl(x, y, QK_K, info->smode);
            if (repick) hq3_xxs_repick(x, y, QK_K, dg);
            break;
        case GGML_TYPE_HQ3_S:
            quantize_row_hq3_s_impl(x, y, QK_K, info->smode);
            if (repick) hq3_s_repick(x, y, QK_K, dg);
            break;
        default: GGML_ABORT("fatal error");
    }
}

// U[j][k] = hq_urow(U, n, j)[k]
static inline const float * hq_urow(const float * U, int64_t n, int64_t j) {
    return U + ((size_t) j*n - (size_t) j*(j - 1)/2 - (size_t) j);
}

// how an element-wise unit decodes: per scale (sub values), d*(code + lmin) for a symmetric type, d*code + m for
// an affine one (lmin = 0); the IQ4 types decode d*kvalues_iq4nl[code]
typedef struct {
    int   sub;
    int   lmin, lmax;
    bool  affine;
    float d[QK_K/16];
    float m[QK_K/16];
} hq_levels;

static void hq_get_levels(enum ggml_type type, const void * GGML_RESTRICT vy, hq_levels * lv) {
    lv->affine = false;
    lv->lmin   = 0;
    switch (type) {
        case GGML_TYPE_HQ4_K:
        case GGML_TYPE_HQ5_K:
            {
                const block_q4_K * b4 = vy;
                const block_q5_K * b5 = vy;
                const float     dd     = GGML_FP16_TO_FP32(type == GGML_TYPE_HQ4_K ? b4->d    : b5->d);
                const float     dmin   = GGML_FP16_TO_FP32(type == GGML_TYPE_HQ4_K ? b4->dmin : b5->dmin);
                const uint8_t * scales = type == GGML_TYPE_HQ4_K ? b4->scales : b5->scales;
                lv->sub = 32; lv->affine = true; lv->lmax = type == GGML_TYPE_HQ4_K ? 15 : 31;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    uint8_t sc, m;
                    get_scale_min_k4(ib, scales, &sc, &m);
                    lv->d[ib] = dd * sc;
                    lv->m[ib] = -(dmin * m);
                }
            } break;
        case GGML_TYPE_HQ4_XS:
            {
                const block_iq4_xs * b = vy;
                lv->sub = 32;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    const int ls = ((b->scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((b->scales_h >> 2*ib) & 3) << 4);
                    lv->d[ib] = GGML_FP16_TO_FP32(b->d) * (ls - 32);
                }
            } break;
        case GGML_TYPE_HQ4_NL: lv->sub = 32; lv->d[0] = GGML_FP16_TO_FP32(((const block_iq4_nl *) vy)->d); break;
        case GGML_TYPE_HQ4_0:  lv->sub = 32; lv->lmin = -8;   lv->lmax = 7;   lv->d[0] = GGML_FP16_TO_FP32(((const block_q4_0 *) vy)->d); break;
        case GGML_TYPE_HQ5_0:  lv->sub = 32; lv->lmin = -16;  lv->lmax = 15;  lv->d[0] = GGML_FP16_TO_FP32(((const block_q5_0 *) vy)->d); break;
        case GGML_TYPE_HQ8_0:  lv->sub = 32; lv->lmin = -128; lv->lmax = 127; lv->d[0] = GGML_FP16_TO_FP32(((const block_q8_0 *) vy)->d); break;
        case GGML_TYPE_HQ4_1:
        case GGML_TYPE_HQ5_1:
            {
                const block_q4_1 * b4 = vy;
                const block_q5_1 * b5 = vy;
                lv->sub = 32; lv->affine = true; lv->lmax = type == GGML_TYPE_HQ4_1 ? 15 : 31;
                lv->d[0] = GGML_FP16_TO_FP32(type == GGML_TYPE_HQ4_1 ? b4->d : b5->d);
                lv->m[0] = GGML_FP16_TO_FP32(type == GGML_TYPE_HQ4_1 ? b4->m : b5->m);
            } break;
        case GGML_TYPE_HQ2_K:
            {
                const block_q2_K * b = vy;
                const float dd   = GGML_FP16_TO_FP32(b->d);
                const float dmin = GGML_FP16_TO_FP32(b->dmin);
                lv->sub = 16; lv->affine = true; lv->lmax = 3;
                for (int j = 0; j < QK_K/16; ++j) {
                    lv->d[j] = dd * (b->scales[j] & 0xF);
                    lv->m[j] = -(dmin * (b->scales[j] >> 4));
                }
            } break;
        case GGML_TYPE_HQ3_K:
            {
                const block_q3_K * b = vy;
                lv->sub = 16; lv->lmin = -4; lv->lmax = 3;
                for (int j = 0; j < QK_K/16; ++j) {
                    lv->d[j] = GGML_FP16_TO_FP32(b->d) * hq_q3_K_scale(b->scales, j);
                }
            } break;
        case GGML_TYPE_HQ6_K:
            {
                const block_q6_K * b = vy;
                lv->sub = 16; lv->lmin = -32; lv->lmax = 31;
                for (int j = 0; j < QK_K/16; ++j) {
                    lv->d[j] = GGML_FP16_TO_FP32(b->d) * b->scales[j];
                }
            } break;
        default: GGML_ABORT("fatal error");
    }
}

// the element-wise step over one unit [j0, j0 + unit): each code at the unit's scales with the rule its
// _impl used, its error fed into the rest of the block [.., b1) and kept in e (indexed from b0)
static void hq_gptq_elements(enum ggml_type type, float * GGML_RESTRICT x, void * GGML_RESTRICT y, const hq_unit_info * info,
                             int64_t j0, int unit, int64_t b0, int64_t b1, const float * GGML_RESTRICT U, int64_t n,
                             float * GGML_RESTRICT e) {
    uint8_t   L[QK_K];
    hq_levels lv;
    const bool iq4 = type == GGML_TYPE_HQ4_XS || type == GGML_TYPE_HQ4_NL;

    hq_get_codes(type, y, L, unit);
    hq_get_levels(type, y, &lv);

    for (int i = 0; i < unit; ++i) {
        const int64_t j  = j0 + i;
        const int     sb = i/lv.sub;
        const float   d  = lv.d[sb];
        float xh;
        if (iq4) {
            L[i] = best_index_int8(16, kvalues_iq4nl, info->idl[sb]*x[j]);
            xh = d*kvalues_iq4nl[L[i]];
        } else if (lv.affine) {
            if (d) {
                L[i] = hq_k_code(x[j], d, -lv.m[sb], lv.lmax);
            }
            xh = d*L[i] + lv.m[sb];
        } else {
            if (d) {
                L[i] = hq_s_code(x[j], d, lv.lmin, lv.lmax) - lv.lmin;
            }
            xh = d*(L[i] + lv.lmin);
        }
        const float * uj  = hq_urow(U, n, j);
        const float   err = (x[j] - xh)/uj[j];
        e[j - b0] = err;
        for (int64_t k = j + 1; k < b1; ++k) {
            x[k] -= err*uj[k];
        }
    }
    hq_set_codes(type, y, L, unit);
}

// ---- the codebook types, one 8-value sign group g (0..31) of a super-block at a time

static bool hq_parity_signs(enum ggml_type type) {
    return type == GGML_TYPE_HQ2_XXS || type == GGML_TYPE_HQ2_XS || type == GGML_TYPE_HQ3_XXS;
}

// the scale of group g, as dequantize_row_iq* computes it
static float hq_cb_db(enum ggml_type type, const void * GGML_RESTRICT vy, int g) {
    const int ib32 = g/4, l = g%4;
    switch (type) {
        case GGML_TYPE_HQ2_XXS:
            {
                const block_iq2_xxs * y = vy;
                uint32_t aux32[2];
                memcpy(aux32, y->qs + 4*ib32, 2*sizeof(uint32_t));
                return GGML_FP16_TO_FP32(y->d) * (0.5f + (aux32[1] >> 28)) * 0.25f;
            }
        case GGML_TYPE_HQ2_XS:
            {
                const block_iq2_xs * y = vy;
                return GGML_FP16_TO_FP32(y->d) * (0.5f + ((y->scales[ib32] >> 4*(l/2)) & 0xf)) * 0.25f;
            }
        case GGML_TYPE_HQ2_S:
            {
                const block_iq2_s * y = vy;
                return GGML_FP16_TO_FP32(y->d) * (0.5f + ((y->scales[ib32] >> 4*(l/2)) & 0xf)) * 0.25f;
            }
        case GGML_TYPE_HQ3_XXS:
            {
                const block_iq3_xxs * y = vy;
                uint32_t aux32;
                memcpy(&aux32, y->qs + QK_K/4 + 4*ib32, sizeof(uint32_t));
                return GGML_FP16_TO_FP32(y->d) * (0.5f + (aux32 >> 28)) * 0.5f;
            }
        case GGML_TYPE_HQ3_S:
            {
                const block_iq3_s * y = vy;
                return GGML_FP16_TO_FP32(y->d) * (1 + 2*((y->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
            }
        default: GGML_ABORT("fatal error");
    }
}

// the stored sign bits of group g: 7 for the parity types, else 8
static uint8_t hq_cb_get_signs(enum ggml_type type, const void * GGML_RESTRICT vy, int g) {
    const int ib32 = g/4, l = g%4;
    switch (type) {
        case GGML_TYPE_HQ2_XXS:
            {
                uint32_t aux32;
                memcpy(&aux32, ((const block_iq2_xxs *) vy)->qs + 4*ib32 + 2, sizeof(uint32_t));
                return (aux32 >> 7*l) & 127;
            }
        case GGML_TYPE_HQ2_XS:  return ((const block_iq2_xs *) vy)->qs[g] >> 9;
        case GGML_TYPE_HQ2_S:   return ((const block_iq2_s *) vy)->qs[QK_K/8 + g];
        case GGML_TYPE_HQ3_XXS:
            {
                uint32_t aux32;
                memcpy(&aux32, ((const block_iq3_xxs *) vy)->qs + QK_K/4 + 4*ib32, sizeof(uint32_t));
                return (aux32 >> 7*l) & 127;
            }
        case GGML_TYPE_HQ3_S:   return ((const block_iq3_s *) vy)->signs[g];
        default: GGML_ABORT("fatal error");
    }
}

static void hq_cb_set_signs(enum ggml_type type, void * GGML_RESTRICT vy, int g, uint8_t s) {
    const int ib32 = g/4, l = g%4;
    switch (type) {
        case GGML_TYPE_HQ2_XXS:
            {
                uint16_t * q = ((block_iq2_xxs *) vy)->qs + 4*ib32 + 2;
                uint32_t aux32;
                memcpy(&aux32, q, sizeof(uint32_t));
                aux32 = (aux32 & ~(127u << 7*l)) | ((uint32_t) (s & 127) << 7*l);
                memcpy(q, &aux32, sizeof(uint32_t));
            } break;
        case GGML_TYPE_HQ2_XS:
            {
                uint16_t * q = ((block_iq2_xs *) vy)->qs + g;
                *q = (uint16_t) ((*q & 511) | ((s & 127) << 9));
            } break;
        case GGML_TYPE_HQ2_S: ((block_iq2_s *) vy)->qs[QK_K/8 + g] = s; break;
        case GGML_TYPE_HQ3_XXS:
            {
                uint8_t * q = ((block_iq3_xxs *) vy)->qs + QK_K/4 + 4*ib32;
                uint32_t aux32;
                memcpy(&aux32, q, sizeof(uint32_t));
                aux32 = (aux32 & ~(127u << 7*l)) | ((uint32_t) (s & 127) << 7*l);
                memcpy(q, &aux32, sizeof(uint32_t));
            } break;
        case GGML_TYPE_HQ3_S: ((block_iq3_s *) vy)->signs[g] = s; break;
        default: GGML_ABORT("fatal error");
    }
}

// grid point m of group g (one for IQ2, two for IQ3)
static int hq_cb_get_code(enum ggml_type type, const void * GGML_RESTRICT vy, int g, int m) {
    const int ib32 = g/4, l = g%4;
    switch (type) {
        case GGML_TYPE_HQ2_XXS: return ((const uint8_t *) (((const block_iq2_xxs *) vy)->qs + 4*ib32))[l];
        case GGML_TYPE_HQ2_XS:  return ((const block_iq2_xs *) vy)->qs[g] & 511;
        case GGML_TYPE_HQ2_S:
            {
                const block_iq2_s * y = vy;
                return y->qs[g] | ((y->qh[ib32] << (8 - 2*l)) & 0x300);
            }
        case GGML_TYPE_HQ3_XXS: return ((const block_iq3_xxs *) vy)->qs[2*g + m];
        case GGML_TYPE_HQ3_S:
            {
                const block_iq3_s * y = vy;
                const int k = 2*l + m;
                return y->qs[8*ib32 + k] | (((y->qh[ib32] >> k) & 1) << 8);
            }
        default: GGML_ABORT("fatal error");
    }
}

static void hq_cb_set_code(enum ggml_type type, void * GGML_RESTRICT vy, int g, int m, int idx) {
    const int ib32 = g/4, l = g%4;
    switch (type) {
        case GGML_TYPE_HQ2_XXS: ((uint8_t *) (((block_iq2_xxs *) vy)->qs + 4*ib32))[l] = (uint8_t) idx; break;
        case GGML_TYPE_HQ2_XS:
            {
                uint16_t * q = ((block_iq2_xs *) vy)->qs + g;
                *q = (uint16_t) ((*q & ~511) | idx);
            } break;
        case GGML_TYPE_HQ2_S:
            {
                block_iq2_s * y = vy;
                y->qs[g] = idx & 255;
                y->qh[ib32] = (y->qh[ib32] & ~(3 << 2*l)) | ((idx >> 8) << 2*l);
            } break;
        case GGML_TYPE_HQ3_XXS: ((block_iq3_xxs *) vy)->qs[2*g + m] = (uint8_t) idx; break;
        case GGML_TYPE_HQ3_S:
            {
                block_iq3_s * y = vy;
                const int k = 2*l + m;
                y->qs[8*ib32 + k] = idx & 255;
                y->qh[ib32] = (y->qh[ib32] & ~(1 << k)) | ((idx >> 8) << k);
            } break;
        default: GGML_ABORT("fatal error");
    }
}

// the group step over one super-block [j0, j0 + QK_K) [QuIP#'s g-block LDLQ, g = 8]: each group's
// signs by its sub-block's rule on the current values, its codes by the exact search at the group's
// scale, then e with U_gg^T e = x_g - xhat_g fed into the rest of the block [.., b1) and kept in e
static void hq_gptq_groups(enum ggml_type type, float * GGML_RESTRICT x, void * GGML_RESTRICT y, const hq_unit_info * info,
                           const hq_dec_grid * dg, int64_t j0, int64_t b0, int64_t b1, const float * GGML_RESTRICT U, int64_t n,
                           float * GGML_RESTRICT e) {
    const bool parity = hq_parity_signs(type);
    const int  sub    = type == GGML_TYPE_HQ2_XS || type == GGML_TYPE_HQ2_S ? 16 : 32;
    const int  gs     = dg->gs;

    for (int g = 0; g < QK_K/8; ++g) {
        const int64_t jg = j0 + 8*g;
        float * xg = x + jg;

        uint8_t s;
        const int mode = info->smode[8*g/sub];
        if (mode == HQ_SIGNS_KEEP) {
            s = hq_cb_get_signs(type, y, g);
        } else {
            float xval[8];
            s = hq_signs8(xg, xval, parity);
            if (mode == HQ_SIGNS_INVERT) {
                s = ~s;
            }
            hq_cb_set_signs(type, y, g, s);
        }
        if (parity) {
            s = ksigns_iq2xs[s & 127];
        }

        float xh[8] = { 0 };
        const float db = hq_cb_db(type, y, g);
        if (db > 0) {
            float xs[8];
            for (int i = 0; i < 8; ++i) {
                xs[i] = s & kmask_iq2xs[i] ? -xg[i] : xg[i];
            }
            for (int m = 0; m < 8/gs; ++m) {
                const int idx = hq_repick(dg, xs + gs*m, db, hq_cb_get_code(type, y, g, m));
                hq_cb_set_code(type, y, g, m, idx);
                for (int i = 0; i < gs; ++i) {
                    const int k = gs*m + i;
                    xh[k] = db * dg->grid[gs*idx + i] * (s & kmask_iq2xs[k] ? -1.f : 1.f);
                }
            }
        }

        float eg[8];
        for (int a = 0; a < 8; ++a) {
            float acc = xg[a] - xh[a];
            for (int b = 0; b < a; ++b) {
                acc -= hq_urow(U, n, jg + b)[jg + a]*eg[b];
            }
            eg[a] = acc/hq_urow(U, n, jg + a)[jg + a];
            e[jg + a - b0] = eg[a];
        }
        for (int a = 0; a < 8; ++a) {
            const float * ua = hq_urow(U, n, jg + a);
            for (int64_t k = jg + 8; k < b1; ++k) {
                x[k] -= eg[a]*ua[k];
            }
        }
    }
}

// x[r][k] -= sum_jj e[r][jj]*u[jj][k] for k in [b1, n) and r < nr; e holds zero rows up to a multiple of 4
static HQ_INLINE void hq_gptq_update_body(float * GGML_RESTRICT x, int64_t n, int nr, const float * GGML_RESTRICT e,
                                          const float * const * u, int64_t nb, int64_t b1) {
    for (int64_t k = b1; k < n; k += 16) {
        for (int r = 0; r < nr; r += 4) {
            const float * e0 = e + (r + 0)*QK_K;
            const float * e1 = e + (r + 1)*QK_K;
            const float * e2 = e + (r + 2)*QK_K;
            const float * e3 = e + (r + 3)*QK_K;
            float out[4][16];
#if defined(__GNUC__) || defined(__clang__)
            typedef float hq_v8 __attribute__((vector_size(32)));
            hq_v8 a00 = { 0 }, a01 = { 0 }, a10 = { 0 }, a11 = { 0 }, a20 = { 0 }, a21 = { 0 }, a30 = { 0 }, a31 = { 0 };
            for (int64_t jj = 0; jj < nb; ++jj) {
                hq_v8 u0, u1;
                memcpy(&u0, u[jj] + k,     sizeof(u0));
                memcpy(&u1, u[jj] + k + 8, sizeof(u1));
                a00 += e0[jj]*u0; a01 += e0[jj]*u1;
                a10 += e1[jj]*u0; a11 += e1[jj]*u1;
                a20 += e2[jj]*u0; a21 += e2[jj]*u1;
                a30 += e3[jj]*u0; a31 += e3[jj]*u1;
            }
            memcpy(out[0], &a00, 32); memcpy(out[0] + 8, &a01, 32);
            memcpy(out[1], &a10, 32); memcpy(out[1] + 8, &a11, 32);
            memcpy(out[2], &a20, 32); memcpy(out[2] + 8, &a21, 32);
            memcpy(out[3], &a30, 32); memcpy(out[3] + 8, &a31, 32);
#else
            memset(out, 0, sizeof(out));
            for (int64_t jj = 0; jj < nb; ++jj) {
                for (int l = 0; l < 16; ++l) {
                    out[0][l] += e0[jj]*u[jj][k + l];
                    out[1][l] += e1[jj]*u[jj][k + l];
                    out[2][l] += e2[jj]*u[jj][k + l];
                    out[3][l] += e3[jj]*u[jj][k + l];
                }
            }
#endif
            for (int i = 0; i < 4 && r + i < nr; ++i) {
                float * xr = x + (r + i)*n + k;
                for (int l = 0; l < 16; ++l) {
                    xr[l] -= out[i][l];
                }
            }
        }
    }
}

static void hq_gptq_update_default(float * x, int64_t n, int nr, const float * e, const float * const * u, int64_t nb, int64_t b1) {
    hq_gptq_update_body(x, n, nr, e, u, nb, b1);
}

#ifdef HQ_AVX2_CLONE
__attribute__((target("avx2,fma")))
static void hq_gptq_update_avx2(float * x, int64_t n, int nr, const float * e, const float * const * u, int64_t nb, int64_t b1) {
    hq_gptq_update_body(x, n, nr, e, u, nb, b1);
}
#endif

static void hq_gptq_update(float * x, int64_t n, int nr, const float * e, const float * const * u, int64_t nb, int64_t b1) {
#ifdef HQ_AVX2_CLONE
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        hq_gptq_update_avx2(x, n, nr, e, u, nb, b1);
        return;
    }
#endif
    hq_gptq_update_default(x, n, nr, e, u, nb, b1);
}

static bool hq_is_elementwise(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_HQ2_XXS:
        case GGML_TYPE_HQ2_XS:
        case GGML_TYPE_HQ2_S:
        case GGML_TYPE_HQ3_XXS:
        case GGML_TYPE_HQ3_S:
            return false;
        default:
            return true;
    }
}

// U == NULL: the uniform quantizer, which doesn't write x
static size_t hq_quantize(enum ggml_type type, float * GGML_RESTRICT x, void * GGML_RESTRICT dst, int64_t nrows, int64_t n,
                          const float * GGML_RESTRICT U) {
    GGML_ASSERT(ggml_is_rotated(type));
    ggml_quantize_init(type);

    const int    unit      = (int) ggml_blck_size(type);
    const size_t row_size  = ggml_row_size(type, n);
    const size_t unit_size = ggml_row_size(type, unit);
    GGML_ASSERT(n % unit == 0);

    hq_dec_grid  dg;
    hq_unit_info info;
    hq_dec_grid_for(&dg, type);

    if (!U) {
        for (int64_t r = 0; r < nrows; ++r) {
            for (int64_t u0 = 0; u0 < n; u0 += unit) {
                hq_encode_unit(type, x + r*n + u0, (char *) dst + r*row_size + (u0/unit)*unit_size, &info, &dg, true);
            }
        }
        return nrows*row_size;
    }

    float e[HQ_GPTQ_ROWS*QK_K];
    const float * u[QK_K];
    for (int64_t r0 = 0; r0 < nrows; r0 += HQ_GPTQ_ROWS) {
        const int nr = (int) MIN(HQ_GPTQ_ROWS, nrows - r0);
        memset(e, 0, sizeof(e));
        for (int64_t b0 = 0; b0 < n; b0 += QK_K) {
            const int64_t b1 = MIN(n, b0 + QK_K);
            for (int r = 0; r < nr; ++r) {
                float * xr = x + (r0 + r)*n;
                char  * yr = (char *) dst + (r0 + r)*row_size;
                for (int64_t u0 = b0; u0 < b1; u0 += unit) {
                    void * yu = yr + (u0/unit)*unit_size;
                    hq_encode_unit(type, xr + u0, yu, &info, &dg, false);
                    if (hq_is_elementwise(type)) {
                        hq_gptq_elements(type, xr, yu, &info, u0, unit, b0, b1, U, n, e + r*QK_K);
                    } else {
                        hq_gptq_groups(type, xr, yu, &info, &dg, u0, b0, b1, U, n, e + r*QK_K);
                    }
                }
            }
            if (b1 < n) {
                for (int64_t j = b0; j < b1; ++j) {
                    u[j - b0] = hq_urow(U, n, j);
                }
                hq_gptq_update(x + r0*n, n, nr, e, u, b1 - b0, b1);
            }
        }
    }
    return nrows*row_size;
}

size_t quantize_hq(enum ggml_type type, const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row) {
    return hq_quantize(type, (float *) src, dst, nrows, n_per_row, NULL);
}

size_t ggml_quantize_rows_gptq(enum ggml_type type, float * rows, void * dst, int64_t nrows, int64_t n_per_row, const float * U) {
    GGML_ASSERT(U);
    return hq_quantize(type, rows, dst, nrows, n_per_row, U);
}
