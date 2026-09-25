#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-impl.h"

#include <assert.h>
#include <float.h>
#include <math.h>
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

static float make_qkx3_quants(int n, int nmax, const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L,
        float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux, float rmin, float rdelta, int nstep) {
    float min = x[0];
    float max = x[0];
    float sum_w = 1;
    float sum_x = x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        sum_w += 1;
        sum_x += x[i];
    }
    if (min > 0) {
        min = 0;
    }
    if (max <= min) {
        memset(L, 0, n);
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        best_mse += diff*diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            sum_l  += l;
            sum_l2 += l*l;
            sum_xl += l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mse = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                mse += diff*diff;
            }
            if (mse < best_mse) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mse = mse;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float make_qp_quants(int n, int nmax, const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L) {
    float max = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    if (max < GROUP_MAX_EPS) {
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) {
        L[i] = nearest_int(iscale * x[i]);
    }
    float scale = 1/iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - scale*L[i];
        best_mse += diff*diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float iscale_is = (0.1f*is + nmax)/max;
        float scale_is = 1/iscale_is;
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale_is*x[i]);
            l = MIN(nmax, l);
            float diff = x[i] - scale_is*l;
            mse += diff*diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MIN(nmax, l);
        L[i] = l;
        sumlx += x[i]*l;
        suml2 += l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float slx = sumlx - x[i]*L[i];
            float sl2 = suml2 - L[i]*L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = MIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += x[i]*new_l;
                    sl2 += new_l*new_l;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = new_l; sumlx = slx; suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) {
            break;
        }
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// ====================== HQ4_K / HQ5_K

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
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }
        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

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
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(31, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * GGML_RESTRICT qh = y[i].qh;
        uint8_t * GGML_RESTRICT ql = y[i].qs;
        memset(qh, 0, QK_K/8);

        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16; qh[j] |= m1;
                }
                int l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16; qh[j] |= m2;
                }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }

        x += QK_K;
    }
}

// ====================== HQ2 (IQ2 grids)

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

static void quantize_row_hq2_xxs_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n) {

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
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = xval[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        if (xval[8*k+i] < min) {
                            min = xval[8*k+i]; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float scale = make_qp_quants(32, kMaxQ+1, xval, (uint8_t*)L);
            float eff_max = scale*kMaxQ;
            if (eff_max <= 0) {
                scales[ib] = 0;
                memset(L, 0, 32);
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

static void quantize_row_hq2_xs_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n) {

    const uint64_t * kgrid_q2xs;
    const int      * kmap_q2xs;
    const uint16_t * kneighbors_q2xs;
    ggml_iq2_entry(GGML_TYPE_IQ2_XS, &kgrid_q2xs, &kmap_q2xs, &kneighbors_q2xs);

    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n/QK_K;

    block_iq2_xs * y = vy;

    float scales[QK_K/16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
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
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = xval[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        if (xval[8*k+i] < min) {
                            min = xval[8*k+i]; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 16);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                continue;
            }
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
            if (scale < 0) {
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

static void quantize_row_hq2_s_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n) {

    const uint64_t * kgrid_q2xs;
    const int      * kmap_q2xs;
    const uint16_t * kneighbors_q2xs;
    ggml_iq2_entry(GGML_TYPE_IQ2_S, &kgrid_q2xs, &kmap_q2xs, &kneighbors_q2xs);

    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n/QK_K;

    block_iq2_s * y = vy;

    float scales[QK_K/16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_s));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            for (int k = 0; k < 2; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 16);
            if (max < GROUP_MAX_EPS_IQ2_S) {
                scales[ib] = 0;
                continue;
            }
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
            if (scale < 0) {
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

static void quantize_row_hq3_xxs_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n) {

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
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = xval[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        if (xval[8*k+i] < min) {
                            min = xval[8*k+i]; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 32);
            if (max < GROUP_MAX_EPS_IQ3_XXS) {
                scales[ib] = 0;
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

static void quantize_row_hq3_s_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n) {

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
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            float max = xval[0];
            for (int i = 1; i < block_size; ++i) max = MAX(max, xval[i]);
            memset(L, 0, block_size);
            if (!max) {
                scales[ib] = 0;
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

static void hq_dec_grid_for(hq_dec_grid * t, enum ggml_type type) {
    const uint64_t * grid2;
    const uint32_t * grid3;
    const int      * map;
    const uint16_t * neighbours;
    switch (type) {
        case GGML_TYPE_HQ2_XXS:
            ggml_iq2_entry(GGML_TYPE_IQ2_XXS, &grid2, &map, &neighbours);
            hq_dec_grid_init(t, (const uint8_t *) iq2xxs_grid, 256, 8, map);
            break;
        case GGML_TYPE_HQ2_XS:
            ggml_iq2_entry(GGML_TYPE_IQ2_XS, &grid2, &map, &neighbours);
            hq_dec_grid_init(t, (const uint8_t *) iq2xs_grid, 512, 8, map);
            break;
        case GGML_TYPE_HQ2_S:
            ggml_iq2_entry(GGML_TYPE_IQ2_S, &grid2, &map, &neighbours);
            hq_dec_grid_init(t, (const uint8_t *) iq2s_grid, 1024, 8, map);
            break;
        case GGML_TYPE_HQ3_XXS:
            ggml_iq3_entry(256, &grid3, &map, &neighbours);
            hq_dec_grid_init(t, (const uint8_t *) iq3xxs_grid, 256, 4, map);
            break;
        case GGML_TYPE_HQ3_S:
            ggml_iq3_entry(512, &grid3, &map, &neighbours);
            hq_dec_grid_init(t, (const uint8_t *) iq3s_grid, 512, 4, map);
            break;
        default:
            break;
    }
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

static void quantize_row_hq4_nl_impl(const int super_block_size, const int block_size, const float * GGML_RESTRICT x,
        ggml_fp16_t * dh, uint8_t * q4, uint16_t * scales_h, uint8_t * scales_l,
        float * scales, uint8_t * L) {

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

static void quantize_row_hq4_nl(const float * GGML_RESTRICT x, block_iq4_nl * GGML_RESTRICT y, int64_t n) {
    GGML_ASSERT(n%QK4_NL == 0);
    uint8_t L[QK4_NL];
    uint16_t unused_h;
    float scale;
    for (int64_t ib = 0; ib < n/QK4_NL; ++ib) {
        quantize_row_hq4_nl_impl(QK4_NL, 32, x + QK4_NL*ib, &y[ib].d, y[ib].qs, &unused_h, NULL, &scale, L);
    }
}

static void quantize_row_hq4_xs(const float * GGML_RESTRICT x, block_iq4_xs * GGML_RESTRICT y, int64_t n) {
    GGML_ASSERT(n%QK_K == 0);
    uint8_t L[QK_K];
    float scales[QK_K/32];
    for (int64_t ibl = 0; ibl < n/QK_K; ++ibl) {
        quantize_row_hq4_nl_impl(QK_K, 32, x + QK_K*ibl, &y[ibl].d, y[ibl].qs, &y[ibl].scales_h, y[ibl].scales_l, scales, L);
    }
}

size_t quantize_hq(enum ggml_type type, const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row) {
    GGML_ASSERT(ggml_is_rotated(type));

    const size_t row_size = ggml_row_size(type, n_per_row);

    hq_dec_grid dg;
    hq_dec_grid_for(&dg, type);

    char * qrow = (char *) dst;
    for (int64_t row = 0; row < nrows; ++row) {
        switch (type) {
            case GGML_TYPE_HQ4_K: quantize_row_hq4_K_impl(src, (block_q4_K *) qrow, n_per_row); break;
            case GGML_TYPE_HQ5_K: quantize_row_hq5_K_impl(src, (block_q5_K *) qrow, n_per_row); break;
            case GGML_TYPE_HQ2_XXS:
                quantize_row_hq2_xxs_impl(src, qrow, n_per_row);
                hq2_xxs_repick(src, (block_iq2_xxs *) qrow, n_per_row, &dg);
                break;
            case GGML_TYPE_HQ2_XS:
                quantize_row_hq2_xs_impl(src, qrow, n_per_row);
                hq2_xs_repick(src, (block_iq2_xs *) qrow, n_per_row, &dg);
                break;
            case GGML_TYPE_HQ2_S:
                quantize_row_hq2_s_impl(src, qrow, n_per_row);
                hq2_s_repick(src, (block_iq2_s *) qrow, n_per_row, &dg);
                break;
            case GGML_TYPE_HQ3_XXS:
                quantize_row_hq3_xxs_impl(src, qrow, n_per_row);
                hq3_xxs_repick(src, (block_iq3_xxs *) qrow, n_per_row, &dg);
                break;
            case GGML_TYPE_HQ3_S:
                quantize_row_hq3_s_impl(src, qrow, n_per_row);
                hq3_s_repick(src, (block_iq3_s *) qrow, n_per_row, &dg);
                break;
            case GGML_TYPE_HQ4_NL: quantize_row_hq4_nl(src, (block_iq4_nl *) qrow, n_per_row); break;
            case GGML_TYPE_HQ4_XS: quantize_row_hq4_xs(src, (block_iq4_xs *) qrow, n_per_row); break;
            default: GGML_ABORT("fatal error");
        }
        src  += n_per_row;
        qrow += row_size;
    }

    return nrows * row_size;
}
