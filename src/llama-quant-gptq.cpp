#include "llama-quant-gptq.h"

#include <cmath>
#include <cstring>
#include <memory>

// x86 builds without AVX2 in their baseline (llama.o isn't built with -march) get an AVX2+FMA clone of
// the fp64 kernels, picked at run time
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && !defined(__AVX2__)
#define GPTQ_AVX2_CLONE
#endif

#if defined(__GNUC__) || defined(__clang__)
#define GPTQ_INLINE inline __attribute__((always_inline))
#else
#define GPTQ_INLINE inline
#endif

#define GPTQ_NB 128 // Cholesky panel width
#define GPTQ_NC 256 // columns per panel-solve task and per trailing-update block
#define GPTQ_MC 64  // rows per trailing-update task

size_t llama_gptq_packed_bytes(int64_t n) {
    return sizeof(float)*(size_t) n*(n + 1)/2;
}

size_t llama_gptq_build_bytes(int64_t n) {
    return sizeof(double)*(size_t) n*n + llama_gptq_packed_bytes(n);
}

bool llama_gptq_normalize(const float * v, int64_t n, std::vector<float> & vbar) {
    double sum = 0.0;
    for (int64_t k = 0; k < n; ++k) {
        if (!(v[k] >= 0.0f)) {
            return false;
        }
        sum += v[k];
    }
    if (!(sum > 0.0)) {
        return false;
    }
    const double mean = sum/n;
    vbar.resize(n);
    for (int64_t k = 0; k < n; ++k) {
        vbar[k] = (float) (v[k]/mean);
    }
    return true;
}

// ====================== blocked right-looking Cholesky, A = U^T U in the upper triangle of A (n x n, row-major)
//
// Every entry of U is computed by one fixed sequence of operations whatever the thread count, so U is
// bitwise reproducible. The lower triangle is scratch.

// the diagonal block [k0, k1)
static bool gptq_chol_diag(double * A, int64_t n, int64_t k0, int64_t k1) {
    for (int64_t p = k0; p < k1; ++p) {
        double * up = A + p*n;
        if (!(up[p] > 0.0)) {
            return false;
        }
        const double d = std::sqrt(up[p]);
        up[p] = d;
        for (int64_t j = p + 1; j < k1; ++j) {
            up[j] /= d;
        }
        for (int64_t q = p + 1; q < k1; ++q) {
            const double s = up[q];
            double * uq = A + q*n;
            for (int64_t j = q; j < k1; ++j) {
                uq[j] -= s*up[j];
            }
        }
    }
    return true;
}

// rows [k0, k1) of U for the columns [j0, j1), then those columns packed for the trailing update:
// pack[(g*nb + p)*8 + l] = U[k0 + p][k1 + 8*g + l], zero past n
static GPTQ_INLINE void gptq_panel_body(double * A, int64_t n, int64_t k0, int64_t k1, int64_t j0, int64_t j1, double * pack) {
    const int64_t nb = k1 - k0;
    for (int64_t p = k0; p < k1; ++p) {
        double * up = A + p*n;
        for (int64_t q = k0; q < p; ++q) {
            const double   s  = A[q*n + p];
            const double * uq = A + q*n;
            for (int64_t j = j0; j < j1; ++j) {
                up[j] -= s*uq[j];
            }
        }
        const double d = up[p];
        for (int64_t j = j0; j < j1; ++j) {
            up[j] /= d;
        }
    }
    for (int64_t j = j0; j < j1; j += 8) {
        double * pg = pack + ((j - k1)/8)*nb*8;
        for (int64_t p = 0; p < nb; ++p) {
            const double * up = A + (k0 + p)*n;
            for (int64_t l = 0; l < 8; ++l) {
                pg[p*8 + l] = j + l < j1 ? up[j + l] : 0.0;
            }
        }
    }
}

// A[i + r][j + l] -= sum_p a[p*8 + r]*b[p*8 + l] for r < nr, l < nc
static GPTQ_INLINE void gptq_kernel_4x8(const double * a, const double * b, int64_t nb, double * c, int64_t ldc, int nr, int nc) {
    double out[4][8];
#if defined(__GNUC__) || defined(__clang__)
    typedef double gptq_v4 __attribute__((vector_size(32)));
    gptq_v4 c00 = { 0 }, c01 = { 0 }, c10 = { 0 }, c11 = { 0 }, c20 = { 0 }, c21 = { 0 }, c30 = { 0 }, c31 = { 0 };
    for (int64_t p = 0; p < nb; ++p) {
        gptq_v4 b0, b1;
        memcpy(&b0, b + p*8,     sizeof(b0));
        memcpy(&b1, b + p*8 + 4, sizeof(b1));
        const double * ap = a + p*8;
        c00 += ap[0]*b0; c01 += ap[0]*b1;
        c10 += ap[1]*b0; c11 += ap[1]*b1;
        c20 += ap[2]*b0; c21 += ap[2]*b1;
        c30 += ap[3]*b0; c31 += ap[3]*b1;
    }
    memcpy(out[0], &c00, 32); memcpy(out[0] + 4, &c01, 32);
    memcpy(out[1], &c10, 32); memcpy(out[1] + 4, &c11, 32);
    memcpy(out[2], &c20, 32); memcpy(out[2] + 4, &c21, 32);
    memcpy(out[3], &c30, 32); memcpy(out[3] + 4, &c31, 32);
#else
    for (int r = 0; r < 4; ++r) {
        for (int l = 0; l < 8; ++l) {
            out[r][l] = 0.0;
        }
    }
    for (int64_t p = 0; p < nb; ++p) {
        for (int r = 0; r < 4; ++r) {
            for (int l = 0; l < 8; ++l) {
                out[r][l] += a[p*8 + r]*b[p*8 + l];
            }
        }
    }
#endif
    for (int r = 0; r < nr; ++r) {
        for (int l = 0; l < nc; ++l) {
            c[r*ldc + l] -= out[r][l];
        }
    }
}

// the trailing update of the rows [i0, i1): A[i][j] -= sum_p U[k0 + p][i]*U[k0 + p][j] for the upper triangle
static GPTQ_INLINE void gptq_update_body(double * A, int64_t n, int64_t k1, int64_t nb, const double * pack, int64_t i0, int64_t i1) {
    for (int64_t jb = k1 + ((i0 - k1)/8)*8; jb < n; jb += GPTQ_NC) {
        const int64_t je = std::min<int64_t>(n, jb + GPTQ_NC);
        for (int64_t i = i0; i < i1; i += 4) {
            const int64_t  ri = i - k1;
            const double * a  = pack + (ri/8)*nb*8 + ri%8;
            const int      nr = (int) std::min<int64_t>(4, i1 - i);
            for (int64_t j = std::max(jb, k1 + (ri/8)*8); j < je; j += 8) {
                gptq_kernel_4x8(a, pack + ((j - k1)/8)*nb*8, nb, A + i*n + j, n, nr, (int) std::min<int64_t>(8, n - j));
            }
        }
    }
}

static void gptq_panel_default(double * A, int64_t n, int64_t k0, int64_t k1, int64_t j0, int64_t j1, double * pack) {
    gptq_panel_body(A, n, k0, k1, j0, j1, pack);
}

static void gptq_update_default(double * A, int64_t n, int64_t k1, int64_t nb, const double * pack, int64_t i0, int64_t i1) {
    gptq_update_body(A, n, k1, nb, pack, i0, i1);
}

#ifdef GPTQ_AVX2_CLONE
__attribute__((target("avx2,fma")))
static void gptq_panel_avx2(double * A, int64_t n, int64_t k0, int64_t k1, int64_t j0, int64_t j1, double * pack) {
    gptq_panel_body(A, n, k0, k1, j0, j1, pack);
}

__attribute__((target("avx2,fma")))
static void gptq_update_avx2(double * A, int64_t n, int64_t k1, int64_t nb, const double * pack, int64_t i0, int64_t i1) {
    gptq_update_body(A, n, k1, nb, pack, i0, i1);
}
#endif

static bool gptq_cholesky(double * A, int64_t n, std::vector<std::thread> & workers, int nthread) {
    auto panel  = gptq_panel_default;
    auto update = gptq_update_default;
#ifdef GPTQ_AVX2_CLONE
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        panel  = gptq_panel_avx2;
        update = gptq_update_avx2;
    }
#endif

    std::vector<double> pack((size_t) GPTQ_NB*(n + 8));

    for (int64_t k0 = 0; k0 < n; k0 += GPTQ_NB) {
        const int64_t k1 = std::min<int64_t>(n, k0 + GPTQ_NB);
        if (!gptq_chol_diag(A, n, k0, k1)) {
            return false;
        }
        if (k1 == n) {
            break;
        }
        const int64_t nb = k1 - k0;
        llama_parallel_rows((n - k1 + GPTQ_NC - 1)/GPTQ_NC, workers, nthread, [&](int64_t t) {
            const int64_t j0 = k1 + t*GPTQ_NC;
            panel(A, n, k0, k1, j0, std::min<int64_t>(n, j0 + GPTQ_NC), pack.data());
        });
        llama_parallel_rows((n - k1 + GPTQ_MC - 1)/GPTQ_MC, workers, nthread, [&](int64_t t) {
            const int64_t i0 = k1 + t*GPTQ_MC;
            update(A, n, k1, nb, pack.data(), i0, std::min<int64_t>(n, i0 + GPTQ_MC));
        });
    }
    return true;
}

bool llama_gptq_factor(const float * v, int64_t n, uint64_t seed, float damp, std::vector<float> & U,
                       std::vector<std::thread> & workers, int nthread) {
    std::vector<float> vbar;
    if (!(damp >= LLAMA_GPTQ_DAMP_MIN) || !ggml_rht_plan(n, nullptr, nullptr) || !llama_gptq_normalize(v, n, vbar)) {
        return false;
    }

    // H^-1 = R*diag(g)*R^T, g = 1/(vbar + damp); column i is R(g * R^T e_i), stored as row i
    std::vector<double> g(n);
    for (int64_t k = 0; k < n; ++k) {
        g[k] = 1.0/((double) vbar[k] + (double) damp);
    }
    std::unique_ptr<double[]> A(new double[(size_t) n*n]);
    const int64_t cols = 64;
    llama_parallel_rows((n + cols - 1)/cols, workers, nthread, [&](int64_t t) {
        std::vector<double> e(n);
        for (int64_t i = t*cols; i < std::min(n, (t + 1)*cols); ++i) {
            std::fill(e.begin(), e.end(), 0.0);
            e[i] = 1.0;
            ggml_rht_inv_f64(e.data(), n, seed);
            for (int64_t k = 0; k < n; ++k) {
                e[k] *= g[k];
            }
            ggml_rht_ref_f64(e.data(), n, seed);
            memcpy(A.get() + (size_t) i*n, e.data(), n*sizeof(double));
        }
    });

    if (!gptq_cholesky(A.get(), n, workers, nthread)) {
        return false;
    }

    U.resize((size_t) n*(n + 1)/2);
    llama_parallel_rows(n, workers, nthread, [&](int64_t j) {
        float        * uj = U.data() + ((size_t) j*n - (size_t) j*(j - 1)/2);
        const double * aj = A.get() + (size_t) j*n;
        for (int64_t k = j; k < n; ++k) {
            uj[k - j] = (float) aj[k];
        }
    });
    return true;
}

size_t llama_gptq_cache::bytes() const {
    size_t total = 0;
    for (const auto & e : entries) {
        total += llama_gptq_packed_bytes(e.n);
    }
    return total;
}

const float * llama_gptq_cache::get(const float * v, int64_t n, uint64_t seed, float damp,
                                    std::vector<std::thread> & workers, int nthread, status * st) {
    status dummy;
    st = st ? st : &dummy;

    std::vector<float> vbar;
    if (!(damp >= LLAMA_GPTQ_DAMP_MIN) || !ggml_rht_plan(n, nullptr, nullptr) || !llama_gptq_normalize(v, n, vbar)) {
        *st = REFUSED;
        return nullptr;
    }
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it->n == n && it->seed == seed && it->damp == damp && it->vbar == vbar) {
            entries.splice(entries.begin(), entries, it);
            *st = HIT;
            return entries.front().U.data();
        }
    }

    const size_t need = llama_gptq_build_bytes(n);
    if (need > cap) {
        *st = TOO_BIG;
        return nullptr;
    }
    while (!entries.empty() && (bytes() + need > cap || entries.size() >= max_entries)) {
        entries.pop_back();
    }

    entry e { std::move(vbar), n, seed, damp, {} };
    if (!llama_gptq_factor(v, n, seed, damp, e.U, workers, nthread)) {
        *st = FAILED;
        return nullptr;
    }
    entries.push_front(std::move(e));
    *st = BUILT;
    return entries.front().U.data();
}

bool llama_tensor_quantize_gptq(ggml_type type, float * rows, void * dst, int64_t nrows, int64_t n_per_row,
                                const float * U, std::vector<std::thread> & workers, int nthread) {
    const size_t row_size = ggml_row_size(type, n_per_row);
    std::atomic<bool> valid { true };
    llama_parallel_rows((nrows + LLAMA_GPTQ_ROW_GROUP - 1)/LLAMA_GPTQ_ROW_GROUP, workers, nthread, [&](int64_t t) {
        const int64_t r0 = t*LLAMA_GPTQ_ROW_GROUP;
        char * out = (char *) dst + r0*row_size;
        const size_t size = ggml_quantize_rows_gptq(type, rows + r0*n_per_row, out,
                                                    std::min<int64_t>(LLAMA_GPTQ_ROW_GROUP, nrows - r0), n_per_row, U);
        if (!ggml_validate_row_data(type, out, size)) {
            valid = false;
        }
    });
    return valid;
}
