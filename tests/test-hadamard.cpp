// Hadamard-rotated quantization (ConvRot) correctness tests.
//
// Covers the invariants the Q4R_*/Q5R_* types rely on:
//   - the Sylvester matrix is orthonormal, symmetric and self-inverse
//   - the FWHT butterfly used by the quantizer computes the same H_n as that matrix
//   - the rotated/base type pairing round-trips and both spellings share a byte layout
//   - a rotated quantize -> dequantize -> inverse-rotate round-trip stays within quant error
//   - quantizer/inference consistency: <H*w, H*x> == <w, x>, which is what makes a rotated
//     weight's GEMM produce the same output as the unrotated one
//   - the imatrix transform is the group mean, i.e. sum_j H_ij^2 * m_j

#include "llama-hadamard.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <random>
#include <vector>

static int g_failed = 0;

static void check(bool ok, const char * name, double got, double tol) {
    printf("  %-58s %10.3e  (tol %.0e)  %s\n", name, got, tol, ok ? "PASS" : "FAIL");
    if (!ok) {
        g_failed++;
    }
}

static std::vector<float> make_random(int64_t n, uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);

    std::vector<float> v(n);
    for (auto & x : v) {
        x = dist(rng);
    }

    return v;
}

// H_n is orthonormal, symmetric and self-inverse, and the butterfly agrees with the matrix
static void test_matrix(int n) {
    printf("H_%d:\n", n);

    std::vector<float> H((size_t) n*n);
    llama_gen_hadamard_matrix(H.data(), n);

    double e_orth = 0.0;
    double e_sym  = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double dot = 0.0;
            for (int k = 0; k < n; k++) {
                dot += (double) H[i*n + k] * H[j*n + k];
            }

            e_orth = std::max(e_orth, std::fabs(dot - (i == j ? 1.0 : 0.0)));
            e_sym  = std::max(e_sym,  (double) std::fabs(H[i*n + j] - H[j*n + i]));
        }
    }

    check(e_orth < 1e-5, "orthonormal: max|H*H^T - I|",   e_orth, 1e-5);
    check(e_sym == 0.0,  "symmetric:   max|H - H^T|",     e_sym,  0.0);

    const std::vector<float> v = make_random(n, 1234 + n);

    // the butterfly must compute the same transform as the materialized matrix - inference uses
    // one or the other depending on whether the backend has a fast FWHT for this size
    std::vector<float> w = v;
    llama_hadamard_inplace(w.data(), n);

    double e_bfly = 0.0;
    for (int i = 0; i < n; i++) {
        double dot = 0.0;
        for (int j = 0; j < n; j++) {
            dot += (double) H[i*n + j] * v[j];
        }
        e_bfly = std::max(e_bfly, std::fabs(dot - w[i]));
    }
    check(e_bfly < 1e-5, "butterfly == matrix: max|H*v - fwht(v)|", e_bfly, 1e-5);

    // H*H == I, so the same routine undoes itself
    llama_hadamard_inplace(w.data(), n);

    double e_inv = 0.0;
    for (int i = 0; i < n; i++) {
        e_inv = std::max(e_inv, (double) std::fabs(w[i] - v[i]));
    }
    check(e_inv < 1e-5, "self-inverse: max|H(H(v)) - v|", e_inv, 1e-5);

    // the imatrix transform: sum_j H_ij^2 * m_j is the group mean, for every row i
    std::vector<float> m = make_random(n, 99 + n);
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        m[i] = std::fabs(m[i]) + 1.0f; // importances are non-negative
        mean += m[i];
    }
    mean /= n;

    double e_imat = 0.0;
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int j = 0; j < n; j++) {
            s += (double) H[i*n + j] * H[i*n + j] * m[j];
        }
        e_imat = std::max(e_imat, std::fabs(s - mean));
    }
    check(e_imat < 1e-5, "imatrix: max|sum_j H_ij^2 m_j - mean(m)|", e_imat, 1e-5);
}

// ggml_get_rotated_type / ggml_get_base_type round-trip, and a rotated type is a byte-level alias
static void test_type_pairing() {
    printf("type pairing:\n");

    int n_pairs = 0;
    int n_bad   = 0;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const ggml_type base = (ggml_type) i;
        const ggml_type rot  = ggml_get_rotated_type(base);

        if (rot == base) {
            continue;
        }

        n_pairs++;
        n_bad += ggml_get_base_type(rot) != base;      // round-trips
        n_bad += !ggml_is_rotated(rot);                // the rotated one is rotated
        n_bad += ggml_is_rotated(base);                // the base one is not
        n_bad += ggml_blck_size(rot) != ggml_blck_size(base);
        n_bad += ggml_type_size(rot) != ggml_type_size(base);
        n_bad += ggml_is_quantized(rot) != ggml_is_quantized(base);
        n_bad += !ggml_fwht_supports_group(ggml_blck_size(rot)); // every group must be FWHT-able

        printf("  %-7s <-> %-7s  blck=%3d  size=%3zu\n",
            ggml_type_name(base), ggml_type_name(rot), (int) ggml_blck_size(base), ggml_type_size(base));
    }

    check(n_pairs == 2, "two rotated variants exist (_K only)", (double) n_pairs, 0.0);
    check(n_bad  == 0,  "pairing invariants hold",    (double) n_bad,  0.0);
}

// Rotate -> quantize as the base type -> dequantize -> inverse-rotate should land within the
// error the unrotated path has, and <H*w, H*x> must equal <w, x> exactly enough that a rotated
// GEMM reproduces the unrotated one.
static void test_round_trip(ggml_type base, int64_t n_per_row, int64_t nrows) {
    const ggml_type rot = ggml_get_rotated_type(base);
    const int64_t   g   = ggml_blck_size(base);

    printf("round-trip %s -> %s (%" PRId64 " x %" PRId64 "):\n",
        ggml_type_name(base), ggml_type_name(rot), n_per_row, nrows);

    const int64_t n = n_per_row * nrows;

    const std::vector<float> w = make_random(n, 4242);

    // --- baseline: quantize unrotated ---
    std::vector<uint8_t> q_plain(ggml_row_size(base, n_per_row) * nrows);
    ggml_quantize_chunk(base, w.data(), q_plain.data(), 0, nrows, n_per_row, nullptr);

    std::vector<float> w_plain(n);
    ggml_get_type_traits(base)->to_float(q_plain.data(), w_plain.data(), n);

    // --- rotated: rotate, quantize as the base type, dequantize, rotate back ---
    std::vector<float> w_rot = w;
    for (int64_t r = 0; r < nrows; r++) {
        for (int64_t c = 0; c < n_per_row; c += g) {
            llama_hadamard_inplace(w_rot.data() + r*n_per_row + c, (int) g);
        }
    }

    std::vector<uint8_t> q_rot(ggml_row_size(base, n_per_row) * nrows);
    ggml_quantize_chunk(base, w_rot.data(), q_rot.data(), 0, nrows, n_per_row, nullptr);

    std::vector<float> w_back(n);
    ggml_get_type_traits(base)->to_float(q_rot.data(), w_back.data(), n);
    for (int64_t r = 0; r < nrows; r++) {
        for (int64_t c = 0; c < n_per_row; c += g) {
            llama_hadamard_inplace(w_back.data() + r*n_per_row + c, (int) g);
        }
    }

    double err_plain = 0.0;
    double err_rot   = 0.0;
    for (int64_t i = 0; i < n; i++) {
        err_plain = std::max(err_plain, (double) std::fabs(w_plain[i] - w[i]));
        err_rot   = std::max(err_rot,   (double) std::fabs(w_back[i]  - w[i]));
    }

    printf("  max|dequant - original|: unrotated %.3e, rotated %.3e\n", err_plain, err_rot);
    check(err_rot < 4.0*err_plain, "rotated round-trip error is comparable", err_rot, 4.0*err_plain);

    // --- what inference actually relies on: <W'_row, H*x> == <W_row, x> ---
    // W' is the *quantized* rotated weight, so this also folds in the quantization error the
    // engine will really see.
    const std::vector<float> x = make_random(n_per_row, 777);

    std::vector<float> x_rot = x;
    for (int64_t c = 0; c < n_per_row; c += g) {
        llama_hadamard_inplace(x_rot.data() + c, (int) g);
    }

    // dequantize the rotated weight without rotating it back - this is literally what the GEMM
    // sees - and pair it with the rotated activation
    std::vector<float> w_quant_rot(n);
    ggml_get_type_traits(base)->to_float(q_rot.data(), w_quant_rot.data(), n);

    double err_dot_plain = 0.0;
    double err_dot_rot   = 0.0;
    for (int64_t r = 0; r < nrows; r++) {
        double ref = 0.0, dot_plain = 0.0, dot_rot = 0.0;
        for (int64_t c = 0; c < n_per_row; c++) {
            const int64_t i = r*n_per_row + c;
            ref       += (double) w[i]           * x[c];     // exact
            dot_plain += (double) w_plain[i]     * x[c];     // <dequant(W), x>
            dot_rot   += (double) w_quant_rot[i] * x_rot[c]; // <dequant(W'), H*x>
        }

        err_dot_plain = std::max(err_dot_plain, std::fabs(dot_plain - ref));
        err_dot_rot   = std::max(err_dot_rot,   std::fabs(dot_rot   - ref));
    }

    printf("  max|dot - exact|:        unrotated %.3e, rotated %.3e\n", err_dot_plain, err_dot_rot);
    check(err_dot_rot < 4.0*err_dot_plain + 1e-4, "rotated GEMM matches the unrotated result",
        err_dot_rot, 4.0*err_dot_plain + 1e-4);
}

int main() {
    ggml_quantize_init(GGML_TYPE_Q4_K);

    test_matrix(256);
    test_matrix(512);

    test_type_pairing();

    test_round_trip(GGML_TYPE_Q4_K, 512, 8);
    test_round_trip(GGML_TYPE_Q5_K, 512, 8);

    if (g_failed > 0) {
        printf("\n%d check(s) FAILED\n", g_failed);
        return 1;
    }

    printf("\nall checks passed\n");
    return 0;
}
