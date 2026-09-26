// Tests for GPTQ error feedback on the Hadamard-rotated (HQ) types: the factor of H^-1, its cache, and
// the encoder.
//
// usage: test-hq-gptq [--quick]    (--quick skips the factor accuracy sweep up to n = 17408)

#include "ggml.h"
#include "ggml-quants.h"
#include "llama-quant-gptq.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int g_failed = 0;

static void check(bool ok, const std::string & name) {
    printf("  %-88s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    fflush(stdout);
    if (!ok) {
        g_failed++;
    }
}

static uint64_t sm64(uint64_t & s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static float unif(uint64_t & s) { return (float) (sm64(s) >> 40) * (1.0f/16777216.0f); }

// roughly normal, from four uniforms: no libm, so bitwise reproducible everywhere
static float gauss(uint64_t & s) {
    const float a = unif(s), b = unif(s), c = unif(s), d = unif(s);
    return (a + b + c + d - 2.0f) * 1.7320508f;
}

enum v_kind { V_CONST, V_SPIKE, V_LOGNORMAL, V_ZEROS, V_ONE_CHANNEL };

static const char * v_name(v_kind k) {
    switch (k) {
        case V_CONST:       return "constant";
        case V_SPIKE:       return "one spike x1e4";
        case V_LOGNORMAL:   return "log-normal";
        case V_ZEROS:       return "log-normal with zeros";
        case V_ONE_CHANNEL: return "one channel";
    }
    return "";
}

static std::vector<float> make_v(v_kind kind, int64_t n, uint64_t seed) {
    std::vector<float> v(n, 1.0f);
    uint64_t s = seed;
    switch (kind) {
        case V_CONST: break;
        case V_SPIKE: v[n/3] = 1e4f; break;
        case V_LOGNORMAL: for (auto & x : v) x = expf(2.0f*gauss(s)); break;
        case V_ZEROS: for (auto & x : v) x = unif(s) < 0.25f ? 0.0f : expf(2.0f*gauss(s)); break;
        case V_ONE_CHANNEL: std::fill(v.begin(), v.end(), 0.0f); v[n/5] = 3.0f; break;
    }
    return v;
}

static double kappa(const std::vector<float> & v, float damp) {
    std::vector<float> vbar;
    llama_gptq_normalize(v.data(), (int64_t) v.size(), vbar);
    float mn = vbar[0], mx = vbar[0];
    for (float x : vbar) { mn = std::min(mn, x); mx = std::max(mx, x); }
    return ((double) mx + damp)/((double) mn + damp);
}

static size_t upos(int64_t n, int64_t j, int64_t k) {
    return (size_t) j*n - (size_t) j*(j - 1)/2 + (k - j);
}

// ||U*H*U^T*y - y||/||y|| for a random y, H = R*(diag(vbar) + damp)*R^T applied with the fp64 transforms
static double uhu_error(const std::vector<float> & U, const std::vector<float> & v, float damp, uint64_t seed, uint64_t yseed) {
    const int64_t n = (int64_t) v.size();
    std::vector<float> vbar;
    llama_gptq_normalize(v.data(), n, vbar);
    std::vector<double> y(n), z(n, 0.0), u(n, 0.0);
    uint64_t s = yseed;
    for (auto & x : y) x = gauss(s);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t k = j; k < n; ++k) {
            z[k] += (double) U[upos(n, j, k)]*y[j];
        }
    }
    ggml_rht_inv_f64(z.data(), n, seed);
    for (int64_t k = 0; k < n; ++k) {
        z[k] *= (double) vbar[k] + damp;
    }
    ggml_rht_ref_f64(z.data(), n, seed);
    double err = 0, ny = 0;
    for (int64_t j = 0; j < n; ++j) {
        double acc = 0;
        for (int64_t k = j; k < n; ++k) {
            acc += (double) U[upos(n, j, k)]*z[k];
        }
        err += (acc - y[j])*(acc - y[j]);
        ny  += y[j]*y[j];
    }
    return sqrt(err/ny);
}

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static const int g_nthread = std::max(1u, std::thread::hardware_concurrency());

static void test_factor() {
    printf("factor:\n");
    std::vector<std::thread> workers;
    const uint64_t seed = 0x48512d524854ull;
    const float damp = LLAMA_GPTQ_DAMP_DEFAULT;

    for (int64_t n : { 256, 1024, 2048, 3072, 5120, 6144, 9728, 17408 }) {
        for (v_kind kind : { V_CONST, V_SPIKE, V_LOGNORMAL, V_ZEROS }) {
            const std::vector<float> v = make_v(kind, n, 7 + n);
            std::vector<float> U;
            const auto t0 = std::chrono::steady_clock::now();
            const bool ok = llama_gptq_factor(v.data(), n, seed, damp, U, workers, g_nthread);
            const double t = seconds_since(t0);
            const double err = ok ? uhu_error(U, v, damp, seed, 3) : 1e30;
            const double bound = 50*ldexp(1.0, -24)*sqrt(kappa(v, damp));
            char buf[256];
            snprintf(buf, sizeof(buf), "n = %5" PRId64 ", v %-22s: |U H U^T y - y|/|y| = %.2e < %.2e (%.1f s)", n, v_name(kind), err, bound, t);
            check(ok && err < bound, buf);
        }
    }

    {
        const int64_t n = 17408;
        const std::vector<float> v = make_v(V_ONE_CHANNEL, n, 1);
        std::vector<float> U;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = llama_gptq_factor(v.data(), n, seed, LLAMA_GPTQ_DAMP_MIN, U, workers, g_nthread);
        const double t = seconds_since(t0);
        const double err = ok ? uhu_error(U, v, LLAMA_GPTQ_DAMP_MIN, seed, 5) : 1e30;
        const double k = kappa(v, LLAMA_GPTQ_DAMP_MIN);
        const double bound = 50*ldexp(1.0, -24)*sqrt(k);
        char buf[256];
        snprintf(buf, sizeof(buf), "worst case (n = 17408, one channel, damp 0.001, kappa %.2e): completes, err %.2e < %.2e", k, err, bound);
        check(ok && err < bound, buf);
        printf("  timing: n = 17408 factor in %.1f s on %d threads\n", t, g_nthread);
        check(t < 900, "n = 17408 factors in under 900 s");
    }

    {
        const int64_t n = 1024;
        const std::vector<float> v = make_v(V_CONST, n, 1);
        std::vector<float> U;
        llama_gptq_factor(v.data(), n, seed, damp, U, workers, g_nthread);
        const double want = 1.0/sqrt(1.0 + damp);
        double err = 0;
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t k = j; k < n; ++k) {
                err = std::max(err, fabs(U[upos(n, j, k)] - (j == k ? want : 0.0)));
            }
        }
        check(err < 1e-6, "constant v: U = (1 + damp)^(-1/2) I within 1e-6");
    }

    {
        const int64_t n = 1024;
        std::vector<float> U(3, 7.0f);
        const std::vector<float> v = make_v(V_SPIKE, n, 1), zero(n, 0.0f);
        std::vector<float> neg = v;
        neg[5] = -1.0f;
        check(!llama_gptq_factor(v.data(), n, seed, 0.0009f, U, workers, g_nthread), "damp 0.0009 is refused");
        check(!llama_gptq_factor(zero.data(), n, seed, damp, U, workers, g_nthread), "an all-zero v is refused");
        check(!llama_gptq_factor(neg.data(), n, seed, damp, U, workers, g_nthread), "a v with a negative entry is refused");
        check(!llama_gptq_factor(v.data(), 1000, seed, damp, U, workers, g_nthread), "a width without an RHT is refused");
        check(U.size() == 3 && U[0] == 7.0f, "... and U is left untouched");
    }

    {
        const int64_t n = 2048;
        const std::vector<float> v = make_v(V_ZEROS, n, 9);
        std::vector<float> U1, U8;
        llama_gptq_factor(v.data(), n, seed, damp, U1, workers, 1);
        llama_gptq_factor(v.data(), n, seed, damp, U8, workers, 8);
        check(U1.size() == U8.size() && memcmp(U1.data(), U8.data(), U1.size()*sizeof(float)) == 0,
              "1 and 8 threads give bitwise-identical U");
    }
}

static void test_cache() {
    printf("factor cache:\n");
    std::vector<std::thread> workers;
    const int64_t n = 1024;
    const float damp = LLAMA_GPTQ_DAMP_DEFAULT;
    std::vector<std::vector<float>> vs;
    for (int i = 0; i < 5; ++i) {
        vs.push_back(make_v(V_LOGNORMAL, n, 100 + i));
    }
    std::vector<float> v2 = vs[0];
    for (auto & x : v2) x *= 2.0f;

    llama_gptq_cache c(1ull << 30);
    llama_gptq_cache::status st;
    const float * U0 = c.get(vs[0].data(), n, 1, damp, workers, g_nthread, &st);
    check(U0 && st == llama_gptq_cache::BUILT, "the first input is built");
    std::vector<float> U;
    llama_gptq_factor(vs[0].data(), n, 1, damp, U, workers, g_nthread);
    check(memcmp(U0, U.data(), U.size()*sizeof(float)) == 0, "... and equals llama_gptq_factor");
    check(c.get(v2.data(), n, 1, damp, workers, g_nthread, &st) == U0 && st == llama_gptq_cache::HIT, "2*v hits (same vbar)");
    c.get(vs[0].data(), n, 2, damp, workers, g_nthread, &st);
    check(st == llama_gptq_cache::BUILT, "another seed misses");
    c.get(vs[0].data(), n, 1, 0.1f, workers, g_nthread, &st);
    check(st == llama_gptq_cache::BUILT, "another damp misses");
    c.get(vs[1].data(), n, 1, damp, workers, g_nthread, &st);
    check(st == llama_gptq_cache::BUILT && c.size() == 4, "a new v misses; 4 entries");
    c.get(vs[0].data(), n, 1, damp, workers, g_nthread, &st);
    check(st == llama_gptq_cache::HIT, "the first input still hits");
    c.get(vs[2].data(), n, 1, damp, workers, g_nthread, &st);
    check(c.size() == 4, "the 5th distinct input evicts one");
    c.get(vs[0].data(), n, 1, damp, workers, g_nthread, &st);
    check(st == llama_gptq_cache::HIT, "... the least recently used, not the first input");
    c.get(vs[0].data(), n, 2, damp, workers, g_nthread, &st);
    check(st == llama_gptq_cache::BUILT, "... which was the seed-2 entry");

    const std::vector<float> zero(n, 0.0f);
    check(!c.get(zero.data(), n, 1, damp, workers, g_nthread, &st) && st == llama_gptq_cache::REFUSED, "an all-zero v is refused");

    llama_gptq_cache small(llama_gptq_build_bytes(n) - 1);
    check(!small.get(vs[0].data(), n, 1, damp, workers, g_nthread, &st) && st == llama_gptq_cache::TOO_BIG && small.size() == 0,
          "memory cap: a factor that can't be built within the cap is refused");
    llama_gptq_cache one(llama_gptq_build_bytes(n) + llama_gptq_packed_bytes(n) - 1);
    one.get(vs[0].data(), n, 1, damp, workers, g_nthread, &st);
    one.get(vs[1].data(), n, 1, damp, workers, g_nthread, &st);
    check(st == llama_gptq_cache::BUILT && one.size() == 1 && one.bytes() == llama_gptq_packed_bytes(n),
          "memory cap: building a second factor evicts the first");
    one.get(vs[0].data(), n, 1, damp, workers, g_nthread, &st);
    check(st == llama_gptq_cache::BUILT, "... so the first is built again");
}

// ====================== the encoder

static const ggml_type all_types[] = { GGML_TYPE_HQ4_K, GGML_TYPE_HQ5_K, GGML_TYPE_HQ4_XS, GGML_TYPE_HQ4_NL,
    GGML_TYPE_HQ3_S, GGML_TYPE_HQ3_XXS, GGML_TYPE_HQ2_S, GGML_TYPE_HQ2_XS, GGML_TYPE_HQ2_XXS };

static const ggml_type elementwise_types[] = { GGML_TYPE_HQ4_K, GGML_TYPE_HQ5_K, GGML_TYPE_HQ4_XS, GGML_TYPE_HQ4_NL };

static const ggml_type codebook_types[] = { GGML_TYPE_HQ3_S, GGML_TYPE_HQ3_XXS, GGML_TYPE_HQ2_S, GGML_TYPE_HQ2_XS, GGML_TYPE_HQ2_XXS };

static bool parity_signs(ggml_type type) {
    return type == GGML_TYPE_HQ2_XXS || type == GGML_TYPE_HQ2_XS || type == GGML_TYPE_HQ3_XXS;
}

static std::vector<int64_t> widths(ggml_type type) {
    if (type == GGML_TYPE_HQ4_NL) {
        return { 96, 800, 2048 };
    }
    return { 256, 768, 2048 };
}

// nrows rows of four kinds: plain, outliers, a zeroed 32-block and super-block, tiny values
static std::vector<float> hash_rows(int64_t nrows, int64_t n, uint64_t seed) {
    std::vector<float> x(nrows*n);
    uint64_t s = seed;
    for (int64_t r = 0; r < nrows; ++r) {
        float * xr = x.data() + r*n;
        const float scale = 0.02f*(float) (1 + r%3);
        for (int64_t j = 0; j < n; ++j) xr[j] = scale*gauss(s);
        switch (r % 4) {
            case 1: for (int64_t j = 5; j < n; j += 37) xr[j] *= 8.0f; break;
            case 2:
                for (int64_t j = 32; j < 64 && j < n; ++j) xr[j] = 0.0f;
                for (int64_t j = 256; j < 512 && j < n; ++j) xr[j] = 0.0f;
                break;
            case 3: for (int64_t j = 0; j < n; j += 2) xr[j] *= 1e-6f; break;
        }
    }
    return x;
}

static uint64_t fnv1a(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

// quantize_hq's output before the shared driver, 8 hash_rows rows for seeds 1..3, on x86-64 with FMA
// (other builds may round differently)
static const struct { ggml_type type; int64_t n; uint64_t hash; } hq_hashes[] = {
    { GGML_TYPE_HQ4_K,    256, 0xd1892cd3bf0805f4ull },
    { GGML_TYPE_HQ4_K,    768, 0xd29cca2c9a9313b6ull },
    { GGML_TYPE_HQ4_K,   2048, 0xc2f6b238d324f2a1ull },
    { GGML_TYPE_HQ5_K,    256, 0x19dbb69b52776925ull },
    { GGML_TYPE_HQ5_K,    768, 0xe0f76f2d51dfbdc4ull },
    { GGML_TYPE_HQ5_K,   2048, 0xf20635a32e5f4331ull },
    { GGML_TYPE_HQ4_XS,   256, 0xa8335de4c1694ac7ull },
    { GGML_TYPE_HQ4_XS,   768, 0x9d175da58dc5c552ull },
    { GGML_TYPE_HQ4_XS,  2048, 0xeeafc35a04cb2d8eull },
    { GGML_TYPE_HQ4_NL,    96, 0x2a716849e54e88b4ull },
    { GGML_TYPE_HQ4_NL,   800, 0x05fe2acb88d10869ull },
    { GGML_TYPE_HQ4_NL,  2048, 0x89ac1e7ff66b03faull },
    { GGML_TYPE_HQ3_S,    256, 0x7d7544167943999dull },
    { GGML_TYPE_HQ3_S,    768, 0x4a0f8d733ff8ba83ull },
    { GGML_TYPE_HQ3_S,   2048, 0xa76b6497da386e3cull },
    { GGML_TYPE_HQ3_XXS,   256, 0x6e1b4346f8d96baeull },
    { GGML_TYPE_HQ3_XXS,   768, 0x3ca98046f6c49cc2ull },
    { GGML_TYPE_HQ3_XXS,  2048, 0xc1857a4378b462d0ull },
    { GGML_TYPE_HQ2_S,    256, 0x26cab4a584650709ull },
    { GGML_TYPE_HQ2_S,    768, 0x42d30a98d57c10a2ull },
    { GGML_TYPE_HQ2_S,   2048, 0xef2d9e6c8fee1255ull },
    { GGML_TYPE_HQ2_XS,   256, 0x4b482002b5bcc05aull },
    { GGML_TYPE_HQ2_XS,   768, 0x3b5b1deaeea7f16bull },
    { GGML_TYPE_HQ2_XS,  2048, 0x38aeb9575117cea2ull },
    { GGML_TYPE_HQ2_XXS,   256, 0xf48122f2ac589e4full },
    { GGML_TYPE_HQ2_XXS,   768, 0xed64bcccf16b3f03ull },
    { GGML_TYPE_HQ2_XXS,  2048, 0x49a31d04927f93a3ull },
};

static std::vector<uint8_t> quantize_uniform(ggml_type type, const std::vector<float> & x, int64_t n) {
    const int64_t nrows = (int64_t) x.size()/n;
    std::vector<uint8_t> q(nrows*ggml_row_size(type, n));
    quantize_hq(type, x.data(), q.data(), nrows, n);
    return q;
}

// ggml_quantize_rows_gptq over the rows in calls of g rows
static std::vector<uint8_t> quantize_gptq(ggml_type type, std::vector<float> x, int64_t n, const std::vector<float> & U, int64_t g = 32) {
    const int64_t nrows = (int64_t) x.size()/n;
    const size_t rs = ggml_row_size(type, n);
    std::vector<uint8_t> q(nrows*rs);
    for (int64_t r0 = 0; r0 < nrows; r0 += g) {
        ggml_quantize_rows_gptq(type, x.data() + r0*n, q.data() + r0*rs, std::min(g, nrows - r0), n, U.data());
    }
    return q;
}

static std::vector<float> scaled_identity(int64_t n, float c) {
    std::vector<float> U((size_t) n*(n + 1)/2, 0.0f);
    for (int64_t j = 0; j < n; ++j) {
        U[upos(n, j, j)] = c;
    }
    return U;
}

static std::vector<float> decode(ggml_type type, const std::vector<uint8_t> & q, int64_t nel) {
    std::vector<float> y(nel);
    ggml_get_type_traits(type)->to_float(q.data(), y.data(), nel);
    return y;
}

// sum over the rows of |diag(sqrt(vbar)) R^T (w - wq)|^2 / the same for w: the output error under H = R diag(vbar) R^T
static double out_err(const std::vector<float> & w, const std::vector<float> & wq, int64_t n, const std::vector<float> & vbar, uint64_t seed) {
    double num = 0, den = 0;
    std::vector<double> d(n), a(n);
    for (size_t r = 0; r*n < w.size(); ++r) {
        for (int64_t k = 0; k < n; ++k) {
            d[k] = (double) w[r*n + k] - wq[r*n + k];
            a[k] = w[r*n + k];
        }
        ggml_rht_inv_f64(d.data(), n, seed);
        ggml_rht_inv_f64(a.data(), n, seed);
        for (int64_t k = 0; k < n; ++k) {
            num += vbar[k]*d[k]*d[k];
            den += vbar[k]*a[k]*a[k];
        }
    }
    return num/den;
}

// a few loud channels over a log-normal floor, as with massive activations
static std::vector<float> spiky_v(int64_t n, uint64_t seed) {
    std::vector<float> v = make_v(V_LOGNORMAL, n, seed);
    for (int64_t k : { n/7, n/3, n/2 + 1, n - 5 }) {
        v[k] = 1e3f;
    }
    return v;
}

static std::vector<float> gauss_rows(int64_t nrows, int64_t n, uint64_t seed) {
    std::vector<float> x(nrows*n);
    uint64_t s = seed;
    for (auto & v : x) v = 0.02f*gauss(s);
    return x;
}

static void test_hashes() {
    printf("quantize_hq output is unchanged (hashes recorded before the shared driver):\n");
    for (ggml_type type : all_types) {
        bool ok = true;
        std::string ns;
        for (const auto & h : hq_hashes) {
            if (h.type != type) continue;
            uint64_t hash = 0xcbf29ce484222325ull;
            for (uint64_t seed = 1; seed <= 3; ++seed) {
                const std::vector<uint8_t> q = quantize_uniform(type, hash_rows(8, h.n, seed), h.n);
                hash = fnv1a(hash, q.data(), q.size());
            }
            ok &= hash == h.hash;
            ns += (ns.empty() ? "" : ", ") + std::to_string(h.n);
        }
        check(ok, std::string(ggml_type_name(type)) + ", n = " + ns);
    }
}

static void test_identity(const std::vector<ggml_type> & types) {
    printf("U = c*I gives the uniform quantizer's bytes:\n");
    for (ggml_type type : types) {
        bool ok = true;
        for (int64_t n : widths(type)) {
            std::vector<float> x = hash_rows(8, n, 5);
            if (type == GGML_TYPE_HQ3_S) {
                // the IQ3_S search doesn't advance its output after an all-zero 32-block of a nonzero
                // super-block, so it misplaces that super-block's later codes; the group step writes
                // every group in place
                for (int64_t r = 2; r < 8; r += 4) {
                    for (int64_t j = 32; j < 64; ++j) x[r*n + j] = 0.001f*(float) (j - 40);
                }
            }
            ok &= quantize_gptq(type, x, n, scaled_identity(n, 0.7f)) == quantize_uniform(type, x, n);
        }
        check(ok, ggml_type_name(type));
    }
}

static void test_encoder(const std::vector<ggml_type> & types, double max_ratio) {
    printf("GPTQ encoder (max output error ratio vs uniform %.2f):\n", max_ratio);
    std::vector<std::thread> workers;
    const uint64_t seed = 0x48512d524854ull;

    for (ggml_type type : types) {
        const int64_t n = type == GGML_TYPE_HQ4_NL ? 800 : 2048;
        const int64_t nrows = 64;
        const std::vector<float> v = spiky_v(n, 3);
        std::vector<float> vbar, U;
        llama_gptq_normalize(v.data(), n, vbar);
        llama_gptq_factor(v.data(), n, seed, LLAMA_GPTQ_DAMP_DEFAULT, U, workers, g_nthread);

        const std::vector<float> x = gauss_rows(nrows, n, 11);
        const std::vector<uint8_t> qu = quantize_uniform(type, x, n);
        const std::vector<uint8_t> qg = quantize_gptq(type, x, n, U);
        const double eu = out_err(x, decode(type, qu, nrows*n), n, vbar, seed);
        const double eg = out_err(x, decode(type, qg, nrows*n), n, vbar, seed);
        char buf[256];
        snprintf(buf, sizeof(buf), "%s, n = %" PRId64 ": output error %.5f -> %.5f (%.2fx)", ggml_type_name(type), n, eu, eg, eg/eu);
        check(eg <= max_ratio*eu, buf);

        const ggml_type base = ggml_get_base_type(type);
        const std::vector<float> yb = decode(base, qg, nrows*n), yh = decode(type, qg, nrows*n);
        bool finite = true;
        for (float y : yb) finite &= std::isfinite(y);
        check(ggml_validate_row_data(type, qg.data(), qg.size()) && finite && yb == yh,
              std::string(ggml_type_name(type)) + ": valid, decodes with " + ggml_type_name(base) + "'s to_float");
        if (parity_signs(type)) {
            bool even = true;
            for (size_t g = 0; g < yb.size(); g += 8) {
                int neg = 0;
                for (int i = 0; i < 8; ++i) neg += std::signbit(yb[g + i]) && yb[g + i] != 0.0f;
                even &= neg % 2 == 0;
            }
            check(even, std::string(ggml_type_name(type)) + ": every decoded group has even sign parity");
        }

        bool same = true;
        for (int64_t g : { 1, 7 }) {
            same &= quantize_gptq(type, x, n, U, g) == qg;
        }
        std::vector<uint8_t> q1(qg.size()), q8(qg.size());
        std::vector<float> x1 = x, x8 = x;
        llama_tensor_quantize_gptq(type, x1.data(), q1.data(), nrows, n, U.data(), workers, 1);
        llama_tensor_quantize_gptq(type, x8.data(), q8.data(), nrows, n, U.data(), workers, 8);
        check(same && q1 == qg && q8 == qg, std::string(ggml_type_name(type)) + ": the same bytes for calls of 1, 7, 32 rows and 1, 8 threads");
    }
}

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "--quick";

    if (!quick) {
        test_factor();
    }
    test_cache();
    test_hashes();
    test_identity({ std::begin(all_types), std::end(all_types) });
    test_encoder({ std::begin(elementwise_types), std::end(elementwise_types) }, 0.7);
    test_encoder({ std::begin(codebook_types), std::end(codebook_types) }, 0.85);

    printf("\n%s: %d failure(s)\n", g_failed ? "FAIL" : "PASS", g_failed);
    return g_failed ? 1 : 0;
}
