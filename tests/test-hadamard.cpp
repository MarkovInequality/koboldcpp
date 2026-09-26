// Unit tests for the randomized Hadamard transform (RHT), the HQ types and the CPU RHT op.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-hadamard.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#define GGML_HAD_DECL(K) static const char test_had_##K[]
#include "ggml-hadamard-tables.h"
#undef GGML_HAD_DECL

static const std::vector<std::pair<int, const char *>> header_orders = {
#define X(K) { K, test_had_##K },
    GGML_HAD_ORDERS(X)
#undef X
};

static int g_failed = 0;

static void check(bool ok, const std::string & name) {
    printf("  %-72s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    if (!ok) {
        g_failed++;
    }
}

static std::vector<float> make_random(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto & x : v) {
        x = dist(rng);
    }
    return v;
}

static std::vector<float> rht(const std::vector<float> & x, uint64_t seed) {
    std::vector<float> y = x;
    ggml_rht_ref(y.data(), (int64_t) y.size(), seed);
    return y;
}

static const std::vector<int64_t> dims_11 = {
    5120, 6144, 10240, 17408, 2560, 4096, 9728, 1024, 2048, 3072,
    14336, 8192, 28672, 11008, 3584, 5376, 21504, 12288, 25600, 18944, 13824, 27648,
};

// runs first: every order's matrix cache is still cold
static void test_thread_safety() {
    printf("concurrent first use of every order:\n");

    std::vector<int64_t> ns;
    for (const auto & [K, rows] : header_orders) {
        if (K % 8 != 0) {
            ns.push_back(4*(int64_t) K);
        }
    }

    const int nth = 16;
    std::vector<std::vector<float>> in(ns.size());
    std::vector<std::vector<std::vector<float>>> out(nth, std::vector<std::vector<float>>(ns.size()));
    for (size_t i = 0; i < ns.size(); ++i) {
        in[i] = make_random(ns[i], 100 + (uint32_t) i);
    }

    std::vector<std::thread> threads;
    for (int t = 0; t < nth; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t k = 0; k < ns.size(); ++k) {
                const size_t i = (k + t) % ns.size();
                out[t][i] = in[i];
                ggml_rht_ref(out[t][i].data(), ns[i], 7);
            }
        });
    }
    for (auto & th : threads) {
        th.join();
    }

    bool same = true;
    for (size_t i = 0; i < ns.size(); ++i) {
        const std::vector<float> ref = rht(in[i], 7);
        for (int t = 0; t < nth; ++t) {
            same &= memcmp(ref.data(), out[t][i].data(), ns[i]*sizeof(float)) == 0;
        }
    }
    check(same, "16 threads, first use of each K, match a single-threaded run");
}

static void test_table() {
    printf("table:\n");

    std::set<int> seen;
    bool unique = true;
    for (const auto & [K, rows] : header_orders) {
        unique &= seen.insert(K).second;
    }
    check(unique, "each order appears once in GGML_HAD_ORDERS");

    std::set<int> expected = { 40 };
    for (int m = 3; 4*m <= GGML_RHT_K_MAX; m += 2) {
        expected.insert(4*m);
    }
    check(seen == expected, "orders are 4*odd for 12..252, plus 40");

    std::set<int> accepted;
    std::vector<float> H((size_t) 512*512);
    for (int K = 0; K <= 512; ++K) {
        if (ggml_hadamard_matrix(K, H.data())) {
            accepted.insert(K);
        }
    }
    check(accepted == expected, "ggml_hadamard_matrix accepts exactly the table orders");

    for (const auto & [K, rows] : header_orders) {
        std::vector<int> h((size_t) K*K);
        for (int i = 0; i < K*K; ++i) {
            h[i] = rows[i] == '+' ? 1 : (rows[i] == '-' ? -1 : 0);
        }

        bool ok = strlen(rows) == (size_t) K*K;
        for (int i = 0; i < K && ok; ++i) {
            for (int j = 0; j < K && ok; ++j) {
                int64_t dot = 0;
                for (int k = 0; k < K; ++k) {
                    dot += h[i*K + k]*h[j*K + k];
                }
                ok = dot == (i == j ? K : 0);
            }
        }

        std::vector<float> Hf((size_t) K*K);
        ggml_hadamard_matrix(K, Hf.data());
        bool rows_ok = true;
        for (int i = 0; i < K*K; ++i) {
            rows_ok &= Hf[i] == (float) h[i];
        }

        const float * cached = ggml_rht_matrix_f32(K);
        const bool cached_ok = cached && memcmp(cached, Hf.data(), Hf.size()*sizeof(float)) == 0;

        check(ok && rows_ok && cached_ok, "H_" + std::to_string(K) + ": H*H^T == K*I, accessor row i == header row i");
    }
}

static void test_plan() {
    printf("decomposition rule:\n");

    struct tc { int64_t n; int K; int64_t P; };
    const std::vector<tc> cases = {
        { 5120, 20, 256 }, { 6144, 12, 512 }, { 10240, 20, 512 }, { 17408, 68, 256 },
        { 2560, 20, 128 }, { 4096, 1, 4096 }, { 9728, 76, 128 }, { 1024, 1, 1024 }, { 2048, 1, 2048 },
        { 3072, 12, 256 }, { 14336, 28, 512 }, { 8192, 1, 8192 }, { 28672, 28, 1024 }, { 11008, 172, 64 },
        { 3584, 28, 128 }, { 5376, 84, 64 }, { 21504, 84, 256 }, { 12288, 12, 1024 }, { 25600, 100, 256 },
        { 18944, 148, 128 }, { 13824, 108, 128 }, { 27648, 108, 256 },
        { 24*256, 12, 512 }, { 128, 1, 128 }, { 40*256, 20, 512 }, { 64*63, 252, 16 },
        { 64*65, 0, 0 }, { 6, 0, 0 }, { 2*63, 0, 0 }, { 0, 0, 0 },
    };

    for (const auto & c : cases) {
        int     K = -1;
        int64_t P = -1;
        const bool ok = ggml_rht_plan(c.n, &K, &P);
        const bool pass = c.K == 0 ? !ok : (ok && K == c.K && P == c.P);
        char buf[128];
        snprintf(buf, sizeof(buf), "n = %" PRId64 " -> %s", c.n,
                 c.K == 0 ? "NONE" : ("(" + std::to_string(c.K) + ", " + std::to_string(c.P) + ")").c_str());
        check(pass, buf);
    }

    bool all = true;
    for (int64_t n = 1; n <= 65536; ++n) {
        int64_t m = n;
        int     t = 0;
        while (m % 2 == 0) { m /= 2; t++; }
        const bool want = m == 1 || (t >= 2 && m <= 63);
        int K; int64_t P;
        const bool got = ggml_rht_plan(n, &K, &P);
        all &= got == want;
        if (got) {
            all &= (int64_t) K*P == n && (P & (P - 1)) == 0 && (K == 1 || (K == 4*m));
        }
    }
    check(all, "n = 1..65536: accepted iff divisible by 4 with odd part <= 63 (or a power of 2)");
}

static uint64_t splitmix64_step(uint64_t & state) {
    state += GGML_RHT_GAMMA;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void test_signs() {
    printf("signs:\n");

    check(ggml_rht_sign_word(0, 64, 0) == 0x2a7b67af6c6ad50eull, "golden (seed 0, n 64) word 0");

    const uint64_t golden42[4] = { 0xe9239fdbd1fd73cbull, 0xce417ca0454148f0ull, 0x70dc16857a11f4f1ull, 0x4b8d8bc782ad1c24ull };
    bool ok = true;
    uint64_t fnv = 0xcbf29ce484222325ull;
    for (int k = 0; k < 80; ++k) {
        const uint64_t w = ggml_rht_sign_word(42, 5120, k);
        if (k < 4) {
            ok &= w == golden42[k];
        }
        for (int b = 0; b < 8; ++b) {
            fnv ^= (w >> (8*b)) & 0xff;
            fnv *= 0x100000001b3ull;
        }
    }
    check(ok && ggml_rht_sign_word(42, 5120, 79) == 0xc7df6955514c4871ull && fnv == 0xcec84e1c346746a6ull,
          "golden (seed 42, n 5120) words 0..79");

    bool stepped = true;
    for (uint64_t seed : { 0ull, 42ull, 0xdeadbeefcafef00dull }) {
        for (int64_t n : { 64ll, 5120ll, 17408ll }) {
            uint64_t state = seed ^ (GGML_RHT_GAMMA * (uint64_t) n);
            for (int64_t k = 0; k < (n + 63)/64; ++k) {
                stepped &= splitmix64_step(state) == ggml_rht_sign_word(seed, n, k);
            }
        }
    }
    check(stepped, "word k == splitmix64 stepped k+1 times from seed ^ gamma*n");
}

static int popcount64(uint64_t x) {
    int c = 0;
    while (x) { x &= x - 1; c++; }
    return c;
}

// R = (1/sqrt(n)) * (H_K (x) H_P) * diag(s), natural order, H_K row-major from the header
static std::vector<double> dense_rht(int64_t n, uint64_t seed) {
    int K; int64_t P;
    ggml_rht_plan(n, &K, &P);

    std::vector<float> HK(1, 1.0f);
    if (K > 1) {
        HK.resize((size_t) K*K);
        ggml_hadamard_matrix(K, HK.data());
    }

    std::vector<double> R((size_t) n*n);
    for (int64_t r = 0; r < n; ++r) {
        const int64_t cr = r / P, jr = r % P;
        for (int64_t c = 0; c < n; ++c) {
            const int64_t cc = c / P, jc = c % P;
            const double hp = popcount64((uint64_t) (jr & jc)) % 2 ? -1.0 : 1.0;
            const double s  = (ggml_rht_sign_word(seed, n, c / 64) >> (c % 64)) & 1 ? -1.0 : 1.0;
            R[r*n + c] = HK[cr*K + cc] * hp * s / sqrt((double) n);
        }
    }
    return R;
}

static void test_reference() {
    printf("reference transform:\n");

    for (int64_t n : dims_11) {
        const std::vector<float> w = make_random(n, 1), x = make_random(n, 2);
        const std::vector<float> Rw = rht(w, 1234), Rx = rht(x, 1234);

        double nx = 0, nRx = 0, wx = 0, RwRx = 0;
        for (int64_t i = 0; i < n; ++i) {
            nx   += (double) x[i]*x[i];
            nRx  += (double) Rx[i]*Rx[i];
            wx   += (double) w[i]*x[i];
            RwRx += (double) Rw[i]*Rx[i];
        }
        const bool ok = fabs(sqrt(nRx) - sqrt(nx)) < 1e-5*sqrt(nx) && fabs(RwRx - wx) < 1e-4*sqrt(nx*nx);
        check(ok, "n = " + std::to_string(n) + ": |Rx| == |x|, <Rw,Rx> == <w,x>");
    }

    for (int64_t n : { 64ll, 192ll, 160ll, 112ll, 272ll, 688ll }) {
        int K; int64_t P;
        ggml_rht_plan(n, &K, &P);
        const uint64_t seed = 99;
        const std::vector<double> R = dense_rht(n, seed);

        double err = 0;
        for (int64_t b = 0; b < n + 4; ++b) {
            std::vector<float> x = b < n ? std::vector<float>(n, 0.0f) : make_random(n, 1000 + (uint32_t) b);
            if (b < n) {
                x[b] = 1.0f;
            }
            const std::vector<float> y = rht(x, seed);
            for (int64_t r = 0; r < n; ++r) {
                double ref = 0;
                for (int64_t c = 0; c < n; ++c) {
                    ref += R[r*n + c]*x[c];
                }
                err = std::max(err, fabs(ref - y[r]));
            }
        }
        check(err < 1e-5, "(K, P) = (" + std::to_string(K) + ", " + std::to_string(P) + "): equals the dense R on basis and random vectors");
    }
}

static void test_f64() {
    printf("fp64 transform:\n");

    std::vector<int64_t> ns;
    for (const auto & [K, rows] : header_orders) {
        if (K % 8 != 0) {
            ns.push_back(64*(int64_t) K);
        }
    }
    for (int64_t n : { 1024ll, 2048ll, 3072ll, 2560ll, 4096ll, 5120ll, 6144ll, 9728ll, 17408ll }) {
        ns.push_back(n);
    }

    for (int64_t n : ns) {
        int K; int64_t P;
        ggml_rht_plan(n, &K, &P);
        const std::vector<float> x = make_random(n, 11);
        const std::vector<float> y = rht(x, 4321);

        std::vector<double> xd(x.begin(), x.end());
        ggml_rht_ref_f64(xd.data(), n, 4321);
        double err = 0, ny = 0, nx = 0, nxd = 0;
        for (int64_t i = 0; i < n; ++i) {
            err += (xd[i] - y[i])*(xd[i] - y[i]);
            ny  += (double) y[i]*y[i];
            nx  += (double) x[i]*x[i];
            nxd += xd[i]*xd[i];
        }
        ggml_rht_inv_f64(xd.data(), n, 4321);
        double inv = 0;
        for (int64_t i = 0; i < n; ++i) {
            inv += (xd[i] - x[i])*(xd[i] - x[i]);
        }
        const std::string what = "(K, P) = (" + std::to_string(K) + ", " + std::to_string(P) + "): ";
        check(sqrt(err/ny) < 1e-6, what + "f64 == ggml_rht_ref to fp32 rounding");
        check(sqrt(inv/nx) < 1e-12 && fabs(sqrt(nxd) - sqrt(nx)) < 1e-12*sqrt(nx), what + "inv(ref(x)) == x, |ref(x)| == |x|");
    }
}

static void test_stage_split() {
    printf("stage split:\n");

    for (int64_t n : { 5120ll, 17408ll, 4096ll, 11008ll, 4032ll }) {
        int K; int64_t P;
        ggml_rht_plan(n, &K, &P);
        const std::vector<float> x   = make_random(n, 5);
        const std::vector<float> ref = rht(x, 77);

        bool ok = true;
        for (int parts : { 1, 3, 8 }) {
            std::vector<float> y = x;
            for (int p = 0; p < parts; ++p) {
                ggml_rht_stage_chunks(y.data(), n, 77, P, (int64_t) K*p/parts, (int64_t) K*(p + 1)/parts);
            }
            for (int p = parts - 1; p >= 0; --p) {
                ggml_rht_stage_mix(y.data(), n, K, P, ggml_rht_matrix_f32(K), P*p/parts, P*(p + 1)/parts);
            }
            ok &= memcmp(y.data(), ref.data(), n*sizeof(float)) == 0;
        }
        check(ok, "n = " + std::to_string(n) + ": 1, 3 and 8-way splits give the bytes of ggml_rht_ref");

        bool ok_out = true;
        std::vector<float> u = x;
        if (K == 1) {
            for (int64_t P1 : { 2ll, 16ll }) {
                u = x;
                const int64_t cs = P/P1;
                for (int64_t c = 0; c < P1; ++c) {
                    ggml_rht_stage_chunks(u.data(), n, 77, cs, c, c + 1);
                }
                std::vector<float> y(n, 0.0f);
                for (int parts : { 1, 3, 8 }) {
                    for (int p = 0; p < parts; ++p) {
                        ggml_rht_stage_fwht_outer(u.data(), y.data(), n, cs, cs*p/parts, cs*(p + 1)/parts);
                    }
                    ok_out &= memcmp(y.data(), ref.data(), n*sizeof(float)) == 0;
                }
            }
            check(ok_out, "n = " + std::to_string(n) + ": H_P1 (x) H_P2 split with the outer stage gives the same bytes");
        } else {
            ggml_rht_stage_chunks(u.data(), n, 77, P, 0, K);
            std::vector<float> y(n, 0.0f);
            for (int jp : { 1, 3 }) {
                for (int gp : { 1, 5 }) {
                    for (int a = 0; a < jp; ++a) {
                        for (int b = 0; b < gp; ++b) {
                            ggml_rht_stage_mix_out(u.data(), y.data(), n, K, P, ggml_rht_matrix_f32(K),
                                                   P*a/jp, P*(a + 1)/jp, K*b/gp, K*(b + 1)/gp);
                        }
                    }
                    ok_out &= memcmp(y.data(), ref.data(), n*sizeof(float)) == 0;
                }
            }
            check(ok_out, "n = " + std::to_string(n) + ": out-of-place mix over position and output splits gives the same bytes");
        }
    }
}

static const std::vector<std::pair<ggml_type, ggml_type>> hq_pairs = {
    { GGML_TYPE_Q4_K,    GGML_TYPE_HQ4_K   }, { GGML_TYPE_Q5_K,    GGML_TYPE_HQ5_K   },
    { GGML_TYPE_IQ2_XXS, GGML_TYPE_HQ2_XXS }, { GGML_TYPE_IQ2_XS,  GGML_TYPE_HQ2_XS  },
    { GGML_TYPE_IQ2_S,   GGML_TYPE_HQ2_S   }, { GGML_TYPE_IQ3_XXS, GGML_TYPE_HQ3_XXS },
    { GGML_TYPE_IQ3_S,   GGML_TYPE_HQ3_S   }, { GGML_TYPE_IQ4_NL,  GGML_TYPE_HQ4_NL  },
    { GGML_TYPE_IQ4_XS,  GGML_TYPE_HQ4_XS  },
};

static void test_types() {
    printf("types:\n");

    const std::map<ggml_type, std::string> names = {
        { GGML_TYPE_HQ4_K, "hq4_K" }, { GGML_TYPE_HQ5_K, "hq5_K" }, { GGML_TYPE_HQ2_XXS, "hq2_xxs" },
        { GGML_TYPE_HQ2_XS, "hq2_xs" }, { GGML_TYPE_HQ2_S, "hq2_s" }, { GGML_TYPE_HQ3_XXS, "hq3_xxs" },
        { GGML_TYPE_HQ3_S, "hq3_s" }, { GGML_TYPE_HQ4_NL, "hq4_nl" }, { GGML_TYPE_HQ4_XS, "hq4_xs" },
    };
    check(GGML_TYPE_COUNT == 159 && GGML_TYPE_HQ4_K == 150 && GGML_TYPE_HQ4_XS == 158, "indices 150..158, GGML_TYPE_COUNT == 159");

    for (const auto & [base, hq] : hq_pairs) {
        const ggml_type_traits * tb = ggml_get_type_traits(base);
        const ggml_type_traits * th = ggml_get_type_traits(hq);
        const bool traits_ok =
            th->blck_size == tb->blck_size && th->blck_size_interleave == tb->blck_size_interleave &&
            th->type_size == tb->type_size && th->is_quantized == tb->is_quantized &&
            th->to_float == tb->to_float && th->from_float_ref == nullptr;

        const ggml_type_traits_cpu * cpu = ggml_get_type_traits_cpu(hq);
        const bool cpu_ok = cpu->from_float == nullptr && cpu->vec_dot == nullptr;

        const bool pair_ok = ggml_get_rotated_type(base) == hq && ggml_get_base_type(hq) == base &&
            ggml_is_rotated(hq) && !ggml_is_rotated(base) && ggml_get_rotated_type(hq) == hq;

        ggml_type parsed = GGML_TYPE_COUNT;
        for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
            if (names.at(hq) == ggml_type_name((ggml_type) t)) {
                parsed = (ggml_type) t;
            }
        }

        check(traits_ok && cpu_ok && pair_ok && parsed == hq,
              std::string(ggml_type_name(hq)) + ": traits == " + ggml_type_name(base) + "'s, no CPU traits, pairing, name");
    }

    bool others = true;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const ggml_type type = (ggml_type) t;
        if (names.count(type)) {
            continue;
        }
        others &= !ggml_is_rotated(type) && ggml_get_base_type(type) == type;
        bool is_base = false;
        for (const auto & p : hq_pairs) {
            is_base |= p.first == type;
        }
        others &= is_base || ggml_get_rotated_type(type) == type;
    }
    check(others, "every non-HQ type is its own base; only the nine bases have a rotated type");
}

static void test_validate() {
    printf("row validation:\n");

    const int64_t n = 512;
    const std::vector<float> x = make_random(n, 3);
    const std::vector<float> imat(n, 1.0f);

    for (const auto & [base, hq] : hq_pairs) {
        std::vector<uint8_t> row(ggml_row_size(base, n));
        ggml_quantize_chunk(base, x.data(), row.data(), 0, 1, n, imat.data());

        const bool valid = ggml_validate_row_data(base, row.data(), row.size()) &&
                           ggml_validate_row_data(hq,   row.data(), row.size());

        std::vector<uint8_t> bad = row;
        const uint16_t nan16 = 0x7e00;
        memcpy(bad.data(), &nan16, sizeof(nan16));
        const bool rejected = !ggml_validate_row_data(base, bad.data(), bad.size()) &&
                              !ggml_validate_row_data(hq,   bad.data(), bad.size());

        check(valid && rejected, std::string(ggml_type_name(hq)) + ": accepts a valid row, rejects a NaN scale like the base type");
    }
}

static std::vector<float> cpu_op(const std::vector<float> & x, int64_t n, int64_t nr, int64_t stride, uint64_t seed, int nth, bool as3d) {
    ggml_init_params ip = { (size_t) 64*1024*1024 + x.size()*8, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * buf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, stride*nr);
    memcpy(buf->data, x.data(), x.size()*sizeof(float));

    ggml_tensor * src = as3d
        ? ggml_view_3d(ctx, buf, n, 2, nr/2, stride*sizeof(float), 2*stride*sizeof(float), 0)
        : ggml_view_2d(ctx, buf, n, nr, stride*sizeof(float), 0);
    ggml_tensor * out = ggml_rht(ctx, src, seed);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, nth);

    std::vector<float> y(n*nr);
    memcpy(y.data(), out->data, y.size()*sizeof(float));
    ggml_free(ctx);
    return y;
}

static void test_cpu_op() {
    printf("CPU op:\n");

    for (int64_t n : { 5120ll, 3072ll, 4096ll, 17408ll, 688ll, 4032ll, 9728ll }) {
        for (int64_t nr : { 1ll, 2ll, 7ll, 16ll, 33ll }) {
            const int64_t stride = n + (nr > 1 ? 16 : 0);
            const std::vector<float> x = make_random(stride*nr, 9 + (uint32_t) nr);

            std::vector<float> ref(n*nr);
            for (int64_t r = 0; r < nr; ++r) {
                std::vector<float> row(x.begin() + r*stride, x.begin() + r*stride + n);
                ggml_rht_ref(row.data(), n, 4242);
                memcpy(ref.data() + r*n, row.data(), n*sizeof(float));
            }

            bool ok = true;
            for (int nth : { 1, 4, 16 }) {
                ok &= memcmp(cpu_op(x, n, nr, stride, 4242, nth, false).data(), ref.data(), ref.size()*sizeof(float)) == 0;
                if (nr % 2 == 0) {
                    ok &= memcmp(cpu_op(x, n, nr, stride, 4242, nth, true).data(), ref.data(), ref.size()*sizeof(float)) == 0;
                }
            }
            check(ok, "n = " + std::to_string(n) + ", rows = " + std::to_string(nr) +
                      (stride != n ? " (padded stride)" : "") + ": 1, 4, 16 threads == ggml_rht_ref");
        }
    }
}

int main() {
    ggml_cpu_init();

    test_thread_safety();
    test_table();
    test_plan();
    test_signs();
    test_reference();
    test_f64();
    test_stage_split();
    test_types();
    test_validate();
    test_cpu_op();

    printf("\n%s: %d failure(s)\n", g_failed ? "FAIL" : "PASS", g_failed);
    return g_failed ? 1 : 0;
}
