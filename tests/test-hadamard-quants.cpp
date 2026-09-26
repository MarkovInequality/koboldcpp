// Tests and measurements for the HQ quantizers (quantize_hq, reached through ggml_quantize_chunk).

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "ggml.h"
#include "ggml-quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

static int g_failed = 0;

static void check(bool ok, const std::string & name) {
    printf("  %-72s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    fflush(stdout);
    if (!ok) {
        g_failed++;
    }
}

struct hq_pair {
    ggml_type hq;
    ggml_type base;
};

static const hq_pair k_pairs[] = {
    { GGML_TYPE_HQ4_K,   GGML_TYPE_Q4_K    },
    { GGML_TYPE_HQ5_K,   GGML_TYPE_Q5_K    },
    { GGML_TYPE_HQ2_XXS, GGML_TYPE_IQ2_XXS },
    { GGML_TYPE_HQ2_XS,  GGML_TYPE_IQ2_XS  },
    { GGML_TYPE_HQ2_S,   GGML_TYPE_IQ2_S   },
    { GGML_TYPE_HQ3_XXS, GGML_TYPE_IQ3_XXS },
    { GGML_TYPE_HQ3_S,   GGML_TYPE_IQ3_S   },
    { GGML_TYPE_HQ4_NL,  GGML_TYPE_IQ4_NL  },
    { GGML_TYPE_HQ4_XS,  GGML_TYPE_IQ4_XS  },
    { GGML_TYPE_HQ4_0,   GGML_TYPE_Q4_0    },
    { GGML_TYPE_HQ4_1,   GGML_TYPE_Q4_1    },
    { GGML_TYPE_HQ5_0,   GGML_TYPE_Q5_0    },
    { GGML_TYPE_HQ5_1,   GGML_TYPE_Q5_1    },
    { GGML_TYPE_HQ8_0,   GGML_TYPE_Q8_0    },
    { GGML_TYPE_HQ2_K,   GGML_TYPE_Q2_K    },
    { GGML_TYPE_HQ3_K,   GGML_TYPE_Q3_K    },
    { GGML_TYPE_HQ6_K,   GGML_TYPE_Q6_K    },
};

static bool is_legacy(ggml_type base) {
    return base == GGML_TYPE_Q4_0 || base == GGML_TYPE_Q4_1 || base == GGML_TYPE_Q5_0 || base == GGML_TYPE_Q5_1 || base == GGML_TYPE_Q8_0;
}

static int n_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n ? (int) n : 4;
}

static void parallel_for(int64_t n, int nth, const std::function<void(int, int64_t, int64_t)> & f) {
    std::vector<std::thread> threads;
    for (int t = 0; t < nth; ++t) {
        const int64_t i0 = n*t/nth;
        const int64_t i1 = n*(t + 1)/nth;
        if (i0 < i1) {
            threads.emplace_back(f, t, i0, i1);
        }
    }
    for (auto & th : threads) {
        th.join();
    }
}

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static uint64_t splitmix64(uint64_t & s) {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static float urand(uint64_t & s) {
    return (float) (splitmix64(s) >> 40) * (1.0f/16777216.0f);
}

enum dist_kind { DIST_GAUSS, DIST_LAPLACE };

static const char * dist_name(dist_kind d) {
    return d == DIST_GAUSS ? "gauss" : "laplace";
}

// unit variance, each row seeded independently so generation can be split across threads
static std::vector<float> make_rows(int64_t nrows, int64_t n, dist_kind d, uint64_t seed) {
    std::vector<float> x(nrows*n);
    parallel_for(nrows, n_threads(), [&](int, int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            uint64_t s = seed ^ (0xd1b54a32d192ed03ull * (uint64_t) (r + 1));
            float * xr = x.data() + r*n;
            for (int64_t i = 0; i < n; ++i) {
                if (d == DIST_GAUSS) {
                    const double u1 = (double) ((splitmix64(s) >> 11) + 1) * 0x1.0p-53;
                    const double u2 = (double) (splitmix64(s) >> 11) * 0x1.0p-53;
                    xr[i] = (float) (std::sqrt(-2.0*std::log(u1)) * std::cos(2.0*M_PI*u2));
                } else {
                    const double u = ((double) (splitmix64(s) >> 11) + 0.5) * 0x1.0p-53 - 0.5;
                    xr[i] = (float) ((u < 0 ? M_SQRT1_2 : -M_SQRT1_2) * std::log(1.0 - 2.0*std::fabs(u)));
                }
            }
        }
    });
    return x;
}

static void rotate_rows(std::vector<float> & x, int64_t n, uint64_t seed) {
    const int64_t nrows = (int64_t) x.size()/n;
    parallel_for(nrows, n_threads(), [&](int, int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            ggml_rht_ref(x.data() + r*n, n, seed);
        }
    });
}

static std::vector<uint8_t> quantize(ggml_type type, const std::vector<float> & x, int64_t n, const float * imatrix, int nth) {
    const int64_t nrows = (int64_t) x.size()/n;
    std::vector<uint8_t> q(nrows*ggml_row_size(type, n));
    parallel_for(nrows, nth, [&](int, int64_t r0, int64_t r1) {
        ggml_quantize_chunk(type, x.data(), q.data(), r0*n, r1 - r0, n, imatrix);
    });
    return q;
}

static std::vector<float> dequantize(ggml_type type, const std::vector<uint8_t> & q, int64_t nrows, int64_t n) {
    std::vector<float> y(nrows*n);
    const size_t rs = ggml_row_size(type, n);
    parallel_for(nrows, n_threads(), [&](int, int64_t r0, int64_t r1) {
        ggml_get_type_traits(type)->to_float(q.data() + r0*rs, y.data() + r0*n, (r1 - r0)*n);
    });
    return y;
}

static double mse(const std::vector<float> & x, const std::vector<float> & y) {
    const int nth = n_threads();
    std::vector<double> part(nth, 0.0);
    parallel_for((int64_t) x.size(), nth, [&](int t, int64_t i0, int64_t i1) {
        double s = 0;
        for (int64_t i = i0; i < i1; ++i) {
            const double d = (double) x[i] - y[i];
            s += d*d;
        }
        part[t] = s;
    });
    double s = 0;
    for (double p : part) {
        s += p;
    }
    return s/(double) x.size();
}

static double quant_mse(ggml_type type, const std::vector<float> & x, int64_t n, const float * imatrix) {
    const std::vector<uint8_t> q = quantize(type, x, n, imatrix, n_threads());
    return mse(x, dequantize(type, q, (int64_t) x.size()/n, n));
}

static bool uses_iq2_grid(ggml_type t) {
    return t == GGML_TYPE_HQ2_XXS || t == GGML_TYPE_HQ2_XS || t == GGML_TYPE_HQ2_S;
}

static bool uses_iq3_grid(ggml_type t) {
    return t == GGML_TYPE_HQ3_XXS || t == GGML_TYPE_HQ3_S;
}

static bool grid_ready(ggml_type hq) {
    const int * map;
    const uint16_t * nbrs;
    if (uses_iq2_grid(hq)) {
        const uint64_t * grid;
        ggml_iq2_entry(ggml_get_base_type(hq), &grid, &map, &nbrs);
        return grid && map && nbrs;
    }
    const uint32_t * grid;
    ggml_iq3_entry(hq == GGML_TYPE_HQ3_XXS ? 256 : 512, &grid, &map, &nbrs);
    return grid && map && nbrs;
}

// ---------------------------------------------------------------------------------------------

// must run before anything else calls ggml_quantize_init
static void test_cold_start() {
    printf("cold start (HQ i-quants quantized before any base-type init):\n");
    const int64_t n = 3072, nrows = 4;
    const std::vector<float> x = make_rows(nrows, n, DIST_GAUSS, 11);
    for (const hq_pair & p : k_pairs) {
        if (!uses_iq2_grid(p.hq) && !uses_iq3_grid(p.hq)) {
            continue;
        }
        const bool cold = !grid_ready(p.hq);
        std::vector<uint8_t> q(nrows*ggml_row_size(p.hq, n));
        ggml_quantize_chunk(p.hq, x.data(), q.data(), 0, nrows, n, nullptr);
        const double rel = mse(x, dequantize(p.hq, q, nrows, n));
        check(cold && grid_ready(p.hq) && rel < 0.25,
              std::string(ggml_type_name(p.hq)) + ": grid cold before, ready after, rel MSE " + std::to_string(rel).substr(0, 6));
    }
}

static void test_requires_imatrix() {
    printf("ggml_quantize_requires_imatrix:\n");
    bool ok = true;
    for (const hq_pair & p : k_pairs) {
        ok = ok && !ggml_quantize_requires_imatrix(p.hq);
    }
    check(ok, "false for every HQ type");
}

// ---------------------------------------------------------------------------------------------
// base quantizers unchanged: FNV-1a of ggml_quantize_chunk output, recorded before the HQ
// quantizers and the grid accessors were added

static void make_hash_input(std::vector<float> & x, std::vector<float> & imatrix, int64_t nrows, int64_t n) {
    uint64_t s = 1234;
    x.resize(nrows*n);
    for (int64_t r = 0; r < nrows; ++r) {
        const float scale = 0.01f*(float) (1 + r);
        for (int64_t i = 0; i < n; ++i) {
            float g = -6.0f;
            for (int k = 0; k < 12; ++k) {
                g += urand(s);
            }
            if (i % 97 == 0) {
                g *= 8.0f;
            }
            x[r*n + i] = scale*g;
        }
    }
    imatrix.resize(n);
    uint64_t t = 99;
    for (int64_t i = 0; i < n; ++i) {
        imatrix[i] = 0.25f + 2.0f*urand(t);
    }
}

static uint64_t fnv1a64(const uint8_t * p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

struct base_hash {
    ggml_type type;
    uint64_t no_imatrix; // 0: the type requires an imatrix
    uint64_t imatrix;
};

static const base_hash k_base_hashes[] = {
    { GGML_TYPE_F16    , 0x4e5505e7d57c8ab8ull, 0x4e5505e7d57c8ab8ull },
    { GGML_TYPE_BF16   , 0x1c5c80e42bdbf391ull, 0x1c5c80e42bdbf391ull },
    { GGML_TYPE_Q1_0   , 0x79a0a49d4796a768ull, 0x79a0a49d4796a768ull },
    { GGML_TYPE_Q2_0   , 0x8e927a24f3bc00eeull, 0x8e927a24f3bc00eeull },
    { GGML_TYPE_Q4_0   , 0x6b01ed9e0e4c1430ull, 0x4209e795b8efd4a4ull },
    { GGML_TYPE_Q4_1   , 0x255314a6c63e0903ull, 0x54d9a6015b58d341ull },
    { GGML_TYPE_Q5_0   , 0x8fa834f5bcca67f0ull, 0x9ab0c889f125a03bull },
    { GGML_TYPE_Q5_1   , 0x0026a5a89c07fa5aull, 0xd35a65a45dcc6271ull },
    { GGML_TYPE_Q8_0   , 0x4cdff26fea05236dull, 0x4cdff26fea05236dull },
    { GGML_TYPE_MXFP4  , 0x46a9dda2c2a07263ull, 0x46a9dda2c2a07263ull },
    { GGML_TYPE_NVFP4  , 0x105b0d8eae821850ull, 0x105b0d8eae821850ull },
    { GGML_TYPE_Q2_K   , 0x01c3d28ad8c19398ull, 0xb87297a4dc0b354bull },
    { GGML_TYPE_Q3_K   , 0x9771379369b0a0dfull, 0x723868c1eb70b2b4ull },
    { GGML_TYPE_Q4_K   , 0x31978fc32000f65bull, 0xf1b374822220b148ull },
    { GGML_TYPE_Q5_K   , 0x3bd6dde6ac2a2a54ull, 0x62ed19a460b3b21full },
    { GGML_TYPE_Q6_K   , 0x25ae7c6f20ecaf52ull, 0xa72dd88a14c7a482ull },
    { GGML_TYPE_TQ1_0  , 0xd8f2078e1249b66bull, 0xd8f2078e1249b66bull },
    { GGML_TYPE_TQ2_0  , 0xab2ef8a91fc90084ull, 0xab2ef8a91fc90084ull },
    { GGML_TYPE_IQ2_XXS, 0x0000000000000000ull, 0x778e3f3206e99ac8ull },
    { GGML_TYPE_IQ2_XS , 0x0000000000000000ull, 0x584ba27a362305e6ull },
    { GGML_TYPE_IQ2_S  , 0x9671c14f554bc98eull, 0x418e926aea8638a7ull },
    { GGML_TYPE_IQ3_XXS, 0xe293b3bab8dfe7b1ull, 0xfee41810c1672effull },
    { GGML_TYPE_IQ3_S  , 0x913ba321224cf8acull, 0x643d9fbec4ac2d13ull },
    { GGML_TYPE_IQ1_S  , 0x0000000000000000ull, 0x2d66b75fde0275b2ull },
    { GGML_TYPE_IQ1_M  , 0xaab44a7c4cc7ceaaull, 0xe381e99abbf02707ull },
    { GGML_TYPE_IQ4_NL , 0x970b6ebe5fbcc7d6ull, 0x1bc505146acaf394ull },
    { GGML_TYPE_IQ4_XS , 0x142ba81e93909ac2ull, 0x681e5597ac7e61a1ull },
};

static void test_base_hashes() {
    printf("base quantizers unchanged (output hashes):\n");
    const int64_t nrows = 8, n = 3072;
    std::vector<float> x, im;
    make_hash_input(x, im, nrows, n);
    for (const base_hash & b : k_base_hashes) {
        uint64_t h[2] = { 0, 0 };
        for (int w = 0; w < 2; ++w) {
            if (w == 0 && ggml_quantize_requires_imatrix(b.type)) {
                continue;
            }
            std::vector<uint8_t> q(nrows*ggml_row_size(b.type, n), 0);
            ggml_quantize_chunk(b.type, x.data(), q.data(), 0, nrows, n, w ? im.data() : nullptr);
            h[w] = fnv1a64(q.data(), q.size());
        }
        check(h[0] == b.no_imatrix && h[1] == b.imatrix, std::string(ggml_type_name(b.type)) + ": without and with imatrix");
    }
}

// ---------------------------------------------------------------------------------------------

static void test_format_and_imatrix() {
    printf("format compatibility, imatrix ignored:\n");
    const int64_t n = 3072, nrows = 16;
    std::vector<float> x = make_rows(nrows, n, DIST_GAUSS, 21);
    rotate_rows(x, n, 5);
    std::vector<float> im(n);
    uint64_t s = 77;
    for (float & v : im) {
        v = 0.01f + 10.0f*urand(s);
    }
    for (const hq_pair & p : k_pairs) {
        const std::string name = ggml_type_name(p.hq);
        const std::vector<uint8_t> q    = quantize(p.hq, x, n, nullptr,   1);
        const std::vector<uint8_t> q_im = quantize(p.hq, x, n, im.data(), 1);
        check(q == q_im, name + ": an imatrix gives byte-identical output");

        const auto * tr_hq   = ggml_get_type_traits(p.hq);
        const auto * tr_base = ggml_get_type_traits(p.base);
        bool ok = tr_hq->to_float && tr_base->to_float;
        if (ok) {
            std::vector<float> y_hq(nrows*n), y_base(nrows*n);
            tr_hq->to_float(q.data(), y_hq.data(), nrows*n);
            tr_base->to_float(q.data(), y_base.data(), nrows*n);
            ok = memcmp(y_hq.data(), y_base.data(), y_hq.size()*sizeof(float)) == 0;
        }
        check(ok, name + ": HQ traits to_float == " + ggml_type_name(p.base) + " to_float, bitwise");
    }
}

static void test_determinism() {
    printf("determinism:\n");
    const int64_t n = 3072, nrows = 61;
    std::vector<float> x = make_rows(nrows, n, DIST_LAPLACE, 31);
    rotate_rows(x, n, 9);
    for (const hq_pair & p : k_pairs) {
        const std::vector<uint8_t> a = quantize(p.hq, x, n, nullptr, 1);
        const std::vector<uint8_t> b = quantize(p.hq, x, n, nullptr, 1);
        const std::vector<uint8_t> c = quantize(p.hq, x, n, nullptr, 7);
        check(a == b && a == c, std::string(ggml_type_name(p.hq)) + ": twice single-threaded and 7-threaded give the same bytes");
    }
}

// ---------------------------------------------------------------------------------------------
// the grid formats, read the way dequantize_row_iq* reads them

struct grid_desc {
    ggml_type hq;
    int gs;         // elements per grid point
    int sub;        // elements sharing one scale
    int ngrid;
    int kmaxq;
    int bits;       // lattice bits per coordinate in the map index
    const uint8_t * dec;
};

struct group_code {
    float   db;
    int     idx;
    uint8_t signs;
};

static void parse_groups(ggml_type base, const uint8_t * row, int64_t n, std::vector<group_code> & out) {
    out.clear();
    for (int64_t ibl = 0; ibl < n/QK_K; ++ibl) {
        switch (base) {
            case GGML_TYPE_IQ2_XXS: {
                const block_iq2_xxs * b = (const block_iq2_xxs *) row + ibl;
                const float d = ggml_fp16_to_fp32(b->d);
                for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                    uint32_t aux32[2];
                    memcpy(aux32, b->qs + 4*ib32, 2*sizeof(uint32_t));
                    const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
                    for (int l = 0; l < 4; ++l) {
                        out.push_back({ db, (int) ((aux32[0] >> 8*l) & 255), ksigns_iq2xs[(aux32[1] >> 7*l) & 127] });
                    }
                }
            } break;
            case GGML_TYPE_IQ2_XS: {
                const block_iq2_xs * b = (const block_iq2_xs *) row + ibl;
                const float d = ggml_fp16_to_fp32(b->d);
                for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                    const float db[2] = { d * (0.5f + (b->scales[ib32] & 0xf)) * 0.25f, d * (0.5f + (b->scales[ib32] >> 4)) * 0.25f };
                    for (int l = 0; l < 4; ++l) {
                        out.push_back({ db[l/2], b->qs[4*ib32 + l] & 511, ksigns_iq2xs[b->qs[4*ib32 + l] >> 9] });
                    }
                }
            } break;
            case GGML_TYPE_IQ2_S: {
                const block_iq2_s * b = (const block_iq2_s *) row + ibl;
                const float d = ggml_fp16_to_fp32(b->d);
                const uint8_t * qs = b->qs;
                const uint8_t * signs = b->qs + QK_K/8;
                for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                    const float db[2] = { d * (0.5f + (b->scales[ib32] & 0xf)) * 0.25f, d * (0.5f + (b->scales[ib32] >> 4)) * 0.25f };
                    for (int l = 0; l < 4; ++l) {
                        out.push_back({ db[l/2], qs[l] | ((b->qh[ib32] << (8 - 2*l)) & 0x300), signs[l] });
                    }
                    qs += 4;
                    signs += 4;
                }
            } break;
            case GGML_TYPE_IQ3_XXS: {
                const block_iq3_xxs * b = (const block_iq3_xxs *) row + ibl;
                const float d = ggml_fp16_to_fp32(b->d);
                const uint8_t * qs = b->qs;
                const uint8_t * scales_and_signs = b->qs + QK_K/4;
                for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                    uint32_t aux32;
                    memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
                    const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
                    for (int l = 0; l < 4; ++l) {
                        const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                        out.push_back({ db, qs[2*l + 0], (uint8_t) (signs & 15) });
                        out.push_back({ db, qs[2*l + 1], (uint8_t) (signs >> 4) });
                    }
                    qs += 8;
                }
            } break;
            case GGML_TYPE_IQ3_S: {
                const block_iq3_s * b = (const block_iq3_s *) row + ibl;
                const float d = ggml_fp16_to_fp32(b->d);
                const uint8_t * qs = b->qs;
                const uint8_t * qh = b->qh;
                const uint8_t * signs = b->signs;
                for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
                    for (int h = 0; h < 2; ++h) {
                        const float db = d * (1 + 2*((b->scales[ib32/2] >> 4*h) & 0xf));
                        for (int l = 0; l < 4; ++l) {
                            out.push_back({ db, qs[2*l + 0] | ((qh[h] << (8 - 2*l)) & 256), (uint8_t) (signs[l] & 15) });
                            out.push_back({ db, qs[2*l + 1] | ((qh[h] << (7 - 2*l)) & 256), (uint8_t) (signs[l] >> 4) });
                        }
                        qs += 8;
                        signs += 4;
                    }
                    qh += 2;
                }
            } break;
            default:
                GGML_ABORT("unexpected type");
        }
    }
}

static const grid_desc k_grids[] = {
    { GGML_TYPE_HQ2_XXS, 8, 32,  256, 3, 2, (const uint8_t *) iq2xxs_grid },
    { GGML_TYPE_HQ2_XS,  8, 16,  512, 3, 2, (const uint8_t *) iq2xs_grid  },
    { GGML_TYPE_HQ2_S,   8, 16, 1024, 3, 2, (const uint8_t *) iq2s_grid   },
    { GGML_TYPE_HQ3_XXS, 4, 32,  256, 8, 3, (const uint8_t *) iq3xxs_grid },
    { GGML_TYPE_HQ3_S,   4, 32,  512, 8, 3, (const uint8_t *) iq3s_grid   },
};

static const grid_desc * find_grid(ggml_type hq) {
    for (const grid_desc & g : k_grids) {
        if (g.hq == hq) {
            return &g;
        }
    }
    return nullptr;
}

static float group_err(const grid_desc & gd, const group_code & c, const float * x) {
    float e = 0;
    for (int i = 0; i < gd.gs; ++i) {
        const float xv = c.signs & (1 << i) ? -x[i] : x[i];
        const float diff = c.db*gd.dec[gd.gs*c.idx + i] - xv;
        e += diff*diff;
    }
    return e;
}

// ---------------------------------------------------------------------------------------------

// Each HQ quantizer is its base quantizer with uniform weights. The base quantizer given the
// imatrix qw = 1/sqrt(sigma2 + x^2) sees weights of 1 +- 1 ulp (Q6_K's takes the imatrix as its weights,
// so it gets ones), so it makes the same choices except at near-ties (0-3 blocks of 512 when this was
// written). The grid types and the legacy types then re-pick their codes at the stored scale, so there
// only the scales (and signs) must match, and no re-picked code may decode worse than the base's.
static void test_copy_fidelity() {
    printf("copy fidelity (base quantizer with a weight-cancelling imatrix):\n");
    const int64_t n = 4096, nrows = 32;
    std::vector<float> x = make_rows(nrows, n, DIST_GAUSS, 41);
    rotate_rows(x, n, 3);
    for (const hq_pair & p : k_pairs) {
        const ggml_type b = p.base;
        const bool legacy = is_legacy(b);
        const int group = b == GGML_TYPE_IQ4_NL || legacy ? 32 : QK_K;
        // the legacy _impls take sigma2 over the whole row
        const int64_t sgroup = legacy ? n : group;
        const float sigma_mul = b == GGML_TYPE_IQ2_XXS || b == GGML_TYPE_IQ2_XS || b == GGML_TYPE_Q2_K || legacy ? 1.0f : 2.0f;
        const size_t rs = ggml_row_size(b, n);
        const size_t bs = ggml_type_size(b)*(group/ggml_blck_size(b));
        // the scale bytes that lead a legacy block: d, or d and m
        const size_t hdr = b == GGML_TYPE_Q4_1 || b == GGML_TYPE_Q5_1 ? 4 : 2;
        std::vector<uint8_t> qb(rs), qh(rs);
        std::vector<float> qw(n), yb(n), yh(n);
        const grid_desc * gd = find_grid(p.hq);
        std::vector<group_code> cb, ch;
        int64_t same = 0, total = 0, repicked = 0, worse = 0;
        for (int64_t r = 0; r < nrows; ++r) {
            const float * xr = x.data() + r*n;
            for (int64_t g = 0; g < n; g += sgroup) {
                float sumx2 = 0;
                for (int64_t i = 0; i < sgroup; ++i) {
                    sumx2 += xr[g + i]*xr[g + i];
                }
                const float sigma2 = sigma_mul*sumx2/sgroup;
                for (int64_t i = 0; i < sgroup; ++i) {
                    qw[g + i] = b == GGML_TYPE_Q6_K ? 1.0f : 1.0f/sqrtf(sigma2 + xr[g + i]*xr[g + i]);
                }
            }
            ggml_quantize_chunk(b,    xr, qb.data(), 0, 1, n, qw.data());
            ggml_quantize_chunk(p.hq, xr, qh.data(), 0, 1, n, nullptr);
            if (legacy) {
                ggml_get_type_traits(b)->to_float(qb.data(), yb.data(), n);
                ggml_get_type_traits(b)->to_float(qh.data(), yh.data(), n);
                for (size_t o = 0; o < rs; o += bs) {
                    total++;
                    if (memcmp(qb.data() + o, qh.data() + o, hdr) != 0) {
                        continue;
                    }
                    same++;
                    const int64_t i0 = (int64_t) (o/bs)*group;
                    for (int64_t i = i0; i < i0 + group; ++i) {
                        repicked += yh[i] != yb[i];
                        worse    += std::fabs(xr[i] - yh[i]) > std::fabs(xr[i] - yb[i]);
                    }
                }
                continue;
            }
            if (!gd) {
                for (size_t o = 0; o < rs; o += bs) {
                    same += memcmp(qb.data() + o, qh.data() + o, bs) == 0;
                    total++;
                }
                continue;
            }
            parse_groups(b, qb.data(), n, cb);
            parse_groups(b, qh.data(), n, ch);
            const size_t gpb = QK_K/gd->gs;
            for (size_t g0 = 0; g0 < cb.size(); g0 += gpb) {
                bool match = true;
                for (size_t g = g0; g < g0 + gpb; ++g) {
                    match = match && cb[g].db == ch[g].db && cb[g].signs == ch[g].signs;
                }
                total++;
                if (!match) {
                    continue;
                }
                same++;
                for (size_t g = g0; g < g0 + gpb; ++g) {
                    repicked += cb[g].idx != ch[g].idx;
                    worse    += group_err(*gd, ch[g], xr + gd->gs*g) > group_err(*gd, cb[g], xr + gd->gs*g);
                }
            }
        }
        const double frac = (double) same/(double) total;
        char buf[160];
        if (legacy) {
            snprintf(buf, sizeof(buf), "%s: %.2f%% of blocks match %s's scale; %lld codes re-picked, %lld worse",
                     ggml_type_name(p.hq), 100.0*frac, ggml_type_name(b), (long long) repicked, (long long) worse);
        } else if (gd) {
            snprintf(buf, sizeof(buf), "%s: %.2f%% of blocks match %s but for codes; %lld re-picked, %lld worse",
                     ggml_type_name(p.hq), 100.0*frac, ggml_type_name(b), (long long) repicked, (long long) worse);
        } else {
            snprintf(buf, sizeof(buf), "%s: %.2f%% of %d-element blocks identical to %s", ggml_type_name(p.hq), 100.0*frac, group, ggml_type_name(b));
        }
        check(frac >= 0.98 && worse == 0, buf);
    }
}

// ---------------------------------------------------------------------------------------------

static const int64_t k_mse_rows = 4096;
static const int64_t k_mse_cols = 4096;

static void test_mse() {
    printf("aggregate MSE, %lld x %lld, HQ vs base (IQ2_XXS/XS base: all-ones imatrix, others: none):\n",
           (long long) k_mse_rows, (long long) k_mse_cols);
    printf("  %-8s %-8s %-10s %12s %12s %8s\n", "dist", "type", "base", "MSE HQ", "MSE base", "ratio");
    const std::vector<float> ones(k_mse_cols, 1.0f);
    for (dist_kind d : { DIST_GAUSS, DIST_LAPLACE }) {
        const std::vector<float> x = make_rows(k_mse_rows, k_mse_cols, d, 1000 + d);
        for (const hq_pair & p : k_pairs) {
            const float * im = ggml_quantize_requires_imatrix(p.base) ? ones.data() : nullptr;
            const double e_hq   = quant_mse(p.hq,   x, k_mse_cols, nullptr);
            const double e_base = quant_mse(p.base, x, k_mse_cols, im);
            printf("  %-8s %-8s %-10s %12.6e %12.6e %8.4f\n", dist_name(d), ggml_type_name(p.hq), ggml_type_name(p.base), e_hq, e_base, e_hq/e_base);
            check(e_hq <= 1.01*e_base, std::string(dist_name(d)) + " " + ggml_type_name(p.hq) + ": MSE <= 1.01 x base");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// neighbour list vs exhaustive nearest-codebook search

struct nn_stats {
    int64_t groups = 0;
    int64_t off_grid = 0;    // rounded lattice point not in the grid: the neighbour list was used
    int64_t nl_disagree = 0;
    double  nl_err = 0;
    double  ex_err = 0;
    int64_t out_disagree = 0;
    double  out_err = 0;
    double  out_ex_err = 0;
    int64_t parse_mismatch = 0;

    void add(const nn_stats & o) {
        groups += o.groups; off_grid += o.off_grid; nl_disagree += o.nl_disagree;
        nl_err += o.nl_err; ex_err += o.ex_err;
        out_disagree += o.out_disagree; out_err += o.out_err; out_ex_err += o.out_ex_err;
        parse_mismatch += o.parse_mismatch;
    }
};

static void test_neighbour_vs_exhaustive() {
    printf("neighbour list vs exhaustive search, rotated Gaussian rows (the table is reported, not asserted):\n");
    printf("  lattice: the quantizer's search at the least-squares scale of the chosen codes\n");
    printf("  output:  the stored codes vs an exhaustive pass over the decode grid at the stored scale\n");
    printf("  %-8s %6s %9s %11s %10s %11s %10s\n", "type", "grid", "off-grid", "lat disagr", "lat gap", "out disagr", "out gain");

    const int64_t n = 5120, nrows = 256;
    std::vector<float> x = make_rows(nrows, n, DIST_GAUSS, 51);
    rotate_rows(x, n, 42);

    for (const grid_desc & gd : k_grids) {
        const ggml_type base = ggml_get_base_type(gd.hq);
        const std::vector<uint8_t> q = quantize(gd.hq, x, n, nullptr, n_threads());
        const std::vector<float> y = dequantize(gd.hq, q, nrows, n);
        const size_t rs = ggml_row_size(gd.hq, n);

        const int8_t * lat;
        const int * map;
        const uint16_t * nbrs;
        if (gd.gs == 8) {
            const uint64_t * g;
            ggml_iq2_entry(base, &g, &map, &nbrs);
            lat = (const int8_t *) g;
        } else {
            const uint32_t * g;
            ggml_iq3_entry(gd.ngrid, &g, &map, &nbrs);
            lat = (const int8_t *) g;
        }

        const int gs = gd.gs;
        auto lat_err = [&](const float * xv, float s, int idx) {
            float e = 0;
            for (int i = 0; i < gs; ++i) {
                const float diff = s*lat[gs*idx + i] - xv[i];
                e += diff*diff;
            }
            return e;
        };
        auto dec_err = [&](const float * xv, float db, int idx) {
            float e = 0;
            for (int i = 0; i < gs; ++i) {
                const float diff = db*gd.dec[gs*idx + i] - xv[i];
                e += diff*diff;
            }
            return e;
        };

        const int nth = n_threads();
        std::vector<nn_stats> part(nth);
        parallel_for(nrows, nth, [&](int t, int64_t r0, int64_t r1) {
            nn_stats st;
            std::vector<group_code> codes;
            std::vector<float> xval(n);
            for (int64_t r = r0; r < r1; ++r) {
                const float * xr = x.data() + r*n;
                parse_groups(base, q.data() + r*rs, n, codes);
                for (size_t g = 0; g < codes.size(); ++g) {
                    for (int i = 0; i < gs; ++i) {
                        const bool neg = codes[g].signs & (1 << i);
                        xval[gs*g + i] = neg ? -xr[gs*g + i] : xr[gs*g + i];
                        const float v = codes[g].db * gd.dec[gs*codes[g].idx + i] * (neg ? -1.0f : 1.0f);
                        st.parse_mismatch += v != y[r*n + gs*g + i];
                    }
                }
                const int gpb = gd.sub/gs;
                for (size_t g0 = 0; g0 < codes.size(); g0 += gpb) {
                    float sumqx = 0, sumq2 = 0;
                    for (int g = (int) g0; g < (int) g0 + gpb; ++g) {
                        for (int i = 0; i < gs; ++i) {
                            const float qv = lat[gs*codes[g].idx + i];
                            sumqx += qv*xval[gs*g + i];
                            sumq2 += qv*qv;
                        }
                    }
                    const float s = sumq2 > 0 ? sumqx/sumq2 : 0.0f;
                    for (int g = (int) g0; g < (int) g0 + gpb; ++g) {
                        const float * xv = xval.data() + gs*g;
                        const group_code & c = codes[g];
                        if (c.db <= 0 || s <= 0) {
                            continue;
                        }
                        st.groups++;

                        const float id = 1/s;
                        int u = 0;
                        for (int i = 0; i < gs; ++i) {
                            int l = (int) nearbyintf(0.5f*(id*xv[i] - 1));
                            l = std::max(0, std::min(gd.kmaxq - 1, l));
                            u |= l << gd.bits*i;
                        }
                        int nl = map[u];
                        if (nl < 0) {
                            st.off_grid++;
                            const uint16_t * nb = nbrs - map[u] - 1;
                            float best = INFINITY;
                            for (int j = 1; j <= nb[0]; ++j) {
                                const float e = lat_err(xv, s, nb[j]);
                                if (e < best) {
                                    best = e;
                                    nl = nb[j];
                                }
                            }
                        }
                        float ex_best = INFINITY;
                        for (int k = 0; k < gd.ngrid; ++k) {
                            ex_best = std::min(ex_best, lat_err(xv, s, k));
                        }
                        const float nl_e = lat_err(xv, s, nl);
                        st.nl_disagree += nl_e > ex_best;
                        st.nl_err += nl_e;
                        st.ex_err += ex_best;

                        float out_best = INFINITY;
                        for (int k = 0; k < gd.ngrid; ++k) {
                            out_best = std::min(out_best, dec_err(xv, c.db, k));
                        }
                        const float out_e = dec_err(xv, c.db, c.idx);
                        st.out_disagree += out_e > out_best;
                        st.out_err += out_e;
                        st.out_ex_err += out_best;
                    }
                }
            }
            part[t] = st;
        });
        nn_stats st;
        for (const nn_stats & p : part) {
            st.add(p);
        }
        const double g = (double) st.groups;
        printf("  %-8s %6d %8.2f%% %10.3f%% %9.4f%% %10.3f%% %9.4f%%\n", ggml_type_name(gd.hq), gd.ngrid,
               100.0*st.off_grid/g, 100.0*st.nl_disagree/g, 100.0*(st.nl_err - st.ex_err)/st.nl_err,
               100.0*st.out_disagree/g, 100.0*(st.out_err - st.out_ex_err)/st.out_err);
        check(st.parse_mismatch == 0, std::string(ggml_type_name(gd.hq)) + ": parsed codes decode to the dequantized values, bitwise");
        check(st.out_err - st.out_ex_err <= 1e-6*st.out_err, std::string(ggml_type_name(gd.hq)) + ": no gain left from an exhaustive pass at the stored scale");
    }
    printf("  (disagreement counts only strictly worse choices; gap and gain are relative to the chosen codes' error)\n");
}

// ---------------------------------------------------------------------------------------------
// round trip through a dense R^T

static void test_round_trip() {
    const int64_t n = 3072, nrows = 64;
    const uint64_t seed = 12345;
    printf("round trip: R^T dequant(quant(R w)) vs w, n = %lld, %lld Laplacian rows with outlier columns:\n", (long long) n, (long long) nrows);

    std::vector<float> R(n*n);
    parallel_for(n, n_threads(), [&](int, int64_t j0, int64_t j1) {
        std::vector<float> e(n);
        for (int64_t j = j0; j < j1; ++j) {
            std::fill(e.begin(), e.end(), 0.0f);
            e[j] = 1.0f;
            ggml_rht_ref(e.data(), n, seed);
            for (int64_t i = 0; i < n; ++i) {
                R[i*n + j] = e[i];
            }
        }
    });

    std::vector<float> w = make_rows(nrows, n, DIST_LAPLACE, 61);
    for (int64_t r = 0; r < nrows; ++r) {
        for (int64_t j = 0; j < n; j += 101) {
            w[r*n + j] *= 20.0f;
        }
    }
    std::vector<float> xr = w;
    rotate_rows(xr, n, seed);

    auto unrotate = [&](const std::vector<float> & y) {
        std::vector<float> out(nrows*n);
        parallel_for(nrows, n_threads(), [&](int, int64_t r0, int64_t r1) {
            std::vector<double> acc(n);
            for (int64_t r = r0; r < r1; ++r) {
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int64_t i = 0; i < n; ++i) {
                    const double yi = y[r*n + i];
                    const float * Ri = R.data() + i*n;
                    for (int64_t j = 0; j < n; ++j) {
                        acc[j] += Ri[j]*yi;
                    }
                }
                for (int64_t j = 0; j < n; ++j) {
                    out[r*n + j] = (float) acc[j];
                }
            }
        });
        return out;
    };

    const double w2 = mse(w, std::vector<float>(w.size(), 0.0f));
    const double exact = mse(w, unrotate(xr))/w2;
    char buf[160];
    snprintf(buf, sizeof(buf), "R^T R w == w without quantization (rel MSE %.1e)", exact);
    check(exact < 1e-10, buf);

    struct tol { ggml_type t; double max_rel; };
    const tol tols[] = {
        { GGML_TYPE_HQ4_K,   0.0063 }, { GGML_TYPE_HQ5_K,   0.0016 },
        { GGML_TYPE_HQ2_XXS, 0.1450 }, { GGML_TYPE_HQ2_XS,  0.1100 }, { GGML_TYPE_HQ2_S,   0.0760 },
        { GGML_TYPE_HQ3_XXS, 0.0410 }, { GGML_TYPE_HQ3_S,   0.0230 },
        { GGML_TYPE_HQ4_NL,  0.0071 }, { GGML_TYPE_HQ4_XS,  0.0072 },
        { GGML_TYPE_HQ4_0,   0.0082 }, { GGML_TYPE_HQ4_1,   0.0061 }, { GGML_TYPE_HQ5_0,   0.0020 },
        { GGML_TYPE_HQ5_1,   0.0014 }, { GGML_TYPE_HQ8_0,   0.00004 },
        { GGML_TYPE_HQ2_K,   0.0870 }, { GGML_TYPE_HQ3_K,   0.0260 }, { GGML_TYPE_HQ6_K,   0.00038 },
    };
    for (const tol & tl : tols) {
        const std::vector<uint8_t> q = quantize(tl.t, xr, n, nullptr, n_threads());
        const std::vector<float> yr = dequantize(tl.t, q, nrows, n);
        const double rel_rot = mse(xr, yr)/w2;
        const double rel     = mse(w, unrotate(yr))/w2;
        snprintf(buf, sizeof(buf), "%s: rel MSE %.5f <= %.4g (rotated domain %.5f)", ggml_type_name(tl.t), rel, tl.max_rel, rel_rot);
        check(rel <= tl.max_rel && std::fabs(rel - rel_rot) <= 1e-3*rel_rot, buf);
    }
}

// ---------------------------------------------------------------------------------------------

int main() {
    const auto t0 = std::chrono::steady_clock::now();

    test_cold_start();
    test_requires_imatrix();
    test_base_hashes();
    test_format_and_imatrix();
    test_determinism();
    test_copy_fidelity();
    printf("  [%.1f s]\n", seconds_since(t0));

    test_round_trip();
    printf("  [%.1f s]\n", seconds_since(t0));

    test_neighbour_vs_exhaustive();
    printf("  [%.1f s]\n", seconds_since(t0));

    test_mse();
    printf("  [%.1f s]\n", seconds_since(t0));

    printf("%s (%d failed)\n", g_failed ? "FAILED" : "ALL PASSED", g_failed);
    return g_failed ? 1 : 0;
}
