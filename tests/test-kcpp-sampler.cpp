// SampleLogits' candidate selection and stop-sequence search (otherarch/kcpp_sampler_util.h) against frozen copies
// of the code they replaced in gpttype_adapter.cpp: the top 3000 candidates must be bitwise identical.
//
// usage: test-kcpp-sampler [ROWS] [--bench]
//   ROWS     rows recorded by koboldcpp with KCPP_SAMPLER_RECORD=ROWS (int32 n_vocab, then n_vocab floats, per row)
//   --bench  per-row time of the old and new selection, at the adapter's compile flags

#include "kcpp_sampler_util.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

// the code SampleLogits ran before kcpp_select_top_k, as it was
namespace ref {

static void sample_top_k(llama_token_data_array * cur_p, int32_t k) {
    if (k <= 0) {
        k = cur_p->size;
    }

    k = std::max(k, (int) 1); //min keep of 1
    k = std::min(k, (int) cur_p->size);

    // Sort scores in descending order
    if (!cur_p->sorted) {
        auto comp = [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        };
        if (k <= 128) {
            std::partial_sort(cur_p->data, cur_p->data + k, cur_p->data + cur_p->size, comp);
        } else {
            constexpr int   nbuckets     = 128;
            constexpr float bucket_low   = -10.0f;
            constexpr float bucket_high  =  10.0f;
            constexpr float bucket_scale = nbuckets/(bucket_high - bucket_low);
            constexpr float bucket_inter = -bucket_low * bucket_scale;

            static thread_local std::vector<int> bucket_idx;
            static thread_local std::vector<int> histo;
            static thread_local std::vector<llama_token_data> tmp_tokens;
            static thread_local std::vector<llama_token_data*> bucket_ptrs;

            bucket_idx.resize(cur_p->size);
            histo.assign(nbuckets, 0);

            for (int i = 0; i < (int)cur_p->size; ++i) {
                const float val = cur_p->data[i].logit;
                int ib = int(bucket_scale * val + bucket_inter); //nbuckets * (val - bucket_low) / (bucket_high - bucket_low);
                ib = std::max(0, std::min(nbuckets-1, ib));
                bucket_idx[i] = ib;
                ++histo[ib];
            }
            int nhave = 0;
            int ib = nbuckets - 1;
            for ( ; ib >= 0; --ib) {
                nhave += histo[ib];
                if (nhave >= k) {
                    break;
                }
            }
            tmp_tokens.resize(nhave);
            auto * ptr = tmp_tokens.data();
            bucket_ptrs.clear();
            bucket_ptrs.reserve(nbuckets - ib);
            for (int j = nbuckets - 1; j >= ib; --j) {
                bucket_ptrs.push_back(ptr);
                ptr += histo[j];
            }
            for (int i = 0; i < (int)cur_p->size; ++i) {
                int j = bucket_idx[i];
                if (j >= ib) {
                    *bucket_ptrs[nbuckets-1-j]++ = cur_p->data[i];
                }
            }

            ptr = tmp_tokens.data();
            int ndone = 0;
            for (int j = nbuckets-1; j > ib; --j) {
                std::sort(ptr, ptr + histo[j], comp);
                ptr += histo[j];
                ndone += histo[j];
            }
            std::partial_sort(ptr, ptr + k - ndone, ptr + histo[ib], comp);

            std::memcpy(cur_p->data, tmp_tokens.data(), k*sizeof(llama_token_data));

        }
        cur_p->sorted = true;
    }
    cur_p->size = k;
}

static bool stop_found(const std::string & text, const std::vector<std::string> & stops, const std::string ** matched) {
    for (const auto & s : stops) {
        if (text.find(s) != std::string::npos) {
            *matched = &s;
            return true;
        }
    }
    return false;
}

} // namespace ref

struct bias {
    int   token_id;
    float bias;
};

// what SampleLogits applies before the top 3000: logit biases in order, then DRY's per-token penalties
struct row_edits {
    std::vector<bias> biases;
    std::map<int, float> penalties;
};

static std::vector<llama_token_data> select_ref(const float * row, int n, const row_edits & e, int k) {
    static std::vector<llama_token_data> candidates;
    candidates.resize(n);
    for (llama_token token_id = 0; token_id < n; token_id++) {
        candidates[token_id] = llama_token_data{token_id, row[token_id], 0.0f};
    }
    for (const auto & b : e.biases) {
        candidates[b.token_id].logit += b.bias;
    }
    llama_token_data_array cp = { candidates.data(), candidates.size(), -1, false };
    for (const auto & [token, penalty] : e.penalties) {
        cp.data[token].logit -= penalty;
    }
    ref::sample_top_k(&cp, k);
    return std::vector<llama_token_data>(cp.data, cp.data + cp.size);
}

static std::vector<llama_token_data> select_new(const float * row, int n, const row_edits & e, int k) {
    static std::vector<float> scratch;
    static std::vector<llama_token_data> out;
    scratch.assign(row, row + n);
    for (const auto & b : e.biases) {
        scratch[b.token_id] += b.bias;
    }
    for (const auto & [token, penalty] : e.penalties) {
        scratch[token] -= penalty;
    }
    kcpp_select_top_k(scratch.data(), n, k, out);
    return out;
}

static std::vector<std::vector<float>> read_rows(const char * path) {
    std::vector<std::vector<float>> rows;
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return rows;
    }
    int32_t n = 0;
    while (fread(&n, sizeof(n), 1, f) == 1 && n > 0) {
        std::vector<float> r(n);
        if (fread(r.data(), sizeof(float), n, f) != (size_t) n) {
            break;
        }
        rows.push_back(std::move(r));
    }
    fclose(f);
    return rows;
}

// rows that exercise the edges: ties, many logits above the top bucket, infinities and NaNs, small vocabularies
static std::vector<std::vector<float>> synthetic_rows(uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::vector<std::vector<float>> rows;
    auto make = [&](int n, auto f) {
        std::vector<float> r(n);
        for (int i = 0; i < n; ++i) {
            r[i] = f(i);
        }
        rows.push_back(std::move(r));
    };
    const int n = 248320;
    make(n, [&](int) { return nd(rng); });
    make(n, [&](int) { return std::round(nd(rng)*2.0f)/2.0f; });            // ties
    make(n, [&](int) { return 12.0f + std::fabs(nd(rng))*0.1f; });           // everything in the top bucket
    make(n, [&](int i) { return i % 7 == 0 ? 9.99f : nd(rng) - 20.0f; });    // a big tied group at the boundary
    make(n, [&](int i) {
        const float v = nd(rng);
        return i % 1009 == 0 ? std::numeric_limits<float>::infinity() : i % 1013 == 0 ? -std::numeric_limits<float>::infinity() :
               i % 1019 == 0 ? std::numeric_limits<float>::quiet_NaN() : i % 1021 == 0 ? 3e38f : v;
    });
    make(100,  [&](int) { return nd(rng); });  // k <= 128 path
    make(2000, [&](int) { return nd(rng); });  // k = n
    make(n, [&](int) { return -1e6f + nd(rng); });                           // everything below the bottom bucket
    return rows;
}

static row_edits make_edits(int n, uint32_t seed, int variant) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tok(0, n - 1);
    std::uniform_real_distribution<float> ud(-5.0f, 5.0f);
    row_edits e;
    if (variant & 1) {
        for (int i = 0; i < 16; ++i) {
            e.biases.push_back({ tok(rng), ud(rng) });
        }
        e.biases.push_back({ tok(rng), -1e9f }); // a custom token ban
        e.biases.push_back({ e.biases[0].token_id, 0.25f }); // a repeated id adds up in order
    }
    if (variant & 2) {
        for (int i = 0; i < 24; ++i) {
            e.penalties[tok(rng)] = std::fabs(ud(rng))*std::pow(1.75f, (float) (i % 6));
        }
    }
    return e;
}

static int test_stop_scanner() {
    int fails = 0;
    std::mt19937 rng(7);
    const std::vector<std::string> pieces = { "a", "b", "<", "/", "s", ">", "\n", "\xe2\x96\x81", "\xe4\xbd\xa0", "ab", "</s", "s>", "</", ">\n" };
    const std::vector<std::vector<std::string>> stop_sets = {
        { "</s>" }, { "\n\n", "</s>" }, { "aba", "bab" }, { "\xe4\xbd\xa0\xe4\xbd\xa0" }, { "ab", "a" }, { "zzz" },
    };
    for (int trial = 0; trial < 2000; ++trial) {
        const auto & stops = stop_sets[trial % stop_sets.size()];
        std::string text;
        kcpp_stop_scanner sc;
        for (int step = 0; step < 64; ++step) {
            const int n_append = rng() % 3; // 0, 1 or 2 pieces between checks
            for (int a = 0; a < n_append; ++a) {
                text += pieces[rng() % pieces.size()];
            }
            const std::string * m_ref = nullptr;
            const bool f_ref = ref::stop_found(text, stops, &m_ref);
            const std::string * m_new = sc.find(text, stops);
            if (f_ref != (m_new != nullptr) || (f_ref && *m_ref != *m_new)) {
                if (fails++ < 5) {
                    printf("  stop scan mismatch at trial %d step %d\n", trial, step);
                }
            }
            if (f_ref) {
                break; // generation stops at the first match
            }
        }
    }
    printf("  %-60s %s\n", "stop sequences: tail scan vs full find (2000 random texts)", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}

int main(int argc, char ** argv) {
    const char * rows_path = nullptr;
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bench") {
            bench = true;
        } else {
            rows_path = argv[i];
        }
    }

    std::vector<std::vector<float>> rows = synthetic_rows(1);
    const size_t n_synthetic = rows.size();
    if (rows_path) {
        auto rec = read_rows(rows_path);
        printf("%zu recorded rows from %s\n", rec.size(), rows_path);
        rows.insert(rows.end(), rec.begin(), rec.end());
    }

    int fails = 0, n_cases = 0;
    for (size_t r = 0; r < rows.size(); ++r) {
        const int n = (int) rows[r].size();
        for (int variant = 0; variant < 4; ++variant) {
            for (uint32_t seed : { 1u, 2u, 3u }) {
                if (variant == 0 && seed > 1) {
                    continue;
                }
                const row_edits e = make_edits(n, seed*7919 + (uint32_t) r, variant);
                for (int k : { 3000, 40, 1 }) {
                    const auto a = select_ref(rows[r].data(), n, e, k);
                    const auto b = select_new(rows[r].data(), n, e, k);
                    n_cases++;
                    if (a.size() != b.size() || memcmp(a.data(), b.data(), a.size()*sizeof(llama_token_data)) != 0) {
                        if (fails++ < 10) {
                            printf("  mismatch: row %zu (%s), n %d, edits %d, seed %u, k %d\n", r, r < n_synthetic ? "synthetic" : "recorded", n, variant, seed, k);
                        }
                    }
                }
            }
        }
    }
    printf("  %-60s %s\n", ("top-k selection: " + std::to_string(n_cases) + " cases bitwise").c_str(), fails ? "FAIL" : "ok");

    fails += test_stop_scanner();

    if (bench) {
        // the real rows if there are any, else the synthetic ones of full width
        std::vector<const std::vector<float> *> brows;
        for (size_t r = rows_path ? n_synthetic : 0; r < rows.size(); ++r) {
            if (rows[r].size() > 100000) {
                brows.push_back(&rows[r]);
            }
        }
        const row_edits e = make_edits((int) brows[0]->size(), 5, 3);
        auto time_us = [&](auto && f) {
            std::vector<double> t;
            for (int rep = 0; rep < 5; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                for (const auto * row : brows) {
                    f(row->data(), (int) row->size(), e, 3000);
                }
                t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()/brows.size());
            }
            std::sort(t.begin(), t.end());
            return t[t.size()/2];
        };
        // rows whose boundary bucket holds far more than the 3000 kept, where the partial sort dominates
        int n_wide = 0;
        for (const auto * row : brows) {
            int histo[128] = {};
            for (float v : *row) {
                histo[std::max(0, std::min(127, int(6.4f*v + 64.0f)))]++;
            }
            int nhave = 0, ib = 127;
            for (; ib >= 0; --ib) {
                if ((nhave += histo[ib]) >= 3000) {
                    break;
                }
            }
            n_wide += histo[ib] > 6000;
        }
        const double t_ref = time_us(select_ref);
        const double t_new = time_us(select_new);
        printf("bench over %zu rows: old fill+bias+DRY+top-k %.1f us/row, new %.1f us/row; %d rows with a boundary bucket > 6000\n",
               brows.size(), t_ref, t_new, n_wide);
    }

    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
