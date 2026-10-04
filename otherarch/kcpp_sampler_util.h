#pragma once

// Sampler helpers of gpttype_adapter.cpp, shared with tests/test-kcpp-sampler.cpp.

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

// sample_top_k's candidates straight from a logits row: the same 128 buckets over [-10, 10], the same sorts on the
// same inputs in the same order, so the result is bitwise what sample_top_k leaves after filling a
// llama_token_data per vocabulary entry, which this skips

static inline void kcpp_select_top_k(const float * row, int n, int k, std::vector<llama_token_data> & out) {
    if (k <= 0) {
        k = n;
    }
    k = std::max(k, 1);
    k = std::min(k, n);

    auto comp = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    };

    if (k <= 128) {
        static thread_local std::vector<llama_token_data> all;
        all.resize(n);
        for (int i = 0; i < n; ++i) {
            all[i] = llama_token_data{i, row[i], 0.0f};
        }
        std::partial_sort(all.data(), all.data() + k, all.data() + n, comp);
        out.assign(all.begin(), all.begin() + k);
        return;
    }

    constexpr int   nbuckets     = 128;
    constexpr float bucket_low   = -10.0f;
    constexpr float bucket_high  =  10.0f;
    constexpr float bucket_scale = nbuckets/(bucket_high - bucket_low);
    constexpr float bucket_inter = -bucket_low * bucket_scale;

    static thread_local std::vector<uint8_t> bucket_idx;
    static thread_local std::vector<llama_token_data> tmp_tokens;
    static thread_local std::vector<llama_token_data*> bucket_ptrs;

    bucket_idx.resize(n);

    // four histograms, so consecutive increments of one bucket don't wait on each other
    int histo4[4][nbuckets] = {};
    int i = 0;
#if defined(__SSE2__)
    // cvttps2dq gives INT_MIN for NaN and out-of-range values exactly like the scalar int(), and mul then add rounds
    // the same, so these are the scalar loop's indices
    {
        const __m128 vscale = _mm_set1_ps(bucket_scale);
        const __m128 vinter = _mm_set1_ps(bucket_inter);
        const __m128i vzero = _mm_setzero_si128();
        const __m128i vmax  = _mm_set1_epi32(nbuckets - 1);
        alignas(16) int32_t b[4];
        for (; i + 4 <= n; i += 4) {
            __m128i v = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vscale, _mm_loadu_ps(row + i)), vinter));
            v = _mm_and_si128(v, _mm_cmpgt_epi32(v, vzero));
            const __m128i over = _mm_cmpgt_epi32(v, vmax);
            v = _mm_or_si128(_mm_andnot_si128(over, v), _mm_and_si128(over, vmax));
            _mm_store_si128((__m128i *) b, v);
            for (int j = 0; j < 4; ++j) {
                bucket_idx[i + j] = (uint8_t) b[j];
                ++histo4[j][b[j]];
            }
        }
    }
#else
    for (; i + 4 <= n; i += 4) {
        for (int j = 0; j < 4; ++j) {
            int ib = int(bucket_scale * row[i + j] + bucket_inter);
            ib = std::max(0, std::min(nbuckets-1, ib));
            bucket_idx[i + j] = (uint8_t) ib;
            ++histo4[j][ib];
        }
    }
#endif
    for (; i < n; ++i) {
        int ib = int(bucket_scale * row[i] + bucket_inter);
        ib = std::max(0, std::min(nbuckets-1, ib));
        bucket_idx[i] = (uint8_t) ib;
        ++histo4[0][ib];
    }
    int histo[nbuckets];
    for (int b = 0; b < nbuckets; ++b) {
        histo[b] = histo4[0][b] + histo4[1][b] + histo4[2][b] + histo4[3][b];
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
    int t = 0;
#if defined(__SSE2__)
    // few entries reach the kept buckets: skip 16 at a time, keeping id order within each bucket
    {
        const __m128i vlow = _mm_set1_epi8((char) (ib - 1)); // indices are 0..127, so signed bytes compare right
        for (; t + 16 <= n; t += 16) {
            int mask = _mm_movemask_epi8(_mm_cmpgt_epi8(_mm_loadu_si128((const __m128i *) (bucket_idx.data() + t)), vlow));
            while (mask) {
                const int l = __builtin_ctz(mask);
                mask &= mask - 1;
                const int j = bucket_idx[t + l];
                *bucket_ptrs[nbuckets-1-j]++ = llama_token_data{t + l, row[t + l], 0.0f};
            }
        }
    }
#endif
    for (; t < n; ++t) {
        const int j = bucket_idx[t];
        if (j >= ib) {
            *bucket_ptrs[nbuckets-1-j]++ = llama_token_data{t, row[t], 0.0f};
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

    out.assign(tmp_tokens.begin(), tmp_tokens.begin() + k);
}

// finds the first of stops (in order) that occurs in an append-only text, looking only at what an occurrence
// ending in the text appended since the last call could cover; reset() for a new text
struct kcpp_stop_scanner {
    size_t scanned = 0;

    void reset() {
        scanned = 0;
    }

    const std::string * find(const std::string & text, const std::vector<std::string> & stops) {
        const std::string * res = nullptr;
        for (const auto & s : stops) {
            const size_t from = s.empty() ? 0 : (scanned >= s.size() - 1 ? scanned - (s.size() - 1) : 0);
            if (text.find(s, from) != std::string::npos) {
                res = &s;
                break;
            }
        }
        scanned = text.size();
        return res;
    }
};
