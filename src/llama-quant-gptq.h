#pragma once

// GPTQ error feedback for the Hadamard-rotated (HQ) types, with H = R*diag(v)*R^T from an imatrix v, or from a full
// input Gram G (hessian-collect)

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <string>
#include <thread>
#include <vector>

// runs fn(r) for r in [0, nrows) across the workers
template <typename F>
static void llama_parallel_rows(int64_t nrows, std::vector<std::thread> & workers, int nthread, F && fn) {
    std::atomic<int64_t> next { 0 };
    auto run = [&]() {
        for (int64_t r; (r = next++) < nrows; ) {
            fn(r);
        }
    };
    for (int t = 0; t < std::min<int64_t>(nthread, nrows) - 1; ++t) {
        workers.emplace_back(run);
    }
    run();
    for (auto & w : workers) {
        w.join();
    }
    workers.clear();
}

// the smallest damp at which the fp64 factorization is guaranteed to complete, for n up to about 30000
constexpr float LLAMA_GPTQ_DAMP_MIN     = 0.001f;
constexpr float LLAMA_GPTQ_DAMP_DEFAULT = 0.01f;
constexpr int   LLAMA_GPTQ_ROW_GROUP    = 32;

// bytes of one packed factor, and the peak bytes of building one
size_t llama_gptq_packed_bytes(int64_t n);
size_t llama_gptq_build_bytes(int64_t n);

// vbar = v/mean(v); false if v is all zero or has a negative entry
bool llama_gptq_normalize(const float * v, int64_t n, std::vector<float> & vbar);

// U = the upper Cholesky factor of H^-1 (H^-1 = U^T U), H = R*(diag(vbar) + damp*I)*R^T with R the RHT of seed,
// packed as ggml_quantize_rows_gptq reads it. The result doesn't depend on nthread.
// false, without factoring, for damp < LLAMA_GPTQ_DAMP_MIN, a v refused by llama_gptq_normalize, or a
// width without an RHT; also false if the factorization breaks down, which the damp floor rules out
// for n up to about 30000
bool llama_gptq_factor(const float * v, int64_t n, uint64_t seed, float damp, std::vector<float> & U,
                       std::vector<std::thread> & workers, int nthread);

// bytes of building a factor from a full Gram
size_t llama_gptq_build_bytes_full(int64_t n);

// U as llama_gptq_factor builds it, for H = R*((1 - alpha)*Gbar + alpha*diag(Gbar) + damp*I)*R^T with Gbar =
// G/mean(diag G): alpha = 1 is the diagonal H of llama_gptq_factor. read_gram fills the n x n Gram. false for damp <
// LLAMA_GPTQ_DAMP_MIN, alpha outside [0, 1], a width without an RHT, a Gram that can't be read or has a negative or
// all-zero diagonal, or a breakdown of the factorization
bool llama_gptq_factor_full(const std::function<bool(float *)> & read_gram, int64_t n, uint64_t seed, float alpha, float damp,
                            std::vector<float> & U, std::vector<std::thread> & workers, int nthread);

// the most recently used factors, keyed by (vbar, n, seed, damp), or for full Grams by (key, n, seed, alpha, damp);
// the cache and a factor being built
// together stay within cap bytes
struct llama_gptq_cache {
    // REFUSED: v or damp is refused by llama_gptq_factor; FAILED: the factorization broke down
    enum status { HIT, BUILT, REFUSED, TOO_BIG, FAILED };

    explicit llama_gptq_cache(size_t cap, size_t max_entries = 4) : cap(cap), max_entries(max_entries) {}

    // nullptr unless the status is HIT or BUILT
    const float * get(const float * v, int64_t n, uint64_t seed, float damp,
                      std::vector<std::thread> & workers, int nthread, status * st = nullptr);

    // the full-Gram factor of the input named key (its owner weight); read_gram is called only to build it
    const float * get_full(const std::string & key, const std::function<bool(float *)> & read_gram, int64_t n, uint64_t seed,
                           float alpha, float damp, std::vector<std::thread> & workers, int nthread, status * st = nullptr);

    size_t size()  const { return entries.size(); }
    size_t bytes() const;

private:
    struct entry {
        std::vector<float> vbar;
        std::string key; // empty for an imatrix factor
        int64_t  n;
        uint64_t seed;
        float    damp;
        float    alpha;
        std::vector<float> U;
    };

    size_t cap;
    size_t max_entries;
    std::list<entry> entries; // most recent first
};

// quantizes nrows rotated rows with the factor U, in groups of LLAMA_GPTQ_ROW_GROUP rows across the
// workers; the rows are modified. false if the output fails validation
bool llama_tensor_quantize_gptq(ggml_type type, float * rows, void * dst, int64_t nrows, int64_t n_per_row,
                                const float * U, std::vector<std::thread> & workers, int nthread);
