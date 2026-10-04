// upstream's tests/test-recurrent-state-rollback.cpp at 53ed051ce, plus the fork's cases (see "fork cases" below).
// The multi-seq split replay rolls back 3 tokens of a 4-token ubatch: a rollback reaches only the states inside the
// last ubatch, and upstream's 3-token tail asked for the state before it, which no snapshot holds.

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <vector>

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context * init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx);
    if (!decode_tokens(ctx, std::vector<llama_token>(n_tokens, 0), n_tokens)) {
        llama_free(ctx);
        return nullptr;
    }
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (collector.buffers.empty()) {
        fprintf(stderr, "%s : no cache buffers found\n", __func__);
        llama_free(ctx);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

static double nmse(const float * a, const float * b, int n) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double diff = (double) a[i] - b[i];
        mse_ab += diff*diff;
        mse_a0 += (double) a[i]*a[i];
    }
    return mse_a0 == 0.0 ? (mse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : mse_ab/mse_a0;
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    llama_context * ctx_roll = make_ctx_multi();
    llama_context * ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_roll) < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0 - 1; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref,  batch) == 0;

        common_batch_clear(batch);
        common_batch_add(batch, tok(s, p0 - 1), p0 - 1, { (llama_seq_id) s }, false);
        ok = ok && llama_decode(ctx_ref, batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0 - 1; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        cleanup();
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll, batch) == 0;
    ok = ok && llama_decode(ctx_ref, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        cleanup();
        return false;
    }

    // the rolled-back state came out of a 4-token ubatch, the reference's out of a 1-token one
    constexpr float nmse_eps = 1e-4f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    double   nmse_ab   = 0.0;
    double   nmse_a0   = 0.0;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            cleanup();
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            const float diff = logit_diff(r, f);
            if (diff > 0.0f && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_ab += d*d;
                nmse_a0 += (double) r*r;
            } else {
                nmse_ab = std::numeric_limits<double>::infinity();
                nmse_a0 = 1.0;
            }
        }
    }
    const double nmse_val = nmse_a0 == 0.0 ? (nmse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_ab/nmse_a0;

    if (nmse_val > nmse_eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, nmse %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, nmse_val, seq_first, pos_first);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff_max, nmse_val);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref, batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    double nmse_tail_ab = 0.0;
    double nmse_tail_a0 = 0.0;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll, batch_one) == 0;
        ok = ok && llama_decode(ctx_ref, batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            diff_tail = std::max(diff_tail, logit_diff(r, f));
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_tail_ab += d*d;
                nmse_tail_a0 += (double) r*r;
            } else {
                nmse_tail_ab = std::numeric_limits<double>::infinity();
                nmse_tail_a0 = 1.0;
            }
        }
    }
    const double nmse_tail = nmse_tail_a0 == 0.0 ? (nmse_tail_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_tail_ab/nmse_tail_a0;

    if (!ok || nmse_tail > nmse_eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g, nmse %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail, nmse_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g, nmse %g)\n", __func__, (double) diff_tail, nmse_tail);
    cleanup();
    return true;
}

static int test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // TODO: use smart pointers
    llama_context * ctx_src = make_ctx(params, model, fill);
    llama_context * ctx_dst = make_ctx(params, model, fill);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float nmse_eps = 0.0;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src, tokens[pos], pos) ||
                !decode_one(ctx_dst, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            const double nmse_val = nmse(logits_src, logits_dst, n_vocab);
            int token_first = -1;
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > 0.0f && token_first < 0) {
                    token_first = token;
                }
            }
            if (nmse_val > nmse_eps) {
                fprintf(stderr, "%s : %s logits mismatch at position %d, first token %d, nmse %g\n",
                        __func__, mode, pos, token_first, nmse_val);
                return false;
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
    //       this is not the case here. add asserts and guardrails to prevent such attempts
    //if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
    //    !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
    //    fprintf(stderr, "%s : partial rollback failed\n", __func__);
    //    return 1;
    //}

    //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    //common_prompt_checkpoint ckpt_partial;
    //ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    //ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    //if (!replay_and_compare("partial")) {
    //    return 1;
    //}

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model, fill);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        const double nmse_dirty = nmse(logits_src_replay[i].data(), logits_dirty, n_vocab);
        int token_first = -1;
        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > 0.0f && token_first < 0) {
                token_first = token;
            }
        }
        if (nmse_dirty > nmse_eps) {
            fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, first token %d, nmse %g\n",
                    __func__, pos, token_first, nmse_dirty);
            return 1;
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);

    if (!test_multi_seq_split_replay(params, model, n_vocab, fill)) {
        return 1;
    }

    return 0;
}

//
// fork cases: the MTP loop's shapes (a verify batch of 1 + n_draft tokens at n_rs_seq = 4, then a rollback), cell
// moves between sequences, fresh sequences sharing a ubatch, and state round trips. A context that rolls back is
// compared with one that never decoded the rejected tokens; where their ubatch shapes differ only rounding may
// differ (nmse <= FORK_EPS), where they match the logits must be identical.
//

static constexpr double FORK_EPS = 1e-4;

struct fork_seq_batch {
    llama_seq_id seq;
    llama_pos    p0;
    uint32_t     n;
};

static llama_token fork_tok(llama_seq_id seq, llama_pos pos, int n_vocab) {
    return (llama_token) ((13*(uint32_t) pos + 101*(uint32_t) seq + 5) % (uint32_t) n_vocab);
}

static llama_context_ptr fork_ctx(const common_params & params, llama_model * model, uint32_t n_rs_seq, uint32_t n_seq_max = 1) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seq_max;
    cparams.n_rs_seq   = n_rs_seq;
    cparams.n_ctx      = 256*n_seq_max;
    cparams.n_batch    = 64;
    cparams.n_ubatch   = 64;
    cparams.kv_unified = false;
    return llama_context_ptr(llama_init_from_model(model, cparams));
}

// decodes the parts in one batch, in the given order, with logits for every token; returns them part by part
static bool fork_decode(llama_context * ctx, const std::vector<fork_seq_batch> & parts, int n_vocab, std::vector<std::vector<float>> * out = nullptr) {
    uint32_t n = 0;
    for (const auto & p : parts) {
        n += p.n;
    }
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (const auto & p : parts) {
        for (uint32_t i = 0; i < p.n; ++i) {
            common_batch_add(batch, fork_tok(p.seq, p.p0 + i, n_vocab), p.p0 + i, { p.seq }, true);
        }
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (ok && out) {
        out->clear();
        for (uint32_t i = 0; i < n; ++i) {
            const float * l = llama_get_logits_ith(ctx, i);
            out->emplace_back(l, l + n_vocab);
        }
    }
    return ok;
}

static bool fork_compare(const char * name, const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b, bool exact) {
    double worst = 0;
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i) {
        worst = std::max(worst, nmse(a[i].data(), b[i].data(), (int) a[i].size()));
        same = same && memcmp(a[i].data(), b[i].data(), a[i].size()*sizeof(float)) == 0;
    }
    const bool ok = a.size() == b.size() && (exact ? same : worst <= FORK_EPS);
    fprintf(stderr, "  %-58s %s (%s, nmse %.3g)\n", name, ok ? "ok" : "FAIL", same ? "identical" : "differ", worst);
    return ok;
}

// a rollback that seq_rm accepts must give the state the sequence had at that point; one that reaches past the
// snapshots of the last ubatch must be refused (a 1-token ubatch keeps no snapshot of the state before it)
static bool fork_test_guard(const common_params & params, llama_model * model, int n_vocab) {
    bool ok = true;
    for (uint32_t n_rs_seq : { 0u, 4u }) {
        auto ctx = fork_ctx(params, model, n_rs_seq);
        auto ref = fork_ctx(params, model, n_rs_seq);
        std::vector<std::vector<float>> l_ctx, l_ref;
        bool good = fork_decode(ctx.get(), { { 0, 0, 16 } }, n_vocab) && fork_decode(ref.get(), { { 0, 0, 16 } }, n_vocab);
        for (llama_pos pos = 16; pos < 20; ++pos) {
            good = good && fork_decode(ctx.get(), { { 0, pos, 1 } }, n_vocab) && (pos == 19 || fork_decode(ref.get(), { { 0, pos, 1 } }, n_vocab));
        }
        const bool accepted = good && llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 19, -1);
        if (accepted) {
            good = good && fork_decode(ctx.get(), { { 0, 19, 1 } }, n_vocab, &l_ctx) && fork_decode(ref.get(), { { 0, 19, 1 } }, n_vocab, &l_ref);
        }
        char name[96];
        snprintf(name, sizeof(name), "n_rs_seq %u: rollback past a 1-token ubatch %s", n_rs_seq, accepted ? "accepted" : "refused");
        ok &= good && (accepted ? fork_compare(name, l_ctx, l_ref, false) : (fprintf(stderr, "  %-58s ok\n", name), true));
    }
    return ok;
}

// prefill, a verify batch of 5, rollback r, then the next verify batch: against a context that decoded only the
// accepted tokens. With n_rs_seq 0 the rollback must be refused.
static bool fork_test_verify(const common_params & params, llama_model * model, int n_vocab) {
    bool ok = true;
    for (uint32_t n_rs_seq : { 0u, 4u }) {
        for (uint32_t r = 0; r <= 4; ++r) {
            auto ctx = fork_ctx(params, model, n_rs_seq);
            auto ref = fork_ctx(params, model, n_rs_seq);
            std::vector<std::vector<float>> l_ctx, l_ref;
            bool good = fork_decode(ctx.get(), { { 0, 0, 16 } }, n_vocab) && fork_decode(ctx.get(), { { 0, 16, 5 } }, n_vocab) &&
                        fork_decode(ref.get(), { { 0, 0, 16 } }, n_vocab) && fork_decode(ref.get(), { { 0, 16, 5 - r } }, n_vocab);
            const llama_pos next = 21 - (llama_pos) r;
            const bool accepted = r == 0 || llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, next, -1);
            char name[96];
            if (n_rs_seq == 0) {
                if (r > 0) {
                    snprintf(name, sizeof(name), "n_rs_seq 0: rollback %u refused", r);
                    fprintf(stderr, "  %-58s %s\n", name, accepted ? "FAIL" : "ok");
                    ok &= good && !accepted;
                }
                continue;
            }
            good = good && accepted && fork_decode(ctx.get(), { { 0, next, 5 } }, n_vocab, &l_ctx) &&
                   fork_decode(ref.get(), { { 0, next, 5 } }, n_vocab, &l_ref);
            snprintf(name, sizeof(name), "n_rs_seq 4: verify 5, rollback %u, verify 5", r);
            ok &= good && fork_compare(name, l_ctx, l_ref, r == 0);
        }
    }
    return ok;
}

// seq 1 forks off seq 0 (seq_cp), a batch ordered [seq 1, seq 0] moves the cells, seq 0 rolls back 2 of its 4 new
// tokens; each sequence then continues alone, against single-sequence references
static bool fork_test_cell_swap(const common_params & params, llama_model * model, int n_vocab) {
    auto ctx  = fork_ctx(params, model, 4, 2);
    auto ref0 = fork_ctx(params, model, 4);
    auto ref1 = fork_ctx(params, model, 4);
    std::vector<std::vector<float>> l0, l1, r0, r1;
    bool good = fork_decode(ctx.get(), { { 0, 0, 16 } }, n_vocab);
    llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, -1, -1);
    good = good && fork_decode(ctx.get(), { { 1, 16, 4 }, { 0, 16, 4 } }, n_vocab);
    good = good && llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 18, -1);
    good = good && fork_decode(ctx.get(), { { 0, 18, 3 } }, n_vocab, &l0) && fork_decode(ctx.get(), { { 1, 20, 3 } }, n_vocab, &l1);

    // the references decode seq 1's tokens as seq 0, so the token ids must follow the context's sequence
    auto decode_as = [&](llama_context * c, llama_seq_id as, llama_pos p0, uint32_t n, std::vector<std::vector<float>> * out) {
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (uint32_t i = 0; i < n; ++i) {
            common_batch_add(batch, fork_tok(as, p0 + i, n_vocab), p0 + i, { 0 }, true);
        }
        bool res = llama_decode(c, batch) == 0;
        llama_batch_free(batch);
        if (res && out) {
            out->clear();
            for (uint32_t i = 0; i < n; ++i) {
                const float * l = llama_get_logits_ith(c, i);
                out->emplace_back(l, l + n_vocab);
            }
        }
        return res;
    };
    good = good && fork_decode(ref0.get(), { { 0, 0, 16 } }, n_vocab) && decode_as(ref0.get(), 0, 16, 2, nullptr) &&
           decode_as(ref0.get(), 0, 18, 3, &r0);
    good = good && fork_decode(ref1.get(), { { 0, 0, 16 } }, n_vocab) && decode_as(ref1.get(), 1, 16, 4, nullptr) &&
           decode_as(ref1.get(), 1, 20, 3, &r1);
    bool ok = good;
    ok &= fork_compare("2 seqs: seq_cp, batch [1, 0], seq 0 rolls back 2 (seq 0)", l0, r0, false);
    ok &= fork_compare("2 seqs: seq_cp, batch [1, 0], seq 0 rolls back 2 (seq 1)", l1, r1, false);
    return ok;
}

// two fresh sequences prefilled in one ubatch, then a verify batch of both, against each decoded alone
static bool fork_test_fresh_pair(const common_params & params, llama_model * model, int n_vocab) {
    auto ctx = fork_ctx(params, model, 4, 2);
    std::vector<std::vector<float>> l, r0, r1;
    bool good = fork_decode(ctx.get(), { { 0, 0, 16 }, { 1, 0, 16 } }, n_vocab) &&
                fork_decode(ctx.get(), { { 0, 16, 5 }, { 1, 16, 5 } }, n_vocab, &l);
    for (llama_seq_id s : { 0, 1 }) {
        auto ref = fork_ctx(params, model, 4, 2);
        good = good && fork_decode(ref.get(), { { s, 0, 16 } }, n_vocab) && fork_decode(ref.get(), { { s, 16, 5 } }, n_vocab, s ? &r1 : &r0);
    }
    std::vector<std::vector<float>> both(r0);
    both.insert(both.end(), r1.begin(), r1.end());
    return good && fork_compare("2 fresh seqs in one ubatch, then a verify of both", l, both, false);
}

// a sequence saved with a rollback pending and loaded into a fresh context continues identically
static bool fork_test_round_trip(const common_params & params, llama_model * model, int n_vocab) {
    auto src = fork_ctx(params, model, 4);
    auto dst = fork_ctx(params, model, 4);
    std::vector<std::vector<float>> l_src, l_dst;
    bool good = fork_decode(src.get(), { { 0, 0, 16 } }, n_vocab) && fork_decode(src.get(), { { 0, 16, 5 } }, n_vocab) &&
                llama_memory_seq_rm(llama_get_memory(src.get()), 0, 19, -1);
    std::vector<uint8_t> state(llama_state_seq_get_size(src.get(), 0));
    good = good && llama_state_seq_get_data(src.get(), state.data(), state.size(), 0) == state.size() &&
           llama_state_seq_set_data(dst.get(), state.data(), state.size(), 0) == state.size();
    good = good && fork_decode(src.get(), { { 0, 19, 5 } }, n_vocab, &l_src) && fork_decode(dst.get(), { { 0, 19, 5 } }, n_vocab, &l_dst);
    return good && fork_compare("save with a rollback pending, load, verify 5", l_src, l_dst, true);
}

static bool fork_tests(const common_params & params, llama_model * model) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    fprintf(stderr, "fork cases:\n");
    bool ok = true;
    ok &= fork_test_guard(params, model, n_vocab);
    ok &= fork_test_verify(params, model, n_vocab);
    ok &= fork_test_cell_swap(params, model, n_vocab);
    ok &= fork_test_fresh_pair(params, model, n_vocab);
    ok &= fork_test_round_trip(params, model, n_vocab);
    fprintf(stderr, "fork cases: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    for (uint8_t fill : { 0, 0x3e }) {
        fprintf(stderr, "%s : testing with cache fill 0x%02x\n", __func__, fill);
        if (test_rollback(params, model, fill) != 0) {
            return 1;
        }
    }

    return fork_tests(params, model) ? 0 : 1;
}
