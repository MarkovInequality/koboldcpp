// MTP draft state across a partial-state restore, as SmartCache checkpoints do it: process a prompt, save the
// target's partial (recurrent) state and the speculative state, draft, process another batch, restore, draft
// again. The drafts (tokens and the last draft step's logits) must match bitwise, and so must the target's logits
// for the next token. Without restoring the speculative state the draft logits must differ: the MTP head's first
// input is the target's hidden row of the last processed token, which only the speculative state carries.
// Also checks the restore order: cutting the target back before loading the partial state fails once the cut is
// outside the rollback window, and loading first makes it succeed.
//
// usage: test-mtp-spec-state MODEL [--bench]   (a hybrid model with MTP layers)
//   --bench: also times saving and loading the partial state into a SmartCache checkpoint buffer, pageable and
//            registered with CUDA (pinned)

#include "llama.h"
#include "llama-model.h"
#include "common.h"
#include "speculative.h"
#include "kcpp_state_buffer.h"
#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * TEXT_A =
    "The lighthouse keeper climbed the spiral stairs every evening at dusk, carrying a can of oil and a rag. "
    "He had done it for thirty years, through storms that rattled the glass and calm nights when the sea lay "
    "flat as a mirror. The ships that passed rarely signalled back, but he kept the lamp burning anyway, because "
    "a light that is only lit when someone is watching is not much of a light at all. One winter a young woman "
    "arrived on the supply boat with a letter from the harbour authority. ";
static const char * TEXT_B =
    "She was to learn the work and, in time, take it over. He read the letter twice, folded it into his coat "
    "pocket, and showed her where the oil was kept.";

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & s, bool bos) {
    std::vector<llama_token> out(s.size() + 8);
    const int n = llama_tokenize(vocab, s.c_str(), (int) s.size(), out.data(), (int) out.size(), bos, false);
    out.resize(n > 0 ? n : 0);
    return out;
}

static bool decode(llama_context * ctx, common_speculative * spec, const std::vector<llama_token> & toks, llama_pos pos0) {
    llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], pos0 + (llama_pos) i, { 0 }, i + 1 == toks.size());
    }
    const bool ok = llama_decode(ctx, batch) == 0 && common_speculative_process(spec, batch);
    llama_batch_free(batch);
    return ok;
}

struct draft_out {
    llama_tokens tokens;
    std::vector<float> logits;
};

static draft_out draft(common_speculative * spec, llama_context * dft, int32_t n_vocab, llama_token id_last, llama_pos pos0) {
    draft_out d;
    llama_tokens prompt;
    auto & dp = common_speculative_get_draft_params(spec, 0);
    dp.drafting = true;
    dp.n_max    = 4;
    dp.pos0     = pos0;
    dp.id_last  = id_last;
    dp.prompt   = &prompt;
    dp.result   = &d.tokens;
    common_speculative_draft(spec);
    const float * lg = llama_get_logits_ith(dft, -1);
    d.logits.assign(lg, lg + n_vocab);
    llama_memory_seq_rm(llama_get_memory(dft), 0, pos0, -1);
    return d;
}

static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

// median milliseconds of saving and of loading the partial state through buf
static void bench_state(llama_context * ctx, kcpp_state_buffer & buf, const char * what) {
    const size_t n = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    std::vector<double> save, load;
    for (int it = 0; it < 23; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        const bool ok_s = llama_state_seq_get_data_ext(ctx, buf.data(), n, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == n;
        auto t1 = std::chrono::steady_clock::now();
        const bool ok_l = llama_state_seq_set_data_ext(ctx, buf.data(), n, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == n;
        auto t2 = std::chrono::steady_clock::now();
        if (!ok_s || !ok_l) {
            printf("  %s: state copy failed\n", what);
            return;
        }
        if (it >= 3) {
            save.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            load.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
    }
    std::sort(save.begin(), save.end());
    std::sort(load.begin(), load.end());
    printf("  %-26s save %6.2f ms (%5.1f GB/s), load %6.2f ms (%5.1f GB/s), medians of 20\n", what, save[save.size()/2],
           n/save[save.size()/2]/1e6, load[load.size()/2], n/load[load.size()/2]/1e6);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s MODEL\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mtp     = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model || model->hparams.n_layer_nextn == 0 || !llama_model_is_hybrid(model)) {
        printf("FAIL: cannot load %s as a hybrid model with MTP layers\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // as koboldcpp sets them up for MTP with 4 drafted tokens
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 4096;
    cp.n_batch  = 512;
    cp.n_ubatch = 512;
    cp.n_rs_seq = 4;
    llama_context * tgt = llama_init_from_model(model, cp);
    llama_context_params dcp = cp;
    dcp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
    dcp.ctx_other     = tgt;
    dcp.n_rs_seq      = 0;
    dcp.n_outputs_max = 1;
    llama_context * dft = llama_init_from_model(model, dcp);
    if (!tgt || !dft) {
        printf("FAIL: cannot create the contexts\n");
        return 1;
    }

    common_params_speculative sp;
    sp.types                  = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    sp.draft.ctx_tgt          = tgt;
    sp.draft.ctx_dft          = dft;
    sp.draft.n_max            = 4;
    sp.draft.n_min            = 0;
    sp.draft.p_min            = 0.0f;
    sp.draft.backend_sampling = false;
    common_speculative * spec = common_speculative_init(sp, 1);
    if (!spec) {
        printf("FAIL: cannot create the MTP speculative state\n");
        return 1;
    }

    const std::vector<llama_token> a = tokenize(vocab, TEXT_A, true);
    const std::vector<llama_token> b = tokenize(vocab, TEXT_B, false);
    const llama_pos n = (llama_pos) a.size();
    if (!decode(tgt, spec, a, 0)) {
        printf("FAIL: prompt decode\n");
        return 1;
    }
    const float * lg = llama_get_logits_ith(tgt, -1);
    llama_token t = 0;
    for (int32_t i = 1; i < n_vocab; ++i) {
        t = lg[i] > lg[t] ? i : t;
    }

    const size_t n_tgt = llama_state_seq_get_size_ext(tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    std::vector<uint8_t> state_tgt(n_tgt);
    std::vector<uint8_t> state_spec;
    const bool saved_tgt  = llama_state_seq_get_data_ext(tgt, state_tgt.data(), n_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == n_tgt;
    const bool saved_spec = common_speculative_get_state(spec, 0, state_spec);
    printf("prompt %d tokens, then %zu; partial target state %zu bytes, speculative state %zu bytes%s\n", n, b.size(),
           n_tgt, state_spec.size(), saved_spec ? "" : " (none saved)");

    const draft_out d1 = draft(spec, dft, n_vocab, t, n);
    if (!decode(tgt, spec, { t }, n)) {
        printf("FAIL: decode of the next token\n");
        return 1;
    }
    const float * l = llama_get_logits_ith(tgt, -1);
    const std::vector<float> ref(l, l + n_vocab);

    int fails = !saved_tgt;
    auto restore = [&](bool with_spec) {
        if (llama_state_seq_set_data_ext(tgt, state_tgt.data(), state_tgt.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != state_tgt.size()) {
            return false;
        }
        if (!llama_memory_seq_rm(llama_get_memory(tgt), 0, n, -1)) {
            return false;
        }
        llama_memory_seq_rm(llama_get_memory(dft), 0, n, -1);
        if (with_spec) {
            common_speculative_set_state(spec, 0, state_spec);
        }
        return true;
    };

    if (!decode(tgt, spec, b, n + 1)) {
        printf("FAIL: second batch decode\n");
        return 1;
    }
    const bool cut_first = llama_memory_seq_rm(llama_get_memory(tgt), 0, n, -1);
    printf("cut before loading the partial state %s  %s\n", cut_first ? "succeeded" : "failed", cut_first ? "FAIL" : "ok");
    fails += cut_first;
    if (!restore(true)) {
        printf("FAIL: restore\n");
        return 1;
    }
    const draft_out d2 = draft(spec, dft, n_vocab, t, n);
    const bool d_same = d1.tokens == d2.tokens && same(d1.logits, d2.logits);
    printf("draft after the restore: %zu tokens, %s  %s\n", d2.tokens.size(),
           d_same ? "tokens and logits bitwise equal" : "differs", d_same ? "ok" : "FAIL");
    fails += !d_same;
    if (!decode(tgt, spec, { t }, n)) {
        printf("FAIL: decode of the next token after the restore\n");
        return 1;
    }
    l = llama_get_logits_ith(tgt, -1);
    const bool t_same = same(ref, std::vector<float>(l, l + n_vocab));
    printf("target logits for the next token after the restore: %s  %s\n", t_same ? "bitwise equal" : "differ", t_same ? "ok" : "FAIL");
    fails += !t_same;

    if (!decode(tgt, spec, b, n + 1) || !restore(false)) {
        printf("FAIL: second restore\n");
        return 1;
    }
    const draft_out d3 = draft(spec, dft, n_vocab, t, n);
    const bool d_differs = !same(d1.logits, d3.logits);
    printf("draft after a restore without the speculative state: %s  %s\n", d_differs ? "logits differ" : "logits equal",
           d_differs ? "ok" : "FAIL (the check can't see a stale draft input)");
    fails += !d_differs;

    if (argc > 2 && std::string(argv[2]) == "--bench") {
        printf("checkpoint copies of the %zu MB partial state:\n", n_tgt >> 20);
        kcpp_state_buffer buf;
        buf.fit(n_tgt);
        bench_state(tgt, buf, "pageable (THP)");
#ifdef GGML_USE_CUDA
        // writable: a checkpoint save copies into it (ggml_backend_cuda_register_host_buffer registers read-only)
        auto t0 = std::chrono::steady_clock::now();
        const bool reg = cudaHostRegister(buf.data(), buf.capacity(), cudaHostRegisterPortable) == cudaSuccess;
        auto t1 = std::chrono::steady_clock::now();
        printf("  registering %zu MB: %s, %.1f ms\n", buf.capacity() >> 20, reg ? "ok" : "failed",
               std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (reg) {
            bench_state(tgt, buf, "registered with CUDA");
            cudaHostUnregister(buf.data());
        }
#endif
    }

    common_speculative_free(spec);
    llama_free(dft);
    llama_free(tgt);
    llama_model_free(model);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
