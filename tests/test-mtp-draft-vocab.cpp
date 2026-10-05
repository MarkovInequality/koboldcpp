// The reduced MTP draft head (llama_set_draft_n_vocab): an MTP context with a draft vocabulary gives logits whose
// kept rows ([0, N) and the control/end-of-generation tail) are the full head's and whose other rows are -inf, for
// a 5-row batch and for single-token draft steps, against an MTP context with the full head. Bitwise, except the
// tail in a multi-row batch: its own small matmul may take another block shape than the full head's (the Blackwell
// MMVQ table widens launches with few warps in flight), so there within 1e-4 relative.
//
// usage: test-mtp-draft-vocab MODEL [N]   (a model with MTP layers; N defaults to 65536)

#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static llama_context * make_mtp_ctx(llama_model * model, llama_context * main_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx         = 512;
    cp.n_batch       = 64;
    cp.n_ubatch      = 64;
    cp.n_outputs_max = 8;
    cp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
    cp.ctx_other     = main_ctx;
    return llama_init_from_model(model, cp);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s MODEL [N]\n", argv[0]);
        return 1;
    }
    const int32_t n_draft = argc > 2 ? atoi(argv[2]) : 65536;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mtp     = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model || model->hparams.n_layer_nextn == 0) {
        printf("FAIL: cannot load %s with MTP layers\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    int32_t tail = n_draft;
    while (tail < n_vocab && !llama_vocab_is_control(vocab, tail) && !llama_vocab_is_eog(vocab, tail)) {
        tail++;
    }
    const bool rotated = model->output && model->is_rotated(model->output);
    printf("n_vocab %d, draft head rows [0, %d) and [%d, %d); output.weight %s\n", n_vocab, n_draft, tail, n_vocab,
           rotated ? "HQ-rotated" : "not rotated");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    llama_context * main_ctx = llama_init_from_model(model, cp);
    llama_context * full = make_mtp_ctx(model, main_ctx);
    llama_context * red  = make_mtp_ctx(model, main_ctx);
    if (!main_ctx || !full || !red) {
        printf("FAIL: cannot create the contexts\n");
        return 1;
    }
    llama_set_draft_n_vocab(red, n_draft);

    const int32_t n_embd = llama_model_n_embd_out(model);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> tok(0, n_draft - 1);

    int fails = 0;
    llama_pos pos = 0;
    // a 5-row batch like a catch-up with outputs, then single-token draft steps
    for (int n : { 5, 1, 1, 1 }) {
        llama_batch batch = llama_batch_init(n, n_embd, 1);
        batch.token = (llama_token *) malloc(sizeof(llama_token)*n); // MTP takes the token and the hidden state
        for (int i = 0; i < n; ++i) {
            batch.token[i]     = tok(rng);
            batch.pos[i]       = pos + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
            for (int j = 0; j < n_embd; ++j) {
                batch.embd[(size_t) i*n_embd + j] = nd(rng);
            }
        }
        batch.n_tokens = n;
        if (llama_decode(full, batch) != 0 || llama_decode(red, batch) != 0) {
            printf("FAIL: decode failed\n");
            return 1;
        }
        for (int i = 0; i < n; ++i) {
            const float * a = llama_get_logits_ith(full, i);
            const float * b = llama_get_logits_ith(red, i);
            int n_diff = 0, n_tail_diff = 0, n_tail_far = 0, n_not_inf = 0;
            for (int32_t t = 0; t < n_vocab; ++t) {
                if (t < n_draft) {
                    n_diff += memcmp(&a[t], &b[t], sizeof(float)) != 0;
                } else if (t >= tail) {
                    n_tail_diff += memcmp(&a[t], &b[t], sizeof(float)) != 0;
                    n_tail_far  += !(std::fabs(a[t] - b[t]) <= 1e-4f*(1.0f + std::fabs(a[t])));
                } else {
                    n_not_inf += !(std::isinf(b[t]) && b[t] < 0);
                }
            }
            const bool ok = n_diff == 0 && n_not_inf == 0 && (n == 1 ? n_tail_diff == 0 : n_tail_far == 0);
            printf("  batch of %d, row %d: head rows %s, tail rows %s, dropped rows %s  %s\n", n, i,
                   n_diff ? (std::to_string(n_diff) + " differ").c_str() : "bitwise equal",
                   n_tail_diff ? (std::to_string(n_tail_diff) + " differ, " + std::to_string(n_tail_far) + " beyond 1e-4").c_str() : "bitwise equal",
                   n_not_inf ? (std::to_string(n_not_inf) + " not -inf").c_str() : "-inf", ok ? "ok" : "FAIL");
            fails += !ok;
        }
        pos += n;
        free(batch.token);
        batch.token = nullptr;
        llama_batch_free(batch);
    }

    llama_free(red);
    llama_free(full);
    llama_free(main_ctx);
    llama_model_free(model);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
