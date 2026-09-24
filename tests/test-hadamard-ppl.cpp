// Perplexity comparison for Hadamard-rotated (ConvRot) models.
//
// Phase 5b of plans/add_hadamard_rotated_quantization_plan.md calls for running the perplexity
// tool on a Q4_K and a Q4R_K build of the same model. This fork has no perplexity tool
// (tools/perplexity/main.cpp is a stub calling an undefined llama_perplexity), so this computes
// the same quantity directly: the mean negative log-likelihood over a sliding window of a text
// file, exp() of which is the perplexity.
//
// A rotated model with a missing or mismatched inference-side rotation does not crash - it
// produces plausible-looking garbage - so this number, not "does it generate text", is the real
// check that the quantizer and the engine agree.
//
// usage: test-hadamard-ppl [-ngl N] <text-file> <model.gguf> [more models...]
//
// -ngl is what makes this worth running twice: the CPU mul_mat takes the FWHT fast path
// unconditionally and never reads the materialized H_g matrix, so a CPU-only run proves the
// quantizer/engine math but exercises none of the backend dispatch and none of the H_g plumbing.

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

static const int N_CTX = 512;

static std::vector<llama_token> tokenize_file(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tokens(text.size() + 16);

    const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
        tokens.data(), (int32_t) tokens.size(), /*add_special =*/ true, /*parse_special =*/ false);
    if (n < 0) {
        fprintf(stderr, "tokenization failed (%d)\n", n);
        return {};
    }

    tokens.resize(n);
    return tokens;
}

// mean NLL over every predicted token, in non-overlapping N_CTX windows (the second half of each
// window only, so every scored token has at least N_CTX/2 tokens of context)
static double eval_nll(llama_context * ctx, const llama_vocab * vocab, const std::vector<llama_token> & tokens, int & n_scored) {
    const int n_vocab = llama_vocab_n_tokens(vocab);

    double sum_nll = 0.0;
    n_scored = 0;

    for (size_t start = 0; start + N_CTX <= tokens.size(); start += N_CTX) {
        std::vector<llama_token> window(tokens.begin() + start, tokens.begin() + start + N_CTX);

        llama_memory_clear(llama_get_memory(ctx), true);

        // logits are needed at every position, which llama_batch_get_one does not request
        llama_batch batch = llama_batch_init(N_CTX, 0, 1);
        batch.n_tokens = N_CTX;
        for (int i = 0; i < N_CTX; i++) {
            batch.token[i]     = window[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }

        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "llama_decode failed (%d)\n", rc);
            return -1.0;
        }

        for (int i = N_CTX/2; i < N_CTX - 1; i++) {
            const float * logits = llama_get_logits_ith(ctx, i);
            if (logits == nullptr) {
                fprintf(stderr, "no logits at %d - the context must be created with all logits enabled\n", i);
                return -1.0;
            }

            // log_softmax of the next token's logit
            float max_logit = logits[0];
            for (int j = 1; j < n_vocab; j++) {
                max_logit = std::max(max_logit, logits[j]);
            }

            double sum_exp = 0.0;
            for (int j = 0; j < n_vocab; j++) {
                sum_exp += std::exp((double) logits[j] - max_logit);
            }

            const double logp = (double) logits[window[i + 1]] - max_logit - std::log(sum_exp);

            sum_nll -= logp;
            n_scored++;
        }
    }

    return n_scored > 0 ? sum_nll / n_scored : -1.0;
}

int main(int argc, char ** argv) {
    int n_gpu_layers = 0;
    int arg          = 1;

    if (arg + 1 < argc && strcmp(argv[arg], "-ngl") == 0) {
        n_gpu_layers = atoi(argv[arg + 1]);
        arg += 2;
    }

    if (argc - arg < 2) {
        fprintf(stderr, "usage: %s [-ngl N] <text-file> <model.gguf> [more models...]\n", argv[0]);
        return 1;
    }

    std::string text;
    {
        FILE * f = fopen(argv[arg], "rb");
        if (!f) {
            fprintf(stderr, "cannot open %s\n", argv[arg]);
            return 1;
        }
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
            text.append(buf, n);
        }
        fclose(f);
    }

    llama_backend_init();

    printf("n_gpu_layers = %d\n\n", n_gpu_layers);
    printf("%-40s %12s %12s %10s\n", "model", "NLL", "perplexity", "tokens");

    int ret = 0;
    for (int m = arg + 1; m < argc; m++) {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = n_gpu_layers;

        llama_model * model = llama_model_load_from_file(argv[m], mparams);
        if (!model) {
            fprintf(stderr, "failed to load %s\n", argv[m]);
            ret = 1;
            continue;
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx        = N_CTX;
        cparams.n_batch      = N_CTX;
        cparams.n_ubatch     = N_CTX;
        cparams.n_outputs_max = N_CTX; // logits at every position

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "failed to create context for %s\n", argv[m]);
            llama_model_free(model);
            ret = 1;
            continue;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);

        const std::vector<llama_token> tokens = tokenize_file(vocab, text);

        int n_scored = 0;
        const double nll = eval_nll(ctx, vocab, tokens, n_scored);

        const char * name = strrchr(argv[m], '/');
        name = name ? name + 1 : argv[m];

        if (nll < 0.0) {
            printf("%-40s %12s %12s %10d\n", name, "FAILED", "-", n_scored);
            ret = 1;
        } else {
            printf("%-40s %12.4f %12.4f %10d\n", name, nll, std::exp(nll), n_scored);
        }
        fflush(stdout);

        llama_free(ctx);
        llama_model_free(model);
    }

    return ret;
}
