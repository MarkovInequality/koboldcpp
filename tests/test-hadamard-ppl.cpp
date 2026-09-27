// Quality of quantized models against a reference: perplexity, KL divergence and top-token
// agreement over non-overlapping windows of a text file, plus bits per weight.
//
// usage: test-hadamard-ppl [-ngl N] [-c N_CTX] [--chunks N] [--cache FILE] [--lora FILE] <text-file> <reference.gguf> [model.gguf...]
//
// The reference's log-probabilities are written to the cache file (FP16) once and streamed back for
// every model; a cache with more windows than asked for is reused. Tokens in the second half of each
// window are scored, so each has at least N_CTX/2 tokens of context. --lora applies a runtime adapter
// to the reference and to every model.
// The fork has no perplexity tool; this is its replacement for these runs.

#include "kl-eval.h"

#include <cstdlib>
#include <cstring>

int main(int argc, char ** argv) {
    int n_gpu_layers = 0;
    int n_ctx        = 512;
    int n_chunks     = 0;
    std::string cache_path;
    std::string lora;

    int arg = 1;
    for (; arg < argc && argv[arg][0] == '-'; arg += 2) {
        if (arg + 1 >= argc) {
            break;
        }
        if (!strcmp(argv[arg], "-ngl")) {
            n_gpu_layers = atoi(argv[arg + 1]);
        } else if (!strcmp(argv[arg], "-c")) {
            n_ctx = atoi(argv[arg + 1]);
        } else if (!strcmp(argv[arg], "--chunks")) {
            n_chunks = atoi(argv[arg + 1]);
        } else if (!strcmp(argv[arg], "--cache")) {
            cache_path = argv[arg + 1];
        } else if (!strcmp(argv[arg], "--lora")) {
            lora = argv[arg + 1];
        } else {
            break;
        }
    }
    if (argc - arg < 2) {
        fprintf(stderr, "usage: %s [-ngl N] [-c N_CTX] [--chunks N] [--cache FILE] [--lora FILE] <text-file> <reference.gguf> [model.gguf...]\n", argv[0]);
        return 1;
    }

    std::string text;
    if (!read_text_file(argv[arg], text)) {
        return 1;
    }
    const std::string ref_path = argv[arg + 1];
    if (cache_path.empty()) {
        cache_path = ref_path + (lora.empty() ? "" : ".lora") + ".kl-cache";
    }

    llama_backend_init();
    const int nth = std::max(1u, std::thread::hardware_concurrency());

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;

    // the reference pass
    llama_model * ref = llama_model_load_from_file(ref_path.c_str(), mp);
    if (!ref) {
        fprintf(stderr, "failed to load %s\n", ref_path.c_str());
        return 1;
    }
    const llama_vocab * vocab  = llama_model_get_vocab(ref);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    const std::vector<llama_token> tokens = tokenize_file(vocab, text);

    const int n_avail = (int) (tokens.size() / n_ctx);
    n_chunks = n_chunks > 0 ? std::min(n_chunks, n_avail) : n_avail;

    printf("%d chunks of %d tokens, %d scored per chunk, n_gpu_layers = %d\n\n", n_chunks, n_ctx, scored_per_chunk(n_ctx), n_gpu_layers);
    printf("%-44s %8s %9s %9s %10s %10s %8s\n", "model", "bpw", "NLL", "PPL", "mean KL", "p99 KL", "top-1 %");

    FILE * cache = cache_open(cache_path, n_vocab, n_ctx, n_chunks, tokens);
    if (!cache) {
        llama_context * ctx = make_ctx(ref, n_ctx, lora);
        const bool ok = cache_write(ctx, cache_path, n_vocab, n_ctx, n_chunks, tokens, nth);
        llama_free(ctx);
        cache = ok ? cache_open(cache_path, n_vocab, n_ctx, n_chunks, tokens) : nullptr;
        if (!cache) {
            return 1;
        }
    }

    std::vector<std::string> models;
    models.push_back(ref_path);
    for (int m = arg + 2; m < argc; ++m) {
        models.push_back(argv[m]);
    }

    int ret = 0;
    for (size_t m = 0; m < models.size(); ++m) {
        llama_model * model = m == 0 ? ref : llama_model_load_from_file(models[m].c_str(), mp);
        if (!model) {
            fprintf(stderr, "failed to load %s\n", models[m].c_str());
            ret = 1;
            continue;
        }
        llama_context * ctx = make_ctx(model, n_ctx, lora);

        window_stats tot;
        const bool ok = score(ctx, cache, tokens, n_ctx, n_chunks, n_vocab, nth, tot);

        const char * name = strrchr(models[m].c_str(), '/');
        name = name ? name + 1 : models[m].c_str();
        const double bpw = 8.0*llama_model_size(model)/llama_model_n_params(model);

        if (!ok) {
            printf("%-44s %8.3f %9s\n", name, bpw, "FAILED");
            ret = 1;
        } else {
            const double nll = tot.nll/tot.n;
            printf("%-44s %8.3f %9.5f %9.4f %10.6f %10.6f %8.3f\n", name, bpw, nll, std::exp(nll), tot.mean_kl(),
                   tot.p99_kl(), tot.top1_pct());
        }
        fflush(stdout);

        llama_free(ctx);
        if (m != 0) {
            llama_model_free(model);
        }
    }

    fclose(cache);
    llama_model_free(ref);
    return ret;
}
