// Quality of quantized models against a reference: perplexity, KL divergence and top-token
// agreement over non-overlapping windows of a text file, plus bits per weight.
//
// usage: test-hadamard-ppl [-ngl N] [--ref-ngl N] [-c N_CTX] [--chunks N] [--parallel N] [--cache FILE] [--lora FILE]
//                          [-ctk TYPE] [-ub N] [--special] [--chat-mask] <text-file> <reference.gguf> [model.gguf...]
//
// The reference's log-probabilities are written to the cache file (FP16) once and streamed back for
// every model; a cache with more windows than asked for is reused, and then the reference is only read for
// its vocabulary. Its own row comes from the cache. Tokens in the second half of each window are scored, so
// each has at least N_CTX/2 tokens of context. --ref-ngl (default: -ngl) places the reference while its
// log-probs are computed; --parallel N decodes N windows per batch, so that weights in RAM cross to the GPU
// once per batch; --special parses special tokens in the text; --chat-mask (with --special) scores only the
// tokens a chat model generates, its own turns and text outside any turn; --lora applies a runtime adapter to
// the reference and to every model; -ctk sets the K and V cache type of every context (default f16); -ub N decodes
// in ubatches of N tokens (with N <= 8 the matrix-vector kernels run, as in decode and speculative verify).
// The fork has no perplexity tool; this is its replacement for these runs.

#include "kl-eval.h"
#include "gguf.h"

#include <cstdlib>
#include <cstring>
#include <unistd.h>

int main(int argc, char ** argv) {
    int n_gpu_layers = 0;
    int ref_ngl      = -1;
    int n_ctx        = 512;
    int n_chunks     = 0;
    int n_par        = 1;
    bool special     = false;
    bool chat        = false;
    std::string cache_path;
    std::string lora;

    int arg = 1;
    for (; arg < argc && argv[arg][0] == '-'; arg += 2) {
        if (!strcmp(argv[arg], "--special") || !strcmp(argv[arg], "--chat-mask")) {
            (argv[arg][2] == 's' ? special : chat) = true;
            arg--;
            continue;
        }
        if (arg + 1 >= argc) {
            break;
        }
        if (!strcmp(argv[arg], "-ngl")) {
            n_gpu_layers = atoi(argv[arg + 1]);
        } else if (!strcmp(argv[arg], "--ref-ngl")) {
            ref_ngl = atoi(argv[arg + 1]);
        } else if (!strcmp(argv[arg], "-c")) {
            n_ctx = atoi(argv[arg + 1]);
        } else if (!strcmp(argv[arg], "--chunks")) {
            n_chunks = atoi(argv[arg + 1]);
        } else if (!strcmp(argv[arg], "--parallel")) {
            n_par = std::max(1, atoi(argv[arg + 1]));
        } else if (!strcmp(argv[arg], "--cache")) {
            cache_path = argv[arg + 1];
        } else if (!strcmp(argv[arg], "--lora")) {
            lora = argv[arg + 1];
        } else if (!strcmp(argv[arg], "-ub")) {
            kl_eval_n_ubatch = atoi(argv[arg + 1]);
        } else if (!strcmp(argv[arg], "-ctk")) {
            bool found = false;
            for (int t = 0; t < GGML_TYPE_COUNT && !found; ++t) {
                if (ggml_type_name((ggml_type) t) && !strcmp(ggml_type_name((ggml_type) t), argv[arg + 1])) {
                    kl_eval_kv_type = (ggml_type) t;
                    found = true;
                }
            }
            if (!found) {
                fprintf(stderr, "unknown cache type %s\n", argv[arg + 1]);
                return 1;
            }
        } else {
            break;
        }
    }
    if (argc - arg < 2) {
        fprintf(stderr, "usage: %s [-ngl N] [--ref-ngl N] [-c N_CTX] [--chunks N] [--parallel N] [--cache FILE] [--lora FILE] "
                        "[-ctk TYPE] [-ub N] [--special] [--chat-mask] <text-file> <reference.gguf> [model.gguf...]\n", argv[0]);
        return 1;
    }
    ref_ngl = ref_ngl < 0 ? n_gpu_layers : ref_ngl;

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

    std::vector<llama_token> tokens;
    std::vector<uint8_t> mask;
    int n_vocab;
    {
        llama_model_params mpv = llama_model_default_params();
        mpv.vocab_only = true;
        llama_model * ref = llama_model_load_from_file(ref_path.c_str(), mpv);
        if (!ref) {
            fprintf(stderr, "failed to load %s\n", ref_path.c_str());
            return 1;
        }
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(ref));
        tokens  = common_tokenize(llama_model_get_vocab(ref), text, true, special);
        if (chat) {
            mask = chat_mask(llama_model_get_vocab(ref), tokens);
        }
        llama_model_free(ref);
    }

    const int n_avail = (int) (tokens.size() / n_ctx);
    n_chunks = n_chunks > 0 ? std::min(n_chunks, n_avail) : n_avail;
    n_par    = std::max(1, std::min(n_par, n_chunks));

    FILE * cache = cache_open(cache_path, n_vocab, n_ctx, n_chunks, tokens);
    if (!cache) {
        fprintf(stderr, "computing the reference's log-probs into %s\n", cache_path.c_str());
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = ref_ngl;
        llama_model * ref = llama_model_load_from_file(ref_path.c_str(), mp);
        llama_context * ctx = ref ? make_ctx(ref, n_ctx, lora, n_par) : nullptr;
        const bool ok = ctx && cache_write(ctx, cache_path, n_vocab, n_ctx, n_chunks, tokens, nth, n_par);
        llama_free(ctx);
        llama_model_free(ref);
        cache = ok ? cache_open(cache_path, n_vocab, n_ctx, n_chunks, tokens) : nullptr;
        if (!cache) {
            fprintf(stderr, "failed to compute the reference's log-probs\n");
            return 1;
        }
    }

    const int n_per_chunk = scored_per_chunk(n_ctx);
    int n_scored = 0;
    for (int c = 0; c < n_chunks; ++c) {
        for (int i = 0; i < n_per_chunk; ++i) {
            n_scored += mask.empty() || mask[(size_t) c*n_ctx + n_ctx/2 + i + 1];
        }
    }
    printf("%d chunks of %d tokens, %d scored per chunk%s, %d scored in all, n_gpu_layers = %d\n\n", n_chunks, n_ctx, n_per_chunk,
           mask.empty() ? "" : " before the chat mask", n_scored, n_gpu_layers);
    printf("%-44s %8s %9s %9s %10s %10s %8s\n", "model", "bpw", "NLL", "PPL", "mean KL", "p99 KL", "top-1 %");

    {
        // the reference's row: its NLL from the cache, its bpw from the file
        double nll = 0;
        for (int c = 0; c < n_chunks; ++c) {
            for (int i = 0; i < n_per_chunk; ++i) {
                if (!mask.empty() && !mask[(size_t) c*n_ctx + n_ctx/2 + i + 1]) {
                    continue;
                }
                const llama_token t = tokens[(size_t) c*n_ctx + n_ctx/2 + i + 1];
                ggml_fp16_t lp;
                const off_t off = sizeof(cache_header) + (((off_t) c*n_per_chunk + i)*n_vocab + t)*(off_t) sizeof(ggml_fp16_t);
                if (pread(fileno(cache), &lp, sizeof(lp), off) != sizeof(lp)) {
                    fprintf(stderr, "failed to read %s\n", cache_path.c_str());
                    return 1;
                }
                nll -= ggml_fp16_to_fp32(lp);
            }
        }
        nll /= n_scored;

        double bytes = 0, params = 0;
        ggml_context * meta = nullptr;
        if (gguf_context * g = gguf_init_from_file(ref_path.c_str(), { /*.no_alloc =*/ true, /*.ctx =*/ &meta })) {
            for (ggml_tensor * t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
                bytes  += ggml_nbytes(t);
                params += ggml_nelements(t);
            }
            gguf_free(g);
            ggml_free(meta);
        }
        const char * name = strrchr(ref_path.c_str(), '/');
        name = name ? name + 1 : ref_path.c_str();
        printf("%-44s %8.3f %9.5f %9.4f %10.6f %10.6f %8.3f\n", name, params > 0 ? 8.0*bytes/params : 0.0, nll, std::exp(nll),
               0.0, 0.0, 100.0);
        fflush(stdout);
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;

    int ret = 0;
    for (int m = arg + 2; m < argc; ++m) {
        llama_model * model = llama_model_load_from_file(argv[m], mp);
        if (!model) {
            fprintf(stderr, "failed to load %s\n", argv[m]);
            ret = 1;
            continue;
        }
        llama_context * ctx = make_ctx(model, n_ctx, lora, n_par);

        window_stats tot;
        const bool ok = score(ctx, cache, tokens, n_ctx, n_chunks, n_vocab, nth, tot, n_par, mask.empty() ? nullptr : &mask);

        const char * name = strrchr(argv[m], '/');
        name = name ? name + 1 : argv[m];
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
        llama_model_free(model);
    }

    fclose(cache);
    return ret;
}
