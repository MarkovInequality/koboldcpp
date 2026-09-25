// Quality of quantized models against a reference: perplexity, KL divergence and top-token
// agreement over non-overlapping windows of a text file, plus bits per weight.
//
// usage: test-hadamard-ppl [-ngl N] [-c N_CTX] [--chunks N] [--cache FILE] [--lora FILE] <text-file> <reference.gguf> [model.gguf...]
//
// The reference's log-probabilities are written to the cache file (FP16) once and streamed back for
// every model. Tokens in the second half of each window are scored, so each has at least N_CTX/2
// tokens of context. --lora applies a runtime adapter to the reference and to every model.
// The fork has no perplexity tool; this is its replacement for these runs.

#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct cache_header {
    uint32_t magic = 0x4c4b5148; // "HQKL"
    int32_t  n_vocab;
    int32_t  n_ctx;
    int32_t  n_chunks;
    uint64_t token_hash;
};

static std::vector<llama_token> tokenize_file(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tokens(text.size() + 16);
    const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), true, false);
    if (n < 0) {
        fprintf(stderr, "tokenization failed (%d)\n", n);
        return {};
    }
    tokens.resize(n);
    return tokens;
}

static void log_softmax(const float * logits, int n, std::vector<float> & out) {
    float mx = logits[0];
    for (int j = 1; j < n; ++j) {
        mx = std::max(mx, logits[j]);
    }
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
        sum += std::exp((double) logits[j] - mx);
    }
    const float lse = mx + (float) std::log(sum);
    out.resize(n);
    for (int j = 0; j < n; ++j) {
        out[j] = logits[j] - lse;
    }
}

struct window_stats {
    double nll = 0, kl = 0;
    int    n = 0, top1 = 0;
    std::vector<float> kls;
};

// runs one window and hands each scored position's log-probs to fn(i, logp)
template <typename F>
static bool run_window(llama_context * ctx, const llama_token * window, int n_ctx, int n_vocab, int nth, F && fn) {
    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    batch.n_tokens = n_ctx;
    for (int i = 0; i < n_ctx; i++) {
        batch.token[i]     = window[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i >= n_ctx/2 && i < n_ctx - 1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed (%d)\n", rc);
        return false;
    }

    const int i0 = n_ctx/2, i1 = n_ctx - 1;
    std::vector<std::thread> threads;
    for (int t = 0; t < nth; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<float> logp;
            for (int i = i0 + t; i < i1; i += nth) {
                log_softmax(llama_get_logits_ith(ctx, i), n_vocab, logp);
                fn(i - i0, logp);
            }
        });
    }
    for (auto & th : threads) {
        th.join();
    }
    return true;
}

static std::string g_lora;

static llama_context * make_ctx(llama_model * model, int n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx         = n_ctx;
    cp.n_batch       = n_ctx;
    cp.n_ubatch      = n_ctx;
    cp.n_outputs_max = n_ctx;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx && !g_lora.empty()) {
        llama_adapter_lora * a = llama_adapter_lora_init(model, g_lora.c_str());
        float scale = 1.0f;
        if (!a || llama_set_adapters_lora(ctx, &a, 1, &scale) != 0) {
            fprintf(stderr, "failed to apply LoRA %s\n", g_lora.c_str());
            exit(1);
        }
    }
    return ctx;
}

int main(int argc, char ** argv) {
    int n_gpu_layers = 0;
    int n_ctx        = 512;
    int n_chunks     = 0;
    std::string cache_path;

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
            g_lora = argv[arg + 1];
        } else {
            break;
        }
    }
    if (argc - arg < 2) {
        fprintf(stderr, "usage: %s [-ngl N] [-c N_CTX] [--chunks N] [--cache FILE] [--lora FILE] <text-file> <reference.gguf> [model.gguf...]\n", argv[0]);
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
    const std::string ref_path = argv[arg + 1];
    if (cache_path.empty()) {
        cache_path = ref_path + (g_lora.empty() ? "" : ".lora") + ".kl-cache";
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
    const int n_per_chunk = n_ctx - 1 - n_ctx/2;

    cache_header hdr;
    hdr.n_vocab  = n_vocab;
    hdr.n_ctx    = n_ctx;
    hdr.n_chunks = n_chunks;
    hdr.token_hash = 1469598103934665603ull;
    for (int i = 0; i < n_chunks*n_ctx; ++i) {
        hdr.token_hash = (hdr.token_hash ^ (uint32_t) tokens[i]) * 1099511628211ull;
    }

    bool have_cache = false;
    if (FILE * f = fopen(cache_path.c_str(), "rb")) {
        cache_header h;
        have_cache = fread(&h, sizeof(h), 1, f) == 1 && memcmp(&h, &hdr, sizeof(h)) == 0;
        fclose(f);
    }

    const size_t row_bytes = (size_t) n_vocab*sizeof(ggml_fp16_t);
    printf("%d chunks of %d tokens, %d scored per chunk, n_gpu_layers = %d\n\n", n_chunks, n_ctx, n_per_chunk, n_gpu_layers);
    printf("%-44s %8s %9s %9s %10s %10s %8s\n", "model", "bpw", "NLL", "PPL", "mean KL", "p99 KL", "top-1 %");

    if (!have_cache) {
        FILE * f = fopen(cache_path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "cannot write %s\n", cache_path.c_str());
            return 1;
        }
        fwrite(&hdr, sizeof(hdr), 1, f);
        llama_context * ctx = make_ctx(ref, n_ctx);
        std::vector<ggml_fp16_t> rows((size_t) n_per_chunk*n_vocab);
        for (int c = 0; c < n_chunks; ++c) {
            const bool ok = run_window(ctx, tokens.data() + (size_t) c*n_ctx, n_ctx, n_vocab, nth, [&](int i, const std::vector<float> & logp) {
                ggml_fp32_to_fp16_row(logp.data(), rows.data() + (size_t) i*n_vocab, n_vocab);
            });
            if (!ok) {
                return 1;
            }
            fwrite(rows.data(), row_bytes, n_per_chunk, f);
        }
        fclose(f);
        llama_free(ctx);
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
        llama_context * ctx = make_ctx(model, n_ctx);

        FILE * f = fopen(cache_path.c_str(), "rb");
        fseek(f, sizeof(cache_header), SEEK_SET);

        std::vector<ggml_fp16_t> rows((size_t) n_per_chunk*n_vocab);
        std::vector<window_stats> st(nth);
        bool ok = true;
        for (int c = 0; c < n_chunks && ok; ++c) {
            if (fread(rows.data(), row_bytes, n_per_chunk, f) != (size_t) n_per_chunk) {
                ok = false;
                break;
            }
            const llama_token * window = tokens.data() + (size_t) c*n_ctx;
            std::vector<std::vector<float>> ref_rows(nth, std::vector<float>(n_vocab));
            ok = run_window(ctx, window, n_ctx, n_vocab, nth, [&](int i, const std::vector<float> & logp) {
                const int t = i % nth;
                auto & s = st[t];
                std::vector<float> & rp = ref_rows[t];
                ggml_fp16_to_fp32_row(rows.data() + (size_t) i*n_vocab, rp.data(), n_vocab);

                double kl = 0.0;
                int arg_ref = 0, arg_m = 0;
                for (int j = 0; j < n_vocab; ++j) {
                    kl += std::exp((double) rp[j]) * ((double) rp[j] - logp[j]);
                    arg_ref = rp[j]   > rp[arg_ref]   ? j : arg_ref;
                    arg_m   = logp[j] > logp[arg_m]   ? j : arg_m;
                }
                s.nll  -= logp[window[n_ctx/2 + i + 1]];
                s.kl   += kl;
                s.kls.push_back((float) kl);
                s.top1 += arg_ref == arg_m;
                s.n++;
            });
        }
        fclose(f);

        window_stats tot;
        for (auto & s : st) {
            tot.nll += s.nll; tot.kl += s.kl; tot.n += s.n; tot.top1 += s.top1;
            tot.kls.insert(tot.kls.end(), s.kls.begin(), s.kls.end());
        }
        std::sort(tot.kls.begin(), tot.kls.end());

        const char * name = strrchr(models[m].c_str(), '/');
        name = name ? name + 1 : models[m].c_str();
        const double bpw = 8.0*llama_model_size(model)/llama_model_n_params(model);

        if (!ok || tot.n == 0) {
            printf("%-44s %8.3f %9s\n", name, bpw, "FAILED");
            ret = 1;
        } else {
            const double nll = tot.nll/tot.n;
            printf("%-44s %8.3f %9.5f %9.4f %10.6f %10.6f %8.3f\n", name, bpw, nll, std::exp(nll), tot.kl/tot.n,
                   tot.kls[(size_t) (0.99*(tot.kls.size() - 1))], 100.0*tot.top1/tot.n);
        }
        fflush(stdout);

        llama_free(ctx);
        if (m != 0) {
            llama_model_free(model);
        }
    }

    llama_model_free(ref);
    return ret;
}
