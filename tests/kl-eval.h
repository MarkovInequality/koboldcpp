#pragma once

// Scoring a model against a reference's log-probabilities, cached to a file (FP16), over non-overlapping
// windows of a text. Tokens in the second half of each window are scored, so each has at least n_ctx/2
// tokens of context. Shared by test-hadamard-ppl and tensor-kl, which read the same cache files.

#include "llama.h"
#include "common/common.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
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

// 1 for the tokens a chat model generates: its own turns (<|im_start|>assistant\n ... <|im_end|>) and text outside any
// turn; 0 for turn headers and system/user/tool turns, where chat models aren't trained to predict. <|endoftext|>
// ends a turn.
static std::vector<uint8_t> chat_mask(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    auto id = [&](const char * s) {
        llama_token t[4];
        return llama_tokenize(vocab, s, (int32_t) strlen(s), t, 4, false, true) == 1 ? t[0] : LLAMA_TOKEN_NULL;
    };
    const llama_token im_start = id("<|im_start|>"), im_end = id("<|im_end|>"), eot = id("<|endoftext|>");
    const llama_token assistant = id("assistant"), nl = id("\n");
    enum { NONE, HEADER, ASSISTANT, OTHER } role = NONE;
    std::vector<uint8_t> mask(tokens.size(), 1);
    for (size_t k = 0; k < tokens.size(); ++k) {
        const llama_token t = tokens[k];
        if (t == im_start) {
            role = HEADER;
            mask[k] = 0;
        } else if (role == HEADER) {
            mask[k] = 0;
            role = t == assistant ? ASSISTANT : OTHER;
            if (k + 1 < tokens.size() && tokens[k + 1] == nl) {
                mask[++k] = 0;
            }
        } else if (t == eot) {
            role = NONE;
        } else {
            mask[k] = role != OTHER;
            role = t == im_end ? NONE : role;
        }
    }
    return mask;
}

static bool read_text_file(const char * path, std::string & text) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        text.append(buf, n);
    }
    fclose(f);
    return true;
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

    double mean_kl() const { return kl/n; }
    double p99_kl()  const { return kls[(size_t) (0.99*(kls.size() - 1))]; }
    double top1_pct() const { return 100.0*top1/n; }
};

// decodes n_win consecutive windows as the parallel sequences of one batch, so that weights that live in
// host memory cross to the GPU once for all of them
static bool decode_windows(llama_context * ctx, const llama_token * tokens, int n_win, int n_ctx) {
    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init(n_win*n_ctx, 0, 1);
    batch.n_tokens = n_win*n_ctx;
    for (int k = 0; k < n_win*n_ctx; k++) {
        const int i = k % n_ctx;
        batch.token[k]     = tokens[k];
        batch.pos[k]       = i;
        batch.n_seq_id[k]  = 1;
        batch.seq_id[k][0] = k / n_ctx;
        batch.logits[k]    = i >= n_ctx/2 && i < n_ctx - 1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed (%d)\n", rc);
        return false;
    }
    return true;
}

// hands the log-probs of each scored position of window w of the last decode to fn(i, logp)
template <typename F>
static void scored_logprobs(llama_context * ctx, int w, int n_ctx, int n_vocab, int nth, F && fn) {
    const int i0 = n_ctx/2, i1 = n_ctx - 1;
    std::vector<std::thread> threads;
    for (int t = 0; t < nth; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<float> logp;
            for (int i = i0 + t; i < i1; i += nth) {
                log_softmax(llama_get_logits_ith(ctx, w*n_ctx + i), n_vocab, logp);
                fn(i - i0, logp);
            }
        });
    }
    for (auto & th : threads) {
        th.join();
    }
}

// n_par windows per batch
static llama_context * make_ctx(llama_model * model, int n_ctx, const std::string & lora = "", int n_par = 1) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx         = n_par*n_ctx;
    cp.n_batch       = n_par*n_ctx;
    cp.n_ubatch      = n_par*n_ctx;
    cp.n_seq_max     = n_par;
    cp.n_outputs_max = n_par*n_ctx;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx && !lora.empty()) {
        llama_adapter_lora * a = llama_adapter_lora_init(model, lora.c_str());
        float scale = 1.0f;
        if (!a || llama_set_adapters_lora(ctx, &a, 1, &scale) != 0) {
            fprintf(stderr, "failed to apply LoRA %s\n", lora.c_str());
            exit(1);
        }
    }
    return ctx;
}

static uint64_t token_hash(const std::vector<llama_token> & tokens, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ (uint32_t) tokens[i]) * 1099511628211ull;
    }
    return h;
}

static int scored_per_chunk(int n_ctx) {
    return n_ctx - 1 - n_ctx/2;
}

// a complete cache of at least n_chunks windows of these tokens, or nullptr
static FILE * cache_open(const std::string & path, int n_vocab, int n_ctx, int n_chunks, const std::vector<llama_token> & tokens) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return nullptr;
    }
    cache_header h;
    bool ok = fread(&h, sizeof(h), 1, f) == 1 && h.magic == cache_header().magic && h.n_vocab == n_vocab && h.n_ctx == n_ctx &&
              h.n_chunks >= n_chunks && (size_t) h.n_chunks*n_ctx <= tokens.size() &&
              h.token_hash == token_hash(tokens, (size_t) h.n_chunks*n_ctx);
    if (ok) {
        const uint64_t need = sizeof(h) + (uint64_t) h.n_chunks*scored_per_chunk(n_ctx)*n_vocab*sizeof(ggml_fp16_t);
        ok = fseeko(f, 0, SEEK_END) == 0 && (uint64_t) ftello(f) >= need;
    }
    if (!ok) {
        fclose(f);
        return nullptr;
    }
    return f;
}

// writes the log-probs of ctx's model for the first n_chunks windows; n_par as given to make_ctx
static bool cache_write(llama_context * ctx, const std::string & path, int n_vocab, int n_ctx, int n_chunks,
                        const std::vector<llama_token> & tokens, int nth, int n_par = 1) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        return false;
    }
    cache_header hdr;
    hdr.n_vocab    = n_vocab;
    hdr.n_ctx      = n_ctx;
    hdr.n_chunks   = n_chunks;
    hdr.token_hash = token_hash(tokens, (size_t) n_chunks*n_ctx);
    fwrite(&hdr, sizeof(hdr), 1, f);

    const int n_per_chunk = scored_per_chunk(n_ctx);
    std::vector<ggml_fp16_t> rows((size_t) n_per_chunk*n_vocab);
    for (int c0 = 0; c0 < n_chunks; c0 += n_par) {
        const int n_win = std::min(n_par, n_chunks - c0);
        if (!decode_windows(ctx, tokens.data() + (size_t) c0*n_ctx, n_win, n_ctx)) {
            fclose(f);
            return false;
        }
        for (int w = 0; w < n_win; ++w) {
            scored_logprobs(ctx, w, n_ctx, n_vocab, nth, [&](int i, const std::vector<float> & logp) {
                ggml_fp32_to_fp16_row(logp.data(), rows.data() + (size_t) i*n_vocab, n_vocab);
            });
            fwrite(rows.data(), (size_t) n_vocab*sizeof(ggml_fp16_t), n_per_chunk, f);
        }
    }
    return fclose(f) == 0;
}

// KL(reference || model), top-1 agreement and NLL of ctx's model over the first n_chunks windows of an open
// cache; n_par as given to make_ctx; with a mask, only the predictions of the tokens it marks
static bool score(llama_context * ctx, FILE * cache, const std::vector<llama_token> & tokens, int n_ctx, int n_chunks,
                  int n_vocab, int nth, window_stats & tot, int n_par = 1, const std::vector<uint8_t> * mask = nullptr) {
    const int    n_per_chunk = scored_per_chunk(n_ctx);
    const size_t row_bytes   = (size_t) n_vocab*sizeof(ggml_fp16_t);
    fseeko(cache, sizeof(cache_header), SEEK_SET);

    std::vector<ggml_fp16_t> rows((size_t) n_per_chunk*n_vocab);
    std::vector<window_stats> st(nth);
    std::vector<std::vector<float>> ref_rows(nth, std::vector<float>(n_vocab));
    for (int c0 = 0; c0 < n_chunks; c0 += n_par) {
        const int n_win = std::min(n_par, n_chunks - c0);
        if (!decode_windows(ctx, tokens.data() + (size_t) c0*n_ctx, n_win, n_ctx)) {
            return false;
        }
        for (int c = c0; c < c0 + n_win; ++c) {
            if (fread(rows.data(), row_bytes, n_per_chunk, cache) != (size_t) n_per_chunk) {
                return false;
            }
            // read once per model: keep the rows from pushing the model's weights out of the page cache
            posix_fadvise(fileno(cache), sizeof(cache_header) + (off_t) c*n_per_chunk*row_bytes, (off_t) n_per_chunk*row_bytes,
                          POSIX_FADV_DONTNEED);
            const llama_token * window = tokens.data() + (size_t) c*n_ctx;
            scored_logprobs(ctx, c - c0, n_ctx, n_vocab, nth, [&](int i, const std::vector<float> & logp) {
                if (mask && !(*mask)[(size_t) c*n_ctx + n_ctx/2 + i + 1]) {
                    return;
                }
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
    }

    tot = window_stats();
    for (auto & s : st) {
        tot.nll += s.nll; tot.kl += s.kl; tot.n += s.n; tot.top1 += s.top1;
        tot.kls.insert(tot.kls.end(), s.kls.begin(), s.kls.end());
    }
    std::sort(tot.kls.begin(), tot.kls.end());
    return tot.n > 0;
}
