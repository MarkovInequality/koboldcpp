// Every architecture with its rotatable weights stored rotated gives the same logits as without: each weight
// the quantizer would rotate (llama_quant_tensor_rotatable) is replaced by R*w row by row and marked rotated,
// so a graph that multiplies it without rotating its input first fails the guard or changes the logits.
//
// usage: test-hadamard-archs [-a REGEX] [-d DEVICE] [-s SEED] [-v]
//        test-hadamard-archs -g [-a REGEX] [-d DEVICE] [-s SEED] [-ngl N] [-pp] [-v]
//
// -g: the batch shapes of MTP verify and small prompts give bitwise the same logits with CUDA graphs as without them
// (a child process reruns everything with GGML_CUDA_DISABLE_GRAPHS=1), and multi-token batches do run as graphs:
// per model, a 16-token prefill, then up to 4 batches of each size in G_TS, the CUDA graph launch and capture counters
// checked per size. Each model also runs quantized to HQ4_K_M (MMVQ/MMQ, RHT, quantized experts). -ngl offloads only
// N layers (a CPU split feeds the GPU), -pp turns on pipeline parallelism (needs two devices, e.g. GGML_CUDA_DEVICES=2).
//
// The random-model fixture (from get_tokens to arch_supported) is upstream's tests/test-llama-archs.cpp at
// 53ed051ce, the fork's last upstream merge, unchanged.

#include "common/common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include "llama-arch.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "llama-model-saver.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

extern bool kcpp_pipeline_parallelism;

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

struct tensor_data_params {
    size_t seed;
    float  stdev;
};

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;
    size_t seed = params.seed;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, params.stdev);

    // TODO: refactor per-tensor initialization logic in a cleaner way

    // note: Mamba A must be negative (state decay)
    const bool is_ssm_a = strstr(tensor->name, "ssm_a") != nullptr;
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            float val = dis(gen);
            tmp[i] = is_ssm_a ? -fabsf(val) : val;
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            float val = dis(gen);
            tmp[i] = ggml_fp32_to_fp16(is_ssm_a ? -fabsf(val) : val);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_HY_V4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        //n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
        n_vocab = 3072; // TODO: should be 4096, but user code cannot get `n_vocab_out` yet [TAG_LLAMA_N_VOCAB_OUT]
    } else if (arch == LLM_ARCH_HRM_TEXT) {
        n_layer = 8; // 1 layer per stack x 2 h-cycles x (3 l-cycles + 1) cache slots
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head_kv = 2; // GQA coverage
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_SPARK2_5 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE ||
            arch == LLM_ARCH_MAPLE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ 0 }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    // note: using a realistic top-k here makes the results unstable and hard to match between CPU and GPU
    //       a large value makes things deterministic since all data is selected by the indexer
    //ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(131072));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    // mrope sections count rope pairs; Ling 3.0 VL files carry [t, h, w] sections
    // summing to n_rot / 2 (n_rot is 64 in this fixture)
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({8, 12, 12, 0}));
    } else {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    }

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 8 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(3));
    }

    if (arch == LLM_ARCH_MAPLE) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, 7.0f);
    }

    // dummy tokenizer: token ids are derived from fixed-size chunks and detokenized as hex ids
    {
        std::vector<std::string> tokenizer_list(n_vocab);
        std::vector<float>       tokenizer_scores(n_vocab, 0.0f);

        ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }

    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(4) : uint32_t(2)); // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,             true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,               n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,              3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,        -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_MAPLE:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static std::string g_log;
static bool        g_verbose = false;

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN || g_verbose) {
        g_log += text;
    }
}

// dev nullptr: the CPU only, unless all_devices
static llama_model_ptr make_model(gguf_context * gguf_ctx, size_t seed, ggml_backend_dev_t dev, int ngl = -1, bool all_devices = false) {
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    mp.devices = dev || !all_devices ? devs : nullptr;
    if (ngl >= 0) {
        mp.n_gpu_layers = ngl;
    }
    // the 1e-6 threshold below assumes this scale (upstream's fixture now defaults to 0.1)
    tensor_data_params tp = { seed, 0.01f };
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx, set_tensor_data, &tp, mp));
    if (!model) {
        throw std::runtime_error("failed to create the model");
    }
    return model;
}

// Each activation is rotated once: no two RHT nodes of a graph read the same rows (one tensor, or reshapes of it
// that keep the row width). Seen through the eval callback's queries, which leave the graph unsplit.
struct rht_census {
    std::map<const ggml_tensor *, std::set<const ggml_tensor *>> by_rows;
    std::set<const ggml_tensor *> seen;
    std::string twice; // a rows tensor rotated by more than one RHT node
};

static bool rht_census_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * c = (rht_census *) user_data;
    if (ask && t->op == GGML_OP_RHT) {
        if (!c->seen.insert(t).second) {
            // the next evaluation of the graph (reused, or rebuilt in the same memory)
            c->by_rows.clear();
            c->seen = { t };
        }
        const ggml_tensor * x = t->src[0];
        while (x->op == GGML_OP_RESHAPE && x->src[0]->ne[0] == x->ne[0]) {
            x = x->src[0];
        }
        if (c->by_rows[x].insert(t).second && c->by_rows[x].size() > 1 && c->twice.empty()) {
            c->twice = x->name;
        }
    }
    return false;
}

static llama_context_ptr make_ctx(llama_model * model, bool encode, bool fa, rht_census * census = nullptr) {
    llama_context_params cp = llama_context_default_params();
    if (census) {
        cp.cb_eval           = rht_census_cb;
        cp.cb_eval_user_data = census;
    }
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_ctx = 0;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    if (!encode) {
        cp.n_ubatch = 64;
    }
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    if (!ctx) {
        throw std::runtime_error("failed to create the context");
    }
    return ctx;
}

static const uint64_t SEED = 0x48512d524854ull;

// every weight the quantizer would rotate, rotated in place and marked; returns their count
static int rotate_weights(llama_model * model) {
    llama_model_quantize_params qp = llama_model_quantize_default_params();
    quantize_state_impl * qs = llama_quant_init(model, &qp);
    int n = 0;
    for (const auto & [name, t] : model->tensors_by_name) {
        if (!llama_quant_tensor_rotatable(qs, t)) {
            continue;
        }
        GGML_ASSERT(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16);
        const int64_t ne0 = t->ne[0], nr = ggml_nrows(t);
        std::vector<float> w(ggml_nelements(t));
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, w.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<ggml_fp16_t> h(w.size());
            ggml_backend_tensor_get(t, h.data(), 0, ggml_nbytes(t));
            ggml_fp16_to_fp32_row(h.data(), w.data(), (int64_t) w.size());
        }
        for (int64_t r = 0; r < nr; ++r) {
            ggml_rht_ref(w.data() + r*ne0, ne0, SEED);
        }
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, w.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<ggml_fp16_t> h(w.size());
            ggml_fp32_to_fp16_row(w.data(), h.data(), (int64_t) w.size());
            ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
        }
        model->rotated_tensors.insert(t);
        n++;
    }
    model->hadamard_seed = SEED;
    llama_quant_free(qs);
    return n;
}

static std::string last_lines(const std::string & s, int n) {
    size_t pos = s.size();
    for (int i = 0; i <= n && pos != std::string::npos && pos > 0; ++i) {
        pos = s.rfind('\n', pos - 1);
    }
    return pos == std::string::npos || pos >= s.size() ? s : s.substr(pos + 1);
}

static const int G_PREFILL = 16;
static const int G_CTX     = 256; // n_kv is capped at the cache size, so it stays the same for every batch
static const int G_TS[]    = { 2, 4, 5, 8, 9, 33, 64 };
static const int G_N_TS    = sizeof(G_TS)/sizeof(G_TS[0]);

struct g_options {
    std::string filter;
    ggml_backend_dev_t dev = nullptr;
    size_t seed = 42;
    int ngl = -1;
    bool pp = false;
};

struct g_counters {
    int64_t (*launches)() = nullptr;
    int64_t (*captures)() = nullptr;
    bool no_vmm = false;
};

static g_counters g_get_counters() {
    g_counters c;
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg) {
        return c;
    }
    c.launches = (int64_t (*)()) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_launch_count");
    c.captures = (int64_t (*)()) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_capture_count");
    auto get_features = (ggml_backend_feature * (*)(ggml_backend_reg_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    for (ggml_backend_feature * f = get_features ? get_features(reg) : nullptr; f && f->name; ++f) {
        c.no_vmm |= strcmp(f->name, "NO_VMM") == 0;
    }
    return c;
}

struct g_run {
    std::vector<float> logits;
    int64_t launches[G_N_TS] = {};
    int64_t captures[G_N_TS] = {};
    int     n_batches[G_N_TS] = {};
};

static g_run g_run_sequence(llama_model * model, const std::vector<llama_token> & tokens, const g_counters & gc) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = G_CTX;
    cp.n_batch         = 64;
    cp.n_ubatch        = 64;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    if (!ctx) {
        throw std::runtime_error("failed to create the context");
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init(64, 0, 1);
    g_run r;
    size_t next = 0;
    auto decode = [&](int n, int pos0) {
        common_batch_clear(batch);
        for (int i = 0; i < n; ++i) {
            common_batch_add(batch, tokens[next++ % tokens.size()], pos0 + i, {0}, true);
        }
        if (llama_decode(ctx.get(), batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to decode batch");
        }
        for (int i = 0; i < n; ++i) {
            const float * l = llama_get_logits_ith(ctx.get(), i);
            r.logits.insert(r.logits.end(), l, l + n_vocab);
        }
    };
    for (int it = 0; it < G_N_TS; ++it) {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        decode(G_PREFILL, 0);
        const int T  = G_TS[it];
        const int nb = std::min(4, (G_CTX - G_PREFILL)/T);
        const int64_t l0 = gc.launches ? gc.launches() : 0;
        const int64_t c0 = gc.captures ? gc.captures() : 0;
        for (int b = 0; b < nb; ++b) {
            decode(T, G_PREFILL + b*T);
        }
        r.launches[it]  = (gc.launches ? gc.launches() : 0) - l0;
        r.captures[it]  = (gc.captures ? gc.captures() : 0) - c0;
        r.n_batches[it] = nb;
    }
    llama_batch_free(batch);
    return r;
}

// the fixture saved, quantized to HQ4_K_M and loaded back
static llama_model_ptr g_make_hq_model(llama_model * src, const g_options & o) {
    const std::string dir = (std::filesystem::temp_directory_path() / ("test-hadamard-archs-" + std::to_string(getpid()))).string();
    std::filesystem::create_directories(dir);
    const std::string f16 = dir + "/src.gguf", hq = dir + "/hq.gguf";
    llama_model_save_to_file(src, f16.c_str());
    llama_model_quantize_params qp = llama_model_quantize_default_params();
    qp.ftype    = LLAMA_FTYPE_MOSTLY_Q4_K_M;
    qp.hadamard = true;
    qp.nthread  = 4;
    const uint32_t rc = llama_model_quantize(f16.c_str(), hq.c_str(), &qp);
    llama_model_ptr model;
    if (rc == 0) {
        llama_model_params mp = llama_model_default_params();
        mp.progress_callback = silent_model_load_progress;
        ggml_backend_dev_t devs[2] = { o.dev, nullptr };
        mp.devices = o.dev ? devs : nullptr;
        if (o.ngl >= 0) {
            mp.n_gpu_layers = o.ngl;
        }
        model.reset(llama_model_load_from_file(hq.c_str(), mp));
    }
    std::filesystem::remove_all(dir);
    if (!model) {
        throw std::runtime_error(rc ? "quantization failed" : "failed to load the quantized model");
    }
    return model;
}

// calls f(label, model) for every model of the -g matrix; a model that fails to build is skipped by both processes
template <typename F>
static void g_for_each_model(const g_options & o, F && f) {
    for (const llm_arch arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_supported(arch) ||
                (!o.filter.empty() && !std::regex_search(llm_arch_name(arch), std::regex(o.filter)))) {
            continue;
        }
        if (arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE ||
                arch == LLM_ARCH_RND1) {
            continue; // encoders and diffusion models don't decode in batches
        }
        for (int cfg = 0; cfg < 4; ++cfg) {
            const bool moe = cfg & 1;
            const bool hq  = cfg & 2;
            if ((moe && !moe_implemented(arch)) || (!moe && moe_mandatory(arch))) {
                continue;
            }
            const std::string label = std::string(llm_arch_name(arch)) + (moe ? " (MoE)" : "") + (hq ? " HQ4_K_M" : "");
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            g_log.clear();
            llama_model_ptr model;
            try {
                model = make_model(gguf_ctx.get(), o.seed, o.dev, o.ngl, true);
                if (hq) {
                    model = g_make_hq_model(model.get(), o);
                }
            } catch (const std::exception & e) {
                f(label, nullptr, moe, std::string(e.what()));
                continue;
            }
            f(label, model.get(), moe, std::string());
        }
    }
}

static void g_write_str(std::ofstream & out, const std::string & s) {
    const uint64_t n = s.size();
    out.write((const char *) &n, sizeof(n));
    out.write(s.data(), n);
}

// the child: every model's logits without CUDA graphs, written to path ("" for a model that failed)
static int g_child(const g_options & o, const std::string & path) {
    std::ofstream out(path, std::ios::binary);
    const std::vector<llama_token> tokens = get_tokens(1024, 128, o.seed);
    g_for_each_model(o, [&](const std::string & label, llama_model * model, bool, const std::string & err) {
        std::string data;
        if (model && err.empty()) {
            try {
                const g_run r = g_run_sequence(model, tokens, g_counters());
                data.assign((const char *) r.logits.data(), r.logits.size()*sizeof(float));
            } catch (const std::exception &) {
            }
        }
        g_write_str(out, label);
        g_write_str(out, data);
    });
    return out.good() ? 0 : 1;
}

static std::map<std::string, std::string> g_read(const std::string & path) {
    std::map<std::string, std::string> res;
    std::ifstream in(path, std::ios::binary);
    auto read_str = [&](std::string & s) {
        uint64_t n = 0;
        if (!in.read((char *) &n, sizeof(n))) {
            return false;
        }
        s.resize(n);
        return (bool) in.read(s.data(), n);
    };
    std::string label, data;
    while (read_str(label) && read_str(data)) {
        res[label] = data;
    }
    return res;
}

static int g_main(const g_options & o, char ** argv) {
    const std::string ref_path = (std::filesystem::temp_directory_path() / ("test-hadamard-archs-g-" + std::to_string(getpid()) + ".bin")).string();
    std::vector<std::string> args = { argv[0], "-g", "--g-child", ref_path, "-s", std::to_string(o.seed) };
    if (!o.filter.empty()) {
        args.insert(args.end(), { "-a", o.filter });
    }
    if (o.dev) {
        args.insert(args.end(), { "-d", ggml_backend_dev_name(o.dev) });
    }
    if (o.ngl >= 0) {
        args.insert(args.end(), { "-ngl", std::to_string(o.ngl) });
    }
    if (o.pp) {
        args.push_back("-pp");
    }
    printf("reference run without CUDA graphs...\n");
    fflush(stdout);
    const std::string child_log = ref_path + ".log";
    const pid_t pid = fork();
    if (pid == 0) {
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
        if (FILE * f = fopen(child_log.c_str(), "w")) {
            dup2(fileno(f), STDOUT_FILENO);
            dup2(fileno(f), STDERR_FILENO);
        }
        std::vector<char *> cargs;
        for (auto & a : args) {
            cargs.push_back(a.data());
        }
        cargs.push_back(nullptr);
        execv(cargs[0], cargs.data());
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    const std::map<std::string, std::string> ref = g_read(ref_path);
    std::filesystem::remove(ref_path);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::ifstream log(child_log);
        printf("FAIL: the reference process failed\n%s\n", std::string(std::istreambuf_iterator<char>(log), {}).c_str());
        std::filesystem::remove(child_log);
        return 1;
    }
    std::filesystem::remove(child_log);

    const g_counters gc = g_get_counters();
    if (!gc.launches || !gc.captures) {
        printf("FAIL: no CUDA graph counters (not a CUDA build?)\n");
        return 1;
    }
    printf("graph launches per batch size %s", gc.no_vmm ? "(NO_VMM: graphs must stay off)" : "");
    for (int it = 0; it < G_N_TS; ++it) {
        printf("%s%d", it ? "/" : " T=", G_TS[it]);
    }
    printf("\n");

    const std::vector<llama_token> tokens = get_tokens(1024, 128, o.seed);
    int n_pass = 0, n_fail = 0, n_skip = 0;
    std::vector<std::string> failed;
    g_for_each_model(o, [&](const std::string & label, llama_model * model, bool moe, const std::string & err) {
        const auto it_ref = ref.find(label);
        if (!model || it_ref == ref.end() || it_ref->second.empty()) {
            printf("  %-40s SKIP (%s)\n", label.c_str(), !err.empty() ? err.c_str() : "fails without graphs");
            n_skip++;
            return;
        }
        std::string status = "PASS";
        std::string counts;
        try {
            const g_run r = g_run_sequence(model, tokens, gc);
            const std::string & rs = it_ref->second;
            if (rs.size() != r.logits.size()*sizeof(float)) {
                status = "FAIL (logit count differs)";
            } else if (memcmp(rs.data(), r.logits.data(), rs.size()) != 0) {
                const float * a = (const float *) rs.data();
                size_t first = 0;
                double max_diff = 0;
                for (size_t i = r.logits.size(); i-- > 0;) {
                    if (memcmp(&a[i], &r.logits[i], sizeof(float)) != 0) {
                        first = i;
                        max_diff = std::max(max_diff, (double) std::fabs(a[i] - r.logits[i]));
                    }
                }
                char buf[128];
                snprintf(buf, sizeof(buf), "FAIL (logits differ from #%zu, max |diff| %.3g)", first, max_diff);
                status = buf;
            }
            for (int it = 0; it < G_N_TS; ++it) {
                counts += (it ? "/" : "") + std::to_string(r.launches[it]);
                // a re-capture on every batch (a graph property that changes between same-shaped batches) shows as
                // captures without matching launches; a dense model must run every batch size as a graph
                const bool churn = r.n_batches[it] >= 3 && r.launches[it] < 2*r.captures[it];
                const bool off   = gc.no_vmm ? r.launches[it] + r.captures[it] > 0 : !moe && r.launches[it] == 0;
                if (status == "PASS" && (churn || off)) {
                    status = "FAIL (" + std::string(churn ? "re-captured" : gc.no_vmm ? "graphs on the legacy pool" : "no graph") +
                             " at T=" + std::to_string(G_TS[it]) + ")";
                }
            }
        } catch (const std::exception & e) {
            status = std::string("FAIL (") + e.what() + ")";
        }
        printf("  %-40s launches %-22s %s\n", label.c_str(), counts.c_str(), status.c_str());
        if (status != "PASS") {
            const std::string tail = last_lines(g_log, 3);
            if (!tail.empty()) {
                printf("%s", tail.c_str());
            }
            failed.push_back(label);
            n_fail++;
        } else {
            n_pass++;
        }
        fflush(stdout);
    });

    printf("\n%d passed, %d failed, %d skipped\n", n_pass, n_fail, n_skip);
    for (const auto & f : failed) {
        printf("  failed: %s\n", f.c_str());
    }
    printf("%s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}

int main(int argc, char ** argv) {
    std::string filter;
    std::string device;
    std::string g_child_path;
    size_t seed = 42;
    bool graphs = false;
    g_options go;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-a" || a == "-d" || a == "-s" || a == "-ngl" || a == "--g-child") && i + 1 < argc) {
            const std::string v = argv[++i];
            if (a == "-a") filter = v; else if (a == "-d") device = v; else if (a == "-ngl") go.ngl = std::stoi(v);
            else if (a == "--g-child") g_child_path = v; else seed = std::stoull(v);
        } else if (a == "-v") {
            g_verbose = true;
        } else if (a == "-g") {
            graphs = true;
        } else if (a == "-pp") {
            go.pp = true;
        } else {
            printf("usage: %s [-a REGEX] [-d DEVICE] [-s SEED] [-v]\n"
                   "       %s -g [-a REGEX] [-d DEVICE] [-s SEED] [-ngl N] [-pp] [-v]\n", argv[0], argv[0]);
            return 1;
        }
    }
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    ggml_backend_dev_t dev = nullptr;
    if (!device.empty()) {
        dev = ggml_backend_dev_by_name(device.c_str());
        if (!dev) {
            printf("unknown device %s\n", device.c_str());
            return 1;
        }
    }

    if (graphs) {
        go.filter = filter;
        go.dev    = dev;
        go.seed   = seed;
        kcpp_pipeline_parallelism = go.pp;
        return g_child_path.empty() ? g_main(go, argv) : g_child(go, g_child_path);
    }

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);
    int n_pass = 0, n_fail = 0, n_skip = 0;
    std::vector<std::string> failed;

    std::string unsupported;
    for (const llm_arch arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || (!filter.empty() && !std::regex_search(llm_arch_name(arch), std::regex(filter)))) {
            continue;
        }
        if (!arch_supported(arch)) {
            unsupported += std::string(unsupported.empty() ? "" : ", ") + llm_arch_name(arch);
            continue;
        }
        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (int cfg = 0; cfg < 4; ++cfg) {
            const bool moe = cfg & 1;
            const bool fa  = !(cfg & 2);
            if ((moe && !moe_implemented(arch)) || (!moe && moe_mandatory(arch))) {
                continue;
            }
            const std::string label = std::string(llm_arch_name(arch)) + (moe ? " (MoE)" : "") + (fa ? "" : " -fa off");
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            g_log.clear();

            std::vector<float> ref;
            try {
                llama_model_ptr model = make_model(gguf_ctx.get(), seed, dev);
                llama_context_ptr ctx = make_ctx(model.get(), encode, fa);
                ref = get_logits(model.get(), ctx.get(), tokens, encode);
            } catch (const std::exception & e) {
                printf("  %-40s SKIP (fails without rotation: %s)\n", label.c_str(), e.what());
                n_skip++;
                continue;
            }
            if (!std::all_of(ref.begin(), ref.end(), [](float x) { return std::isfinite(x); })) {
                printf("  %-40s SKIP (non-finite logits without rotation)\n", label.c_str());
                n_skip++;
                continue;
            }

            std::string status;
            int n_rot = 0;
            double err = -1;
            try {
                llama_model_ptr model = make_model(gguf_ctx.get(), seed, dev);
                n_rot = rotate_weights(model.get());
                rht_census census;
                llama_context_ptr ctx = make_ctx(model.get(), encode, fa, &census);
                err = nmse(ref, get_logits(model.get(), ctx.get(), tokens, encode));
                status = n_rot > 0 && err < 1e-6 ? "PASS" : "FAIL";
                if (!census.twice.empty()) {
                    status = "FAIL (" + census.twice + " rotated by two RHT nodes)";
                }
            } catch (const std::exception & e) {
                status = std::string("FAIL (") + e.what() + ")";
            }
            printf("  %-40s %3d rotated, nmse %9.2e  %s\n", label.c_str(), n_rot, err, status.c_str());
            if (status != "PASS") {
                const std::string tail = last_lines(g_log, 3);
                if (!tail.empty()) {
                    printf("%s", tail.c_str());
                }
                failed.push_back(label);
                n_fail++;
            } else {
                n_pass++;
            }
            fflush(stdout);
        }
    }

    printf("\n%d passed, %d failed, %d skipped (broken without rotation)\n", n_pass, n_fail, n_skip);
    printf("not built by the fixture: %s\n", unsupported.c_str());
    for (const auto & f : failed) {
        printf("  failed: %s\n", f.c_str());
    }
    printf("%s\n", n_fail ? "FAIL" : "PASS");
    return n_fail ? 1 : 0;
}
