// Synthetic model files for image generation tests: a tensor table taken from a model block or from a recorded file
// layout, filled with deterministic values and written as safetensors. Tiny configs keep the tests download-free.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "ggml.h"
#include "model/common/ggml_block.hpp"
#include "model/te/llm.hpp"

namespace synth {

struct Tensor {
    std::string name;
    std::string dtype;           // F32, F16 or BF16
    std::vector<int64_t> shape;  // file order, outermost first
    std::vector<float> delta;    // added to the generated values when not empty
    float gain = 1.f;            // multiplies the generated values
};

inline int64_t numel(const Tensor& t) {
    int64_t n = 1;
    for (auto d : t.shape) {
        n *= d;
    }
    return n;
}

inline uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h = (h ^ c) * 1099511628211ull;
    }
    return h;
}

// Norm scales sit near 1, other vectors near 0, matrices are uniform in +-1/sqrt(fan_in).
inline std::vector<float> values(const Tensor& t, uint64_t seed) {
    const int64_t n = numel(t);
    std::vector<float> v(n);
    uint64_t s     = fnv1a(t.name) ^ seed;
    auto next      = [&]() {
        s += 0x9e3779b97f4a7c15ull;
        uint64_t z = s;
        z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return (float)((z ^ (z >> 31)) >> 40) / (float)(1 << 24) * 2.f - 1.f;
    };
    const bool vec   = t.shape.empty() || n == t.shape[0];
    const bool scale = vec && (t.name.find("gamma") != std::string::npos ||
                               (t.name.find("norm") != std::string::npos && t.name.size() > 7 &&
                                t.name.compare(t.name.size() - 7, 7, ".weight") == 0));
    const float a    = scale ? 0.1f : vec ? 0.02f : 1.f / std::sqrt((float)(n / t.shape[0]));
    for (int64_t i = 0; i < n; ++i) {
        v[i] = ((scale ? 1.f : 0.f) + a * next()) * t.gain;
    }
    if (!t.delta.empty()) {
        for (int64_t i = 0; i < n; ++i) {
            v[i] += t.delta[i];
        }
    }
    return v;
}

inline uint16_t to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

inline bool write_safetensors(const std::string& path, const std::vector<Tensor>& tensors, uint64_t seed) {
    std::ostringstream header;
    header << "{\"__metadata__\":{\"format\":\"pt\"}";
    size_t offset = 0;
    for (const auto& t : tensors) {
        size_t bytes = (size_t)numel(t) * (t.dtype == "F32" ? 4 : 2);
        header << ",\"" << t.name << "\":{\"dtype\":\"" << t.dtype << "\",\"shape\":[";
        for (size_t i = 0; i < t.shape.size(); ++i) {
            header << (i ? "," : "") << t.shape[i];
        }
        header << "],\"data_offsets\":[" << offset << "," << offset + bytes << "]}";
        offset += bytes;
    }
    header << "}";
    std::string h = header.str();
    h.append((8 - h.size() % 8) % 8, ' ');
    std::ofstream out(path, std::ios::binary);
    uint64_t len = h.size();
    out.write((const char*)&len, 8);
    out.write(h.data(), h.size());
    for (const auto& t : tensors) {
        auto v = values(t, seed);
        if (t.dtype == "F32") {
            out.write((const char*)v.data(), v.size() * 4);
            continue;
        }
        std::vector<uint16_t> half(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            half[i] = t.dtype == "BF16" ? to_bf16(v[i]) : ggml_fp32_to_fp16(v[i]);
        }
        out.write((const char*)half.data(), half.size() * 2);
    }
    return (bool)out;
}

// The block's own parameters, matrices as F16 and vectors as F32.
inline std::vector<Tensor> from_block(GGMLBlock& block, const String2TensorStorage& storage = {}, const std::string& prefix = "") {
    ggml_init_params params = {(size_t)65536 * ggml_tensor_overhead(), nullptr, true};
    ggml_context* ctx       = ggml_init(params);
    block.init(ctx, storage, prefix);
    std::map<std::string, ggml_tensor*> tensors;
    block.get_param_tensors(tensors, prefix);
    std::vector<Tensor> out;
    for (const auto& [name, t] : tensors) {
        Tensor st{name, ggml_n_dims(t) > 1 ? "F16" : "F32", {}, {}};
        for (int i = ggml_n_dims(t) - 1; i >= 0; --i) {
            st.shape.push_back(t->ne[i]);
        }
        out.push_back(std::move(st));
    }
    ggml_free(ctx);
    return out;
}

// A recorded layout: "name dtype dims..." per line, '#' comments.
inline std::vector<Tensor> from_table(const std::string& path) {
    std::vector<Tensor> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ls(line);
        Tensor t;
        ls >> t.name >> t.dtype;
        int64_t d;
        while (ls >> d) {
            t.shape.push_back(d);
        }
        out.push_back(std::move(t));
    }
    return out;
}

inline void add_storage(String2TensorStorage& storage, const std::string& name, std::vector<int64_t> ne) {
    storage[name] = TensorStorage(name, GGML_TYPE_F16, ne.data(), (int)ne.size(), 0);
}

// A Qwen3-VL text encoder (32 heads of 128 as the architecture fixes) with an optional vision tower, named as the HF
// checkpoints are, vision under model.visual.
inline std::vector<Tensor> qwen3vl(int64_t hidden, int layers, bool vision, int64_t vision_hidden = 256) {
    String2TensorStorage storage;
    add_storage(storage, "model.embed_tokens.weight", {hidden, 151936});
    add_storage(storage, "model.layers.0.mlp.gate_proj.weight", {hidden, 3 * hidden});
    add_storage(storage, "model.layers." + std::to_string(layers - 1) + ".input_layernorm.weight", {hidden});
    if (vision) {
        add_storage(storage, "visual.patch_embed.proj.weight", {16, 16, 2, 3 * vision_hidden});
        add_storage(storage, "visual.patch_embed.proj.bias", {vision_hidden});
        add_storage(storage, "visual.pos_embed.weight", {vision_hidden, 64});
        add_storage(storage, "visual.blocks.1.norm1.weight", {vision_hidden});
        add_storage(storage, "visual.blocks.0.mlp.linear_fc1.weight", {vision_hidden, 2 * vision_hidden});
        add_storage(storage, "visual.merger.linear_fc2.weight", {4 * vision_hidden, hidden});
    }
    bool enable_vision = vision;
    auto config        = LLM::LLMConfig::detect_from_weights(storage, "", LLM::LLMArch::QWEN3_VL, enable_vision);
    LLM::LLM model(config, enable_vision, false);
    auto tensors = from_block(model, storage);
    for (auto& t : tensors) {
        if (t.name.rfind("visual.", 0) == 0) {
            t.name = "model." + t.name;
        }
    }
    return tensors;
}

}  // namespace synth
