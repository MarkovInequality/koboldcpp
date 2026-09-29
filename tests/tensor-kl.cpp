// Quantization sensitivity of each weight of a model at its operating point: for each weight of a model
// (usually a quantized mix), replaces that weight by the same weight quantized to each given type from the
// float reference, and measures how much the KL divergence and top-token agreement against the reference
// change from the unperturbed model's. Scoring is test-hadamard-ppl's, from the same cache files: the
// reference runs once, and later runs reuse its log-probs. The unperturbed model is scored once per run.
//
// usage: tensor-kl [-ngl N] [-perturbngl N] [-ot REGEX=DEVICE] [-c N_CTX] [--chunks N] [--parallel N] [--cache FILE]
//                  [--imatrix FILE] [--types T,T,...] [--variants-mb N] [--group tensor|layer|kind] [--tensors REGEX]
//                  [--layers A[-B]] <text-file> <reference.gguf> <model.gguf>
//
// Prints a line per weight in the format of a --tensor-type-file, with the results in place of the type
// and the weight's type in the model as a comment:
//   ^blk\.0\.ffn_down\.weight$=[(q4_k, <KL change>, <top-1 change in points>), (hq4_k, ...)]  # iq4_xs
// then, as # comments, a ranking for each type. --group layer perturbs the weights of one block at a time
// (^blk\.N\.), kind one kind of weight in every block at a time (^blk\.\d+\.ffn_down\.weight$).
//
// The reference supplies each weight's values, dequantized (and rotated back for its HQ weights) if it
// isn't F32, F16 or BF16 - a near-lossless one like an HQ8_0 file makes a fast stand-in for the float model
// when the model to perturb is the reference itself. The values go through the same quantizers as
// quantize_gguf: plain types use the imatrix, HQ types are rotated with the model's seed and quantized with
// GPTQ when there is an imatrix entry (--no-gptq, --gptq-damp as in quantize_gguf). A variant
// replaces the weight as a tensor of its own type on the weight's device, so it runs on that type's kernels,
// rotated input included for an HQ type, exactly as in a file quantized that way. Each weight is quantized
// to all the types in one step, which reads it and rotates it once and builds its GPTQ factor once; the
// variants are kept in RAM, one type at a time where they'd take more than --variants-mb.
//
// -ngl is for the reference, -perturbngl (default: -ngl) for the model; --parallel N evaluates N windows per
// batch, so that weights that don't fit in VRAM cross to the GPU once per batch; -ot places weights as
// llama.cpp's --override-tensor does, for both models. The reference's settings name its default cache file.
// When the model is the reference itself, a reference computed on a different path (-ngl, --parallel, -ot)
// shrinks the changes, top-1 most.

#include "kl-eval.h"

#include "llama-ext.h"
#include "llama-model.h"
#include "llama-quant-gptq.h"
#include "common/imatrix-loader.h"

#include "ggml-backend.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <atomic>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <malloc.h>
#include <regex>
#include <set>
#include <strings.h>
#include <unistd.h>
#include <unordered_map>

// --types same and hq: per weight, its type in the model and that type's HQ twin
static const ggml_type TYPE_SAME = (ggml_type) (GGML_TYPE_COUNT + 1);
static const ggml_type TYPE_HQ   = (ggml_type) (GGML_TYPE_COUNT + 2);

static bool is_float(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

// the reference's values of a row: dequantized, and rotated back for an HQ type
static void row_to_f32(ggml_type type, const void * src, float * dst, int64_t n, uint64_t seed) {
    switch (type) {
        case GGML_TYPE_F32:  memcpy(dst, src, n*sizeof(float)); return;
        case GGML_TYPE_F16:  ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n); return;
        case GGML_TYPE_BF16: ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, n); return;
        default: break;
    }
    const bool rot = ggml_is_rotated(type);
    ggml_get_type_traits(rot ? ggml_get_base_type(type) : type)->to_float(src, dst, n);
    if (rot) {
        std::vector<double> d(dst, dst + n);
        ggml_rht_inv_f64(d.data(), n, seed);
        std::copy(d.begin(), d.end(), dst);
    }
}

static std::string type_label(ggml_type type) {
    if (type == TYPE_SAME || type == TYPE_HQ) {
        return type == TYPE_SAME ? "same" : "hq";
    }
    std::string s = ggml_type_name(type);
    for (char & c : s) {
        c = (char) tolower((unsigned char) c);
    }
    return s;
}

struct quantizer {
    uint64_t seed;
    bool     gptq;
    float    damp;
    int      nth;
    llama_gptq_cache cache { (size_t) 8 << 30 };
    std::vector<std::thread> workers;
    std::set<int64_t> warned;

    void rotate(float * w, int64_t n, int64_t nrows) {
        llama_parallel_rows(nrows, workers, nth, [&](int64_t r) { ggml_rht_ref(w + r*n, n, seed); });
    }

    // quantizes nrows rows of n values, already rotated for an HQ type, into dst; GPTQ modifies the rows
    bool quantize(ggml_type type, float * w, int64_t n, int64_t nrows, const float * im, uint8_t * dst) {
        const size_t row_size = ggml_row_size(type, n);
        const float * U = nullptr;
        if (ggml_is_rotated(type) && im && gptq) {
            llama_gptq_cache::status st;
            U = cache.get(im, n, seed, damp, nth, &st);
            if ((st == llama_gptq_cache::TOO_BIG || st == llama_gptq_cache::FAILED) && warned.insert(n).second) {
                fprintf(stderr, "warning: no GPTQ factor for width %" PRId64 " - uniform weights\n", n);
            }
        }
        if (U) {
            return llama_tensor_quantize_gptq(type, w, dst, nrows, n, U, workers, nth);
        }
        const int64_t rows_per_job = std::max<int64_t>(1, 32768/n);
        std::atomic<bool> valid { true };
        llama_parallel_rows((nrows + rows_per_job - 1)/rows_per_job, workers, nth, [&](int64_t j) {
            const int64_t r0 = j*rows_per_job;
            const size_t  sz = ggml_quantize_chunk(type, w, dst, r0*n, std::min(rows_per_job, nrows - r0), n, im);
            if (!ggml_validate_row_data(type, dst + r0*row_size, sz)) {
                valid = false;
            }
        });
        return valid;
    }
};

// the reference file, where the weights' values are read from, and its seed for HQ weights
static int      g_ref_fd   = -1;
static uint64_t g_ref_seed = 0;

struct target {
    ggml_tensor * t;
    int         layer;      // -1 outside the blocks
    std::string kind;       // the name without "blk.N." and ".weight"
    std::string label;      // the name without ".weight"
    const float * im = nullptr;
    bool        rotatable;  // --hadamard could store it rotated
    int64_t     src_offs;   // the weight's values in the reference file
    ggml_type   src_type;

    // the weight as the model has it
    ggml_type type0;
    size_t    nb0[GGML_MAX_DIMS];
    ggml_backend_buffer_t buffer0;
    void *    data0;
    void *    extra0;
    bool      rot0;

    ggml_backend_buffer_t var_buf = nullptr; // the variant in place, if any
    std::vector<std::vector<uint8_t>> var;   // quantized variants, by type index

    // rows [r0, r0 + nr) of the reference's values, via buf
    const uint8_t * read_src(int64_t r0, int64_t nr, std::vector<uint8_t> & buf) const {
        const size_t row_bytes = ggml_row_size(src_type, t->ne[0]);
        const size_t size = nr*row_bytes;
        const off_t  offs = src_offs + r0*row_bytes;
        buf.resize(size);
        for (size_t done = 0; done < size; ) {
            const ssize_t n = pread(g_ref_fd, buf.data() + done, size - done, offs + done);
            GGML_ASSERT(n > 0);
            done += n;
        }
        // read once per weight: keep it from pushing the model's weights out of the page cache
        posix_fadvise(g_ref_fd, offs, size, POSIX_FADV_DONTNEED);
        return buf.data();
    }

    ggml_type resolve(ggml_type type) const {
        return type == TYPE_SAME ? (rot0 ? ggml_get_rotated_type(type0) : type0) :
               type == TYPE_HQ   ? ggml_get_rotated_type(type0) : type;
    }

    bool applies(ggml_type type) const {
        if (type == TYPE_HQ && !ggml_is_rotated(resolve(type))) {
            return false;
        }
        type = resolve(type);
        return t->ne[0] % ggml_blck_size(type) == 0 &&
               (!ggml_is_rotated(type) || rotatable) &&
               (im || !ggml_quantize_requires_imatrix(type));
    }
};

static size_t variant_bytes(const ggml_tensor * t, ggml_type type) {
    return t->ne[1]*t->ne[2]*ggml_row_size(type, t->ne[0]);
}

// quantizes the reference's values to each of the types in ks that apply, into var[k]: reads and converts
// them once, rotates them once for all HQ types, and builds the GPTQ factor once. In slabs of rows, so that
// the float copies of a large weight (token_embd) stay small
static bool prepare(quantizer & qz, const std::vector<ggml_type> & types, const std::vector<size_t> & ks, target & tg) {
    ggml_tensor * t = tg.t;
    const int64_t n = t->ne[0], nrows = t->ne[1], n_expert = t->ne[2];
    const size_t  row_bytes = ggml_row_size(tg.src_type, n);
    const int64_t slab = std::max<int64_t>(1, ((int64_t) 64 << 20)/n);

    std::vector<size_t> todo;
    bool any_rot = false;
    for (size_t k : ks) {
        if (tg.applies(types[k])) {
            todo.push_back(k);
            tg.var[k].resize(variant_bytes(t, tg.resolve(types[k])));
            any_rot |= ggml_is_rotated(tg.resolve(types[k]));
        }
    }
    std::vector<float>   w, w_rot, work;
    std::vector<uint8_t> in;
    for (int64_t e = 0; e < n_expert && !todo.empty(); ++e) {
        for (int64_t r0 = 0; r0 < nrows; r0 += slab) {
            const int64_t   nr  = std::min(slab, nrows - r0);
            const uint8_t * src = tg.read_src(e*nrows + r0, nr, in);
            w.resize(nr*n);
            llama_parallel_rows(nr, qz.workers, qz.nth, [&](int64_t r) {
                row_to_f32(tg.src_type, src + r*row_bytes, w.data() + r*n, n, g_ref_seed);
            });
            if (any_rot) {
                w_rot = w;
                qz.rotate(w_rot.data(), n, nr);
            }
            for (size_t k : todo) {
                const ggml_type type = tg.resolve(types[k]);
                float * x = w.data();
                if (ggml_is_rotated(type)) {
                    work = w_rot;
                    x = work.data();
                }
                uint8_t * dst = tg.var[k].data() + (e*nrows + r0)*ggml_row_size(type, n);
                if (!qz.quantize(type, x, n, nr, tg.im ? tg.im + e*n : nullptr, dst)) {
                    fprintf(stderr, "%s: %s: quantized data validation failed\n", t->name, ggml_type_name(type));
                    return false;
                }
            }
        }
    }
    return true;
}

// set when a weight moved into or out of the rotated set since the last replan
static bool g_rotation_changed = false;

static void set_rotated(llama_model * model, const ggml_tensor * t, bool rot) {
    g_rotation_changed |= model->is_rotated(t) != rot;
    if (rot) {
        model->rotated_tensors.insert(t);
    } else {
        model->rotated_tensors.erase(t);
    }
}

// puts the weight back as the model has it
static void restore(llama_model * model, target & tg) {
    if (!tg.var_buf) {
        return;
    }
    ggml_tensor * t = tg.t;
    t->type   = tg.type0;
    t->buffer = tg.buffer0;
    t->data   = tg.data0;
    t->extra  = tg.extra0;
    memcpy(t->nb, tg.nb0, sizeof(t->nb));
    set_rotated(model, t, tg.rot0);
    ggml_backend_buffer_free(tg.var_buf);
    tg.var_buf = nullptr;
}

// replaces the weight by its variant k, a tensor of that type in a buffer of its own on the weight's device,
// or restores it if it has none
static bool install(llama_model * model, const std::vector<ggml_type> & types, size_t k, target & tg) {
    restore(model, tg);
    if (tg.var[k].empty()) {
        return true;
    }
    ggml_tensor * t = tg.t;
    const ggml_type vtype = tg.resolve(types[k]);
    const bool      rot  = ggml_is_rotated(vtype);
    const ggml_type type = rot ? ggml_get_base_type(vtype) : vtype;

    // a host weight may be in a mapped file or a repacked buffer, neither of which can take a new tensor
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_is_host(tg.buffer0) ? ggml_backend_cpu_buffer_type()
                                                                              : ggml_backend_buffer_get_type(tg.buffer0);
    t->type   = type;
    t->nb[0]  = ggml_type_size(type);
    t->nb[1]  = ggml_row_size(type, t->ne[0]);
    for (int i = 2; i < GGML_MAX_DIMS; ++i) {
        t->nb[i] = t->nb[i - 1]*t->ne[i - 1];
    }
    t->buffer = nullptr;
    t->data   = nullptr;
    t->extra  = nullptr;
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, t));
    if (!buf || ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: can't allocate its %s variant in %s\n", tg.t->name, ggml_type_name(vtype), ggml_backend_buft_name(buft));
        return false;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(t, tg.var[k].data(), 0, tg.var[k].size());
    set_rotated(model, t, rot);
    tg.var_buf = buf;
    return true;
}

// The graph allocator keeps its plan while a new graph has as many nodes as the last, but moving a weight into
// or out of the rotated set changes which nodes read a shared input, and a reused plan can then give that input's
// memory to a later result. Toggling a setting that forces a reserve makes the next decode plan afresh; the
// models scored here are causal.
static void replan_if_needed(llama_context * ctx) {
    if (g_rotation_changed) {
        llama_set_causal_attn(ctx, false);
        llama_set_causal_attn(ctx, true);
        g_rotation_changed = false;
    }
}

static bool parse_types(const std::string & list, std::vector<ggml_type> & types) {
    size_t pos = 0;
    while (pos <= list.size()) {
        const size_t end = std::min(list.find(',', pos), list.size());
        const std::string name = list.substr(pos, end - pos);
        if (name == "same" || name == "hq") {
            types.push_back(name == "same" ? TYPE_SAME : TYPE_HQ);
            pos = end + 1;
            continue;
        }
        ggml_type found = GGML_TYPE_COUNT;
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            const char * tn = ggml_type_name((ggml_type) i);
            if (tn && strcasecmp(tn, name.c_str()) == 0) {
                found = (ggml_type) i;
            }
        }
        if (found == GGML_TYPE_COUNT || !ggml_is_quantized(found)) {
            fprintf(stderr, "not a quantized type: '%s'\n", name.c_str());
            return false;
        }
        types.push_back(found);
        pos = end + 1;
    }
    return !types.empty();
}

static void quiet_log(ggml_log_level level, const char * text, void *) {
    static ggml_log_level last = GGML_LOG_LEVEL_NONE;
    if (level != GGML_LOG_LEVEL_CONT) {
        last = level;
    }
    if (last >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static std::string join(const std::set<std::string> & s) {
    std::string out;
    for (const std::string & x : s) {
        out += (out.empty() ? "" : ",") + x;
    }
    return out;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
        "usage: %s [options] <text-file> <reference.gguf> <model.gguf>\n"
        "\n"
        "For each weight of <model.gguf>, replaces it by the same weight quantized to each of --types from\n"
        "<reference.gguf> (dequantized if it isn't F32/F16/BF16), and prints how much the KL divergence and top-1\n"
        "agreement against the reference change from the unperturbed model's, as a --tensor-type-file:\n"
        "  ^blk\\.0\\.ffn_down\\.weight$=[(q4_k, <KL change>, <top-1 change in points>), ...]  # <type now>\n"
        "then a ranking for each type. See tools/quantize/README.md, \"Per-weight sensitivity\".\n"
        "\n"
        "  -ngl N              GPU layers of the reference, used only to compute its log-probs (default: 0)\n"
        "  -perturbngl N       GPU layers of the model to perturb (default: -ngl)\n"
        "  -ot REGEX=DEVICE    place matching weights on DEVICE (CUDA0, CPU, ...) in both models; repeatable\n"
        "  -c N                tokens per window; the last N/2-1 are scored (default: 512)\n"
        "  --chunks N          number of windows (default: 10)\n"
        "  --parallel N        windows per batch (default: 1)\n"
        "  --cache FILE        the reference's log-probs (default: <reference>.ngl<N>.p<P>[.ot<hash>].kl-cache)\n"
        "  --imatrix FILE      importance matrix: plain types use it, HQ types get GPTQ (default: none)\n"
        "  --types T,T,...     quantized types to try (default: q4_K,hq4_K); same is each weight's type in the\n"
        "                      model, requantized, and hq that type's HQ twin (weights without one are kept)\n"
        "  --variants-mb N     RAM for one weight's variants; above it, one type at a time (default: 4096)\n"
        "  --activations A     on CUDA, how the GEMMs round: f32, dequantized F32 cuBLAS GEMMs - no rounding to\n"
        "                      re-roll in the layers after a changed weight, which would put a floor under every\n"
        "                      change; f16, F16 cuBLAS GEMMs; q8, the quantized kernels, as in inference (default: f32)\n"
        "  --group G           tensor, layer or kind: what is perturbed at once (default: tensor)\n"
        "  --tensors REGEX     only the weights whose name matches (default: all)\n"
        "  --layers A[-B]      only the weights of blocks A to B (default: all, with output and token_embd)\n"
        "\n"
        "  --no-gptq, --gptq-damp D   as in quantize_gguf\n", prog);
}

int main(int argc, char ** argv) {
    int  n_gpu_layers  = 0;
    int  perturb_ngl   = -1;
    int  n_ctx         = 512;
    int  n_chunks      = 10;
    int  n_par         = 1;
    size_t variants_budget = (size_t) 4096 << 20;
    bool layers_given  = false;
    int  layer_min     = 0;
    int  layer_max     = INT32_MAX;
    std::string cache_path, imatrix_path, types_arg = "q4_K,hq4_K", tensors_arg, group_by = "tensor", activations = "f32";
    std::vector<std::string> overrides;
    const llama_model_quantize_params qparams = llama_model_quantize_default_params();
    bool  gptq      = qparams.hq_gptq;
    float gptq_damp = qparams.hq_gptq_damp;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return 0;
        }
    }

    int arg = 1;
    for (; arg + 1 < argc && argv[arg][0] == '-'; arg += 2) {
        const char * a = argv[arg];
        const char * v = argv[arg + 1];
        if (!strcmp(a, "--no-gptq")) {
            gptq = false;
            arg--;
        } else if (!strcmp(a, "--gptq-damp")) {
            gptq_damp = (float) atof(v);
        } else if (!strcmp(a, "-ngl")) {
            n_gpu_layers = atoi(v);
        } else if (!strcmp(a, "-perturbngl")) {
            perturb_ngl = atoi(v);
        } else if (!strcmp(a, "-ot")) {
            overrides.push_back(v);
        } else if (!strcmp(a, "-c")) {
            n_ctx = atoi(v);
        } else if (!strcmp(a, "--chunks")) {
            n_chunks = atoi(v);
        } else if (!strcmp(a, "--parallel")) {
            n_par = atoi(v);
        } else if (!strcmp(a, "--variants-mb")) {
            variants_budget = (size_t) atoll(v) << 20;
        } else if (!strcmp(a, "--cache")) {
            cache_path = v;
        } else if (!strcmp(a, "--imatrix")) {
            imatrix_path = v;
        } else if (!strcmp(a, "--types")) {
            types_arg = v;
        } else if (!strcmp(a, "--group")) {
            group_by = v;
        } else if (!strcmp(a, "--activations")) {
            activations = v;
        } else if (!strcmp(a, "--tensors")) {
            tensors_arg = v;
        } else if (!strcmp(a, "--layers")) {
            layers_given = true;
            if (sscanf(v, "%d-%d", &layer_min, &layer_max) == 1) {
                layer_max = layer_min;
            }
        } else {
            break;
        }
    }
    std::vector<ggml_type> types;
    if (argc - arg != 3 || n_chunks < 1 || n_par < 1 || !parse_types(types_arg, types) ||
        (group_by != "tensor" && group_by != "layer" && group_by != "kind") || (activations != "f32" && activations != "f16" && activations != "q8")) {
        print_usage(argv[0]);
        return 1;
    }
    if (perturb_ngl < 0) {
        perturb_ngl = n_gpu_layers;
    }
    const std::string ref_path   = argv[arg + 1];
    const std::string model_path = argv[arg + 2];
    std::string text;
    if (!read_text_file(argv[arg], text)) {
        return 1;
    }

    // perturbed weights change their type and data, which a reused graph wouldn't see
    setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
#ifdef GGML_USE_CUDA
    // Any rounding of intermediate values - quantized GEMMs' 8-bit input, and to a lesser degree F16 GEMMs -
    // re-rolls in every layer after a changed weight, a floor under every change as large as small real ones.
    // f32 runs every weight as a dequantized F32 cuBLAS GEMM, which removes it.
    ggml_cuda_set_mul_mat_q(activations == "q8");
    if (activations == "f32") {
        setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32", 1);
    }
#endif
    const int nth = std::max(1u, std::thread::hardware_concurrency());

    std::vector<std::string> ot_patterns;
    std::vector<llama_model_tensor_buft_override> ot;
    ot_patterns.reserve(overrides.size());
    for (const std::string & o : overrides) {
        const size_t eq = o.find('=');
        ggml_backend_buffer_type_t buft = nullptr;
        for (size_t i = 0; eq != std::string::npos && i < ggml_backend_dev_count(); ++i) {
            ggml_backend_buffer_type_t b = ggml_backend_dev_buffer_type(ggml_backend_dev_get(i));
            if (o.compare(eq + 1, std::string::npos, ggml_backend_buft_name(b)) == 0) {
                buft = b;
            }
        }
        if (!buft) {
            fprintf(stderr, "-ot %s: needs REGEX=DEVICE with a device's buffer type (CUDA0, CPU, ...)\n", o.c_str());
            return 1;
        }
        ot_patterns.push_back(o.substr(0, eq));
        ot.push_back({ ot_patterns.back().c_str(), buft });
    }
    ot.push_back({ nullptr, nullptr });

    // tokens, and the reference's log-probs unless they are cached
    std::vector<llama_token> tokens;
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
        tokens  = common_tokenize(llama_model_get_vocab(ref), text, true);
        llama_model_free(ref);
    }
    n_chunks = std::min(n_chunks, (int) (tokens.size() / n_ctx));
    n_par    = std::min(n_par, n_chunks);
    if (cache_path.empty()) {
        uint64_t h = 1469598103934665603ull;
        for (const std::string & o : overrides) {
            for (char c : o + '\n') {
                h = (h ^ (uint8_t) c) * 1099511628211ull;
            }
        }
        char ot_tag[32] = "";
        if (!overrides.empty()) {
            snprintf(ot_tag, sizeof(ot_tag), ".ot%08x", (uint32_t) (h ^ (h >> 32)));
        }
        cache_path = ref_path + ".ngl" + std::to_string(n_gpu_layers) + ".p" + std::to_string(n_par) + ot_tag +
                     (activations == "f32" ? "" : "." + activations + "a") + ".kl-cache";
    }

    FILE * cache = cache_open(cache_path, n_vocab, n_ctx, n_chunks, tokens);
    if (cache) {
        fprintf(stderr, "using the reference's log-probs in %s\n", cache_path.c_str());
    } else {
        fprintf(stderr, "computing the reference's log-probs into %s\n", cache_path.c_str());
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers          = n_gpu_layers;
        mp.tensor_buft_overrides = ot.data();
        llama_model * ref = llama_model_load_from_file(ref_path.c_str(), mp);
        llama_context * ctx = ref ? make_ctx(ref, n_ctx, "", n_par) : nullptr;
        const bool ok = ctx && cache_write(ctx, cache_path, n_vocab, n_ctx, n_chunks, tokens, nth, n_par);
        llama_free(ctx);
        llama_model_free(ref);
        cache = ok ? cache_open(cache_path, n_vocab, n_ctx, n_chunks, tokens) : nullptr;
        if (!cache) {
            fprintf(stderr, "failed to compute the reference's log-probs\n");
            return 1;
        }
    }

    // the reference's tensors, where the weights' values come from
    ggml_context * ref_meta = nullptr;
    gguf_context * ref_gguf = gguf_init_from_file(ref_path.c_str(), { /*.no_alloc =*/ true, /*.ctx =*/ &ref_meta });
    if (!ref_gguf) {
        fprintf(stderr, "failed to read %s\n", ref_path.c_str());
        return 1;
    }
    const int64_t split_kid = gguf_find_key(ref_gguf, "split.count");
    if (split_kid >= 0 && gguf_get_val_u16(ref_gguf, split_kid) > 1) {
        fprintf(stderr, "%s: a split reference isn't supported\n", ref_path.c_str());
        return 1;
    }
    g_ref_fd = open(ref_path.c_str(), O_RDONLY);
    GGML_ASSERT(g_ref_fd >= 0);
    const int64_t seed_kid = gguf_find_key(ref_gguf, "hadamard.seed");
    g_ref_seed = seed_kid >= 0 ? gguf_get_val_u64(ref_gguf, seed_kid) : 0;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers          = perturb_ngl;
    mp.tensor_buft_overrides = ot.data();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }
    if (llama_vocab_n_tokens(llama_model_get_vocab(model)) != n_vocab) {
        fprintf(stderr, "%s and %s have different vocabularies\n", ref_path.c_str(), model_path.c_str());
        return 1;
    }
    llama_context * ctx = make_ctx(model, n_ctx, "", n_par);
    if (!ctx) {
        return 1;
    }

    std::unordered_map<std::string, std::vector<float>> imatrix;
    common_imatrix im;
    if (!imatrix_path.empty() && !common_imatrix_load(imatrix_path, im)) {
        fprintf(stderr, "failed to load the imatrix %s\n", imatrix_path.c_str());
        return 1;
    }
    common_imatrix_means(im, imatrix);

    // HQ variants use the model's seed, or quantize_gguf's default for a model without rotated weights
    quantizer qz;
    if (!model->hadamard_seed) {
        model->hadamard_seed = qparams.hadamard_seed;
    }
    qz.seed = model->hadamard_seed;
    qz.nth  = nth;
    qz.gptq = gptq;
    qz.damp = gptq_damp;
    if (!(qz.damp >= LLAMA_GPTQ_DAMP_MIN)) {
        fprintf(stderr, "--gptq-damp %g: needs at least %g\n", qz.damp, LLAMA_GPTQ_DAMP_MIN);
        return 1;
    }

    // the weights to perturb, those outside the blocks first
    quantize_state_impl * qs = llama_quant_init(model, &qparams);
    // a tied output.weight loads token_embd.weight a second time; both copies are perturbed
    const std::regex tensors_re(tensors_arg.empty() ? "." : tensors_arg);
    std::vector<target> targets;
    std::set<std::string> no_imatrix;
    for (const auto & [name, t] : model->tensors_by_name) {
        int il = -1, n_prefix = 0;
        if (sscanf(name.c_str(), "blk.%d.%n", &il, &n_prefix) != 1) {
            il = -1;
        }
        if ((layers_given && (il < layer_min || il > layer_max)) ||
            !llama_quant_tensor_allows_quantization(qs, t) || !std::regex_search(name, tensors_re) || !t->buffer || !t->data) {
            continue;
        }
        GGML_ASSERT(ggml_is_contiguous(t) && t->ne[3] == 1 && !t->view_src);

        const int64_t id = gguf_find_tensor(ref_gguf, name.c_str());
        const ggml_tensor * rt = id >= 0 ? ggml_get_tensor(ref_meta, name.c_str()) : nullptr;
        const ggml_type rbase = rt && ggml_is_rotated(rt->type) ? ggml_get_base_type(rt->type) : rt ? rt->type : GGML_TYPE_COUNT;
        if (!rt || !ggml_are_same_shape(rt, t) || (!is_float(rbase) && !ggml_get_type_traits(rbase)->to_float) ||
            (ggml_is_rotated(rt->type) && !g_ref_seed)) {
            fprintf(stderr, "%s: the reference needs it in a type it can be read from, of the same shape\n", name.c_str());
            return 1;
        }

        target tg;
        tg.t         = t;
        tg.layer     = il;
        tg.label     = name.substr(0, name.size() - strlen(".weight"));
        tg.kind      = tg.label.substr(il >= 0 ? n_prefix : 0);
        tg.rotatable = llama_quant_tensor_rotatable(qs, t);
        tg.src_offs  = gguf_get_data_offset(ref_gguf) + gguf_get_tensor_offset(ref_gguf, id);
        tg.src_type  = rt->type;
        tg.type0     = t->type;
        memcpy(tg.nb0, t->nb, sizeof(tg.nb0));
        tg.buffer0   = t->buffer;
        tg.data0     = t->data;
        tg.extra0    = t->extra;
        tg.rot0      = model->is_rotated(t);
        if (!imatrix.empty()) {
            const auto it = imatrix.find(name);
            if (it == imatrix.end()) {
                no_imatrix.insert(name);
            } else if (it->second.size() != (size_t) (t->ne[0]*t->ne[2])) {
                fprintf(stderr, "imatrix size %zu is different from tensor size %" PRId64 " for %s\n",
                        it->second.size(), t->ne[0]*t->ne[2], t->name);
                return 1;
            } else {
                tg.im = it->second.data();
            }
        }
        targets.push_back(std::move(tg));
    }
    llama_quant_free(qs);
    if (targets.empty()) {
        fprintf(stderr, "no weights to perturb in the selected layers and tensors\n");
        return 1;
    }
    if (!no_imatrix.empty()) {
        fprintf(stderr, "warning: %zu of the weights have no imatrix entry: %s\n", no_imatrix.size(), join(no_imatrix).c_str());
    }
    std::stable_sort(targets.begin(), targets.end(), [](const target & a, const target & b) { return a.layer < b.layer; });

    // the groups, named by the --tensor-type pattern that selects them
    struct group {
        std::string pattern;
        std::vector<size_t> idx;
        std::set<std::string> names;
        std::set<std::string> types0;
        int64_t n_params = 0;
    };
    std::vector<group> groups;
    std::unordered_map<std::string, size_t> group_of;
    for (size_t i = 0; i < targets.size(); ++i) {
        const target & tg = targets[i];
        std::string pattern = "^" + regex_escape(tg.label) + "\\.weight$";
        if (tg.layer >= 0 && group_by == "layer") {
            pattern = "^blk\\." + std::to_string(tg.layer) + "\\.";
        } else if (tg.layer >= 0 && group_by == "kind") {
            pattern = "^blk\\.\\d+\\." + regex_escape(tg.kind) + "\\.weight$";
        }
        const auto [it, fresh] = group_of.emplace(pattern, groups.size());
        if (fresh) {
            groups.push_back({ pattern, {}, {}, {}, 0 });
        }
        group & g = groups[it->second];
        g.idx.push_back(i);
        g.types0.insert(type_label(tg.rot0 ? ggml_get_rotated_type(tg.type0) : tg.type0));
        if (g.names.insert(tg.t->name).second) {
            g.n_params += ggml_nelements(tg.t);
        }
    }

    const auto now  = [] { return std::chrono::steady_clock::now(); };
    const auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    const auto t_start = now();

    // the unperturbed model, once
    window_stats base;
    if (!score(ctx, cache, tokens, n_ctx, n_chunks, n_vocab, nth, base, n_par)) {
        return 1;
    }
    const double t_eval = secs(t_start, now());
    const bool mismatch = ref_path == model_path && base.mean_kl() > 1e-5;
    if (mismatch) {
        fprintf(stderr, "warning: the unperturbed model differs from its own reference (KL %.2e): the changes come out "
                        "smaller than against a matching reference - use the same -ngl and -perturbngl\n", base.mean_kl());
    }
    fprintf(stderr, "one evaluation takes %.1f s; %zu weights or groups x %zu types: at least %.0f min\n",
            t_eval, groups.size(), types.size(), groups.size()*types.size()*t_eval/60);

    const char * rname = strrchr(ref_path.c_str(), '/');
    const char * mname = strrchr(model_path.c_str(), '/');
    printf("# reference %s, perturbing %s by %s\n", rname ? rname + 1 : ref_path.c_str(), mname ? mname + 1 : model_path.c_str(),
           group_by.c_str());
    printf("# %d chunks of %d tokens, %d scored per chunk, %d per batch; -ngl %d, -perturbngl %d; %s activations%s%s%s\n", n_chunks, n_ctx,
           scored_per_chunk(n_ctx), n_par, n_gpu_layers, perturb_ngl, activations.c_str(), imatrix_path.empty() ? "" : "; imatrix ", imatrix_path.c_str(),
           !imatrix_path.empty() && qz.gptq ? " (GPTQ for HQ)" : "");
    printf("# unperturbed model: KL %.6f, top-1 %.3f %%%s\n", base.mean_kl(), base.top1_pct(),
           mismatch ? " - differs from its own reference, so the changes are too small" : "");
    printf("# pattern=[(type, KL change, top-1 change in percentage points), ...]  # the weight's type in the model\n");
    fflush(stdout);

    struct result {
        bool   done = false;
        double dkl = 0, dtop1 = 0;
    };
    std::vector<std::vector<result>> res(groups.size(), std::vector<result>(types.size()));
    for (size_t ig = 0; ig < groups.size(); ++ig) {
        const group & g = groups[ig];
        fprintf(stderr, "[%zu/%zu] %s:", ig + 1, groups.size(), g.pattern.c_str());
        std::string line = g.pattern + "=[";

        // all types in one preparation step if their variants fit in the budget, else one at a time
        size_t need = 0;
        for (size_t i : g.idx) {
            targets[i].var.assign(types.size(), {});
            for (ggml_type type : types) {
                need += targets[i].applies(type) ? variant_bytes(targets[i].t, targets[i].resolve(type)) : 0;
            }
        }
        std::vector<std::vector<size_t>> steps;
        for (size_t k = 0; k < types.size(); ++k) {
            bool applies = false;
            for (size_t i : g.idx) {
                applies |= targets[i].applies(types[k]);
            }
            if (!applies) {
                continue;
            }
            if (steps.empty() || need > variants_budget) {
                steps.emplace_back();
            }
            steps.back().push_back(k);
        }

        for (const auto & step : steps) {
            const auto t0 = now();
            bool ok = true;
            for (size_t i : g.idx) {
                ok = ok && prepare(qz, types, step, targets[i]);
            }
            // the GPTQ factors (up to 3 GB to build) are rarely reused across weights, and weights in RAM
            // evaluate fast only while they all fit in the page cache
            qz.cache = llama_gptq_cache((size_t) 8 << 30);
            malloc_trim(0);
            if (!ok) {
                return 1;
            }
            fprintf(stderr, " prepare %.1f s,", secs(t0, now()));

            for (size_t k : step) {
                const auto t1 = now();
                for (size_t i : g.idx) {
                    if (!install(model, types, k, targets[i])) {
                        return 1;
                    }
                }
                replan_if_needed(ctx);
                window_stats st;
                if (!score(ctx, cache, tokens, n_ctx, n_chunks, n_vocab, nth, st, n_par)) {
                    return 1;
                }
                for (size_t i : g.idx) {
                    targets[i].var[k] = {};
                }
                result & r = res[ig][k];
                r = { true, st.mean_kl() - base.mean_kl(), st.top1_pct() - base.top1_pct() };
                char buf[128];
                snprintf(buf, sizeof(buf), "%s(%s, %.3e, %+.3f)", line.back() == '[' ? "" : ", ", type_label(types[k]).c_str(),
                         r.dkl, r.dtop1);
                line += buf;
                fprintf(stderr, " %s %.1f s", type_label(types[k]).c_str(), secs(t1, now()));
            }
        }
        for (size_t i : g.idx) {
            restore(model, targets[i]);
        }

        printf("%s]  # %s\n", line.c_str(), join(g.types0).c_str());
        fflush(stdout);
        const double elapsed = secs(t_start, now());
        fprintf(stderr, " | %.0f min left\n", elapsed/(ig + 1)*(groups.size() - ig - 1)/60);
    }

    size_t w = 7;
    for (const group & g : groups) {
        w = std::max(w, g.pattern.size());
    }
    for (size_t it = 0; it < types.size(); ++it) {
        std::vector<size_t> order;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            if (res[ig][it].done) {
                order.push_back(ig);
            }
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return res[a][it].dkl > res[b][it].dkl; });

        printf("#\n# ranking for %s, largest KL change first\n", type_label(types[it]).c_str());
        printf("# %4s  %-*s %10s %8s %8s  %s\n", "rank", (int) w, "pattern", "KL change", "top-1", "Mparams", "now");
        for (size_t r = 0; r < order.size(); ++r) {
            const group &  g  = groups[order[r]];
            const result & rs = res[order[r]][it];
            printf("# %4zu  %-*s %10.3e %+8.3f %8.1f  %s\n", r + 1, (int) w, g.pattern.c_str(), rs.dkl, rs.dtop1, g.n_params/1e6,
                   join(g.types0).c_str());
        }
    }

    fclose(cache);
    llama_free(ctx);
    llama_model_free(model);
    gguf_free(ref_gguf);
    ggml_free(ref_meta);
    close(g_ref_fd);
    return 0;
}
