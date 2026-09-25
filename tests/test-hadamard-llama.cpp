// Loading, saving, device policy and graph tests for Hadamard-rotated (HQ) models.
//
// usage: test-hadamard-llama <workdir> <model.gguf>...
//
// Each model is a small non-HQ quantization. The test relabels its eligible tensors to their HQ
// types and adds hadamard.seed; the data isn't really rotated, which doesn't matter here.

#include "llama.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "gguf.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static int g_failed = 0;

static void check(bool ok, const std::string & name) {
    printf("  %-84s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    fflush(stdout);
    if (!ok) {
        g_failed++;
    }
}

static std::string g_log;

static void log_cb(ggml_log_level level, const char * text, void * ud) {
    (void) ud;
    if (level >= GGML_LOG_LEVEL_WARN) {
        g_log += text;
    }
}

static const uint64_t SEED = 0x1234abcdull;

static bool g_rotate_embd = false;

static bool eligible(const char * name, ggml_type type, int64_t ne0) {
    return ggml_get_rotated_type(type) != type && ggml_rht_plan(ne0, nullptr, nullptr) &&
           (g_rotate_embd || strcmp(name, "token_embd.weight") != 0);
}

// copies src to dst with every eligible tensor relabelled to its HQ type
static std::set<std::string> make_hq_file(const std::string & src, const std::string & dst, bool with_seed,
                                          std::map<std::string, ggml_type> * hq_types = nullptr) {
    ggml_context * ctx = nullptr;
    gguf_context * in = gguf_init_from_file(src.c_str(), { false, &ctx });
    if (!in) {
        throw std::runtime_error("failed to read " + src);
    }

    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    if (with_seed) {
        gguf_set_val_u64(out, "hadamard.seed", SEED);
    }

    std::set<std::string> relabelled;
    for (int64_t i = 0; i < gguf_get_n_tensors(in); ++i) {
        ggml_tensor * t = ggml_get_tensor(ctx, gguf_get_tensor_name(in, i));
        gguf_add_tensor(out, t);
        if (eligible(t->name, t->type, t->ne[0])) {
            gguf_set_tensor_type(out, t->name, ggml_get_rotated_type(t->type));
            relabelled.insert(t->name);
            if (hq_types) {
                (*hq_types)[t->name] = ggml_get_rotated_type(t->type);
            }
        }
    }
    gguf_write_to_file(out, dst.c_str(), false);
    gguf_free(out);
    gguf_free(in);
    ggml_free(ctx);
    return relabelled;
}

static llama_model * load(const std::string & path, bool extra_bufts = true, ggml_backend_dev_t dev = nullptr) {
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    mp.devices         = devs;
    mp.n_gpu_layers    = dev ? 99 : 0;
    mp.use_extra_bufts = extra_bufts;
    return llama_model_load_from_file(path.c_str(), mp);
}

struct eval_record {
    int n_rht = 0;
    int n_hadamard_inputs = 0;
    int n_lora_on_rht = 0;
    int n_lora_mm = 0;
};

static const ggml_tensor * strip_views(const ggml_tensor * t) {
    while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
        t = t->src[0];
    }
    return t;
}

static bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * rec = (eval_record *) ud;
    if (!ask) {
        return true;
    }
    if (t->op == GGML_OP_RHT) {
        rec->n_rht++;
    }
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (t->src[j] && strncmp(t->src[j]->name, "hadamard_rot", 12) == 0) {
            rec->n_hadamard_inputs++;
        }
    }
    if (t->op == GGML_OP_MUL_MAT && strstr(t->src[0]->name, ".lora_a")) {
        rec->n_lora_mm++;
        if (strip_views(t->src[1])->op == GGML_OP_RHT) {
            rec->n_lora_on_rht++;
        }
    }
    return false;
}

static std::vector<float> run_logits(llama_model * model, eval_record * rec = nullptr, llama_adapter_lora * lora = nullptr) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 128;
    cp.n_batch   = 32;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    if (rec) {
        cp.cb_eval           = eval_cb;
        cp.cb_eval_user_data = rec;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return {};
    }
    if (lora) {
        float scale = 1.0f;
        llama_set_adapters_lora(ctx, &lora, 1, &scale);
    }

    std::vector<llama_token> toks = { 785, 6722, 315, 9625, 374, 12095, 13, 576 };
    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    const int ret = llama_decode(ctx, batch);

    std::vector<float> logits;
    if (ret == 0) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * l = llama_get_logits_ith(ctx, -1);
        logits.assign(l, l + n_vocab);
    }
    llama_free(ctx);
    return logits;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return INFINITY;
    }
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double) (a[i] - b[i])*(a[i] - b[i]);
        den += (double) a[i]*a[i];
    }
    return num/den;
}

static std::vector<uint8_t> tensor_bytes(const ggml_tensor * t) {
    std::vector<uint8_t> v(ggml_nbytes(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size());
    return v;
}

static std::map<std::string, const ggml_tensor *> tensor_map(const llama_model * m) {
    std::map<std::string, const ggml_tensor *> r;
    for (const auto & [name, t] : m->tensors_by_name) {
        r[name] = t;
    }
    return r;
}

// a synthetic rank-4 adapter on the first layer's attn_q and ffn_down
static void make_lora(const llama_model * model, const std::string & path) {
    const auto tm = tensor_map(model);
    ggml_context * ctx = ggml_init({ 64*1024*1024, nullptr, false });
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.type", "adapter");
    gguf_set_val_str(g, "general.architecture", model->arch_name().c_str());
    gguf_set_val_str(g, "adapter.type", "lora");
    gguf_set_val_f32(g, "adapter.lora.alpha", 4.0f);

    uint32_t state = 1;
    auto rnd = [&]() { state = state*1664525u + 1013904223u; return ((state >> 8) & 0xffff)/65536.0f - 0.5f; };

    for (const char * name : { "blk.0.attn_q.weight", "blk.0.ffn_down.weight" }) {
        const ggml_tensor * w = tm.at(name);
        const int64_t r = 4;
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w->ne[0], r);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, w->ne[1]);
        ggml_set_name(a, (std::string(name) + ".lora_a").c_str());
        ggml_set_name(b, (std::string(name) + ".lora_b").c_str());
        for (int64_t i = 0; i < ggml_nelements(a); ++i) ((float *) a->data)[i] = 0.05f*rnd();
        for (int64_t i = 0; i < ggml_nelements(b); ++i) ((float *) b->data)[i] = 0.05f*rnd();
        gguf_add_tensor(g, a);
        gguf_add_tensor(g, b);
    }
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
}

static int expected_rhts(const llama_model * m) {
    int n = 0;
    for (const auto & l : m->layers) {
        n += m->is_rotated(l.wq) || m->is_rotated(l.wk) || m->is_rotated(l.wv) || m->is_rotated(l.wqkv);
        n += m->is_rotated(l.wo);
        n += m->is_rotated(l.ffn_gate) || m->is_rotated(l.ffn_up);
        n += m->is_rotated(l.ffn_down);
    }
    n += m->output && m->is_rotated(m->output);
    return n;
}

static bool guard_throws(const llama_model * m, const std::function<ggml_tensor * (ggml_context *)> & build) {
    ggml_context * ctx = ggml_init({ 16*1024*1024, nullptr, true });
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, build(ctx));
    bool threw = false;
    try {
        llm_graph_check_hadamard_rotation(gf, *m);
    } catch (const std::exception &) {
        threw = true;
    }
    ggml_free(ctx);
    return threw;
}

static void test_guard(const llama_model * m) {
    printf("guard:\n");

    ggml_tensor * w  = (ggml_tensor *) m->layers[0].wq;
    ggml_tensor * w2 = (ggml_tensor *) m->layers[0].ffn_up;
    const int64_t n = w->ne[0];
    const uint64_t seed = m->hadamard_seed;

    auto x = [&](ggml_context * c) { return ggml_new_tensor_2d(c, GGML_TYPE_F32, n, 3); };

    check(!guard_throws(m, [&](ggml_context * c) { return ggml_mul_mat(c, w, ggml_rht(c, x(c), seed)); }),
          "accepts MUL_MAT(rotated, RHT(x))");
    check(!guard_throws(m, [&](ggml_context * c) {
              ggml_tensor * v = ggml_view_2d(c, w, n, 8, w->nb[1], 0);
              return ggml_mul_mat(c, v, ggml_reshape_2d(c, ggml_rht(c, x(c), seed), n, 3)); }),
          "accepts a view of a rotated weight with a reshaped RHT input");
    check(guard_throws(m, [&](ggml_context * c) { return ggml_mul_mat(c, w, x(c)); }),
          "refuses an unrotated input");
    check(guard_throws(m, [&](ggml_context * c) { return ggml_mul_mat(c, w, ggml_rht(c, x(c), seed + 1)); }),
          "refuses an RHT input with the wrong seed");
    check(guard_throws(m, [&](ggml_context * c) {
              ggml_tensor * y = ggml_rht(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 2*n, 3), seed);
              return ggml_mul_mat(c, w, ggml_reshape_2d(c, y, n, 6)); }),
          "refuses an RHT input of the wrong width");
    check(guard_throws(m, [&](ggml_context * c) {
              return ggml_get_rows(c, w, ggml_new_tensor_1d(c, GGML_TYPE_I32, 2)); }),
          "refuses GET_ROWS on a rotated weight");
    check(guard_throws(m, [&](ggml_context * c) { return ggml_add(c, ggml_new_tensor(c, GGML_TYPE_F32, 2, w->ne), w); }),
          "refuses ADD with a rotated weight");
    check(guard_throws(m, [&](ggml_context * c) {
              return ggml_mul_mat(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, n, 4), ggml_view_2d(c, w, n, 2, w->nb[1], 0)); }),
          "refuses a rotated weight as src1");
    check(guard_throws(m, [&](ggml_context * c) {
              return ggml_cpy(c, ggml_new_tensor(c, w->type, 2, w->ne), w); }),
          "refuses CPY into a rotated weight");
    check(guard_throws(m, [&](ggml_context * c) {
              return ggml_set_rows(c, w, ggml_new_tensor_2d(c, GGML_TYPE_F32, n, 2), ggml_new_tensor_1d(c, GGML_TYPE_I64, 2)); }),
          "refuses SET_ROWS into a rotated weight");
    check(guard_throws(m, [&](ggml_context * c) {
              ggml_tensor * v = ggml_view_2d(c, w2, w2->ne[0], 1, w2->nb[1], 0);
              return ggml_scale_inplace(c, v, 2.0f); }),
          "refuses an in-place op on a view of a rotated weight");
    check(guard_throws(m, [&](ggml_context * c) {
              ggml_tensor * v = ggml_view_2d(c, w, n, 8, w->nb[1], 0);
              return ggml_add(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, n, 8), v); }),
          "refuses ADD on a view of a rotated weight");
}

static void test_model(const std::string & dir, const std::string & base_path) {
    printf("\n%s:\n", base_path.c_str());

    const std::string hq_path = dir + "/test-hq.gguf";
    std::map<std::string, ggml_type> hq_types;
    const std::set<std::string> relabelled = make_hq_file(base_path, hq_path, true, &hq_types);
    check(!relabelled.empty(), "relabelled " + std::to_string(relabelled.size()) + " tensors to HQ types");

    // the CPU repack buffer can't be read back, so bytes are compared on loads without it
    llama_model * base      = load(base_path, true);
    llama_model * base_flat = load(base_path, false);
    llama_model * hq        = load(hq_path, true);
    llama_model * hq_flat   = load(hq_path, false);
    if (!base || !base_flat || !hq || !hq_flat) {
        check(false, "load");
        return;
    }

    printf("demotion:\n");
    {
        const auto tb = tensor_map(base_flat);
        const auto tbr = tensor_map(base);
        const auto th = tensor_map(hq_flat);
        const auto tr = tensor_map(hq);
        bool types_ok = true, bytes_ok = true, set_ok = true, buft_ok = true;
        for (const auto & [name, t] : th) {
            const ggml_tensor * b = tb.at(name);
            types_ok &= t->type == b->type && memcmp(t->nb, b->nb, sizeof(t->nb)) == 0 && !ggml_is_rotated(t->type);
            set_ok   &= hq_flat->is_rotated(t) == (relabelled.count(name) > 0);
            if (relabelled.count(name)) {
                bytes_ok &= tensor_bytes(t) == tensor_bytes(b);
                buft_ok  &= strcmp(ggml_backend_buffer_name(tr.at(name)->buffer), ggml_backend_buffer_name(tbr.at(name)->buffer)) == 0;
            }
        }
        check(types_ok, "HQ tensors load with their base type and the same nb[]");
        check(bytes_ok, "... and the same bytes");
        check(set_ok, "is_rotated holds for exactly the HQ tensors");

        ggml_context * ctx = ggml_init({ 1024*1024, nullptr, true });
        ggml_tensor * t = (ggml_tensor *) th.at(*relabelled.begin());
        ggml_tensor * v = ggml_view_1d(ctx, ggml_view_1d(ctx, t, 256, 0), 256, 0);
        check(hq_flat->is_rotated(v), "is_rotated follows views");
        ggml_free(ctx);

        check(buft_ok, "HQ weights get the same buffer types as their base types (CPU repack)");

        std::vector<std::string> splits;
        llama_model_loader ml(nullptr, nullptr, nullptr, hq_path, splits, nullptr, LLAMA_LOAD_MODE_MMAP,
                              false, false, true, nullptr, nullptr);
        bool meta_ok = ml.hadamard_seed == SEED;
        for (const auto & [name, type] : hq_types) {
            meta_ok &= ml.weights_map.at(name).tensor->type == type;
        }
        check(meta_ok, "the loader's meta tensors keep the HQ types, and it reads hadamard.seed");
    }

    // the repacked kernels differ numerically from the plain ones, so the bound comes from the base model
    const std::vector<float> logits      = run_logits(hq);
    const std::vector<float> logits_flat = run_logits(hq_flat);
    const double err_hq   = nmse(logits, logits_flat);
    const double err_base = nmse(run_logits(base), run_logits(base_flat));
    printf("  (nmse repack vs plain: HQ %.3g, base %.3g)\n", err_hq, err_base);
    check(!logits.empty() && err_hq < std::max(1e-5, 10*err_base), "logits with and without the CPU repack match as closely as the base model's");

    printf("save round trip:\n");
    {
        const std::string saved = dir + "/test-hq-saved.gguf";
        llama_model_save_to_file(hq_flat, saved.c_str());

        gguf_context * a = gguf_init_from_file(hq_path.c_str(), { true, nullptr });
        gguf_context * s = gguf_init_from_file(saved.c_str(), { true, nullptr });
        bool types_ok = s != nullptr;
        for (int64_t i = 0; s && i < gguf_get_n_tensors(s); ++i) {
            const char * name = gguf_get_tensor_name(s, i);
            const int64_t ia = gguf_find_tensor(a, name);
            types_ok &= ia >= 0 && gguf_get_tensor_type(a, ia) == gguf_get_tensor_type(s, i);
        }
        check(types_ok, "the saved file has the source's HQ types");
        const int64_t kid = s ? gguf_find_key(s, "hadamard.seed") : -1;
        check(kid >= 0 && gguf_get_val_u64(s, kid) == SEED, "the saved file has the same hadamard.seed");
        if (s) gguf_free(s);
        gguf_free(a);

        llama_model * reloaded = load(saved, false);
        bool bytes_ok = reloaded != nullptr;
        if (reloaded) {
            const auto t0 = tensor_map(hq_flat);
            for (const auto & [name, t] : tensor_map(reloaded)) {
                bytes_ok &= t0.count(name) && tensor_bytes(t) == tensor_bytes(t0.at(name)) &&
                            reloaded->is_rotated(t) == hq_flat->is_rotated(t0.at(name));
            }
            check(bytes_ok, "byte-identical tensor data and the same rotated set after reloading");
            check(run_logits(reloaded) == logits_flat, "identical logits after reloading");
            llama_model_free(reloaded);
        } else {
            check(false, "reload the saved model");
        }

        const std::string saved_base = dir + "/test-base-saved.gguf";
        llama_model_save_to_file(base_flat, saved_base.c_str());
        gguf_context * sb = gguf_init_from_file(saved_base.c_str(), { true, nullptr });
        check(sb && gguf_find_key(sb, "hadamard.seed") < 0, "a non-HQ model saves without hadamard.seed");
        if (sb) gguf_free(sb);
    }

    printf("graph:\n");
    {
        eval_record rec;
        run_logits(hq, &rec);
        const int want = expected_rhts(hq);
        check(rec.n_rht == want, "one RHT per distinct rotated activation (" + std::to_string(rec.n_rht) + " == " + std::to_string(want) + ")");
        check(rec.n_hadamard_inputs == 0, "no Hadamard matrix graph inputs");

        const std::string lora_path = dir + "/test-lora.gguf";
        make_lora(hq, lora_path);
        llama_adapter_lora * lora = llama_adapter_lora_init(hq, lora_path.c_str());
        eval_record rec_lora;
        run_logits(hq, &rec_lora, lora);
        check(lora && rec_lora.n_lora_mm == 2 && rec_lora.n_lora_on_rht == 0, "the LoRA branch multiplies the unrotated input");
        check(rec_lora.n_rht == want, "the LoRA branch adds no rotations");
    }

    test_guard(hq);

    printf("refusals:\n");
    {
        const std::string noseed = dir + "/test-hq-noseed.gguf";
        make_hq_file(base_path, noseed, false);
        g_log.clear();
        llama_model * m = load(noseed);
        check(m == nullptr && g_log.find("ConvRot") != std::string::npos, "a file with HQ types and no hadamard.seed is refused, with the ConvRot hint");
        if (m) llama_model_free(m);

        const std::string embd = dir + "/test-hq-embd.gguf";
        g_rotate_embd = true;
        make_hq_file(base_path, embd, true);
        g_rotate_embd = false;
        m = load(embd);
        g_log.clear();
        const bool ran = m && !run_logits(m).empty();
        check(m && !ran && g_log.find("Hadamard-rotated weight token_embd.weight is used by") != std::string::npos,
              "a rotated token_embd is refused by the guard (GET_ROWS)");
        if (m) llama_model_free(m);
    }

    llama_model_free(base);
    llama_model_free(base_flat);
    llama_model_free(hq);
    llama_model_free(hq_flat);
}

// the device policy: a CPU-only RPC server stands in for any backend without GGML_OP_RHT
static void test_rpc(const std::string & base_path, const std::string & hq_path) {
    printf("\ndevice policy (RPC):\n");

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    const char * endpoint = "127.0.0.1:50931";
    std::thread([=]() {
        ggml_backend_dev_t devs[1] = { cpu };
        ggml_backend_rpc_start_server(endpoint, nullptr, 2, 1, devs);
    }).detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    ggml_backend_reg_t reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_dev_t rpc = reg && ggml_backend_reg_dev_count(reg) > 0 ? ggml_backend_reg_dev_get(reg, 0) : nullptr;
    if (!rpc) {
        check(false, "connect to the local RPC server");
        return;
    }

    llama_model * m = load(base_path, true, rpc);
    check(m != nullptr, "a non-HQ model loads with an RPC device");
    if (m) llama_model_free(m);

    g_log.clear();
    m = load(hq_path, true, rpc);
    check(m == nullptr && g_log.find("only supported on the CPU and the CUDA backend") != std::string::npos,
          "an HQ model with an RPC device in the device list is refused");
    if (m) llama_model_free(m);
}

// the relabelled files aren't really rotated, which makes their logits chaotic, so the CUDA
// comparison uses a real HQ model requantized from the base file
static void test_cuda(const std::string & dir, const std::string & base_path) {
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && !gpu; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        ggml_backend_reg_t r = ggml_backend_dev_backend_reg(d);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && r && std::string(ggml_backend_reg_name(r)) == "CUDA") {
            gpu = d;
        }
    }
    if (!gpu) {
        printf("\n(no CUDA device - skipping the CUDA load)\n");
        return;
    }

    printf("\ndevice policy (CUDA):\n");
    const std::string hq_path = dir + "/test-hq-real.gguf";
    llama_model_quantize_params qp = llama_model_quantize_default_params();
    qp.ftype            = LLAMA_FTYPE_MOSTLY_Q4_K_M;
    qp.allow_requantize = true;
    qp.hadamard         = true;
    qp.pure             = true;
    check(llama_model_quantize(base_path.c_str(), hq_path.c_str(), &qp) == 0, "requantize the base model with --hadamard");

    llama_model * m  = load(hq_path, true, gpu);
    llama_model * mc = load(hq_path, true);
    llama_model * b  = load(base_path, true, gpu);
    llama_model * bc = load(base_path, true);
    check(m != nullptr, "an HQ model loads on CUDA");
    if (m && mc && b && bc) {
        const double err_hq   = nmse(run_logits(m), run_logits(mc));
        const double err_base = nmse(run_logits(b), run_logits(bc));
        printf("  (nmse CUDA vs CPU: HQ %.3g, base %.3g)\n", err_hq, err_base);
        check(err_hq < std::max(1e-5, 10*err_base), "HQ logits on CUDA match the CPU as closely as the base model's");
    }
    for (llama_model * x : { m, mc, b, bc }) {
        if (x) llama_model_free(x);
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <workdir> <model.gguf>...\n", argv[0]);
        return 1;
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    const std::string dir = argv[1];
    for (int i = 2; i < argc; ++i) {
        test_model(dir, argv[i]);
    }

    const std::string hq_path = dir + "/test-hq.gguf";
    make_hq_file(argv[2], hq_path, true);
    test_rpc(argv[2], hq_path);
    test_cuda(dir, argv[2]);

    llama_backend_free();

    printf("\n%s: %d failure(s)\n", g_failed ? "FAIL" : "PASS", g_failed);
    return g_failed ? 1 : 0;
}
