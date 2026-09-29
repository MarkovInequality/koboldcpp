// Quantizer tests for the Hadamard-rotated (HQ) types, through llama_model_quantize.
//
// usage: test-hadamard-quantize <workdir> <model-bf16.gguf>
//
// The source is pruned to two layers first, so each quantization takes a few seconds.

#include "llama.h"
#include "llama-quant-gptq.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

static int g_failed = 0;

static void check(bool ok, const std::string & name) {
    printf("  %-88s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    fflush(stdout);
    if (!ok) {
        g_failed++;
    }
}

static std::string g_log;

static void log_cb(ggml_log_level level, const char * text, void * ud) {
    (void) ud;
    if (level >= GGML_LOG_LEVEL_INFO) {
        g_log += text;
    }
}

static std::string g_dir;
static std::string path(const std::string & name) { return g_dir + "/" + name; }

static int quantize(const std::string & src, const std::string & dst, llama_ftype ftype,
                    const std::function<void(llama_model_quantize_params &)> & opts = nullptr) {
    llama_model_quantize_params p = llama_model_quantize_default_params();
    p.ftype            = ftype;
    p.nthread          = 8;
    p.allow_requantize = true;
    if (opts) {
        opts(p);
    }
    g_log.clear();
    return (int) llama_model_quantize(src.c_str(), dst.c_str(), &p);
}

struct gguf_file {
    gguf_context * g   = nullptr;
    ggml_context * ctx = nullptr;

    explicit gguf_file(const std::string & p) { g = gguf_init_from_file(p.c_str(), { false, &ctx }); }
    ~gguf_file() { if (g) gguf_free(g); if (ctx) ggml_free(ctx); }

    ggml_tensor * t(const std::string & name) const { return ggml_get_tensor(ctx, name.c_str()); }

    bool has_key(const char * key) const { return gguf_find_key(g, key) >= 0; }

    float f32(const char * key) const {
        const int64_t k = gguf_find_key(g, key);
        return k < 0 ? -1.0f : gguf_get_val_f32(g, k);
    }

    uint64_t seed() const {
        const int64_t k = gguf_find_key(g, "hadamard.seed");
        return k < 0 ? 0 : gguf_get_val_u64(g, k);
    }

    std::string str(const char * key) const {
        const int64_t k = gguf_find_key(g, key);
        return k < 0 ? "" : gguf_get_val_str(g, k);
    }
};

static std::vector<float> to_f32(const ggml_tensor * t, int64_t i0 = 0, int64_t n = -1) {
    const int64_t ne0 = t->ne[0];
    if (n < 0) {
        n = ggml_nelements(t) - i0;
    }
    std::vector<float> out(n);
    const char * data = (const char *) t->data + (i0/ne0)*ggml_row_size(t->type, ne0);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), data, n*sizeof(float));
    } else {
        ggml_get_type_traits(t->type)->to_float(data, out.data(), n);
    }
    return out;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double) (a[i] - b[i])*(a[i] - b[i]);
        den += (double) b[i]*b[i];
    }
    return num/den;
}

static bool same_bytes(const ggml_tensor * a, const ggml_tensor * b) {
    return a && b && a->type == b->type && ggml_nbytes(a) == ggml_nbytes(b) && memcmp(a->data, b->data, ggml_nbytes(a)) == 0;
}

static bool files_identical(const std::string & a, const std::string & b) {
    FILE * fa = fopen(a.c_str(), "rb");
    FILE * fb = fopen(b.c_str(), "rb");
    bool same = fa && fb;
    std::vector<char> ba(1 << 20), bb(1 << 20);
    while (same) {
        const size_t na = fread(ba.data(), 1, ba.size(), fa);
        const size_t nb = fread(bb.data(), 1, bb.size(), fb);
        same = na == nb && memcmp(ba.data(), bb.data(), na) == 0;
        if (na == 0) {
            break;
        }
    }
    if (fa) fclose(fa);
    if (fb) fclose(fb);
    return same;
}

static size_t log_count(const std::string & needle) {
    size_t n = 0;
    for (size_t pos = 0; (pos = g_log.find(needle, pos)) != std::string::npos; ++pos) n++;
    return n;
}

// a synthetic imatrix entry per expert: a spread-out floor with two loud channels, as with massive activations
static std::vector<float> synth_v(int64_t n, int64_t nexp, uint32_t salt) {
    std::vector<float> v(n*nexp);
    for (size_t j = 0; j < v.size(); ++j) v[j] = 0.1f + (float) (((j + salt)*2654435761u) % 1000)/100.0f;
    for (int64_t e = 0; e < nexp; ++e) {
        v[e*n + 3] = 2000.0f;
        v[e*n + n/2 + 7*e + 1] = 2000.0f;
    }
    return v;
}

struct imatrix_set {
    std::vector<std::string> names;
    std::vector<std::vector<float>> vals;
    std::vector<llama_model_imatrix_data> data;

    void add(const std::string & name, std::vector<float> v) { names.push_back(name); vals.push_back(std::move(v)); }

    const std::vector<float> & at(const std::string & name) const {
        for (size_t i = 0; i < names.size(); ++i) if (names[i] == name) return vals[i];
        static const std::vector<float> none;
        return none;
    }

    const llama_model_imatrix_data * get() {
        data.clear();
        for (size_t i = 0; i < names.size(); ++i) data.push_back({ names[i].c_str(), vals[i].data(), vals[i].size() });
        data.push_back({ nullptr, nullptr, 0 });
        return data.data();
    }
};

// the in-sample output error of rotated rows wq against w under H = R*diag(v)*R^T: sum |diag(sqrt(v)) R^T (w - wq)|^2 / sum |diag(sqrt(v)) R^T w|^2
static double out_err(const std::vector<float> & w, const std::vector<float> & wq, int64_t n, const std::vector<float> & v, uint64_t seed) {
    double num = 0, den = 0;
    std::vector<double> d(n), a(n);
    for (size_t r = 0; r*n < w.size(); ++r) {
        for (int64_t k = 0; k < n; ++k) {
            d[k] = (double) w[r*n + k] - wq[r*n + k];
            a[k] = w[r*n + k];
        }
        ggml_rht_inv_f64(d.data(), n, seed);
        ggml_rht_inv_f64(a.data(), n, seed);
        for (int64_t k = 0; k < n; ++k) {
            num += v[k]*d[k]*d[k];
            den += v[k]*a[k]*a[k];
        }
    }
    return num/den;
}

// R*w for every row of the first nrows rows
static std::vector<float> rotate_rows(std::vector<float> w, int64_t n, uint64_t seed) {
    for (size_t r = 0; r*n < w.size(); ++r) {
        ggml_rht_ref(w.data() + r*n, n, seed);
    }
    return w;
}

// relabels every eligible tensor of src to its HQ type; the data isn't rotated
static void relabel(const std::string & src, const std::string & dst, bool with_seed) {
    gguf_file in(src);
    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in.g);
    gguf_remove_key(out, "hadamard.seed");
    if (with_seed) {
        gguf_set_val_u64(out, "hadamard.seed", 5);
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(in.g); ++i) {
        ggml_tensor * t = in.t(gguf_get_tensor_name(in.g, i));
        gguf_add_tensor(out, t);
        if (ggml_get_rotated_type(t->type) != t->type && strcmp(t->name, "token_embd.weight") != 0 &&
                ggml_rht_plan(t->ne[0], nullptr, nullptr)) {
            gguf_set_tensor_type(out, t->name, ggml_get_rotated_type(t->type));
        }
    }
    gguf_write_to_file(out, dst.c_str(), false);
    gguf_free(out);
}

// the rank-one part of make_hessian's Grams: u u^T with u of the size of the diagonal's typical entry
static std::vector<float> gram_u(const std::vector<float> & v) {
    double mean = 0;
    for (float x : v) mean += x;
    mean /= v.size();
    std::vector<float> u(v.size());
    for (size_t k = 0; k < u.size(); ++k) u[k] = (float) (std::sqrt(mean)*std::sin(0.37*k + 1.0));
    return u;
}

// a hessian-collect file for the 2D weights of im: G = diag(v) + u u^T, each weight the owner of its Gram
static void make_hessian(const gguf_file & model, const imatrix_set & im, const std::string & dst) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.type", "imatrix");
    const char * ds = "synthetic";
    gguf_set_arr_str(g, "imatrix.datasets", &ds, 1);
    gguf_set_val_bool(g, "hessian.complete", true);
    size_t mem = 0;
    for (const auto & name : im.names) {
        const int64_t n = model.t(name)->ne[0];
        mem += 3*ggml_tensor_overhead() + (size_t) (n*n + n + 1)*sizeof(float) + 3*GGML_MEM_ALIGN;
    }
    ggml_context * ctx = ggml_init({ mem, nullptr, false });
    for (const auto & name : im.names) {
        if (model.t(name)->ne[2] != 1) continue;
        const std::vector<float> & v = im.at(name);
        const std::vector<float> u = gram_u(v);
        const int64_t n = (int64_t) v.size();
        ggml_tensor * gr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
        ggml_tensor * s2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
        ggml_tensor * ct = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                ((float *) gr->data)[i*n + j] = (i == j ? v[i] : 0.0f) + u[i]*u[j];
            }
            ((float *) s2->data)[i] = ((float *) gr->data)[i*n + i];
        }
        ((float *) ct->data)[0] = 1.0f;
        ggml_set_name(gr, (name + ".in_gram").c_str());
        ggml_set_name(s2, (name + ".in_sum2").c_str());
        ggml_set_name(ct, (name + ".counts").c_str());
        gguf_add_tensor(g, gr);
        gguf_add_tensor(g, s2);
        gguf_add_tensor(g, ct);
    }
    gguf_write_to_file(g, dst.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
}

// sum over the rows of d^T G d, d = R^T (w - wq) in the unrotated space, for make_hessian's G, relative to w's
static double gram_err(const std::vector<float> & w, const std::vector<float> & wq, int64_t n, const std::vector<float> & v, uint64_t seed) {
    const std::vector<float> u = gram_u(v);
    double num = 0, den = 0;
    std::vector<double> d(n), a(n);
    for (size_t r = 0; r*n < w.size(); ++r) {
        for (int64_t k = 0; k < n; ++k) {
            d[k] = (double) w[r*n + k] - wq[r*n + k];
            a[k] = w[r*n + k];
        }
        ggml_rht_inv_f64(d.data(), n, seed);
        ggml_rht_inv_f64(a.data(), n, seed);
        double ud = 0, ua = 0;
        for (int64_t k = 0; k < n; ++k) {
            num += v[k]*d[k]*d[k];
            den += v[k]*a[k]*a[k];
            ud += u[k]*d[k];
            ua += u[k]*a[k];
        }
        num += ud*ud;
        den += ua*ua;
    }
    return num/den;
}

// a rank-4 adapter; A and B are laid out as llama-adapter.cpp expects (flipped for token_embd)
static void make_lora(const gguf_file & model, const std::string & dst, const std::vector<std::string> & names,
                      const char * arch = nullptr, bool alora = false) {
    ggml_context * ctx = ggml_init({ 256*1024*1024, nullptr, false });
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.type", "adapter");
    const int64_t ka = gguf_find_key(model.g, "general.architecture");
    gguf_set_val_str(g, "general.architecture", arch ? arch : gguf_get_val_str(model.g, ka));
    gguf_set_val_str(g, "adapter.type", "lora");
    gguf_set_val_f32(g, "adapter.lora.alpha", 8.0f);
    if (alora) {
        const uint32_t toks[2] = { 1, 2 };
        gguf_set_arr_data(g, "adapter.alora.invocation_tokens", GGUF_TYPE_UINT32, toks, 2);
    }

    uint32_t state = 7;
    auto rnd = [&]() { state = state*1664525u + 1013904223u; return ((state >> 8) & 0xffff)/65536.0f - 0.5f; };

    const int64_t r = 4;
    for (const auto & name : names) {
        const ggml_tensor * w = model.t(name);
        const int64_t n_in  = w ? w->ne[0] : 1024;
        const int64_t n_out = w ? w->ne[1] : 1024;
        const bool embd = name == "token_embd.weight";
        ggml_tensor * a = embd ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, n_out) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, r);
        ggml_tensor * b = embd ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, n_in)  : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, n_out);
        ggml_set_name(a, (name + ".lora_a").c_str());
        ggml_set_name(b, (name + ".lora_b").c_str());
        for (int64_t i = 0; i < ggml_nelements(a); ++i) ((float *) a->data)[i] = 0.2f*rnd();
        for (int64_t i = 0; i < ggml_nelements(b); ++i) ((float *) b->data)[i] = 0.2f*rnd();
        gguf_add_tensor(g, a);
        gguf_add_tensor(g, b);
    }
    gguf_write_to_file(g, dst.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
}

// scale * B*A for a weight row range, from the adapter file, as llama-adapter.cpp computes it
static std::vector<float> lora_delta(const std::string & lora, const std::string & name, int64_t n_in, int64_t nrows, float scale) {
    gguf_file f(lora);
    const ggml_tensor * a = f.t(name + ".lora_a");
    const ggml_tensor * b = f.t(name + ".lora_b");
    const int64_t r = b->ne[0];
    const float s = scale*8.0f/r;
    const bool embd = name == "token_embd.weight";
    const float * A = (const float *) a->data;
    const float * B = (const float *) b->data;
    std::vector<float> d(n_in*nrows, 0.0f);
    for (int64_t o = 0; o < nrows; ++o) {
        for (int64_t i = 0; i < n_in; ++i) {
            double acc = 0;
            for (int64_t k = 0; k < r; ++k) {
                acc += embd ? (double) A[o*r + k]*B[i*r + k] : (double) B[o*r + k]*A[k*n_in + i];
            }
            d[o*n_in + i] = s*acc;
        }
    }
    return d;
}

static std::vector<float> add(std::vector<float> a, const std::vector<float> & b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
    return a;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <workdir> <model-bf16.gguf>\n", argv[0]);
        return 1;
    }
    g_dir = argv[1];
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    const std::string src = path("hqq-src.gguf");
    {
        std::vector<int32_t> prune;
        gguf_file full(argv[2]);
        const int64_t k = gguf_find_key(full.g, "qwen3.block_count");
        const int n_layer = k < 0 ? 28 : (int) gguf_get_val_u32(full.g, k);
        for (int i = 2; i < n_layer; ++i) prune.push_back(i);
        prune.push_back(-1);
        const int rc = quantize(argv[2], src, LLAMA_FTYPE_MOSTLY_BF16, [&](auto & p) { p.prune_layers = prune.data(); });
        if (rc != 0) {
            fprintf(stderr, "failed to prepare the source:\n%s", g_log.c_str());
            return 1;
        }
    }
    gguf_file srcf(src);
    const std::string q = "blk.0.attn_q.weight", fd = "blk.1.ffn_down.weight", emb = "token_embd.weight";
    const int64_t n_embd = srcf.t(q)->ne[0];

    printf("rotation:\n");
    const std::string hq = path("hqq-hq.gguf");
    check(quantize(src, hq, LLAMA_FTYPE_MOSTLY_Q4_K_M, [](auto & p) { p.hadamard = true; }) == 0, "--hadamard Q4_K_M");
    {
        gguf_file f(hq);
        check(f.t(q)->type == GGML_TYPE_HQ4_K && !ggml_is_rotated(f.t(emb)->type), "attn_q is HQ4_K, token_embd stays unrotated");
        const uint64_t seed = f.seed();
        const std::vector<float> w  = to_f32(srcf.t(q), 0, 64*n_embd);
        const std::vector<float> wq = to_f32(f.t(q), 0, 64*n_embd);
        const double err_rot   = nmse(wq, rotate_rows(w, n_embd, seed));
        const double err_plain = nmse(wq, w);
        printf("  (nmse vs R*w: %.4g, vs w: %.4g)\n", err_rot, err_plain);
        check(err_rot < 0.01 && err_plain > 0.5, "stored rows are R*w with the file's seed");
        check(f.has_key("hadamard.seed") && seed != 0, "hadamard.seed is written");
        int n_hadamard_keys = 0;
        for (int64_t i = 0; i < gguf_get_n_kv(f.g); ++i) {
            n_hadamard_keys += strncmp(gguf_get_key(f.g, i), "hadamard.", 9) == 0;
        }
        check(n_hadamard_keys == 1 && !f.has_key("general.merged_loras"), "no other hadamard.* key, no general.merged_loras without --lora");
        bool has_hq6 = false;
        for (int64_t i = 0; i < gguf_get_n_tensors(f.g); ++i) {
            has_hq6 |= f.t(gguf_get_tensor_name(f.g, i))->type == GGML_TYPE_HQ6_K;
        }
        check(has_hq6, "the mix's Q6_K tensors become HQ6_K");
    }
    const std::string hq40 = path("hqq-hq40.gguf");
    {
        check(quantize(src, hq40, LLAMA_FTYPE_MOSTLY_Q4_0, [](auto & p) { p.hadamard = true; }) == 0, "--hadamard Q4_0");
        gguf_file f(hq40);
        const std::vector<float> w  = to_f32(srcf.t(q), 0, 64*n_embd);
        const std::vector<float> wq = to_f32(f.t(q), 0, 64*n_embd);
        const double err_rot = nmse(wq, rotate_rows(w, n_embd, f.seed()));
        printf("  (HQ4_0 nmse vs R*w: %.4g)\n", err_rot);
        check(f.t(q)->type == GGML_TYPE_HQ4_0 && err_rot < 0.01 && !ggml_is_rotated(f.t(emb)->type), "attn_q is HQ4_0 and stores R*w");
    }

    printf("seed:\n");
    {
        const std::string hq2 = path("hqq-hq2.gguf"), hq_s = path("hqq-hq-seed.gguf"), hq_1t = path("hqq-hq-1t.gguf");
        quantize(src, hq2, LLAMA_FTYPE_MOSTLY_Q4_K_M, [](auto & p) { p.hadamard = true; });
        check(files_identical(hq, hq2), "two runs with the default seed give byte-identical files");
        quantize(src, hq_1t, LLAMA_FTYPE_MOSTLY_Q4_K_M, [](auto & p) { p.hadamard = true; p.nthread = 1; });
        check(files_identical(hq, hq_1t), "1 thread and 8 threads give byte-identical files");

        quantize(src, hq_s, LLAMA_FTYPE_MOSTLY_Q4_K_M, [](auto & p) { p.hadamard = true; p.hadamard_seed = 77; p.hadamard_seed_set = true; });
        gguf_file a(hq), b(hq_s);
        bool only_hq = b.seed() == 77 && a.seed() != 77;
        for (int64_t i = 0; i < gguf_get_n_tensors(a.g); ++i) {
            const ggml_tensor * ta = a.t(gguf_get_tensor_name(a.g, i));
            const ggml_tensor * tb = b.t(ta->name);
            only_hq &= ggml_is_rotated(ta->type) ? !same_bytes(ta, tb) : same_bytes(ta, tb);
        }
        bool kv_same = gguf_get_n_kv(a.g) == gguf_get_n_kv(b.g);
        for (int64_t i = 0; kv_same && i < gguf_get_n_kv(a.g); ++i) {
            kv_same &= gguf_find_key(b.g, gguf_get_key(a.g, i)) >= 0;
        }
        check(only_hq && kv_same, "--hadamard-seed changes the HQ tensors and hadamard.seed, and nothing else");

        const std::string plain = path("hqq-plain.gguf");
        quantize(src, plain, LLAMA_FTYPE_MOSTLY_Q4_K_M);
        gguf_file c(plain);
        check(!c.has_key("hadamard.seed"), "no hadamard.seed without rotated tensors");
    }

    printf("imatrix (GPTQ):\n");
    imatrix_set im_all, im_part;
    for (int64_t i = 0; i < gguf_get_n_tensors(srcf.g); ++i) {
        const ggml_tensor * t = srcf.t(gguf_get_tensor_name(srcf.g, i));
        if (ggml_n_dims(t) < 2) continue;
        std::vector<float> v = synth_v(t->ne[0], t->ne[2], (uint32_t) i);
        if (!strstr(t->name, "attn_q.weight")) im_part.add(t->name, v);
        im_all.add(t->name, std::move(v));
    }
    const std::string hq_im = path("hqq-hq-im.gguf");
    {
        quantize(src, hq_im, LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) { p.hadamard = true; p.imatrix = im_all.get(); });
        gguf_file a(hq), b(hq_im);
        size_t n_hq = 0;
        bool all_differ = true;
        for (int64_t i = 0; i < gguf_get_n_tensors(a.g); ++i) {
            const ggml_tensor * ta = a.t(gguf_get_tensor_name(a.g, i));
            if (!ggml_is_rotated(ta->type)) continue;
            n_hq++;
            all_differ &= !same_bytes(ta, b.t(ta->name)) && b.t(ta->name)->type == ta->type;
        }
        check(all_differ && n_hq > 0, "the imatrix is used (GPTQ on by default): every HQ tensor differs from the uniform one");
        check(log_count("use the imatrix through GPTQ") == 1 && g_log.find(std::to_string(n_hq) + " HQ tensors use the imatrix through GPTQ (damp 0.01)") != std::string::npos &&
              log_count("(gptq)") == n_hq, "one info line with the count, and each GPTQ tensor's line says (gptq)");
        check(b.f32("quantize.hq.gptq_damp") == 0.01f && !a.has_key("quantize.hq.gptq_damp"), "quantize.hq.gptq_damp is written only when GPTQ ran");

        const std::string hq_part = path("hqq-hq-im-part.gguf");
        quantize(src, hq_part, LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) { p.hadamard = true; p.imatrix = im_part.get(); });
        gguf_file c(hq_part);
        bool part_ok = true;
        for (int64_t i = 0; i < gguf_get_n_tensors(a.g); ++i) {
            const ggml_tensor * ta = a.t(gguf_get_tensor_name(a.g, i));
            if (!ggml_is_rotated(ta->type)) continue;
            part_ok &= strstr(ta->name, "attn_q.weight") ? same_bytes(ta, c.t(ta->name)) : same_bytes(b.t(ta->name), c.t(ta->name));
        }
        check(part_ok && log_count("2 HQ tensors have no usable imatrix entry") == 1,
              "tensors without an entry are byte-identical to the uniform ones, with one warning");

        const std::string hq_1t = path("hqq-hq-im-1t.gguf");
        quantize(src, hq_1t, LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) { p.hadamard = true; p.imatrix = im_all.get(); p.nthread = 1; });
        check(files_identical(hq_im, hq_1t), "the same seed and imatrix give identical files at 1 and 8 threads");

        const std::string hq_off = path("hqq-hq-im-off.gguf");
        quantize(src, hq_off, LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) { p.hadamard = true; p.imatrix = im_all.get(); p.hq_gptq = false; });
        gguf_file d(hq_off);
        bool off_same = true;
        for (int64_t i = 0; i < gguf_get_n_tensors(a.g); ++i) {
            const ggml_tensor * ta = a.t(gguf_get_tensor_name(a.g, i));
            if (ggml_is_rotated(ta->type)) off_same &= same_bytes(ta, d.t(ta->name));
        }
        check(off_same && log_count("the imatrix is ignored") == 1 && log_count("(gptq)") == 0 && !d.has_key("quantize.hq.gptq_damp"),
              "hq_gptq = false: the uniform bytes, one imatrix-ignored warning, no damp key");

        check(quantize(src, path("hqq-bad.gguf"), LLAMA_FTYPE_MOSTLY_Q4_K_M,
                       [&](auto & p) { p.hadamard = true; p.imatrix = im_all.get(); p.hq_gptq_damp = 0.0001f; }) != 0 &&
              g_log.find("GPTQ damp") != std::string::npos, "hq_gptq_damp = 0.0001 is refused");
        const std::string hq_d = path("hqq-hq-im-damp.gguf");
        quantize(src, hq_d, LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) { p.hadamard = true; p.imatrix = im_all.get(); p.hq_gptq_damp = 0.1f; });
        gguf_file e(hq_d);
        check(e.f32("quantize.hq.gptq_damp") == 0.1f && !same_bytes(e.t(q), b.t(q)), "hq_gptq_damp = 0.1 is used and recorded");

        const std::string hq_copy = path("hqq-hq-im-copy.gguf");
        quantize(hq_im, hq_copy, LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) { p.hadamard = true; p.imatrix = im_all.get(); });
        gguf_file f(hq_copy);
        bool copies = true;
        for (int64_t i = 0; i < gguf_get_n_tensors(b.g); ++i) {
            const ggml_tensor * tb = b.t(gguf_get_tensor_name(b.g, i));
            copies &= same_bytes(tb, f.t(tb->name));
        }
        check(copies && log_count("(gptq)") == 0, "HQ -> the same HQ type with an imatrix stays a byte copy");
        check(f.f32("quantize.hq.gptq_damp") == 0.01f, "... and keeps the source's quantize.hq.gptq_damp");

        const std::string iq = path("hqq-hq-iq2xxs.gguf");
        check(quantize(src, iq, LLAMA_FTYPE_MOSTLY_IQ2_XXS, [](auto & p) { p.hadamard = true; }) == 0,
              "--hadamard IQ2_XXS runs without an imatrix");

        const std::string hq40im = path("hqq-hq40-im.gguf");
        quantize(src, hq40im, LLAMA_FTYPE_MOSTLY_Q4_0, [&](auto & p) { p.hadamard = true; p.imatrix = im_all.get(); });
        gguf_file u(hq40), g(hq40im);
        const int64_t nrq = srcf.t(q)->ne[1];
        const std::vector<float> want = rotate_rows(to_f32(srcf.t(q), 0, nrq*n_embd), n_embd, g.seed());
        const double eu = out_err(want, to_f32(u.t(q), 0, nrq*n_embd), n_embd, im_all.at(q), u.seed());
        const double eg = out_err(want, to_f32(g.t(q), 0, nrq*n_embd), n_embd, im_all.at(q), g.seed());
        printf("  (HQ4_0 attn_q output error: uniform %.4g, GPTQ %.4g)\n", eu, eg);
        check(g.t(q)->type == GGML_TYPE_HQ4_0 && log_count("(gptq)") > 0 && eg < 0.7*eu, "--hadamard --imatrix Q4_0: GPTQ beats uniform HQ4_0");
    }

    printf("full Hessian (--hessian):\n");
    {
        const std::string hfile = path("hqq-hessian.gguf");
        make_hessian(srcf, im_all, hfile);
        const std::string hq_h = path("hqq-hq-hess.gguf");
        const int rc = quantize(src, hq_h, LLAMA_FTYPE_MOSTLY_Q4_K_M,
                                [&](auto & p) { p.hadamard = true; p.imatrix = im_all.get(); p.hessian = hfile.c_str(); });
        gguf_file a(hq), b(hq_im), h(hq_h);
        size_t n_hq = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(h.g); ++i) {
            n_hq += ggml_is_rotated(h.t(gguf_get_tensor_name(h.g, i))->type);
        }
        check(rc == 0 && log_count("(gptq, full H)") == n_hq && n_hq > 0 &&
              g_log.find(std::to_string(n_hq) + " of them with the full Hessian (alpha 0.1)") != std::string::npos,
              "every HQ tensor uses its full Gram, with one info line");
        check(h.f32("quantize.hq.gptq_damp") == 0.01f && h.f32("quantize.hq.gptq_hessian_alpha") == 0.1f &&
              h.str("quantize.hq.gptq_hessian") == "synthetic", "the three GPTQ keys are written");
        check(!b.has_key("quantize.hq.gptq_hessian_alpha") && !b.has_key("quantize.hq.gptq_hessian"),
              "a run without --hessian and without HQ copies has no Hessian keys");

        const int64_t nrq = srcf.t(q)->ne[1];
        const std::vector<float> want = rotate_rows(to_f32(srcf.t(q), 0, nrq*n_embd), n_embd, h.seed());
        const double eu = gram_err(want, to_f32(a.t(q), 0, nrq*n_embd), n_embd, im_all.at(q), h.seed());
        const double ed = gram_err(want, to_f32(b.t(q), 0, nrq*n_embd), n_embd, im_all.at(q), h.seed());
        const double ef = gram_err(want, to_f32(h.t(q), 0, nrq*n_embd), n_embd, im_all.at(q), h.seed());
        printf("  (attn_q output error under the full G: uniform %.4g, diagonal GPTQ %.4g, full-H GPTQ %.4g)\n", eu, ed, ef);
        check(ef < 0.7*ed && ed < eu, "under the full Gram, full-H GPTQ beats diagonal GPTQ, which beats uniform");

        const std::string hq_hc = path("hqq-hq-hess-copy.gguf");
        quantize(hq_h, hq_hc, LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) { p.hadamard = true; p.imatrix = im_all.get(); });
        gguf_file c(hq_hc);
        check(log_count("(gptq") == 0 && c.f32("quantize.hq.gptq_damp") == 0.01f &&
              c.f32("quantize.hq.gptq_hessian_alpha") == 0.1f && c.str("quantize.hq.gptq_hessian") == "synthetic",
              "HQ tensors copied from a full-H file keep all three keys");
    }

    printf("unsupported widths:\n");
    {
        const int64_t wide = 256*65;
        const std::string syn = path("hqq-syn.gguf");
        {
            gguf_context * out = gguf_init_empty();
            gguf_set_kv(out, srcf.g);
            ggml_context * ctx = ggml_init({ 2*(size_t) wide*n_embd*sizeof(float) + 1024*1024, nullptr, false });
            for (int64_t i = 0; i < gguf_get_n_tensors(srcf.g); ++i) {
                ggml_tensor * t = srcf.t(gguf_get_tensor_name(srcf.g, i));
                if (strstr(t->name, "ffn_down.weight")) {
                    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, wide, n_embd);
                    ggml_set_name(w, t->name);
                    for (int64_t j = 0; j < ggml_nelements(w); ++j) ((float *) w->data)[j] = 0.01f*(float) ((j*40503u) % 200 - 100);
                    gguf_add_tensor(out, w);
                } else {
                    gguf_add_tensor(out, t);
                }
            }
            gguf_write_to_file(out, syn.c_str(), false);
            gguf_free(out);
            ggml_free(ctx);
        }
        const std::string o = path("hqq-syn-q.gguf");
        const int rc = quantize(syn, o, LLAMA_FTYPE_MOSTLY_Q4_K_M, [](auto & p) { p.hadamard = true; p.pure = true; });
        size_t n_warn = 0;
        for (size_t pos = 0; (pos = g_log.find("width 16640 has no supported Hadamard rotation", pos)) != std::string::npos; ++pos) n_warn++;
        gguf_file f(o);
        check(rc == 0 && f.g && f.t("blk.0.ffn_down.weight")->type == GGML_TYPE_Q4_K &&
              f.t("blk.1.ffn_down.weight")->type == GGML_TYPE_Q4_K && f.t(q)->type == GGML_TYPE_HQ4_K && n_warn == 1,
              "--hadamard keeps the base type for width 64*65*4, warning once per width");

        const llama_model_tensor_override tto[2] = { { "blk\\.0\\.ffn_down", GGML_TYPE_HQ4_K }, { nullptr, GGML_TYPE_COUNT } };
        const int rc2 = quantize(syn, path("hqq-syn-q2.gguf"), LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) { p.tt_overrides = tto; });
        check(rc2 != 0 && g_log.find("no supported Hadamard rotation") != std::string::npos,
              "an explicit --tensor-type ...=hq4_K for that width is refused");
    }

    printf("output.weight without token_embd:\n");
    {
        // CodeShell-style: the embeddings come from output.weight, so it must stay unrotated
        const std::string noemb = path("hqq-noemb.gguf"), o = path("hqq-noemb-q.gguf");
        gguf_context * out = gguf_init_empty();
        gguf_set_kv(out, srcf.g);
        for (int64_t i = 0; i < gguf_get_n_tensors(srcf.g); ++i) {
            ggml_tensor * t = srcf.t(gguf_get_tensor_name(srcf.g, i));
            if (strcmp(t->name, "token_embd.weight") == 0) {
                ggml_set_name(t, "output.weight");
                gguf_add_tensor(out, t);
                ggml_set_name(t, "token_embd.weight");
            } else {
                gguf_add_tensor(out, t);
            }
        }
        gguf_write_to_file(out, noemb.c_str(), false);
        gguf_free(out);
        const int rc = quantize(noemb, o, LLAMA_FTYPE_MOSTLY_Q4_K_M, [](auto & p) { p.hadamard = true; });
        gguf_file f(o);
        check(rc == 0 && f.g && !ggml_is_rotated(f.t("output.weight")->type) && ggml_is_rotated(f.t(q)->type),
              "--hadamard leaves output.weight unrotated when there's no token_embd");
    }

    printf("refusals:\n");
    {
        check(quantize(src, path("hqq-bad.gguf"), LLAMA_FTYPE_MOSTLY_Q4_K_M, [](auto & p) { p.token_embedding_type = GGML_TYPE_HQ4_K; }) != 0,
              "HQ token embeddings are refused");

        const std::string noseed = path("hqq-noseed.gguf");
        relabel(path("hqq-plain.gguf"), noseed, false);
        check(quantize(noseed, path("hqq-bad.gguf"), LLAMA_FTYPE_MOSTLY_Q4_K_M, [](auto & p) { p.hadamard = true; }) != 0 &&
              g_log.find("ConvRot") != std::string::npos, "requantizing a file with HQ types and no hadamard.seed is refused");

        check(quantize(hq, path("hqq-bad.gguf"), LLAMA_FTYPE_MOSTLY_Q4_K_M, [](auto & p) {
                  p.hadamard = true; p.hadamard_seed = 12345; p.hadamard_seed_set = true; }) != 0,
              "requantizing with a --hadamard-seed that differs from the file's is refused");
        check(quantize(hq, path("hqq-bad.gguf"), LLAMA_FTYPE_MOSTLY_Q4_K_M) != 0, "HQ -> non-HQ is refused");
    }

    printf("requantize:\n");
    {
        const std::string h5 = path("hqq-h5.gguf"), h5c = path("hqq-h5-copy.gguf"), h4 = path("hqq-h5-hq4xs.gguf");
        quantize(src, h5, LLAMA_FTYPE_MOSTLY_Q5_K_M, [](auto & p) { p.hadamard = true; p.pure = true; });
        quantize(h5, h5c, LLAMA_FTYPE_MOSTLY_Q5_K_M, [](auto & p) { p.hadamard = true; p.pure = true; });
        gguf_file a(h5), b(h5c);
        check(a.t(q)->type == GGML_TYPE_HQ5_K && same_bytes(a.t(q), b.t(q)) && same_bytes(a.t(fd), b.t(fd)), "HQ5_K -> HQ5_K is a plain copy");

        check(quantize(h5, h4, LLAMA_FTYPE_MOSTLY_IQ4_XS, [](auto & p) { p.hadamard = true; p.pure = true; }) == 0, "HQ5_K -> HQ4_XS");
        gguf_file c(h4);
        const std::vector<float> w5 = to_f32(a.t(q), 0, 64*n_embd);
        const std::vector<float> w4 = to_f32(c.t(q), 0, 64*n_embd);
        const double err = nmse(w4, w5);
        printf("  (nmse HQ4_XS vs HQ5_K: %.4g)\n", err);
        check(c.t(q)->type == GGML_TYPE_HQ4_XS && err < 0.02 && c.seed() == a.seed(), "... is not re-rotated and keeps the seed");

        const std::string h4g = path("hqq-h5-hq4xs-gptq.gguf");
        quantize(h5, h4g, LLAMA_FTYPE_MOSTLY_IQ4_XS, [&](auto & p) { p.hadamard = true; p.pure = true; p.imatrix = im_all.get(); });
        gguf_file g(h4g);
        const int64_t nrq = srcf.t(q)->ne[1];
        const std::vector<float> w5all = to_f32(a.t(q), 0, nrq*n_embd);
        const double e_u = out_err(w5all, to_f32(c.t(q), 0, nrq*n_embd), n_embd, im_all.at(q), a.seed());
        const double e_g = out_err(w5all, to_f32(g.t(q), 0, nrq*n_embd), n_embd, im_all.at(q), a.seed());
        int n_seed_keys = 0;
        for (int64_t i = 0; i < gguf_get_n_kv(g.g); ++i) n_seed_keys += strcmp(gguf_get_key(g.g, i), "hadamard.seed") == 0;
        printf("  (attn_q output error, HQ5_K -> HQ4_XS: uniform %.4g, GPTQ %.4g)\n", e_u, e_g);
        check(g.t(q)->type == GGML_TYPE_HQ4_XS && log_count("(gptq)") > 0 && e_g < 0.7*e_u && g.seed() == a.seed() && n_seed_keys == 1,
              "HQ5_K -> HQ4_XS with an imatrix: GPTQ in the inherited seed's space beats uniform; one seed");

        const std::string part = path("hqq-part.gguf"), part2 = path("hqq-part-req.gguf");
        const llama_model_tensor_override only_down[2] = { { "ffn_down", GGML_TYPE_HQ4_K }, { nullptr, GGML_TYPE_COUNT } };
        quantize(src, part, LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) {
            p.tt_overrides = only_down; p.hadamard_seed = 99; p.hadamard_seed_set = true; });
        quantize(part, part2, LLAMA_FTYPE_MOSTLY_IQ4_XS, [](auto & p) { p.hadamard = true; p.pure = true; });
        gguf_file pf(part), d(part2);
        check(pf.t(fd)->type == GGML_TYPE_HQ4_K && !ggml_is_rotated(pf.t(q)->type) && pf.seed() == 99, "a partially rotated file (only ffn_down)");
        const std::vector<float> ws = to_f32(srcf.t(q), 0, 64*n_embd);
        const std::vector<float> wd = to_f32(d.t(q), 0, 64*n_embd);
        check(d.seed() == 99 && nmse(wd, rotate_rows(ws, n_embd, 99)) < 0.02, "a requantized file keeps the source seed");
    }

    printf("--lora merge:\n");
    {
        const std::string lora = path("hqq-lora.gguf");
        make_lora(srcf, lora, { q, fd, emb });
        std::vector<llama_model_quantize_lora> loras = { { lora.c_str(), 0.5f }, { nullptr, 0.0f } };

        const std::string mf16 = path("hqq-merge-f16.gguf");
        const int rc = quantize(src, mf16, LLAMA_FTYPE_MOSTLY_F16, [&](auto & p) { p.loras = loras.data(); });
        check(rc == 0, "f16 source + --lora -> F16");
        if (rc != 0) {
            printf("%s", g_log.c_str());
            return 1;
        }
        gguf_file f(mf16);
        bool ok = true;
        for (const auto & name : { q, fd, emb }) {
            const ggml_tensor * w = srcf.t(name);
            const int64_t nr = 16;
            const std::vector<float> want = add(to_f32(w, 0, nr*w->ne[0]), lora_delta(lora, name, w->ne[0], nr, 0.5f));
            const double err = nmse(to_f32(f.t(name), 0, nr*w->ne[0]), want);
            ok &= err < 1e-5;
        }
        check(ok, "merged weights are W + scale*alpha/r * B*A (incl. token_embd's flipped layout)");
        {
            const int64_t k = gguf_find_key(f.g, "general.merged_loras");
            check(k >= 0 && gguf_get_arr_n(f.g, k) == 1 && std::string(gguf_get_arr_str(f.g, k, 0)) == "hqq-lora.gguf:0.5",
                  "general.merged_loras records the adapter");
        }

        const std::string mhq = path("hqq-merge-hq.gguf");
        quantize(src, mhq, LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) { p.loras = loras.data(); p.hadamard = true; p.pure = true; });
        gguf_file g(mhq);
        const int64_t nr = 16;
        const std::vector<float> wq  = to_f32(srcf.t(q), 0, nr*n_embd);
        const std::vector<float> dq  = lora_delta(lora, q, n_embd, nr, 0.5f);
        const std::vector<float> got = to_f32(g.t(q), 0, nr*n_embd);
        const double e_merged = nmse(got, rotate_rows(add(wq, dq), n_embd, g.seed()));
        const double e_plain  = nmse(got, rotate_rows(wq, n_embd, g.seed()));
        printf("  (nmse vs R(W+D): %.4g, vs R*W: %.4g)\n", e_merged, e_plain);
        check(g.t(q)->type == GGML_TYPE_HQ4_K && e_merged < 0.01 && e_plain > 2*e_merged, "unrotated source: the delta is added before the rotation");
        check(g.t(emb)->type == GGML_TYPE_Q4_K && nmse(to_f32(g.t(emb), 0, nr*n_embd), add(to_f32(srcf.t(emb), 0, nr*n_embd), lora_delta(lora, emb, n_embd, nr, 0.5f))) < 0.01,
              "token_embd merges and stays unrotated");

        const std::string mhq2 = path("hqq-merge-hq2.gguf");
        quantize(path("hqq-h5.gguf"), mhq2, LLAMA_FTYPE_MOSTLY_Q5_K_M, [&](auto & p) { p.loras = loras.data(); p.hadamard = true; p.pure = true; });
        gguf_file h5(path("hqq-h5.gguf")), m2(mhq2);
        const std::vector<float> base5 = to_f32(h5.t(q), 0, nr*n_embd);
        const std::vector<float> got2  = to_f32(m2.t(q), 0, nr*n_embd);
        const double e2 = nmse(got2, add(base5, rotate_rows(dq, n_embd, h5.seed())));
        printf("  (nmse HQ source + R*D: %.4g)\n", e2);
        check(e2 < 0.01, "HQ source: the delta is rotated with the file's seed, then added");

        const std::string mim = path("hqq-merge-hq-im.gguf");
        const int rc_im = quantize(src, mim, LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) {
            p.loras = loras.data(); p.hadamard = true; p.pure = true; p.imatrix = im_all.get(); });
        gguf_file gi(mim);
        const int64_t nrq = srcf.t(q)->ne[1];
        const std::vector<float> want = rotate_rows(add(to_f32(srcf.t(q), 0, nrq*n_embd), lora_delta(lora, q, n_embd, nrq, 0.5f)), n_embd, gi.seed());
        const double eu = out_err(want, to_f32(g.t(q), 0, nrq*n_embd), n_embd, im_all.at(q), g.seed());
        const double eg = out_err(want, to_f32(gi.t(q), 0, nrq*n_embd), n_embd, im_all.at(q), gi.seed());
        printf("  (attn_q output error vs R(W+D): uniform %.4g, GPTQ %.4g)\n", eu, eg);
        check(rc_im == 0 && log_count("(gptq)") > 0 && eg < 0.7*eu, "--lora with an imatrix: merged, then GPTQ");

        std::vector<llama_model_quantize_lora> l2;
        const std::string bad_arch = path("hqq-lora-arch.gguf"), bad_alora = path("hqq-lora-alora.gguf"), bad_name = path("hqq-lora-name.gguf");
        make_lora(srcf, bad_arch, { q }, "llama");
        make_lora(srcf, bad_alora, { q }, nullptr, true);
        make_lora(srcf, bad_name, { "blk.9.attn_q.weight" });
        for (const auto & [p, what] : std::vector<std::pair<std::string, std::string>>{
                 { bad_arch, "a different architecture" }, { bad_alora, "an activated LoRA" }, { bad_name, "an adapter tensor with no model tensor" } }) {
            l2 = { { p.c_str(), 1.0f }, { nullptr, 0.0f } };
            check(quantize(src, path("hqq-bad.gguf"), LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & pp) { pp.loras = l2.data(); }) != 0,
                  "refuses " + what);
        }
    }

    printf("3D experts:\n");
    {
        const std::string syn3 = path("hqq-exps.gguf"), o3 = path("hqq-exps-q.gguf");
        const std::string en = "blk.0.ffn_down_exps.weight";
        const int64_t nr = 64, nexp = 2;
        {
            gguf_context * out = gguf_init_empty();
            gguf_set_kv(out, srcf.g);
            ggml_context * ctx = ggml_init({ (size_t) n_embd*nr*nexp*sizeof(float) + 1024*1024, nullptr, false });
            for (int64_t i = 0; i < gguf_get_n_tensors(srcf.g); ++i) {
                gguf_add_tensor(out, srcf.t(gguf_get_tensor_name(srcf.g, i)));
            }
            ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, nr, nexp);
            ggml_set_name(w, en.c_str());
            uint32_t state = 11;
            for (int64_t j = 0; j < ggml_nelements(w); ++j) {
                state = state*1664525u + 1013904223u;
                ((float *) w->data)[j] = 0.02f*(((state >> 8) & 0xffff)/32768.0f - 1.0f);
            }
            gguf_add_tensor(out, w);
            gguf_write_to_file(out, syn3.c_str(), false);
            gguf_free(out);
            ggml_free(ctx);
        }
        imatrix_set im3;
        im3.add(en, synth_v(n_embd, nexp, 5));
        const int rc = quantize(syn3, o3, LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) { p.hadamard = true; p.pure = true; p.imatrix = im3.get(); });
        gguf_file in(syn3), f(o3);
        const ggml_tensor * t = rc == 0 ? f.t(en) : nullptr;
        check(t && t->type == GGML_TYPE_HQ4_K && log_count("1 HQ tensors use the imatrix through GPTQ") == 1, "a two-expert 3D tensor is quantized with GPTQ");
        if (t) {
            std::vector<std::thread> workers;
            const size_t rs = ggml_row_size(GGML_TYPE_HQ4_K, n_embd);
            bool ok = true, other_differs = true;
            for (int64_t e = 0; e < nexp; ++e) {
                const float * src_e = (const float *) in.t(en)->data + e*nr*n_embd;
                auto encode = [&](int64_t ev) {
                    std::vector<float> rows = rotate_rows(std::vector<float>(src_e, src_e + nr*n_embd), n_embd, f.seed());
                    std::vector<float> U;
                    llama_gptq_factor(im3.at(en).data() + ev*n_embd, n_embd, f.seed(), LLAMA_GPTQ_DAMP_DEFAULT, U, 8);
                    std::vector<uint8_t> qb(nr*rs);
                    llama_tensor_quantize_gptq(GGML_TYPE_HQ4_K, rows.data(), qb.data(), nr, n_embd, U.data(), workers, 8);
                    return qb;
                };
                const uint8_t * got = (const uint8_t *) t->data + e*nr*rs;
                ok            &= memcmp(got, encode(e).data(), nr*rs) == 0;
                other_differs &= memcmp(got, encode(1 - e).data(), nr*rs) != 0;
            }
            check(ok, "each expert equals its slice quantized alone with its own imatrix slice");
            check(other_differs, "... and differs from using the other expert's slice");

            imatrix_set im3z;
            std::vector<float> vz = im3.at(en);
            std::fill(vz.begin() + n_embd, vz.end(), 0.0f);
            im3z.add(en, vz);
            const std::string o3z = path("hqq-exps-qz.gguf");
            quantize(syn3, o3z, LLAMA_FTYPE_MOSTLY_Q4_K_M, [&](auto & p) { p.hadamard = true; p.pure = true; p.imatrix = im3z.get(); });
            gguf_file fz(o3z);
            const float * src_1 = (const float *) in.t(en)->data + nr*n_embd;
            std::vector<float> rows1 = rotate_rows(std::vector<float>(src_1, src_1 + nr*n_embd), n_embd, fz.seed());
            std::vector<uint8_t> u1(nr*rs);
            ggml_quantize_chunk(GGML_TYPE_HQ4_K, rows1.data(), u1.data(), 0, nr, n_embd, nullptr);
            const uint8_t * gz = (const uint8_t *) fz.t(en)->data;
            check(memcmp(gz, t->data, nr*rs) == 0 && memcmp(gz + nr*rs, u1.data(), nr*rs) == 0 &&
                  log_count("1 experts of GPTQ tensors have an all-zero imatrix slice") == 1,
                  "an expert with an all-zero imatrix slice gets the uniform bytes, with a warning");
        }
    }

    llama_backend_free();
    printf("\n%s: %d failure(s)\n", g_failed ? "FAIL" : "PASS", g_failed);
    return g_failed ? 1 : 0;
}
