// FLASH_ATTN_EXT with quantized K/V on CUDA: the MMA kernel reading the quantized tiles against the same kernel on
// the F16 copy (GGML_CUDA_FA_Q_CONVERT=1). The tiles hold the same bytes, so with a KV range short enough that both
// launches split it alike (512) the outputs are bitwise equal; longer ranges may be split at other points (the
// quantized kernel needs less shared memory, so more blocks fit), within 1e-6 nmse. The quantized kernel must have
// run (ggml_backend_cuda_fattn_mma_q_count).
//
// usage: test-fattn-mma-q [-v]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct fa_case {
    ggml_type type;
    int hs, n_kv_head, gqa, n_kv, n_tokens;
};

static std::vector<float> run(ggml_backend_t backend, const fa_case & c, uint32_t seed) {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int n_head = c.n_kv_head*c.gqa;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.hs, c.n_tokens, n_head, 1);
    // K/V as views of a cache twice as long, like the KV cache
    ggml_tensor * k0 = ggml_new_tensor_4d(ctx, c.type, c.hs, 2*c.n_kv, c.n_kv_head, 1);
    ggml_tensor * v0 = ggml_new_tensor_4d(ctx, c.type, c.hs, 2*c.n_kv, c.n_kv_head, 1);
    ggml_tensor * k = ggml_view_4d(ctx, k0, c.hs, c.n_kv, c.n_kv_head, 1, k0->nb[1], k0->nb[2], k0->nb[3], 0);
    ggml_tensor * v = ggml_view_4d(ctx, v0, c.hs, c.n_kv, c.n_kv_head, 1, v0->nb[1], v0->nb[2], v0->nb[3], 0);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_tokens, 1, 1);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/std::sqrt((float) c.hs), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto fill_f32 = [&](ggml_tensor * t) {
        std::vector<float> d(ggml_nelements(t));
        for (auto & x : d) x = nd(rng);
        ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
    };
    auto fill_q = [&](ggml_tensor * t) {
        std::vector<float> d(ggml_nelements(t));
        for (auto & x : d) x = nd(rng);
        std::vector<uint8_t> qd(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, d.data(), qd.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        ggml_backend_tensor_set(t, qd.data(), 0, qd.size());
    };
    fill_f32(q);
    fill_q(k0);
    fill_q(v0);
    {
        // causal over the last n_tokens positions
        std::vector<ggml_fp16_t> md(ggml_nelements(m));
        for (int64_t j = 0; j < m->ne[1]; ++j) {
            for (int64_t i = 0; i < m->ne[0]; ++i) {
                const bool visible = j < c.n_tokens && i <= c.n_kv - c.n_tokens + j;
                md[j*m->ne[0] + i] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(m, md.data(), 0, md.size()*sizeof(ggml_fp16_t));
    }
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main(int argc, char ** argv) {
    const bool verbose = argc > 1 && std::string(argv[1]) == "-v";
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1); // each run dispatches again
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        printf("no CUDA device\n");
        return 1;
    }

    std::vector<fa_case> cases;
    for (ggml_type type : { GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
        for (int hs : { 128, 256 }) {
            for (int gqa : { 4, 6, 8 }) {
                for (int n_kv : { 512, 8192 }) {
                    for (int nt : { 1, 2, 3, 5, 8 }) {
                        cases.push_back({ type, hs, 4, gqa, n_kv, nt });
                    }
                }
            }
        }
    }

    auto mma_q_count = (int64_t (*)()) ggml_backend_reg_get_proc_address(ggml_backend_reg_by_name("CUDA"), "ggml_backend_cuda_fattn_mma_q_count");
    if (!mma_q_count) {
        printf("no ggml_backend_cuda_fattn_mma_q_count\n");
        return 1;
    }

    int fails = 0, n_bitwise = 0;
    for (const auto & c : cases) {
        unsetenv("GGML_CUDA_FA_Q_CONVERT");
        const int64_t n0 = mma_q_count();
        const std::vector<float> a = run(backend, c, 7);
        const bool ran_q = mma_q_count() > n0;
        setenv("GGML_CUDA_FA_Q_CONVERT", "1", 1);
        const int64_t n1 = mma_q_count();
        const std::vector<float> b = run(backend, c, 7);
        const bool ran_q_ref = mma_q_count() > n1;

        double err = 0, ref = 0;
        bool finite = true;
        for (size_t i = 0; i < a.size(); ++i) {
            finite &= std::isfinite(a[i]);
            err += (double) (a[i] - b[i])*(a[i] - b[i]);
            ref += (double) b[i]*b[i];
        }
        const bool same = memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
        n_bitwise += same;
        const double nmse = err/ref;
        const bool ok = finite && (c.n_kv <= 512 ? same : nmse < 1e-6) && ran_q && !ran_q_ref;
        if (!ok || verbose) {
            printf("  %-5s hs %3d gqa %d kv %5d tokens %d: %s (nmse %.3g)%s%s\n", ggml_type_name(c.type), c.hs, c.gqa, c.n_kv,
                   c.n_tokens, same ? "bitwise equal" : "differ", nmse, ran_q && !ran_q_ref ? "" : " (wrong kernel)", ok ? "" : "  FAIL");
        }
        fails += !ok;
    }
    printf("%zu cases, %d bitwise equal to the F16-copy path, %d failed\n%s\n", cases.size(), n_bitwise, fails, fails ? "FAIL" : "PASS");
    ggml_backend_free(backend);
    return fails ? 1 : 0;
}
