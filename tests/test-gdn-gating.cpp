// The gated-delta-net gating on CUDA: beta = sigmoid(beta projection) and g = softplus(alpha projection + dt)*a, as
// qwen35 builds them, feeding a GDN that reads its state from the cache and writes its snapshots back. The backend
// computes the gating inside the GDN (ggml_backend_cuda_gdn_gating_count); with g also a graph output it runs the
// gating nodes instead. Both must give the same bytes: the kernel uses the unary ops' own helpers.
//
// usage: test-gdn-gating [-v]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct gdn_case {
    int64_t H, S_v, n_tokens, n_seqs, K;
};

// the attention output and the cache after the snapshots
static std::vector<float> run(ggml_backend_t backend, const gdn_case & c, bool g_output) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    const int64_t H = c.H, S_v = c.S_v, T = c.n_tokens, S = c.n_seqs, D = S_v*S_v*H, n_embd = 512;
    const int64_t n_rows    = c.K + 2;
    const int64_t n_written = std::min(T, c.K);

    ggml_tensor * x       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, T, S);
    ggml_tensor * w_beta  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, H);
    ggml_tensor * w_alpha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, H);
    ggml_tensor * dt      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * a       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * q       = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, S);
    ggml_tensor * k       = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, S);
    ggml_tensor * v       = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, S);
    ggml_tensor * states  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_rows*S);
    ggml_tensor * idx     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);

    ggml_tensor * beta  = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w_beta, x), 1, H, T, S));
    ggml_tensor * alpha = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w_alpha, x), H, T, S);
    ggml_tensor * gate  = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, dt)), a);
    if (g_output) {
        ggml_set_output(gate);
    }
    gate = ggml_reshape_4d(ctx, gate, 1, H, T, S);

    ggml_tensor * state = ggml_reshape_4d(ctx, ggml_get_rows(ctx, states, idx), S_v, S_v, H, S);
    ggml_tensor * out   = ggml_gated_delta_net(ctx, ggml_l2_norm(ctx, q, 1e-6f), ggml_l2_norm(ctx, k, 1e-6f), v, gate, beta, state, c.K);
    ggml_tensor * src   = ggml_view_3d(ctx, out, D, S, n_written, ggml_row_size(out->type, D),
            ggml_row_size(out->type, D*S), ggml_row_size(out->type, S_v*H*T*S));
    ggml_tensor * dst   = ggml_view_3d(ctx, states, D, S, n_written, states->nb[1], S*states->nb[1], 0);
    ggml_tensor * cpy   = ggml_cpy(ctx, src, dst);
    ggml_tensor * attn  = ggml_cont(ctx, ggml_view_1d(ctx, out, S_v*H*T*S, 0));
    ggml_set_output(attn);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cpy);
    ggml_build_forward_expand(gf, attn);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    for (ggml_tensor * t : { x, w_beta, w_alpha, dt, a, q, k, v, states }) {
        std::vector<float> d(ggml_nelements(t));
        for (auto & e : d) {
            e = t == a ? -0.1f - 1.9f*(ud(rng) + 1.0f)/2 : ud(rng); // a = -exp(A_log)
        }
        ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
    }
    std::vector<int32_t> rows(S);
    for (int64_t s = 0; s < S; ++s) {
        rows[s] = (int32_t) (3*s);
    }
    ggml_backend_tensor_set(idx, rows.data(), 0, rows.size()*sizeof(int32_t));

    ggml_backend_graph_compute(backend, gf);
    std::vector<float> res(ggml_nelements(attn) + ggml_nelements(states));
    ggml_backend_tensor_get(attn, res.data(), 0, ggml_nbytes(attn));
    ggml_backend_tensor_get(states, res.data() + ggml_nelements(attn), 0, ggml_nbytes(states));
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
    auto gating_count = (int64_t (*)()) ggml_backend_reg_get_proc_address(ggml_backend_reg_by_name("CUDA"), "ggml_backend_cuda_gdn_gating_count");
    if (!gating_count) {
        printf("no ggml_backend_cuda_gdn_gating_count\n");
        return 1;
    }

    std::vector<gdn_case> cases;
    for (int64_t S : { 1, 2 }) {
        for (int64_t T : { 1, 5, 64 }) {
            for (int64_t K : { 1, 5 }) {
                cases.push_back({ 48, 128, T, S, K });
            }
        }
    }
    cases.push_back({ 16, 64, 5, 1, 5 });

    int fails = 0;
    for (const auto & c : cases) {
        const int64_t n0 = gating_count();
        const std::vector<float> fused = run(backend, c, false);
        const int64_t n1 = gating_count();
        const std::vector<float> ref = run(backend, c, true);
        const int64_t n2 = gating_count();

        const bool same = memcmp(fused.data(), ref.data(), fused.size()*sizeof(float)) == 0;
        const bool ok   = same && n1 > n0 && n2 == n1;
        if (!ok || verbose) {
            printf("  H %2lld S_v %3lld tokens %2lld seqs %lld K %lld: %s%s%s\n", (long long) c.H, (long long) c.S_v,
                   (long long) c.n_tokens, (long long) c.n_seqs, (long long) c.K, same ? "bitwise equal" : "differ",
                   n1 > n0 && n2 == n1 ? "" : " (fusion counter wrong)", ok ? "" : "  FAIL");
        }
        fails += !ok;
    }
    printf("%zu cases, %d failed\n%s\n", cases.size(), fails, fails ? "FAIL" : "PASS");
    ggml_backend_free(backend);
    return fails ? 1 : 0;
}
