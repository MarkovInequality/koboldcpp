// A gated-delta-net layer on CUDA as qwen35 builds it: beta = sigmoid(beta projection) and g = softplus(alpha
// projection + dt)*a, q and k L2-normed heads of the conv output, and a GDN that reads its state from the cache and
// writes its snapshots back. Two variants must give the same bytes:
// - the gating computed inside the GDN (ggml_backend_cuda_gdn_gating_count) and, with g also a graph output, by the
//   gating nodes: the kernel uses the unary ops' own helpers;
// - q and k normed by one op over both (build_gdn_l2_norm_qk) and each by its own;
// - each warp of the GDN kernel owning 2 or 4 state columns, or as many as it picks, and one (GGML_CUDA_GDN_COLS=1).
// Prompt batches (64+ tokens before the snapshot tail) take the chunked kernels (ggml_backend_cuda_gdn_chunked_count),
// which must agree with the recurrent kernel (GGML_CUDA_GDN_CHUNKED=0) within NMSE_CHUNKED, attention outputs and
// states each.
//
// usage: test-gdn-gating [-v]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const double NMSE_CHUNKED = 1e-9;

static double nmse(const float * a, const float * b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0.0 ? num / den : num;
}

struct gdn_case {
    int64_t H, H_k, S_v, n_tokens, n_seqs, K;
};

static ggml_tensor * l2_norm(ggml_context * ctx, ggml_tensor * x) {
    const float n = x->ne[0];
    return ggml_scale(ctx, ggml_rms_norm(ctx, x, 1e-6f/n), 1.0f/sqrtf(n));
}

// the attention output and the cache after the snapshots
static std::vector<float> run(ggml_backend_t backend, const gdn_case & c, bool g_output, bool merged_qk) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    const int64_t H = c.H, H_k = c.H_k, S_v = c.S_v, T = c.n_tokens, S = c.n_seqs, D = S_v*S_v*H, n_embd = 512;
    const int64_t n_rows    = c.K + 2;
    const int64_t n_written = std::min(T, c.K);

    ggml_tensor * x       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, T, S);
    ggml_tensor * w_beta  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, H);
    ggml_tensor * w_alpha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, H);
    ggml_tensor * dt      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * a       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * qk_in   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v*(2*H_k + 1), T, S);
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

    // q and k heads of a wider row, as in the conv output
    const size_t  hs = ggml_row_size(GGML_TYPE_F32, S_v);
    ggml_tensor * q  = ggml_view_4d(ctx, qk_in, S_v, H_k, T, S, hs, qk_in->nb[1], qk_in->nb[2], 0);
    ggml_tensor * k  = ggml_view_4d(ctx, qk_in, S_v, H_k, T, S, hs, qk_in->nb[1], qk_in->nb[2], H_k*hs);
    if (merged_qk) {
        ggml_tensor * qk = l2_norm(ctx, ggml_view_4d(ctx, qk_in, S_v, 2*H_k, T, S, hs, qk_in->nb[1], qk_in->nb[2], 0));
        q = ggml_view_4d(ctx, qk, S_v, H_k, T, S, qk->nb[1], qk->nb[2], qk->nb[3], 0);
        k = ggml_view_4d(ctx, qk, S_v, H_k, T, S, qk->nb[1], qk->nb[2], qk->nb[3], H_k*qk->nb[1]);
    } else {
        q = l2_norm(ctx, q);
        k = l2_norm(ctx, k);
    }

    ggml_tensor * state = ggml_reshape_4d(ctx, ggml_get_rows(ctx, states, idx), S_v, S_v, H, S);
    ggml_tensor * out   = ggml_gated_delta_net(ctx, q, k, v, gate, beta, state, c.K);
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
    for (ggml_tensor * t : { x, w_beta, w_alpha, dt, a, qk_in, v, states }) {
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
    auto gating_count  = (int64_t (*)()) ggml_backend_reg_get_proc_address(ggml_backend_reg_by_name("CUDA"), "ggml_backend_cuda_gdn_gating_count");
    auto chunked_count = (int64_t (*)()) ggml_backend_reg_get_proc_address(ggml_backend_reg_by_name("CUDA"), "ggml_backend_cuda_gdn_chunked_count");
    if (!gating_count || !chunked_count) {
        printf("no ggml_backend_cuda_gdn_gating_count or ggml_backend_cuda_gdn_chunked_count\n");
        return 1;
    }

    std::vector<gdn_case> cases;
    for (int64_t S : { 1, 2 }) {
        for (int64_t T : { 1, 5, 33, 64, 1024 }) {
            for (int64_t K : { 1, 5 }) {
                cases.push_back({ 48, 16, 128, T, S, K });
            }
        }
    }
    cases.push_back({ 16, 16, 64, 5, 1, 5 });

    int fails = 0;
    double worst = 0.0;
    for (const auto & c : cases) {
        const int64_t tail     = c.K > 1 ? std::min(c.n_tokens, c.K) : 0;
        const bool    chunked  = c.S_v == 128 && c.n_tokens - tail >= 64;
        const size_t  n_attn   = (size_t) (c.S_v*c.H*c.n_tokens*c.n_seqs);

        setenv("GGML_CUDA_GDN_CHUNKED", "0", 1);
        unsetenv("GGML_CUDA_GDN_COLS");
        const int64_t m0 = chunked_count();
        const std::vector<float> recurrent = run(backend, c, false, false);
        unsetenv("GGML_CUDA_GDN_CHUNKED");
        const int64_t m1 = chunked_count();
        const std::vector<float> dflt = run(backend, c, false, false);
        const int64_t m2 = chunked_count();
        const double e_attn  = nmse(dflt.data(), recurrent.data(), n_attn);
        const double e_state = nmse(dflt.data() + n_attn, recurrent.data() + n_attn, dflt.size() - n_attn);
        worst = std::max(worst, std::max(e_attn, e_state));
        const bool path_ok = m1 == m0 && (m2 > m1) == chunked && e_attn <= NMSE_CHUNKED && e_state <= NMSE_CHUNKED;

        bool same_cols = true;
        setenv("GGML_CUDA_GDN_COLS", "1", 1);
        const std::vector<float> one_col = run(backend, c, false, false);
        for (const char * nc : { "2", "4", "" }) {
            if (*nc) {
                setenv("GGML_CUDA_GDN_COLS", nc, 1);
            } else {
                unsetenv("GGML_CUDA_GDN_COLS");
            }
            const std::vector<float> r = run(backend, c, false, false);
            same_cols &= memcmp(r.data(), one_col.data(), r.size()*sizeof(float)) == 0;
        }

        const int64_t n0 = gating_count();
        const std::vector<float> fused = run(backend, c, false, false);
        const int64_t n1 = gating_count();
        const std::vector<float> ref = run(backend, c, true, false);
        const int64_t n2 = gating_count();
        const std::vector<float> merged = run(backend, c, false, true);

        const bool same_gating = memcmp(fused.data(), ref.data(), fused.size()*sizeof(float)) == 0;
        const bool same_qk     = memcmp(fused.data(), merged.data(), fused.size()*sizeof(float)) == 0;
        const bool counted     = n1 > n0 && n2 == n1;
        const bool ok          = same_gating && same_qk && same_cols && counted && path_ok;
        if (!ok || verbose) {
            printf("  H %2lld/%2lld S_v %3lld tokens %4lld seqs %lld K %lld: gating %s, merged q/k norm %s, columns per warp %s, "
                   "%s nmse attn %.1e state %.1e%s%s\n",
                   (long long) c.H, (long long) c.H_k, (long long) c.S_v, (long long) c.n_tokens, (long long) c.n_seqs,
                   (long long) c.K, same_gating ? "bitwise equal" : "differs", same_qk ? "bitwise equal" : "differs",
                   same_cols ? "bitwise equal" : "differ", chunked ? "chunked vs recurrent" : "recurrent", e_attn, e_state,
                   counted ? "" : " (fusion counter wrong)", ok ? "" : "  FAIL");
        }
        fails += !ok;
    }
    printf("%zu cases, %d failed, worst chunked nmse %.1e\n%s\n", cases.size(), fails, worst, fails ? "FAIL" : "PASS");
    ggml_backend_free(backend);
    return fails ? 1 : 0;
}
