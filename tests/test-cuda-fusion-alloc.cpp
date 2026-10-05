// CUDA fusions whose memory checks depend on where the graph allocator puts tensors, which test-backend-ops (each
// tensor in its own allocation) can't show. The graphs here are allocated by ggml_gallocr as the scheduler does:
// - a matmul + reshape + add of a computed residual: the allocator puts the sum in place of the residual, which the
//   fused MMVQ kernel allows (each block reads its rows' residual before it writes them). It must fuse
//   (ggml_backend_cuda_mmvq_fused_count) and give the bytes of the graph with the matmul an output, which is not fused.
//
// usage: test-cuda-fusion-alloc [-v]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct mm_case {
    ggml_type type;
    int64_t   k, n, m;
};

struct run_result {
    std::vector<float> out;
    bool in_place = false; // the sum took the residual's memory
};

static run_result run_reshape_add(ggml_backend_t backend, const mm_case & c, bool mm_output) {
    ggml_init_params ip_w = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(ip_w);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, c.type, c.k, c.n);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);

    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k, c.m);
    ggml_tensor * r0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.n, 1, c.m);
    ggml_set_input(x);
    ggml_set_input(r0);
    ggml_tensor * r  = ggml_scale(ctx, r0, 2.0f);
    ggml_tensor * mm = ggml_mul_mat(ctx, w, x);
    if (mm_output) {
        ggml_set_output(mm);
    }
    ggml_tensor * out = ggml_add(ctx, r, ggml_reshape_3d(ctx, mm, c.n, 1, c.m));
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::vector<float> wf(c.k*c.n);
    for (auto & e : wf) e = ud(rng);
    std::vector<uint8_t> wq(ggml_nbytes(w));
    ggml_quantize_chunk(c.type, wf.data(), wq.data(), 0, c.n, c.k, nullptr);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    for (ggml_tensor * t : { x, r0 }) {
        std::vector<float> d(ggml_nelements(t));
        for (auto & e : d) e = ud(rng);
        ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
    }

    ggml_backend_graph_compute(backend, gf);
    run_result res;
    res.in_place = out->data == r->data;
    res.out.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.out.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
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
    auto fused_count = (int64_t (*)()) ggml_backend_reg_get_proc_address(ggml_backend_reg_by_name("CUDA"), "ggml_backend_cuda_mmvq_fused_count");
    if (!fused_count) {
        printf("no ggml_backend_cuda_mmvq_fused_count\n");
        return 1;
    }

    std::vector<mm_case> cases;
    for (ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_K }) {
        for (int64_t m : { 1, 2, 5 }) {
            cases.push_back({ type, 6144, 5120, m });
        }
    }

    int fails = 0;
    for (const auto & c : cases) {
        const int64_t n0 = fused_count();
        const run_result fused = run_reshape_add(backend, c, false);
        const int64_t n1 = fused_count();
        const run_result ref = run_reshape_add(backend, c, true);
        const int64_t n2 = fused_count();

        const bool same = memcmp(fused.out.data(), ref.out.data(), fused.out.size()*sizeof(float)) == 0;
        const bool ok   = fused.in_place && same && n1 > n0 && n2 == n1;
        if (!ok || verbose) {
            printf("  reshape add %-5s %lldx%lld, %lld columns: sum %s the residual, %s, %s%s\n", ggml_type_name(c.type),
                   (long long) c.k, (long long) c.n, (long long) c.m, fused.in_place ? "in place of" : "apart from",
                   n1 > n0 ? "fused" : "not fused", same ? "bitwise equal" : "differs", ok ? "" : "  FAIL");
        }
        fails += !ok;
    }
    printf("%zu cases, %d failed\n%s\n", cases.size(), fails, fails ? "FAIL" : "PASS");
    ggml_backend_free(backend);
    return fails ? 1 : 0;
}
