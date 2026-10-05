// MUL_MAT bandwidth at decode/verify shapes (1..8 columns) through the CUDA backend, the path MMVQ takes.
// Each case multiplies a cycle of weight copies of at least 384 MB (L2 is 96 MB on the 5090) in one graph, so CUDA
// graphs launch it and no weight byte comes from L2; GB/s is against 1792 GB/s and a measured LDG.128 read roofline.
//
// With -e add each product gets a same-shape residual added, with -e glu pairs of copies are gate and up into a
// SWIGLU: the epilogues MMVQ fuses (GGML_CUDA_DISABLE_FUSION=1 for the unfused kernels).
//
// usage: bench-mmvq [-t TYPES] [-k KS] [-n NS] [-c NCOLS] [-r REPS] [-e none|add|glu]   (comma lists; types by ggml name)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

static __global__ void read_kernel(const int4 * __restrict__ p, size_t n, int * out) {
    int acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const int4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x7fffffff) {
        *out = acc;
    }
}

static double ldg128_roofline_gbs(size_t bytes) {
    int4 * p;
    int * out;
    cudaMalloc(&p, bytes);
    cudaMalloc(&out, sizeof(int));
    cudaMemset(p, 1, bytes);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    double best = 0;
    for (int rep = 0; rep < 10; ++rep) {
        cudaEventRecord(e0);
        read_kernel<<<sms*8, 256>>>(p, bytes/sizeof(int4), out);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        best = std::max(best, bytes/(ms*1e-3)/1e9);
    }
    cudaFree(p);
    cudaFree(out);
    return best;
}

template <typename T>
static std::vector<T> parse_list(const char * s, T (*conv)(const std::string &)) {
    std::vector<T> res;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        res.push_back(conv(item));
    }
    return res;
}

static ggml_type type_by_name(const std::string & name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && name == n) {
            return (ggml_type) t;
        }
    }
    fprintf(stderr, "unknown type %s\n", name.c_str());
    exit(1);
}

int main(int argc, char ** argv) {
    std::vector<ggml_type> types = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0, GGML_TYPE_Q3_K };
    std::vector<int64_t> ks = { 5120, 6144, 17408 };
    std::vector<int64_t> ns = { 48, 96, 1024, 6144, 10240, 17408 };
    std::vector<int64_t> ncols = { 1, 2, 3, 4, 5, 6, 7, 8 };
    int reps = 10;
    std::string epilogue = "none";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        auto to_i64 = [](const std::string & s) { return (int64_t) std::stoll(s); };
        if (a == "-t") types = parse_list<ggml_type>(argv[i + 1], type_by_name);
        else if (a == "-k") ks = parse_list<int64_t>(argv[i + 1], to_i64);
        else if (a == "-n") ns = parse_list<int64_t>(argv[i + 1], to_i64);
        else if (a == "-c") ncols = parse_list<int64_t>(argv[i + 1], to_i64);
        else if (a == "-r") reps = std::stoi(argv[i + 1]);
        else if (a == "-e") epilogue = argv[i + 1];
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }

    const size_t cycle_bytes = 384ull << 20;
    const double roof = ldg128_roofline_gbs(cycle_bytes);
    printf("LDG.128 read roofline: %.0f GB/s (nominal 1792)\n", roof);
    printf("%-7s %6s %6s %5s %6s %9s %8s %7s %7s\n", "type", "K", "N", "ncols", "copies", "us/matmul", "GB/s", "%1792", "%roof");

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (ggml_type type : types) {
        for (int64_t K : ks) {
            if (K % ggml_blck_size(type) != 0) {
                continue;
            }
            for (int64_t N : ns) {
                const size_t w_bytes = ggml_row_size(type, K)*N;
                const int n_copies = 2*(int) std::min<size_t>(2048, (cycle_bytes + 2*w_bytes - 1)/(2*w_bytes));

                // one quantized random weight, copied n_copies times on the device
                std::vector<float> wf(K*N);
                for (auto & v : wf) v = nd(rng)*0.02f;
                std::vector<uint8_t> wq(w_bytes);
                ggml_quantize_chunk(type, wf.data(), wq.data(), 0, N, K, nullptr);

                for (int64_t nc : ncols) {
                    ggml_init_params ip = { ggml_tensor_overhead()*(4*n_copies + 8) + ggml_graph_overhead_custom(3*n_copies + 8, false), nullptr, true };
                    ggml_context * ctx = ggml_init(ip);
                    std::vector<ggml_tensor *> ws(n_copies);
                    for (auto & w : ws) w = ggml_new_tensor_2d(ctx, type, K, N);
                    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, nc);
                    ggml_tensor * r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, nc);
                    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 3*n_copies + 8, false);
                    for (int c = 0; c < n_copies; ++c) {
                        if (epilogue == "glu") {
                            ggml_build_forward_expand(gf, ggml_swiglu_split(ctx, ggml_mul_mat(ctx, ws[c], x), ggml_mul_mat(ctx, ws[c + 1], x)));
                            ++c;
                        } else if (epilogue == "add") {
                            ggml_build_forward_expand(gf, ggml_add(ctx, ggml_mul_mat(ctx, ws[c], x), r));
                        } else {
                            ggml_build_forward_expand(gf, ggml_mul_mat(ctx, ws[c], x));
                        }
                    }
                    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
                    if (!buf) {
                        fprintf(stderr, "allocation failed\n");
                        return 1;
                    }
                    ggml_backend_tensor_set(ws[0], wq.data(), 0, w_bytes);
                    for (int c = 1; c < n_copies; ++c) {
                        cudaMemcpy(ws[c]->data, ws[0]->data, w_bytes, cudaMemcpyDeviceToDevice);
                    }
                    std::vector<float> xf(K*nc);
                    for (auto & v : xf) v = nd(rng);
                    ggml_backend_tensor_set(x, xf.data(), 0, xf.size()*sizeof(float));
                    std::vector<float> rf(N*nc, 1.0f);
                    ggml_backend_tensor_set(r, rf.data(), 0, rf.size()*sizeof(float));

                    // at least 300 ms, so the clocks are up
                    for (const auto w0 = std::chrono::steady_clock::now(); std::chrono::steady_clock::now() - w0 < std::chrono::milliseconds(300);) {
                        ggml_backend_graph_compute(backend, gf);
                        ggml_backend_synchronize(backend);
                    }
                    std::vector<double> t;
                    for (int r = 0; r < reps; ++r) {
                        const auto t0 = std::chrono::steady_clock::now();
                        ggml_backend_graph_compute(backend, gf);
                        ggml_backend_synchronize(backend);
                        t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()/n_copies);
                    }
                    std::sort(t.begin(), t.end());
                    const double us = t[t.size()/2];
                    const double gbs = w_bytes/(us*1e-6)/1e9;
                    printf("%-7s %6lld %6lld %5lld %6d %9.2f %8.0f %6.1f%% %6.1f%%\n", ggml_type_name(type), (long long) K, (long long) N,
                           (long long) nc, n_copies, us, gbs, 100*gbs/1792, 100*gbs/roof);
                    fflush(stdout);
                    ggml_backend_buffer_free(buf);
                    ggml_free(ctx);
                }
            }
        }
    }
    ggml_backend_free(backend);
    return 0;
}
