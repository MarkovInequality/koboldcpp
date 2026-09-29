// Correctness and timing of the GGML_OP_RHT CUDA kernels against ggml_rht_ref.
//
//   bench-rht            check every width class and the Q8_1 outputs, then time the reference shapes (CSV)
//   bench-rht --sweep    also time the runtime variants (warps per CTA, K = 1 chunk width)
//   bench-rht --once     launch each timed variant once, for ncu
//   -n N[,N...], -r ROWS, -k A|B  restrict the timed shapes

#include "ggml-cuda/rht-impl.cuh"
#include "ggml-threading.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

void ggml_critical_section_start(void) {}
void ggml_critical_section_end(void) {}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  in %s at %s:%d\n  %s\n", msg, func, file, line, stmt);
    abort();
}

static const uint64_t SEED    = 0x5eed5eed12345678ull;
static const double   MAX_ERR = 2e-6;

static uint64_t rng_state = 1;
static float frand() {
    rng_state = rng_state*6364136223846793005ull + 1442695040888963407ull;
    return (float) ((rng_state >> 40) & 0xFFFFFF)/(float) 0x800000 - 1.0f;
}

static const char * kernel_name(const rht_shape & sh, int64_t rows, const rht_cfg & cfg, const int out = GGML_CUDA_SRC1_F32) {
    if (rht_use_b(sh, rows, cfg, out)) {
        return "B";
    }
    return sh.multipass ? "A+strided" : sh.M == 1 ? "A1" : "A";
}

static std::string cfg_name(const rht_shape & sh, int64_t rows, const rht_cfg & cfg) {
    char buf[64];
    if (rht_use_b(sh, rows, cfg)) {
        snprintf(buf, sizeof(buf), "Bw%d", cfg.b_warps);
    } else {
        snprintf(buf, sizeof(buf), "Aw%d", cfg.a_warps);
    }
    std::string s = buf;
    if (cfg.generic && sh.K > 1) {
        s += "/gen";
    }
    if (sh.K == 1) {
        s += "/Pc" + std::to_string(sh.Pc);
    }
    return s;
}

// rows of width n with arbitrary row strides (in floats): ne = {n, ne1, ne2, ne3}, stride = {s1, s2, s3}
struct rht_case {
    int64_t n;
    int64_t ne[4];
    int64_t st[4];
    size_t  size() const { return (size_t) (st[3]*ne[3]); }
    int64_t rows() const { return ne[1]*ne[2]*ne[3]; }
    int64_t off(int64_t r) const {
        const int64_t i1 = r % ne[1];
        const int64_t i2 = (r / ne[1]) % ne[2];
        const int64_t i3 = r / (ne[1]*ne[2]);
        return i1*st[1] + i2*st[2] + i3*st[3];
    }
};

static rht_case make_case(int64_t n, int64_t rows, int64_t pad = 0) {
    rht_case c;
    c.n = n;
    c.ne[0] = n; c.ne[1] = rows; c.ne[2] = 1; c.ne[3] = 1;
    c.st[0] = 1; c.st[1] = n + pad; c.st[2] = c.st[1]*rows; c.st[3] = c.st[2];
    return c;
}

static rht_case make_case_4d(int64_t n, int64_t ne1, int64_t ne2, int64_t ne3, int64_t pad1, int64_t pad2, int64_t pad3) {
    rht_case c;
    c.n = n;
    c.ne[0] = n; c.ne[1] = ne1; c.ne[2] = ne2; c.ne[3] = ne3;
    c.st[0] = 1;
    c.st[1] = n + pad1;
    c.st[2] = c.st[1]*ne1 + pad2;
    c.st[3] = c.st[2]*ne2 + pad3;
    return c;
}

static void launch(const rht_shape & sh, const rht_cfg & cfg, const rht_case & c, const float * src, void * dst, float * tmp, cudaStream_t st,
        const ggml_cuda_src1_fmt out = GGML_CUDA_SRC1_F32) {
    size_t nb[4];
    for (int i = 0; i < 4; ++i) {
        nb[i] = c.st[i]*sizeof(float);
    }
    const rht_args a = rht_make_args(sh, src, c.ne, nb, dst, out, SEED, tmp);
    rht_launch(a, sh, cfg, st);
}

// what quantize.cu makes of the F32 rotation: the consumers' own conversion
static void quantize_ref(const rht_case & c, const float * y, void * q, const ggml_cuda_src1_fmt fmt, cudaStream_t st) {
    const int64_t npad = GGML_PAD(c.n, MATRIX_ROW_PADDING);
    const int64_t s1 = c.n;
    const int64_t s2 = c.n*c.ne[1];
    const int64_t s3 = s2*c.ne[2];
    if (fmt == GGML_CUDA_SRC1_Q8_1) {
        quantize_row_q8_1_cuda(y, nullptr, q, GGML_TYPE_Q4_K, c.n, s1, s2, s3, npad, c.ne[1], c.ne[2], c.ne[3], st);
    } else {
        const ggml_type type = fmt == GGML_CUDA_SRC1_Q8_1_MMQ_D4 ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
        quantize_mmq_q8_1_cuda(y, nullptr, q, type, c.n, s1, s2, s3, npad, c.ne[1], c.ne[2], c.ne[3], st);
    }
}

static size_t q8_1_size(const rht_case & c) {
    return (size_t) (c.rows()*GGML_PAD(c.n, MATRIX_ROW_PADDING))*sizeof(block_q8_1)/QK8_1;
}

// max |y - ref| / max |ref| over all rows
static double check(const rht_case & c, const rht_cfg & cfg, cudaStream_t st, const char ** kname = nullptr) {
    rht_shape sh;
    if (!rht_make_shape(c.n, cfg, sh)) {
        fprintf(stderr, "n = %ld not accepted\n", (long) c.n);
        exit(1);
    }
    const int64_t rows = c.rows();

    std::vector<float> x(c.size());
    for (auto & v : x) {
        v = frand();
    }

    float * d_x;
    float * d_y;
    float * d_t = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, x.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t) (rows*c.n)*sizeof(float)));
    const size_t nt = rht_tmp_floats(sh, rows, cfg);
    if (nt) {
        CUDA_CHECK(cudaMalloc(&d_t, nt*sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0xFF, (size_t) (rows*c.n)*sizeof(float)));

    launch(sh, cfg, c, d_x, d_y, d_t, st);
    CUDA_CHECK(cudaStreamSynchronize(st));

    std::vector<float> y((size_t) (rows*c.n));
    CUDA_CHECK(cudaMemcpy(y.data(), d_y, y.size()*sizeof(float), cudaMemcpyDeviceToHost));

    double max_diff = 0.0;
    double max_ref  = 0.0;
    std::vector<float> ref(c.n);
    for (int64_t r = 0; r < rows; ++r) {
        memcpy(ref.data(), x.data() + c.off(r), c.n*sizeof(float));
        ggml_rht_ref(ref.data(), c.n, SEED);
        for (int64_t i = 0; i < c.n; ++i) {
            const double d = fabs((double) y[r*c.n + i] - ref[i]);
            max_diff = std::isnan(d) ? INFINITY : std::max(max_diff, d);
            max_ref  = std::max(max_ref, (double) fabs(ref[i]));
        }
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    if (d_t) {
        CUDA_CHECK(cudaFree(d_t));
    }
    if (kname) {
        *kname = kernel_name(sh, rows, cfg);
    }
    return max_ref > 0 ? max_diff/max_ref : max_diff;
}

static __global__ void sign_words(const uint64_t * seeds, const int64_t * ns, const int64_t * ks, uint64_t * out, int count) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = ggml_rht_sign_word_impl(seeds[i], ns[i], ks[i]);
    }
}

static bool check_sign_mixer() {
    std::vector<uint64_t> seeds;
    std::vector<int64_t>  ns;
    std::vector<int64_t>  ks;
    for (uint64_t seed : std::initializer_list<uint64_t>{0, 1, 42, SEED, 0xFFFFFFFFFFFFFFFFull, 0x8000000000000000ull}) {
        for (int64_t n : {1ll, 64ll, 5120ll, 17408ll, 1ll << 31, (1ll << 40) + 12}) {
            for (int64_t k = 0; k < 600; k += 1 + k/7) {
                seeds.push_back(seed);
                ns.push_back(n);
                ks.push_back(k);
            }
        }
    }
    const int count = (int) seeds.size();

    uint64_t * d_s; int64_t * d_n; int64_t * d_k; uint64_t * d_o;
    CUDA_CHECK(cudaMalloc(&d_s, count*sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_n, count*sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_k, count*sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_o, count*sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_s, seeds.data(), count*sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n, ns.data(),    count*sizeof(int64_t),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, ks.data(),    count*sizeof(int64_t),  cudaMemcpyHostToDevice));
    sign_words<<<(count + 127)/128, 128>>>(d_s, d_n, d_k, d_o, count);
    CUDA_CHECK(cudaGetLastError());

    std::vector<uint64_t> out(count);
    CUDA_CHECK(cudaMemcpy(out.data(), d_o, count*sizeof(uint64_t), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < count; ++i) {
        if (out[i] != ggml_rht_sign_word(seeds[i], ns[i], ks[i])) {
            bad++;
        }
    }
    CUDA_CHECK(cudaFree(d_s)); CUDA_CHECK(cudaFree(d_n)); CUDA_CHECK(cudaFree(d_k)); CUDA_CHECK(cudaFree(d_o));
    printf("# device sign mixer: %d words, %d mismatches\n", count, bad);
    return bad == 0;
}

static int n_checked = 0;
static int n_failed  = 0;

static void expect(const rht_case & c, const rht_cfg & cfg, cudaStream_t st, const char * what) {
    const char * kname;
    const double err = check(c, cfg, st, &kname);
    n_checked++;
    if (!(err <= MAX_ERR)) {
        n_failed++;
        rht_shape sh;
        rht_make_shape(c.n, cfg, sh);
        printf("# FAIL %-10s n=%ld (K=%d P=%ld) rows=%ld kernel=%s err=%.3g\n",
            what, (long) c.n, sh.K, (long) sh.P, (long) c.rows(), kname, err);
    }
}

static void check_all(cudaStream_t st) {
    rht_cfg a;
    a.kernel = 1;
    rht_cfg b;
    b.kernel = 2;

    std::vector<int64_t> widths;
    for (int64_t n = 1; n <= (1 << 15); n *= 2) {
        widths.push_back(n);
    }
    for (int K = 12; K <= 252; K += 8) {
        for (int64_t P = 1; K*P <= (1 << 15); P *= 2) {
            widths.push_back(K*P);
        }
    }
    for (int64_t n : widths) {
        expect(make_case(n, 3), a, st, "A");
        expect(make_case(n, 3), b, st, "B");
    }

    // strided passes: wide K = 1 rows and K > 1 with P > RHT_CHUNK_MAX
    for (int64_t n : {1ll << 16, 1ll << 17, 1ll << 20, 12ll << 11, 12ll << 13, 20ll << 12, 36ll << 11, 252ll << 11}) {
        expect(make_case(n, 2), a, st, "wide");
        expect(make_case(n, 9), b, st, "wide");
    }

    // K = 1 chunk widths
    for (int64_t pc : {32, 64, 128, 256, 512}) {
        rht_cfg ca = a;
        ca.k1_chunk = pc;
        rht_cfg cb = b;
        cb.k1_chunk = pc;
        for (int64_t n : {1ll << 10, 1ll << 12, 1ll << 14, 1ll << 15, 1ll << 17}) {
            expect(make_case(n, 3), ca, st, "k1chunk");
            expect(make_case(n, 3), cb, st, "k1chunk");
        }
    }

    // row counts around the A/B switch, strides, 3D/4D, CTA sizes
    for (int64_t n : {2560ll, 4096ll, 9728ll, 5120ll, 6144ll, 17408ll, 11008ll, 3072ll, 4032ll, 36ll*64}) {
        rht_cfg def;
        for (int64_t rows : {1, 7, 8, 9, 512}) {
            expect(make_case(n, rows), def, st, "rows");
        }
        expect(make_case(n, 5, 3), a, st, "pad");
        expect(make_case(n, 5, 17), b, st, "pad");
        expect(make_case_4d(n, 3, 4, 2, 1, 5, 7), a, st, "4d");
        expect(make_case_4d(n, 3, 4, 2, 1, 5, 7), b, st, "4d");
        for (int w : {1, 2, 8}) {
            rht_cfg cw = a;
            cw.a_warps = w;
            expect(make_case(n, 2), cw, st, "a_warps");
        }
        for (int w : {1, 4, 32}) {
            rht_cfg cw = b;
            cw.b_warps = w;
            expect(make_case(n, 2), cw, st, "b_warps");
        }
    }
    printf("# correctness: %d cases, %d failed (max rel err %.1g)\n", n_checked, n_failed, MAX_ERR);
}

// the Q8_1 an RHT kernel writes must be byte-identical to quantizing its F32 output with quantize.cu
static bool q8_1_identical(const rht_case & c, const rht_cfg & cfg, const ggml_cuda_src1_fmt fmt, cudaStream_t st) {
    rht_shape sh;
    GGML_ASSERT(rht_make_shape(c.n, cfg, sh));
    const int64_t rows = c.rows();
    const size_t  nq   = q8_1_size(c);
    const size_t  ny   = (size_t) (rows*c.n)*sizeof(float);

    std::vector<float> x(c.size());
    for (auto & v : x) {
        v = frand();
    }
    float * d_x;
    float * d_y;
    char  * d_ref;
    char  * d_q;
    float * d_t = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, x.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, ny));
    CUDA_CHECK(cudaMalloc(&d_ref, nq));
    CUDA_CHECK(cudaMalloc(&d_q, std::max(nq, ny)));
    const size_t nt = std::max(rht_tmp_floats(sh, rows, cfg), rht_tmp_floats(sh, rows, cfg, fmt));
    if (nt) {
        CUDA_CHECK(cudaMalloc(&d_t, nt*sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_q, 0xAB, std::max(nq, ny)));

    // the F32 through the kernel that writes the Q8_1: the kernels may sum in different orders
    rht_cfg cfg_f32 = cfg;
    if (!rht_use_b(sh, rows, cfg, fmt)) {
        cfg_f32.kernel = 1;
    }
    launch(sh, cfg_f32, c, d_x, d_y, d_t, st);
    quantize_ref(c, d_y, d_ref, fmt, st);
    launch(sh, cfg, c, d_x, d_q, d_t, st, fmt);
    CUDA_CHECK(cudaStreamSynchronize(st));

    std::vector<char> ref(nq), q(nq);
    CUDA_CHECK(cudaMemcpy(ref.data(), d_ref, nq, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(q.data(),   d_q,   nq, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_q));
    if (d_t) {
        CUDA_CHECK(cudaFree(d_t));
    }
    return memcmp(ref.data(), q.data(), nq) == 0;
}

static void check_q8_1_all(cudaStream_t st) {
    int n_cases = 0;
    int n_bad   = 0;
    for (int64_t n : {2560ll, 5376ll, 11008ll, 4608ll, 12ll << 11, 36ll*16, 252ll*16, 1024ll, 4096ll, 17408ll, 3072ll}) {
        for (int kernel : {1, 2}) {
            rht_cfg cfg;
            cfg.kernel = kernel;
            for (const rht_case & c : {make_case(n, 1), make_case(n, 7), make_case(n, 64), make_case_4d(n, 5, 3, 2, 0, 0, 0)}) {
                for (ggml_cuda_src1_fmt fmt : {GGML_CUDA_SRC1_Q8_1, GGML_CUDA_SRC1_Q8_1_MMQ_D4, GGML_CUDA_SRC1_Q8_1_MMQ_DS4}) {
                    n_cases++;
                    if (!q8_1_identical(c, cfg, fmt, st)) {
                        n_bad++;
                        rht_shape sh;
                        rht_make_shape(n, cfg, sh);
                        printf("# FAIL q8_1 fmt=%d n=%ld rows=%ld kernel=%s\n", (int) fmt, (long) n, (long) c.rows(), kernel_name(sh, c.rows(), cfg, fmt));
                    }
                }
            }
        }
    }
    n_failed += n_bad;
    printf("# q8_1 outputs: %d cases, %d not byte-identical to quantize.cu\n", n_cases, n_bad);
}

static bool once = false;

// best of 5: us per call of f, captured count times in one CUDA graph
template <typename F>
static double time_graph_of(F f, int count, cudaStream_t st) {
    f(0);
    if (once) {
        CUDA_CHECK(cudaStreamSynchronize(st));
        return 0.0;
    }
    cudaGraph_t     graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < count; ++i) {
        f(i);
    }
    CUDA_CHECK(cudaStreamEndCapture(st, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(exec, st));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    const int reps = std::max(1, 4096/count);
    double best = INFINITY;
    for (int k = 0; k < 5; ++k) {
        CUDA_CHECK(cudaEventRecord(e0, st));
        for (int r = 0; r < reps; ++r) {
            CUDA_CHECK(cudaGraphLaunch(exec, st));
        }
        CUDA_CHECK(cudaEventRecord(e1, st));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = std::min(best, 1000.0*ms/(reps*count));
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return best;
}

// decode: 64 dependent launches, each reading the previous output
static double time_dependent(const rht_shape & sh, const rht_cfg & cfg, int64_t rows, cudaStream_t st) {
    const rht_case c = make_case(sh.n, rows);
    float * buf[2];
    float * tmp = nullptr;
    for (auto & b : buf) {
        CUDA_CHECK(cudaMalloc(&b, c.size()*sizeof(float)));
        CUDA_CHECK(cudaMemset(b, 0, c.size()*sizeof(float)));
    }
    const size_t nt = rht_tmp_floats(sh, rows, cfg);
    if (nt) {
        CUDA_CHECK(cudaMalloc(&tmp, nt*sizeof(float)));
    }
    const double us = time_graph_of([&](int i) { launch(sh, cfg, c, buf[i % 2], buf[(i + 1) % 2], tmp, st); }, 64, st);
    for (auto & b : buf) {
        CUDA_CHECK(cudaFree(b));
    }
    if (tmp) {
        CUDA_CHECK(cudaFree(tmp));
    }
    return us;
}

// Q8_1 for the consumers: emitted by the RHT kernel, or F32 followed by quantize.cu's conversion.
// Decode: 64 launches in a graph; prompt processing: buffer sets larger than L2.
static double time_q8_1(const rht_shape & sh, const rht_cfg & cfg, int64_t rows, const ggml_cuda_src1_fmt fmt, bool emit, cudaStream_t st) {
    const rht_case c     = make_case(sh.n, rows);
    const size_t   bytes = c.size()*sizeof(float);
    const int      nbuf  = rows >= 64 ? (int) std::max<size_t>(2, (384u << 20)/(2*bytes) + 1) : 2;

    std::vector<float *> x(nbuf), y(nbuf);
    std::vector<char *>  q(nbuf);
    for (int i = 0; i < nbuf; ++i) {
        CUDA_CHECK(cudaMalloc(&x[i], bytes));
        CUDA_CHECK(cudaMalloc(&y[i], bytes));
        CUDA_CHECK(cudaMalloc(&q[i], std::max(bytes, q8_1_size(c))));
        CUDA_CHECK(cudaMemset(x[i], 0, bytes));
    }
    float * tmp = nullptr;
    const size_t nt = rht_tmp_floats(sh, rows, cfg, emit ? fmt : GGML_CUDA_SRC1_F32);
    if (nt) {
        CUDA_CHECK(cudaMalloc(&tmp, nt*sizeof(float)));
    }

    const double us = time_graph_of([&](int i) {
        const int k = i % nbuf;
        if (emit) {
            launch(sh, cfg, c, x[k], q[k], tmp, st, fmt);
        } else {
            launch(sh, cfg, c, x[k], y[k], tmp, st);
            quantize_ref(c, y[k], q[k], fmt, st);
        }
    }, rows >= 64 ? nbuf : 64, st);

    for (int i = 0; i < nbuf; ++i) {
        CUDA_CHECK(cudaFree(x[i]));
        CUDA_CHECK(cudaFree(y[i]));
        CUDA_CHECK(cudaFree(q[i]));
    }
    if (tmp) {
        CUDA_CHECK(cudaFree(tmp));
    }
    return us;
}

static __global__ void copy_f32(const float4 * x, float4 * y, int64_t n4) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x*blockDim.x) {
        y[i] = x[i];
    }
}

// prompt processing: cycles through buffer sets larger than L2; with copy = true a copy of the same bytes
static double time_streaming(const rht_shape & sh, const rht_cfg & cfg, int64_t rows, cudaStream_t st, bool copy) {
    const rht_case c     = make_case(sh.n, rows);
    const size_t   bytes = c.size()*sizeof(float);
    const int      nbuf  = (int) std::max<size_t>(2, (384u << 20)/(2*bytes) + 1);

    std::vector<float *> x(nbuf), y(nbuf);
    for (int i = 0; i < nbuf; ++i) {
        CUDA_CHECK(cudaMalloc(&x[i], bytes));
        CUDA_CHECK(cudaMalloc(&y[i], bytes));
        CUDA_CHECK(cudaMemset(x[i], 0, bytes));
        CUDA_CHECK(cudaMemset(y[i], 0, bytes));
    }
    float * tmp = nullptr;
    const size_t nt = copy ? 0 : rht_tmp_floats(sh, rows, cfg);
    if (nt) {
        CUDA_CHECK(cudaMalloc(&tmp, nt*sizeof(float)));
    }

    const double us = time_graph_of([&](int i) {
        if (copy) {
            copy_f32<<<1024, 256, 0, st>>>((const float4 *) x[i], (float4 *) y[i], (int64_t) (bytes/16));
            CUDA_CHECK(cudaGetLastError());
        } else {
            launch(sh, cfg, c, x[i], y[i], tmp, st);
        }
    }, nbuf, st);

    for (int i = 0; i < nbuf; ++i) {
        CUDA_CHECK(cudaFree(x[i]));
        CUDA_CHECK(cudaFree(y[i]));
    }
    if (tmp) {
        CUDA_CHECK(cudaFree(tmp));
    }
    return us;
}

static void bench(int64_t n, const rht_cfg & cfg, int64_t rows, cudaStream_t st) {
    rht_shape sh;
    GGML_ASSERT(rht_make_shape(n, cfg, sh));

    const double err = check(make_case(n, std::min<int64_t>(rows, 64)), cfg, st);
    if (!(err <= MAX_ERR)) {
        n_failed++;
    }

    const bool   pp    = rows >= 64;
    const double us    = pp ? time_streaming(sh, cfg, rows, st, false) : time_dependent(sh, cfg, rows, st);
    const double bytes = 2.0*n*rows*sizeof(float);
    double gbs = 0.0, copy_gbs = 0.0;
    if (pp && !once) {
        gbs      = bytes/us*1e-3;
        copy_gbs = bytes/time_streaming(sh, cfg, rows, st, true)*1e-3;
    }
    printf("%ld,%d,%ld,%ld,%s,%s,%.2g,%.2f,%.0f,%.0f,%.0f\n", (long) n, sh.K, (long) sh.P, (long) rows,
        kernel_name(sh, rows, cfg), cfg_name(sh, rows, cfg).c_str(), err, us, gbs, copy_gbs, copy_gbs > 0 ? 100.0*gbs/copy_gbs : 0.0);
    fflush(stdout);
}

// rows of a Q4_K consumer on this GPU: MMVQ for one row, MMQ (DS4) otherwise
static void bench_q8_1(int64_t n, const rht_cfg & cfg, int64_t rows, cudaStream_t st) {
    rht_shape sh;
    GGML_ASSERT(rht_make_shape(n, cfg, sh));
    const ggml_cuda_src1_fmt fmt = rows == 1 ? GGML_CUDA_SRC1_Q8_1 : GGML_CUDA_SRC1_Q8_1_MMQ_DS4;
    const char * fmt_name = rows == 1 ? "q8_1" : "q8_1_mmq";

    for (bool emit : {false, true}) {
        const double us = time_q8_1(sh, cfg, rows, fmt, emit, st);
        const double gbs = rows >= 64 && !once ? (double) (n*rows*sizeof(float) + q8_1_size(make_case(n, rows)))/us*1e-3 : 0.0;
        printf("%ld,%d,%ld,%ld,%s,%s%s,,%.2f,%.0f,,\n", (long) n, sh.K, (long) sh.P, (long) rows, kernel_name(sh, rows, cfg, emit ? fmt : GGML_CUDA_SRC1_F32),
            emit ? "emit_" : "f32+quantize_", fmt_name, us, gbs);
    }
    fflush(stdout);
}

int main(int argc, char ** argv) {
    bool    sweep      = false;
    bool    skip_check = false;
    std::vector<int64_t> only_n;
    int64_t only_rows  = 0;
    int     only_k     = 0;
    int     a_warps    = rht_cfg().a_warps;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--once")) {
            once = true;
        } else if (!strcmp(argv[i], "--sweep")) {
            sweep = true;
        } else if (!strcmp(argv[i], "--no-check")) {
            skip_check = true;
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            for (char * p = argv[++i]; *p; ) {
                only_n.push_back(strtoll(p, &p, 10));
                p += *p == ',';
            }
        } else if (!strcmp(argv[i], "-r") && i + 1 < argc) {
            only_rows = atoll(argv[++i]);
        } else if (!strcmp(argv[i], "-aw") && i + 1 < argc) {
            a_warps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-k") && i + 1 < argc) {
            only_k = argv[++i][0] == 'A' ? 1 : 2;
        } else {
            fprintf(stderr, "usage: %s [--once] [--sweep] [--no-check] [-n width[,width...]] [-r rows] [-k A|B] [-aw warps]\n", argv[0]);
            return 1;
        }
    }

    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreate(&st));

    bool ok = check_sign_mixer();
    if (!skip_check && !once) {
        check_all(st);
        check_q8_1_all(st);
    }

    std::vector<int64_t> shapes = { 2560, 4096, 9728, 5120, 6144, 17408, 11008, 3072 };
    if (!only_n.empty()) {
        shapes = only_n;
    }

    printf("n,K,P,rows,kernel,variant,err,us,GBps,copy_GBps,pct_copy\n");
    for (int64_t n : shapes) {
        rht_cfg a;
        a.kernel  = 1;
        a.a_warps = a_warps;
        rht_cfg b;
        b.kernel = 2;
        for (int64_t rows : only_rows ? std::vector<int64_t>{only_rows} : std::vector<int64_t>{1, 8, 512}) {
            if (only_k != 2) {
                bench(n, a, rows, st);
            }
            if (only_k != 1) {
                bench(n, b, rows, st);
            }
            bench_q8_1(n, rht_cfg(), rows, st);
        }
        if (sweep) {
            rht_shape sh;
            rht_make_shape(n, rht_cfg(), sh);
            for (int w : {1, 2, 8}) {
                rht_cfg c = a;
                c.a_warps = w;
                bench(n, c, 1, st);
            }
            for (int w : {4, 8, 16, 32}) {
                rht_cfg c = b;
                c.b_warps = w;
                bench(n, c, 512, st);
            }
            if (sh.K > 1 && sh.mix > 1) {
                rht_cfg ga = a;
                ga.generic = true;
                rht_cfg gb = b;
                gb.generic = true;
                bench(n, ga, 1, st);
                bench(n, gb, 512, st);
            }
            if (sh.K == 1) {
                for (int64_t pc : {128, 256, 512}) {
                    rht_cfg ca = a;
                    ca.k1_chunk = pc;
                    rht_cfg cb = b;
                    cb.k1_chunk = pc;
                    bench(n, ca, 1, st);
                    bench(n, cb, 512, st);
                }
            }
        }
    }

    CUDA_CHECK(cudaStreamDestroy(st));

    ok = ok && n_failed == 0;
    printf("# %s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
