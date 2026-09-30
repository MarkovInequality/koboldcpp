#pragma once

// Kernels of GGML_OP_RHT: y = (1/sqrt(n)) * (H_K (x) H_P) * diag(s) * x per row, natural element order
// i = c*P + j (see ggml_rht_ref). Included by rht.cu and tests/bench-rht.cu only: the Hadamard tables
// below are device data, which every including translation unit would carry.

#include "common.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "ggml-hadamard.h"

#include <type_traits>
#include <utility>

#define GGML_HAD_DECL(K) static constexpr char rht_had_str_##K[]
#include "ggml-hadamard-tables.h"
#undef GGML_HAD_DECL

// orders whose mix is unrolled at compile time; every other table order runs the generic mix
#define RHT_UNROLLED_ORDERS(X) X(12) X(20) X(28) X(68) X(76) X(84) X(100) X(108) X(148) X(172)

#define RHT_CHUNK_MAX   1024 // widest chunk a warp transforms in registers
#define RHT_MIX_MAX     32   // widest power-of-2 mix (K = 1)
#define RHT_A_WARPS_MAX 8
#define RHT_B_WARPS_MAX 32
#define RHT_B_REGS      32   // row elements a kernel-B lane holds
#define RHT_B_SMEM      (48*1024)
#define RHT_B_MAX       (WARP_SIZE*RHT_B_WARPS_MAX*RHT_B_REGS) // widest row of kernel B
#define RHT_GA          4
#define RHT_GB          8

// ±1 bits of H_K: bit c%32 of w[r*NW + c/32] is set iff H[r][c] = -1 (r = row of the header's strings)
template <int K>
struct rht_bits {
    static constexpr int NW = (K + 31)/32;
    uint32_t w[K*NW];
};

template <int K>
static constexpr rht_bits<K> rht_pack_bits(const char (&rows)[K*K + 1]) {
    rht_bits<K> b = {};
    for (int r = 0; r < K; ++r) {
        for (int c = 0; c < K; ++c) {
            if (rows[r*K + c] == '-') {
                b.w[r*rht_bits<K>::NW + c/32] |= 1u << (c % 32);
            }
        }
    }
    return b;
}

#define RHT_BITS_DECL(K) static __device__ const rht_bits<K> rht_bits_##K = rht_pack_bits<K>(rht_had_str_##K);
GGML_HAD_ORDERS(RHT_BITS_DECL)
#undef RHT_BITS_DECL

static __device__ __forceinline__ const uint32_t * rht_bits_ptr(const int K) {
    switch (K) {
#define RHT_BITS_CASE(K) case K: return rht_bits_##K.w;
        GGML_HAD_ORDERS(RHT_BITS_CASE)
#undef RHT_BITS_CASE
        default: return nullptr;
    }
}

// constant (r, c) folds the table load away, leaving FADDs with sign modifiers
template <int K> struct rht_had;
#define RHT_HAD_DECL(K)                                                                          \
    template <> struct rht_had<K> {                                                              \
        static __device__ __forceinline__ bool neg(const int r, const int c) {                   \
            return (rht_bits_##K.w[r*rht_bits<K>::NW + c/32] >> (c % 32)) & 1;                   \
        }                                                                                        \
    };
RHT_UNROLLED_ORDERS(RHT_HAD_DECL)
#undef RHT_HAD_DECL

// What the kernels rotate in place of the source rows x: x*w, scaled by 1/rms(x) after the transform (a fused
// RMS_NORM + MUL), or act(g)*u (a fused GLU, g = x)
enum rht_prologue {
    RHT_PRO_NONE,
    RHT_PRO_NORM,
    RHT_PRO_SWIGLU,
    RHT_PRO_GEGLU,
};

struct rht_args {
    const char * src;
    float      * tmp;   // rows*n floats for the passes of kernel A, then with RHT_PRO_NORM rows*units sums of squares
    void       * dst;
    int          out;   // ggml_cuda_src1_fmt
    int          K;     // table order, for the generic mix
    int          M;     // mix order: K, or the number of chunks of a K = 1 row
    int          nrows;
    int          S;     // kernel B: slabs per group
    int64_t      n;
    int64_t      Pc;    // chunk transformed by one warp: n = M*Pm, Pm = Pc * (strided passes)
    int64_t      Pm;    // stride of the mix inputs
    int64_t      npad;  // Q8_1 row width, n padded to MATRIX_ROW_PADDING
    int64_t      nb1, nb2, nb3;
    uint3        ne1, ne2;
    uint64_t     seed;
    float        scale;
    int          pro;   // rht_prologue
    int          units; // units per row (kernel A pass 1)
    const char * src2;  // RHT_PRO_NORM: w (n floats); GLU: u, rows at nb21, nb22, nb23 like the source's
    int64_t      nb21, nb22, nb23;
    float        eps;
};

static __device__ __forceinline__ const float * rht_src_row(const rht_args & a, const uint32_t row) {
    const uint2 q1 = fast_div_modulo(row,  a.ne1);
    const uint2 q2 = fast_div_modulo(q1.x, a.ne2);
    return (const float *) (a.src + q1.y*a.nb1 + q2.y*a.nb2 + q2.x*a.nb3);
}

static __device__ __forceinline__ const float * rht_src2_row(const rht_args & a, const uint32_t row) {
    if (a.pro == RHT_PRO_NORM) {
        return (const float *) a.src2;
    }
    const uint2 q1 = fast_div_modulo(row,  a.ne1);
    const uint2 q2 = fast_div_modulo(q1.x, a.ne2);
    return (const float *) (a.src2 + q1.y*a.nb21 + q2.y*a.nb22 + q2.x*a.nb23);
}

static __device__ __forceinline__ float rht_norm_scale(const rht_args & a, const float sumsq) {
    return rsqrtf(sumsq/a.n + a.eps);
}

// the row's sum of squares from kernel A pass 1, added in the same order by every warp
static __device__ __forceinline__ float rht_row_sumsq(const rht_args & a, const int64_t row, const int lane) {
    const float * ps = a.tmp + a.n*a.nrows + row*a.units;
    float sum = 0.0f;
    for (int u = lane; u < a.units; u += WARP_SIZE) {
        sum += ps[u];
    }
    return warp_reduce_sum(sum);
}

static __device__ __forceinline__ float rht_flip(const float v, const uint32_t mask, const int bit) {
    return __int_as_float(__float_as_int(v) ^ ((mask << (31 - bit)) & 0x80000000u));
}

static __device__ __forceinline__ void rht_store_f32(
        const rht_args & a, const float scale, const int64_t row, const int64_t e0, const int lane, const float v) {
    const int64_t i = e0 + lane;
    if (i < a.n) {
        ((float *) a.dst)[row*a.n + i] = v*scale;
    }
}

// lane t < 8 packs values 4t ... 4t+3 of the block, as a thread of quantize_mmq_q8_1 does
template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void rht_store_mmq(
        void * dst, const uint2 ci, const int64_t ne1, const int64_t npad, const int64_t e0, const int lane, const float x) {
    const int t = lane % 8;
    float4 x4;
    x4.x = __shfl_sync(0xFFFFFFFF, x, 4*t + 0, WARP_SIZE);
    x4.y = __shfl_sync(0xFFFFFFFF, x, 4*t + 1, WARP_SIZE);
    x4.z = __shfl_sync(0xFFFFFFFF, x, 4*t + 2, WARP_SIZE);
    x4.w = __shfl_sync(0xFFFFFFFF, x, 4*t + 3, WARP_SIZE);
    const q8_1_mmq_vals v = q8_1_mmq_quantize4<ds_layout>(x4);
    if (lane < 8) {
        q8_1_mmq_store4<ds_layout>(q8_1_mmq_block(dst, ci.x, ci.y, e0, ne1, npad), (int) (e0 % QK8_1_MMQ) + 4*t, v);
    }
}

// Q8_1 blocks of up to 4 output chunks, element e0 + k*Pm + lane of chunk k. Out of line: inlined next to
// the unrolled mixes, the epilogues slow down their F32 path.
static __device__ __noinline__ void rht_store_q8_1(
        void * dst, const int out, const uint3 ne1, const int64_t npad, const float scale,
        const int64_t row, const int64_t e0, const int64_t Pm, const int count, const int lane, const float4 v) {
    const uint2 ci  = fast_div_modulo((uint32_t) row, ne1);
    const float x[4] = { v.x*scale, v.y*scale, v.z*scale, v.w*scale };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (k < count) {
            switch (out) {
                case GGML_CUDA_SRC1_Q8_1:
                    q8_1_quantize_block(q8_1_block(dst, row, e0 + k*Pm, npad), lane, x[k]);
                    break;
                case GGML_CUDA_SRC1_Q8_1_MMQ_D4:
                    rht_store_mmq<MMQ_Q8_1_DS_LAYOUT_D4>(dst, ci, ne1.z, npad, e0 + k*Pm, lane, x[k]);
                    break;
                case GGML_CUDA_SRC1_Q8_1_MMQ_DS4:
                    rht_store_mmq<MMQ_Q8_1_DS_LAYOUT_DS4>(dst, ci, ne1.z, npad, e0 + k*Pm, lane, x[k]);
                    break;
            }
        }
    }
}

// Every store covers 32 consecutive elements of an output chunk, one block_q8_1, with all lanes of the warp:
// lane stores element c*Pm + e0 + lane for the chunks c0 ... c0 + count-1 (count <= G). Q8_1 needs n % 32 == 0.
template <int G>
static __device__ __forceinline__ void rht_store(const rht_args & a, const float scale,
        const int64_t row, const int64_t e0, const int lane, const int c0, const int count, const float (&v)[G]) {
    if (a.out == GGML_CUDA_SRC1_F32) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (g < count) {
                rht_store_f32(a, scale, row, (c0 + g)*a.Pm + e0, lane, v[g]);
            }
        }
        return;
    }
#pragma unroll
    for (int g = 0; g < G; g += 4) {
        if (g < count) {
            const float4 v4 = make_float4(v[g], g + 1 < G ? v[g + 1] : 0.0f, g + 2 < G ? v[g + 2] : 0.0f, g + 3 < G ? v[g + 3] : 0.0f);
            rht_store_q8_1(a.dst, a.out, a.ne1, a.npad, scale, row, (c0 + g)*a.Pm + e0, a.Pm, min(count - g, 4), lane, v4);
        }
    }
}

// the zero blocks of values n ... npad-1 of a Q8_1 row, as the quantize kernels write them for x = 0
static __device__ __noinline__ void rht_store_pad(void * dst, const int out, const uint3 ne1, const int64_t n, const int64_t npad,
        const int64_t row, const int lane) {
    const uint2 ci = fast_div_modulo((uint32_t) row, ne1);
    for (int64_t i = n + 4*lane; i < npad; i += 4*WARP_SIZE) {
        if (out == GGML_CUDA_SRC1_Q8_1) {
            block_q8_1 * b = q8_1_block(dst, row, i, npad);
            *(int *) (b->qs + i % QK8_1) = 0;
            if (i % QK8_1 == 0) {
                b->ds = make_half2(0.0f, 0.0f);
            }
        } else {
            block_q8_1_mmq * b = q8_1_mmq_block(dst, ci.x, ci.y, i, ne1.z, npad);
            *(int *) (b->qs + i % QK8_1_MMQ) = 0;
            if (i % QK8_1 == 0) {
                b->d4[(i % QK8_1_MMQ)/QK8_1] = 0.0f;
            }
        }
    }
}

// One lane quantizes the 32 values x of the block at element i0 alone, in the arithmetic and summation order of
// the warp-wide quantize kernels, so the bytes are the same.
static __device__ __noinline__ void rht_quantize_block_lane(
        void * dst, const int out, const uint3 ne1, const int64_t npad, const int64_t row, const int64_t i0, const float * x) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < QK8_1; ++i) {
        amax = fmaxf(amax, fabsf(x[i]));
    }
    if (out == GGML_CUDA_SRC1_Q8_1) {
        // warp_reduce_sum's xor tree
        float s[QK8_1];
#pragma unroll
        for (int i = 0; i < QK8_1; ++i) {
            s[i] = x[i];
        }
#pragma unroll
        for (int off = QK8_1/2; off > 0; off >>= 1) {
#pragma unroll
            for (int i = 0; i < off; ++i) {
                s[i] += s[i + off];
            }
        }
        block_q8_1 * y = q8_1_block(dst, row, i0, npad);
        const float d = amax / 127.0f;
#pragma unroll
        for (int i = 0; i < QK8_1; i += 4) {
            char4 q;
            q.x = amax == 0.0f ? 0 : roundf(x[i + 0] / d);
            q.y = amax == 0.0f ? 0 : roundf(x[i + 1] / d);
            q.z = amax == 0.0f ? 0 : roundf(x[i + 2] / d);
            q.w = amax == 0.0f ? 0 : roundf(x[i + 3] / d);
            *(char4 *) (y->qs + i) = q;
        }
        y->ds = make_half2(d, s[0]);
        return;
    }
    const uint2 ci = fast_div_modulo((uint32_t) row, ne1);
    block_q8_1_mmq * y = q8_1_mmq_block(dst, ci.x, ci.y, i0, ne1.z, npad);
    const int iqs = (int) (i0 % QK8_1_MMQ);
    const float d_inv = 127.0f / amax;
#pragma unroll
    for (int i = 0; i < QK8_1; i += 4) {
        char4 q;
        q.x = roundf(x[i + 0]*d_inv);
        q.y = roundf(x[i + 1]*d_inv);
        q.z = roundf(x[i + 2]*d_inv);
        q.w = roundf(x[i + 3]*d_inv);
        ((char4 *) y->qs)[(iqs + i)/4] = q;
    }
    const float d = 1.0f / d_inv;
    if (out == GGML_CUDA_SRC1_Q8_1_MMQ_DS4) {
        // q8_1_mmq_quantize4: sums of 4 per thread, then an xor tree over the 8 threads of the block
        float s[QK8_1/4];
#pragma unroll
        for (int j = 0; j < QK8_1/4; ++j) {
            s[j] = x[4*j] + x[4*j + 1] + x[4*j + 2] + x[4*j + 3];
        }
#pragma unroll
        for (int off = QK8_1/8; off > 0; off >>= 1) {
#pragma unroll
            for (int j = 0; j < off; ++j) {
                s[j] += s[j + off];
            }
        }
        y->ds4[iqs/QK8_1] = make_half2(d, s[0]);
    } else {
        y->d4[iqs/QK8_1] = d;
    }
}

// A warp holds a unit of E*32 row elements starting at base, lane-strided: v[e] = x[base + 32*e + lane].
// For E > 1 the unit is one chunk (Pc = 32*E); for E = 1 it may hold several chunks (Pc <= 32).

// bit e set: element base + 32*e + lane is negated
template <int E>
static __device__ __forceinline__ uint32_t rht_sign_mask(const uint64_t seed, const int64_t n, const int64_t base, const int lane) {
    if constexpr (E == 1) {
        const uint64_t w = ggml_rht_sign_word_impl(seed, n, base >> 6);
        const uint32_t h = base & 32 ? (uint32_t) (w >> 32) : (uint32_t) w;
        return (h >> lane) & 1;
    } else {
        const uint64_t w  = ggml_rht_sign_word_impl(seed, n, (base >> 6) + (lane & (E/2 - 1)));
        const uint32_t lo = (uint32_t) w;
        const uint32_t hi = (uint32_t) (w >> 32);
        uint32_t m = 0;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const uint32_t h = __shfl_sync(0xFFFFFFFF, e & 1 ? hi : lo, e/2, WARP_SIZE);
            m |= ((h >> lane) & 1) << e;
        }
        return m;
    }
}

template <int E>
static __device__ __forceinline__ void rht_load(float * v, const float * x, const int64_t base, const int lane, const int64_t n) {
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int64_t i = base + e*WARP_SIZE + lane;
        v[e] = E > 1 || i < n ? x[i] : 0.0f;
    }
}

// rht_load through the prologue; ss collects x^2 for RHT_PRO_NORM. Same arithmetic as the GLU kernels.
template <int E>
static __device__ __forceinline__ void rht_load_pro(
        float * v, const rht_args & a, const float * x, const float * x2, const int64_t base, const int lane, float & ss) {
    float u[E];
    rht_load<E>(v, x, base, lane, a.n);
    rht_load<E>(u, x2, base, lane, a.n);
    switch (a.pro) {
        case RHT_PRO_NORM:
#pragma unroll
            for (int e = 0; e < E; ++e) {
                ss  += v[e]*v[e];
                v[e] *= u[e];
            }
            break;
        case RHT_PRO_SWIGLU:
#pragma unroll
            for (int e = 0; e < E; ++e) {
                v[e] = ggml_cuda_op_silu_single(v[e])*u[e];
            }
            break;
        default:
#pragma unroll
            for (int e = 0; e < E; ++e) {
                v[e] = ggml_cuda_op_gelu_single(v[e])*u[e];
            }
            break;
    }
}

// natural-order Sylvester H_Pc on each chunk of the unit: lane bits first, then register bits
template <int E>
static __device__ __forceinline__ void rht_fwht(float * v, const int lane, const int64_t Pc) {
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h <<= 1) {
        if (h < Pc) {
            const uint32_t flip = lane & h ? 0x80000000u : 0u;
#pragma unroll
            for (int e = 0; e < E; ++e) {
                const float o = __shfl_xor_sync(0xFFFFFFFF, v[e], h, WARP_SIZE);
                v[e] = o + __int_as_float(__float_as_int(v[e]) ^ flip);
            }
        }
    }
#pragma unroll
    for (int h = 1; h < E; h <<= 1) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            if (!(e & h)) {
                const float x0 = v[e];
                const float x1 = v[e + h];
                v[e]     = x0 + x1;
                v[e + h] = x0 - x1;
            }
        }
    }
}

// E up to EMAX only: kernel B takes n <= RHT_B_MAX, so an unrolled order K has chunks of at most RHT_B_MAX/K
template <int EMAX, typename F>
static __device__ __forceinline__ void rht_switch_e(const int E, F f) {
    switch (E) {
        case  1: f(std::integral_constant<int, 1>{}); break;
        case  2: if constexpr (EMAX >=  2) { f(std::integral_constant<int,  2>{}); } break;
        case  4: if constexpr (EMAX >=  4) { f(std::integral_constant<int,  4>{}); } break;
        case  8: if constexpr (EMAX >=  8) { f(std::integral_constant<int,  8>{}); } break;
        case 16: if constexpr (EMAX >= 16) { f(std::integral_constant<int, 16>{}); } break;
        case 32: if constexpr (EMAX >= 32) { f(std::integral_constant<int, 32>{}); } break;
    }
}

static constexpr __host__ __device__ int rht_pow2_floor(const int x) {
    return x >= 32 ? 32 : x >= 16 ? 16 : x >= 8 ? 8 : x >= 4 ? 4 : x >= 2 ? 2 : 1;
}

// Mixes. ld(c) returns the lane's input from chunk c; they produce the outputs of chunks t*G ... t*G + G-1.

// the sums over the input chunks C0 ... C1-1 of tile T
template <int K, int G, int T, int C0, int C1, typename F>
static __device__ __forceinline__ void rht_mix_tile(F ld, float (&acc)[G]) {
#pragma unroll
    for (int c = C0; c < C1; ++c) {
        const float v = ld(c);
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (T*G + g < K) {
                const float s = rht_had<K>::neg(T*G + g, c) ? -v : v;
                acc[g] = c == C0 ? s : acc[g] + s;
            }
        }
    }
}

template <int K, int G, int KS, int T, typename F, int... P>
static __device__ __forceinline__ void rht_mix_tile_part(const int p, F ld, float (&acc)[G], std::integer_sequence<int, P...>) {
    ((p == P ? rht_mix_tile<K, G, T, P*(K/KS), (P + 1)*(K/KS)>(ld, acc) : void()), ...);
}

// part p of KS of tile t; one instance per tile and part keeps each unrolled body under the compiler's
// full-unroll limit
template <int K, int G, int KS, typename F, int... T>
static __device__ __forceinline__ void rht_mix_unrolled(const int t, const int p, F ld, float (&acc)[G], std::integer_sequence<int, T...>) {
    ((t == T ? rht_mix_tile_part<K, G, KS, T>(p, ld, acc, std::make_integer_sequence<int, KS>{}) : void()), ...);
}

// Kernel A's pass 2 splits the input chunks of the larger orders over KS warps: each warp runs a different
// stretch of unrolled code, and one warp alone waits on the instruction fetches of all of it. Order 76 has an
// unsplit instance too, faster for it beyond 2 rows (RTX 5090).
static constexpr __host__ __device__ int rht_ksplit(const int mix) {
    return mix >= 68 ? 4 : 1;
}

static constexpr int rht_ksplit_rows_max(const int mix) {
    return mix == 76 ? 2 : INT_MAX;
}

// outputs of the rows r0 ... r0 + G-1 of H_K (r0 may differ between lanes); rows >= K give zeros
template <int G, typename F>
static __device__ __forceinline__ void rht_mix_generic(const uint32_t * bits, const int K, const int r0, F ld, float (&acc)[G]) {
    const int        NW = (K + 31)/32;
    const uint32_t * rw = bits + r0*NW;

#pragma unroll
    for (int g = 0; g < G; ++g) {
        acc[g] = 0.0f;
    }

    for (int cw = 0; cw < NW; ++cw) {
        uint32_t wd[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            wd[g] = r0 + g < K ? __brev(rw[g*NW + cw]) : 0;
        }
        const int nc = min(32, K - cw*32);
#pragma unroll 8
        for (int cb = 0; cb < nc; ++cb) {
            const float v = ld(cw*32 + cb);
#pragma unroll
            for (int g = 0; g < G; ++g) {
                acc[g] += __int_as_float(__float_as_int(v) ^ (wd[g] & 0x80000000u));
                wd[g] <<= 1;
            }
        }
    }
}

template <int M, typename F>
static __device__ __forceinline__ void rht_mix_sylvester(F ld, float * v) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
        v[m] = ld(m);
    }
#pragma unroll
    for (int h = 1; h < M; h <<= 1) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            if (!(m & h)) {
                const float x0 = v[m];
                const float x1 = v[m + h];
                v[m]     = x0 + x1;
                v[m + h] = x0 - x1;
            }
        }
    }
}

// output chunks per warp for a kernel's G: the generic mix holds a sign word per chunk as well
static constexpr __host__ __device__ int rht_tile_rows(const int mix, const int G) {
    return mix == 0 && G > 4 ? 4 : G;
}

// MIX: an unrolled order, 0 = generic, 1 = Sylvester H_M (K = 1). st(c0, count, v) stores the lane's outputs
// of chunks c0 ... c0 + count-1, v[] holding at least G of them.
template <int MIX, int G, typename F, typename S>
static __device__ __forceinline__ void rht_mix_out(const rht_args & a, const int t, const uint32_t * bits, F ld, S st) {
    if constexpr (MIX == 1) {
        float v[RHT_MIX_MAX];
        switch (a.M) {
            case  1: v[0] = ld(0);                    break;
            case  2: rht_mix_sylvester< 2>(ld, v); break;
            case  4: rht_mix_sylvester< 4>(ld, v); break;
            case  8: rht_mix_sylvester< 8>(ld, v); break;
            case 16: rht_mix_sylvester<16>(ld, v); break;
            case 32: rht_mix_sylvester<32>(ld, v); break;
        }
#pragma unroll
        for (int m = 0; m < RHT_MIX_MAX; m += 4) {
            if (m < a.M) {
                const float v4[4] = { v[m], v[m + 1], v[m + 2], v[m + 3] };
                st(m, min(a.M - m, 4), v4);
            }
        }
    } else {
        float acc[G];
        if constexpr (MIX == 0) {
            rht_mix_generic<G>(bits, a.K, t*G, ld, acc);
        } else {
            rht_mix_unrolled<MIX, G, 1>(t, 0, ld, acc, std::make_integer_sequence<int, (MIX + G - 1)/G>{});
        }
        st(t*G, min(a.M - t*G, G), acc);
    }
}

// Chunks narrower than a warp (Pm < 32, generic mix only): each lane computes one output of a 32-element
// block, so its output row of H_K depends on the lane. ld(i) returns element i of the transformed row;
// returns the lane's output of elements 32*b ... 32*b + 31.
template <typename F>
static __device__ __forceinline__ float rht_mix_narrow(const rht_args & a, const int64_t b, const int lane, const uint32_t * bits, F ld) {
    const int64_t e = min(b*WARP_SIZE + lane, a.n - 1);
    const int64_t j = e % a.Pm;
    float acc[1];
    rht_mix_generic<1>(bits, a.K, (int) (e / a.Pm), [&](const int c) { return ld(c*a.Pm + j); }, acc);
    return acc[0];
}

// Kernel A (few rows): spread each row over many SMs in two passes through a scratch buffer in L2.
// One CTA per row would run all K*n adds of the mix on one SM, while the GEMV that follows uses all of them.

// pass 1: one warp per (unit, row): signs and H_Pc; writes tmp, or dst when M == 1
template <int E>
__launch_bounds__(RHT_A_WARPS_MAX*WARP_SIZE)
static __global__ void rht_chunks(const rht_args a) {
    const int     lane  = threadIdx.x;
    const int64_t units = (a.n + E*WARP_SIZE - 1)/(E*WARP_SIZE);
    const int64_t w     = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;

    ggml_cuda_pdl_lc();
    if (w >= units*a.nrows) {
        return;
    }

    const int      row  = w / units;
    const int64_t  base = (w % units)*E*WARP_SIZE;
    const uint32_t sg   = rht_sign_mask<E>(a.seed, a.n, base, lane);

    ggml_cuda_pdl_sync();

    float v[E];
    float ss = 0.0f;
    if (a.pro == RHT_PRO_NONE) {
        rht_load<E>(v, rht_src_row(a, row), base, lane, a.n);
    } else {
        rht_load_pro<E>(v, a, rht_src_row(a, row), rht_src2_row(a, row), base, lane, ss);
    }
#pragma unroll
    for (int e = 0; e < E; ++e) {
        v[e] = rht_flip(v[e], sg, e);
    }
    rht_fwht<E>(v, lane, a.Pc);

    if (a.M == 1) {
        // the whole row is this warp's unit
        const float scale = a.pro == RHT_PRO_NORM ? a.scale*rht_norm_scale(a, warp_reduce_sum(ss)) : a.scale;
        if (a.out == GGML_CUDA_SRC1_F32) {
#pragma unroll
            for (int e = 0; e < E; ++e) {
                rht_store_f32(a, scale, row, base + e*WARP_SIZE, lane, v[e]);
            }
            return;
        }
        // block e is v[e] across the lanes: transposed, lane e quantizes it alone instead of E warp reductions in turn
        extern __shared__ float rht_tr[];
        float * t = rht_tr + threadIdx.y*WARP_SIZE*(WARP_SIZE + 1);
#pragma unroll
        for (int e = 0; e < E; ++e) {
            t[e*(WARP_SIZE + 1) + lane] = v[e]*scale;
        }
        __syncwarp();
        if (lane < E) {
            rht_quantize_block_lane(a.dst, a.out, a.ne1, a.npad, row, (int64_t) lane*WARP_SIZE, t + lane*(WARP_SIZE + 1));
        }
        rht_store_pad(a.dst, a.out, a.ne1, a.n, a.npad, row, lane);
        return;
    }

    float * y = a.tmp + row*a.n;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int64_t i = base + e*WARP_SIZE + lane;
        if (E > 1 || i < a.n) {
            y[i] = v[e];
        }
    }
    if (a.pro == RHT_PRO_NORM) {
        ss = warp_reduce_sum(ss);
        if (lane == 0) {
            a.tmp[a.n*a.nrows + row*a.units + w % units] = ss;
        }
    }
}

// rows with Pm > Pc: H_L at stride s, in place in tmp, one thread per L elements
template <int L>
__launch_bounds__(256)
static __global__ void rht_strided(const rht_args a, const int64_t s) {
    const int64_t per_row = a.n/L;
    const int64_t t       = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    ggml_cuda_pdl_lc();
    if (t >= per_row*a.nrows) {
        return;
    }

    const int64_t row = t / per_row;
    const int64_t k   = t % per_row;
    float * x = a.tmp + row*a.n + (k / s)*L*s + k % s;

    ggml_cuda_pdl_sync();

    float v[L];
    rht_mix_sylvester<L>([&](const int l) { return x[l*s]; }, v);
#pragma unroll
    for (int l = 0; l < L; ++l) {
        x[l*s] = v[l];
    }
}

// pass 2: one warp per (32 positions j, tile of G output chunks, row): mix, scale, epilogue.
// With KS > 1, KS consecutive warps of a CTA each sum a part of the input chunks of one job.
template <int MIX, int G, int KS>
__launch_bounds__(RHT_A_WARPS_MAX*WARP_SIZE)
static __global__ void rht_mix_split(const rht_args a) {
    static_assert(KS == 1 || KS == rht_tile_rows(MIX, G), "each warp of a split job finishes one output chunk");
    const int     lane   = threadIdx.x;
    const int64_t w      = ((int64_t) blockIdx.x*blockDim.y + threadIdx.y)/KS;
    const bool    narrow = MIX == 0 && a.Pm < WARP_SIZE;
    const int64_t slabs  = narrow ? (a.n + WARP_SIZE - 1)/WARP_SIZE : a.Pm/WARP_SIZE;
    constexpr int GT     = rht_tile_rows(MIX, G);
    const int     NT     = MIX == 1 || narrow ? 1 : (a.M + GT - 1)/GT;
    const int64_t jobs   = slabs*NT;

    ggml_cuda_pdl_lc();

    if constexpr (KS > 1) {
        __shared__ float part_acc[RHT_A_WARPS_MAX][GT][WARP_SIZE];

        const int  p      = threadIdx.y % KS;
        const bool active = w < jobs*a.nrows;
        const int     t   = w / (slabs*a.nrows);
        const int     row = active ? (w / slabs) % a.nrows : 0;
        const int64_t s   = w % slabs;

        ggml_cuda_pdl_sync();

        if (active && p == 0 && t == 0 && s == 0 && a.out != GGML_CUDA_SRC1_F32) {
            rht_store_pad(a.dst, a.out, a.ne1, a.n, a.npad, row, lane);
        }
        const float scale = a.pro == RHT_PRO_NORM ? a.scale*rht_norm_scale(a, rht_row_sumsq(a, row, lane)) : a.scale;

        float acc[GT];
        if (active) {
            const float * xs = a.tmp + row*a.n + s*WARP_SIZE + lane;
            rht_mix_unrolled<MIX, GT, KS>(t, p, [&](const int c) { return xs[c*a.Pm]; }, acc, std::make_integer_sequence<int, (MIX + GT - 1)/GT>{});
        }
#pragma unroll
        for (int g = 0; g < GT; ++g) {
            part_acc[threadIdx.y][g][lane] = acc[g];
        }
        __syncthreads();
        // warp p of the job finishes output chunk p of the tile, adding the parts in order
        if (active && t*GT + p < a.M) {
            const int w0 = threadIdx.y - p;
            float v[1] = { part_acc[w0][p][lane] };
#pragma unroll
            for (int q = 1; q < KS; ++q) {
                v[0] += part_acc[w0 + q][p][lane];
            }
            rht_store(a, scale, row, s*WARP_SIZE, lane, t*GT + p, 1, v);
        }
    } else {
        if (w >= jobs*a.nrows) {
            return;
        }

        // consecutive warps take the same tile, for every slab and row, so they share its instructions
        const int        t    = w / (slabs*a.nrows);
        const int        row  = (w / slabs) % a.nrows;
        const int64_t    s    = w % slabs;
        const float    * x    = a.tmp + row*a.n;
        const uint32_t * bits = MIX == 0 ? rht_bits_ptr(a.K) : nullptr;

        ggml_cuda_pdl_sync();

        if (a.out != GGML_CUDA_SRC1_F32 && t == 0 && s == 0) {
            rht_store_pad(a.dst, a.out, a.ne1, a.n, a.npad, row, lane);
        }
        const float scale = a.pro == RHT_PRO_NORM ? a.scale*rht_norm_scale(a, rht_row_sumsq(a, row, lane)) : a.scale;

        if constexpr (MIX == 0) {
            if (narrow) {
                const float v[1] = { rht_mix_narrow(a, s, lane, bits, [&](const int64_t i) { return x[i]; }) };
                rht_store(a, scale, row, s*WARP_SIZE, lane, 0, 1, v);
                return;
            }
        }

        const float * xs = x + s*WARP_SIZE + lane;
        rht_mix_out<MIX, GT>(a, t, bits, [&](const int c) { return xs[c*a.Pm]; },
            [&](const int c0, const int count, const auto & v) { rht_store(a, scale, row, s*WARP_SIZE, lane, c0, count, v); });
    }
}

// Kernel B (many rows): one CTA per row, one pass. The warps keep the transformed chunks in registers and
// hand them to the mix through shared memory in groups of S slabs, [S][M][32]: slab = 32 consecutive j.
// F32 output only: with the row held in registers, the Q8_1 epilogues push the kernel past its 64 registers.
template <int MIX, int G>
__launch_bounds__(RHT_B_WARPS_MAX*WARP_SIZE, 1)
static __global__ void rht_rows(const rht_args a) {
    extern __shared__ float rht_smem[];

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int W    = blockDim.y;

    const int E = a.Pc >= WARP_SIZE ? a.Pc/WARP_SIZE : 1;
    const int U = (a.n + E*WARP_SIZE - 1)/(E*WARP_SIZE);
    const int Q = (U + W - 1)/W;

    constexpr int EMAX = MIX > 1 ? rht_pow2_floor(RHT_B_MAX/(MIX*WARP_SIZE)) : RHT_B_REGS;

    ggml_cuda_pdl_lc();

    // unit u = w + q*W sits in v[q*E ... q*E + E-1]
    float    v[RHT_B_REGS];
    uint32_t sg = 0;
    rht_switch_e<EMAX>(E, [&](auto ec) {
        constexpr int E_ = decltype(ec)::value;
#pragma unroll
        for (int q = 0; q < RHT_B_REGS/E_; ++q) {
            const int u = w + q*W;
            if (q < Q && u < U) {
                sg |= rht_sign_mask<E_>(a.seed, a.n, (int64_t) u*E_*WARP_SIZE, lane) << (q*E_);
            }
        }
    });

    ggml_cuda_pdl_sync();

    const uint32_t * bits = MIX == 0 ? rht_bits_ptr(a.K) : nullptr;

    const int row = blockIdx.x;

    const float * x  = rht_src_row(a, row);
    const float * x2 = a.pro == RHT_PRO_NONE ? nullptr : rht_src2_row(a, row);
    float ss = 0.0f;
    rht_switch_e<EMAX>(E, [&](auto ec) {
        constexpr int E_ = decltype(ec)::value;
#pragma unroll
        for (int q = 0; q < RHT_B_REGS/E_; ++q) {
            const int u = w + q*W;
            if (q < Q && u < U) {
                if (a.pro == RHT_PRO_NONE) {
                    rht_load<E_>(v + q*E_, x, (int64_t) u*E_*WARP_SIZE, lane, a.n);
                } else {
                    rht_load_pro<E_>(v + q*E_, a, x, x2, (int64_t) u*E_*WARP_SIZE, lane, ss);
                }
            }
        }
#pragma unroll
        for (int q = 0; q < RHT_B_REGS/E_; ++q) {
            const int u = w + q*W;
            if (q < Q && u < U) {
#pragma unroll
                for (int e = 0; e < E_; ++e) {
                    v[q*E_ + e] = rht_flip(v[q*E_ + e], sg, q*E_ + e);
                }
                rht_fwht<E_>(v + q*E_, lane, a.Pc);
            }
        }
    });

    // the sums of squares pass through the start of the shared memory, before any stage uses it
    float scale = a.scale;
    if (a.pro == RHT_PRO_NORM) {
        ss = warp_reduce_sum(ss);
        if (lane == 0) {
            rht_smem[w] = ss;
        }
        __syncthreads();
        float sumsq = 0.0f;
        for (int i = 0; i < W; ++i) {
            sumsq += rht_smem[i];
        }
        scale *= rht_norm_scale(a, sumsq);
        __syncthreads();
    }

    if constexpr (MIX == 0) {
        if (a.Pm < WARP_SIZE) {
#pragma unroll
            for (int q = 0; q < RHT_B_REGS; ++q) {
                const int64_t i = (int64_t) (w + q*W)*WARP_SIZE + lane;
                if (q < Q && i < a.n) {
                    rht_smem[i] = v[q];
                }
            }
            __syncthreads();
            for (int b = w; b < U; b += W) {
                rht_store_f32(a, scale, row, (int64_t) b*WARP_SIZE, lane, rht_mix_narrow(a, b, lane, bits, [&](const int64_t i) { return rht_smem[i]; }));
            }
            return;
        }
    }

    const int     S  = a.S;
    constexpr int GT = rht_tile_rows(MIX, G);
    const int     NT = MIX == 1 ? 1 : (a.M + GT - 1)/GT;

    for (int g0 = 0; g0 < E; g0 += S) {
        rht_switch_e<EMAX>(E, [&](auto ec) {
            constexpr int E_ = decltype(ec)::value;
#pragma unroll
            for (int q = 0; q < RHT_B_REGS/E_; ++q) {
                const int u = w + q*W;
                if (q < Q && u < U) {
#pragma unroll
                    for (int e = 0; e < E_; ++e) {
                        if (e >= g0 && e < g0 + S) {
                            rht_smem[((e - g0)*a.M + u)*WARP_SIZE + lane] = v[q*E_ + e];
                        }
                    }
                }
            }
        });
        __syncthreads();

        for (int job = w; job < S*NT; job += W) {
            const int     s  = job % S;
            const int     t  = job / S;
            const float * xs = rht_smem + s*a.M*WARP_SIZE + lane;
            const int64_t e0 = (int64_t) (g0 + s)*WARP_SIZE;
            rht_mix_out<MIX, GT>(a, t, bits, [&](const int c) { return xs[c*WARP_SIZE]; },
                [&](const int c0, const int count, const auto & v) {
#pragma unroll
                    for (int g = 0; g < (int) (sizeof(v)/sizeof(v[0])); ++g) {
                        if (g < count) {
                            rht_store_f32(a, scale, row, (c0 + g)*a.Pm + e0, lane, v[g]);
                        }
                    }
                });
        }
        __syncthreads();
    }
}

// host side

struct rht_shape {
    int     K;
    int     M;
    int     mix;       // MIX template argument
    bool    multipass; // strided passes between kernel A's passes
    int64_t n;
    int64_t P;
    int64_t Pc;
    int64_t Pm;
};

struct rht_cfg {
    int     kernel     = 0;             // 0: by rows, 1: A, 2: B (A where B can't run)
    int     rows_a_max = 8;
    int     a_warps    = 2;
    int     b_warps    = 0;             // target, raised as needed to fit the row in registers; 0: by shape
    int64_t k1_chunk   = RHT_CHUNK_MAX; // chunk width of K = 1 rows
    int64_t k1_a_max   = 16384;         // K = 1 rows up to this width stream faster through kernel A
    bool    generic    = false;         // generic mix for the unrolled orders too
};

static bool rht_is_unrolled(const int K) {
    switch (K) {
#define RHT_CASE(K) case K: return true;
        RHT_UNROLLED_ORDERS(RHT_CASE)
#undef RHT_CASE
        default: return false;
    }
}

// K = 1 rows: H_P = H_M (x) H_Pc, H_M done by the mix. Rows with Pc = RHT_CHUNK_MAX and a mix over more
// than RHT_MIX_MAX chunks (K = 1), or K > 1 with P > RHT_CHUNK_MAX, get strided passes in between.
static bool rht_make_shape(const int64_t n, const rht_cfg & cfg, rht_shape & sh) {
    int     K;
    int64_t P;
    if (!ggml_rht_plan(n, &K, &P)) {
        return false;
    }
    sh.K         = K;
    sh.n         = n;
    sh.P         = P;
    sh.multipass = false;
    if (K == 1) {
        sh.Pc = std::min<int64_t>(P, cfg.k1_chunk);
        if (P/sh.Pc > RHT_MIX_MAX) {
            sh.Pc = std::min<int64_t>(P/RHT_MIX_MAX, RHT_CHUNK_MAX);
        }
        sh.M         = (int) std::min<int64_t>(P/sh.Pc, RHT_MIX_MAX);
        sh.multipass = P/sh.Pc > RHT_MIX_MAX;
        sh.mix       = 1;
    } else {
        sh.Pc  = std::min<int64_t>(P, RHT_CHUNK_MAX);
        sh.M   = K;
        sh.multipass = P > RHT_CHUNK_MAX;
        sh.mix = rht_is_unrolled(K) && P >= WARP_SIZE && !cfg.generic ? K : 0;
    }
    sh.Pm = n/sh.M;
    return true;
}

static bool rht_use_b(const rht_shape & sh, const int64_t nrows, const rht_cfg & cfg, const int out = GGML_CUDA_SRC1_F32) {
    const bool b_ok = !sh.multipass && sh.n <= RHT_B_MAX && !(sh.K == 1 && sh.n < WARP_SIZE) &&
        out == GGML_CUDA_SRC1_F32;
    switch (cfg.kernel) {
        case 1:  return false;
        case 2:  return b_ok;
        default: return b_ok && nrows > cfg.rows_a_max && !(sh.K == 1 && sh.n <= cfg.k1_a_max);
    }
}

static int64_t rht_units(const rht_shape & sh) {
    const int64_t E = std::max<int64_t>(sh.Pc/WARP_SIZE, 1);
    return (sh.n + E*WARP_SIZE - 1)/(E*WARP_SIZE);
}

static size_t rht_tmp_floats(const rht_shape & sh, const int64_t nrows, const rht_cfg & cfg, const int out = GGML_CUDA_SRC1_F32,
        const int pro = RHT_PRO_NONE) {
    if (rht_use_b(sh, nrows, cfg, out) || (sh.M == 1 && !sh.multipass)) {
        return 0;
    }
    return (size_t) ((sh.n + (pro == RHT_PRO_NORM ? rht_units(sh) : 0))*nrows);
}

static rht_args rht_make_args(const rht_shape & sh, const void * src, const int64_t * ne, const size_t * nb,
        void * dst, const int out, const uint64_t seed, float * tmp) {
    rht_args a;
    a.src   = (const char *) src;
    a.tmp   = tmp;
    a.dst   = dst;
    a.out   = out;
    a.K     = sh.K;
    a.M     = sh.M;
    a.nrows = (int) (ne[1]*ne[2]*ne[3]);
    a.S     = 1;
    a.n     = sh.n;
    a.Pc    = sh.Pc;
    a.Pm    = sh.Pm;
    a.npad  = GGML_PAD(sh.n, MATRIX_ROW_PADDING);
    a.nb1   = nb[1];
    a.nb2   = nb[2];
    a.nb3   = nb[3];
    a.ne1   = init_fastdiv_values(ne[1]);
    a.ne2   = init_fastdiv_values(ne[2]);
    a.seed  = seed;
    a.scale = (float) (1.0 / sqrt((double) sh.n));
    a.pro   = RHT_PRO_NONE;
    a.units = (int) rht_units(sh);
    a.src2  = nullptr;
    a.nb21  = 0;
    a.nb22  = 0;
    a.nb23  = 0;
    a.eps   = 0.0f;
    return a;
}

template <int G>
static void (*rht_mix_split_kernel(const int mix, const bool split))(const rht_args) {
    switch (mix) {
        case 0: return rht_mix_split<0, G, 1>;
        case 1: return rht_mix_split<1, G, 1>;
#define RHT_CASE(K) case K: \
            if constexpr (rht_ksplit(K) > 1 && rht_ksplit_rows_max(K) < INT_MAX) { \
                return split ? rht_mix_split<K, G, rht_ksplit(K)> : rht_mix_split<K, G, 1>; \
            } else { \
                return rht_mix_split<K, G, rht_ksplit(K)>; \
            }
        RHT_UNROLLED_ORDERS(RHT_CASE)
#undef RHT_CASE
    }
    GGML_ABORT("fatal error");
}

template <int G>
static void (*rht_rows_kernel(const int mix))(const rht_args) {
    switch (mix) {
        case 0: return rht_rows<0, G>;
        case 1: return rht_rows<1, G>;
#define RHT_CASE(K) case K: return rht_rows<K, G>;
        RHT_UNROLLED_ORDERS(RHT_CASE)
#undef RHT_CASE
    }
    GGML_ABORT("fatal error");
}

static void (*rht_chunks_kernel(const int64_t E))(const rht_args) {
    switch (E) {
        case  1: return rht_chunks< 1>;
        case  2: return rht_chunks< 2>;
        case  4: return rht_chunks< 4>;
        case  8: return rht_chunks< 8>;
        case 16: return rht_chunks<16>;
        case 32: return rht_chunks<32>;
    }
    GGML_ABORT("fatal error");
}

static void (*rht_strided_kernel(const int64_t L))(const rht_args, const int64_t) {
    switch (L) {
        case  2: return rht_strided< 2>;
        case  4: return rht_strided< 4>;
        case  8: return rht_strided< 8>;
        case 16: return rht_strided<16>;
        case 32: return rht_strided<32>;
    }
    GGML_ABORT("fatal error");
}

// a.tmp must hold rht_tmp_floats() floats
template <int GA = RHT_GA, int GB = RHT_GB>
static void rht_launch(rht_args a, const rht_shape & sh, const rht_cfg & cfg, cudaStream_t stream) {
    if (a.nrows == 0) {
        return;
    }

    const int64_t E = std::max<int64_t>(sh.Pc/WARP_SIZE, 1);

    if (rht_use_b(sh, a.nrows, cfg, a.out)) {
        // measured on an RTX 5090: more warps for the mix of wide rows, fewer for narrow chunks of short rows
        const int b_warps = cfg.b_warps > 0 ? cfg.b_warps : sh.n >= 16384 ? 32 : sh.Pc <= 128 && sh.n <= 8192 ? 8 : 16;
        const int U  = (int) ((sh.n + E*WARP_SIZE - 1)/(E*WARP_SIZE));
        const int Wt = std::max(1, std::min(b_warps, RHT_B_WARPS_MAX));
        const int Q  = std::min<int>((U + Wt - 1)/Wt, RHT_B_REGS/E);
        const int W  = (U + Q - 1)/Q;
        GGML_ASSERT(W <= RHT_B_WARPS_MAX);

        size_t smem;
        if (sh.mix == 0 && sh.Pm < WARP_SIZE) {
            smem = sh.n*sizeof(float);
        } else {
            int S = 1;
            while (2*S <= E && 2*S*sh.M*WARP_SIZE*sizeof(float) <= RHT_B_SMEM) {
                S *= 2;
            }
            a.S  = S;
            smem = (size_t) S*sh.M*WARP_SIZE*sizeof(float);
        }
        GGML_ASSERT(smem <= RHT_B_SMEM);

        const ggml_cuda_kernel_launch_params p(dim3(a.nrows), dim3(WARP_SIZE, W), smem, stream);
        ggml_cuda_kernel_launch(rht_rows_kernel<GB>(sh.mix), p, a);
        return;
    }

    const dim3    block(WARP_SIZE, cfg.a_warps);
    const int64_t units = (sh.n + E*WARP_SIZE - 1)/(E*WARP_SIZE);
    auto grid = [&](const int64_t warps) {
        return dim3((unsigned) ((warps + cfg.a_warps - 1)/cfg.a_warps));
    };

    const size_t smem1 = sh.M == 1 && !sh.multipass && a.out != GGML_CUDA_SRC1_F32 ?
        (size_t) cfg.a_warps*WARP_SIZE*(WARP_SIZE + 1)*sizeof(float) : 0;
    ggml_cuda_kernel_launch(rht_chunks_kernel(E), ggml_cuda_kernel_launch_params(grid(units*a.nrows), block, smem1, stream), a);
    if (sh.M == 1 && !sh.multipass) {
        return;
    }

    for (int64_t s = sh.Pc; s < sh.Pm; ) {
        const int64_t L = std::min<int64_t>(sh.Pm/s, RHT_MIX_MAX);
        const int64_t threads = a.nrows*(sh.n/L);
        const ggml_cuda_kernel_launch_params p(dim3((unsigned) ((threads + 255)/256)), dim3(256), 0, stream);
        ggml_cuda_kernel_launch(rht_strided_kernel(L), p, a, s);
        s *= L;
    }

    const bool    narrow = sh.mix == 0 && sh.Pm < WARP_SIZE;
    const int64_t slabs  = narrow ? (sh.n + WARP_SIZE - 1)/WARP_SIZE : sh.Pm/WARP_SIZE;
    const int64_t GT     = rht_tile_rows(sh.mix, GA);
    const int64_t NT     = sh.mix == 1 || narrow ? 1 : (sh.M + GT - 1)/GT;
    const bool    split  = sh.mix > 1 && a.nrows <= rht_ksplit_rows_max(sh.mix);
    const int     KS     = split ? rht_ksplit(sh.mix) : 1;
    const int     wpb    = std::min(cfg.a_warps*KS, RHT_A_WARPS_MAX);
    const int64_t warps  = slabs*NT*a.nrows*KS;
    const ggml_cuda_kernel_launch_params p2(dim3((unsigned) ((warps + wpb - 1)/wpb)), dim3(WARP_SIZE, wpb), 0, stream);
    ggml_cuda_kernel_launch(rht_mix_split_kernel<GA>(sh.mix, split), p2, a);
}
