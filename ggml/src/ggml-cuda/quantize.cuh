#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

// How the data of an RHT node is stored during a graph evaluation: the node writes the Q8_1 its consumers
// would otherwise quantize it to (ggml_cuda_rht_plan_q8_1 in ggml-cuda.cu).
enum ggml_cuda_src1_fmt {
    GGML_CUDA_SRC1_F32,
    GGML_CUDA_SRC1_Q8_1,        // quantize_row_q8_1_cuda
    GGML_CUDA_SRC1_Q8_1_MMQ_D4, // quantize_mmq_q8_1_cuda
    GGML_CUDA_SRC1_Q8_1_MMQ_DS4,
};

ggml_cuda_src1_fmt ggml_cuda_get_src1_fmt(const ggml_tensor * t);

// Per-block packing of the Q8_1 layouts, shared by the quantize kernels and the RHT epilogue.

// quantize_row_q8_1_cuda: rows of ne0 (padded) values, one block_q8_1 per 32
static __device__ __forceinline__ block_q8_1 * q8_1_block(void * vy, const int64_t row, const int64_t i0, const int64_t ne0) {
    return (block_q8_1 *) vy + (row*ne0 + i0)/QK8_1;
}

// one value per lane of a 32-lane group, iqs = lane
static __device__ __forceinline__ void q8_1_quantize_block(block_q8_1 * y, const int iqs, const float xi) {
    const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    const float sum  = warp_reduce_sum<QK8_1>(xi);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y->qs[iqs] = q;
    if (iqs == 0) {
        y->ds = make_half2(d, sum);
    }
}

// quantize_mmq_q8_1_cuda: per channel, the 128-value blocks of its ne1 columns interleaved, block (k, i1) at k*ne1 + i1
static __device__ __forceinline__ block_q8_1_mmq * q8_1_mmq_block(
        void * vy, const int64_t channel, const int64_t i1, const int64_t i0, const int64_t ne1, const int64_t ne0) {
    return (block_q8_1_mmq *) vy + channel*(ne1*(ne0/QK8_1_MMQ)) + (i0/QK8_1_MMQ)*ne1 + i1;
}

struct q8_1_mmq_vals {
    char4 q;
    float d;
    float sum;
};

// 4 consecutive values per thread; the threads of a scale group are consecutive lanes
template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ q8_1_mmq_vals q8_1_mmq_quantize4(const float4 xi) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    q8_1_mmq_vals v;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        v.sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            v.sum += __shfl_xor_sync(0xFFFFFFFF, v.sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    v.q.x = roundf(xi.x*d_inv);
    v.q.y = roundf(xi.y*d_inv);
    v.q.z = roundf(xi.z*d_inv);
    v.q.w = roundf(xi.w*d_inv);
    v.d   = 1.0f / d_inv;
    return v;
}

// iqs: index of the first of the 4 values in the block
template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void q8_1_mmq_store4(block_q8_1_mmq * y, const int iqs, const q8_1_mmq_vals & v) {
    ((char4 *) y->qs)[iqs/4] = v.q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y->d2s6[2 + iqs/16] = v.sum;
            if (iqs % 64 == 0) {
                y->d2s6[iqs/64] = v.d;
            }
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y->ds4[iqs/32] = make_half2(v.d, v.sum);
        } else {
            y->d4[iqs/32] = v.d;
        }
    }
}

typedef void (*quantize_cuda_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_fp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             float *         scale,
                             ggml_type       type_src0,
                             bool            use_aligned_float8,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);

// quantize each token once and scatter the block to its compact rows (via the inverse map)
void quantize_scatter_mmq_fp4_cuda(const float *   x,
                                   const int32_t * ids_src1_inv,
                                   void *          vy,
                                   float *         scale,
                                   ggml_type       type_src0,
                                   bool            use_aligned_float8,
                                   int64_t         ne00,
                                   int64_t         stride_token,
                                   int64_t         ne0,
                                   int64_t         n_tokens,
                                   int64_t         nrows_dst,
                                   int             n_expert_used,
                                   cudaStream_t    stream);

void quantize_scatter_mmq_q8_1_cuda(const float *   x,
                                    const int32_t * ids_src1_inv,
                                    void *          vy,
                                    ggml_type       type_src0,
                                    int64_t         ne00,
                                    int64_t         stride_token,
                                    int64_t         ne0,
                                    int64_t         n_tokens,
                                    int64_t         nrows_dst,
                                    int             n_expert_used,
                                    cudaStream_t    stream);
