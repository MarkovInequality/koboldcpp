#pragma once

// MMA flash attention reading quantized K/V. Each 8-value piece of a row is dequantized with the conversion path's
// arithmetic (dequantize.cuh, then a round-to-nearest cast to half) straight into the swizzled F16 tile, so the
// tile holds the bytes the converted F16 copy would and the rest of the kernel is the F16 one.

#include "common.cuh"
#include "dequantize.cuh"
#include "mma.cuh"

template <ggml_type type> struct fattn_mma_q_type;

template <> struct fattn_mma_q_type<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0, qr = QR4_0, block_size = sizeof(block_q4_0), align = alignof(block_q4_0);
    static constexpr dequantize_kernel_t dequantize = dequantize_q4_0;
};
template <> struct fattn_mma_q_type<GGML_TYPE_Q5_1> {
    static constexpr int qk = QK5_1, qr = QR5_1, block_size = sizeof(block_q5_1), align = alignof(block_q5_1);
    static constexpr dequantize_kernel_t dequantize = dequantize_q5_1;
};
template <> struct fattn_mma_q_type<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0, qr = QR8_0, block_size = sizeof(block_q8_0), align = alignof(block_q8_0);
    static constexpr dequantize_kernel_t dequantize = dequantize_q8_0;
};

// bytes from a row's start to its value n, n a multiple of the block size
template <ggml_type type>
static __device__ __forceinline__ int fattn_mma_q_row_offset(const int n) {
    return n/fattn_mma_q_type<type>::qk * fattn_mma_q_type<type>::block_size;
}

// values e0..e0+7 of row as 4 half2, e0 a multiple of 8
template <ggml_type type>
static __device__ __forceinline__ int4 fattn_mma_q_dequantize8(const char * __restrict__ row, const int e0) {
    using tq = fattn_mma_q_type<type>;
    const int ib = e0 / tq::qk;
    const int e  = e0 % tq::qk;
    half h[8];
    if constexpr (tq::qr == 1) {
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            float2 v;
            tq::dequantize(row, ib, e + j, v);
            h[j + 0] = __float2half(v.x);
            h[j + 1] = __float2half(v.y);
        }
    } else {
        // one call gives value iqs and value iqs + qk/2 of the block
        const bool high = e >= tq::qk/2;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float2 v;
            tq::dequantize(row, ib, (high ? e - tq::qk/2 : e) + j, v);
            h[j] = __float2half(high ? v.y : v.x);
        }
    }
    int4 res;
    memcpy(&res, h, sizeof(res));
    return res;
}

// the quantized counterpart of flash_attn_ext_f16_load_tile (synchronous): D2 half2 columns of nbatch_fa rows
// starting at row k_VKQ_0, KV pointing at the first value of the columns in row 0
template<ggml_type type, int stride_tile, bool swz, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_q_load_tile(
        const char * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV_bytes,
        const int k_VKQ_0, const int i_sup) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int chunks_per_row = D2/4; // 16 bytes, 8 values each
    const int nchunks        = nbatch_fa*chunks_per_row;
    for (int c = threadIdx.y*warp_size + threadIdx.x; c < nchunks; c += nwarps*warp_size) {
        const int i = c / chunks_per_row;
        const int k = c - i*chunks_per_row;
        int4 v = make_int4(0, 0, 0, 0);
        if (!oob_check || i < i_sup) {
            v = fattn_mma_q_dequantize8<type>(KV + int64_t(k_VKQ_0 + i)*stride_KV_bytes, 8*k);
        }
        *(int4 *) ((char *) tile_KV + ggml_cuda_mma::swizzle_bytes<swz, half2>(i, 4*k, stride_tile)) = v;
    }
}
