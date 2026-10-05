#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
#include "unary.cuh"

#include <atomic>

// each warp owns NC columns of the state: their recurrences are independent, so a warp overlaps their latencies and
// loads q and k once for all of them; a column's arithmetic is the same for any NC
template <int S_v, bool KDA, bool keep_rs_t, int NC>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * state_rows,
                                     int64_t       state_rows_stride,
                                     const ggml_cuda_gdn_gating gating) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns NC columns, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col0     = (blockIdx.z * blockDim.y + threadIdx.y) * NC;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // or, reading the cache in place, row state_rows[sequence] of curr_state (each warp reads its column before it
    // writes that column of any slot, so the row may be one the kernel writes)
    const int64_t state_in_offset      = (state_rows ? state_rows[sequence] * state_rows_stride : sequence * H * S_v * S_v) +
                                         h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col0 * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[NC][rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < NC; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i   = r * warp_size + lane;
            s_shard[c][r] = curr_state[c * S_v + i];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        // as the skipped nodes compute them (binbcast add and mul, the unary ops)
        const float beta_val = gating.alpha ? ggml_cuda_op_sigmoid_single(gating.beta_in[gb_offset]) : *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(gating.alpha ?
                ggml_cuda_op_softplus_single(gating.alpha[gb_offset] + gating.dt[h_idx]) * gating.a[h_idx] : *g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_col[NC];
#pragma unroll
            for (int c = 0; c < NC; c++) {
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[c][r] * k_reg[r];
                }
                kv_col[c] = warp_reduce_sum<warp_size>(kv_shard);
            }

#pragma unroll
            for (int c = 0; c < NC; c++) {
                // delta[col] = (v[col] - g * kv[col]) * beta
                float delta_col = (v_t[col0 + c] - g_val * kv_col[c]) * beta_val;

                // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r]  = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[c][r] * q_reg[r];
                }

                float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                if (lane == 0) {
                    attn_data[col0 + c] = attn_col * scale;
                }
            }
        } else {
#pragma unroll
            for (int c = 0; c < NC; c++) {
                // kv[col] = sum_i g[i] * S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    kv_shard += expf(g_t[i]) * s_shard[c][r] * k_reg[r];
                }

                float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - kv[col]) * beta
                float delta_col = (v_t[col0 + c] - kv_col) * beta_val;

                // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    s_shard[c][r]  = expf(g_t[i]) * s_shard[c][r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[c][r] * q_reg[r];
                }

                float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                if (lane == 0) {
                    attn_data[col0 + c] = attn_col * scale;
                }
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < NC; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        curr_state[(col0 + c) * S_v + i] = s_shard[c][r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i                 = r * warp_size + lane;
                state[(col0 + c) * S_v + i] = s_shard[c][r];
            }
        }
    }
}

template <int S_v, bool KDA, bool keep_rs_t, typename... Args>
static void launch_gated_delta_net_cols(int nc, const ggml_cuda_kernel_launch_params & launch_params, Args... args) {
    switch (nc) {
        case 1: ggml_cuda_kernel_launch(gated_delta_net_cuda<S_v, KDA, keep_rs_t, 1>, launch_params, args...); break;
        case 2: ggml_cuda_kernel_launch(gated_delta_net_cuda<S_v, KDA, keep_rs_t, 2>, launch_params, args...); break;
        case 4: ggml_cuda_kernel_launch(gated_delta_net_cuda<S_v, KDA, keep_rs_t, 4>, launch_params, args...); break;
        default: GGML_ABORT("fatal error");
    }
}

// Columns per warp: two for prompt batches once the one-column grid fills 3/4 of the resident block slots. A warp then
// overlaps two chains and the SM issues half the q/k loads; below that, halving the blocks leaves SMs short of warps
// (1024 tokens on an RTX 5090: 4.5 blocks per SM +34 %, 6.0 +6 %, 7.5 -7 %, 12 -30 %), and a few tokens are bound by
// the state's loads, which want every warp. GGML_CUDA_GDN_COLS=1|2|4 overrides it (tests).
template <int S_v, bool KDA, bool keep_rs_t>
static int gated_delta_net_cols(int64_t blocks_one_col, int block_size, int64_t n_tokens) {
    const char * env = getenv("GGML_CUDA_GDN_COLS");
    if (env) {
        const int nc = atoi(env);
        if (nc == 1 || nc == 2 || nc == 4) {
            return S_v >= 64 ? nc : 1;
        }
    }
    if (S_v < 64 || n_tokens <= 16) {
        return 1;
    }
    const int device = ggml_cuda_get_device();
    static int max_blocks_per_sm[GGML_CUDA_MAX_DEVICES] = {};
    if (max_blocks_per_sm[device] == 0) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm[device],
            gated_delta_net_cuda<S_v, KDA, keep_rs_t, 1>, block_size, 0));
    }
    const int64_t slots = (int64_t) max_blocks_per_sm[device] * ggml_cuda_info().devices[device].nsm;
    return 4*blocks_one_col >= 3*slots ? 2 : 1;
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * state_rows, int64_t state_rows_stride,
        const ggml_cuda_gdn_gating & gating, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);
    const int64_t blocks_one_col = H * n_seqs * ((S_v + num_warps - 1) / num_warps);
    const int block_size = block_dims.x * block_dims.y;
    int nc = 1;
    switch (S_v) {
        case 64:  nc = gated_delta_net_cols<64,  KDA, keep_rs_t>(blocks_one_col, block_size, n_tokens); break;
        case 128: nc = gated_delta_net_cols<128, KDA, keep_rs_t>(blocks_one_col, block_size, n_tokens); break;
        default: break;
    }
    dim3      grid_dims(H, n_seqs, (S_v + num_warps*nc - 1) / (num_warps*nc));

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
#define GDN_ARGS q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, \
                 sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows, state_rows_stride, gating
    switch (S_v) {
        case 16:  launch_gated_delta_net_cols<16,  KDA, keep_rs_t>(nc, launch_params, GDN_ARGS); break;
        case 32:  launch_gated_delta_net_cols<32,  KDA, keep_rs_t>(nc, launch_params, GDN_ARGS); break;
        case 64:  launch_gated_delta_net_cols<64,  KDA, keep_rs_t>(nc, launch_params, GDN_ARGS); break;
        case 128: launch_gated_delta_net_cols<128, KDA, keep_rs_t>(nc, launch_params, GDN_ARGS); break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
#undef GDN_ARGS
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache,
        const ggml_tensor * state_rows, const ggml_cuda_gdn_gating * gating) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    // with state_rows (the GET_ROWS that gathers the state), its source rows and indices instead of its output
    const float * s_d   = (const float *) (state_rows ? state_rows->src[0]->data : src_state->data);
    const ggml_cuda_gdn_gating gating_d = gating ? *gating : ggml_cuda_gdn_gating{};
    const int32_t * rows_d    = state_rows ? (const int32_t *) state_rows->src[1]->data : nullptr;
    const int64_t rows_stride = state_rows ? state_rows->src[0]->nb[1] / sizeof(float) : 0;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(!gating || !kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(state_rows || ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rows_d, rows_stride, gating_d, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rows_d, rows_stride, gating_d, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rows_d, rows_stride, gating_d, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rows_d, rows_stride, gating_d, stream);
        }
    }
}

static std::atomic<int64_t> gdn_state_in_place_count{0};
static std::atomic<int64_t> gdn_gating_count{0};

int64_t ggml_cuda_gdn_state_in_place_count() {
    return gdn_state_in_place_count.load();
}

int64_t ggml_cuda_gdn_gating_count() {
    return gdn_gating_count.load();
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * state_rows,
                                  const ggml_cuda_gdn_gating * gating) {
    gdn_state_in_place_count += state_rows != nullptr;
    gdn_gating_count += gating != nullptr;
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr, state_rows, gating);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache,
        const ggml_tensor * state_rows, const ggml_cuda_gdn_gating * gating) {
    gdn_state_in_place_count += state_rows != nullptr;
    gdn_gating_count += gating != nullptr;
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache, state_rows, gating);
}
