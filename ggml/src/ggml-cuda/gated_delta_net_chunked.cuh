// Chunked gated delta rule for prompt batches (one gate per head, the state's columns S_v = D).
//
// Within a chunk of C tokens starting from state S (D x D, S[i][col] with i over k and col over v), with G the
// chunk's cumulative log decay, gamma_j = exp(G_j) and w_j = exp(G_last - G_j), the recurrence
//     S_t = g_t S_{t-1} + k_t delta_t^T,  delta_t = beta_t (v_t - (g_t S_{t-1})^T k_t),  o_t = S_t^T q_t
// unrolls to (WY form, as in the Gated DeltaNet paper)
//     U = T diag(beta) (V - diag(gamma) K S),  T = (I + A)^-1,  A[j][i] = beta_j exp(G_j - G_i) k_j.k_i (i < j)
//     O = diag(gamma) Q S + P U,                P[j][i] = exp(G_j - G_i) q_j.k_i (i <= j)
//     S' = gamma_last S + K^T diag(w) U
// with U the chunk's deltas. gdn_chunk_prep computes T, P, gamma, w and beta for every chunk in parallel, k.k and q.k
// once per q/k head; gdn_chunk_scan carries a block of W state columns through the chunks (the columns are
// independent). All in FP32: against the recurrent kernel the 27B's layers differ by an NMSE of ~5e-12.

#pragma once

#include "common.cuh"
#include "unary.cuh"

static constexpr int GDN_CHUNK_C = 32;  // tokens per chunk
static constexpr int GDN_CHUNK_W = 16;  // state columns per scan block

// per (sequence, head, chunk): T [C][C], P [C][C], gamma [C], w [C], beta [C], gamma_last
static constexpr __host__ __device__ int gdn_chunk_stride(int C) {
    return (2*C*C + 3*C + 1 + 3) & ~3;
}

// one block per chunk and q/k head: k_i.k_j and q_i.k_j once, then a warp per v head that shares them (lane = column)
template <int C, int D>
__global__ void __launch_bounds__(256) gdn_chunk_prep(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ g,
        const float * __restrict__ beta, const ggml_cuda_gdn_gating gating, float * __restrict__ prep,
        const int64_t H, const int64_t n_tok, const int n_chunks, const int64_t neqk1, const int64_t rq3,
        const int64_t sq1, const int64_t sq2, const int64_t sq3, const int64_t sb1, const int64_t sb2, const int64_t sb3) {
    static_assert(C == 32 && D % 4 == 0, "one warp scans a chunk");
    constexpr int DP = D + 4; // padded rows: rows 8 apart in a load phase start on distinct banks
    const int     chunk = blockIdx.x;
    const int64_t hk    = blockIdx.y;
    const int64_t seq   = blockIdx.z;
    const int64_t t0    = (int64_t) chunk*C;
    const int     n     = (int) min((int64_t) C, n_tok - t0);
    const int64_t iq3   = seq / rq3;
    const int     tid   = threadIdx.x;
    const int     lane  = tid % 32;
    const int     warp  = tid / 32;

    __shared__ __align__(16) float ks[C][DP];
    __shared__ __align__(16) float qs[C][DP];
    __shared__ float KK[C][C + 1];
    __shared__ float QK[C][C + 1];

    for (int idx = tid; idx < C*D/4; idx += blockDim.x) {
        const int rr = idx / (D/4);
        const int d4 = idx % (D/4);
        float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 qv = kv;
        if (rr < n) {
            const int64_t off = iq3*sq3 + (t0 + rr)*sq2 + hk*sq1 + 4*d4;
            kv = *(const float4 *) (k + off);
            qv = *(const float4 *) (q + off);
        }
        *(float4 *) &ks[rr][4*d4] = kv;
        *(float4 *) &qs[rr][4*d4] = qv;
    }
    __syncthreads();

    // rows i0, i0+1 of K (threads 0..127) or Q (128..255) times rows j, j+8, j+16, j+24 of K
    {
        const int   t  = tid % 128;
        const int   i0 = (t / 8) * 2;
        const int   j  = t % 8;
        const float (*X)[DP] = tid < 128 ? ks : qs;
        float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll 4
        for (int d = 0; d < D; d += 4) {
            const float4 x0 = *(const float4 *) &X[i0][d];
            const float4 x1 = *(const float4 *) &X[i0 + 1][d];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const float4 y = *(const float4 *) &ks[j + 8*m][d];
                acc[0][m] += x0.x*y.x + x0.y*y.y + x0.z*y.z + x0.w*y.w;
                acc[1][m] += x1.x*y.x + x1.y*y.y + x1.z*y.z + x1.w*y.w;
            }
        }
        float (*Y)[C + 1] = tid < 128 ? KK : QK;
#pragma unroll
        for (int a = 0; a < 2; a++) {
#pragma unroll
            for (int m = 0; m < 4; m++) {
                Y[i0 + a][j + 8*m] = acc[a][m];
            }
        }
    }
    __syncthreads();

    const int n_rep = (int) (H / neqk1);
    for (int rep = warp; rep < n_rep; rep += blockDim.x / 32) {
        const int64_t h   = hk + rep*neqk1;
        float *       out = prep + ((seq*H + h)*n_chunks + chunk)*gdn_chunk_stride(C);

        float a = 0.0f; // padded tokens: no decay, no update
        float b = 0.0f;
        if (lane < n) {
            const int64_t gb = seq*sb3 + (t0 + lane)*sb2 + h*sb1;
            // as the skipped nodes compute them (binbcast add and mul, the unary ops)
            b = gating.alpha ? ggml_cuda_op_sigmoid_single(gating.beta_in[gb]) : beta[gb];
            a = gating.alpha ? ggml_cuda_op_softplus_single(gating.alpha[gb] + gating.dt[h]) * gating.a[h] : g[gb];
        }
#pragma unroll
        for (int off = 1; off < C; off <<= 1) {
            const float y = __shfl_up_sync(0xffffffff, a, off);
            if (lane >= off) {
                a += y;
            }
        }
        const float G = a;

        // column `lane` of A and P; T = (I + A)^-1 by forward substitution, A's row j from lane i by shuffles
        float Acol[C];
#pragma unroll
        for (int j = 0; j < C; j++) {
            const float Gj    = __shfl_sync(0xffffffff, G, j);
            const float bj    = __shfl_sync(0xffffffff, b, j);
            const float decay = lane <= j ? expf(Gj - G) : 0.0f;
            Acol[j] = lane < j ? bj * decay * KK[j][lane] : 0.0f;
            out[C*C + j*C + lane] = decay * QK[j][lane];
        }
        float Tcol[C];
#pragma unroll
        for (int j = 0; j < C; j++) {
            float s = lane == j ? 1.0f : 0.0f;
#pragma unroll
            for (int i = 0; i < j; i++) {
                s -= __shfl_sync(0xffffffff, Acol[j], i) * Tcol[i];
            }
            Tcol[j] = s;
            out[j*C + lane] = s;
        }

        const float G_last = __shfl_sync(0xffffffff, G, n - 1);
        out[2*C*C + lane]       = expf(G);          // gamma
        out[2*C*C + C + lane]   = expf(G_last - G); // w
        out[2*C*C + 2*C + lane] = b;                // beta
        if (lane == 0) {
            out[2*C*C + 3*C] = expf(G_last);        // gamma_last
        }
    }
}

// rows r0, r0+1 times columns c0..c0+3 of A S (A: C x D rows in smem, S: D x W), this lane's quarter of D (the
// d-quads q, q+4, ...), summed over the group's 4 lanes, which are 4 apart: a phase of 8 lanes reads 4 column quads
// of 2 rows 4 apart, 16 banks apart with SW = W + 4
template <int D, int W, int SW>
static __device__ __forceinline__ void gdn_rows_times_state(const float (*A)[D], const float (*S)[SW], int r0, int c0, int q,
        float out[2][4]) {
#pragma unroll
    for (int a = 0; a < 2; a++) {
#pragma unroll
        for (int b = 0; b < 4; b++) {
            out[a][b] = 0.0f;
        }
    }
#pragma unroll 4
    for (int m = q; m < D/4; m += 4) {
        const int    d  = 4*m;
        const float4 a0 = *(const float4 *) &A[r0][d];
        const float4 a1 = *(const float4 *) &A[r0 + 1][d];
        const float  x0[4] = {a0.x, a0.y, a0.z, a0.w};
        const float  x1[4] = {a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float4 sj = *(const float4 *) &S[d + j][c0];
            out[0][0] += x0[j]*sj.x; out[0][1] += x0[j]*sj.y; out[0][2] += x0[j]*sj.z; out[0][3] += x0[j]*sj.w;
            out[1][0] += x1[j]*sj.x; out[1][1] += x1[j]*sj.y; out[1][2] += x1[j]*sj.z; out[1][3] += x1[j]*sj.w;
        }
    }
#pragma unroll
    for (int a = 0; a < 2; a++) {
#pragma unroll
        for (int b = 0; b < 4; b++) {
            out[a][b] += __shfl_xor_sync(0xffffffff, out[a][b], 4);
            out[a][b] += __shfl_xor_sync(0xffffffff, out[a][b], 8);
        }
    }
}

// state_in/state_rows as in gated_delta_net_cuda; state_out: [D, D, H, n_seqs] like slot 0 of the kernel's output
template <int C, int D, int W>
__global__ void __launch_bounds__(256, 3) gdn_chunk_scan(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ prep, const float * state_in, const int32_t * state_rows, const int64_t state_rows_stride,
        float * __restrict__ attn, float * state_out, const int64_t H, const int64_t n_tok, const int64_t n_tok_attn,
        const int n_chunks, const int64_t neqk1, const int64_t rq3,
        const int64_t sq1, const int64_t sq2, const int64_t sq3, const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const float scale) {
    static_assert(C == 32 && W == 16 && D % 16 == 0, "thread layout");
    constexpr int SW = W + 4; // padded state rows
    const int     col0 = blockIdx.x * W;
    const int64_t h    = blockIdx.y;
    const int64_t seq  = blockIdx.z;
    const int64_t iq1  = h % neqk1;
    const int64_t iq3  = seq / rq3;
    const int     tid  = threadIdx.x;

    __shared__ __align__(16) float KQ[C][D];   // the chunk's k, then its q
    __shared__ __align__(16) float S[D][SW];
    __shared__ __align__(16) float Bs[C][W];
    __shared__ __align__(16) float Us[C][W];   // the deltas, then scaled by w for the update
    __shared__ float gam[C];
    __shared__ float wv[C];
    __shared__ float bet[C];
    __shared__ float gam_last;

    // the block reads all of its columns before it writes any, so state_out may alias the input rows
    const float * s_in = state_in + (state_rows ? state_rows[seq]*state_rows_stride : seq*H*D*D) + h*D*D;
    for (int idx = tid; idx < W*D; idx += blockDim.x) {
        const int c = idx / D;
        const int d = idx % D;
        S[d][c] = s_in[(col0 + c)*D + d];
    }

    // C x W products: rows r0, r0+1, columns c0..c0+3, quarter qd of D; (r, c..c+1) for the C x C products;
    // rows d0..d0+3, columns cu..cu+1 of the D x W update
    const int qd  = (tid >> 2) & 3;
    const int r0  = (tid >> 4) * 2;
    const int c0  = (tid & 3) * 4;
    const int r   = tid / 8;
    const int c   = (tid % 8) * 2;
    const int d0  = (tid / 8) * 4;
    const int cu  = (tid % 8) * 2;

    auto load_rows = [&](const float * src, int64_t t0, int n) {
        for (int idx = tid; idx < C*D/4; idx += blockDim.x) {
            const int rr = idx / (D/4);
            const int d4 = idx % (D/4);
            float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (rr < n) {
                x = *(const float4 *) (src + iq3*sq3 + (t0 + rr)*sq2 + iq1*sq1 + 4*d4);
            }
            *(float4 *) &KQ[rr][4*d4] = x;
        }
    };

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        const int64_t t0 = (int64_t) chunk*C;
        const int     n  = (int) min((int64_t) C, n_tok - t0);
        const float * pc = prep + ((seq*H + h)*n_chunks + chunk)*gdn_chunk_stride(C);
        load_rows(k, t0, n);
        if (tid < C) {
            gam[tid] = pc[2*C*C + tid];
            wv[tid]  = pc[2*C*C + C + tid];
            bet[tid] = pc[2*C*C + 2*C + tid];
        }
        if (tid == 0) {
            gam_last = pc[2*C*C + 3*C];
        }
        __syncthreads();

        // B = diag(beta) (V - diag(gamma) K S)
        {
            float x[2][4];
            gdn_rows_times_state<D, W, SW>(KQ, S, r0, c0, qd, x);
            // lane qd finishes row r0 + qd/2, columns c0 + 2*(qd%2) and the next
            const int    ra = qd / 2;
            const int    rr = r0 + ra;
            const int    cb = c0 + 2*(qd % 2);
            const float  xa = qd % 2 ? x[ra][2] : x[ra][0];
            const float  xb = qd % 2 ? x[ra][3] : x[ra][1];
            float v0 = 0.0f;
            float v1 = 0.0f;
            if (rr < n) {
                const float * vr = v + seq*sv3 + (t0 + rr)*sv2 + h*sv1 + col0 + cb;
                v0 = vr[0];
                v1 = vr[1];
            }
            Bs[rr][cb]     = bet[rr] * (v0 - gam[rr] * xa);
            Bs[rr][cb + 1] = bet[rr] * (v1 - gam[rr] * xb);
        }
        __syncthreads();

        // U = T B (T and P are zero above the diagonal)
        float u0 = 0.0f;
        float u1 = 0.0f;
        {
            const float * tr = pc + r*C;
            for (int i = 0; i <= r; i += 4) {
                const float4 t4 = *(const float4 *) &tr[i];
                const float  ti[4] = {t4.x, t4.y, t4.z, t4.w};
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float2 b = *(const float2 *) &Bs[i + j][c];
                    u0 += ti[j] * b.x;
                    u1 += ti[j] * b.y;
                }
            }
            Us[r][c]     = u0;
            Us[r][c + 1] = u1;
        }
        __syncthreads();

        // O's P U, before Us is scaled
        float p0 = 0.0f;
        float p1 = 0.0f;
        {
            const float * pr = pc + C*C + r*C;
            for (int i = 0; i <= r; i += 4) {
                const float4 p4 = *(const float4 *) &pr[i];
                const float  pi[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float2 u = *(const float2 *) &Us[i + j][c];
                    p0 += pi[j] * u.x;
                    p1 += pi[j] * u.y;
                }
            }
        }
        __syncthreads();
        Us[r][c]     = u0 * wv[r];
        Us[r][c + 1] = u1 * wv[r];
        __syncthreads();

        // the update K^T diag(w) U, added to gamma_last S once the outputs have read S
        float acc[4][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
        for (int i = 0; i < n; i++) {
            const float4 kd = *(const float4 *) &KQ[i][d0];
            const float2 u  = *(const float2 *) &Us[i][cu];
            acc[0][0] += kd.x*u.x; acc[0][1] += kd.x*u.y;
            acc[1][0] += kd.y*u.x; acc[1][1] += kd.y*u.y;
            acc[2][0] += kd.z*u.x; acc[2][1] += kd.z*u.y;
            acc[3][0] += kd.w*u.x; acc[3][1] += kd.w*u.y;
        }
        __syncthreads();
        load_rows(q, t0, n);
        __syncthreads();

        // O = diag(gamma) Q S + P U
        {
            float o[2][4];
            gdn_rows_times_state<D, W, SW>(KQ, S, r0, c0, qd, o);
            // P U of (r, c..c+1) from the lane that computed it: lane (r, c) is tid r*8 + c/2
            const int    ra = qd / 2;
            const int    rr = r0 + ra;
            const int    cb = c0 + 2*(qd % 2);
            const int    src_lane = (rr*8 + cb/2) % 32;
            const float  pa = __shfl_sync(0xffffffff, p0, src_lane);
            const float  pb = __shfl_sync(0xffffffff, p1, src_lane);
            const float  oa = qd % 2 ? o[ra][2] : o[ra][0];
            const float  ob = qd % 2 ? o[ra][3] : o[ra][1];
            if (rr < n) {
                float * ar = attn + ((seq*n_tok_attn + t0 + rr)*H + h)*D + col0 + cb;
                ar[0] = (gam[rr] * oa + pa) * scale;
                ar[1] = (gam[rr] * ob + pb) * scale;
            }
        }
        __syncthreads();
#pragma unroll
        for (int a = 0; a < 4; a++) {
            S[d0 + a][cu]     = gam_last * S[d0 + a][cu]     + acc[a][0];
            S[d0 + a][cu + 1] = gam_last * S[d0 + a][cu + 1] + acc[a][1];
        }
        __syncthreads();
    }

    float * s_out = state_out + (seq*H + h)*D*D;
    for (int idx = tid; idx < W*D; idx += blockDim.x) {
        const int cl = idx / D;
        const int d  = idx % D;
        s_out[(col0 + cl)*D + d] = S[d][cl];
    }
}
