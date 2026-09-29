#include "rht.cuh"
#include "rht-impl.cuh"

void ggml_cuda_op_rht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src, dst));

    uint64_t seed;
    memcpy(&seed, dst->op_params, sizeof(seed));

    const rht_cfg cfg;
    rht_shape     sh;
    GGML_ASSERT(rht_make_shape(src->ne[0], cfg, sh));

    const int64_t nrows = ggml_nrows(src);
    GGML_ASSERT(nrows <= INT_MAX && src->ne[1] <= UINT32_MAX && src->ne[2] <= UINT32_MAX);

    const ggml_cuda_src1_fmt out = ggml_cuda_get_src1_fmt(dst);

    // kernel B writes F32 only: its Q8_1 comes from one quantize pass over it, instead of one per consumer
    const bool via_f32 = out != GGML_CUDA_SRC1_F32 && rht_use_b(sh, nrows, cfg);

    ggml_cuda_pool_alloc<float> tmp(ctx.pool());
    const size_t n_tmp = rht_tmp_floats(sh, nrows, cfg, via_f32 ? GGML_CUDA_SRC1_F32 : out);
    if (n_tmp > 0) {
        tmp.alloc(n_tmp);
    }
    ggml_cuda_pool_alloc<float> y(ctx.pool());
    if (via_f32) {
        y.alloc(ggml_nelements(src));
    }

    const rht_args a = rht_make_args(sh, src->data, src->ne, src->nb, via_f32 ? (void *) y.ptr : dst->data,
        via_f32 ? GGML_CUDA_SRC1_F32 : out, seed, tmp.ptr);
    rht_launch(a, sh, cfg, ctx.stream());

    if (via_f32) {
        const int64_t n    = src->ne[0];
        const int64_t npad = GGML_PAD(n, MATRIX_ROW_PADDING);
        const int64_t s2   = n*src->ne[1];
        const int64_t s3   = s2*src->ne[2];
        if (out == GGML_CUDA_SRC1_Q8_1) {
            quantize_row_q8_1_cuda(y.ptr, nullptr, dst->data, GGML_TYPE_Q8_0, n, n, s2, s3, npad, src->ne[1], src->ne[2], src->ne[3], ctx.stream());
        } else {
            // any type of the layout: quantize_mmq_q8_1_cuda only picks the layout by it
            const ggml_type type = out == GGML_CUDA_SRC1_Q8_1_MMQ_DS4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
            quantize_mmq_q8_1_cuda(y.ptr, nullptr, dst->data, type, n, n, s2, s3, npad, src->ne[1], src->ne[2], src->ne[3], ctx.stream());
        }
    }
}

// Whether to write fmt for n_consumers instead of F32: only where it's no more work than F32 plus a quantize
// per consumer. The one-consumer limits were measured on an RTX 5090.
bool ggml_cuda_rht_write_q8_1(const ggml_tensor * dst, const int fmt, const int n_consumers) {
    const rht_cfg cfg;
    rht_shape     sh;
    if (dst->ne[0] % QK8_1 != 0 || !rht_make_shape(dst->ne[0], cfg, sh)) {
        return false;
    }
    if (n_consumers >= 2) {
        return true;
    }
    if (rht_use_b(sh, ggml_nrows(dst), cfg)) {
        return false;
    }
    return sh.K < 64 || (fmt == GGML_CUDA_SRC1_Q8_1 && sh.K < 100);
}
