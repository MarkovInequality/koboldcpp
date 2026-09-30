#include "rht.cuh"
#include "rht-impl.cuh"

#include <atomic>

static std::atomic<int64_t> rht_q8_1_count{0};
static std::atomic<int64_t> rht_fused_count{0};

int64_t ggml_cuda_rht_q8_1_count() {
    return rht_q8_1_count.load();
}

int64_t ggml_cuda_rht_fused_count() {
    return rht_fused_count.load();
}

// what a rotation reads: rows of x (strides nbx), through a prologue
struct rht_input {
    const void   * x;
    const size_t * nbx;
    int            pro  = RHT_PRO_NONE;
    const void   * x2   = nullptr; // RHT_PRO_NORM: w; GLU: u
    const size_t * nbx2 = nullptr; // GLU: u's strides
    float          eps  = 0.0f;
};

// kernel B writes F32 only: its Q8_1 comes from one quantize pass over it, instead of one per consumer
static bool rht_via_f32(const rht_shape & sh, const int64_t nrows, const rht_cfg & cfg, const ggml_cuda_src1_fmt out) {
    return out != GGML_CUDA_SRC1_F32 && rht_use_b(sh, nrows, cfg);
}

static void rht_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const rht_input & in) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));

    uint64_t seed;
    memcpy(&seed, dst->op_params, sizeof(seed));

    const rht_cfg cfg;
    rht_shape     sh;
    GGML_ASSERT(rht_make_shape(dst->ne[0], cfg, sh));

    const int64_t nrows = ggml_nrows(dst);
    GGML_ASSERT(nrows <= INT_MAX && dst->ne[1] <= UINT32_MAX && dst->ne[2] <= UINT32_MAX);

    const ggml_cuda_src1_fmt out     = ggml_cuda_get_src1_fmt(dst);
    const bool               via_f32 = rht_via_f32(sh, nrows, cfg, out);

    ggml_cuda_pool_alloc<float> tmp(ctx.pool());
    const size_t n_tmp = rht_tmp_floats(sh, nrows, cfg, via_f32 ? GGML_CUDA_SRC1_F32 : out, in.pro);
    if (n_tmp > 0) {
        tmp.alloc(n_tmp);
    }
    ggml_cuda_pool_alloc<float> y(ctx.pool());
    if (via_f32) {
        y.alloc(ggml_nelements(dst));
    }

    rht_args a = rht_make_args(sh, in.x, dst->ne, in.nbx, via_f32 ? (void *) y.ptr : dst->data,
        via_f32 ? GGML_CUDA_SRC1_F32 : out, seed, tmp.ptr);
    a.pro  = in.pro;
    a.src2 = (const char *) in.x2;
    a.eps  = in.eps;
    if (in.nbx2) {
        a.nb21 = in.nbx2[1];
        a.nb22 = in.nbx2[2];
        a.nb23 = in.nbx2[3];
    }
    rht_launch(a, sh, cfg, ctx.stream());

    if (out != GGML_CUDA_SRC1_F32) {
        rht_q8_1_count++;
    }

    if (via_f32) {
        const int64_t n    = dst->ne[0];
        const int64_t npad = GGML_PAD(n, MATRIX_ROW_PADDING);
        const int64_t s2   = n*dst->ne[1];
        const int64_t s3   = s2*dst->ne[2];
        if (out == GGML_CUDA_SRC1_Q8_1) {
            quantize_row_q8_1_cuda(y.ptr, nullptr, dst->data, GGML_TYPE_Q8_0, n, n, s2, s3, npad, dst->ne[1], dst->ne[2], dst->ne[3], ctx.stream());
        } else {
            // any type of the layout: quantize_mmq_q8_1_cuda only picks the layout by it
            const ggml_type type = out == GGML_CUDA_SRC1_Q8_1_MMQ_DS4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
            quantize_mmq_q8_1_cuda(y.ptr, nullptr, dst->data, type, n, n, s2, s3, npad, dst->ne[1], dst->ne[2], dst->ne[3], ctx.stream());
        }
    }
}

void ggml_cuda_op_rht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(src->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_are_same_shape(src, dst));

    rht_input in;
    in.x   = src->data;
    in.nbx = src->nb;
    rht_run(ctx, dst, in);
}

// Whether to write fmt for n_consumers instead of F32: only where it's no more work than F32 plus a quantize
// per consumer. Kernel B quantizes its F32 temporary once, the same work as one consumer's quantize, and then
// never writes dst while reading its inputs, so a fused producer's input may share dst's memory. The kernel A
// limits were measured on an RTX 5090.
bool ggml_cuda_rht_write_q8_1(const ggml_tensor * dst, const int fmt, const int n_consumers) {
    const rht_cfg cfg;
    rht_shape     sh;
    if (dst->ne[0] % QK8_1 != 0 || !rht_make_shape(dst->ne[0], cfg, sh)) {
        return false;
    }
    if (n_consumers >= 2 || rht_use_b(sh, ggml_nrows(dst), cfg)) {
        return true;
    }
    GGML_UNUSED(fmt);
    return ggml_nrows(dst) == 1 || sh.K < 64;
}

static bool rht_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_backend_buffer_get_alloc_size(b->buffer, b) && b0 < a0 + ggml_backend_buffer_get_alloc_size(a->buffer, a);
}

// A fused rotation writes dst while it reads the producers' inputs, and the graph allocator may have put dst
// in the block of an input freed by the producer. That is safe when the launch reads everything before writing
// (kernel A in two passes; the F32 temporary of via_f32), when the input doesn't overlap dst, or when an F32 dst
// has exactly the input's rows: every kernel reads a row completely before writing it.
static bool rht_fusion_safe(const ggml_tensor * dst, std::initializer_list<const ggml_tensor *> inputs) {
    const rht_cfg cfg;
    rht_shape     sh;
    if (!rht_make_shape(dst->ne[0], cfg, sh)) {
        return false;
    }
    const int64_t            nrows = ggml_nrows(dst);
    const ggml_cuda_src1_fmt out   = ggml_cuda_get_src1_fmt(dst);
    if (rht_via_f32(sh, nrows, cfg, out) || (!rht_use_b(sh, nrows, cfg, out) && (sh.M > 1 || sh.multipass))) {
        return true;
    }
    for (const ggml_tensor * t : inputs) {
        if (t->op == GGML_OP_NONE || !rht_overlap(dst, t)) {
            continue;
        }
        const bool same_rows = out == GGML_CUDA_SRC1_F32 && t->data == dst->data &&
            t->nb[1] == dst->nb[1] && t->nb[2] == dst->nb[2] && t->nb[3] == dst->nb[3];
        if (!same_rows) {
            return false;
        }
    }
    return true;
}

// Where kernel B's mix is compute-bound, reading the prologue's inputs costs more than the producer's pass saves
// (RTX 5090, 512 rows: order 172 is 0.5-7.5 % slower fused, 148 20-27 % faster).
bool ggml_cuda_rht_fusion_pays(const ggml_tensor * dst) {
    const rht_cfg cfg;
    rht_shape     sh;
    return rht_make_shape(dst->ne[0], cfg, sh) && !(rht_use_b(sh, ggml_nrows(dst), cfg) && sh.K > 148);
}

// ggml_can_fuse for {GLU, RHT}, except that the GLU may be written over its gate (ggml_glu_split_inplace), a view
// ggml_can_fuse refuses: here the view's source is the GLU's own input, with no other consumer
static bool rht_glu_can_fuse(const ggml_cgraph * cgraph, const int i) {
    const ggml_op ops[] = { GGML_OP_GLU, GGML_OP_RHT };
    const ggml_tensor * glu = cgraph->nodes[i];
    if (!glu->view_src) {
        return ggml_can_fuse(cgraph, i, ops, 2);
    }
    if (i + 1 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * rht = cgraph->nodes[i + 1];
    const ggml_tensor * g   = glu->src[0];
    auto use_count = [&](const ggml_tensor * t) {
        return cgraph->use_counts[ggml_hash_find(&cgraph->visited_hash_set, t)];
    };
    return glu->op == GGML_OP_GLU && rht->op == GGML_OP_RHT && (glu->flags & GGML_TENSOR_FLAG_COMPUTE) &&
        (rht->flags & GGML_TENSOR_FLAG_COMPUTE) && !(glu->flags & GGML_TENSOR_FLAG_OUTPUT) && use_count(glu) == 1 &&
        rht->src[0] == glu && ggml_are_same_shape(glu, rht) &&
        glu->view_src == g && glu->view_offs == 0 && !g->view_src && !(g->flags & GGML_TENSOR_FLAG_OUTPUT) && use_count(g) == 1;
}

static bool rht_f32_rows(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float);
}

int ggml_cuda_rht_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    const ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_RMS_NORM) {
        const ggml_op ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RHT };
        if (!ggml_can_fuse(cgraph, i, ops, 3)) {
            return 0;
        }
        const ggml_tensor * mul = cgraph->nodes[i + 1];
        ggml_tensor       * rht = cgraph->nodes[i + 2];
        const ggml_tensor * x   = node->src[0];
        const ggml_tensor * w   = mul->src[0] == node ? mul->src[1] : mul->src[0];
        if (rht->src[0] != mul || !rht_f32_rows(x) || node->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
                w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != x->ne[0] || ggml_nelements(w) != w->ne[0] ||
                !ggml_cuda_rht_fusion_pays(rht) || !rht_fusion_safe(rht, { x })) {
            return 0;
        }
        rht_input in;
        in.x   = x->data;
        in.nbx = x->nb;
        in.pro = RHT_PRO_NORM;
        in.x2  = w->data;
        in.eps = ggml_get_op_params_f32(node, 0);
        rht_run(ctx, rht, in);
        rht_fused_count++;
        return 2;
    }

    if (node->op == GGML_OP_GLU) {
        if (!rht_glu_can_fuse(cgraph, i)) {
            return 0;
        }
        ggml_tensor       * rht = cgraph->nodes[i + 1];
        const ggml_glu_op   op  = ggml_get_glu_op(node);
        const ggml_tensor * g   = node->src[0];
        const ggml_tensor * u   = node->src[1];
        if (rht->src[0] != node || (op != GGML_GLU_OP_SWIGLU && op != GGML_GLU_OP_GEGLU) || node->type != GGML_TYPE_F32 ||
                !rht_f32_rows(g) || !ggml_is_contiguous_1(g) || (u && (!rht_f32_rows(u) || !ggml_is_contiguous_1(u))) ||
                !ggml_cuda_rht_fusion_pays(rht) || !rht_fusion_safe(rht, { g, u ? u : g })) {
            return 0;
        }
        // as the GLU kernel: without src1, the two halves of each row, swapped by op param 1
        const int64_t nc      = node->ne[0];
        const bool    swapped = !u && ggml_get_op_params_i32(node, 1) != 0;
        rht_input in;
        in.x    = (const char *) g->data + (!u && swapped ? nc*sizeof(float) : 0);
        in.nbx  = g->nb;
        in.pro  = op == GGML_GLU_OP_SWIGLU ? RHT_PRO_SWIGLU : RHT_PRO_GEGLU;
        in.x2   = u ? u->data : (const char *) g->data + (swapped ? 0 : nc*sizeof(float));
        in.nbx2 = u ? u->nb : g->nb;
        rht_run(ctx, rht, in);
        rht_fused_count++;
        return 1;
    }

    return 0;
}
