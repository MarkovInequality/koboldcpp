#pragma once

#include "common.cuh"

void ggml_cuda_op_rht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// whether the RHT node dst writes the Q8_1 layout fmt (a ggml_cuda_src1_fmt) for its n_consumers consumers
bool ggml_cuda_rht_write_q8_1(const ggml_tensor * dst, int fmt, int n_consumers);

// runs {RMS_NORM, MUL, RHT} or {GLU, RHT} starting at node i as one rotation; returns the nodes to skip, 0 if it didn't
int ggml_cuda_rht_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// keeps the inputs of the rotations ggml_cuda_rht_try_fuse will fuse allocated until their RHT node
void ggml_cuda_rht_add_alloc_deps(ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params);

// whether fusing a producer into the RHT node dst is faster, by its shape and kernel
bool ggml_cuda_rht_fusion_pays(const ggml_tensor * dst);

// rotations that wrote Q8_1 / ran fused, for tests
int64_t ggml_cuda_rht_q8_1_count();
int64_t ggml_cuda_rht_fused_count();
