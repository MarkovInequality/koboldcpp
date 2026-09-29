#pragma once

#include "common.cuh"

void ggml_cuda_op_rht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// whether the RHT node dst writes the Q8_1 layout fmt (a ggml_cuda_src1_fmt) for its n_consumers consumers
bool ggml_cuda_rht_write_q8_1(const ggml_tensor * dst, int fmt, int n_consumers);
