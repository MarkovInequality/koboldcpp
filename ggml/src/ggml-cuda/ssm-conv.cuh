#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// the conv state update of the gated-delta-net models as one kernel: concat(state, x^T) -> copies of conv windows
// (the rollback snapshots) -> ssm_conv -> [bias add] -> silu, for up to GGML_CUDA_SSM_CONV_UPDATE_MAX_T tokens
#define GGML_CUDA_SSM_CONV_UPDATE_MAX_T    32
#define GGML_CUDA_SSM_CONV_UPDATE_MAX_SNAP 16

void ggml_cuda_op_ssm_conv_update(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, ggml_tensor * const * snaps,
                                  int n_snap, const ggml_tensor * ssm_conv, const ggml_tensor * bias_add, ggml_tensor * silu);

// launches of ggml_cuda_op_ssm_conv_update, for tests
int64_t ggml_cuda_ssm_conv_update_count();
