#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// g and beta computed in the kernel from the inputs of the skipped nodes that make them: g = softplus(alpha + dt)*a and
// beta = sigmoid(beta_in), alpha and beta_in laid out as beta, dt and a one value per head
struct ggml_cuda_gdn_gating {
    const float * alpha;
    const float * dt;
    const float * a;
    const float * beta_in;
};

// state_rows: the GET_ROWS that gathers the state (one sequence), skipped: the kernel reads its source rows in place
void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * state_rows = nullptr,
                                  const ggml_cuda_gdn_gating * gating = nullptr);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache,
                                              const ggml_tensor * state_rows = nullptr,
                                              const ggml_cuda_gdn_gating * gating = nullptr);

// GDN launches that read the state in place / that computed the gating, for tests
int64_t ggml_cuda_gdn_state_in_place_count();
int64_t ggml_cuda_gdn_gating_count();
