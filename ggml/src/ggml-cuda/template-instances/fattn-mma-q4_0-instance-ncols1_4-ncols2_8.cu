// MMA flash attention reading q4_0 K/V tiles (fattn-mma-q.cuh)

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_Q_CASE(GGML_TYPE_Q4_0, 128, 128, 4, 8);
DECL_FATTN_MMA_Q_CASE(GGML_TYPE_Q4_0, 256, 256, 4, 8);
