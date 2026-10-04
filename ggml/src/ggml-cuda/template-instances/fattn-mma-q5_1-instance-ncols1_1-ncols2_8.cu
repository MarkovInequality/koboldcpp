// MMA flash attention reading q5_1 K/V tiles (fattn-mma-q.cuh)

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_Q_CASE(GGML_TYPE_Q5_1, 128, 128, 1, 8);
DECL_FATTN_MMA_Q_CASE(GGML_TYPE_Q5_1, 256, 256, 1, 8);
