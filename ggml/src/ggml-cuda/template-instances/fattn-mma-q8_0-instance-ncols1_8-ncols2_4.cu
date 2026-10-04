// MMA flash attention reading q8_0 K/V tiles (fattn-mma-q.cuh)

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_Q_CASE(GGML_TYPE_Q8_0, 128, 128, 8, 4);
DECL_FATTN_MMA_Q_CASE(GGML_TYPE_Q8_0, 256, 256, 8, 4);
