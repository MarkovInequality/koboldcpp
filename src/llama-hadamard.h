#pragma once

#include <cassert>
#include <cmath>

// orthonormal Walsh-Hadamard rotation matrix (Sylvester construction)
// H is symmetric, orthonormal (H * H^T == I) and self-inverse (H * H == I)
// n must be a power of 2
inline void llama_gen_hadamard_matrix(float * out, int n) {
    assert(n > 0);
    assert((n & (n - 1)) == 0); // must be power of 2

    out[0] = 1.0f / sqrtf((float) n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = out[i * n + j];

                out[(i + s) * n + (j    )] =  val;
                out[(i    ) * n + (j + s)] =  val;
                out[(i + s) * n + (j + s)] = -val;
            }
        }
    }
}

// in-place orthonormal Walsh-Hadamard transform (O(n log n) butterfly)
// computes vec = H_n * vec, where H_n is the orthonormal Walsh-Hadamard of size n
// n must be a power of 2
inline void llama_hadamard_inplace(float * vec, int n) {
    assert(n > 0);
    assert((n & (n - 1)) == 0); // must be power of 2

    const float scale = 1.0f / sqrtf((float) n);

    for (int j = 0; j < n; j++) {
        vec[j] *= scale;
    }

    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; j++) {
                const float u = vec[i + j];
                const float v = vec[i + len + j];
                vec[i + j] = u + v;
                vec[i + len + j] = u - v;
            }
        }
    }
}
