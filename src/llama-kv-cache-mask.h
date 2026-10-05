#pragma once

#include "llama-kv-cells.h"

#include <cstdint>
#include <vector>

// A causal KQ mask row (no SWA or ALiBi) for a sequence that every used cell carries: keep the cells at positions up
// to p1, drop the rest and the empty ones (position -1); with 2D positions (M-RoPE) also drop the cells at p1 whose 2D
// position is past (p1_x, p1_y). Appends to idxs the cells at positions from p_near on, the ones the sequence's later
// rows in the ubatch can differ in. Same result as the per-cell checks; p_near <= p1, so idxs holds every cell at p1.
template <typename T>
static void llama_kv_mask_row_causal(const llama_kv_cells & cells, int64_t n_kv, llama_pos p1, bool is_2d, llama_pos p1_x,
                                     llama_pos p1_y, llama_pos p_near, T keep, T drop, T * row, std::vector<uint32_t> & idxs) {
    const llama_pos * pos = cells.pos_data();
    for (int64_t j = 0; j < n_kv; ++j) {
        row[j] = pos[j] >= 0 && pos[j] <= p1 ? keep : drop;
    }
    const size_t idxs0 = idxs.size();
    for (int64_t j = 0; j < n_kv; ++j) {
        if (pos[j] >= 0 && pos[j] >= p_near) {
            idxs.push_back((uint32_t) j);
        }
    }
    if (is_2d) {
        for (size_t k = idxs0; k < idxs.size(); ++k) {
            const uint32_t j = idxs[k];
            if (pos[j] == p1 && cells.ext_get(j).is_2d_gt(p1_x, p1_y)) {
                row[j] = drop;
            }
        }
    }
}
