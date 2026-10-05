// The causal KQ mask row of a sequence that every used cell carries (llama_kv_mask_row_causal, the fast path of
// llama_kv_cache::set_input_kq_mask) against the per-cell checks of the generic path, over random cell layouts:
// empty cells, cells used without a sequence, cells shared by several sequences, positions out of order, repeated
// and past the row's, with and without 2D positions (M-RoPE). The fast path is only taken where llama_kv_cells::seq_has_all_used holds, checked against a
// count of the cells; the row and the cells recorded for the sequence's later rows must equal the generic ones.
//
// usage: test-kv-mask

#include "llama-kv-cells.h"
#include "llama-kv-cache-mask.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(42);
    int fails = 0, n_fast = 0, n_cases = 0;
    for (int it = 0; it < 4000; ++it) {
        const uint32_t n = 1 + rng() % 600;
        llama_kv_cells cells;
        cells.resize(n);
        const int n_seq   = 1 + rng() % 3;
        const int p_range = 1 + rng() % 800;
        const int layout  = rng() % 4; // 0: empty cells, 1: also cells without a sequence, 2: shared, 3: one sequence, all used
        for (uint32_t i = 0; i < n; ++i) {
            if (layout != 3 && rng() % 4 == 0) {
                continue;
            }
            cells.pos_set(i, layout == 3 ? (llama_pos) i : (llama_pos) (rng() % p_range));
            llama_kv_cell_ext ext;
            ext.x = rng() % 4;
            ext.y = rng() % 4;
            cells.ext_set(i, ext);
            if (layout == 1 && rng() % 8 == 0) {
                continue;
            }
            cells.seq_add(i, layout == 3 ? 0 : rng() % n_seq);
            const llama_seq_id s2 = rng() % n_seq;
            if (layout == 2 && rng() % 3 == 0 && !cells.seq_has(i, s2)) {
                cells.seq_add(i, s2);
            }
        }
        for (int q = 0; q < 4; ++q) {
            const llama_seq_id seq_id = rng() % n_seq;
            const llama_pos    p1     = rng() % (p_range + 4);
            const llama_pos    p_near = (llama_pos) (rng() % (p1 + 33)) - 32;
            const bool         is_2d  = rng() % 2;
            const llama_pos    p1_x   = rng() % 4;
            const llama_pos    p1_y   = rng() % 4;
            const int64_t      n_kv   = 1 + rng() % n;

            uint32_t n_used = 0, n_carry = 0;
            for (uint32_t i = 0; i < n; ++i) {
                n_used  += !cells.is_empty(i);
                n_carry += !cells.is_empty(i) && cells.seq_has(i, seq_id);
            }
            const bool all = n_used == n_carry;
            ++n_cases;
            if (cells.seq_has_all_used(seq_id) != all) {
                printf("  case %d: seq_has_all_used %d, counted %d  FAIL\n", it, !all, all);
                ++fails;
                continue;
            }
            if (!all) {
                continue;
            }
            ++n_fast;

            // the generic path's row: per-cell checks, recording the cells near the sequence's positions
            const float keep = 0.0f, drop = -INFINITY;
            std::vector<float>    ref(n_kv);
            std::vector<uint32_t> ref_idxs;
            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j) || !cells.seq_has(j, seq_id)) {
                    ref[j] = drop;
                    continue;
                }
                const llama_pos p0 = cells.pos_get(j);
                if (p0 >= p_near) {
                    ref_idxs.push_back((uint32_t) j);
                }
                ref[j] = p0 > p1 || (is_2d && p0 == p1 && cells.ext_get(j).is_2d_gt(p1_x, p1_y)) ? drop : keep;
            }

            std::vector<float>    row(n_kv, 1.0f);
            std::vector<uint32_t> idxs;
            llama_kv_mask_row_causal(cells, n_kv, p1, is_2d, p1_x, p1_y, p_near, keep, drop, row.data(), idxs);
            if (row != ref || idxs != ref_idxs) {
                printf("  case %d (n %u, n_kv %lld, layout %d): fast row differs  FAIL\n", it, n, (long long) n_kv, layout);
                ++fails;
            }
        }
    }
    printf("%d rows (%d through the fast path), %d failed\n%s\n", n_cases, n_fast, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
