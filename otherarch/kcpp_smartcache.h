#pragma once

// SmartCache restore points of gpttype_adapter.cpp, shared with tests/test-kcpp-smartcache.cpp: the checkpoint lists
// of the live context and the slots, the choice of restore point, and where a save of the live context goes.
//
// A checkpoint holds a model's state at a position that the attention KV can't give back by itself (the recurrent
// part, cut back with seq_rm otherwise). It never changes once written, so the live context and slots share them.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

enum class kcpp_ckpt_kind : uint8_t {
    latest, // the end of a request's prompt; the only kind with logits to sample from
    tail,   // kcpp_ckpt_tail tokens before it
    system, // the end of the system prompt
};

inline const char * kcpp_ckpt_kind_name(kcpp_ckpt_kind k) {
    return k == kcpp_ckpt_kind::latest ? "latest" : k == kcpp_ckpt_kind::tail ? "tail" : "system";
}

constexpr int kcpp_ckpt_capacity = 8;  // per context
constexpr int kcpp_ckpt_tail     = 32; // the tail checkpoint's distance from the end; contexts this short get none
constexpr int kcpp_ckpt_newest   = 3;  // never evicted

template <class Payload>
struct kcpp_ckpt {
    int            pos    = 0; // the state after tokens [0, pos)
    uint64_t       serial = 0; // creation order
    kcpp_ckpt_kind kind   = kcpp_ckpt_kind::latest;
    Payload        data;
};

template <class Payload>
using kcpp_ckpt_ptr = std::shared_ptr<kcpp_ckpt<Payload>>;

// Every checkpoint of a model has the same size: one dropped by every holder goes back to a free list with its
// buffers, so the next one needs no allocation, zero-fill or page faults.
template <class Payload>
class kcpp_ckpt_pool {
public:
    kcpp_ckpt_ptr<Payload> acquire() {
        std::unique_ptr<kcpp_ckpt<Payload>> obj;
        {
            std::lock_guard<std::mutex> lock(st->mtx);
            if (!st->free.empty()) {
                obj = std::move(st->free.back());
                st->free.pop_back();
            }
        }
        if (!obj) {
            obj.reset(new kcpp_ckpt<Payload>());
        }
        obj->serial = ++st->serial;
        std::weak_ptr<state> w = st;
        return kcpp_ckpt_ptr<Payload>(obj.release(), [w](kcpp_ckpt<Payload> * p) {
            if (auto s = w.lock()) {
                std::lock_guard<std::mutex> lock(s->mtx);
                s->free.emplace_back(p);
            } else {
                delete p;
            }
        });
    }

    // a newer creation serial for an existing checkpoint: it counts as just made
    void touch(kcpp_ckpt<Payload> & c) {
        std::lock_guard<std::mutex> lock(st->mtx);
        c.serial = ++st->serial;
    }

    // frees the dropped checkpoints' memory
    void release_free() {
        std::lock_guard<std::mutex> lock(st->mtx);
        st->free.clear();
    }

    size_t n_free() const {
        std::lock_guard<std::mutex> lock(st->mtx);
        return st->free.size();
    }

private:
    struct state {
        std::mutex mtx;
        std::vector<std::unique_ptr<kcpp_ckpt<Payload>>> free;
        uint64_t serial = 0;
    };
    std::shared_ptr<state> st = std::make_shared<state>();
};

template <class Payload>
struct kcpp_ckpt_list {
    std::vector<kcpp_ckpt_ptr<Payload>> items; // by position

    size_t size() const { return items.size(); }
    bool empty() const { return items.empty(); }
    void clear() { items.clear(); }

    // a checkpoint at an existing position replaces it; past the capacity one is evicted, ctx_len being the
    // context's length (the request's prompt length while the live context gains checkpoints)
    void add(kcpp_ckpt_ptr<Payload> c, int ctx_len) {
        items.erase(std::remove_if(items.begin(), items.end(), [&](const kcpp_ckpt_ptr<Payload> & x) { return x->pos == c->pos; }), items.end());
        items.insert(std::upper_bound(items.begin(), items.end(), c->pos,
                                      [](int pos, const kcpp_ckpt_ptr<Payload> & x) { return pos < x->pos; }), std::move(c));
        while ((int) items.size() > kcpp_ckpt_capacity) {
            items.erase(items.begin() + evict_index(ctx_len));
        }
    }

    // a cut back to pos drops the checkpoints past it
    void truncate(int pos) {
        items.erase(std::remove_if(items.begin(), items.end(), [&](const kcpp_ckpt_ptr<Payload> & x) { return x->pos > pos; }), items.end());
    }

    kcpp_ckpt_ptr<Payload> find(int pos) const {
        for (const auto & c : items) {
            if (c->pos == pos) {
                return c;
            }
        }
        return nullptr;
    }

    // the latest checkpoint at or before pos; one at the prompt's end (prompt_len) must be a latest checkpoint, the
    // others leave nothing to decode and no logits to sample from
    kcpp_ckpt_ptr<Payload> at_or_before(int pos, int prompt_len) const {
        for (auto it = items.rbegin(); it != items.rend(); ++it) {
            if ((*it)->pos <= pos && ((*it)->pos < prompt_len || (*it)->kind == kcpp_ckpt_kind::latest)) {
                return *it;
            }
        }
        return nullptr;
    }

    // Never evicted: the newest by creation, and the system checkpoint (or the oldest, when there is none). First go
    // the checkpoints before 40 % of the context (earliest first), so the rest gather in the later part of a
    // conversation, where edits that don't save the context fall; then the one whose neighbors are closest together
    // (position 0 left of the first, the context's end right of the last; the older one on ties).
    int evict_index(int ctx_len) const {
        std::vector<int> by_serial(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            by_serial[i] = (int) i;
        }
        std::sort(by_serial.begin(), by_serial.end(), [&](int a, int b) { return items[a]->serial < items[b]->serial; });
        std::vector<bool> kept(items.size(), false);
        for (int k = 0; k < kcpp_ckpt_newest && k < (int) by_serial.size(); ++k) {
            kept[by_serial[by_serial.size() - 1 - k]] = true;
        }
        int sys = -1;
        for (size_t i = 0; i < items.size(); ++i) {
            if (items[i]->kind == kcpp_ckpt_kind::system) {
                sys = (int) i;
                break;
            }
        }
        kept[sys >= 0 ? sys : by_serial[0]] = true;

        for (size_t i = 0; i < items.size(); ++i) {
            if (!kept[i] && (int64_t) items[i]->pos*10 < (int64_t) ctx_len*4) {
                return (int) i;
            }
        }
        int best = -1;
        int64_t best_gap = 0;
        for (size_t i = 0; i < items.size(); ++i) {
            if (kept[i]) {
                continue;
            }
            const int64_t left  = i > 0 ? items[i - 1]->pos : 0;
            const int64_t right = i + 1 < items.size() ? items[i + 1]->pos : std::max(ctx_len, items[i]->pos);
            const int64_t gap   = right - left;
            if (best < 0 || gap < best_gap || (gap == best_gap && items[i]->serial < items[best]->serial)) {
                best = (int) i;
                best_gap = gap;
            }
        }
        return best >= 0 ? best : 0;
    }
};

inline int kcpp_shared_prefix(const std::vector<int> & a, const std::vector<int> & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return (int) i;
}

// a context a request could continue: the live context (src -1) or a slot
template <class Payload>
struct kcpp_ctx_view {
    int                              src       = -1;
    const std::vector<int> *         tokens    = nullptr;
    const kcpp_ckpt_list<Payload> *  ckpts     = nullptr;
    int                              limit     = 0;    // no checkpoint past this (the live context: its decoded tokens)
    uint64_t                         last_used = 0;
    bool                             usable    = true; // a slot holds a snapshot with the request's media
};

template <class Payload>
struct kcpp_restore {
    int                    src    = -1;
    int                    point  = 0;     // the prompt is processed from here
    bool                   full   = false; // the prompt contains the whole context: continue from its end
    kcpp_ckpt_ptr<Payload> ckpt;           // else the checkpoint at point (none: start over)
    int                    shared = 0;     // the common prefix of the context and the prompt
};

// The furthest restore point: a context's end if the prompt contains it, else its latest checkpoint at or before
// their common prefix. Ties go to the live context, then to the most recently used slot.
template <class Payload>
kcpp_restore<Payload> kcpp_choose_restore(const std::vector<int> & prompt, const std::vector<kcpp_ctx_view<Payload>> & ctxs) {
    kcpp_restore<Payload> best;
    uint64_t best_used = 0;
    bool have = false;
    for (const auto & c : ctxs) {
        if (!c.usable) {
            continue;
        }
        kcpp_restore<Payload> r;
        r.src    = c.src;
        r.shared = kcpp_shared_prefix(*c.tokens, prompt);
        if (r.shared == (int) c.tokens->size()) {
            r.full  = true;
            r.point = r.shared;
        } else if (c.ckpts) {
            r.ckpt = c.ckpts->at_or_before(std::min(r.shared, c.limit), (int) prompt.size());
            r.point = r.ckpt ? r.ckpt->pos : 0;
        }
        if (c.src >= 0 && r.point == 0) {
            continue; // loading a slot to start over gains nothing
        }
        const bool better = !have || r.point > best.point ||
                            (r.point == best.point && best.src >= 0 && (c.src < 0 || c.last_used > best_used));
        if (better) {
            best = r;
            best_used = c.last_used;
            have = true;
        }
    }
    return best;
}

// The live context goes to a slot before a cut or a slot load discards part of it when the prompt keeps less than
// 70 % of it. The denominator is the live context's length: the shorter of the two would make a short new
// conversation that shares only the system block look like a continuation.
inline bool kcpp_keep_is_low(int shared, int live_len) {
    return (int64_t) shared*10 < (int64_t) live_len*7;
}

struct kcpp_slot_view {
    bool     empty     = true;
    bool     identical = false; // the live context's tokens and media
    int      prefix    = 0;     // > 0: the slot's tokens are a prefix of the live context's (an older snapshot of it)
    uint64_t last_used = 0;
};

enum class kcpp_save_action { none, touch, save };

// Where a save of the live context goes: nowhere for a tiny context; an identical slot is only touched; else the
// slot holding an older snapshot of it, an empty slot, or the least recently used one, never the one about to be
// loaded (exclude).
inline kcpp_save_action kcpp_save_target(int live_len, const std::vector<kcpp_slot_view> & slots, int exclude, int & slot) {
    slot = -1;
    if (live_len <= kcpp_ckpt_tail) {
        return kcpp_save_action::none;
    }
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].identical) {
            slot = (int) i;
            return kcpp_save_action::touch;
        }
    }
    int older = -1, empty = -1, lru = -1;
    for (int i = 0; i < (int) slots.size(); ++i) {
        if (i == exclude) {
            continue;
        }
        if (slots[i].prefix > 0 && (older < 0 || slots[i].prefix > slots[older].prefix)) {
            older = i;
        }
        if (slots[i].empty && empty < 0) {
            empty = i;
        }
        if (lru < 0 || slots[i].last_used < slots[lru].last_used) {
            lru = i;
        }
    }
    slot = older >= 0 ? older : empty >= 0 ? empty : lru;
    return slot >= 0 ? kcpp_save_action::save : kcpp_save_action::none;
}
