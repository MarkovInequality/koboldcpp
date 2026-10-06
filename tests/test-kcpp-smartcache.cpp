// SmartCache restore points (otherarch/kcpp_smartcache.h), without a model: checkpoint eviction (capacity, the early
// part first, then the smallest gap, what is never evicted, duplicates), lookup and truncation, sharing and recycling
// of checkpoint objects, the choice of restore point across the live context and slots, the keep fraction, where a
// save of the live context goes, and the checkpoint positions of a simulated agentic session.

#include "kcpp_smartcache.h"

#include <cstdio>
#include <string>
#include <vector>

using payload = std::vector<char>;
using ckpt_list = kcpp_ckpt_list<payload>;

static int fails = 0;

static void expect(bool ok, const std::string & what) {
    printf("  %-74s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    fails += !ok;
}

static std::vector<int> positions(const ckpt_list & l) {
    std::vector<int> p;
    for (const auto & c : l.items) {
        p.push_back(c->pos);
    }
    return p;
}

static std::string str(const std::vector<int> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i ? " " : "") + std::to_string(v[i]);
    }
    return s + "]";
}

static kcpp_ckpt_pool<payload> pool;

static void add(ckpt_list & l, int pos, int ctx_len, kcpp_ckpt_kind kind = kcpp_ckpt_kind::tail) {
    auto c = pool.acquire();
    c->pos = pos;
    c->kind = kind;
    l.add(c, ctx_len);
}

static void check_eviction() {
    printf("eviction\n");
    {
        ckpt_list l;
        add(l, 1000, 10000, kcpp_ckpt_kind::system);
        for (int p = 2000; p <= 9000; p += 1000) {
            add(l, p, 10000);
        }
        expect(positions(l) == std::vector<int>({1000, 3000, 4000, 5000, 6000, 7000, 8000, 9000}),
               "capacity 8; the earliest before 40 % of the context goes first: " + str(positions(l)));
    }
    {
        ckpt_list l;
        add(l, 100, 10000, kcpp_ckpt_kind::system);
        for (int p : {5000, 5100, 6000, 7000, 8000, 9000, 9500, 9900}) {
            add(l, p, 10000);
        }
        expect(positions(l) == std::vector<int>({100, 5000, 6000, 7000, 8000, 9000, 9500, 9900}),
               "none before 40 %: the one whose neighbors are closest goes: " + str(positions(l)));
    }
    {
        ckpt_list l;
        add(l, 1000, 13000, kcpp_ckpt_kind::system);
        for (int p : {5300, 7300, 6300, 8300, 10300, 12000, 12500, 13000}) {
            add(l, p, 13000);
        }
        expect(positions(l) == std::vector<int>({1000, 5300, 6300, 8300, 10300, 12000, 12500, 13000}),
               "equal gaps: the older one goes: " + str(positions(l)));
    }
    {
        ckpt_list l;
        add(l, 30000, 100000, kcpp_ckpt_kind::system);
        for (int p : {50000, 60000, 70000, 80000, 90000, 1000, 2000, 3000}) {
            add(l, p, 100000);
        }
        expect(positions(l) == std::vector<int>({1000, 2000, 3000, 30000, 50000, 70000, 80000, 90000}),
               "the 3 newest and the system checkpoint stay, even before 40 %: " + str(positions(l)));
    }
    {
        ckpt_list l;
        for (int p : {1000, 50000, 60000, 70000, 80000, 90000, 95000, 99000, 100000}) {
            add(l, p, 100000);
        }
        expect(positions(l) == std::vector<int>({1000, 50000, 60000, 70000, 80000, 95000, 99000, 100000}),
               "without a system checkpoint the oldest stays: " + str(positions(l)));
    }
    {
        ckpt_list l;
        add(l, 5000, 10000);
        auto first = l.items[0];
        add(l, 5000, 10000, kcpp_ckpt_kind::latest);
        expect(l.size() == 1 && l.items[0] != first && l.items[0]->kind == kcpp_ckpt_kind::latest,
               "a checkpoint at an existing position replaces it");
    }
}

static void check_lookup() {
    printf("lookup and invalidation\n");
    ckpt_list l;
    add(l, 2000, 10000, kcpp_ckpt_kind::system);
    add(l, 9968, 10000, kcpp_ckpt_kind::tail);
    add(l, 10000, 10000, kcpp_ckpt_kind::latest);
    auto pos = [](const kcpp_ckpt_ptr<payload> & c) { return c ? c->pos : -1; };
    expect(pos(l.at_or_before(9999, 20000)) == 9968, "the latest checkpoint at or before a position");
    expect(pos(l.at_or_before(10000, 20000)) == 10000 && pos(l.at_or_before(1999, 20000)) == -1, "at the position itself; none before the first");
    expect(pos(l.at_or_before(10000, 10000)) == 10000, "at the prompt's end: a latest checkpoint (it has logits)");
    expect(pos(l.at_or_before(9968, 9968)) == 2000, "at the prompt's end: not a tail checkpoint (nothing to decode, no logits)");
    l.truncate(9968);
    expect(positions(l) == std::vector<int>({2000, 9968}), "a cut drops the checkpoints past it: " + str(positions(l)));
}

static void check_sharing() {
    printf("sharing and recycling\n");
    kcpp_ckpt_pool<payload> p;
    ckpt_list a, b;
    auto c = p.acquire();
    c->pos = 100;
    c->data.resize(1 << 20);
    const kcpp_ckpt<payload> * obj = c.get();
    const char * data = c->data.data();
    a.add(c, 1000);
    b.add(c, 1000);
    c.reset();
    a.clear();
    expect(b.size() == 1 && b.items[0]->data.data() == data && p.n_free() == 0, "a shared checkpoint survives one holder dropping it");
    b.clear();
    expect(p.n_free() == 1, "the last drop returns the object to the free list");
    auto d = p.acquire();
    d->data.resize(1 << 20);
    expect(d.get() == obj && d->data.data() == data && p.n_free() == 0, "the next checkpoint reuses its object and buffers");
    const uint64_t s0 = d->serial;
    p.touch(*d);
    expect(d->serial > s0, "a touched checkpoint counts as the newest");
    d.reset();
    p.release_free();
    expect(p.n_free() == 0, "the free list can be released");
}

static void check_matching() {
    printf("matching\n");
    std::vector<int> prompt(1000);
    for (int i = 0; i < 1000; ++i) {
        prompt[i] = i;
    }
    std::vector<int> live(prompt.begin(), prompt.begin() + 950);
    live.push_back(-1);
    std::vector<int> slot_full(prompt.begin(), prompt.begin() + 900);
    std::vector<int> slot_part(prompt.begin(), prompt.begin() + 800);
    slot_part.push_back(-1);
    ckpt_list live_ck, part_ck, part_ck2;
    add(live_ck, 500, 951, kcpp_ckpt_kind::latest);
    add(live_ck, 940, 951, kcpp_ckpt_kind::tail);
    add(part_ck, 780, 801, kcpp_ckpt_kind::tail);
    add(part_ck2, 780, 801, kcpp_ckpt_kind::tail);

    using view = kcpp_ctx_view<payload>;
    auto v = [](int src, const std::vector<int> & t, const ckpt_list * ck, int limit, uint64_t used, bool usable = true) {
        view x;
        x.src = src; x.tokens = &t; x.ckpts = ck; x.limit = limit; x.last_used = used; x.usable = usable;
        return x;
    };
    auto r = kcpp_choose_restore<payload>(prompt, {v(-1, live, &live_ck, 950, 0), v(0, slot_full, nullptr, 1 << 30, 5)});
    expect(r.src == -1 && r.point == 940 && r.ckpt && !r.full && r.shared == 950, "the furthest restore point: the live context's checkpoint at 940");
    r = kcpp_choose_restore<payload>(prompt, {v(-1, live, &live_ck, 930, 0), v(0, slot_full, nullptr, 1 << 30, 5)});
    expect(r.src == 0 && r.point == 900 && r.full, "the live context's checkpoints stop at its decoded tokens; a slot it contains wins");
    std::vector<int> live900(prompt.begin(), prompt.begin() + 900);
    live900.push_back(-1);
    ckpt_list ck900;
    add(ck900, 900, 901, kcpp_ckpt_kind::latest);
    r = kcpp_choose_restore<payload>(prompt, {v(-1, live900, &ck900, 900, 0), v(0, slot_full, nullptr, 1 << 30, 5)});
    expect(r.src == -1 && r.point == 900, "a tie goes to the live context");
    r = kcpp_choose_restore<payload>(prompt, {v(-1, live, nullptr, 950, 0), v(0, slot_part, &part_ck, 1 << 30, 3), v(1, slot_part, &part_ck2, 1 << 30, 7)});
    expect(r.src == 1 && r.point == 780, "between slots, the most recently used");
    r = kcpp_choose_restore<payload>(prompt, {v(-1, live, &live_ck, 950, 0), v(0, prompt, nullptr, 1 << 30, 5, false)});
    expect(r.src == -1 && r.point == 940, "a slot with other media isn't used");
    std::vector<int> other = {5, 6, 7};
    r = kcpp_choose_restore<payload>(prompt, {v(-1, other, nullptr, 3, 0), v(0, other, nullptr, 1 << 30, 5)});
    expect(r.src == -1 && r.point == 0 && !r.ckpt, "nothing usable: start over in the live context");
    r = kcpp_choose_restore<payload>(slot_full, {v(-1, live, &live_ck, 950, 0)});
    expect(r.point == 500, "a prompt that ends before a tail checkpoint restores an earlier one");
}

static void check_keep() {
    printf("keep fraction\n");
    expect(kcpp_keep_is_low(12000, 52000), "a 13k prompt sharing 12k with a 52k live context: save (12k / 52k, not 12k / 13k)");
    expect(!kcpp_keep_is_low(36400, 52000) && kcpp_keep_is_low(36399, 52000), "70 % keeps the live context without a save, below it saves");
}

static void check_save_order() {
    printf("save order\n");
    auto slot = [](bool empty, bool identical, int prefix, uint64_t used) {
        kcpp_slot_view s;
        s.empty = empty; s.identical = identical; s.prefix = prefix; s.last_used = used;
        return s;
    };
    int t;
    expect(kcpp_save_target(32, {slot(true, false, 0, 0)}, -1, t) == kcpp_save_action::none, "a context of 32 tokens or fewer isn't saved");
    expect(kcpp_save_target(100, {slot(false, false, 50, 1), slot(false, true, 0, 2)}, 1, t) == kcpp_save_action::touch && t == 1,
           "an identical slot is only touched");
    expect(kcpp_save_target(100, {slot(true, false, 0, 0), slot(false, false, 40, 9), slot(false, false, 60, 8)}, -1, t) == kcpp_save_action::save && t == 2,
           "an older snapshot of the live context is replaced first (the longest)");
    expect(kcpp_save_target(100, {slot(false, false, 0, 1), slot(false, false, 60, 8)}, 1, t) == kcpp_save_action::save && t == 0,
           "...but never the slot about to be loaded");
    expect(kcpp_save_target(100, {slot(false, false, 0, 1), slot(true, false, 0, 0), slot(true, false, 0, 0)}, -1, t) == kcpp_save_action::save && t == 1,
           "then an empty slot");
    expect(kcpp_save_target(100, {slot(false, false, 0, 4), slot(false, false, 0, 2), slot(false, false, 0, 3)}, 1, t) == kcpp_save_action::save && t == 2,
           "then the least recently used, never the one about to be loaded");
}

// an agentic session: the system prompt ends at 2k, a 27.5k first prompt, then turns that add a 150-token reply and
// 3k tokens of input; each request adds latest - 32 and latest, the first one also the system checkpoint
static std::vector<int> simulate(int turns) {
    const int S = 2000;
    ckpt_list l;
    int L = 27500, r = 0;
    for (int t = 0; t < turns; ++t) {
        if (t > 0) {
            r = L + 150;
            L = r + 3000;
        }
        if (r < S && S < L) {
            add(l, S, L, kcpp_ckpt_kind::system);
        }
        if (L - r > kcpp_ckpt_tail && L - kcpp_ckpt_tail > S) {
            add(l, L - kcpp_ckpt_tail, L, kcpp_ckpt_kind::tail);
        }
        add(l, L, L, kcpp_ckpt_kind::latest);
    }
    return positions(l);
}

static void check_placement() {
    printf("placement in a simulated agentic session\n");
    const std::vector<std::pair<int, std::vector<int>>> want = {
        {10,  {2000, 27468, 33768, 40100, 46400, 52700, 55818, 55850}},
        {40,  {2000, 84168, 96800, 118850, 144050, 147200, 150318, 150350}},
        {100, {2000, 178700, 207050, 266868, 292100, 336200, 339318, 339350}},
    };
    for (const auto & w : want) {
        const std::vector<int> got = simulate(w.first);
        expect(got == w.second, std::to_string(w.first) + " turns: " + str(got));
    }
}

int main() {
    check_eviction();
    check_lookup();
    check_sharing();
    check_matching();
    check_keep();
    check_save_order();
    check_placement();
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
