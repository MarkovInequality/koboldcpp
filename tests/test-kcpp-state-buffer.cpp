// The SmartCache state buffers (otherarch/kcpp_state_buffer.h): fit sizes a buffer through growth, shrinking and
// regrowth with every byte writable, keeps the memory where the new size fits the capacity, and on Linux grows in
// place, keeping the pages it has (bytes written before growing are still there), with transparent huge pages
// unless THP is disabled. A std::vector freed and reallocated on growth fails both Linux checks.
//
// usage: test-kcpp-state-buffer [--bench]
//   --bench: grow a buffer to 1 GB and then 2 GB, against a buffer freed and reallocated on growth (the old way)

#include "kcpp_state_buffer.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

static const size_t MB = (size_t) 1 << 20;

static void fill(uint8_t * p, size_t n, uint8_t seed) {
    for (size_t i = 0; i < n; i += 4093) {
        p[i] = (uint8_t) (seed + i);
    }
    if (n) {
        p[n - 1] = (uint8_t) (seed + n - 1);
    }
}

static bool check(const uint8_t * p, size_t n, uint8_t seed) {
    for (size_t i = 0; i < n; i += 4093) {
        if (p[i] != (uint8_t) (seed + i)) {
            return false;
        }
    }
    return n == 0 || p[n - 1] == (uint8_t) (seed + n - 1);
}

static int check_fit() {
    int fails = 0;
    kcpp_state_buffer buf;
    uint8_t seed = 1;
    // from nothing, shrinking, regrowing within and past the capacity, empty
    for (size_t n : { (size_t) 1000, 3*MB, 100*MB, 40*MB, 120*MB, 300*MB, 300*MB + 7, 5*MB, 290*MB, (size_t) 0, 64*MB }) {
        const uint8_t * before = buf.data();
        const size_t    cap    = buf.capacity();
        buf.fit(n);
        bool ok = buf.size() == n && buf.capacity() >= n && buf.empty() == (n == 0);
        if (n <= cap) {
            ok = ok && buf.data() == before;
        }
        fill(buf.data(), n, seed);
        ok = ok && check(buf.data(), n, seed);
        printf("fit %10zu bytes: capacity %10zu, %s  %s\n", n, buf.capacity(), n <= cap ? "reused" : "grown ", ok ? "ok" : "FAIL");
        fails += !ok;
        ++seed;
    }
    buf.clear();
    buf.shrink_to_fit();
    const bool ok = buf.data() == nullptr && buf.capacity() == 0;
    printf("clear + shrink_to_fit frees: %s\n", ok ? "ok" : "FAIL");
    fails += !ok;

    kcpp_state_buffer a, b;
    a.fit(10*MB);
    fill(a.data(), a.size(), 7);
    b = std::move(a);
    const bool moved = a.data() == nullptr && a.size() == 0 && b.size() == 10*MB && check(b.data(), b.size(), 7);
    printf("move: %s\n", moved ? "ok" : "FAIL");
    fails += !moved;
    return fails;
}

#if defined(__linux__)
static bool thp_enabled() {
    std::ifstream f("/sys/kernel/mm/transparent_hugepage/enabled");
    std::string s;
    std::getline(f, s);
    return !s.empty() && s.find("[never]") == std::string::npos;
}

// AnonHugePages of the mapping that contains p, in kB; -1 if not found
static long anon_huge_kb(const void * p) {
    std::ifstream f("/proc/self/smaps");
    std::string line;
    bool in = false;
    const uintptr_t a = (uintptr_t) p;
    while (std::getline(f, line)) {
        unsigned long lo, hi;
        if (sscanf(line.c_str(), "%lx-%lx ", &lo, &hi) == 2) {
            in = a >= lo && a < hi;
        } else if (in && line.rfind("AnonHugePages:", 0) == 0) {
            return atol(line.c_str() + 14);
        }
    }
    return -1;
}

static int check_linux() {
    int fails = 0;
    kcpp_state_buffer buf;
    buf.fit(300*MB);
    fill(buf.data(), buf.size(), 3);
    buf.fit(700*MB);
    const bool kept = check(buf.data(), 300*MB, 3);
    printf("growth 300 -> 700 MB keeps the bytes written: %s\n", kept ? "ok" : "FAIL");
    fails += !kept;
    if (thp_enabled()) {
        const long kb = anon_huge_kb(buf.data());
        const bool ok = kb > 0;
        printf("AnonHugePages %ld kB: %s\n", kb, ok ? "ok" : "FAIL");
        fails += !ok;
    } else {
        printf("transparent huge pages disabled, huge page check skipped\n");
    }
    return fails;
}
#endif

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the fit this replaced: freed and reallocated on growth, pages touched one by one
static void old_fit(std::vector<uint8_t> & buf, size_t n) {
    if (buf.capacity() < n) {
        buf.clear();
        buf.shrink_to_fit();
        buf.reserve(n + n/4);
    }
    buf.resize(n);
    for (size_t i = 0; i < n; i += 4096) {
        buf[i] = 0;
    }
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        for (int rep = 0; rep < 3; ++rep) {
            double t = now_s();
            {
                std::vector<uint8_t> b;
                old_fit(b, 1024*MB);
                old_fit(b, 2048*MB);
            }
            const double t_old = now_s() - t;
            t = now_s();
            {
                kcpp_state_buffer b;
                b.fit(1024*MB);
                b.fit(2048*MB);
            }
            const double t_new = now_s() - t;
            printf("fit 1 GB, then 2 GB, then free: reallocating with 4 KB pages %.3f s, kcpp_state_buffer %.3f s\n", t_old, t_new);
        }
        return 0;
    }

    int fails = check_fit();
#if defined(__linux__)
    fails += check_linux();
#endif
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
