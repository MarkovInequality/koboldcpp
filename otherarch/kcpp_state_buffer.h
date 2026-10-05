#pragma once

// SmartCache state buffers of gpttype_adapter.cpp, shared with tests/test-kcpp-state-buffer.cpp.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

#if defined(__linux__)
#include <sys/mman.h>
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
#endif

// A byte buffer whose new bytes are left uninitialized: zero-filling a ~400 MB state costs far more than the copy
// into it. On Linux it is one anonymous mapping with transparent huge pages that grows in place (mremap), so a slot
// re-saved after its state grew keeps the pages it has and only faults in the rest: a freed page is slow to get
// back (WSL2 returns it to the host within seconds; faulting it in again cost ~0.4 s/GB, 4 KB pages ~0.65 s/GB).
class kcpp_state_buffer
{
public:
    kcpp_state_buffer() = default;
    kcpp_state_buffer(const kcpp_state_buffer &) = delete;
    kcpp_state_buffer & operator=(const kcpp_state_buffer &) = delete;
    kcpp_state_buffer(kcpp_state_buffer && o) noexcept { swap(o); }
    kcpp_state_buffer & operator=(kcpp_state_buffer && o) noexcept {
        if (this != &o) {
            release();
            swap(o);
        }
        return *this;
    }
    ~kcpp_state_buffer() { release(); }

    uint8_t * data() { return ptr; }
    const uint8_t * data() const { return ptr; }
    size_t size() const { return n; }
    size_t capacity() const { return cap; }
    bool empty() const { return n == 0; }
    void clear() { n = 0; }
    // like std::vector's, a request: the memory goes only once the buffer is empty
    void shrink_to_fit() {
        if (n == 0) {
            release();
        }
    }

    // sizes the buffer to bytes, with every byte backed by memory; grows with headroom
    void fit(size_t bytes) {
        if (bytes > cap) {
            const size_t step    = (size_t) 2 << 20;
            const size_t new_cap = (bytes + bytes/4 + step - 1)/step*step;
#if defined(__linux__)
            void * p = ptr ? mremap(ptr, cap, new_cap, MREMAP_MAYMOVE)
                           : mmap(nullptr, new_cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) {
                throw std::bad_alloc();
            }
            madvise(p, new_cap, MADV_HUGEPAGE);
#else
            release();
            void * p = malloc(new_cap);
            if (p == nullptr) {
                throw std::bad_alloc();
            }
#endif
            ptr = (uint8_t *) p;
            cap = new_cap;
        }
        if (bytes > backed) {
            prefault(ptr + backed, bytes - backed);
            backed = bytes;
        }
        n = bytes;
    }

private:
    uint8_t * ptr    = nullptr;
    size_t    n      = 0;
    size_t    cap    = 0;
    size_t    backed = 0; // bytes faulted in so far

    void swap(kcpp_state_buffer & o) {
        uint8_t * p = ptr; ptr = o.ptr; o.ptr = p;
        size_t t;
        t = n;      n      = o.n;      o.n      = t;
        t = cap;    cap    = o.cap;    o.cap    = t;
        t = backed; backed = o.backed; o.backed = t;
    }

    void release() {
        if (ptr) {
#if defined(__linux__)
            munmap(ptr, cap);
#else
            free(ptr);
#endif
        }
        ptr = nullptr;
        n = cap = backed = 0;
    }

    // faults the pages in before a device-to-host copy writes them: faulting them in during the copy costs about
    // twice as much
    static void prefault(uint8_t * p, size_t len) {
#if defined(__linux__)
        const uintptr_t page = 4096;
        const uintptr_t beg  = (uintptr_t) p & ~(page - 1);
        if (madvise((void *) beg, (uintptr_t) p + len - beg, MADV_POPULATE_WRITE) == 0) {
            return;
        }
#endif
        for (size_t i = 0; i < len; i += 4096) {
            p[i] = 0;
        }
    }
};
