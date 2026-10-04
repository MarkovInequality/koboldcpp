// GPU contention probe: replays graphs of short dependent kernels and reports the wall time lost to
// preemptions (gaps between kernels and stretches inside them). Run it next to every benchmark session.
//
// usage: stall [n_kernels] [spin_ns]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void spin_kernel(unsigned long long * ts, int i, int spin_ns) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long s = globaltimer(), e = s;
        while (e - s < (unsigned long long) spin_ns) {
            e = globaltimer();
        }
        ts[2*i] = s;
        ts[2*i + 1] = e;
    }
}

int main(int argc, char ** argv) {
    const int n    = argc > 1 ? atoi(argv[1]) : 200000;
    const int spin = argc > 2 ? atoi(argv[2]) : 10000;
    const int G    = 1000; // kernels per graph, about one decode step
    if (n < 2*G || n % G != 0) {
        fprintf(stderr, "n_kernels must be a multiple of %d and at least %d\n", G, 2*G);
        return 1;
    }

    unsigned long long * d;
    cudaMalloc(&d, 16ull*n);
    cudaStream_t st;
    cudaStreamCreate(&st);

    std::vector<cudaGraphExec_t> execs;
    for (int b = 0; b < n/G; b++) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < G; i++) {
            spin_kernel<<<1, 32, 0, st>>>(d, b*G + i, spin);
        }
        cudaStreamEndCapture(st, &g);
        cudaGraphInstantiate(&ge, g, 0);
        cudaGraphDestroy(g);
        execs.push_back(ge);
    }
    for (auto & ge : execs) {
        cudaGraphLaunch(ge, st);
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    std::vector<unsigned long long> h(2*n);
    cudaMemcpy(h.data(), d, 16ull*n, cudaMemcpyDeviceToHost);

    const double total = (h[2*n - 1] - h[0])/1e3;
    double busy = 0, lost = 0, in_lost = 0;
    int n_in = 0;
    std::vector<double> gaps, stalls, at;
    for (int i = 0; i < n; i++) {
        const double dur = (h[2*i + 1] - h[2*i])/1e3;
        busy += dur;
        if (dur - spin/1e3 > 50) {
            n_in++;
            in_lost += dur - spin/1e3;
        }
        if (i > 0) {
            const double gap = (h[2*i] - h[2*i - 1])/1e3;
            gaps.push_back(gap);
            if (gap > 50) {
                stalls.push_back(gap);
                at.push_back(h[2*i - 1]/1e3);
                lost += gap;
            }
        }
    }
    std::sort(gaps.begin(), gaps.end());
    printf("n=%d spin=%.1fus total=%.1f ms, kernel busy %.1f ms, median gap %.2f us, p99 gap %.2f us\n",
           n, spin/1e3, total/1e3, busy/1e3, gaps[gaps.size()/2], gaps[gaps.size()*99/100]);
    printf("stalls>50us between kernels: %zu, lost %.2f ms (%.1f%% of wall); in-kernel stretches>50us: %d, lost %.2f ms\n",
           stalls.size(), lost/1e3, 100*lost/total, n_in, in_lost/1e3);
    printf("contention: %.1f%% of wall lost\n", 100*(lost + in_lost)/total);
    if (!stalls.empty()) {
        std::vector<double> sz(stalls);
        std::sort(sz.begin(), sz.end());
        printf("stall size p10 %.0f p50 %.0f p90 %.0f max %.0f us\n",
               sz[sz.size()/10], sz[sz.size()/2], sz[sz.size()*9/10], sz.back());
    }
    if (at.size() > 1) {
        std::vector<double> iv;
        for (size_t i = 1; i < at.size(); i++) {
            iv.push_back(at[i] - at[i - 1]);
        }
        std::sort(iv.begin(), iv.end());
        printf("interval between stalls p10 %.0f p50 %.0f p90 %.0f us\n", iv[iv.size()/10], iv[iv.size()/2], iv[iv.size()*9/10]);
    }
    return 0;
}
