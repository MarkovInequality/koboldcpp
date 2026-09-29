// Collects, for every distinct input x of a model's weight GEMMs, the Gram matrix G = sum x x^T of that input
// over a set of documents (GPTQ's Hessian up to scale), before any Hadamard rotation of it, so it serves any HQ
// seed and any plain type. Writes an imatrix GGUF: each weight keeps <w>.in_sum2 [n, 1] (= diag G) and
// <w>.counts [1, 1], and each distinct input adds <owner>.in_gram F32 [n, n], shared with the weights that read
// the same input through hessian.alias.names / hessian.alias.owners. The owner is the first of those weights in
// the model file's tensor order.
//
// The Grams don't fit in memory at once, so the documents run once per pass, each pass accumulating the Grams
// of a group of layers (on the GPU with cuBLAS SYRK in the CUDA build, in fp64 on the CPU otherwise or with
// --accum cpu) and writing them at their offsets in the file. A run that stops can --resume.
//
// usage: hessian-collect -m MODEL.gguf (--docs DOCS.jsonl | --text FILE --chunk N [--chunks K]) -o OUT.gguf
//            [-ngl N] [-ot REGEX=DEVICE] [--no-mmap] [--ubatch 8192] [--ctx N] [-ctk TYPE] [-ctv TYPE]
//            [--max-seqs 8] [--output-stride 8] [--group-layers N|auto] [--layers 0,31,63,out|A-B]
//            [--resume] [--imatrix-out FILE] [--dataset NAME] [--accum cpu|cuda] [--threads N]
//            [--stop-after-pass N] [--no-op-offload] [--trim-docs] [--dry-run] [--test-hook]
//
// --trim-docs drops the tokens after a document's last counted one, which can't reach a counted row (for runs
// that count a narrow window of long documents). --dry-run prints the input groups and each input's layout and
// stops; --test-hook checks that recording the GEMM inputs leaves the logits bitwise unchanged and stops.
//
// DOCS.jsonl has one document per line, {"id", "text", "count": [[start, end), ...]}: "count" lists the token
// spans that enter the Grams and is optional (the whole document counts); the other tokens are context only.
// Each document is its own sequence from position 0. The LM head's input exists only where logits are
// requested: every --output-stride-th counted position.

#include "llama.h"
#include "llama-context.h"
#include "common/common.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#ifdef GGML_USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] static void fatal(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

#ifdef GGML_USE_CUDA
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) fatal("%s: %s", #x, cudaGetErrorString(e_)); } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) fatal("%s: cuBLAS status %d", #x, (int) s_); } while (0)
#endif

struct document {
    std::string              id;
    std::vector<llama_token> tok;
    std::vector<uint8_t>     counted;
    std::vector<uint8_t>     output;   // logits requested here
    int64_t                  n_counted = 0;
};

struct group {
    std::string              owner;
    std::vector<std::string> weights;   // owner first
    int64_t                  n     = 0;
    int                      layer = 0; // n_layer for weights outside the blocks (the LM head)
    bool                     selected = false;
    bool                     active   = false;
    int64_t                  rows     = 0;
    std::string              layout;    // of the input node, for --dry-run

    std::vector<double>      acc64;     // CPU accumulator, upper triangle of a row-major n x n
    float *                  acc32 = nullptr; // CUDA accumulator, column-major n x n, SYRK upper
};

struct tensor_slot {
    std::string name;
    size_t      offset; // in the file
    int64_t     ne0, ne1;
};

struct batch_plan {
    std::vector<int> docs;
    int64_t          len; // padded length of each sequence; 0 for a lone long document
};

struct collector {
    llama_context *         ctx = nullptr;
    int                     n_layer = 0;
    std::vector<group>      groups;
    std::unordered_map<std::string, int> group_of_weight;

    std::unordered_map<const ggml_tensor *, int> x_group;
    uint64_t                graph_id = UINT64_MAX;

    const std::vector<document> * docs = nullptr;
    std::vector<int>        seq_doc; // current batch

    bool                    use_cuda = false;
    int                     n_threads = 1;
    double                  t_accum = 0;
    std::string             error;

    std::vector<std::pair<int64_t, int64_t>> runs;
    std::vector<int32_t>    row_tok;
    std::vector<float>      host_rows;
    std::vector<double>     xt;

#ifdef GGML_USE_CUDA
    cublasHandle_t          cublas = nullptr;
    cudaStream_t            stream = nullptr;
    float *                 staging = nullptr;
    int64_t                 staging_rows = 0;
    int64_t                 staging_n = 0;
    size_t                  min_free = SIZE_MAX;
    size_t                  free_at_start = 0;
#endif

    void refresh() {
        const auto st = ctx->get_mm_inputs();
        if (st.graph_id == graph_id) {
            return;
        }
        graph_id = st.graph_id;
        x_group.clear();
        for (const auto & r : *st.inputs) {
            auto it = group_of_weight.find(r.w->name);
            if (it != group_of_weight.end()) {
                x_group[r.x] = it->second;
            }
        }
    }

    llama_token pad = 0;

    // every token of the ubatch is the document's token at its (sequence, position): checks the accessor's
    // positions and sequence ids against the batch the tool built
    bool check_ubatch(const llama_ubatch * ub) {
        for (uint32_t i = 0; i < ub->n_tokens; ++i) {
            const int s = ub->seq_id[i][0];
            if (s < 0 || s >= (int) seq_doc.size()) {
                error = "ubatch token with sequence id " + std::to_string(s) + " outside the batch";
                return false;
            }
            const document & d = (*docs)[seq_doc[s]];
            const llama_pos p = ub->pos[i];
            const llama_token want = p < (llama_pos) d.tok.size() ? d.tok[p] : pad;
            if (!ub->token || ub->token[i] != want) {
                error = "ubatch token " + std::to_string(i) + " (seq " + std::to_string(s) + ", pos " + std::to_string(p) +
                        ") is not the document's token there";
                return false;
            }
        }
        return true;
    }

    // rows of x in the current ubatch -> token indices of the ubatch, then runs of counted rows; where the rows
    // are the ubatch's outputs, only the documents' own outputs count, not a logit forced for the decode
    bool counted_runs(const ggml_tensor * x, const llama_ubatch * ub) {
        const int64_t nrows = ggml_nrows(x);
        const bool outputs = nrows != (int64_t) ub->n_tokens;
        row_tok.clear();
        if (!outputs) {
            for (uint32_t i = 0; i < ub->n_tokens; ++i) {
                row_tok.push_back(i);
            }
        } else {
            for (uint32_t i = 0; i < ub->n_tokens; ++i) {
                if (ub->output && ub->output[i]) {
                    row_tok.push_back(i);
                }
            }
            if ((int64_t) row_tok.size() != nrows) {
                error = std::string("input of ") + groups[x_group[x]].owner + " has rows that are neither the ubatch's tokens nor its outputs";
                return false;
            }
        }
        runs.clear();
        int64_t r0 = -1;
        for (int64_t r = 0; r <= nrows; ++r) {
            bool c = false;
            if (r < nrows) {
                const int t = row_tok[r];
                const int s = ub->seq_id[t][0];
                const document & d = (*docs)[seq_doc[s]];
                const llama_pos p = ub->pos[t];
                c = p < (llama_pos) d.tok.size() && (outputs ? d.output[p] : d.counted[p]);
            }
            if (c && r0 < 0) {
                r0 = r;
            } else if (!c && r0 >= 0) {
                runs.emplace_back(r0, r);
                r0 = -1;
            }
        }
        return true;
    }

    static bool uniform_rows(const ggml_tensor * x) {
        return x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) &&
               (x->ne[2] == 1 || x->nb[2] == x->nb[1]*x->ne[1]) && (x->ne[3] == 1 || x->nb[3] == x->nb[2]*x->ne[2]);
    }

    // rows [r0, r1) of x as a contiguous host array
    const float * host_rows_of(const ggml_tensor * x, int64_t r0, int64_t r1) {
        const int64_t n = x->ne[0];
        host_rows.resize((size_t) (r1 - r0)*n);
        if (uniform_rows(x)) {
            const size_t rs = n*sizeof(float);
            if (ggml_backend_buffer_is_host(x->buffer) && x->nb[1] == rs) {
                return (const float *) x->data + r0*n;
            }
            if (x->nb[1] == rs) {
                ggml_backend_tensor_get(x, host_rows.data(), r0*rs, (r1 - r0)*rs);
                return host_rows.data();
            }
            std::vector<uint8_t> span((r1 - r0 - 1)*x->nb[1] + rs);
            ggml_backend_tensor_get(x, span.data(), r0*x->nb[1], span.size());
            for (int64_t r = r0; r < r1; ++r) {
                memcpy(host_rows.data() + (r - r0)*n, span.data() + (r - r0)*x->nb[1], rs);
            }
            return host_rows.data();
        }
        std::vector<uint8_t> all(ggml_nbytes(x));
        ggml_backend_tensor_get(x, all.data(), 0, all.size());
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t i1 = r % x->ne[1], i2 = (r / x->ne[1]) % x->ne[2], i3 = r / (x->ne[1]*x->ne[2]);
            const uint8_t * row = all.data() + i1*x->nb[1] + i2*x->nb[2] + i3*x->nb[3];
            for (int64_t j = 0; j < n; ++j) {
                host_rows[(r - r0)*n + j] = *(const float *) (row + j*x->nb[0]);
            }
        }
        return host_rows.data();
    }

    // G += X^T X in fp64 over rows [r0, r1) of x; G is the upper triangle of a row-major n x n
    void accum_cpu(group & g, const ggml_tensor * x, int64_t r0, int64_t r1) {
        const int64_t n = g.n, k = r1 - r0;
        const float * rows = host_rows_of(x, r0, r1);
        xt.resize((size_t) n*k);
        for (int64_t r = 0; r < k; ++r) {
            for (int64_t j = 0; j < n; ++j) {
                xt[(size_t) j*k + r] = rows[r*n + j];
            }
        }
        const int64_t T = 32;
        const int64_t n_tiles = (n + T - 1)/T;
        std::atomic<int64_t> next{0};
        auto work = [&]() {
            for (int64_t ti; (ti = next.fetch_add(1)) < n_tiles;) {
                const int64_t i0 = ti*T, i1 = std::min(n, i0 + T);
                for (int64_t j0 = i0; j0 < n; j0 += T) {
                    const int64_t j1 = std::min(n, j0 + T);
                    for (int64_t i = i0; i < i1; ++i) {
                        const double * a = xt.data() + (size_t) i*k;
                        double * gi = g.acc64.data() + (size_t) i*n;
                        for (int64_t j = std::max(j0, i); j < j1; ++j) {
                            const double * b = xt.data() + (size_t) j*k;
                            double s = 0.0;
                            for (int64_t r = 0; r < k; ++r) {
                                s += a[r]*b[r];
                            }
                            gi[j] += s;
                        }
                    }
                }
            }
        };
        std::vector<std::thread> th;
        for (int t = 1; t < n_threads; ++t) {
            th.emplace_back(work);
        }
        work();
        for (auto & t : th) {
            t.join();
        }
    }

#ifdef GGML_USE_CUDA
    void accum_cuda(group & g, const ggml_tensor * x, int64_t r0, int64_t r1) {
        const int64_t n = g.n;
        const float one = 1.0f;
        if (!ggml_backend_buffer_is_host(x->buffer) && uniform_rows(x)) {
            const float * a = (const float *) ((const char *) x->data + r0*x->nb[1]);
            CUBLAS_CHECK(cublasSsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, (int) n, (int) (r1 - r0), &one, a,
                                     (int) (x->nb[1]/sizeof(float)), &one, g.acc32, (int) n));
            return;
        }
        const int64_t chunk = 1024;
        if (staging_n < n) {
            if (staging) {
                CUDA_CHECK(cudaFree(staging));
            }
            CUDA_CHECK(cudaMalloc(&staging, (size_t) n*chunk*sizeof(float)));
            staging_n = n;
        }
        for (int64_t c0 = r0; c0 < r1; c0 += chunk) {
            const int64_t c1 = std::min(r1, c0 + chunk);
            const float * h = host_rows_of(x, c0, c1);
            CUDA_CHECK(cudaMemcpyAsync(staging, h, (size_t) (c1 - c0)*n*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUBLAS_CHECK(cublasSsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, (int) n, (int) (c1 - c0), &one, staging,
                                     (int) n, &one, g.acc32, (int) n));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
#endif

    bool observe(ggml_tensor * x) {
        group & g = groups[x_group[x]];
        const auto st = ctx->get_mm_inputs();
        if (!st.ubatch) {
            error = "no ubatch while the graph runs";
            return false;
        }
        if (x->ne[0] != g.n) {
            error = "input width of " + g.owner + " changed";
            return false;
        }
        if (!check_ubatch(st.ubatch) || !counted_runs(x, st.ubatch)) {
            return false;
        }
        const double t0 = now_s();
        for (const auto & [r0, r1] : runs) {
#ifdef GGML_USE_CUDA
            if (use_cuda) {
                accum_cuda(g, x, r0, r1);
            } else
#endif
            {
                accum_cpu(g, x, r0, r1);
            }
            g.rows += r1 - r0;
        }
#ifdef GGML_USE_CUDA
        if (use_cuda && !runs.empty()) {
            // ggml may overwrite x as soon as we return
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
#endif
        t_accum += now_s() - t0;
        return true;
    }
};

static bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * c = (collector *) ud;
    c->refresh();
    // only the current pass's inputs are asked for: the scheduler splits the graph and syncs at each one
    auto it = c->x_group.find(t);
    if (it == c->x_group.end() || !c->groups[it->second].active) {
        return !ask;
    }
    if (ask) {
        return true;
    }
    if (!c->observe(t)) {
        fprintf(stderr, "%s\n", c->error.c_str());
        exit(1);
    }
    return true;
}

// ---------------------------------------------------------------- documents and batches

static void finish_doc(document & d, int stride) {
    d.output.assign(d.tok.size(), 0);
    int64_t k = 0;
    for (size_t i = 0; i < d.tok.size(); ++i) {
        if (d.counted[i]) {
            d.output[i] = k % stride == 0;
            ++k;
        }
    }
    d.n_counted = k;
}

static std::vector<document> load_docs(const std::string & path, const llama_vocab * vocab, int stride, bool trim) {
    std::ifstream in(path);
    if (!in) {
        fatal("cannot open %s", path.c_str());
    }
    std::vector<document> docs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const json j = json::parse(line);
        document d;
        d.id  = j.value("id", std::to_string(docs.size()));
        d.tok = common_tokenize(vocab, j.at("text").get<std::string>(), true, true);
        if (j.contains("count") && !j["count"].is_null()) {
            d.counted.assign(d.tok.size(), 0);
            int64_t prev_end = 0;
            for (const auto & sp : j["count"]) {
                const int64_t a = sp.at(0).get<int64_t>(), b = sp.at(1).get<int64_t>();
                if (a < prev_end || b < a || b > (int64_t) d.tok.size()) {
                    fatal("%s: count span [%" PRId64 ", %" PRId64 ") is out of order or outside its %zu tokens", d.id.c_str(), a, b, d.tok.size());
                }
                std::fill(d.counted.begin() + a, d.counted.begin() + b, 1);
                prev_end = b;
            }
            if (trim) {
                // causal: tokens after the last counted one can't reach a counted row
                d.tok.resize(std::max<int64_t>(1, prev_end));
                d.counted.resize(d.tok.size());
            }
        } else {
            d.counted.assign(d.tok.size(), 1);
        }
        finish_doc(d, stride);
        docs.push_back(std::move(d));
    }
    return docs;
}

// upstream llama-imatrix's chunks: consecutive n-token pieces of one text, the first token of each replaced by
// BOS where the vocab adds one
static std::vector<document> text_chunks(const std::string & path, const llama_vocab * vocab, int n, int max_chunks, int stride) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fatal("cannot open %s", path.c_str());
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<llama_token> all = common_tokenize(vocab, text, true, true);
    std::vector<document> docs;
    for (size_t c = 0; (c + 1)*n <= all.size() && (max_chunks <= 0 || (int) docs.size() < max_chunks); ++c) {
        document d;
        d.id = "chunk" + std::to_string(c);
        d.tok.assign(all.begin() + c*n, all.begin() + (c + 1)*n);
        if (llama_vocab_get_add_bos(vocab)) {
            d.tok[0] = llama_vocab_bos(vocab);
        }
        d.counted.assign(n, 1);
        finish_doc(d, stride);
        docs.push_back(std::move(d));
    }
    return docs;
}

// documents that fit a ubatch are packed k at a time with similar lengths, each padded to the longest of its
// batch: the hybrid memory cuts a batch into ubatches with equal tokens per sequence, and padding after a
// document leaves its rows unchanged. Longer documents run alone, a ubatch at a time.
static std::vector<batch_plan> plan_batches(const std::vector<document> & docs, int64_t n_ubatch, int max_seqs) {
    std::vector<int> order;
    std::vector<batch_plan> out;
    for (int i = 0; i < (int) docs.size(); ++i) {
        if ((int64_t) docs[i].tok.size() > n_ubatch) {
            out.push_back({ { i }, 0 });
        } else {
            order.push_back(i);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (docs[a].tok.size() != docs[b].tok.size()) {
            return docs[a].tok.size() > docs[b].tok.size();
        }
        return docs[a].id < docs[b].id;
    });
    for (size_t i = 0; i < order.size();) {
        batch_plan b{ {}, (int64_t) docs[order[i]].tok.size() };
        while (i < order.size() && (int) b.docs.size() < max_seqs && (int64_t) (b.docs.size() + 1)*b.len <= n_ubatch) {
            b.docs.push_back(order[i++]);
        }
        out.push_back(std::move(b));
    }
    return out;
}

static void decode_or_die(llama_context * ctx, llama_batch & b) {
    const int rc = llama_decode(ctx, b);
    if (rc != 0) {
        fatal("llama_decode failed (%d)", rc);
    }
}

static int64_t run_batch(llama_context * ctx, collector & col, const std::vector<document> & docs, const batch_plan & bp,
                         int64_t n_ubatch, llama_token pad) {
    llama_memory_clear(llama_get_memory(ctx), true);
    col.seq_doc = bp.docs;
    int64_t n_tok = 0;
    if (bp.len == 0) {
        const document & d = docs[bp.docs[0]];
        const int64_t L = d.tok.size();
        llama_batch b = llama_batch_init((int32_t) n_ubatch, 0, 1);
        for (int64_t c0 = 0; c0 < L; c0 += n_ubatch) {
            const int64_t c1 = std::min(L, c0 + n_ubatch);
            b.n_tokens = (int32_t) (c1 - c0);
            bool any = false;
            for (int64_t i = c0; i < c1; ++i) {
                const int k = (int) (i - c0);
                b.token[k] = d.tok[i];
                b.pos[k] = (llama_pos) i;
                b.n_seq_id[k] = 1;
                b.seq_id[k][0] = 0;
                b.logits[k] = d.output[i];
                any |= d.output[i] != 0;
            }
            if (!any) {
                b.logits[b.n_tokens - 1] = 1;
            }
            decode_or_die(ctx, b);
            n_tok += c1 - c0;
        }
        llama_batch_free(b);
        return n_tok;
    }
    const int64_t k = bp.docs.size();
    llama_batch b = llama_batch_init((int32_t) (k*bp.len), 0, 1);
    b.n_tokens = (int32_t) (k*bp.len);
    bool any = false;
    for (int64_t s = 0; s < k; ++s) {
        const document & d = docs[bp.docs[s]];
        for (int64_t i = 0; i < bp.len; ++i) {
            const int64_t j = s*bp.len + i;
            const bool real = i < (int64_t) d.tok.size();
            b.token[j] = real ? d.tok[i] : pad;
            b.pos[j] = (llama_pos) i;
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = (llama_seq_id) s;
            b.logits[j] = real && d.output[i];
            any |= b.logits[j] != 0;
        }
    }
    if (!any) {
        b.logits[b.n_tokens - 1] = 1;
    }
    decode_or_die(ctx, b);
    llama_batch_free(b);
    return k*bp.len;
}

// ---------------------------------------------------------------- file layout

struct layout {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    size_t         meta_size = 0;
    size_t         total_size = 0;
    std::map<std::string, tensor_slot> slots;
    std::vector<uint8_t> layers_done;
};

static void set_header(layout & L, const std::string & dataset, int n_docs, int chunk_size, const std::string & model_name,
                       int stride, const std::vector<std::string> & alias_names, const std::vector<std::string> & alias_owners, bool complete) {
    gguf_context * g = L.gguf;
    gguf_set_val_str(g, "general.type", "imatrix");
    const char * ds = dataset.c_str();
    gguf_set_arr_str(g, "imatrix.datasets", &ds, 1);
    gguf_set_val_u32(g, "imatrix.chunk_count", (uint32_t) n_docs);
    gguf_set_val_u32(g, "imatrix.chunk_size", (uint32_t) chunk_size);
    gguf_set_val_u32(g, "hessian.version", 1);
    gguf_set_val_str(g, "hessian.source_model", model_name.c_str());
    gguf_set_val_u32(g, "hessian.output_stride", (uint32_t) stride);
    std::vector<const char *> an, ao;
    for (size_t i = 0; i < alias_names.size(); ++i) {
        an.push_back(alias_names[i].c_str());
        ao.push_back(alias_owners[i].c_str());
    }
    gguf_set_arr_str(g, "hessian.alias.names", an.data(), an.size());
    gguf_set_arr_str(g, "hessian.alias.owners", ao.data(), ao.size());
    gguf_set_arr_data(g, "hessian.layers_done", GGUF_TYPE_UINT8, L.layers_done.data(), L.layers_done.size());
    gguf_set_val_bool(g, "hessian.complete", complete);
}

static std::vector<uint8_t> header_bytes(const layout & L) {
    std::vector<uint8_t> buf(gguf_get_meta_size(L.gguf));
    gguf_get_meta_data(L.gguf, buf.data());
    return buf;
}

static void pwrite_all(int fd, const void * data, size_t size, size_t off) {
    const uint8_t * p = (const uint8_t *) data;
    while (size > 0) {
        const ssize_t w = pwrite(fd, p, std::min(size, (size_t) 1 << 30), (off_t) off);
        if (w <= 0) {
            fatal("write failed at offset %zu: %s", off, strerror(errno));
        }
        p += w; off += w; size -= w;
    }
}

static void pread_all(int fd, void * data, size_t size, size_t off) {
    uint8_t * p = (uint8_t *) data;
    while (size > 0) {
        const ssize_t r = pread(fd, p, std::min(size, (size_t) 1 << 30), (off_t) off);
        if (r <= 0) {
            fatal("read failed at offset %zu: %s", off, strerror(errno));
        }
        p += r; off += r; size -= r;
    }
}

// ---------------------------------------------------------------- main

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m MODEL.gguf (--docs DOCS.jsonl | --text FILE --chunk N [--chunks K]) -o OUT.gguf\n"
        "    [-ngl N] [-ot REGEX=DEVICE] [--no-mmap] [--ubatch 8192] [--ctx N] [-ctk TYPE] [-ctv TYPE]\n"
        "    [--max-seqs 8] [--output-stride 8] [--group-layers N|auto] [--layers 0,31,63,out|A-B]\n"
        "    [--resume] [--imatrix-out FILE] [--dataset NAME] [--accum cpu|cuda] [--threads N] [--stop-after-pass N]\n"
        "    [--no-op-offload] [--trim-docs] [--dry-run] [--test-hook]\n", argv0);
}

static ggml_type parse_cache_type(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * name = ggml_type_name((ggml_type) t);
        if (name && s == name) {
            return (ggml_type) t;
        }
    }
    fatal("unknown cache type %s", s.c_str());
}

static std::set<int> parse_layers(const std::string & spec, int n_layer) {
    std::set<int> out;
    size_t p = 0;
    while (p <= spec.size()) {
        const size_t q = std::min(spec.find(',', p), spec.size());
        const std::string item = spec.substr(p, q - p);
        if (item == "out" || item == "output") {
            out.insert(n_layer);
        } else if (!item.empty()) {
            const size_t d = item.find('-');
            const int a = std::stoi(item.substr(0, d));
            const int b = d == std::string::npos ? a : std::stoi(item.substr(d + 1));
            for (int l = a; l <= b; ++l) {
                if (l < 0 || l > n_layer) {
                    fatal("--layers: %d is not a layer of this model (0-%d, out)", l, n_layer - 1);
                }
                out.insert(l);
            }
        }
        p = q + 1;
    }
    return out;
}

static const ggml_tensor * view_base(const ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

int main(int argc, char ** argv) {
    std::string model_path, docs_path, text_path, out_path, imatrix_out, dataset, accum, layers_spec, group_spec = "auto";
    int ngl = 0, max_seqs = 8, stride = 8, chunk = 512, max_chunks = 0, stop_after = 0;
    int64_t n_ubatch = 8192, n_ctx = 0;
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    bool use_mmap = true, resume = false, dry_run = false, test_hook = false, op_offload = true, trim_docs = false;
    ggml_type type_k = GGML_TYPE_F16, type_v = GGML_TYPE_F16;
    std::vector<std::string> overrides;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage(argv[0]);
                fatal("%s needs a value", a.c_str());
            }
            return argv[++i];
        };
        if (a == "-m") model_path = next();
        else if (a == "--docs") docs_path = next();
        else if (a == "--text") text_path = next();
        else if (a == "--chunk") chunk = std::stoi(next());
        else if (a == "--chunks") max_chunks = std::stoi(next());
        else if (a == "-o") out_path = next();
        else if (a == "-ngl") ngl = std::stoi(next());
        else if (a == "-ot") overrides.push_back(next());
        else if (a == "--no-mmap") use_mmap = false;
        else if (a == "--ubatch") n_ubatch = std::stoll(next());
        else if (a == "--ctx") n_ctx = std::stoll(next());
        else if (a == "-ctk") type_k = parse_cache_type(next());
        else if (a == "-ctv") type_v = parse_cache_type(next());
        else if (a == "--max-seqs") max_seqs = std::stoi(next());
        else if (a == "--output-stride") stride = std::stoi(next());
        else if (a == "--group-layers") group_spec = next();
        else if (a == "--layers") layers_spec = next();
        else if (a == "--resume") resume = true;
        else if (a == "--imatrix-out") imatrix_out = next();
        else if (a == "--dataset") dataset = next();
        else if (a == "--accum") accum = next();
        else if (a == "--threads") n_threads = std::stoi(next());
        else if (a == "--stop-after-pass") stop_after = std::stoi(next());
        else if (a == "--dry-run") dry_run = true;
        else if (a == "--test-hook") test_hook = true;
        else if (a == "--no-op-offload") op_offload = false;
        else if (a == "--trim-docs") trim_docs = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { usage(argv[0]); fatal("unknown argument %s", a.c_str()); }
    }
    if (model_path.empty() || out_path.empty() || docs_path.empty() == text_path.empty() || stride < 1) {
        usage(argv[0]);
        return 1;
    }
#ifdef GGML_USE_CUDA
    const bool use_cuda = accum != "cpu";
#else
    if (accum == "cuda") {
        fatal("--accum cuda needs the CUDA build (hessian-collect-cuda)");
    }
    const bool use_cuda = false;
#endif
    if (dataset.empty()) {
        const std::string & p = docs_path.empty() ? text_path : docs_path;
        dataset = p.substr(p.find_last_of('/') + 1);
    }

#ifdef GGML_USE_CUDA
    size_t free_at_start = 0;
    if (use_cuda) {
        size_t total_b;
        CUDA_CHECK(cudaMemGetInfo(&free_at_start, &total_b));
    }
#endif
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) {
            fputs(text, stderr);
        }
    }, nullptr);
    llama_backend_init();

    std::vector<std::string> ot_patterns;
    std::vector<llama_model_tensor_buft_override> ot;
    ot_patterns.reserve(overrides.size());
    for (const std::string & o : overrides) {
        const size_t eq = o.find('=');
        ggml_backend_buffer_type_t buft = nullptr;
        for (size_t i = 0; eq != std::string::npos && i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            for (ggml_backend_buffer_type_t b : { ggml_backend_dev_buffer_type(dev), ggml_backend_dev_host_buffer_type(dev) }) {
                if (b && o.compare(eq + 1, std::string::npos, ggml_backend_buft_name(b)) == 0) {
                    buft = b;
                }
            }
        }
        if (!buft) {
            fatal("-ot %s: needs REGEX=DEVICE with a device's buffer type (CUDA0, CPU, CUDA_Host for pinned host memory, ...)", o.c_str());
        }
        ot_patterns.push_back(o.substr(0, eq));
        ot.push_back({ ot_patterns.back().c_str(), buft });
    }
    ot.push_back({ nullptr, nullptr });

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    if (!use_mmap) {
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
    }
    if (!overrides.empty()) {
        mp.tensor_buft_overrides = ot.data();
    }
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fatal("failed to load %s", model_path.c_str());
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer = llama_model_n_layer(model);

    const double t_tok0 = now_s();
    std::vector<document> docs = docs_path.empty() ? text_chunks(text_path, vocab, chunk, max_chunks, stride) : load_docs(docs_path, vocab, stride, trim_docs);
    if (docs.empty()) {
        fatal("no documents");
    }
    int64_t max_len = 0, n_proc = 0, n_counted = 0, n_outputs = 0;
    for (const auto & d : docs) {
        max_len = std::max<int64_t>(max_len, d.tok.size());
        n_proc += d.tok.size();
        n_counted += d.n_counted;
        n_outputs += std::count(d.output.begin(), d.output.end(), 1);
    }
    if (n_ctx == 0) {
        n_ctx = std::max(n_ubatch, (max_len + 255)/256*256);
    }
    if (max_len > n_ctx) {
        fatal("the longest document has %" PRId64 " tokens, more than --ctx %" PRId64, max_len, n_ctx);
    }
    fprintf(stderr, "%zu documents: %" PRId64 " tokens, %" PRId64 " counted, %" PRId64 " LM-head rows, longest %" PRId64 " (tokenized in %.1f s)\n",
            docs.size(), n_proc, n_counted, n_outputs, max_len, now_s() - t_tok0);
    const std::vector<batch_plan> batches = plan_batches(docs, n_ubatch, max_seqs);
    int64_t n_padded = 0;
    for (const auto & b : batches) {
        n_padded += b.len ? b.len*b.docs.size() : docs[b.docs[0]].tok.size();
    }
    fprintf(stderr, "%zu batches, %" PRId64 " tokens with padding\n", batches.size(), n_padded);

    const llama_token pad = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : 0;
    collector col;
    col.docs = &docs;
    col.pad = pad;
    col.n_layer = n_layer;
    col.use_cuda = use_cuda;
    col.n_threads = n_threads;
#ifdef GGML_USE_CUDA
    col.free_at_start = free_at_start;
#endif

    if (test_hook) {
        const document & d0 = docs[0];
        const int n = (int) std::min<size_t>(256, d0.tok.size());
        const int n_vocab = llama_vocab_n_tokens(vocab);
        llama_context_params hp = llama_context_default_params();
        hp.n_ctx = hp.n_batch = hp.n_ubatch = std::max(256, n);
        hp.n_threads = hp.n_threads_batch = n_threads;
        llama_context * hc = llama_init_from_model(model, hp);
        auto logits = [&](bool rec) {
            hc->set_collect_mm_inputs(rec);
            llama_memory_clear(llama_get_memory(hc), true);
            llama_batch b = llama_batch_init(n, 0, 1);
            b.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                b.token[i] = d0.tok[i];
                b.pos[i] = i;
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = 1;
            }
            decode_or_die(hc, b);
            llama_batch_free(b);
            const float * l = llama_get_logits(hc);
            return std::vector<float>(l, l + (size_t) n*n_vocab);
        };
        const std::vector<float> off = logits(false), on = logits(true);
        const size_t n_rec = hc->get_mm_inputs().inputs->size();
        const std::vector<float> off2 = logits(false);
        const bool same = memcmp(off.data(), on.data(), off.size()*sizeof(float)) == 0;
        const bool same2 = memcmp(off.data(), off2.data(), off.size()*sizeof(float)) == 0;
        printf("test-hook: %d tokens, %zu GEMM inputs recorded, logits with the flag on %s, flag off again %s\n",
               n, n_rec, same ? "bitwise identical" : "DIFFERENT", same2 ? "bitwise identical" : "DIFFERENT");
        llama_free(hc);
        llama_model_free(model);
        return same && same2 && n_rec > 0 ? 0 : 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) n_ctx;
    cp.n_batch         = (uint32_t) n_ubatch;
    cp.n_ubatch        = (uint32_t) n_ubatch;
    cp.n_seq_max       = (uint32_t) max_seqs;
    cp.n_outputs_max   = (uint32_t) std::min<int64_t>(n_ubatch, (n_ubatch + stride - 1)/stride + max_seqs);
    cp.kv_unified      = true;
    cp.type_k          = type_k;
    cp.type_v          = type_v;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.op_offload      = op_offload;
    cp.cb_eval           = eval_cb;
    cp.cb_eval_user_data = &col;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fatal("failed to create the context");
    }
    col.ctx = ctx;
    ctx->set_collect_mm_inputs(true);

    // one small decode finds the weight GEMMs, their inputs and the groups sharing an input
    std::vector<std::string> file_order;
    {
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * mg = gguf_init_from_file(model_path.c_str(), gp);
        if (!mg) {
            fatal("cannot read %s", model_path.c_str());
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(mg); ++i) {
            file_order.push_back(gguf_get_tensor_name(mg, i));
        }
        gguf_free(mg);
    }
    std::unordered_map<std::string, int> file_rank;
    for (size_t i = 0; i < file_order.size(); ++i) {
        file_rank[file_order[i]] = (int) i;
    }
    {
        const document & d0 = docs[0];
        const int n = (int) std::min<size_t>(64, d0.tok.size());
        llama_batch b = llama_batch_init(n, 0, 1);
        b.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            b.token[i] = d0.tok[i];
            b.pos[i] = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            // one output reaches the LM head; the context holds only about n_ubatch/stride of them
            b.logits[i] = i == n - 1;
        }
        col.seq_doc = { 0 };
        decode_or_die(ctx, b);
        llama_batch_free(b);
        llama_memory_clear(llama_get_memory(ctx), true);

        const auto st = ctx->get_mm_inputs();
        ggml_cgraph * gf = st.gf;
        std::set<const ggml_tensor *> nodes;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            nodes.insert(ggml_graph_node(gf, i));
        }
        std::map<const ggml_tensor *, std::vector<std::string>> by_x;
        std::set<std::string> recorded;
        for (const auto & r : *st.inputs) {
            if (!file_rank.count(r.w->name)) {
                continue;
            }
            if (r.id) {
                fatal("%s is a MUL_MAT_ID weight: expert (MoE) Grams are not supported", r.w->name);
            }
            if (!nodes.count(r.x)) {
                fatal("the input of %s is not a computed node of the graph (a leaf or an elided node) - it can't be observed", r.w->name);
            }
            if (r.x->type != GGML_TYPE_F32 || r.x->nb[0] != sizeof(float)) {
                fatal("the input of %s is %s with nb[0] = %zu, not F32 rows", r.w->name, ggml_type_name(r.x->type), r.x->nb[0]);
            }
            if (recorded.count(r.w->name)) {
                fatal("%s is multiplied with more than one input", r.w->name);
            }
            recorded.insert(r.w->name);
            by_x[r.x].push_back(r.w->name);
        }
        // coverage: every weight a GEMM of this graph reads must have gone through the hook
        std::set<std::string> missing;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            const ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            const ggml_tensor * w = view_base(node->src[0]);
            if (w && file_rank.count(w->name) && !recorded.count(w->name)) {
                missing.insert(w->name);
            }
        }
        if (!missing.empty()) {
            for (const auto & m : missing) {
                fprintf(stderr, "not recorded: %s (multiplied outside build_mm / build_lora_mm)\n", m.c_str());
            }
            fatal("%zu weights are multiplied without going through the GEMM-input hook", missing.size());
        }
        for (auto & [x, ws] : by_x) {
            std::sort(ws.begin(), ws.end(), [&](const std::string & a, const std::string & b) { return file_rank[a] < file_rank[b]; });
            group g;
            g.owner = ws[0];
            g.weights = ws;
            g.n = x->ne[0];
            char lay[256];
            snprintf(lay, sizeof(lay), "%s %s ne [%" PRId64 ", %" PRId64 ", %" PRId64 "] nb [%zu, %zu, %zu] %s rows",
                     ggml_op_desc(x), ggml_type_name(x->type), x->ne[0], x->ne[1], x->ne[2], x->nb[0], x->nb[1], x->nb[2],
                     collector::uniform_rows(x) ? "uniform" : "non-uniform");
            g.layout = lay;
            int il = -1;
            if (sscanf(g.owner.c_str(), "blk.%d.", &il) != 1) {
                il = n_layer;
            } else if (il >= n_layer) {
                continue;
            }
            g.layer = il;
            for (const auto & w : ws) {
                if (sscanf(w.c_str(), "blk.%d.", &il) == 1 ? il != g.layer : g.layer != n_layer) {
                    fatal("%s and %s share an input across layers", g.owner.c_str(), w.c_str());
                }
            }
            col.groups.push_back(std::move(g));
        }
        std::sort(col.groups.begin(), col.groups.end(), [&](const group & a, const group & b) {
            return a.layer != b.layer ? a.layer < b.layer : a.owner < b.owner;
        });
        for (int gi = 0; gi < (int) col.groups.size(); ++gi) {
            for (const auto & w : col.groups[gi].weights) {
                col.group_of_weight[w] = gi;
            }
        }
        int n_w = 0;
        for (const auto & g : col.groups) {
            n_w += (int) g.weights.size();
        }
        std::set<int> have_layers;
        for (const auto & g : col.groups) {
            have_layers.insert(g.layer);
        }
        fprintf(stderr, "%d weights in %zu input groups over %zu layers (incl. the LM head: %s)\n", n_w, col.groups.size(),
                have_layers.size(), have_layers.count(n_layer) ? "yes" : "no");
        col.graph_id = UINT64_MAX;
        if (dry_run) {
            for (const auto & g : col.groups) {
                printf("layer %2d  n %5" PRId64 "  %-28s", g.layer, g.n, g.layout.c_str());
                for (const auto & w : g.weights) {
                    printf(" %s", w.c_str());
                }
                printf("\n");
            }
            printf("%d weights in %zu groups\n", n_w, col.groups.size());
            llama_free(ctx);
            llama_model_free(model);
            return 0;
        }
    }

    const std::set<int> sel = layers_spec.empty() ? [&] { std::set<int> s; for (const auto & g : col.groups) s.insert(g.layer); return s; }()
                                                   : parse_layers(layers_spec, n_layer);
    for (auto & g : col.groups) {
        g.selected = sel.count(g.layer) > 0;
    }

    // file layout: tensors ordered by (layer, weight name), so a pass writes one contiguous range
    layout L;
    L.layers_done.assign(n_layer + 1, 0);
    std::vector<std::string> alias_names, alias_owners;
    std::vector<std::pair<int, std::string>> order;
    for (const auto & g : col.groups) {
        if (!g.selected) {
            continue;
        }
        for (const auto & w : g.weights) {
            order.emplace_back(g.layer, w);
        }
    }
    std::sort(order.begin(), order.end());
    for (const auto & [il, w] : order) {
        const std::string & owner = col.groups[col.group_of_weight[w]].owner;
        if (w != owner) {
            alias_names.push_back(w);
            alias_owners.push_back(owner);
        }
    }
    {
        L.gguf = gguf_init_empty();
        ggml_init_params ip = { /*.mem_size =*/ ggml_tensor_overhead()*(order.size()*3 + 8), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        L.meta = ggml_init(ip);
        for (const auto & [il, w] : order) {
            const group & g = col.groups[col.group_of_weight[w]];
            ggml_tensor * s2 = ggml_new_tensor_2d(L.meta, GGML_TYPE_F32, g.n, 1);
            ggml_format_name(s2, "%s.in_sum2", w.c_str());
            ggml_tensor * ct = ggml_new_tensor_2d(L.meta, GGML_TYPE_F32, 1, 1);
            ggml_format_name(ct, "%s.counts", w.c_str());
            gguf_add_tensor(L.gguf, s2);
            gguf_add_tensor(L.gguf, ct);
            if (w == g.owner) {
                ggml_tensor * gr = ggml_new_tensor_2d(L.meta, GGML_TYPE_F32, g.n, g.n);
                ggml_format_name(gr, "%s.in_gram", w.c_str());
                gguf_add_tensor(L.gguf, gr);
            }
        }
        const std::string model_name = model_path.substr(model_path.find_last_of('/') + 1);
        set_header(L, dataset, (int) docs.size(), (int) max_len, model_name, stride, alias_names, alias_owners, false);
        L.meta_size = gguf_get_meta_size(L.gguf);
        for (int64_t i = 0; i < gguf_get_n_tensors(L.gguf); ++i) {
            tensor_slot s;
            s.name = gguf_get_tensor_name(L.gguf, i);
            s.offset = L.meta_size + gguf_get_tensor_offset(L.gguf, i);
            const ggml_tensor * t = ggml_get_tensor(L.meta, s.name.c_str());
            s.ne0 = t->ne[0];
            s.ne1 = t->ne[1];
            L.total_size = std::max(L.total_size, s.offset + ggml_nbytes(t));
            L.slots[s.name] = s;
        }
    }

    int fd;
    if (resume) {
        fd = open(out_path.c_str(), O_RDWR);
        if (fd < 0) {
            fatal("--resume: cannot open %s", out_path.c_str());
        }
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * og = gguf_init_from_file(out_path.c_str(), gp);
        if (!og) {
            fatal("--resume: %s is not a GGUF file", out_path.c_str());
        }
        const int64_t k = gguf_find_key(og, "hessian.layers_done");
        if (k < 0 || gguf_get_arr_n(og, k) != L.layers_done.size()) {
            fatal("--resume: %s has no matching hessian.layers_done", out_path.c_str());
        }
        memcpy(L.layers_done.data(), gguf_get_arr_data(og, k), L.layers_done.size());
        const int64_t kc = gguf_find_key(og, "hessian.complete");
        const bool was_complete = kc >= 0 && gguf_get_val_bool(og, kc);
        gguf_free(og);
        gguf_set_arr_data(L.gguf, "hessian.layers_done", GGUF_TYPE_UINT8, L.layers_done.data(), L.layers_done.size());
        gguf_set_val_bool(L.gguf, "hessian.complete", was_complete);
        const std::vector<uint8_t> want = header_bytes(L);
        std::vector<uint8_t> have(want.size());
        pread_all(fd, have.data(), have.size(), 0);
        if (have != want) {
            fatal("--resume: the header of %s doesn't match this model, these documents and these options", out_path.c_str());
        }
        int n_done = 0;
        for (auto v : L.layers_done) {
            n_done += v;
        }
        fprintf(stderr, "resuming: %d of %zu layer slots done\n", n_done, L.layers_done.size());
    } else {
        fd = open(out_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            fatal("cannot create %s", out_path.c_str());
        }
        if (posix_fallocate(fd, 0, (off_t) L.total_size) != 0 && ftruncate(fd, (off_t) L.total_size) != 0) {
            fatal("cannot allocate %zu bytes for %s", L.total_size, out_path.c_str());
        }
        const std::vector<uint8_t> h = header_bytes(L);
        pwrite_all(fd, h.data(), h.size(), 0);
    }
    fprintf(stderr, "output %s: %.2f GB\n", out_path.c_str(), L.total_size/1e9);

    // passes over the layers still to do
    std::vector<int> todo;
    std::map<int, size_t> layer_bytes;
    for (const auto & g : col.groups) {
        if (g.selected && !L.layers_done[g.layer]) {
            if (todo.empty() || todo.back() != g.layer) {
                todo.push_back(g.layer);
            }
            layer_bytes[g.layer] += (size_t) g.n*g.n*(use_cuda ? sizeof(float) : sizeof(double));
        }
    }
    std::sort(todo.begin(), todo.end());
    todo.erase(std::unique(todo.begin(), todo.end()), todo.end());

#ifdef GGML_USE_CUDA
    if (use_cuda) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&col.stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasCreate(&col.cublas));
        CUBLAS_CHECK(cublasSetStream(col.cublas, col.stream));
        CUBLAS_CHECK(cublasSetMathMode(col.cublas, CUBLAS_PEDANTIC_MATH));
    }
#endif
    int group_layers;
    if (group_spec == "auto") {
        size_t budget;
        size_t max_layer = 0;
        for (auto & [l, b] : layer_bytes) {
            if (l < n_layer) {
                max_layer = std::max(max_layer, b);
            }
        }
#ifdef GGML_USE_CUDA
        if (use_cuda) {
            size_t free_b, total_b;
            CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            const size_t margin = (size_t) 1 << 30;
            budget = free_b > margin ? free_b - margin : 0;
            fprintf(stderr, "VRAM: %.2f GB free after the model and context\n", free_b/1e9);
        } else
#endif
        {
            budget = (size_t) sysconf(_SC_AVPHYS_PAGES)*sysconf(_SC_PAGE_SIZE)/2;
        }
        group_layers = max_layer ? (int) std::max<size_t>(1, budget/max_layer) : 1;
    } else {
        group_layers = std::stoi(group_spec);
    }
    std::vector<std::vector<int>> passes;
    {
        std::vector<int> blocks;
        bool out = false;
        for (int l : todo) {
            if (l == n_layer) {
                out = true;
            } else {
                blocks.push_back(l);
            }
        }
        for (size_t i = 0; i < blocks.size(); i += group_layers) {
            passes.emplace_back(blocks.begin() + i, blocks.begin() + std::min(blocks.size(), i + group_layers));
        }
        if (out) {
            if (passes.empty()) {
                passes.emplace_back();
            }
            passes.back().push_back(n_layer);
        }
    }
    fprintf(stderr, "%zu layer slots to do in %zu passes of up to %d layers\n", todo.size(), passes.size(), group_layers);

    std::vector<float> gbuf;
    int n_pass_done = 0;
    for (const auto & pass : passes) {
        const std::set<int> in_pass(pass.begin(), pass.end());
        for (auto & g : col.groups) {
            g.active = g.selected && in_pass.count(g.layer) > 0;
            g.rows = 0;
            if (!g.active) {
                continue;
            }
#ifdef GGML_USE_CUDA
            if (use_cuda) {
                CUDA_CHECK(cudaMalloc(&g.acc32, (size_t) g.n*g.n*sizeof(float)));
                CUDA_CHECK(cudaMemset(g.acc32, 0, (size_t) g.n*g.n*sizeof(float)));
                continue;
            }
#endif
            g.acc64.assign((size_t) g.n*g.n, 0.0);
        }
#ifdef GGML_USE_CUDA
        if (use_cuda) {
            size_t free_b, total_b;
            CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            col.min_free = std::min(col.min_free, free_b);
        }
#endif
        const double t0 = now_s();
        col.t_accum = 0;
        int64_t n_tok = 0;
        for (size_t bi = 0; bi < batches.size(); ++bi) {
            n_tok += run_batch(ctx, col, docs, batches[bi], n_ubatch, pad);
            if (isatty(2) || bi + 1 == batches.size() || bi % 50 == 49) {
                const double el = now_s() - t0;
                fprintf(stderr, "\rpass %d/%zu: batch %zu/%zu, %.0f tok/s, accumulation %.1f s ", n_pass_done + 1, passes.size(),
                        bi + 1, batches.size(), n_tok/el, col.t_accum);
                if (!isatty(2)) {
                    fputc('\n', stderr);
                }
            }
        }
#ifdef GGML_USE_CUDA
        if (use_cuda) {
            size_t free_b, total_b;
            CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            col.min_free = std::min(col.min_free, free_b);
        }
#endif
        const double t_run = now_s() - t0;
        const double tw0 = now_s();
        for (auto & g : col.groups) {
            if (!g.active) {
                continue;
            }
            const int64_t n = g.n;
            gbuf.resize((size_t) n*n);
#ifdef GGML_USE_CUDA
            if (use_cuda) {
                CUDA_CHECK(cudaMemcpy(gbuf.data(), g.acc32, gbuf.size()*sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaFree(g.acc32));
                g.acc32 = nullptr;
                // column-major upper = row-major lower: mirror it into the upper half
                const int64_t T = 64;
                for (int64_t i0 = 0; i0 < n; i0 += T) {
                    for (int64_t j0 = i0; j0 < n; j0 += T) {
                        for (int64_t i = i0; i < std::min(n, i0 + T); ++i) {
                            for (int64_t j = std::max(j0, i + 1); j < std::min(n, j0 + T); ++j) {
                                gbuf[i*n + j] = gbuf[j*n + i];
                            }
                        }
                    }
                }
            } else
#endif
            {
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t j = i; j < n; ++j) {
                        gbuf[i*n + j] = gbuf[j*n + i] = (float) g.acc64[(size_t) i*n + j];
                    }
                }
                std::vector<double>().swap(g.acc64);
            }
            std::vector<float> diag(n);
            for (int64_t i = 0; i < n; ++i) {
                diag[i] = gbuf[i*n + i];
                if (!(diag[i] >= 0.0f)) {
                    fatal("%s: diagonal element %" PRId64 " is %g", g.owner.c_str(), i, diag[i]);
                }
            }
            for (float v : gbuf) {
                if (!std::isfinite(v)) {
                    fatal("%s: non-finite Gram entry", g.owner.c_str());
                }
            }
            pwrite_all(fd, gbuf.data(), gbuf.size()*sizeof(float), L.slots.at(g.owner + ".in_gram").offset);
            const float cnt = (float) g.rows;
            for (const auto & w : g.weights) {
                pwrite_all(fd, diag.data(), diag.size()*sizeof(float), L.slots.at(w + ".in_sum2").offset);
                pwrite_all(fd, &cnt, sizeof(cnt), L.slots.at(w + ".counts").offset);
            }
            g.active = false;
        }
        if (fdatasync(fd) != 0) {
            fatal("fdatasync failed: %s", strerror(errno));
        }
        for (int l : pass) {
            L.layers_done[l] = 1;
        }
        bool complete = true;
        for (const auto & g : col.groups) {
            complete &= !g.selected || L.layers_done[g.layer];
        }
        gguf_set_arr_data(L.gguf, "hessian.layers_done", GGUF_TYPE_UINT8, L.layers_done.data(), L.layers_done.size());
        gguf_set_val_bool(L.gguf, "hessian.complete", complete);
        const std::vector<uint8_t> h = header_bytes(L);
        if (h.size() != L.meta_size) {
            fatal("header size changed (%zu -> %zu)", L.meta_size, h.size());
        }
        pwrite_all(fd, h.data(), h.size(), 0);
        fdatasync(fd);
        std::string ls;
        for (int l : pass) {
            ls += (ls.empty() ? "" : ",") + (l == n_layer ? std::string("out") : std::to_string(l));
        }
        fprintf(stderr, "pass %d/%zu [layers %s]: %.1f s, %.0f tok/s, accumulation %.1f s, write %.1f s\n", n_pass_done + 1, passes.size(),
                ls.c_str(), t_run, n_tok/t_run, col.t_accum, now_s() - tw0);
        ++n_pass_done;
        if (stop_after > 0 && n_pass_done >= stop_after && n_pass_done < (int) passes.size()) {
            fprintf(stderr, "stopping after %d passes (--stop-after-pass)\n", n_pass_done);
            close(fd);
            return 0;
        }
    }

    bool complete = true;
    for (const auto & g : col.groups) {
        complete &= !g.selected || L.layers_done[g.layer];
    }
    if (complete && !imatrix_out.empty()) {
        gguf_context * ig = gguf_init_empty();
        gguf_set_val_str(ig, "general.type", "imatrix");
        const char * ds = dataset.c_str();
        gguf_set_arr_str(ig, "imatrix.datasets", &ds, 1);
        gguf_set_val_u32(ig, "imatrix.chunk_count", (uint32_t) docs.size());
        gguf_set_val_u32(ig, "imatrix.chunk_size", (uint32_t) max_len);
        size_t mem = ggml_tensor_overhead()*(order.size()*2 + 8);
        for (const auto & [il, w] : order) {
            mem += GGML_PAD(col.groups[col.group_of_weight[w]].n*sizeof(float), GGML_MEM_ALIGN) + GGML_PAD(sizeof(float), GGML_MEM_ALIGN);
        }
        ggml_init_params ip = { mem, nullptr, false };
        ggml_context * ictx = ggml_init(ip);
        for (const auto & [il, w] : order) {
            const int64_t n = col.groups[col.group_of_weight[w]].n;
            ggml_tensor * s2 = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, n, 1);
            ggml_format_name(s2, "%s.in_sum2", w.c_str());
            ggml_tensor * ct = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, 1, 1);
            ggml_format_name(ct, "%s.counts", w.c_str());
            pread_all(fd, s2->data, ggml_nbytes(s2), L.slots.at(w + ".in_sum2").offset);
            pread_all(fd, ct->data, ggml_nbytes(ct), L.slots.at(w + ".counts").offset);
            gguf_add_tensor(ig, s2);
            gguf_add_tensor(ig, ct);
        }
        if (!gguf_write_to_file(ig, imatrix_out.c_str(), false)) {
            fatal("cannot write %s", imatrix_out.c_str());
        }
        gguf_free(ig);
        ggml_free(ictx);
        fprintf(stderr, "wrote %s\n", imatrix_out.c_str());
    }
    close(fd);

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    fprintf(stderr, "done%s: peak RSS %.2f GB", complete ? "" : " (incomplete)", ru.ru_maxrss/1e6);
#ifdef GGML_USE_CUDA
    if (use_cuda && col.min_free != SIZE_MAX) {
        size_t free_b, total_b;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        fprintf(stderr, ", peak VRAM used by this run %.2f GB (%.2f of %.2f GB free at its start)", (col.free_at_start - col.min_free)/1e9,
                col.free_at_start/1e9, total_b/1e9);
    }
#endif
    fputc('\n', stderr);

    gguf_free(L.gguf);
    ggml_free(L.meta);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
