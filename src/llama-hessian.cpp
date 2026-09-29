#include "llama-hessian.h"
#include "llama-mmap.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

static bool hessian_remove_suffix(std::string & s, const char * suffix) {
    const size_t n = strlen(suffix);
    if (s.size() < n || s.compare(s.size() - n, n, suffix) != 0) {
        return false;
    }
    s.resize(s.size() - n);
    return true;
}

llama_hessian::llama_hessian()  = default;
llama_hessian::~llama_hessian() = default;

// llama_file reads short past the end of a buffered file instead of failing, so the range is checked here
bool llama_hessian::read_at(void * dst, size_t size, size_t offset) const {
    if (!file || offset > file->size() || size > file->size() - offset) {
        return false;
    }
    try {
        file->seek(offset, SEEK_SET);
        file->read_raw(dst, size);
    } catch (const std::exception &) {
        return false;
    }
    return true;
}

bool llama_hessian::open(const std::string & path) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        fprintf(stderr, "%s: %s is not a GGUF file\n", __func__, path.c_str());
        return false;
    }
    const size_t data_off = gguf_get_data_offset(g);
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        std::string name = gguf_get_tensor_name(g, i);
        const size_t off = data_off + gguf_get_tensor_offset(g, i);
        const ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
        if (t->type != GGML_TYPE_F32) {
            continue;
        }
        if (hessian_remove_suffix(name, ".in_gram")) {
            if (t->ne[0] != t->ne[1]) {
                fprintf(stderr, "%s: %s.in_gram is not square\n", __func__, name.c_str());
                gguf_free(g);
                ggml_free(meta);
                return false;
            }
            grams[name] = { off, t->ne[0] };
            owners[name] = name;
        } else if (hessian_remove_suffix(name, ".in_sum2")) {
            sum2_off[name] = off;
        } else if (hessian_remove_suffix(name, ".counts")) {
            count_off[name] = off;
        }
    }
    const int64_t kn = gguf_find_key(g, "hessian.alias.names");
    const int64_t ko = gguf_find_key(g, "hessian.alias.owners");
    if ((kn < 0) != (ko < 0) || (kn >= 0 && gguf_get_arr_n(g, kn) != gguf_get_arr_n(g, ko))) {
        fprintf(stderr, "%s: %s has mismatched alias arrays\n", __func__, path.c_str());
        gguf_free(g);
        ggml_free(meta);
        return false;
    }
    for (int64_t i = 0; kn >= 0 && i < (int64_t) gguf_get_arr_n(g, kn); ++i) {
        const std::string w = gguf_get_arr_str(g, kn, i), o = gguf_get_arr_str(g, ko, i);
        if (!grams.count(o)) {
            fprintf(stderr, "%s: alias %s names %s, which has no Gram\n", __func__, w.c_str(), o.c_str());
            gguf_free(g);
            ggml_free(meta);
            return false;
        }
        owners[w] = o;
    }
    const int64_t kd = gguf_find_key(g, "imatrix.datasets");
    for (int64_t i = 0; kd >= 0 && i < (int64_t) gguf_get_arr_n(g, kd); ++i) {
        datasets.push_back(gguf_get_arr_str(g, kd, i));
    }
    const int64_t kc = gguf_find_key(g, "hessian.complete");
    complete = kc >= 0 && gguf_get_val_bool(g, kc);
    gguf_free(g);
    ggml_free(meta);

    try {
        file = std::make_unique<llama_file>(path.c_str(), "rb");
    } catch (const std::exception & e) {
        fprintf(stderr, "%s: %s\n", __func__, e.what());
        return false;
    }
    return true;
}

bool llama_hessian::has(const std::string & w) const {
    return owners.count(w) > 0;
}

int64_t llama_hessian::n(const std::string & w) const {
    auto it = owners.find(w);
    return it == owners.end() ? 0 : grams.at(it->second).n;
}

std::string llama_hessian::owner(const std::string & w) const {
    auto it = owners.find(w);
    return it == owners.end() ? std::string() : it->second;
}

double llama_hessian::count(const std::string & w) const {
    auto it = count_off.find(w);
    float c = 0.0f;
    if (it == count_off.end() || !read_at(&c, sizeof(c), it->second)) {
        return 0.0;
    }
    return c;
}

std::vector<std::string> llama_hessian::weights() const {
    std::vector<std::string> out;
    for (const auto & [w, o] : owners) {
        out.push_back(w);
    }
    return out;
}

bool llama_hessian::read(const std::string & w, float * dst, int64_t row0, int64_t n_rows) const {
    auto it = owners.find(w);
    if (it == owners.end()) {
        return false;
    }
    const gram_info & gi = grams.at(it->second);
    if (n_rows < 0) {
        n_rows = gi.n - row0;
    }
    if (row0 < 0 || row0 + n_rows > gi.n) {
        return false;
    }
    return read_at(dst, (size_t) n_rows*gi.n*sizeof(float), gi.offset + (size_t) row0*gi.n*sizeof(float));
}

bool llama_hessian::read_sum2(const std::string & w, float * dst) const {
    auto it = sum2_off.find(w);
    return it != sum2_off.end() && read_at(dst, (size_t) n(w)*sizeof(float), it->second);
}
