#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_file;

// A hessian file from hessian-collect: an imatrix GGUF in which each distinct GEMM input also carries its Gram
// G = sum x x^T (unrotated, a raw sum over .counts rows) as <owner>.in_gram F32 [n, n]; the other weights that
// read the same input name their owner in hessian.alias.names / hessian.alias.owners. Grams are read on demand,
// in row slabs, never the whole file.
class llama_hessian {
public:
    llama_hessian();
    llama_hessian(const llama_hessian &) = delete;
    llama_hessian & operator=(const llama_hessian &) = delete;
    ~llama_hessian();

    bool open(const std::string & fname);

    bool        has(const std::string & weight) const;   // has a Gram, its own or through an alias
    int64_t     n(const std::string & weight) const;     // input width, 0 without a Gram
    std::string owner(const std::string & weight) const; // the weight whose .in_gram holds it
    double      count(const std::string & weight) const; // rows summed into it
    std::vector<std::string> weights() const;            // every weight with a Gram

    // rows [row0, row0 + n_rows) of the weight's Gram, row-major, into dst (n_rows*n floats); n_rows < 0: to the end
    bool read(const std::string & weight, float * dst, int64_t row0 = 0, int64_t n_rows = -1) const;
    // the weight's .in_sum2 (diag G)
    bool read_sum2(const std::string & weight, float * dst) const;

    std::vector<std::string> datasets;
    bool                     complete = false;

private:
    struct gram_info {
        size_t  offset;
        int64_t n;
    };

    bool read_at(void * dst, size_t size, size_t offset) const;

    std::unique_ptr<llama_file> file;
    std::map<std::string, std::string> owners;   // weight -> owner, owners included
    std::map<std::string, gram_info>   grams;    // owner -> its .in_gram
    std::map<std::string, size_t>      sum2_off; // weight -> its .in_sum2
    std::map<std::string, size_t>      count_off;
};
