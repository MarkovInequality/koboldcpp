// Tokenizes text as hessian-collect does (the vocab's own BOS rule, special tokens parsed), for the calibration
// scripts. Loads only the vocab. Reads one JSON string per line on stdin and writes the token ids of each as a
// JSON array on its own line; with --detokenize, reads JSON arrays of ids and writes JSON strings.
//
// usage: hessian-tokenize MODEL.gguf [--detokenize]

#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [--detokenize]\n", argv[0]);
        return 1;
    }
    const bool detok = argc > 2 && !strcmp(argv[2], "--detokenize");

    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::ios::sync_with_stdio(false);
    std::string line;
    std::vector<llama_token> ids;
    while (std::getline(std::cin, line)) {
        nlohmann::json in;
        try {
            in = nlohmann::json::parse(line);
        } catch (const std::exception & e) {
            std::cout << nlohmann::json({{"error", e.what()}}).dump() << '\n' << std::flush;
            continue;
        }
        if (detok) {
            ids = in.get<std::vector<llama_token>>();
            std::string out(ids.size()*8 + 16, '\0');
            int n = llama_detokenize(vocab, ids.data(), (int32_t) ids.size(), out.data(), (int32_t) out.size(), false, true);
            if (n < 0) {
                out.resize(-n);
                n = llama_detokenize(vocab, ids.data(), (int32_t) ids.size(), out.data(), (int32_t) out.size(), false, true);
            }
            out.resize(n);
            std::cout << nlohmann::json(out).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n' << std::flush;
        } else {
            const std::string text = in.get<std::string>();
            ids.resize(text.size() + 16);
            int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), true, true);
            if (n < 0) {
                ids.resize(-n);
                n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), true, true);
            }
            ids.resize(n);
            std::cout << nlohmann::json(ids).dump() << '\n' << std::flush;
        }
    }

    llama_model_free(model);
    return 0;
}
