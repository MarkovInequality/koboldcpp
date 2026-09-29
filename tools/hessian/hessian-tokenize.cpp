// Tokenizes text as hessian-collect does (the vocab's own BOS rule, special tokens parsed), for the calibration
// scripts. Loads only the vocab. Reads one JSON string per line on stdin and writes the token ids of each as a
// JSON array on its own line; with --detokenize, reads JSON arrays of ids and writes JSON strings.
//
// usage: hessian-tokenize MODEL.gguf [--detokenize]

#include "llama.h"
#include "common/common.h"

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
    while (std::getline(std::cin, line)) {
        nlohmann::json in;
        try {
            in = nlohmann::json::parse(line);
        } catch (const std::exception & e) {
            std::cout << nlohmann::json({{"error", e.what()}}).dump() << '\n' << std::flush;
            continue;
        }
        if (detok) {
            const std::string out = common_detokenize(vocab, in.get<std::vector<llama_token>>(), true);
            std::cout << nlohmann::json(out).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n' << std::flush;
        } else {
            std::cout << nlohmann::json(common_tokenize(vocab, in.get<std::string>(), true, true)).dump() << '\n' << std::flush;
        }
    }

    llama_model_free(model);
    return 0;
}
