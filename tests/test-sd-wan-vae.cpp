// The Wan VAE code that Qwen Image 2.1 shares with the video models, without downloaded weights: a tiny synthetic
// LingBot video model decodes a 9-frame clip through the published Wan 2.1 VAE layout (causal 3D convolutions with the
// temporal feature cache), on the CPU. The frame hash is compared with a golden; record it with the build to compare
// against (CPU results depend on the instruction set, so goldens are per machine).
//
//   test-sd-wan-vae [--golden FILE]   check (default tools/perf/golden/test-sd-wan-vae.txt)
//   test-sd-wan-vae --record FILE     write the golden
//   -v                                show the library log

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "stable-diffusion.h"
#include "model/diffusion/lingbot_video.hpp"
#include "sd-synthetic.h"

namespace fs = std::filesystem;

static bool verbose = false;

static void log_cb(sd_log_level_t level, const char* text, void*) {
    if (verbose || level >= SD_LOG_ERROR) {
        fputs(text, stderr);
    }
}

static bool write_models(const std::string& dir, const std::string& data) {
    // the default width: head count detection divides it by the q-norm size before it reads the real width
    LingBotVideo::LingBotVideoConfig config;
    config.depth             = 1;
    config.intermediate_size = 512;
    config.text_dim          = 128;
    LingBotVideo::LingBotVideoModel model(config);
    fs::create_directories(dir);
    return synth::write_safetensors(dir + "/dit.safetensors", synth::from_block(model), 1) &&
           synth::write_safetensors(dir + "/vae.safetensors", synth::from_table(data + "/wan-2.1-vae.tensors"), 2) &&
           synth::write_safetensors(dir + "/llm.safetensors", synth::qwen3vl(config.text_dim, 2, false), 3);
}

static bool generate(sd_ctx_t* ctx, std::vector<uint8_t>& bytes, int& frames, int& w, int& h, int& c) {
    sd_vid_gen_params_t g;
    sd_vid_gen_params_init(&g);
    g.prompt                         = "a paper boat drifting on a pond";
    g.negative_prompt                = "";
    g.width                          = 64;
    g.height                         = 64;
    g.video_frames                   = 9;
    g.seed                           = 42;
    g.sample_params.sample_steps     = 2;
    g.sample_params.guidance.txt_cfg = 4.f;
    sd_image_t* out                  = nullptr;
    frames                           = 0;
    if (!generate_video(ctx, &g, &out, &frames, nullptr, nullptr) || out == nullptr || frames < 1) {
        return false;
    }
    bytes.clear();
    w = out[0].width;
    h = out[0].height;
    c = out[0].channel;
    for (int i = 0; i < frames; ++i) {
        bytes.insert(bytes.end(), out[i].data, out[i].data + (size_t)out[i].width * out[i].height * out[i].channel);
        free(out[i].data);
    }
    free(out);
    return true;
}

int main(int argc, char** argv) {
    std::string golden = "tools/perf/golden/test-sd-wan-vae.txt", record;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-v")) {
            verbose = true;
        } else if (!strcmp(argv[i], "--golden") && i + 1 < argc) {
            golden = argv[++i];
        } else if (!strcmp(argv[i], "--record") && i + 1 < argc) {
            record = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [-v] [--golden FILE | --record FILE]\n", argv[0]);
            return 1;
        }
    }
    sd_set_log_callback(log_cb, nullptr);
    sd_set_progress_callback([](int, int, float, void*) {}, nullptr);
    if (!fs::exists("tests/data/wan-2.1-vae.tensors")) {
        fprintf(stderr, "tests/data/wan-2.1-vae.tensors not found; run from the repository root\n");
        return 1;
    }

    std::string dir = (fs::temp_directory_path() / ("test-sd-wan-vae-" + std::to_string(getpid()))).string();
    if (!write_models(dir, "tests/data")) {
        fprintf(stderr, "could not write the synthetic models to %s\n", dir.c_str());
        return 1;
    }
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    std::string d = dir + "/dit.safetensors", v = dir + "/vae.safetensors", l = dir + "/llm.safetensors";
    p.diffusion_model_path = d.c_str();
    p.vae_path             = v.c_str();
    p.llm_path             = l.c_str();
    p.n_threads            = 4;
    sd_ctx_t* ctx          = new_sd_ctx(&p);
    std::vector<uint8_t> clip, again;
    int frames = 0, w = 0, h = 0, c = 0;
    bool ok = ctx != nullptr && generate(ctx, clip, frames, w, h, c);
    int frames2, w2, h2, c2;
    bool same = ok && generate(ctx, again, frames2, w2, h2, c2) && again == clip;
    if (ctx) {
        free_sd_ctx(ctx);
    }
    fs::remove_all(dir);

    uint64_t hash = 1469598103934665603ull;
    for (auto b : clip) {
        hash = (hash ^ b) * 1099511628211ull;
    }
    char hex[17];
    snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)hash);
    printf("  %d frames of %dx%dx%d, hash %s\n", frames, w, h, c, hex);
    bool good = ok && frames == 9 && w == 64 && h == 64 && c == 3 && same;
    printf("  %-58s %s\n", "a 9-frame 64x64 clip, bitwise equal on a repeat", good ? "ok" : "FAIL");
    auto [lo, hi] = std::minmax_element(clip.begin(), clip.end());
    bool varied   = ok && *hi - *lo > 16;
    printf("  %-58s %s\n", "the frames are not flat", varied ? "ok" : "FAIL");
    good &= varied;
    if (!record.empty()) {
        std::ofstream(record) << hex << "\n";
        printf("recorded %s\n", record.c_str());
    } else {
        std::string want;
        std::ifstream(golden) >> want;
        bool match = want == hex;
        printf("  %-58s %s%s\n", "matches the golden", match ? "ok" : "FAIL", want.empty() ? "  (no golden; --record one)" : "");
        good &= match;
    }
    printf("%s\n", good ? "PASS" : "FAIL");
    return good ? 0 : 1;
}
