// Qwen Image 2.1 without downloaded weights: the upstream fixes checked exactly, then generation through the public API
// with synthetic models (a tiny diffusion model and Qwen3-VL encoder, the published VAE layout with deterministic values).
//
//   test-sd-qwen21                     every check, on the CPU
//   test-sd-qwen21 --write-models DIR  write the synthetic diffusion model, VAE and encoder for tools/perf/sd-e2e.py
//   -v                                 show the library log

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "stable-diffusion.h"
#include "kcpp_sd_extensions.h"
#include "core/ggml_extend.h"
#include "model/diffusion/qwen_image_2_1.hpp"
#include "model_loader.h"
#include "runtime/denoiser.hpp"
#include "sd-synthetic.h"

namespace fs = std::filesystem;

static bool verbose = false;
static bool errors_expected = false;
static int failures = 0;
static std::string library_log;

static void check(bool ok, const char* what, const std::string& detail = "") {
    printf("  %-58s %s%s%s\n", what, ok ? "ok" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    failures += !ok;
}

static void log_cb(sd_log_level_t level, const char* text, void*) {
    library_log += text;
    if (verbose || (level >= SD_LOG_ERROR && !errors_expected)) {
        fputs(text, stderr);
    }
}

static bool logged_since(size_t mark, const std::string& what) {
    return library_log.find(what, mark) != std::string::npos;
}

// ---- the flow schedule (b167b94) ----

static void check_schedule() {
    printf("flow schedule\n");
    double worst = 0;
    bool terminal = true;
    for (int seq : {256, 1024, 4096, 8192, 16384}) {
        for (uint32_t n : {1u, 2u, 8u, 20u, 50u}) {
            auto got = FluxScheduler(seq, VERSION_QWEN_IMAGE_2_1).get_sigmas(n, 0.f, 1.f, nullptr);
            // the official Qwen Image 2.1 defaults: an exponential shift interpolated between (256, 0.5) and
            // (8192, 0.9), stretched so the last model step lands on shift_terminal 0.02
            double mu = 0.5 + (0.9 - 0.5) * (seq - 256) / (8192.0 - 256.0);
            std::vector<double> want(n + 1);
            for (uint32_t i = 0; i <= n; ++i) {
                double t = 1.0 - (double)i / n;
                want[i]  = t <= 0 ? 0 : std::exp(mu) / (std::exp(mu) + (1 / t - 1));
            }
            if (n > 1) {
                double k = (1 - want[n - 1]) / (1 - 0.02);
                for (uint32_t i = 0; i < n; ++i) {
                    want[i] = 1 - (1 - want[i]) / k;
                }
            }
            want[n] = 0;
            for (uint32_t i = 0; i <= n; ++i) {
                worst = std::max(worst, std::fabs(got[i] - want[i]));
            }
            terminal &= n < 2 || std::fabs(got[n - 1] - 0.02f) < 1e-6f;
        }
    }
    char detail[64];
    snprintf(detail, sizeof(detail), "largest difference %.1e", worst);
    check(worst < 1e-5, "Qwen Image 2.1 sigmas match the official schedule", detail);
    check(terminal, "the last model step lands on 0.02");

    bool same = true;
    for (int seq : {256, 1024, 4096, 9000}) {
        for (uint32_t n : {1u, 4u, 28u}) {
            auto got  = FluxScheduler(seq, VERSION_FLUX).get_sigmas(n, 0.f, 1.f, nullptr);
            float m   = (1.15f - 0.5f) / (4096.0f - 256.0f);
            float b   = 0.5f - m * 256.0f;
            float mu  = (float)seq * m + b;
            for (uint32_t i = 0; i <= n; ++i) {
                float t = 1.0f - (float)i / (float)n;
                float s = t <= 0.0f ? 0.0f : flux_time_shift(mu, 1.0f, t);
                same &= memcmp(&s, &got[i], 4) == 0;
            }
        }
    }
    check(same, "Flux sigmas are bitwise those of the fixed anchors");
}

// ---- the VAE convolution scale (0a9340c) ----

static void check_conv_scale() {
    printf("convolution scale\n");
    const int W = 6, H = 6, IC = 4, OC = 3, K = 3, OW = W - K + 1, OH = H - K + 1;
    std::vector<float> wv(K * K * IC * OC), xv(W * H * IC);
    synth::Tensor wt{"w", "F32", {(int64_t)wv.size()}, {}}, xt{"x", "F32", {(int64_t)xv.size()}, {}};
    wv      = synth::values(wt, 1);
    auto xr = synth::values(xt, 2);
    for (auto& v : xr) {
        v /= 0.02f;  // to +-1
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    auto run = [&](float magnitude, float scale, std::vector<float>& out, std::vector<double>& want) {
        for (size_t i = 0; i < xv.size(); ++i) {
            xv[i] = xr[i] * magnitude;
        }
        ggml_init_params params = {64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
        ggml_context* ctx       = ggml_init(params);
        ggml_tensor* w          = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, K, K, 1, IC * OC);
        ggml_tensor* x          = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, 1, IC);
        ggml_tensor* y          = ggml_ext_conv_3d(ctx, cpu, x, w, nullptr, IC, 1, 1, 1, 0, 0, 0, 1, 1, 1, false, scale);
        ggml_cgraph* gf         = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        std::vector<ggml_fp16_t> wh(wv.size());
        std::vector<float> wq(wv.size());
        for (size_t i = 0; i < wv.size(); ++i) {
            wh[i] = ggml_fp32_to_fp16(wv[i]);
            wq[i] = ggml_fp16_to_fp32(wh[i]);
        }
        ggml_backend_tensor_set(w, wh.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_graph_compute(cpu, gf);
        out.resize(ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        want.assign(OW * OH * OC, 0.0);
        for (int oc = 0; oc < OC; ++oc)
            for (int oy = 0; oy < OH; ++oy)
                for (int ox = 0; ox < OW; ++ox)
                    for (int ic = 0; ic < IC; ++ic)
                        for (int ky = 0; ky < K; ++ky)
                            for (int kx = 0; kx < K; ++kx)
                                want[ox + OW * (oy + OH * oc)] += (double)wq[kx + K * (ky + K * (ic + IC * oc))] * xv[(ox + kx) + W * ((oy + ky) + H * ic)];
    };
    auto rel_err = [](const std::vector<float>& got, const std::vector<double>& want) -> double {
        double e = 0, m = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            if (!std::isfinite(got[i])) {
                return INFINITY;
            }
            e = std::max(e, std::fabs(got[i] - want[i]));
            m = std::max(m, std::fabs(want[i]));
        }
        return e / m;
    };
    std::vector<float> out;
    std::vector<double> want;
    run(1.f, 1.f, out, want);
    check(rel_err(out, want) < 1e-2, "unscaled small inputs match a direct convolution");
    run(2e5f, 1.f, out, want);
    check(!std::isfinite(rel_err(out, want)), "unscaled inputs past the F16 range overflow");
    run(2e5f, 1.f / 128.f, out, want);
    double e = rel_err(out, want);
    check(e < 1e-2, "scaled by 1/128 they stay finite and match", "relative error " + std::to_string(e));
    ggml_backend_free(cpu);
}

// ---- the token layout ----

static void check_layout() {
    printf("token layout\n");
    using Qwen::QwenImage21Layout;
    auto text = QwenImage21Layout::build(5, sd::Tensor<int32_t>(), {{4, 4}});
    check(text.prefix_length == 5 && text.positions.size() == 21 && text.segments.size() == 2 &&
              text.positions[5] == std::vector<float>{5, -2, -2},
          "text then the target image");

    auto slots = sd::Tensor<int32_t>::zeros({8});
    for (int i = 2; i < 6; ++i) {
        slots[i] = 1;
    }
    auto edit = QwenImage21Layout::build(8, slots, {{4, 4}, {4, 4}});
    bool ok   = edit.prefix_length == 2 + 16 + 2 && edit.positions.size() == 36 && edit.segments.size() == 4 &&
              edit.segments[1].image_index == 0 && edit.segments[1].context_start == 2 &&
              edit.positions[2] == std::vector<float>{2, -2, -2} && edit.positions[18] == std::vector<float>{6, 6, 6} &&
              edit.positions[20] == std::vector<float>{8, -2, -2};
    check(ok, "a reference image's vision slots take its latent grid");

    bool threw = false;
    slots[5]   = 0;
    try {
        QwenImage21Layout::build(8, slots, {{4, 4}, {4, 4}});
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "slots that don't cover the reference latent are rejected");
}

// ---- synthetic models ----

static const int64_t LLM_HIDDEN = 128, DIT_INTER = 512;

static std::vector<synth::Tensor> qwen21_dit() {
    Qwen::QwenImage21Config config;
    config.hidden_size       = 256;
    config.context_dim       = LLM_HIDDEN;
    config.intermediate_size = DIT_INTER;
    config.num_layers        = 2;
    config.fused_mlp         = true;
    Qwen::QwenImage21Model model(config);
    auto tensors = synth::from_block(model);
    for (auto& t : tensors) {
        t.dtype = "F32";
        // gates are tanh(modulation); at random-init scale they sit near 0 and the text barely reaches the image
        if (t.name.find("modulation") != std::string::npos) {
            t.gain = 32.f;
        }
    }
    return tensors;
}

static std::string data_dir(const char* argv0) {
    for (auto dir : {fs::path("tests/data"), fs::path(argv0).parent_path() / "tests/data"}) {
        if (fs::exists(dir / "qwen-image-2.1-vae.tensors")) {
            return dir.string();
        }
    }
    fprintf(stderr, "tests/data/qwen-image-2.1-vae.tensors not found; run from the repository root\n");
    exit(1);
}

static bool write_models(const std::string& dir, const std::string& data) {
    fs::create_directories(dir);
    return synth::write_safetensors(dir + "/dit.safetensors", qwen21_dit(), 1) &&
           synth::write_safetensors(dir + "/vae.safetensors", synth::from_table(data + "/qwen-image-2.1-vae.tensors"), 2) &&
           synth::write_safetensors(dir + "/llm.safetensors", synth::qwen3vl(LLM_HIDDEN, 2, true), 3);
}

// ---- generation ----

struct Image {
    int w = 0, h = 0, c = 0;
    std::vector<uint8_t> px;
};

static sd_ctx_t* load(const std::string& dir, const std::string& dit, const std::string& llm, const char* model_args = "",
                      const std::string& taesd = "") {
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    p.taesd_path           = taesd.c_str();
    std::string d = dir + "/" + dit, v = dir + "/vae.safetensors", l = dir + "/" + llm;
    p.diffusion_model_path = d.c_str();
    p.vae_path             = v.c_str();
    p.llm_path             = l.c_str();
    p.n_threads            = sd_get_num_physical_cores();
    p.model_args           = model_args;
    return new_sd_ctx(&p);
}

static bool generate(sd_ctx_t* ctx, Image* out, const Image* ref = nullptr, const std::string& lora = "") {
    sd_img_gen_params_t g;
    sd_img_gen_params_init(&g);
    g.prompt                             = "a red fox sitting in the snow";
    g.negative_prompt                    = "";
    g.width                              = 64;
    g.height                             = 64;
    g.seed                               = 42;
    g.sample_params.sample_steps         = 3;
    g.sample_params.guidance.txt_cfg     = 4.f;
    sd_image_t r{};
    if (ref) {
        r                  = {(uint32_t)ref->w, (uint32_t)ref->h, (uint32_t)ref->c, const_cast<uint8_t*>(ref->px.data())};
        g.ref_images       = &r;
        g.ref_images_count = 1;
    }
    sd_lora_t l{false, 1.f, lora.c_str()};
    if (!lora.empty()) {
        g.loras      = &l;
        g.lora_count = 1;
    }
    sd_image_t* images = nullptr;
    int n              = 0;
    if (!generate_image(ctx, &g, &images, &n) || n < 1 || images == nullptr || images[0].data == nullptr) {
        return false;
    }
    out->w = images[0].width;
    out->h = images[0].height;
    out->c = images[0].channel;
    out->px.assign(images[0].data, images[0].data + (size_t)out->w * out->h * out->c);
    for (int i = 0; i < n; ++i) {
        free(images[i].data);
    }
    free(images);
    return true;
}

static double mean_diff(const Image& a, const Image& b) {
    if (a.px.size() != b.px.size() || a.px.empty()) {
        return INFINITY;
    }
    double s = 0;
    for (size_t i = 0; i < a.px.size(); ++i) {
        s += std::abs((int)a.px[i] - (int)b.px[i]);
    }
    return s / a.px.size();
}

static int max_diff(const Image& a, const Image& b) {
    if (a.px.size() != b.px.size() || a.px.empty()) {
        return 256;
    }
    int m = 0;
    for (size_t i = 0; i < a.px.size(); ++i) {
        m = std::max(m, std::abs((int)a.px[i] - (int)b.px[i]));
    }
    return m;
}

static std::string fmt(double v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.3f", v);
    return buf;
}

static void check_detection(const std::string& dir) {
    printf("model detection\n");
    auto version = [&](const std::string& file) {
        ModelLoader loader;
        loader.init_from_file(dir + "/" + file, "model.diffusion_model.");
        return loader.get_sd_version();
    };
    check(version("dit.safetensors") == VERSION_QWEN_IMAGE_2_1, "Qwen Image 2.1 from its text norm");
    std::vector<synth::Tensor> qwen = {{"img_in.weight", "F16", {8, 64}, {}}, {"transformer_blocks.0.img_mod.1.weight", "F16", {8, 8}, {}}};
    synth::write_safetensors(dir + "/qwen.safetensors", qwen, 4);
    check(version("qwen.safetensors") == VERSION_QWEN_IMAGE, "Qwen Image still detected");
    qwen.push_back({"time_text_embed.addition_t_embedding.weight", "F16", {8, 8}, {}});
    synth::write_safetensors(dir + "/qwen.safetensors", qwen, 4);
    check(version("qwen.safetensors") == VERSION_QWEN_IMAGE_LAYERED, "Qwen Image Layered still detected");
}

struct PreviewSeen {
    int calls = 0, channels = 0;
};

static void check_generation(const std::string& dir, const std::string& repo) {
    printf("generation (synthetic models, CPU)\n");
    sd_ctx_t* ctx = load(dir, "dit.safetensors", "llm.safetensors");
    if (ctx == nullptr) {
        check(false, "load the synthetic models");
        return;
    }
    check(std::string(sd_get_model_version_name(ctx)) == "Qwen Image 2.1", "loads as Qwen Image 2.1");
    auto info = kcpp_sd::get_model_info(ctx);
    check(info.is_qwenimg && info.supports_ref_image && info.vae_scale_factor == 16, "koboldcpp sees an editing Qwen model, VAE 16x");

    Image t2i, again, edit;
    PreviewSeen preview;
    sd_set_preview_callback([](int, int count, sd_image_t* frames, bool, void* data) {
        auto* seen = (PreviewSeen*)data;
        seen->calls++;
        seen->channels = count > 0 ? (int)frames[0].channel : 0;
    }, PREVIEW_PROJ, 1, true, false, &preview);
    size_t mark = library_log.size();
    bool ok     = generate(ctx, &t2i);
    sd_set_preview_callback(nullptr, PREVIEW_NONE, 1, true, false, nullptr);
    check(ok && t2i.w == 64 && t2i.h == 64 && t2i.c == 4, "text-to-image gives 64x64 RGBA");
    check(logged_since(mark, "cached prefix") && logged_since(mark, "tokens, f32)"), "the text prefix was cached (F32 without flash attention)");
    check(preview.calls > 0 && preview.channels == 4, "latent previews come out RGBA");
    auto [lo, hi] = std::minmax_element(t2i.px.begin(), t2i.px.end());
    check(ok && *hi - *lo > 16, "the image is not flat");
    check(generate(ctx, &again) && again.px == t2i.px, "a repeat is bitwise equal");
    bool edit_ok = generate(ctx, &edit, &t2i);
    check(edit_ok && edit.w == 64 && mean_diff(edit, t2i) > 1, "an edit through the vision encoder", "mean difference " + fmt(mean_diff(edit, t2i)));

    // a LoRA trained on the unfused gate and projection must land on the halves of the fused gate_up weight
    const int64_t H = 256, R = 4;
    std::vector<synth::Tensor> lora = {
        {"diffusion_model.transformer_blocks.0.img_mlp.gate_layer.lora_down.weight", "F32", {R, H}, {}},
        {"diffusion_model.transformer_blocks.0.img_mlp.gate_layer.lora_up.weight", "F32", {DIT_INTER, R}, {}},
        {"diffusion_model.transformer_blocks.0.img_mlp.proj.lora_down.weight", "F32", {R, H}, {}},
        {"diffusion_model.transformer_blocks.0.img_mlp.proj.lora_up.weight", "F32", {DIT_INTER, R}, {}},
    };
    synth::write_safetensors(dir + "/lora.safetensors", lora, 5);
    auto delta = [&](int part) {
        auto down = synth::values(lora[2 * part], 5), up = synth::values(lora[2 * part + 1], 5);
        std::vector<float> d(DIT_INTER * H, 0.f);
        for (int64_t o = 0; o < DIT_INTER; ++o)
            for (int64_t i = 0; i < H; ++i)
                for (int64_t r = 0; r < R; ++r)
                    d[o * H + i] += up[o * R + r] * down[r * H + i];
        return d;
    };
    auto gate = delta(0), proj = delta(1);
    for (int swapped = 0; swapped < 2; ++swapped) {
        auto dit = qwen21_dit();
        for (auto& t : dit) {
            if (t.name == "transformer_blocks.0.img_mlp.gate_up.weight") {
                t.delta = swapped ? proj : gate;
                t.delta.insert(t.delta.end(), (swapped ? gate : proj).begin(), (swapped ? gate : proj).end());
            }
        }
        synth::write_safetensors(dir + (swapped ? "/dit-swapped.safetensors" : "/dit-baked.safetensors"), dit, 1);
    }
    Image with_lora, baked, swapped;
    bool lora_ok = generate(ctx, &with_lora, nullptr, dir + "/lora.safetensors");
    free_sd_ctx(ctx);
    for (auto [file, img] : {std::pair{"dit-baked.safetensors", &baked}, std::pair{"dit-swapped.safetensors", &swapped}}) {
        sd_ctx_t* c = load(dir, file, "llm.safetensors");
        lora_ok &= c && generate(c, img);
        free_sd_ctx(c);
    }
    double d_baked = mean_diff(with_lora, baked), d_swapped = mean_diff(with_lora, swapped), d_none = mean_diff(with_lora, t2i);
    check(lora_ok && d_none > 1 && d_baked < 0.1 * d_swapped, "a gate/proj LoRA matches its delta baked into gate_up",
          "mean difference " + fmt(d_baked) + " vs " + fmt(d_swapped) + " with the halves swapped");

    // the prefix cache reuses step-independent keys and values; only matrix sizes change
    Image off, q8;
    sd_ctx_t* c = load(dir, "dit.safetensors", "llm.safetensors", "qwen_image_2_1_prefix_cache=false");
    mark          = library_log.size();
    bool cache_ok = c && generate(c, &off) && !logged_since(mark, "cached prefix");
    free_sd_ctx(c);
    c    = load(dir, "dit.safetensors", "llm.safetensors", "qwen_image_2_1_prefix_cache_type=q8_0");
    mark = library_log.size();
    cache_ok &= c && generate(c, &q8) && logged_since(mark, "tokens, q8_0)");
    free_sd_ctx(c);
    double d_off = mean_diff(off, t2i), d_q8 = mean_diff(q8, t2i);
    if (verbose) {
        printf("    largest differences: cache off %d, q8_0 %d, edit %d, lora %d, lora swapped %d\n", max_diff(off, t2i), max_diff(q8, t2i),
               max_diff(edit, t2i), max_diff(with_lora, t2i), max_diff(with_lora, swapped));
    }
    check(cache_ok && max_diff(off, t2i) <= 1, "prefix cache on and off agree", "mean difference " + fmt(d_off));
    check(cache_ok && d_q8 < 0.5, "a q8_0 prefix cache stays close", "mean difference " + fmt(d_q8));

    synth::write_safetensors(dir + "/llm-novision.safetensors", synth::qwen3vl(LLM_HIDDEN, 2, false), 3);
    c = load(dir, "dit.safetensors", "llm-novision.safetensors");
    Image refused;
    errors_expected = true;
    check(c && !generate(c, &refused, &t2i), "an edit without vision weights fails instead of ignoring the image");
    errors_expected = false;
    free_sd_ctx(c);

    // koboldcpp passes embd_res/taesd.embd and swaps in the family's tiny VAE; the Wan 2.1 one can't decode 2.1 latents
    c = load(dir, "dit.safetensors", "llm.safetensors", "", repo + "/embd_res/taesd.embd");
    Image with_taesd;
    check(c && generate(c, &with_taesd) && with_taesd.px == t2i.px, "with koboldcpp's TAESD path it still decodes with the 2.1 VAE");
    free_sd_ctx(c);
}

int main(int argc, char** argv) {
    std::string write_dir;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-v")) {
            verbose = true;
        } else if (!strcmp(argv[i], "--write-models") && i + 1 < argc) {
            write_dir = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [-v] [--write-models DIR]\n", argv[0]);
            return 1;
        }
    }
    sd_set_log_callback(log_cb, nullptr);
    sd_set_progress_callback([](int, int, float, void*) {}, nullptr);
    std::string data = data_dir(argv[0]);
    if (!write_dir.empty()) {
        return write_models(write_dir, data) ? 0 : 1;
    }

    check_schedule();
    check_conv_scale();
    check_layout();

    std::string dir = (fs::temp_directory_path() / ("test-sd-qwen21-" + std::to_string(getpid()))).string();
    if (!write_models(dir, data)) {
        fprintf(stderr, "could not write the synthetic models to %s\n", dir.c_str());
        return 1;
    }
    check_detection(dir);
    check_generation(dir, (fs::path(data) / ".." / "..").string());
    fs::remove_all(dir);

    printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
