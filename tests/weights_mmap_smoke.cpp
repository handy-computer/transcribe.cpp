// weights_mmap_smoke.cpp - memory-mapped model weights
// (transcribe_model_load_params::no_mmap, TRANSCRIBE_NO_MMAP).
//
// Fixture-driven, runs in CI. Covers:
//   - every fixture family that loads on the CPU maps its weights by default
//     and copies them with no_mmap or TRANSCRIBE_NO_MMAP (read off the load
//     log). Mapped weights are read-only, so a family that writes into its
//     GGUF weights at load (as a fusion done in place would) crashes here;
//   - mapped and copied models give bit-identical results (silero VAD
//     probabilities; ecapa langid candidates, including the Q8_0 fixture
//     whose widened tensors are copied while the rest are mapped);
//   - a caller built before no_mmap existed (shorter struct_size) gets the
//     default;
//   - a valid GGUF whose tensors are misaligned for the device loads by
//     copying, with the same results, instead of aborting;
//   - a truncated GGUF fails to load cleanly;
//   - two models mapped from one file coexist and free in either order;
//   - a model under a non-ASCII path maps (Windows wide-path conversion);
//   - with Metal available, the same holds for Metal-mapped weights.

#include "transcribe.h"
#include "transcribe/langid.h"
#include "transcribe/vad.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

std::string fixture(const char * name) {
    return std::string(TRANSCRIBE_TEST_FIXTURES_DIR) + "/" + name;
}

// ---- log capture ------------------------------------------------------------

std::mutex               g_log_mu;
std::vector<std::string> g_log;

void capture_log(transcribe_log_level level, const char * msg, void * /*userdata*/) {
    if (msg == nullptr || level == TRANSCRIBE_LOG_LEVEL_DEBUG || level == TRANSCRIBE_LOG_LEVEL_CONT) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_log_mu);
    g_log.emplace_back(msg);
}

void clear_log() {
    std::lock_guard<std::mutex> lock(g_log_mu);
    g_log.clear();
}

bool log_has(const char * needle) {
    std::lock_guard<std::mutex> lock(g_log_mu);
    for (const auto & line : g_log) {
        if (line.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// ---- loading ----------------------------------------------------------------

enum class Mode {
    Default,        // params defaults: map where possible
    NoMmapParam,    // no_mmap = true
    NoMmapEnv,      // defaults + TRANSCRIBE_NO_MMAP=1
    OldStructSize,  // struct_size of a caller built before no_mmap existed
};

const char * mode_name(Mode mode) {
    switch (mode) {
        case Mode::Default:
            return "default";
        case Mode::NoMmapParam:
            return "no_mmap";
        case Mode::NoMmapEnv:
            return "TRANSCRIBE_NO_MMAP";
        case Mode::OldStructSize:
            return "old struct_size";
    }
    return "?";
}

bool expect_mapped(Mode mode) {
    return mode == Mode::Default || mode == Mode::OldStructSize;
}

// Load `path` and clear+fill the log with what the load said.
transcribe_model * load(const std::string &        path,
                        Mode                       mode,
                        transcribe_backend_request backend = TRANSCRIBE_BACKEND_CPU,
                        transcribe_status *        st_out  = nullptr) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend = backend;
    if (mode == Mode::NoMmapParam) {
        mp.no_mmap = true;
    }
    if (mode == Mode::OldStructSize) {
        // A garbage byte where the field would be must not be read.
        mp.no_mmap     = true;
        mp.struct_size = offsetof(transcribe_model_load_params, no_mmap);
    }
    if (mode == Mode::NoMmapEnv) {
        set_env("TRANSCRIBE_NO_MMAP", "1");
    }
    clear_log();
    transcribe_model *      m  = nullptr;
    const transcribe_status st = transcribe_model_load_file(path.c_str(), &mp, &m);
    if (mode == Mode::NoMmapEnv) {
        set_env("TRANSCRIBE_NO_MMAP", nullptr);
    }
    if (st_out != nullptr) {
        *st_out = st;
    }
    return m;
}

// Load and check the mapping decision the log reports.
transcribe_model * load_checked(const std::string &        path,
                                Mode                       mode,
                                const char *               tag,
                                transcribe_backend_request backend = TRANSCRIBE_BACKEND_CPU) {
    transcribe_status  st = TRANSCRIBE_OK;
    transcribe_model * m  = load(path, mode, backend, &st);
    if (st != TRANSCRIBE_OK || m == nullptr) {
        std::fprintf(stderr, "FAIL load %s (%s): %s\n", path.c_str(), mode_name(mode), transcribe_status_string(st));
        ++g_failures;
        return nullptr;
    }
    const std::string mapped = std::string(tag) + ": weights memory-mapped";
    const std::string copied = std::string(tag) + ": weights copied";
    if (expect_mapped(mode)) {
        if (!log_has(mapped.c_str())) {
            std::fprintf(stderr, "FAIL %s (%s): expected \"%s\" in the load log\n", path.c_str(), mode_name(mode),
                         mapped.c_str());
            ++g_failures;
        }
    } else {
        if (!log_has(copied.c_str()) || !log_has("(memory mapping disabled)")) {
            std::fprintf(stderr, "FAIL %s (%s): expected \"%s ... (memory mapping disabled)\" in the load log\n",
                         path.c_str(), mode_name(mode), copied.c_str());
            ++g_failures;
        }
    }
    return m;
}

// ---- workloads ----------------------------------------------------------------

// Deterministic test signal: bursts of a chirp over low noise.
std::vector<float> signal(int n) {
    std::vector<float> v(static_cast<size_t>(n));
    uint32_t           seed = 12345;
    for (int i = 0; i < n; ++i) {
        seed                      = seed * 1664525u + 1013904223u;
        const float noise         = (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f) * 0.01f;
        const float t             = static_cast<float>(i) / 16000.0f;
        const bool  burst         = (i / 8000) % 3 != 0;
        v[static_cast<size_t>(i)] = noise + (burst ? 0.3f * std::sin(6.2831853f * (200.0f + 400.0f * t) * t) : 0.0f);
    }
    return v;
}

std::vector<float> vad_probs(transcribe_model * m) {
    std::vector<float> out;
    if (m == nullptr) {
        return out;
    }
    transcribe_vad_session_params sp;
    transcribe_vad_session_params_init(&sp);
    transcribe_vad_session * s = nullptr;
    CHECK(transcribe_vad_session_init(m, &sp, &s) == TRANSCRIBE_OK);
    if (s == nullptr) {
        return out;
    }
    const std::vector<float> pcm = signal(3 * 16000);
    CHECK(transcribe_vad_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    transcribe_vad_result r;
    transcribe_vad_result_init(&r);
    CHECK(transcribe_vad_get_result(s, &r) == TRANSCRIBE_OK);
    const float * p = transcribe_vad_probs(s);
    if (p != nullptr && r.n_probs > 0) {
        out.assign(p, p + r.n_probs);
    }
    transcribe_vad_session_free(s);
    CHECK(!out.empty());
    return out;
}

struct LangidOut {
    std::vector<int32_t> index;
    std::vector<float>   p;
    std::vector<float>   logit;
};

LangidOut langid(transcribe_model * m) {
    LangidOut out;
    if (m == nullptr) {
        return out;
    }
    transcribe_langid_session_params sp;
    transcribe_langid_session_params_init(&sp);
    sp.n_threads                  = 2;
    transcribe_langid_session * s = nullptr;
    CHECK(transcribe_langid_session_init(m, &sp, &s) == TRANSCRIBE_OK);
    if (s == nullptr) {
        return out;
    }
    const std::vector<float> pcm = signal(2 * 16000);
    CHECK(transcribe_langid_run(s, pcm.data(), static_cast<int>(pcm.size()), nullptr) == TRANSCRIBE_OK);
    transcribe_langid_result r;
    transcribe_langid_result_init(&r);
    CHECK(transcribe_langid_get_result(s, &r) == TRANSCRIBE_OK);
    for (int i = 0; i < r.n_candidates; ++i) {
        transcribe_langid_candidate c;
        transcribe_langid_candidate_init(&c);
        CHECK(transcribe_langid_get_candidate(s, i, &c) == TRANSCRIBE_OK);
        out.index.push_back(c.index);
        out.p.push_back(c.p);
        out.logit.push_back(c.logit);
    }
    transcribe_langid_session_free(s);
    CHECK(!out.index.empty());
    return out;
}

bool same_floats(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

bool same_langid(const LangidOut & a, const LangidOut & b) {
    return a.index == b.index && same_floats(a.p, b.p) && same_floats(a.logit, b.logit);
}

// ---- tests --------------------------------------------------------------------

const Mode k_modes[] = { Mode::Default, Mode::NoMmapParam, Mode::NoMmapEnv, Mode::OldStructSize };

// Load-only families: mapping must not disturb load (and nothing may write
// into a mapped weight while loading).
void test_load_only_families() {
    struct Fixture {
        const char * file;
        const char * tag;
    };

    const Fixture fixtures[] = {
        { "arch_cohere_minimal.gguf",       "cohere"       },
        { "arch_qwen3_asr_minimal.gguf",    "qwen3_asr"    },
        { "arch_granite5_ctc_minimal.gguf", "granite5_ctc" },
    };
    for (const Fixture & f : fixtures) {
        for (Mode mode : k_modes) {
            transcribe_model * m = load_checked(fixture(f.file), mode, f.tag);
            transcribe_model_free(m);
        }
    }
}

void test_silero_equivalence() {
    std::vector<float> baseline;
    for (Mode mode : k_modes) {
        transcribe_model * m     = load_checked(fixture("arch_silero_vad.gguf"), mode, "silero_vad");
        std::vector<float> probs = vad_probs(m);
        if (baseline.empty()) {
            baseline = probs;
        } else if (!same_floats(probs, baseline)) {
            std::fprintf(stderr, "FAIL silero_vad (%s): probabilities differ from the mapped run\n", mode_name(mode));
            ++g_failures;
        }
        transcribe_model_free(m);
    }
}

void test_ecapa_equivalence() {
    // minimal: every tensor mapped. q8_0: the Q8_0 tensors are widened to
    // F16 at load, so they are copied while the rest stay mapped.
    for (const char * file : { "arch_ecapa_tdnn_minimal.gguf", "arch_ecapa_tdnn_q8_0.gguf" }) {
        LangidOut baseline;
        bool      have = false;
        for (Mode mode : k_modes) {
            transcribe_model * m = load_checked(fixture(file), mode, "ecapa_tdnn");
            if (expect_mapped(mode) && std::strstr(file, "q8_0") != nullptr) {
                CHECK(log_has("copied (converted at load)"));
            }
            const LangidOut out = langid(m);
            if (!have) {
                baseline = out;
                have     = true;
            } else if (!same_langid(out, baseline)) {
                std::fprintf(stderr, "FAIL %s (%s): langid differs from the mapped run\n", file, mode_name(mode));
                ++g_failures;
            }
            transcribe_model_free(m);
        }
    }
}

// Valid GGUF, tensors misaligned for the device: copied, same results, no
// abort (ggml_backend_cpu_buffer_from_ptr asserts on an unaligned base).
void test_misaligned_file() {
    transcribe_model * ref  = load(fixture("arch_silero_vad.gguf"), Mode::NoMmapParam);
    const auto         want = vad_probs(ref);
    transcribe_model_free(ref);

    transcribe_status  st = TRANSCRIBE_OK;
    transcribe_model * m = load(fixture("arch_silero_vad_misaligned.gguf"), Mode::Default, TRANSCRIBE_BACKEND_CPU, &st);
    CHECK(st == TRANSCRIBE_OK && m != nullptr);
    CHECK(log_has("silero_vad: weights copied"));
    CHECK(log_has("misaligned"));
    CHECK(same_floats(vad_probs(m), want));
    transcribe_model_free(m);
}

std::string temp_path(const char * stem) {
    std::error_code ec;
    auto            dir = std::filesystem::temp_directory_path(ec);
    if (ec) {
        dir = std::filesystem::path(".");
    }
    return (dir / (std::string("transcribe_weights_mmap_") + stem + "_" + std::to_string(std::rand()) + ".gguf"))
        .string();
}

// A file cut off inside the data section fails to load, in both modes.
void test_truncated_file() {
    std::ifstream     in(fixture("arch_silero_vad.gguf"), std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(bytes.size() > 4096);
    const std::string path = temp_path("truncated");
    {
        std::ofstream out(path, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
    }
    for (Mode mode : { Mode::Default, Mode::NoMmapParam }) {
        transcribe_status  st = TRANSCRIBE_OK;
        transcribe_model * m  = load(path, mode, TRANSCRIBE_BACKEND_CPU, &st);
        CHECK(st != TRANSCRIBE_OK);
        CHECK(m == nullptr);
        transcribe_model_free(m);
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// Two models mapped from one file are independent; free order is free.
void test_two_models_one_file() {
    const std::string  path = fixture("arch_silero_vad.gguf");
    transcribe_model * a    = load_checked(path, Mode::Default, "silero_vad");
    transcribe_model * b    = load_checked(path, Mode::Default, "silero_vad");
    const auto         pa   = vad_probs(a);
    const auto         pb   = vad_probs(b);
    CHECK(same_floats(pa, pb));
    transcribe_model_free(a);
    CHECK(same_floats(vad_probs(b), pb));
    transcribe_model_free(b);
}

// A non-ASCII (UTF-8) path maps too: on Windows the mapping opens the file
// through the UTF-8 -> wide conversion, and a failure there would silently
// fall back to copying. The directory name is spelled in escaped UTF-8 bytes
// so the source stays ASCII.
void test_utf8_path() {
    namespace fs           = std::filesystem;
    const char *    k_name = "mmap-caf\xc3\xa9-\xe6\x97\xa5\xe6\x9c\xac";
    std::error_code ec;
    fs::path        dir = fs::temp_directory_path(ec);
    if (ec) {
        dir = fs::path(".");
    }
    dir /= fs::u8path(k_name);
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    CHECK(!ec);
    const fs::path dst = dir / "arch_silero_vad.gguf";
    fs::copy_file(fs::u8path(fixture("arch_silero_vad.gguf")), dst, fs::copy_options::overwrite_existing, ec);
    CHECK(!ec);

    transcribe_model * m = load_checked(dst.u8string(), Mode::Default, "silero_vad");
    CHECK(!vad_probs(m).empty());
    transcribe_model_free(m);
    fs::remove_all(dir, ec);
}

// Metal-mapped weights (Apple silicon): same decision log, same results as
// copied Metal weights.
void test_metal() {
    if (!transcribe_backend_available(TRANSCRIBE_BACKEND_METAL)) {
        std::fprintf(stderr, "weights_mmap_smoke: no Metal device; skipping the Metal case\n");
        return;
    }
    LangidOut baseline;
    bool      have = false;
    for (Mode mode : { Mode::Default, Mode::NoMmapParam }) {
        transcribe_model * m =
            load_checked(fixture("arch_ecapa_tdnn_minimal.gguf"), mode, "ecapa_tdnn", TRANSCRIBE_BACKEND_METAL);
        const LangidOut out = langid(m);
        if (!have) {
            baseline = out;
            have     = true;
        } else {
            CHECK(same_langid(out, baseline));
        }
        transcribe_model_free(m);
    }
}

}  // namespace

int main() {
    set_env("TRANSCRIBE_NO_MMAP", nullptr);
    transcribe_log_set(capture_log, nullptr);

    test_load_only_families();
    test_silero_equivalence();
    test_ecapa_equivalence();
    test_misaligned_file();
    test_truncated_file();
    test_two_models_one_file();
    test_utf8_path();
    test_metal();

    transcribe_log_set(nullptr, nullptr);
    if (g_failures != 0) {
        std::fprintf(stderr, "weights_mmap_smoke: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("weights_mmap_smoke: ok\n");
    return 0;
}
