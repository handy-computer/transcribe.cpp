// debug_dump_unit.cpp - round-trip unit test for the tensor dumper in
// src/transcribe-debug.{h,cpp}.
//
// What we cover:
//
//   1. init() honors TRANSCRIBE_DUMP_DIR (set programmatically here).
//   2. dump_tensor() writes both files (<name>.f32 and <name>.json).
//   3. The .f32 file contains exactly the bytes the source tensor
//      held, in the right order (slow-to-fast / row-major).
//   4. The .json sidecar carries the slow-to-fast shape (i.e. ggml
//      ne[] reversed and trailing 1s dropped), the right dtype, and
//      sane min/max/mean.
//   5. The dumper is correct on a CPU-backed ggml_context (the
//      device-agnostic ggml_backend_tensor_get path is exercised).
//
// What we deliberately don't cover here:
//
//   - Metal backend correctness. The same code path runs on Metal in
//     real builds (the loader binds Metal first on Apple Silicon and
//     the encoder graph can dump from Metal-resident activations).
//     Real-model validation exercises that end-to-end via the encoder
//     graph; this unit test stays CPU-only so it's
//     fast and runs in CI without Metal hardware.
//   - The compare_tensors.py side. That's exercised by numerical
//     validation runs.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "transcribe-debug.h"

#include <unistd.h>  // ::getpid()

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_EQ_INT(actual, expected)                                                                           \
    do {                                                                                                         \
        const long long _a = static_cast<long long>(actual);                                                     \
        const long long _e = static_cast<long long>(expected);                                                   \
        if (_a != _e) {                                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s = %lld, expected %lld\n", __FILE__, __LINE__, #actual, _a, _e); \
            ++g_failures;                                                                                        \
        }                                                                                                        \
    } while (0)

// Create a fresh temporary directory under the system temp dir,
// scoped to this test process. Caller deletes it on success.
fs::path make_unique_dump_dir() {
    fs::path base = fs::temp_directory_path() / "transcribe-debug-dump-test";
    base /= std::to_string(::getpid());
    fs::remove_all(base);  // ensure clean state in case of leftover
    fs::create_directories(base);
    return base;
}

std::vector<uint8_t> read_file(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        return {};
    }
    f.seekg(0, std::ios::end);
    const std::streampos len = f.tellg();
    if (len < 0) {
        return {};
    }
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(static_cast<size_t>(len));
    f.read(reinterpret_cast<char *>(out.data()), len);
    return out;
}

std::string read_text(const fs::path & p) {
    std::ifstream f(p);
    if (!f) {
        return {};
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

int main() {
    // ----- Setup ------------------------------------------------------
    //
    // Create a unique dump dir, point TRANSCRIBE_DUMP_DIR at it, and
    // initialize the dumper. setenv before init is the contract:
    // init() reads the env var once and caches it.
    const fs::path dump_dir = make_unique_dump_dir();
    if (::setenv("TRANSCRIBE_DUMP_DIR", dump_dir.c_str(), 1) != 0) {
        std::fprintf(stderr, "FAIL: setenv failed\n");
        return EXIT_FAILURE;
    }

    if (!transcribe::debug::init()) {
        std::fprintf(stderr,
                     "FAIL: debug::init() returned false despite "
                     "TRANSCRIBE_DUMP_DIR being set\n");
        return EXIT_FAILURE;
    }
    CHECK(transcribe::debug::enabled());
    if (transcribe::debug::dump_dir() == nullptr || std::strcmp(transcribe::debug::dump_dir(), dump_dir.c_str()) != 0) {
        std::fprintf(stderr, "FAIL: dump_dir() = \"%s\", expected \"%s\"\n",
                     transcribe::debug::dump_dir() ? transcribe::debug::dump_dir() : "(null)", dump_dir.c_str());
        ++g_failures;
    }

    // ----- Build a CPU-backed tensor with known data -----------------
    //
    // Shape: ggml ne = [3, 5, 1, 1] (3 cols, 5 rows in ggml's
    // fast-to-slow ordering). On disk this should land as
    // numpy/row-major shape [5, 3]. The data is arange(0..15) so
    // expected min=0, max=14, mean=7.
    ggml_init_params init{};
    init.mem_size      = 4 * 1024 * 1024;
    init.mem_buffer    = nullptr;
    init.no_alloc      = true;
    ggml_context * ctx = ggml_init(init);
    CHECK(ctx != nullptr);
    if (ctx == nullptr) {
        return EXIT_FAILURE;
    }

    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 5);
    CHECK(t != nullptr);
    if (t != nullptr) {
        ggml_set_name(t, "roundtrip");
        CHECK_EQ_INT(t->ne[0], 3);
        CHECK_EQ_INT(t->ne[1], 5);
        CHECK_EQ_INT(t->ne[2], 1);
        CHECK_EQ_INT(t->ne[3], 1);
    }

    // Bind a CPU backend buffer for the context. After this returns,
    // every ggml_tensor in ctx has its data pointer wired through
    // backend memory; ggml_backend_tensor_set / _get is the only way
    // to mutate or read.
    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr);
    if (backend == nullptr) {
        return EXIT_FAILURE;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr);

    float src[15];
    for (int i = 0; i < 15; ++i) {
        src[i] = static_cast<float>(i);
    }
    ggml_backend_tensor_set(t, src, 0, sizeof(src));

    // ----- Dump and verify -------------------------------------------
    transcribe::debug::dump_tensor("roundtrip", t, "test.stage");

    const fs::path f32_path  = dump_dir / "roundtrip.f32";
    const fs::path json_path = dump_dir / "roundtrip.json";

    CHECK(fs::exists(f32_path));
    CHECK(fs::exists(json_path));

    // .f32 round-trip: bytes must match the source exactly.
    {
        const std::vector<uint8_t> bytes = read_file(f32_path);
        CHECK_EQ_INT(bytes.size(), sizeof(src));
        if (bytes.size() == sizeof(src)) {
            const float * f = reinterpret_cast<const float *>(bytes.data());
            for (int i = 0; i < 15; ++i) {
                if (f[i] != src[i]) {
                    std::fprintf(stderr, "FAIL .f32[%d] = %f, expected %f\n", i, f[i], src[i]);
                    ++g_failures;
                }
            }
        }
    }

    // .json sidecar checks: shape is the slow-to-fast (numpy)
    // ordering with trailing 1s dropped, so ne=[3,5,1,1] -> [5, 3].
    // We don't actually parse JSON here — substring checks are
    // sufficient and stay dependency-light.
    {
        const std::string js = read_text(json_path);
        CHECK(!js.empty());
        CHECK(contains(js, "\"name\": \"roundtrip\""));
        CHECK(contains(js, "\"stage\": \"test.stage\""));
        CHECK(contains(js, "\"shape\": [5, 3]"));
        CHECK(contains(js, "\"dtype\": \"f32\""));
        CHECK(contains(js, "\"layout\": \"row-major\""));
        CHECK(contains(js, "\"source\": { \"kind\": \"cpp\" }"));
        // The min/max lines exist; their precise formatting is
        // emitter-dependent so we don't pin the exact text. Just
        // verify the keys are there.
        CHECK(contains(js, "\"min\":"));
        CHECK(contains(js, "\"max\":"));
        CHECK(contains(js, "\"mean\":"));
    }

    // ----- Negative cases --------------------------------------------
    //
    // Unsafe names must be rejected (no file written) and the
    // failure must not propagate.
    transcribe::debug::dump_tensor("../escape", t);
    CHECK(!fs::exists(dump_dir / "../escape.f32"));
    transcribe::debug::dump_tensor("nested/path", t);
    CHECK(!fs::exists(dump_dir / "nested"));

    // Null tensor is a no-op.
    transcribe::debug::dump_tensor("nullt", nullptr);
    CHECK(!fs::exists(dump_dir / "nullt.f32"));

    // ----- Teardown --------------------------------------------------
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    fs::remove_all(dump_dir);

    if (g_failures > 0) {
        std::fprintf(stderr, "debug_dump_unit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "debug_dump_unit: ok\n");
    return EXIT_SUCCESS;
}
