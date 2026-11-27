// session_threads_unit.cpp - a session's n_threads bounds the threads the mel
// front-end starts during a run.
//
// The bug this pins: parakeet's one-shot, batch, VAD and buffered-streaming
// paths (and the one-shot paths of canary, cohere, sortformer and whisper)
// called MelFrontend::compute() without the session's n_threads, so the
// STFT fell back to default_n_threads() and started its own worker threads
// even for a session created with n_threads = 1. A caller that sets
// n_threads = 1 to keep all CPU work on its own thread (a real-time audio
// thread, or a process with a fixed CPU budget) got extra threads anyway.
//
// The test counts pthread_create calls made during transcribe_run and
// transcribe_run_batch on the minimal parakeet fixture. The fixture's toy
// encoder (subsampling factor 2) is rejected when its graph is built, so
// each run stops right after the mel front-end and returns an error; the
// count covers the front-end. With n_threads = 1 the count must be zero. A
// control session with n_threads = 2 must start at least one thread in each
// call, which proves the counter sees the library's threads and that both
// calls reached the front-end.
//
// Linux only: the counter interposes pthread_create from the test
// executable, which glibc's dynamic linker resolves ahead of libc. Other
// platforms skip with 77.

#include "transcribe.h"

#include <sys/stat.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(__linux__)
#    include <dlfcn.h>
#    include <pthread.h>
#endif

#ifndef TRANSCRIBE_TEST_FIXTURES_DIR
#    error "TRANSCRIBE_TEST_FIXTURES_DIR must be defined by the build system"
#endif

#if defined(__linux__)

namespace {
std::atomic<int> g_thread_starts{ 0 };
}  // namespace

// Default visibility overrides the build's -fvisibility=hidden, so the
// dynamic linker binds libstdc++'s std::thread to this definition.
extern "C" __attribute__((visibility("default"))) int pthread_create(pthread_t *            thread,
                                                                     const pthread_attr_t * attr,
                                                                     void * (*start)(void *),
                                                                     void * arg) {
    using pthread_create_fn         = int (*)(pthread_t *, const pthread_attr_t *, void * (*) (void *), void *);
    static pthread_create_fn create = reinterpret_cast<pthread_create_fn>(dlsym(RTLD_NEXT, "pthread_create"));
    g_thread_starts.fetch_add(1);
    return create(thread, attr, start, arg);
}

namespace {

const std::string g_fixtures_dir = TRANSCRIBE_TEST_FIXTURES_DIR;

int g_failures = 0;

#    define CHECK(cond)                                                              \
        do {                                                                         \
            if (!(cond)) {                                                           \
                std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
                ++g_failures;                                                        \
            }                                                                        \
        } while (0)

// One second of 16 kHz audio: about 100 mel frames, enough for the STFT to
// split across every thread default_n_threads() could pick.
std::vector<float> make_tone() {
    std::vector<float> pcm(16000);
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = (i % 40 < 20) ? 0.1f : -0.1f;
    }
    return pcm;
}

struct ThreadStarts {
    int run;
    int batch;
};

bool measure(transcribe_model * model, int n_threads, ThreadStarts & out) {
    transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    sp.n_threads                 = n_threads;
    transcribe_session * session = nullptr;
    if (transcribe_session_init(model, &sp, &session) != TRANSCRIBE_OK || session == nullptr) {
        std::fprintf(stderr, "FAIL session_init n_threads=%d\n", n_threads);
        ++g_failures;
        return false;
    }

    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    const std::vector<float> pcm = make_tone();

    g_thread_starts.store(0);
    const transcribe_status run_st = transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp);
    out.run                        = g_thread_starts.load();

    const float * const batch_pcm[2] = { pcm.data(), pcm.data() };
    const int           batch_n[2]   = { static_cast<int>(pcm.size()), static_cast<int>(pcm.size()) };
    g_thread_starts.store(0);
    const transcribe_status batch_st = transcribe_run_batch(session, batch_pcm, batch_n, 2, &rp);
    out.batch                        = g_thread_starts.load();

    std::fprintf(stdout, "n_threads=%d: run %s, %d thread(s) started; batch %s, %d thread(s) started\n", n_threads,
                 transcribe_status_string(run_st), out.run, transcribe_status_string(batch_st), out.batch);
    transcribe_session_free(session);
    return true;
}

}  // namespace

int main() {
    const std::string path = g_fixtures_dir + "/tokenizer_minimal.gguf";
    struct stat       st{};
    if (::stat(path.c_str(), &st) != 0) {
        std::fprintf(stderr,
                     "session_threads_unit: fixture not found: %s\n"
                     "  regenerate with: cmake --build build --target fixtures\n",
                     path.c_str());
        return EXIT_FAILURE;
    }

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    transcribe_model * model = nullptr;
    if (transcribe_model_load_file(path.c_str(), &mp, &model) != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL load %s\n", path.c_str());
        return EXIT_FAILURE;
    }

    ThreadStarts single{};
    if (measure(model, 1, single)) {
        CHECK(single.run == 0);
        CHECK(single.batch == 0);
    }
    ThreadStarts control{};
    if (measure(model, 2, control)) {
        CHECK(control.run >= 1);
        CHECK(control.batch >= 1);
    }

    transcribe_model_free(model);
    if (g_failures > 0) {
        std::fprintf(stderr, "session_threads_unit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "session_threads_unit: ok\n");
    return EXIT_SUCCESS;
}

#else

int main() {
    std::fprintf(stdout, "session_threads_unit: skipped (pthread_create interposition is Linux-only)\n");
    return 77;
}

#endif
