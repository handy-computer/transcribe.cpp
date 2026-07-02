// teardown_safety_unit.cpp - safe_* teardown wrappers under noexcept callers.

#include "ggml-backend.h"
#include "transcribe-backend.h"
#include "transcribe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

std::vector<std::string> g_log;

void capture_cb(transcribe_log_level, const char * msg, void *) {
    g_log.push_back(msg != nullptr ? msg : "");
}

bool log_contains(const char * needle) {
    for (const auto & m : g_log) {
        if (m.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

void set_env(const char * key, const char * value) {
#if defined(_WIN32)
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

void unset_env(const char * key) {
#if defined(_WIN32)
    _putenv_s(key, "");  // MSVC CRT: empty value removes the variable
#else
    unsetenv(key);
#endif
}

void test_wrappers_contain_injected_throw() {
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(be != nullptr);
    ggml_backend_buffer_t buf = nullptr;
    if (be != nullptr) {
        buf = ggml_backend_alloc_buffer(be, 256);
        CHECK(buf != nullptr);
    }

    g_log.clear();
    set_env("TRANSCRIBE_TEST_TEARDOWN_THROW", "1");
    // If a wrapper leaks the injected throw, this noexcept call terminates.
    [&]() noexcept {
        transcribe::safe_buffer_free(buf);
    }();
    [&]() noexcept {
        transcribe::safe_backend_free(be);
    }();
    unset_env("TRANSCRIBE_TEST_TEARDOWN_THROW");
    // The warning proves the hook fired and was contained.
    CHECK(log_contains("fault injection"));
}

void test_null_is_noop_even_with_hook() {
    g_log.clear();
    set_env("TRANSCRIBE_TEST_TEARDOWN_THROW", "1");
    [&]() noexcept {
        transcribe::safe_backend_free(nullptr);
        transcribe::safe_buffer_free(nullptr);
        transcribe::safe_sched_free(nullptr);
    }();
    unset_env("TRANSCRIBE_TEST_TEARDOWN_THROW");
    CHECK(!log_contains("fault injection"));
}

#if !defined(_WIN32)
void test_empty_hook_value_is_inert() {
    // POSIX-only: Windows cannot represent an empty-valued environment variable.
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(be != nullptr);
    g_log.clear();
    set_env("TRANSCRIBE_TEST_TEARDOWN_THROW", "");
    [&]() noexcept {
        transcribe::safe_backend_free(be);
    }();
    unset_env("TRANSCRIBE_TEST_TEARDOWN_THROW");
    CHECK(!log_contains("fault injection"));
}
#endif

}  // namespace

int main() {
    transcribe_init_backends_default();
    transcribe_log_set(&capture_cb, nullptr);

    test_wrappers_contain_injected_throw();
    test_null_is_noop_even_with_hook();
#if !defined(_WIN32)
    test_empty_hook_value_is_inert();
#endif

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
