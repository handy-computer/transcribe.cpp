// run_on_threads_unit.cpp - transcribe::run_on_threads joins every launched
// thread on every path and hands worker / launch exceptions to the caller
// instead of calling std::terminate.

#include "transcribe-batch-util.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>
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

// std::thread wrapper whose construction fails on the k-th launch, like a
// system out of threads. Counts joins so the test can see every launched
// thread was joined before the error surfaced.
struct FlakyThread {
    static int              fail_at;  // 0-based launch index that throws; -1 = never
    static int              launches;
    static std::atomic<int> joins;

    std::thread th;

    template <typename F, typename... A> explicit FlakyThread(F && f, A &&... a) {
        if (launches++ == fail_at) {
            throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
        }
        th = std::thread(std::forward<F>(f), std::forward<A>(a)...);
    }

    void join() {
        th.join();
        joins.fetch_add(1);
    }
};

int              FlakyThread::fail_at  = -1;
int              FlakyThread::launches = 0;
std::atomic<int> FlakyThread::joins{ 0 };

void reset_flaky(int fail_at) {
    FlakyThread::fail_at  = fail_at;
    FlakyThread::launches = 0;
    FlakyThread::joins    = 0;
}

void test_every_tid_runs_once() {
    for (int n : { 0, 1, 2, 7 }) {
        std::vector<std::atomic<int>> hits(static_cast<size_t>(n > 0 ? n : 1));
        transcribe::run_on_threads(n, [&](int tid) { hits[static_cast<size_t>(tid)].fetch_add(1); });
        for (auto & h : hits) {
            CHECK(h.load() == 1);
        }
    }
}

void test_worker_throw_rethrown_after_join() {
    std::atomic<int> done{ 0 };
    bool             caught = false;
    try {
        transcribe::run_on_threads(4, [&](int tid) {
            if (tid == 2) {
                throw std::bad_alloc();
            }
            done.fetch_add(1);
        });
    } catch (const std::bad_alloc &) {
        caught = true;  // exception type preserved across the thread boundary
    }
    CHECK(caught);
    CHECK(done.load() == 3);  // every other thread ran to completion and was joined
}

void test_launch_failure_joins_launched_threads() {
    // Launch index 0 is tid 1. Failing at index 2 (tid 3) means tids 1 and 2
    // were launched and must be joined; tid 0 must not run.
    reset_flaky(2);
    std::atomic<int> ran{ 0 };
    bool             tid0_ran = false;
    bool             caught   = false;
    try {
        transcribe::run_on_threads<FlakyThread>(5, [&](int tid) {
            if (tid == 0) {
                tid0_ran = true;
            }
            ran.fetch_add(1);
        });
    } catch (const std::system_error &) {
        caught = true;
    }
    CHECK(caught);
    CHECK(!tid0_ran);
    CHECK(FlakyThread::joins.load() == 2);
    CHECK(ran.load() == 2);
}

void test_parallel_for_all_propagates() {
    bool caught = false;
    try {
        (void) transcribe::parallel_for_all(16, 4, [](int i) -> bool {
            if (i == 9) {
                throw std::runtime_error("work");
            }
            return true;
        });
    } catch (const std::runtime_error &) {
        caught = true;
    }
    CHECK(caught);
}

}  // namespace

int main() {
    test_every_tid_runs_once();
    test_worker_throw_rethrown_after_join();
    test_launch_failure_joins_launched_threads();
    test_parallel_for_all_propagates();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("run_on_threads_unit: ok\n");
    return EXIT_SUCCESS;
}
