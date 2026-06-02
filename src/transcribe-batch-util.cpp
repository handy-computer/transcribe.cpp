// transcribe-batch-util.cpp - see transcribe-batch-util.h.

#include "transcribe-batch-util.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace transcribe {

bool parallel_for_all(int n, int n_threads,
                      const std::function<bool(int)> & work)
{
    if (n <= 0) return true;
    if (n_threads <= 0) {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
    }
    n_threads = std::max(1, std::min(n, n_threads));

    std::atomic<int>  next{0};
    std::atomic<bool> all_ok{true};
    auto worker = [&]() {
        int i;
        while ((i = next.fetch_add(1, std::memory_order_relaxed)) < n) {
            if (!work(i)) all_ok.store(false, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(n_threads - 1));
    for (int w = 0; w < n_threads - 1; ++w) pool.emplace_back(worker);
    worker();  // the calling thread participates
    for (auto & th : pool) th.join();

    return all_ok.load(std::memory_order_relaxed);
}

} // namespace transcribe
