// transcribe-batch-util.h - shared helpers for the offline batch path
// (transcribe_run_batch family run_batch() hooks).
//
// The per-utterance feature extraction (mel / kaldi-fbank / LFR) that every
// family runs at the top of run_batch() is pure host code with no
// cross-utterance state, so it is embarrassingly parallel across the batch.
// When the encoder runs on a fast accelerator the host feature extraction is
// frequently the dominant wall cost, so batching it across CPU workers is the
// single biggest end-to-end lever. Families differ in their frontend type and
// compute() signature, so the work itself is supplied as a per-index callable;
// this header owns only the parallel-dispatch boilerplate.

#pragma once

#include <functional>

namespace transcribe {

// Invoke `work(i)` for every i in [0, n) across up to `n_threads` worker
// threads (clamped to [1, n]; n_threads <= 0 means hardware_concurrency()).
// The caller's `work` MUST be reentrant: each index runs on exactly one
// thread, with no shared mutable state between indices (write per-index
// outputs into caller-owned, index-disjoint storage). Returns true iff every
// invocation returned true; a false return from any index does not stop the
// others (the caller decides what to do with a partial failure — typically
// fall back to the per-utterance path for the whole call).
bool parallel_for_all(int n, int n_threads,
                      const std::function<bool(int)> & work);

} // namespace transcribe
