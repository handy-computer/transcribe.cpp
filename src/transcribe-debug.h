// transcribe-debug.h - per-stage tensor dumping for the numerical
// accuracy harness.
//
// Phase 4 step 2 of the encoder port. Provides a single internal API
// the encoder graph builder can call to write a named tensor's
// contents to disk in the same on-disk format that
// scripts/dump_reference.py emits, so scripts/compare_tensors.py can
// diff a C++ dump dir against a parakeet-mlx dump dir without
// translation.
//
// On-disk format (matches scripts/dump_reference.py::write_dump):
//
//   <dump_dir>/<name>.f32   raw little-endian fp32, row-major (C order)
//   <dump_dir>/<name>.json  sidecar metadata
//
// The JSON sidecar carries:
//
//   { "name": "...",
//     "stage": "...",          // optional provenance label
//     "shape": [d0, d1, ...],  // numpy/row-major (slow-to-fast)
//     "dtype": "f32",
//     "layout": "row-major",
//     "min": <float>, "max": <float>, "mean": <float>,
//     "source": { "kind": "cpp" } }
//
// Critical layout note: ggml ne[] is fast-to-slow (ne[0] = innermost,
// most-contiguous dim). Numpy / dump_reference.py uses slow-to-fast
// (last axis varies fastest). The dumper converts by reversing ne[]
// and dropping trailing 1s, so a ggml tensor with ne=[1024, 275, 1, 1]
// (typical encoder activation [d_model, T, B=1]) lands on disk as
// shape=[275, 1024] — same as the parakeet-mlx [T, d_model] reference
// after squeezing the batch dim.
//
// Activation: dumping is gated on the TRANSCRIBE_DUMP_DIR environment
// variable. When unset, init() is a no-op and every dump_tensor call
// returns immediately. The encoder graph builder can sprinkle
// dump_tensor() calls freely without affecting normal-build cost.
//
// Thread safety: the dump dir is captured once at init() and the
// per-call file writes are not synchronized. The harness assumes
// single-threaded use during numerical accuracy bringup. Concurrent
// dumps would race on file creation and produce truncated output —
// not a goal for v1.

#pragma once

struct ggml_tensor;

namespace transcribe::debug {

// Initialize the debug dumper from environment. Reads
// TRANSCRIBE_DUMP_DIR. If unset or empty, the dumper stays disabled
// and dump_tensor() is a no-op. Idempotent: safe to call multiple
// times in one process; only the first call has effect. Returns true
// if the dumper is enabled after this call.
bool init();

// True if the dumper has been init()'d AND TRANSCRIBE_DUMP_DIR was
// set. Cheap enough to gate hot-path code paths on.
bool enabled();

// The directory the dumper is writing into, or nullptr if disabled.
// Lifetime is the process; the returned pointer stays valid as long
// as the dumper is enabled. Useful for tests.
const char * dump_dir();

// Dump a tensor to <dump_dir>/<name>.{f32,json}.
//
// `tensor` may live on any backend; the dumper copies the bytes via
// ggml_backend_tensor_get, which is the universal API for reading
// tensor data regardless of backend (host buffer ⇒ memcpy; discrete
// GPU ⇒ readback).
//
// `name` becomes the filename stem. It must be a non-empty string
// containing only filesystem-safe characters (no '/' or '\').
//
// `stage` is an optional provenance label written into the JSON
// sidecar; pass nullptr to omit.
//
// On any failure (disabled, null tensor, dtype other than fp32, IO
// error, malloc error), logs to stderr and returns without throwing.
// Dumping is debug-only and must never affect compute correctness or
// propagate errors out of a graph build.
void dump_tensor(const char *               name,
                 const struct ggml_tensor * tensor,
                 const char *               stage = nullptr);

// Dump a host-side fp32 buffer to <dump_dir>/<name>.{f32,json}.
//
// Used by code paths that don't run on a ggml backend — phase 5's
// decoder runs the LSTM + joint + TDT loop directly on host floats
// because the per-step compute is small and a backend graph would
// add lifetime complexity for no perf win.
//
// `data` points at `n_elem` contiguous fp32 values. `shape` is the
// numpy/row-major (slow-to-fast) shape and must satisfy
// product(shape) == n_elem; pass `{n_elem}` for a 1D vector.
//
// Same name / stage / failure semantics as the ggml-tensor variant.
void dump_host_f32(const char *           name,
                   const float *          data,
                   long long              n_elem,
                   const long long *      shape,
                   int                    n_dims,
                   const char *           stage = nullptr);

} // namespace transcribe::debug
