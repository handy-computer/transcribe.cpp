// arch/parakeet/parakeet.h - Parakeet-family internal model and context
// types.
//
// This header is INTERNAL to the Parakeet sources. It defines the
// concrete classes that derive from transcribe_model / transcribe_context
// for this family. It is also visible to tests/transcribe_tokenizer_smoke
// (via the test target's PRIVATE include path) so the test can pull a
// const Tokenizer * out of a model loaded through the public C ABI
// without inventing a temporary public accessor.
//
// Other source files outside src/arch/parakeet/ should not include this
// header. The central dispatcher in transcribe.cpp talks to the family
// only through the Arch trait and the base classes.

#pragma once

#include "decoder.h"
#include "transcribe-context.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend *        ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::parakeet {

// Family defaults — applied before transcribe::read_capability_kv runs,
// so capability KV present overrides the default and capability KV
// absent leaves the default in place. Defined in capabilities.cpp.
//
// There is no per-variant resolver: variant identity is descriptive
// metadata, not a behavioral discriminator. Differences between v2
// and v3 (or any future Parakeet variant) are expressed as
// stt.capability.* / general.languages KV that the converter writes
// and the family-agnostic readers in transcribe-meta.h consume.
// RESUME.md "Decisions still load-bearing" has the rationale and the
// llama.cpp / whisper.cpp comparison that drove it.
void apply_family_invariants(transcribe_capabilities & caps);

// Concrete model. Owns the ggml_context that holds every weight
// tensor's data buffer (allocated by gguf_init_from_file with
// no_alloc=false during load). The destructor frees the ggml_context;
// every borrowed ggml_tensor* in `weights` is invalidated at that
// moment. The first gguf_context (the loader's header-only inspection
// pass) is no longer kept alive past load() — its only consumer is
// the tokenizer ingest, which copies all the data it needs. The
// second gguf_context (the one created by load() to read tensor
// metadata) is freed inside load() once the weights have been
// catalogued.
struct ParakeetModel final : public transcribe_model {
    Tokenizer       tok;
    ParakeetHParams hparams;
    ParakeetWeights weights;
    ggml_context *  ctx_meta = nullptr;

    // Runtime backends. Discovered at load() time via ggml's device
    // registry (GPU → ACCEL/BLAS → CPU). The first entry is the
    // preferred backend for weights; the scheduler dispatches ops
    // to the best backend per-op. All are freed in ~ParakeetModel
    // in reverse order.
    std::vector<ggml_backend_t> backends;
    ggml_backend_buffer_t       backend_buffer = nullptr;

    // Fused BN parameters live in a separate ggml context + buffer,
    // computed at load time from the raw BN tensors. Freed in dtor.
    ggml_context *          bn_fused_ctx    = nullptr;
    ggml_backend_buffer_t   bn_fused_buffer = nullptr;

    // Mel front-end. Constructed once at load() time from the
    // hparams (precomputes the periodic Hann window + Slaney mel
    // filterbank). MelFrontend is documented as const-after-ctor
    // and thread-safe across compute() calls, so it's shared by
    // every context derived from this model. std::optional because
    // MelFrontend has no default constructor; emplace() in load()
    // after the hparams are read.
    std::optional<transcribe::MelFrontend> mel;

    // Host mirror of the predictor + joint weights, populated at
    // load() time from the backend tensors via
    // build_host_decoder_weights. The decoder runs on host (see
    // decoder.h for the rationale); having a const-after-load
    // mirror lets every context share the same arrays without
    // each transcribe_run paying a backend readback cost. The
    // memory cost is small relative to the encoder weights
    // (~35 MB on v2, ~73 MB on v3 vs ~2.4 GB total).
    HostDecoderWeights host_decoder;

    ParakeetModel() = default;
    ~ParakeetModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

// Concrete context. Owns a per-call compute context and a persistent
// multi-backend scheduler that dispatches encoder graph ops to the
// best available backend (GPU for matmuls, BLAS for CPU matmuls,
// CPU for everything else).
struct ParakeetContext final : public transcribe_context {
    // Compute context: holds the cgraph and intermediate tensor
    // metadata. Created with no_alloc=true; data lives in the
    // sched-managed buffers. Reset at the start of every run() call.
    ggml_context *        compute_ctx    = nullptr;

    // Multi-backend scheduler. Persists across calls; manages compute
    // buffer allocation and dispatches ops to the best backend.
    // Reuses buffers when the graph topology is unchanged.
    ggml_backend_sched_t  sched          = nullptr;

    // Encoder forward output, borrowed pointer into compute_ctx
    // tensors. Invalidated when compute_ctx is reset on the next
    // run() call.
    ggml_tensor * encoder_out = nullptr;

    // Per-context scratch reused across runs. Keeping these on the
    // context avoids re-allocating the same medium-sized host buffers
    // every call while preserving the exact arithmetic.
    std::vector<float> mel_buf;
    std::vector<float> pos_buf;
    std::vector<float> pos_div_term;
    std::vector<float> enc_host;
    std::vector<TdtToken> raw_tokens;

    // Per-call timings (t_mel_us, t_encode_us, t_decode_us) live
    // on the base transcribe_context now and are surfaced via the
    // public transcribe_get_timings API.

    // Phase 5: TDT decode result storage lives on the base
    // (transcribe_context::tokens / words / segments / full_text /
    // result_kind / has_result), populated by Parakeet::run during
    // decode. No per-family result fields here.

    ParakeetContext() = default;
    ~ParakeetContext() override;
};

} // namespace transcribe::parakeet
