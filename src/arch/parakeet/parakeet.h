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
#include "transcribe-backend.h"
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

    // Runtime backend plan. Resolved at load() time from
    // transcribe_model_params::backend via
    // transcribe::load_common::init_backends. See
    // transcribe-backend.h for the plan semantics. The plan's
    // `scheduler_list` owns the backends; the destructor frees
    // them in reverse order. Helpers that key off "is this CPU?"
    // check `plan.primary_kind == BackendKind::Cpu` directly.
    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // Fused BN parameters live in a separate ggml context + buffer,
    // computed at load time from the raw BN tensors. Freed in dtor.
    ggml_context *          bn_fused_ctx    = nullptr;
    ggml_backend_buffer_t   bn_fused_buffer = nullptr;

    // On a CPU primary backend the conformer 1×1 pointwise conv weights
    // are dequantized from their on-disk F16 form to F32 at load time.
    // The current quantizer (transcribe-quantize, post commit 6fee9b9)
    // routes every Conformer ConvPw to F16 across all presets so that
    // GPU backends with native F16 compute (Metal, Vulkan, CUDA) get
    // the bandwidth halving for free; on CPUs without native F16
    // arithmetic (Zen 2 and earlier) the per-matmul F16→F32 upconvert
    // erases the bandwidth win and dwarfs the original cost. Promoting
    // back to F32 at load time pays the conversion exactly once.
    //
    // Today's parakeet GGUFs predate the universal F16 conv_pw policy
    // and ship F32 conv_pw, so the promotion is a no-op for them. The
    // wiring is in place so the next regen does the right thing
    // automatically. The promotion is also a no-op on every non-CPU
    // primary backend.
    //
    // The CohereBlock comment about ~235 MB of "wasted" originals
    // applies symmetrically: ggml backend buffers don't support
    // freeing individual tensor slots, so the F16 source tensors stay
    // resident in the main weight buffer after promotion. The wired
    // weight slot is repointed at the F32 copy and the encoder graph
    // never touches the originals again.
    ggml_context *          conv_pw_f32_ctx    = nullptr;
    ggml_backend_buffer_t   conv_pw_f32_buffer = nullptr;

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

    // KV type for flash attention, resolved from the context params.
    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;

    ParakeetContext() = default;
    ~ParakeetContext() override;
};

} // namespace transcribe::parakeet
