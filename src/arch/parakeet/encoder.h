// arch/parakeet/encoder.h - Parakeet Conformer encoder graph builder.
//
// Phase 4 step 3 of the encoder port. The .cpp file builds a
// ggml_cgraph that mirrors NeMo's Conformer encoder forward. The C++
// encoder reads dims from ParakeetHParams and tensor pointers from
// ParakeetWeights; both are populated by the loader.
//
// Sub-stages of step 3:
//
//   3a: pre_encode subsampling stack only. Validates layout + conv2d
//       wiring + linear projection + the compute-context lifetime
//       against the reference enc.pre_encode.out on samples/jfk.wav.
//   3b: macaron FF1 on block 0.
//   3c: relative-position MHSA on block 0.
//   3d: conv module on block 0.
//   3e: FF2 + final norm on block 0.
//   3f: loop over 24 blocks, validate against block 1, 12, 23, final.
//
// Each sub-stage adds named dump points consumed by
// scripts/compare_tensors.py. The first-divergent-block-wins debug
// strategy is the entire dev loop.
//
// This header is INTERNAL to src/arch/parakeet/. The public C ABI
// only sees Parakeet::run, which builds and runs the graph below.

#pragma once

#include "ggml.h" // ggml_type

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace transcribe::parakeet {

struct ParakeetHParams;
struct ParakeetWeights;

// Result of building one pass of the encoder graph. The caller is
// responsible for:
//
//   1. allocating a backend buffer for the compute_ctx (typically via
//      ggml_backend_alloc_ctx_tensors against the model's backend),
//   2. uploading the mel data into `mel_in` via ggml_backend_tensor_set,
//   3. running the cgraph via ggml_backend_graph_compute,
//   4. reading or dumping `out` via ggml_backend_tensor_get / the
//      transcribe::debug dumper.
//
// Layout convention (see RESUME.md "Where we are" for the long form):
//   - The reference uses [B, T, C] (channels-last). ggml ne is
//     fast-to-slow, so the natural encoder activation lives at
//     ne=[d_model, T, B=1, 1].
//   - For pre_encode (Conv2d), we treat T_mel as W (ggml ne[0]) and
//     n_mels as H (ne[1]); this matches the row-major layout the
//     C++ MelFrontend already produces, so the input tensor receives
//     the mel buffer with no transpose.
//   - After three stride-2 convs, the encoder activation enters the
//     conformer blocks at ne=[d_model, T_enc, 1, 1] where T_enc =
//     floor(T_mel / 8) (with the per-conv padding/kernel formula
//     applied three times — see encoder.cpp).
// Named intermediate dump points the encoder graph builder
// publishes. The driver in Parakeet::run iterates this set after
// graph_compute and feeds each tensor to transcribe::debug::dump_tensor
// (gated on TRANSCRIBE_DUMP_DIR; zero-cost when unset). Any tensor
// stored here must also be passed to transcribe::debug::mark_tensor_for_dump()
// while building the graph; otherwise the scheduler may reuse its
// buffer before the post-compute dump pass reads it. Adding a new
// sub-stage means appending to this struct + populating/preserving it
// in build_encoder_graph; Parakeet::run doesn't need to know the names.
struct EncoderDumps {
    // Pre-encode (3a). ne=[d_model, T_enc, 1, 1].
    ggml_tensor * pre_encode_out = nullptr;
    // Block 0 sub-step outputs (3b-3e). Each ne=[d_model, T_enc, 1, 1].
    ggml_tensor * block0_after_ff1  = nullptr;
    ggml_tensor * block0_after_attn = nullptr;
    ggml_tensor * block0_after_conv = nullptr;
    ggml_tensor * block0_after_ff2  = nullptr;
    ggml_tensor * block0_out        = nullptr;
    // Spot-check blocks (3f). ne=[d_model, T_enc, 1, 1].
    ggml_tensor * block12_out = nullptr;
    ggml_tensor * block23_out = nullptr;
    // Final encoder output (3f). ne=[d_model, T_enc, 1, 1].
    // Currently == block23_out (the last block's norm_out is the
    // encoder's exit point); kept as a separate field so phase 5's
    // wiring isn't coupled to the block count.
    ggml_tensor * final_out = nullptr;
};

struct EncoderBuild {
    // Input handle, ne=[T_mel, n_mels, 1, 1] f32. Caller fills via
    // ggml_backend_tensor_set with the row-major [n_mels, n_frames]
    // mel buffer from MelFrontend::compute (the memory layout is
    // identical because MelFrontend's row-major matches ggml's
    // ne[0]=fastest convention for this 2-D tensor).
    ggml_tensor * mel_in = nullptr;

    // Sinusoidal positional embedding handle, ne=[d_model, 2*T_enc-1, 1, 1].
    // Sub-stages 3c+ need this; the driver computes the buffer
    // host-side from T_enc (which is read from `out->ne[1]` after
    // build) and uploads via ggml_backend_tensor_set. Null in 3a/3b
    // because the FF1 sub-stage doesn't reference it.
    ggml_tensor * pos_emb_in = nullptr;

    // Encoder forward output, ne=[d_model, T_enc, 1, 1] f32. Equal
    // to dumps.final_out; provided as a separate field so callers
    // that don't care about intermediates can ignore the dumps
    // struct entirely.
    ggml_tensor * out = nullptr;

    // Named intermediate handles for the dump harness. Any field
    // that's nullptr is "not yet wired in this sub-stage" and the
    // driver simply skips it.
    EncoderDumps dumps {};

    // The forward graph. ggml_build_forward_expand has been called
    // with `out`.
    ggml_cgraph * graph = nullptr;
};

// Build a fresh encoder forward graph in `compute_ctx`. The context
// must have been created with `no_alloc=true` so the tensors carry
// metadata only; the caller allocates a backend buffer for them
// after this returns. `n_mel_frames` is the time dimension of the
// mel input (the second dim of the row-major [n_mels, n_frames]
// MelFrontend output) and shapes the input handle accordingly.
//
// Returns an EncoderBuild with all three tensor handles populated.
// On any failure (insufficient mem_size, weights/hparams mismatch,
// degenerate frame count) the returned struct has nullptr fields and
// a diagnostic is logged via stderr; the caller must check.
// kv_type: GGML type for K/V activations in flash attention.
// GGML_TYPE_COUNT means "auto" (f16 for quantized weights, f32 for f32).
// backend_name: primary backend name (e.g. "MTL0", "Vulkan0", "CPU") for
// auto-detecting optimal conv strategy. Env vars override if set.
EncoderBuild build_encoder_graph(ggml_context *          compute_ctx,
                                 const ParakeetWeights & weights,
                                 const ParakeetHParams & hp,
                                 int                     n_mel_frames,
                                 ggml_type               kv_type = GGML_TYPE_COUNT,
                                 const char *            backend_name = "");

} // namespace transcribe::parakeet
