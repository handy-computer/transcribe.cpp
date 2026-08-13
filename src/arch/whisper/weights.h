// arch/whisper/weights.h - whisper family binding onto the shared Whisper
// encoder-decoder graph. INTERNAL to src/arch/whisper/.
//
// The tensor catalog, hparams, weight slots, KV cache, and both graph
// builders live in src/whisper_graph/ and are shared with the
// `crisperwhisper` family (see whisper_graph.h for why). This header pulls
// those names into `transcribe::whisper` under their historical
// Whisper-prefixed spellings so the family's model.cpp / bin_load.cpp read
// exactly as they did before the extraction, and binds the two
// family-specific parameters: the `stt.whisper.*` KV prefix and the
// "whisper" tag used in load-error messages.

#pragma once

#include "whisper_graph/kv_cache.h"
#include "whisper_graph/whisper_graph.h"

namespace transcribe::whisper {

// Architecture / frontend types. Same structs the family always used; they
// now have one definition shared with crisperwhisper.
using WhisperHParams  = whisper_graph::HParams;
using WhisperFrontend = whisper_graph::Frontend;
using WhisperEncStem  = whisper_graph::EncStem;
using WhisperEncTop   = whisper_graph::EncTop;
using WhisperEncBlock = whisper_graph::EncBlock;
using WhisperDecTop   = whisper_graph::DecTop;
using WhisperDecBlock = whisper_graph::DecBlock;
using WhisperWeights  = whisper_graph::Weights;

// Backend-resident graph state.
using WhisperKvCache = whisper_graph::KvCache;
using WhisperEncOut  = whisper_graph::EncOut;

using whisper_graph::enc_out_init;
using whisper_graph::k_cross_kv_pad;
using whisper_graph::kv_cache_init;
using whisper_graph::kv_cache_init_batched;
using whisper_graph::kv_pad_self_attn;

// This family's KV namespace and error-message tag. crisperwhisper binds
// "stt.crisperwhisper" / "crisperwhisper" to the same readers.
inline constexpr const char k_kv_prefix[]  = "stt.whisper";
inline constexpr const char k_family_tag[] = "whisper";

inline transcribe_status read_whisper_hparams(const gguf_context * gguf, WhisperHParams & hp) {
    return whisper_graph::read_hparams(gguf, k_kv_prefix, k_family_tag, hp);
}

inline transcribe_status build_whisper_weights(ggml_context *         ctx_meta,
                                               const WhisperHParams & hp,
                                               WhisperWeights &       weights) {
    return whisper_graph::build_weights(ctx_meta, hp, k_family_tag, weights);
}

inline transcribe_status install_mel_from_buffers(const WhisperHParams &                   hp,
                                                  std::vector<float>                       filterbank,
                                                  std::vector<float>                       window,
                                                  std::optional<transcribe::MelFrontend> & out_mel) {
    return whisper_graph::install_mel_from_buffers(hp, k_family_tag, std::move(filterbank), std::move(window),
                                                   out_mel);
}

}  // namespace transcribe::whisper
