// arch/crisperwhisper/weights.h - crisperwhisper family binding onto the
// shared Whisper encoder-decoder graph, plus this family's own decode
// contract. INTERNAL to src/arch/crisperwhisper/.
//
// The graph, tensor catalog, weight slots and KV cache come from
// src/whisper_graph/ unchanged — CrisperWhisper 2.0 is a Whisper fine-tune
// and its safetensors layout is identical (see docs/porting/families/
// crisperwhisper.md). This header binds the family's KV prefix
// (stt.crisperwhisper.*) and adds CwDecodeContract, which holds everything
// the reference does ABOVE the graph and whisper has no analogue for.
//
// Nothing here hardcodes a token id. The `turbo` variant shifts every special
// token above 50357 by +1, so every id is read from GGUF KV that the Stage-3
// converter resolved from tokenizer.json by literal content.

#pragma once

#include "whisper_graph/kv_cache.h"
#include "whisper_graph/whisper_graph.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace transcribe::crisperwhisper {

// Shared graph types, under this family's spelling.
using CwHParams  = whisper_graph::HParams;
using CwWeights  = whisper_graph::Weights;
using CwKvCache  = whisper_graph::KvCache;
using CwEncOut   = whisper_graph::EncOut;
using CwEncBlock = whisper_graph::EncBlock;
using CwDecBlock = whisper_graph::DecBlock;

using whisper_graph::enc_out_init;
using whisper_graph::k_cross_kv_pad;
using whisper_graph::kv_cache_init;
using whisper_graph::kv_cache_init_batched;
using whisper_graph::kv_pad_self_attn;

inline constexpr const char k_kv_prefix[]  = "stt.crisperwhisper";
inline constexpr const char k_family_tag[] = "crisperwhisper";

// Reference: crisperwhisper 2.0.2. transcribe(mode=...) defaults to verbatim.
enum class CwMode { Verbatim, Intended };

// Everything the reference does above the graph. Read from
// stt.crisperwhisper.* by read_cw_contract().
struct CwDecodeContract {
    // Mode tags, in emission order. The decoder prompt opens with the whole
    // block: "[verbatim_1][verbatim_2]...[verbatim_5]" (prompt.py
    // PromptBuilder._mode_tags_text). Tag COUNT is a reference constructor
    // argument, so read the list rather than assuming 5.
    std::vector<int32_t> verbatim_tag_ids;
    std::vector<int32_t> intended_tag_ids;

    // Prompt markers (prompt.py PROMPT_MARKER_TOKENS).
    int32_t vtx_id  = -1;  // <vtx>   verbatimize open   (not in scope)
    int32_t evtx_id = -1;  // <evtx>  verbatimize close  (not in scope)
    int32_t htx_id  = -1;  // <htx>   hotwords open      (Pro models only)
    int32_t ehtx_id = -1;  // <ehtx>  hotwords close     (Pro models only)
    int32_t ctx_id  = -1;  // <ctx>   long-form context open
    int32_t ectx_id = -1;  // <ectx>  long-form context close

    // The 15 vocal-event tokens ([UM], [UH], [laughter], ...). All carry
    // `special: false` upstream and are INTENDED OUTPUT in verbatim mode:
    // they must survive detokenization. Held so the runtime can tell them
    // apart from prompt artifacts without a hardcoded id range.
    std::vector<int32_t> event_ids;

    // Mode tags + the six markers. The reference strips these by explicit id
    // (prompt.py strip_prompt_artifacts), NOT via the tokenizer's
    // special-token machinery, because none of them is flagged special.
    std::vector<int32_t> prompt_artifact_ids;

    // Word timing: (layer, head) pairs whose cross-attention the Viterbi
    // aligner averages. Flattened 2N in the GGUF. Per-variant — turbo has 4
    // decoder layers, so its pairs are not interchangeable with the others'.
    std::vector<std::pair<int32_t, int32_t>> alignment_heads;

    // Long-form (<ctx> conditional continuation). Defaults mirror
    // crisperwhisper/longform/base.py::LongformConfig.
    float   longform_chunk_duration = 30.0f;
    float   longform_stride         = 26.0f;
    int32_t longform_context_words  = 12;
    int32_t longform_drop_words     = 2;

    // Contract shape flags. Both are true for every shipped 2.0 checkpoint;
    // carried so a future variant that changes them fails loudly at load
    // instead of silently decoding with the wrong prompt.
    bool mode_tags_before_prefix = true;
    bool always_no_timestamps    = true;

    CwMode default_mode = CwMode::Verbatim;

    const std::vector<int32_t> & tags_for(CwMode m) const {
        return m == CwMode::Verbatim ? verbatim_tag_ids : intended_tag_ids;
    }

    bool is_prompt_artifact(int32_t id) const {
        for (const int32_t a : prompt_artifact_ids) {
            if (a == id) {
                return true;
            }
        }
        return false;
    }
};

inline transcribe_status read_cw_hparams(const gguf_context * gguf, CwHParams & hp) {
    return whisper_graph::read_hparams(gguf, k_kv_prefix, k_family_tag, hp);
}

inline transcribe_status build_cw_weights(ggml_context * ctx_meta, const CwHParams & hp, CwWeights & weights) {
    return whisper_graph::build_weights(ctx_meta, hp, k_family_tag, weights);
}

inline transcribe_status install_cw_mel(const CwHParams &                        hp,
                                        std::vector<float>                       filterbank,
                                        std::vector<float>                       window,
                                        std::optional<transcribe::MelFrontend> & out_mel) {
    return whisper_graph::install_mel_from_buffers(hp, k_family_tag, std::move(filterbank), std::move(window), out_mel);
}

// Read the stt.crisperwhisper.* decode contract and validate it against the
// decoder shape in `hp` (alignment-head bounds, non-empty tag blocks).
transcribe_status read_cw_contract(const gguf_context * gguf, const CwHParams & hp, CwDecodeContract & out);

}  // namespace transcribe::crisperwhisper
