// arch/crisperwhisper/weights.cpp - read the stt.crisperwhisper.* decode
// contract. The graph hparams and tensor catalog come from
// src/whisper_graph/; only the family-specific contract is read here.

#include "weights.h"

#include "gguf.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"

#include <cstring>

namespace transcribe::crisperwhisper {

namespace {

constexpr const char * kTag = k_family_tag;

// Required int32-array KV. Absent or wrong type is fatal: every array in the
// contract is load-bearing (a missing mode-tag block means we cannot build a
// prompt at all).
transcribe_status read_required_i32_array(const gguf_context * gguf, const char * key, std::vector<int32_t> & out) {
    const auto st = transcribe::read_int32_array_kv(gguf, key, out);
    if (st == transcribe::KvResult::Ok) {
        return TRANSCRIBE_OK;
    }
    if (st == transcribe::KvResult::Absent) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: required KV %s is missing", kTag, key);
    } else {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: KV %s has wrong type (expected int32 array)", kTag, key);
    }
    return TRANSCRIBE_ERR_GGUF;
}

// Required token-id KV.
transcribe_status read_required_token(const gguf_context * gguf, const char * key, int32_t & out) {
    uint32_t   v  = 0;
    const auto st = transcribe::read_uint32_kv(gguf, key, v);
    if (st == transcribe::KvResult::Ok) {
        out = static_cast<int32_t>(v);
        return TRANSCRIBE_OK;
    }
    log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: required token-id KV %s is missing or ill-typed", kTag, key);
    return TRANSCRIBE_ERR_GGUF;
}

}  // namespace

transcribe_status read_cw_contract(const gguf_context * gguf, const CwHParams & hp, CwDecodeContract & out) {
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // ---- mode tags ----
    if (const auto st =
            read_required_i32_array(gguf, "stt.crisperwhisper.mode.verbatim_token_ids", out.verbatim_tag_ids);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (const auto st =
            read_required_i32_array(gguf, "stt.crisperwhisper.mode.intended_token_ids", out.intended_tag_ids);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (out.verbatim_tag_ids.empty() || out.intended_tag_ids.empty()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: mode tag blocks must be non-empty (verbatim=%zu, intended=%zu)", kTag,
                out.verbatim_tag_ids.size(), out.intended_tag_ids.size());
        return TRANSCRIBE_ERR_GGUF;
    }

    // ---- prompt markers ----
    struct MarkerSlot {
        const char * key;
        int32_t *    slot;
    };

    const MarkerSlot markers[] = {
        { "stt.crisperwhisper.marker.verbatimize_start_token_id", &out.vtx_id  },
        { "stt.crisperwhisper.marker.verbatimize_end_token_id",   &out.evtx_id },
        { "stt.crisperwhisper.marker.hotword_start_token_id",     &out.htx_id  },
        { "stt.crisperwhisper.marker.hotword_end_token_id",       &out.ehtx_id },
        { "stt.crisperwhisper.marker.context_start_token_id",     &out.ctx_id  },
        { "stt.crisperwhisper.marker.context_end_token_id",       &out.ectx_id },
    };
    for (const auto & m : markers) {
        if (const auto st = read_required_token(gguf, m.key, *m.slot); st != TRANSCRIBE_OK) {
            return st;
        }
    }

    // ---- event + artifact id lists ----
    if (const auto st = read_required_i32_array(gguf, "stt.crisperwhisper.event_token_ids", out.event_ids);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (const auto st =
            read_required_i32_array(gguf, "stt.crisperwhisper.prompt_artifact_token_ids", out.prompt_artifact_ids);
        st != TRANSCRIBE_OK) {
        return st;
    }

    // An event token in the artifact list would silently delete verbatim
    // output — the whole point of the family. Cheap to check, fatal if wrong.
    for (const int32_t ev : out.event_ids) {
        if (out.is_prompt_artifact(ev)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "%s: event token %d is also listed as a prompt artifact; "
                    "verbatim output would be stripped",
                    kTag, ev);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // ---- word timing alignment heads (flattened (layer, head) pairs) ----
    std::vector<int32_t> heads_flat;
    if (const auto st = read_required_i32_array(gguf, "stt.crisperwhisper.word_timing.alignment_heads", heads_flat);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (heads_flat.empty() || (heads_flat.size() % 2) != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: alignment_heads must be a non-empty flattened pair list (got %zu)",
                kTag, heads_flat.size());
        return TRANSCRIBE_ERR_GGUF;
    }
    out.alignment_heads.clear();
    out.alignment_heads.reserve(heads_flat.size() / 2);
    for (size_t i = 0; i + 1 < heads_flat.size(); i += 2) {
        const int32_t layer = heads_flat[i];
        const int32_t head  = heads_flat[i + 1];
        if (layer < 0 || layer >= hp.dec_n_layers || head < 0 || head >= hp.dec_n_heads) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "%s: alignment head (%d, %d) out of range for %d decoder layers x %d heads", kTag, layer, head,
                    hp.dec_n_layers, hp.dec_n_heads);
            return TRANSCRIBE_ERR_GGUF;
        }
        out.alignment_heads.emplace_back(layer, head);
    }

    // ---- long-form geometry ----
    if (const auto st = transcribe::read_required_f32_kv(gguf, "stt.crisperwhisper.longform.chunk_duration", kTag,
                                                         out.longform_chunk_duration);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (const auto st =
            transcribe::read_required_f32_kv(gguf, "stt.crisperwhisper.longform.stride", kTag, out.longform_stride);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (const auto st = transcribe::read_required_u32_kv(gguf, "stt.crisperwhisper.longform.context_words", kTag,
                                                         out.longform_context_words);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (const auto st = transcribe::read_required_u32_kv(gguf, "stt.crisperwhisper.longform.drop_words", kTag,
                                                         out.longform_drop_words);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (!(out.longform_stride > 0.0f) || !(out.longform_chunk_duration > 0.0f) ||
        out.longform_stride > out.longform_chunk_duration) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: long-form stride (%g) must be in (0, chunk_duration=%g]", kTag,
                static_cast<double>(out.longform_stride), static_cast<double>(out.longform_chunk_duration));
        return TRANSCRIBE_ERR_GGUF;
    }

    // ---- contract shape flags ----
    if (const auto st = transcribe::read_optional_bool_kv(gguf, "stt.crisperwhisper.mode_tags_before_prefix", kTag,
                                                          true, out.mode_tags_before_prefix);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (const auto st = transcribe::read_optional_bool_kv(gguf, "stt.crisperwhisper.always_no_timestamps", kTag, true,
                                                          out.always_no_timestamps);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (!out.mode_tags_before_prefix) {
        // The reference emits encode(tags) + [sot, lang, task, notimestamps]
        // with no <|startofprev|> wrapper. A checkpoint claiming otherwise
        // needs a prompt builder we have not written.
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "%s: stt.crisperwhisper.mode_tags_before_prefix=false is not supported "
                "(only the mode-tags-then-Whisper-prefix contract is implemented)",
                kTag);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!out.always_no_timestamps) {
        // Timestamp-token decoding is a whole decode contract this family
        // does not implement (no timestamp rules, no segment assembly).
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "%s: stt.crisperwhisper.always_no_timestamps=false is not supported "
                "(the runtime never decodes timestamp tokens)",
                kTag);
        return TRANSCRIBE_ERR_GGUF;
    }

    // ---- default mode ----
    std::string mode_str;
    if (const auto st =
            transcribe::read_optional_string_kv(gguf, "stt.crisperwhisper.mode.default", kTag, "verbatim", mode_str);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (mode_str == "verbatim") {
        out.default_mode = CwMode::Verbatim;
    } else if (mode_str == "intended") {
        out.default_mode = CwMode::Intended;
    } else {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: unknown stt.crisperwhisper.mode.default \"%s\"", kTag,
                mode_str.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }

    return TRANSCRIBE_OK;
}

}  // namespace transcribe::crisperwhisper
