// arch/cohere/model.cpp - Cohere ASR family handler.
//
// Load/init_context/run lifecycle for the Cohere ASR encoder-decoder
// model. The encoder is a conformer (identical to Parakeet except FFN
// has bias). The decoder is an autoregressive Transformer that runs
// on the ggml graph (not host-side like Parakeet's LSTM).

#include "cohere.h"
#include "decoder.h"
#include "encoder.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "transcribe-arch.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-env.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-mel.h"
#include "transcribe-meta.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace transcribe::cohere {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model, CohereModel>);
static_assert(std::is_base_of_v<transcribe_session, CohereSession>);

CohereSession::~CohereSession() {
    kv_cache.free();
    if (sched != nullptr) {
        safe_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
    encoder_out = nullptr;
}

bool kv_cache_init(CohereKvCache & cache,
                   ggml_backend_t  backend,
                   int             n_ctx,
                   int             T_enc,
                   int             n_state,
                   int             n_layer,
                   ggml_type       kv_type) {
    if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "cohere kv_cache: unsupported kv_type=%d "
                "(only F16/F32)",
                static_cast<int>(kv_type));
        return false;
    }

    // Allocate 4 tensors: self K, self V, cross K, cross V.
    const size_t ctx_size = 4 * ggml_tensor_overhead() + 256;

    ggml_init_params params{};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    cache.ctx = ggml_init(params);
    if (cache.ctx == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere kv_cache: ggml_init failed");
        return false;
    }

    const int64_t self_elements  = static_cast<int64_t>(n_state) * n_layer * n_ctx;
    const int64_t cross_elements = static_cast<int64_t>(n_state) * n_layer * T_enc;

    cache.self_k  = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
    cache.self_v  = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
    cache.cross_k = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);
    cache.cross_v = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);

    ggml_set_name(cache.self_k, "kv_self_k");
    ggml_set_name(cache.self_v, "kv_self_v");
    ggml_set_name(cache.cross_k, "kv_cross_k");
    ggml_set_name(cache.cross_v, "kv_cross_v");

    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    if (cache.buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere kv_cache: buffer alloc failed");
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
        return false;
    }

    ggml_backend_buffer_clear(cache.buffer, 0);

    cache.n_ctx           = n_ctx;
    cache.T_enc           = T_enc;
    cache.n               = 0;
    cache.head            = 0;
    cache.cross_populated = false;

    const size_t total_bytes =
        ggml_nbytes(cache.self_k) + ggml_nbytes(cache.self_v) + ggml_nbytes(cache.cross_k) + ggml_nbytes(cache.cross_v);
    log_msg(TRANSCRIBE_LOG_LEVEL_INFO,
            "cohere kv_cache: allocated %.1f MB (%s) "
            "(self: %d session x %d layers, cross: %d T_enc x %d layers)",
            static_cast<double>(total_bytes) / (1024.0 * 1024.0), ggml_type_name(kv_type), n_ctx, n_layer, T_enc,
            n_layer);

    return true;
}

bool kv_cache_init_batched(CohereKvCache & cache,
                           ggml_backend_t  backend,
                           int             n_ctx,
                           int             T_enc,
                           int             n_state,
                           int             n_layer,
                           int             n_batch,
                           ggml_type       kv_type) {
    if (n_batch <= 1) {
        if (!kv_cache_init(cache, backend, n_ctx, T_enc, n_state, n_layer, kv_type)) {
            return false;
        }
        cache.n_batch = 1;
        return true;
    }
    if (kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_F32) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere kv_cache(batched): unsupported kv_type");
        return false;
    }

    const size_t     ctx_size = 4 * ggml_tensor_overhead() + 256;
    ggml_init_params params{};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    cache.ctx         = ggml_init(params);
    if (cache.ctx == nullptr) {
        return false;
    }

    const int64_t self_elements  = static_cast<int64_t>(n_state) * n_layer * n_ctx * n_batch;
    const int64_t cross_elements = static_cast<int64_t>(n_state) * n_layer * T_enc * n_batch;

    cache.self_k  = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
    cache.self_v  = ggml_new_tensor_1d(cache.ctx, kv_type, self_elements);
    cache.cross_k = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);
    cache.cross_v = ggml_new_tensor_1d(cache.ctx, kv_type, cross_elements);
    ggml_set_name(cache.self_k, "kv_self_k");
    ggml_set_name(cache.self_v, "kv_self_v");
    ggml_set_name(cache.cross_k, "kv_cross_k");
    ggml_set_name(cache.cross_v, "kv_cross_v");

    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    if (cache.buffer == nullptr) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(cache.buffer, 0);

    cache.n_ctx           = n_ctx;
    cache.T_enc           = T_enc;
    cache.n               = 0;
    cache.head            = 0;
    cache.n_batch         = n_batch;
    cache.cross_populated = false;
    return true;
}

CohereModel::~CohereModel() {
    if (bn_fused_ctx != nullptr) {
        ggml_free(bn_fused_ctx);
        bn_fused_ctx = nullptr;
    }
    if (bn_fused_buffer != nullptr) {
        safe_buffer_free(bn_fused_buffer);
        bn_fused_buffer = nullptr;
    }
    if (conv_pw_f32_ctx != nullptr) {
        ggml_free(conv_pw_f32_ctx);
        conv_pw_f32_ctx = nullptr;
    }
    if (conv_pw_f32_buffer != nullptr) {
        safe_buffer_free(conv_pw_f32_buffer);
        conv_pw_f32_buffer = nullptr;
    }
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (backend_buffer != nullptr) {
        safe_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary      = nullptr;
    plan.primary_kind = transcribe::BackendKind::Unknown;
}

// Long-form windowing (see run() below for the rationale).
//
// Cohere Transcribe was trained on clips of at most 35 s -- `max_audio_clip_s`
// in the HF feature-extractor config, recorded in the porting intake. That
// value is not written to the GGUF today, so the trained window lives here as
// a family constant; emitting it as `stt.cohere.max_audio_clip_s` and reading
// it in read_cohere_hparams() (with this as the fallback for already-published
// GGUFs) would be the tidier follow-up.
constexpr double kTrainedClipSeconds = 35.0;

// Fractions of a window searched backwards for a pause to cut on. The wider
// pass is a fallback when the narrow one finds only continuous speech.
constexpr double kSplitSearchFraction     = 0.25;
constexpr double kSplitSearchFractionWide = 0.50;

// A frame counts as a pause only if it is this much quieter than the mean of
// the searched region. A minimum always exists; a pause does not.
constexpr float kPauseRatio = 0.30f;

// Search back from `target` for a genuine pause to cut on: the quietest 20 ms
// frame within the trailing `search` samples, accepted only when it is clearly
// quieter than that region's own mean amplitude.
//
// Returns the cut offset, or -1 when the region is continuous speech and no
// pause qualifies -- the caller decides what to do rather than getting a
// silently arbitrary cut.
//
// A pause is preferred but not required: windows overlap by kOverlapMs, so a
// cut that lands mid-word still leaves that word whole in the next window, and
// token_seam() removes the duplicate when the two are joined. Cutting on a
// pause simply makes the seam easier to find.
int pick_split(const float * pcm, int n_samples, int target, int search) {
    constexpr int frame = 320;  // 20 ms @ 16 kHz
    if (target >= n_samples || search < frame * 2 || target < frame * 2) {
        return -1;
    }

    const int lo   = std::max(frame, target - search);
    const int step = frame / 2;

    double sum_amp  = 0.0;
    int    n_frames = 0;
    int    best_at  = -1;
    float  best_amp = std::numeric_limits<float>::max();

    for (int at = lo; at + frame <= target; at += step) {
        float sum = 0.0f;
        for (int i = 0; i < frame; ++i) {
            sum += std::fabs(pcm[at + i]);
        }
        const float amp = sum / static_cast<float>(frame);

        sum_amp += amp;
        ++n_frames;

        if (amp < best_amp) {
            best_amp = amp;
            best_at  = at + frame / 2;  // cut in the middle of the pause
        }
    }

    if (n_frames == 0 || best_at < 0) {
        return -1;
    }

    // Reject a "quietest" frame that is really just ordinary speech.
    const float mean_amp = static_cast<float>(sum_amp / n_frames);
    if (best_amp > mean_amp * kPauseRatio) {
        return -1;
    }
    return best_at;
}

// Overlap carried between consecutive windows. A boundary that lands in
// continuous speech splits a word; repeating a second of audio on both sides
// means the word survives whole in at least one window, and token_seam() drops
// the duplicate when the windows are joined.
constexpr int kOverlapMs = 1000;

// Seam search widths, in encoder frames' worth of tokens, derived from the
// overlap rather than hardcoded.
//
// The two sides are deliberately ASYMMETRIC, and the small one is the load-
// bearing part. Only the first ~1 s of a new window can legitimately duplicate
// the previous one, so `current_search` must not reach past the overlap: a
// wider window lets a genuinely repeated sentence match the accumulated tail
// and be deleted as if it were overlap. `previous_search` can be looser because
// the at-previous-end rule already constrains where a match may finish.
//
// Mirrors the canary long-form path (PR #112), which computes the same two
// widths from its own overlap.
int seam_search_previous(const CohereHParams & hp) {
    const int frame_ms =
        hp.fe_sample_rate > 0 ? hp.fe_hop_length * hp.enc_subsampling_factor * 1000 / hp.fe_sample_rate : 0;
    const int delay_frames = std::max(1, frame_ms > 0 ? kOverlapMs / frame_ms : 12);
    return delay_frames * 2;
}

int seam_search_current(const CohereHParams & hp) {
    const int frame_ms =
        hp.fe_sample_rate > 0 ? hp.fe_hop_length * hp.enc_subsampling_factor * 1000 / hp.fe_sample_rate : 0;
    const int delay_frames = std::max(1, frame_ms > 0 ? kOverlapMs / frame_ms : 12);
    return std::max(1, delay_frames * 3 / 5);
}

// Find where to join two consecutive windows by matching the tail of the
// accumulated hypothesis against the head of the new window.
//
// The match is only accepted when it runs to the *end* of the previous
// hypothesis. A match that stops earlier means the repeated text sits in the
// interior, which is ambiguous -- the speaker may genuinely have repeated the
// phrase, and trimming it would delete real speech. A single-token match is
// accepted only at the very start of the new window, where it cannot be
// coincidence.
//
// Ported from canary_token_seam() in the canary long-form path (PR #112) so the
// two families stitch identically. Timestamp-free by construction, which is
// what makes it usable here: cohere advertises max_timestamp_kind == NONE and
// has no alignment data to fall back on.
TokenSeam token_seam(const std::vector<int> & previous,
                     const std::vector<int> & current,
                     int                      previous_search,
                     int                      current_search) {
    TokenSeam seam{ static_cast<int>(previous.size()), 0, false };
    if (previous.empty() || current.empty() || previous_search <= 0 || current_search <= 0) {
        return seam;
    }

    const int previous_begin = std::max(0, static_cast<int>(previous.size()) - previous_search);
    const int current_end    = std::min(static_cast<int>(current.size()), current_search);

    int best_length   = 0;
    int best_curr_end = 0;

    for (int i = previous_begin; i < static_cast<int>(previous.size()); ++i) {
        for (int j = 0; j < current_end; ++j) {
            int length = 0;
            while (i + length < static_cast<int>(previous.size()) && j + length < current_end &&
                   previous[i + length] == current[j + length]) {
                ++length;
            }
            const bool at_previous_end = (i + length) == static_cast<int>(previous.size());
            if (!at_previous_end || !(length >= 2 || (length == 1 && j == 0))) {
                continue;
            }
            if (length > best_length) {
                best_length   = length;
                best_curr_end = j + length;
            }
        }
    }

    // A seam that would swallow the entire new window is never overlap -- the
    // window covers ~34 s of fresh audio and the overlap is 1 s. Treat it as a
    // failed match and append whole; repeating a phrase is recoverable, dropping
    // a window is the exact defect this path exists to prevent.
    if (best_length > 0 && best_curr_end < static_cast<int>(current.size())) {
        seam.current_skip = best_curr_end;
        seam.matched      = true;
    }
    return seam;
}

namespace {

constexpr float kBnEps = 1e-5f;

// Input-length contract (canonical reference; see docs/input-limits.md).
// Cohere ASR has THREE bounds, only one of which a caller ever sees:
//   (a) TRAINED CLIP SPAN — kTrainedClipSeconds, the window the model was
//       trained on. run() windows longer audio to it, so this is what actually
//       shapes behaviour. Because run() handles it, caps.max_audio_ms is 0
//       (no practical limit) and the family sits in bucket 1.
//   (b) ENCODER pos-emb table — enc_pos_emb_max_len. T_enc must stay within
//       the trained span or the runtime table aliases. No longer the public
//       limit: each window is ~35 s, far under it, so the up-front gate in
//       run_window() is now a per-window assertion rather than a caller-facing
//       bound.
//   (c) DECODER self-KV — dec_max_seq and a 512 max-new-tokens cap. Bounds the
//       *output* per window: an over-budget transcript is kept as a partial
//       result and flagged via transcribe_was_truncated(), not rejected.

// Predicted encoder frame count T_enc for a given mel frame count. The
// FastConformer pre-encode downsamples time via three stride-2, kernel-3,
// pad-(k-1)/2 convs, each mapping T_in -> floor((T_in - 1)/2) + 1. We fold
// that exact recurrence here so the prediction matches the graph's T_enc
// (net result floor(T_mel/8)). Pure host-side count; no graph math.
int cohere_predict_t_enc(int mel_n_frames, int subsampling_factor) {
    if (mel_n_frames <= 0 || subsampling_factor <= 0) {
        return 0;
    }
    // Derive the stride-2 stage count from the factor (log2); fall back to
    // floor(T_mel / factor) if it is not a power of two.
    int stages = 0;
    for (int f = subsampling_factor; f > 1; f >>= 1) {
        ++stages;
    }
    if ((1 << stages) != subsampling_factor) {
        return mel_n_frames / subsampling_factor;
    }
    int t = mel_n_frames;
    for (int s = 0; s < stages; ++s) {
        t = (t - 1) / 2 + 1;  // floor((T_in - 1)/2) + 1, k=3 s=2 p=1
        if (t <= 0) {
            return 0;
        }
    }
    return t;
}

// transcribe_capabilities::max_audio_ms: longest audio whose T_enc still
// fits the trained pos-emb span enc_pos_emb_max_len. Inverts the rate:
//   ms = T_enc * subsampling_factor * hop_length * 1000 / sr
// at T_enc == enc_pos_emb_max_len. Returns 0 (unknown) if any rate hparam
// is missing, so a misconfigured model is never advertised with a wrong value.
int64_t cohere_max_audio_ms(const CohereHParams & hp) {
    if (hp.enc_pos_emb_max_len <= 0 || hp.enc_subsampling_factor <= 0 || hp.fe_hop_length <= 0 ||
        hp.fe_sample_rate <= 0) {
        return 0;
    }
    const int64_t mel_frames = static_cast<int64_t>(hp.enc_pos_emb_max_len) * hp.enc_subsampling_factor;
    return mel_frames * hp.fe_hop_length * 1000 / hp.fe_sample_rate;
}

// Effective decoder self-KV ceiling, in tokens: the model's trained
// dec_max_seq, optionally lowered — never raised — by the caller's session
// n_ctx knob. Used to size / clamp the autoregressive self-KV cache.
int cohere_dec_ctx_ceiling(int32_t n_ctx_knob, const CohereHParams & hp) {
    int ceiling = hp.dec_max_seq > 0 ? hp.dec_max_seq : 1024;
    if (n_ctx_knob > 0 && n_ctx_knob < ceiling) {
        ceiling = n_ctx_knob;
    }
    return ceiling;
}

transcribe_status fuse_batch_norm(CohereModel & m) {
    const size_t n_blocks = m.weights.blocks.size();
    if (n_blocks == 0) {
        return TRANSCRIBE_OK;
    }

    const int64_t d            = m.hparams.enc_d_model;
    const size_t  tensor_bytes = static_cast<size_t>(d) * sizeof(float);

    const size_t     ctx_size = n_blocks * 2 * ggml_tensor_overhead() + 256;
    ggml_init_params params   = { ctx_size, nullptr, true };
    m.bn_fused_ctx            = ggml_init(params);
    if (m.bn_fused_ctx == nullptr) {
        return TRANSCRIBE_ERR_BACKEND;
    }

    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b              = m.weights.blocks[i];
        b.conv_bn_fused_scale = ggml_new_tensor_1d(m.bn_fused_ctx, GGML_TYPE_F32, d);
        b.conv_bn_fused_bias  = ggml_new_tensor_1d(m.bn_fused_ctx, GGML_TYPE_F32, d);
    }

    m.bn_fused_buffer = ggml_backend_alloc_ctx_tensors(m.bn_fused_ctx, m.plan.scheduler_list.back());
    if (m.bn_fused_buffer == nullptr) {
        return TRANSCRIBE_ERR_BACKEND;
    }

    std::vector<float> bn_w(d), bn_b(d), rm(d), rv(d);
    std::vector<float> fused_s(d), fused_b(d);

    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.weights.blocks[i];
        ggml_backend_tensor_get(b.conv_bn_w, bn_w.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_b, bn_b.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_rm, rm.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_rv, rv.data(), 0, tensor_bytes);

        for (int64_t c = 0; c < d; ++c) {
            const float s = bn_w[c] / std::sqrt(rv[c] + kBnEps);
            fused_s[c]    = s;
            fused_b[c]    = bn_b[c] - rm[c] * s;
        }

        ggml_backend_tensor_set(b.conv_bn_fused_scale, fused_s.data(), 0, tensor_bytes);
        ggml_backend_tensor_set(b.conv_bn_fused_bias, fused_b.data(), 0, tensor_bytes);
    }

    return TRANSCRIBE_OK;
}

// Fold each encoder layer's Q bias into its pos_bias_u / pos_bias_v at
// load time. In rel_pos_mhsa:
//   q_u = (W_q x + q_b) + pos_u = W_q x + (q_b + pos_u)
//   q_v = (W_q x + q_b) + pos_v = W_q x + (q_b + pos_v)
// so pre-adding q_b into pos_u/pos_v is identical and drops the explicit
// `q = q + q_b` graph op. We null attn_q_b after so the builder skips the add.
transcribe_status fuse_encoder_q_bias(CohereModel & m) {
    const size_t n_blocks = m.weights.blocks.size();
    if (n_blocks == 0) {
        return TRANSCRIBE_OK;
    }

    const int64_t d_model  = m.hparams.enc_d_model;
    const int64_t n_heads  = m.hparams.enc_n_heads;
    const int64_t head_dim = n_heads > 0 ? d_model / n_heads : 0;

    if (head_dim * n_heads != d_model) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "cohere: d_model (%lld) != head_dim*n_heads; "
                "skipping encoder Q-bias fusion",
                (long long) d_model);
        return TRANSCRIBE_OK;
    }

    const size_t       nbytes = static_cast<size_t>(d_model) * sizeof(float);
    std::vector<float> q_bias(d_model);
    std::vector<float> pos_u(d_model);
    std::vector<float> pos_v(d_model);

    size_t fused = 0;
    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.weights.blocks[i];
        if (b.attn_q_b == nullptr || b.attn_pos_u == nullptr || b.attn_pos_v == nullptr) {
            continue;
        }

        ggml_backend_tensor_get(b.attn_q_b, q_bias.data(), 0, nbytes);
        ggml_backend_tensor_get(b.attn_pos_u, pos_u.data(), 0, nbytes);
        ggml_backend_tensor_get(b.attn_pos_v, pos_v.data(), 0, nbytes);

        // pos_u/v [head_dim, n_heads] and q_b [d_model=head_dim*n_heads]
        // share the same linear element layout, so a flat add is correct.
        for (int64_t j = 0; j < d_model; ++j) {
            pos_u[j] += q_bias[j];
            pos_v[j] += q_bias[j];
        }

        ggml_backend_tensor_set(b.attn_pos_u, pos_u.data(), 0, nbytes);
        ggml_backend_tensor_set(b.attn_pos_v, pos_v.data(), 0, nbytes);

        // Drop the reference so the graph builder skips the add.
        // Tensor memory stays allocated inside ctx_meta.
        b.attn_q_b = nullptr;
        ++fused;
    }

    if (fused > 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "cohere: fused Q bias into pos_u/pos_v for %zu encoder blocks", fused);
    }
    return TRANSCRIBE_OK;
}

// On a CPU primary backend, dequantize the conformer 1×1 pointwise conv
// weights (pw1, pw2) from F16 back to F32: Zen 2 (and anything else without
// native F16 compute) pays an F16->F32 upconvert per matmul that outweighs
// the bandwidth win. GPU backends skip this and keep the F16 weights.
transcribe_status promote_conv_pw_to_f32_on_cpu(CohereModel & m) {
    std::vector<load_common::ConvPwF32Slot> slots;
    slots.reserve(m.weights.blocks.size() * 2);
    for (auto & b : m.weights.blocks) {
        if (b.conv_pw1_w != nullptr && b.conv_pw1_w->type == GGML_TYPE_F16) {
            slots.push_back({ &b.conv_pw1_w, b.conv_pw1_w });
        }
        if (b.conv_pw2_w != nullptr && b.conv_pw2_w->type == GGML_TYPE_F16) {
            slots.push_back({ &b.conv_pw2_w, b.conv_pw2_w });
        }
    }
    return load_common::promote_conv_pw_f16_to_f32_on_cpu(m.plan, slots, "cohere", &m.conv_pw_f32_ctx,
                                                          &m.conv_pw_f32_buffer);
}

constexpr const char k_default_variant[] = "cohere-asr";

// Forward declarations for the Arch trait below.
extern transcribe_status load(Loader &, const transcribe_model_load_params *, transcribe_model **);
extern transcribe_status init_context(transcribe_model *, const transcribe_session_params *, transcribe_session **);
extern transcribe_status run(transcribe_session *, const float *, int, const transcribe_run_params *);

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    // Backend and device selection are resolved below via load_common.

    const int64_t t_load_start = ggml_time_us();

    auto m       = std::make_unique<CohereModel>();
    m->arch      = &arch;
    m->t_load_us = 0;

    if (loader.variant().empty()) {
        m->variant = k_default_variant;
    } else {
        m->variant = loader.variant();
    }
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) {
        return st;
    }

    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m); st != TRANSCRIBE_OK) {
        return st;
    }

    if (const transcribe_status st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) {
        return st;
    }

    if (const transcribe_status st = read_cohere_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }

    // No practical input-length limit: run() windows long audio to the trained
    // clip span (kTrainedClipSeconds) before it ever reaches the encoder, so the
    // encoder pos-emb table is never the binding bound for a caller. This puts
    // cohere in the "chunked / unbounded" bucket of docs/input-limits.md,
    // alongside whisper and parakeet. cohere_max_audio_ms() is still the
    // per-window ceiling and is asserted inside run_window().
    m->caps.max_audio_ms = 0;

    // Basis for the session-level limits query (transcribe_session_get_limits).
    // The LimitsBasis is a single context cap, filled from the DECODER side
    // (dec_max_seq) so effective_n_ctx and max_kv_bytes are exact. It bounds the
    // OUTPUT transcript per window, which is why lowering n_ctx shrinks the
    // per-window generation budget without changing how much audio the family
    // accepts.
    if (m->hparams.dec_max_seq > 0 && m->hparams.enc_subsampling_factor > 0 && m->hparams.fe_hop_length > 0 &&
        m->hparams.fe_sample_rate > 0) {
        m->limits.has_context_cap        = true;
        // Take the audio bound straight from caps (0 = unbounded) rather than
        // deriving it from the decoder ceiling: run() windows the input, so the
        // amount of audio accepted does not shrink with n_ctx. Reporting
        // effective_max_audio_ms == 0 keeps the session query consistent with
        // caps.max_audio_ms.
        m->limits.audio_from_caps        = true;
        m->limits.model_max_ctx          = m->hparams.dec_max_seq;
        // Fixed control-token preamble (see run()'s prompt_pieces). Audio is
        // in cross-KV, so there is no audio-token overhead here.
        m->limits.prompt_overhead        = 10;
        m->limits.gen_reserve            = 512;  // max-new-tokens cap in run()
        // ms-per-audio-token = subsampling_factor * hop_length * 1000 / sr.
        m->limits.ms_per_audio_token     = static_cast<double>(m->hparams.enc_subsampling_factor) *
                                           m->hparams.fe_hop_length * 1000.0 / m->hparams.fe_sample_rate;
        // Self-KV stores dec_hidden per layer, two tensors (K, V); no GQA.
        m->limits.kv_elems_per_ctx_token = (int64_t) m->hparams.dec_hidden * m->hparams.dec_n_layers * 2;
    }

    m->hparams.vocab_size   = m->tok.n_tokens();
    m->hparams.bos_token_id = m->tok.bos_id();
    m->hparams.eos_token_id = m->tok.eos_id();

    // Hard-fail at load time if the tokenizer supplied no EOS token id —
    // a missing EOS is a GGUF-builder bug that should surface at conversion.
    if (m->hparams.eos_token_id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "cohere: GGUF tokenizer has no eos_token_id -- "
                "regenerate with an up-to-date converter");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Mel frontend.
    {
        transcribe::MelConfig cfg{};
        cfg.sample_rate  = m->hparams.fe_sample_rate;
        cfg.num_mels     = m->hparams.fe_num_mels;
        cfg.n_fft        = m->hparams.fe_n_fft;
        cfg.win_length   = m->hparams.fe_win_length;
        cfg.hop_length   = m->hparams.fe_hop_length;
        cfg.pre_emphasis = m->hparams.fe_pre_emphasis;
        cfg.f_min        = m->hparams.fe_f_min;
        cfg.f_max        = m->hparams.fe_f_max;
        cfg.pad_mode     = m->hparams.fe_pad_mode;

        // Load checkpoint filterbank/window from GGUF if present (exact
        // trained values). read_f32_tensor_checked: Absent -> compute from
        // hparams; BadType/BadSize/ReadErr -> hard fail.
        {
            using R = load_common::ReadF32Result;

            const size_t fb_elems = static_cast<size_t>(cfg.num_mels) * static_cast<size_t>(cfg.n_fft / 2 + 1);
            const auto   fb_rc    = load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(), "frontend.mel_filterbank", fb_elems, "cohere", cfg.filterbank);
            if (fb_rc != R::Ok && fb_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }

            const size_t win_elems = static_cast<size_t>(cfg.win_length);
            const auto   win_rc = load_common::read_f32_tensor_checked(loader.gguf(), loader.path(), "frontend.window",
                                                                       win_elems, "cohere", cfg.window);
            if (win_rc != R::Ok && win_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
        }

        m->mel.emplace(cfg);
    }

    // Reopen with no_alloc.
    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (const transcribe_status st = build_cohere_weights(m->ctx_meta, m->hparams, m->weights); st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }

    // Resolve the backend plan (see transcribe-load-common.h). The conv
    // pointwise F16->F32 promotion below depends on a strict-CPU plan
    // (primary_kind == BackendKind::Cpu), which only TRANSCRIBE_BACKEND_CPU
    // reliably produces.
    const transcribe_backend_request backend_req = (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;

    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, (params != nullptr) ? params->gpu_device : 0, "cohere", m->plan);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }

    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (weights_buffer == nullptr) {
        gguf_free(gguf_data);
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere: ggml_backend_alloc_ctx_tensors failed");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Stream tensor data from the GGUF file into the backend buffer
    // slots. See transcribe-load-common.h for the shared loop.
    if (const transcribe_status st =
            transcribe::load_common::stream_tensor_data(loader.path(), gguf_data, m->ctx_meta, "cohere");
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }

    gguf_free(gguf_data);

    if (const transcribe_status st = fuse_batch_norm(*m); st != TRANSCRIBE_OK) {
        return st;
    }

    // Fuse encoder Q bias into pos_bias_u/v (drops one add per layer).
    if (const transcribe_status st = fuse_encoder_q_bias(*m); st != TRANSCRIBE_OK) {
        return st;
    }

    // CPU only: dequantize conv pointwise weights to F32 (see function doc).
    if (const transcribe_status st = promote_conv_pw_to_f32_on_cpu(*m); st != TRANSCRIBE_OK) {
        return st;
    }

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(transcribe_model *                model,
                               const transcribe_session_params * params,
                               transcribe_session **             out_ctx) {
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto cc       = std::make_unique<CohereSession>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;
    // Cache the caller's n_ctx knob; for cohere it lowers the DECODER
    // self-KV ceiling only (see cohere_dec_ctx_ceiling).
    cc->n_ctx     = transcribe_session_params_n_ctx(params);

    // Flash-attention policy — see CohereSession (cohere.h). Encoder
    // auto-disables on Metal; decoder defaults on; env overrides apply globally.
    auto *     cm       = static_cast<CohereModel *>(model);
    const bool is_metal = (cm->plan.primary_kind == transcribe::BackendKind::Metal);

    cc->encoder_use_flash = !is_metal;
    cc->decoder_use_flash = true;

    transcribe::flash::apply_env_overrides(cc->encoder_use_flash, cc->decoder_use_flash);

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// Transcribe one window of PCM. This is the single-pass path: mel -> encoder
// -> autoregressive decode -> committed result. run() below drives it once for
// short-form audio and repeatedly for long-form.
transcribe_status run_window(transcribe_session *          session,
                             const float *                 pcm,
                             int                           n_samples,
                             const transcribe_run_params * params) {
    if (session == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<CohereSession *>(session);
    auto * cm = static_cast<CohereModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Pre-run abort check.
    if (cc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    transcribe::debug::init();

    // ----- Mel front-end -------------------------------------------
    if (!cm->mel.has_value()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere run: model has no MelFrontend (load skipped?)");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int64_t t_mel_start  = ggml_time_us();
    int           mel_n_mels   = 0;
    int           mel_n_frames = 0;
    if (const transcribe_status mst =
            cm->mel->compute(pcm, static_cast<size_t>(n_samples), cc->mel_buf, mel_n_mels, mel_n_frames);
        mst != TRANSCRIBE_OK) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere run: MelFrontend::compute failed (%s)",
                transcribe_status_string(mst));
        return mst;
    }
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    // ----- Input-length gate -----
    // T_enc must stay within the trained pos-emb span enc_pos_emb_max_len
    // or the runtime pos table aliases past the trained range. T_enc is a
    // deterministic function of mel_n_frames, so reject an over-length clip
    // here, before building the encoder graph.
    if (cm->hparams.enc_pos_emb_max_len > 0) {
        const int t_enc_pred = cohere_predict_t_enc(mel_n_frames, cm->hparams.enc_subsampling_factor);
        if (t_enc_pred > cm->hparams.enc_pos_emb_max_len) {
            const double max_s = static_cast<double>(cohere_max_audio_ms(cm->hparams)) / 1000.0;
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "cohere run: input too long — %d encoder frames exceed the %d "
                                "the model supports (~%.0f s max). See "
                                "transcribe_capabilities.max_audio_ms.",
                                t_enc_pred, cm->hparams.enc_pos_emb_max_len, max_s);
            return TRANSCRIBE_ERR_INPUT_TOO_LONG;
        }
    }

    // ----- Reset per-call compute state ----------------------------
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    cc->encoder_out = nullptr;

    // ----- Build encoder graph -------------------------------------
    {
        // 48 encoder blocks + decoder = large graph. 8 MB metadata arena.
        ggml_init_params init_params{};
        init_params.mem_size   = 8 * 1024 * 1024;
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;
        cc->compute_ctx        = ggml_init(init_params);
        if (cc->compute_ctx == nullptr) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "cohere run: ggml_init for compute_ctx failed — out of memory.");
            return TRANSCRIBE_ERR_OOM;
        }
    }

    ggml_type resolved_kv = GGML_TYPE_COUNT;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) {
        resolved_kv = GGML_TYPE_F32;
    }
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F16) {
        resolved_kv = GGML_TYPE_F16;
    }

    EncoderBuild eb = build_encoder_graph(cc->compute_ctx, cm->weights, cm->hparams, mel_n_frames, resolved_kv,
                                          cc->encoder_use_flash, cm->backend.c_str());
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // ----- Allocate + compute encoder graph -------------------------
    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(cm->plan.scheduler_list.size()), 16384, false, true);
        if (cc->sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere run: ggml_backend_sched_new failed");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere run: encoder graph allocation failed — out of memory.");
        return TRANSCRIBE_ERR_OOM;
    }

    // Upload mel.
    ggml_backend_tensor_set(eb.mel_in, cc->mel_buf.data(), 0, cc->mel_buf.size() * sizeof(float));

    transcribe::debug::dump_tensor("enc.mel.in", eb.mel_in, "encoder.mel");

    // Sinusoidal positional embedding (same as Parakeet).
    if (eb.pos_emb_in != nullptr) {
        const int d_model = cm->hparams.enc_d_model;
        const int pos_len = static_cast<int>(eb.pos_emb_in->ne[1]);
        const int T_enc   = (pos_len + 1) / 2;

        cc->pos_buf.assign(static_cast<size_t>(pos_len) * d_model, 0.0f);

        cc->pos_div_term.resize(static_cast<size_t>(d_model / 2));
        const float ln_10000 = std::log(10000.0f);
        for (int k = 0; k < d_model / 2; ++k) {
            cc->pos_div_term[static_cast<size_t>(k)] =
                std::exp(static_cast<float>(2 * k) * (-ln_10000 / static_cast<float>(d_model)));
        }

        for (int i = 0; i < pos_len; ++i) {
            const float pos = static_cast<float>((T_enc - 1) - i);
            float *     row = cc->pos_buf.data() + static_cast<size_t>(i) * d_model;
            for (int k = 0; k < d_model / 2; ++k) {
                const float div = cc->pos_div_term[static_cast<size_t>(k)];
                row[2 * k]      = std::sin(pos * div);
                row[2 * k + 1]  = std::cos(pos * div);
            }
        }

        ggml_backend_tensor_set(eb.pos_emb_in, cc->pos_buf.data(), 0, cc->pos_buf.size() * sizeof(float));

        transcribe::debug::dump_tensor("enc.pos_emb", eb.pos_emb_in, "encoder.pos_emb");
    }

    // Set thread count.
    transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);

    // Compute encoder graph.
    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, eb.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere run: encoder graph compute failed (%d)", static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_encode_us = ggml_time_us() - t_enc_start;

    // Dump encoder intermediates.
    auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
        if (t != nullptr) {
            transcribe::debug::dump_tensor(name, t, stage);
        }
    };

    // Pre-encode sub-stage dumps for debugging.
    try_dump("enc.pre_encode.out", eb.dumps.pre_encode_out, "encoder.pre_encode");
    try_dump("enc.block.0.out", eb.dumps.block0_out, "encoder.block0.out");
    {
        char bname[64];
        std::snprintf(bname, sizeof(bname), "enc.block.%d.out", cm->hparams.enc_n_layers / 2 - 1);
        try_dump(bname, eb.dumps.block_mid_out, "encoder.block_mid.out");
    }
    {
        char bname[64];
        std::snprintf(bname, sizeof(bname), "enc.block.%d.out", cm->hparams.enc_n_layers - 1);
        try_dump(bname, eb.dumps.block_last_out, "encoder.block_last.out");
    }
    try_dump("enc.final", eb.dumps.final_out, "encoder.final");
    try_dump("enc_dec_proj.out", eb.dumps.enc_dec_proj_out, "encoder.enc_dec_proj");

    cc->encoder_out = eb.out;

    // Read encoder output (after enc-dec projection) to host.
    const int d_enc = static_cast<int>(eb.out->ne[0]);
    const int T_enc = static_cast<int>(eb.out->ne[1]);
    if (d_enc <= 0 || T_enc <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "cohere run: encoder output has degenerate shape "
                "[%d, %d]",
                d_enc, T_enc);
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->enc_host.resize(static_cast<size_t>(d_enc) * static_cast<size_t>(T_enc));
    ggml_backend_tensor_get(eb.out, cc->enc_host.data(), 0, cc->enc_host.size() * sizeof(float));

    // ----- Decoder with KV cache ------------------------------------
    //
    // Steps:
    //   1. Initialize KV cache (if not already done for this T_enc).
    //   2. Compute cross-attention K/V once from encoder output.
    //   3. Prompt pass: process all prompt tokens, populate self-attn
    //      KV cache, get first predicted token.
    //   4. Autoregressive loop: single-token step passes using cached KV.

    // Build the prompt tokens from tokenizer vocabulary.
    //
    // Prompt structure (matches Transformers get_decoder_prompt_ids):
    //   ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|>
    //   <|{lang}|> <|{lang}|>
    //   <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
    //
    const char * lang = (params && params->language) ? params->language : "en";

    // The dispatcher (transcribe_run) rejects unsupported caller languages
    // before this handler; the NULL -> "en" default applies only when the
    // caller specified none.

    const std::string              lang_token    = std::string("<|") + lang + "|>";
    const std::vector<std::string> prompt_pieces = {
        "\xe2\x96\x81",  // ▁ (U+2581, sentencepiece space prefix)
        "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>", lang_token, lang_token, "<|pnc|>",
        "<|noitn|>",          "<|notimestamp|>",       "<|nodiarize|>",
    };

    std::vector<int32_t> prompt_ids;
    prompt_ids.reserve(prompt_pieces.size());
    for (const auto & piece : prompt_pieces) {
        const int id = cm->tok.find(piece);
        if (id < 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere run: unknown prompt token '%s'", piece.c_str());
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        prompt_ids.push_back(id);
    }
    const int prompt_len = static_cast<int>(prompt_ids.size());

    // ----- Initialize KV cache -----
    {
        // Free any existing cache if T_enc changed.
        if (cc->kv_cache.buffer != nullptr && cc->kv_cache.T_enc != T_enc) {
            cc->kv_cache.free();
        }

        if (cc->kv_cache.buffer == nullptr) {
            // Decoder self-KV ceiling: the model's trained dec_max_seq,
            // optionally lowered (never raised) by the caller's n_ctx knob.
            const int n_ctx      = cohere_dec_ctx_ceiling(cc->n_ctx, cm->hparams);
            // Decoder cache dtype: honor user override, else default to
            // F16 (weights are bf16, so F16 KV is lossless-enough and
            // halves autoregressive memory bandwidth).
            ggml_type cache_type = resolved_kv;
            if (cache_type == GGML_TYPE_COUNT) {
                cache_type = GGML_TYPE_F16;
            }
            if (!kv_cache_init(cc->kv_cache, cm->plan.primary, n_ctx, T_enc, static_cast<int>(cm->hparams.dec_hidden),
                               cm->hparams.dec_n_layers, cache_type)) {
                transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                    "cohere run: KV cache allocation failed — out of memory. "
                                    "Lower transcribe_session_params.n_ctx or shorten the audio.");
                return TRANSCRIBE_ERR_OOM;
            }
        } else {
            // Cache exists with same T_enc; just reset self-attn state.
            cc->kv_cache.n               = 0;
            cc->kv_cache.head            = 0;
            cc->kv_cache.cross_populated = false;
        }
    }

    // Fresh compute context. Null cc->encoder_out first: it was allocated in
    // the context being freed (its data is already in cc->enc_host), so the
    // stale pointer must not outlive the ggml_free.
    auto new_compute_ctx = [&](size_t mem_size) -> bool {
        if (cc->compute_ctx != nullptr) {
            ggml_free(cc->compute_ctx);
            cc->compute_ctx = nullptr;
        }
        cc->encoder_out = nullptr;
        ggml_init_params init_params{};
        init_params.mem_size   = mem_size;
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;
        cc->compute_ctx        = ggml_init(init_params);
        return cc->compute_ctx != nullptr;
    };

    // Helper to find the causal mask input tensor by name.
    auto find_mask_input = [&]() -> ggml_tensor * {
        for (ggml_tensor * t = ggml_get_first_tensor(cc->compute_ctx); t != nullptr;
             t               = ggml_get_next_tensor(cc->compute_ctx, t)) {
            if (std::strcmp(t->name, "dec.causal_mask") == 0 && t->type == GGML_TYPE_F32) {
                return t;
            }
        }
        return nullptr;
    };

    // ----- Compute cross-attention K/V -----
    const int64_t t_dec_start = ggml_time_us();
    {
        if (!new_compute_ctx(4 * 1024 * 1024)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "cohere run: ggml_init for cross_kv failed — out of memory.");
            return TRANSCRIBE_ERR_OOM;
        }

        DecoderBuild cross_db = build_cross_kv_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, T_enc);
        if (cross_db.graph == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere run: build_cross_kv_graph failed");
            return TRANSCRIBE_ERR_GGUF;
        }

        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, cross_db.graph)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "cohere run: cross_kv graph allocation failed — out of memory.");
            return TRANSCRIBE_ERR_OOM;
        }

        // Upload encoder output.
        ggml_backend_tensor_set(cross_db.encoder_out_in, cc->enc_host.data(), 0, cc->enc_host.size() * sizeof(float));

        if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, cross_db.graph);
            gs != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere run: cross_kv compute failed (%d)", static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }
        cc->kv_cache.cross_populated = true;
    }

    // ----- Prompt pass with KV cache -----
    {
        if (!new_compute_ctx(4 * 1024 * 1024)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "cohere run: ggml_init for decoder prompt failed — out of memory.");
            return TRANSCRIBE_ERR_OOM;
        }

        // Skip log_softmax and emit a GPU argmax over the last position.
        // Debug dumps need the pre-head state, so they take the slower path.
        const bool   prompt_skip_softmax = !transcribe::debug::enabled();
        DecoderBuild db = build_decoder_graph_kv(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, prompt_len,
                                                 /*n_past=*/0, T_enc,
                                                 /*skip_log_softmax=*/prompt_skip_softmax, cc->decoder_use_flash);
        if (db.out == nullptr || db.graph == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere run: build_decoder_graph_kv (prompt) failed");
            return TRANSCRIBE_ERR_GGUF;
        }

        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, db.graph)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "cohere run: decoder prompt graph allocation failed — out of memory.");
            return TRANSCRIBE_ERR_OOM;
        }

        // Upload token IDs and position IDs.
        ggml_backend_tensor_set(db.token_ids_in, prompt_ids.data(), 0, prompt_ids.size() * sizeof(int32_t));
        std::vector<int32_t> pos_ids(prompt_len);
        for (int i = 0; i < prompt_len; ++i) {
            pos_ids[i] = i;
        }
        ggml_backend_tensor_set(db.pos_ids_in, pos_ids.data(), 0, pos_ids.size() * sizeof(int32_t));

        // Causal mask for prompt pass: [n_kv=prompt_len, prompt_len].
        if (prompt_len > 1) {
            ggml_tensor * mask_input = find_mask_input();
            if (mask_input != nullptr) {
                const int          n_kv = prompt_len;
                std::vector<float> mask_data(static_cast<size_t>(n_kv) * prompt_len);
                for (int q = 0; q < prompt_len; ++q) {
                    for (int k = 0; k < n_kv; ++k) {
                        mask_data[static_cast<size_t>(q) * n_kv + k] = (k <= q) ? 0.0f : -1e9f;
                    }
                }
                ggml_backend_tensor_set(mask_input, mask_data.data(), 0, mask_data.size() * sizeof(float));
            }
        }

        if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, db.graph); gs != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere run: decoder prompt compute failed (%d)", static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }

        // Dump decoder intermediates (prompt pass only).
        try_dump("dec.token_emb", db.dumps.token_emb, "decoder.embedding");
        try_dump("dec.pos_emb", db.dumps.pos_emb, "decoder.position_embedding");
        try_dump("dec.embed_norm", db.dumps.embed_norm, "decoder.embed_norm");
        for (int i = 0; i < cm->hparams.dec_n_layers; ++i) {
            char bname[32], stage[48];
            std::snprintf(bname, sizeof(bname), "dec.block.%d.out", i);
            std::snprintf(stage, sizeof(stage), "decoder.block%d.out", i);
            try_dump(bname, db.dumps.block_out[i], stage);
        }
        try_dump("dec.out_before_head", db.dumps.out_before_head, "decoder.out_before_head");
        try_dump("dec.logits_raw", db.dumps.logits_raw, "decoder.logits_raw");
        try_dump("dec.logits", db.dumps.logits, "decoder.logits");

        // Update KV cache state: prompt_len tokens are now cached.
        cc->kv_cache.n    = prompt_len;
        cc->kv_cache.head = prompt_len;

        // Greedy decode from last prompt position.
        cc->clear_result();

        // Load-time validation guarantees eos_token_id >= 0; no
        // fallback is needed here. See the tokenizer.eos_id() check
        // in cohere::load() at the top of this file.
        const int eos_id     = cm->hparams.eos_token_id;
        const int max_tokens = std::min(512, cc->kv_cache.n_ctx - prompt_len);

        // Pick the first generated token. Fast path reads a single
        // int32 argmax that the GPU computed; debug path reads the
        // full log_softmax'd logits for dumping and argmaxes on host.
        int next_token = 0;
        if (prompt_skip_softmax && db.argmax_out != nullptr) {
            int32_t argmax_id = 0;
            ggml_backend_tensor_get(db.argmax_out, &argmax_id, 0, sizeof(int32_t));
            next_token = argmax_id;
        } else {
            const int64_t      vocab_size = db.out->ne[0];
            std::vector<float> logits_host(static_cast<size_t>(vocab_size) * prompt_len);
            ggml_backend_tensor_get(db.out, logits_host.data(), 0, logits_host.size() * sizeof(float));
            const float * last_logits = logits_host.data() + static_cast<size_t>(prompt_len - 1) * vocab_size;
            float         best        = last_logits[0];
            for (int j = 1; j < static_cast<int>(vocab_size); ++j) {
                if (last_logits[j] > best) {
                    best       = last_logits[j];
                    next_token = j;
                }
            }
        }

        std::vector<int> generated_ids;
        if (next_token != eos_id) {
            generated_ids.push_back(next_token);
        }

        // Commit accumulated generated_ids as the run's segment + full
        // text. Called both on normal loop exit and on abort so the
        // public contract (partial result on TRANSCRIBE_ERR_ABORTED)
        // holds. Mirrors the result-shape rationale below: cohere
        // advertises max_timestamp_kind == NONE so we expose a single
        // text-only segment with zeroed timings and no token/word
        // substructure.
        auto commit_result = [&]() {
            cc->t_decode_us = ggml_time_us() - t_dec_start;

            // Publish the raw ids so the long-form path in run() can stitch
            // windows in token space. Set before the empty check so a window
            // that produced nothing clears the previous window's ids.
            cc->window_ids = generated_ids;

            if (generated_ids.empty()) {
                return;
            }

            const transcribe::Tokenizer & tok = cm->tok;

            std::string full = tok.decode(generated_ids.data(), static_cast<int>(generated_ids.size()));
            cc->raw_text     = full;  // pre-trim decode, via transcribe_raw_text
            if (!full.empty() && full.front() == ' ') {
                full.erase(full.begin());
            }

            transcribe_session::SegmentEntry seg;
            seg.t0_ms       = 0;
            seg.t1_ms       = 0;
            seg.first_token = 0;
            seg.n_tokens    = 0;
            seg.first_word  = 0;
            seg.n_words     = 0;
            seg.text        = full;

            cc->segments.push_back(std::move(seg));
            cc->full_text   = std::move(full);
            cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
            cc->has_result  = true;
        };

        // ----- Autoregressive step passes -----
        // Two loop variants by primary backend kind:
        //   GPU: build_step_graph — one static-topology graph; KV writes via
        //     ggml_set_rows at runtime kv_idx, flash-attn over a fixed
        //     max_n_kv window. Avoids per-step graph_build + sched_alloc,
        //     which dominate GPU dispatch overhead.
        //   CPU: build_decoder_graph_kv per step — n_kv grows with n_past
        //     (reads only the populated prefix). CPU has no dispatch overhead
        //     to amortize, so the static-graph bandwidth tax is a net loss.
        int n_past = prompt_len;

        const bool primary_is_gpu = cm->plan.primary_kind != transcribe::BackendKind::Cpu &&
                                    cm->plan.primary_kind != transcribe::BackendKind::Accel &&
                                    cm->plan.primary_kind != transcribe::BackendKind::Unknown;

        if (primary_is_gpu) {
            // ---------- Static-graph step path (GPU) ----------
            // max_n_kv: next power of two (1024 floor) — Vulkan/Metal flash
            // dispatches faster on pow2 ne[1], and the static graph amortizes
            // the slight bandwidth cost.
            int max_n_kv = 1024;
            while (max_n_kv < prompt_len + max_tokens) {
                max_n_kv *= 2;
            }
            if (max_n_kv > cc->kv_cache.n_ctx) {
                max_n_kv = cc->kv_cache.n_ctx;
            }

            if (!new_compute_ctx(8 * 1024 * 1024)) {
                transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                    "cohere run: new_compute_ctx failed (step) — out of memory.");
                commit_result();
                return TRANSCRIBE_ERR_OOM;
            }
            StepBuild sb = build_step_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, max_n_kv, T_enc,
                                            cc->decoder_use_flash);
            if (sb.graph == nullptr || sb.argmax_out == nullptr) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere run: build_step_graph failed");
                commit_result();
                return TRANSCRIBE_ERR_GGUF;
            }
            ggml_backend_sched_reset(cc->sched);
            if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
                transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                    "cohere run: step graph allocation failed — out of memory.");
                commit_result();
                return TRANSCRIBE_ERR_OOM;
            }

            // Mask buffer: full max_n_kv span, reused host-side. Positions
            // already populated by the prompt pass [0, prompt_len) start
            // attendable; remaining slots are -inf until each step flips
            // its newly-written position to attendable.
            const ggml_fp16_t        mask_zero    = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t        mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
            std::vector<ggml_fp16_t> step_mask(max_n_kv, mask_neg_inf);
            for (int p = 0; p < prompt_len; ++p) {
                step_mask[p] = mask_zero;
            }

            for (int step = 1; step < max_tokens && next_token != eos_id; ++step) {
                if (cc->poll_abort()) {
                    commit_result();
                    return TRANSCRIBE_ERR_ABORTED;
                }
                if (n_past + 1 > max_n_kv) {
                    // Hit the self-KV window before EOS: keep the partial
                    // transcript, flag truncation, warn — never discard it.
                    cc->was_truncated = true;
                    transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                                        "cohere run: output truncated at %d tokens — decode "
                                        "reached the context/budget before end-of-stream; "
                                        "transcript may be incomplete.",
                                        static_cast<int>(generated_ids.size()));
                    break;
                }

                int32_t token_val = next_token;
                int32_t pos_val   = n_past;
                int64_t kv_val    = n_past;
                ggml_backend_tensor_set(sb.token_id_in, &token_val, 0, sizeof(int32_t));
                ggml_backend_tensor_set(sb.pos_id_in, &pos_val, 0, sizeof(int32_t));
                ggml_backend_tensor_set(sb.kv_idx_in, &kv_val, 0, sizeof(int64_t));

                step_mask[n_past] = mask_zero;
                ggml_backend_tensor_set(sb.mask_in, step_mask.data(), 0,
                                        static_cast<size_t>(max_n_kv) * sizeof(ggml_fp16_t));

                if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, sb.graph);
                    gs != GGML_STATUS_SUCCESS) {
                    log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "cohere run: step compute failed (%d, n_past=%d)",
                            static_cast<int>(gs), n_past);
                    commit_result();
                    return TRANSCRIBE_ERR_GGUF;
                }

                n_past += 1;
                cc->kv_cache.n    = n_past;
                cc->kv_cache.head = n_past;

                int32_t argmax_id = 0;
                ggml_backend_tensor_get(sb.argmax_out, &argmax_id, 0, sizeof(int32_t));
                next_token = argmax_id;

                if (next_token != eos_id) {
                    generated_ids.push_back(next_token);
                }
            }
        } else {
            // ---------- Dynamic-graph step path (CPU) ----------
            // Reserve scheduler buffers with a worst-case single-token graph
            // (n_past = n_ctx - 1) so alloc_graph never reallocates in-loop.
            if (new_compute_ctx(4 * 1024 * 1024)) {
                const int    worst_n_past = cc->kv_cache.n_ctx - 1;
                DecoderBuild db_reserve =
                    build_decoder_graph_kv(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache,
                                           /*n_tokens=*/1, worst_n_past, T_enc,
                                           /*skip_log_softmax=*/true, cc->decoder_use_flash);
                if (db_reserve.graph != nullptr) {
                    ggml_backend_sched_reserve(cc->sched, db_reserve.graph);
                }
            }

            for (int step = 1; step < max_tokens && next_token != eos_id; ++step) {
                if (cc->poll_abort()) {
                    commit_result();
                    return TRANSCRIBE_ERR_ABORTED;
                }

                if (n_past + 1 > cc->kv_cache.n_ctx) {
                    // Filled the self-KV cache before EOS: keep the partial
                    // transcript, flag truncation, warn — never discard it.
                    cc->was_truncated = true;
                    transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                                        "cohere run: output truncated at %d tokens — decode "
                                        "reached the context/budget before end-of-stream; "
                                        "transcript may be incomplete.",
                                        static_cast<int>(generated_ids.size()));
                    break;
                }

                if (!new_compute_ctx(4 * 1024 * 1024)) {
                    break;
                }

                DecoderBuild db_step = build_decoder_graph_kv(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache,
                                                              /*n_tokens=*/1, n_past, T_enc,
                                                              /*skip_log_softmax=*/true, cc->decoder_use_flash);
                if (db_step.out == nullptr || db_step.graph == nullptr) {
                    break;
                }

                ggml_backend_sched_reset(cc->sched);
                if (!ggml_backend_sched_alloc_graph(cc->sched, db_step.graph)) {
                    break;
                }

                int32_t token_id = next_token;
                int32_t pos_id   = n_past;
                ggml_backend_tensor_set(db_step.token_ids_in, &token_id, 0, sizeof(int32_t));
                ggml_backend_tensor_set(db_step.pos_ids_in, &pos_id, 0, sizeof(int32_t));

                if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, db_step.graph);
                    gs != GGML_STATUS_SUCCESS) {
                    break;
                }

                n_past += 1;
                cc->kv_cache.n    = n_past;
                cc->kv_cache.head = n_past;

                int32_t argmax_id = 0;
                ggml_backend_tensor_get(db_step.argmax_out, &argmax_id, 0, sizeof(int32_t));
                next_token = argmax_id;

                if (next_token != eos_id) {
                    generated_ids.push_back(next_token);
                }
            }
        }

        // A clean decode stops at EOS. If the loop instead ran out of budget
        // (max-new cap) with a non-EOS token pending, the transcript is
        // truncated. The break sites flag the self-KV-full case; this covers
        // the normal-exit-at-budget case.
        if (!cc->was_truncated && next_token != eos_id) {
            cc->was_truncated = true;
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                                "cohere run: output truncated at %d tokens — decode reached "
                                "the context/budget before end-of-stream; transcript may be "
                                "incomplete.",
                                static_cast<int>(generated_ids.size()));
        }

        // Build the result. max_timestamp_kind == NONE means text but no
        // alignment data: full_text plus one segment (text == full_text,
        // zeroed timings and counts), with empty tokens/words.
        commit_result();
    }

    // Output truncation is a hard status: the partial transcript is committed
    // and stays readable (like an aborted run), but we surface the truncation
    // rather than reporting a clean OK.
    return cc->was_truncated ? TRANSCRIBE_ERR_OUTPUT_TRUNCATED : TRANSCRIBE_OK;
}

// Long-form entry point.
//
// The encoder's positional table spans ~400 s, so a clip well past the trained
// window still encodes cleanly and the up-front gate lets it through. The
// decoder is the problem: its cross-attention has no monotonicity constraint
// (unlike an RNN-T, which walks the encoder frames in order), so past the
// trained span it drifts, emits EOS early, and drops the tail. The run then
// returns TRANSCRIBE_OK with a silently incomplete transcript -- exactly what
// docs/input-limits.md promises never happens.
//
// So window the audio to the trained span. Windows overlap by kOverlapMs and
// are stitched in token space by token_seam(), so a boundary that lands
// mid-word costs nothing: the word survives whole in the next window and the
// duplicate is trimmed at the join. Boundaries still prefer a pause where one
// is detectable, which makes the seam easier to find. Short-form audio takes
// the single-pass path unchanged.
transcribe_status run(transcribe_session *          session,
                      const float *                 pcm,
                      int                           n_samples,
                      const transcribe_run_params * params) {
    if (session == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<CohereSession *>(session);
    auto * cm = static_cast<CohereModel *>(cc->model);
    if (cm == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int sample_rate    = cm->hparams.fe_sample_rate > 0 ? cm->hparams.fe_sample_rate : 16000;
    const int window_samples = static_cast<int>(kTrainedClipSeconds * sample_rate);

    // Short-form: one window, identical to the pre-windowing path.
    if (window_samples <= 0 || n_samples <= window_samples) {
        return run_window(session, pcm, n_samples, params);
    }

    const int search      = static_cast<int>(window_samples * kSplitSearchFraction);
    const int search_wide = static_cast<int>(window_samples * kSplitSearchFractionWide);

    const int overlap_samples = std::min(kOverlapMs * sample_rate / 1000, window_samples / 2);

    // Windows are stitched in TOKEN space, then decoded once at the end. Two
    // reasons: the seam search needs ids, and a single decode reproduces the
    // tokenizer's own spacing exactly. Gluing decoded strings would either
    // inject separators the model never emitted (fatal for unspaced scripts
    // like Chinese and Japanese, both advertised by this family) or require
    // re-deriving word boundaries from text.
    std::vector<int> merged_ids;
    int64_t          mel_us    = 0;
    int64_t          encode_us = 0;
    int64_t          decode_us = 0;
    int              offset    = 0;

    // Fold the accumulated ids into the session result. Windowing is an
    // implementation detail: the caller still sees a single text-only segment,
    // matching max_timestamp_kind == NONE.
    auto commit_windows = [&]() {
        cc->clear_result();
        cc->t_mel_us    = mel_us;
        cc->t_encode_us = encode_us;
        cc->t_decode_us = decode_us;

        if (merged_ids.empty()) {
            return;
        }

        // raw_text keeps the untrimmed decode (the transcribe_raw_text
        // contract); full_text applies the same single leading-space trim
        // run_window() applies to a one-window run.
        std::string raw_joined = cm->tok.decode(merged_ids.data(), static_cast<int>(merged_ids.size()));
        std::string full       = raw_joined;
        if (!full.empty() && full.front() == ' ') {
            full.erase(full.begin());
        }

        transcribe_session::SegmentEntry seg;
        seg.t0_ms       = 0;
        seg.t1_ms       = 0;
        seg.first_token = 0;
        seg.n_tokens    = 0;
        seg.first_word  = 0;
        seg.n_words     = 0;
        seg.text        = full;

        cc->raw_text = std::move(raw_joined);
        cc->segments.push_back(std::move(seg));
        cc->full_text   = std::move(full);
        cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        cc->has_result  = true;
    };

    int n_windows = 0;
    while (offset < n_samples) {
        const int remaining = n_samples - offset;
        int       take      = remaining;

        if (remaining > window_samples) {
            // Prefer a pause; widen the search once before settling for a
            // mid-speech cut.
            int split = pick_split(pcm + offset, remaining, window_samples, search);
            if (split < 0) {
                split = pick_split(pcm + offset, remaining, window_samples, search_wide);
            }
            if (split > 0) {
                take = split;
            } else {
                take = window_samples;
                transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                                    "cohere run: no pause near the window boundary at sample %d — "
                                    "cutting mid-speech; a word may be split across windows.",
                                    offset + window_samples);
            }
        }
        if (take <= 0) {
            take = std::min(remaining, window_samples);
        }

        // Drop the previous window's state *before* running. run_window() can
        // return early (abort, mel failure, over-length gate, OOM) before it
        // reaches its own clear_result(), and stale text would otherwise be
        // captured a second time as if it belonged to this window. clear_result()
        // does not touch the timing fields, so zero those explicitly or an
        // early-returning window re-adds the previous window's numbers.
        cc->clear_result();
        cc->window_ids.clear();
        cc->t_mel_us    = 0;
        cc->t_encode_us = 0;
        cc->t_decode_us = 0;

        const transcribe_status st = run_window(session, pcm + offset, take, params);
        ++n_windows;

        mel_us += cc->t_mel_us;
        encode_us += cc->t_encode_us;
        decode_us += cc->t_decode_us;

        if (!cc->window_ids.empty()) {
            if (merged_ids.empty()) {
                merged_ids = cc->window_ids;
            } else {
                // The windows share `overlap_samples` of audio, so the new one
                // re-transcribes the tail of the previous. Drop that duplicate
                // prefix; when no seam is found the text is appended whole,
                // which repeats a little rather than losing anything.
                const TokenSeam seam = token_seam(merged_ids, cc->window_ids, seam_search_previous(cm->hparams),
                                                  seam_search_current(cm->hparams));
                if (!seam.matched) {
                    transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                                        "cohere run: no token seam at window %d — appending whole "
                                        "(overlap may repeat a few words).",
                                        n_windows);
                }
                merged_ids.insert(merged_ids.end(), cc->window_ids.begin() + seam.current_skip, cc->window_ids.end());
            }
        }

        // OUTPUT_TRUNCATED keeps its partial text and is surfaced at the end
        // through was_truncated; any other non-OK status stops the run, but the
        // windows already decoded stay readable (the aborted-run contract).
        if (st != TRANSCRIBE_OK && st != TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
            commit_windows();
            return st;
        }

        // Step back by the overlap so the next window re-reads the tail of this
        // one. Never on the final window (nothing follows), and never far
        // enough to stall: `take` always exceeds the overlap because
        // overlap_samples is capped at half a window.
        const int advance = (offset + take >= n_samples) ? take : take - overlap_samples;
        offset += advance > 0 ? advance : take;
    }

    commit_windows();

    transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                        "cohere run: long-form windowed into %d windows (<= %.0f s each, %d ms overlap)", n_windows,
                        kTrainedClipSeconds, kOverlapMs);

    return cc->was_truncated ? TRANSCRIBE_ERR_OUTPUT_TRUNCATED : TRANSCRIBE_OK;
}

// ===========================================================================
// Offline batched decode (transcribe_run_batch)
// ===========================================================================
//
// The compute-bound encoder stays SERIAL per utterance (batching a heavy
// conformer regresses on mixed lengths). The host-side mel is parallelized
// and the bandwidth-bound autoregressive DECODE is batched, so B utterances
// amortize the per-step weight reads.
//
// Cohere-specific wrinkle: each utterance has its own encoder output
// (variable T_enc), so the cross KV cache carries a batch dim and a
// per-utterance cross-pad mask discards frames past T_enc[b]. The uniform
// prompt feeds through the same batched step graph (prompt_len steps).

// Encoder for one utterance from a precomputed mel buffer → host [hidden, T_enc].
transcribe_status encode_one_to_host(CohereSession *            cc,
                                     CohereModel *              cm,
                                     const std::vector<float> & mel_buf,
                                     int                        mel_n_frames,
                                     std::vector<float> &       enc_host_out,
                                     int &                      T_enc_out,
                                     int64_t &                  enc_us) {
    if (mel_n_frames <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    cc->encoder_out = nullptr;
    {
        ggml_init_params ip{};
        ip.mem_size     = 8 * 1024 * 1024;
        ip.mem_buffer   = nullptr;
        ip.no_alloc     = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx == nullptr) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "cohere encode_one_to_host: ggml_init failed — out of memory.");
            return TRANSCRIBE_ERR_OOM;
        }
    }

    ggml_type resolved_kv = GGML_TYPE_COUNT;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) {
        resolved_kv = GGML_TYPE_F32;
    }
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F16) {
        resolved_kv = GGML_TYPE_F16;
    }

    EncoderBuild eb = build_encoder_graph(cc->compute_ctx, cm->weights, cm->hparams, mel_n_frames, resolved_kv,
                                          cc->encoder_use_flash, cm->backend.c_str());
    if (eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(cm->plan.scheduler_list.size()), 16384, false, true);
        if (cc->sched == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "cohere encode_one_to_host: encoder graph allocation failed — out of memory.");
        return TRANSCRIBE_ERR_OOM;
    }

    ggml_backend_tensor_set(eb.mel_in, mel_buf.data(), 0, mel_buf.size() * sizeof(float));

    if (eb.pos_emb_in != nullptr) {
        const int          d_model = cm->hparams.enc_d_model;
        const int          pos_len = static_cast<int>(eb.pos_emb_in->ne[1]);
        const int          T_enc   = (pos_len + 1) / 2;
        std::vector<float> pos_buf(static_cast<size_t>(pos_len) * d_model, 0.0f);
        const float        ln_10000 = std::log(10000.0f);
        std::vector<float> div_term(static_cast<size_t>(d_model / 2));
        for (int k = 0; k < d_model / 2; ++k) {
            div_term[k] = std::exp(static_cast<float>(2 * k) * (-ln_10000 / static_cast<float>(d_model)));
        }
        for (int i = 0; i < pos_len; ++i) {
            const float p   = static_cast<float>((T_enc - 1) - i);
            float *     row = pos_buf.data() + static_cast<size_t>(i) * d_model;
            for (int k = 0; k < d_model / 2; ++k) {
                row[2 * k]     = std::sin(p * div_term[k]);
                row[2 * k + 1] = std::cos(p * div_term[k]);
            }
        }
        ggml_backend_tensor_set(eb.pos_emb_in, pos_buf.data(), 0, pos_buf.size() * sizeof(float));
    }

    transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);

    const int64_t t0 = ggml_time_us();
    if (ggml_backend_sched_graph_compute(cc->sched, eb.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_GGUF;
    }
    enc_us += ggml_time_us() - t0;

    const int d_enc = static_cast<int>(eb.out->ne[0]);
    const int T_enc = static_cast<int>(eb.out->ne[1]);
    if (d_enc <= 0 || T_enc <= 0) {
        return TRANSCRIBE_ERR_GGUF;
    }
    enc_host_out.resize(static_cast<size_t>(d_enc) * T_enc);
    ggml_backend_tensor_get(eb.out, enc_host_out.data(), 0, enc_host_out.size() * sizeof(float));
    T_enc_out = T_enc;
    return TRANSCRIBE_OK;
}

transcribe_status run_batch_serial(CohereSession *               cc,
                                   const float * const *         pcm,
                                   const int *                   n_samples,
                                   int                           n,
                                   const transcribe_run_params * params) {
    // Session-level truncation is the OR across utterances; the exact
    // per-utterance status stays in batch_results. Same shape as arch/moss.
    bool any_truncated = false;

    for (int i = 0; i < n; ++i) {
        if (cc->poll_abort()) {
            cc->was_truncated = any_truncated;
            return TRANSCRIBE_ERR_ABORTED;
        }

        // Per-utterance state. transcribe_run_batch() resets was_truncated once
        // for the whole call, so without this a single truncated utterance would
        // leave every later one flagged truncated as well. clear_result() covers
        // the case where run() is skipped entirely (invalid args) and the
        // previous utterance's text would otherwise be captured again.
        cc->clear_result();
        cc->was_truncated = false;
        cc->t_mel_us      = 0;
        cc->t_encode_us   = 0;
        cc->t_decode_us   = 0;

        const transcribe_status st = (pcm[i] == nullptr || n_samples[i] <= 0) ? TRANSCRIBE_ERR_INVALID_ARG :
                                                                                run(cc, pcm[i], n_samples[i], params);
        any_truncated              = any_truncated || st == TRANSCRIBE_ERR_OUTPUT_TRUNCATED;

        // Capture unconditionally, matching run_batched_encdec_*: a non-OK
        // status that still produced text (OUTPUT_TRUNCATED, ABORTED) keeps that
        // partial transcript readable. Discarding it here would contradict
        // docs/input-limits.md, which guarantees the partial output survives.
        cc->batch_results.push_back(cc->capture_result(st));

        // A cancel that lands *inside* an utterance ends the batch. The
        // poll_abort() at the top of the loop is not enough on its own: an
        // edge-triggered callback may report true only once, and run() has
        // already consumed it, so the remaining utterances would transcribe
        // normally. Returning ABORTED lets the dispatcher fill the remaining
        // slots via pad_batch_results_aborted().
        if (st == TRANSCRIBE_ERR_ABORTED) {
            cc->was_truncated = any_truncated;
            return TRANSCRIBE_ERR_ABORTED;
        }
    }

    cc->was_truncated = any_truncated;
    return TRANSCRIBE_OK;
}

transcribe_status run_batch(transcribe_session *          session,
                            const float * const *         pcm,
                            const int *                   n_samples,
                            int                           n,
                            const transcribe_run_params * params) {
    if (session == nullptr || pcm == nullptr || n_samples == nullptr || n <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<CohereSession *>(session);
    auto * cm = static_cast<CohereModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() || !cm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Batched decode requires the flash step path; fall back to serial for
    // n==1, the CPU/manual path, or when debug dumping is active.
    const bool primary_is_gpu = cm->plan.primary_kind != transcribe::BackendKind::Cpu &&
                                cm->plan.primary_kind != transcribe::BackendKind::Accel &&
                                cm->plan.primary_kind != transcribe::BackendKind::Unknown;

    // Long-form windowing lives in run(), which the serial path calls per
    // utterance. The batched decoder assumes exactly one encode per utterance
    // and has nowhere to fan one input out into several windows, so a batch
    // carrying any long clip goes serial. Without this, the same audio would
    // be silently truncated or not depending on backend and batch size.
    const int sample_rate    = cm->hparams.fe_sample_rate > 0 ? cm->hparams.fe_sample_rate : 16000;
    const int window_samples = static_cast<int>(kTrainedClipSeconds * sample_rate);
    bool      any_long_form  = false;
    for (int i = 0; i < n; ++i) {
        if (n_samples[i] > window_samples) {
            any_long_form = true;
            break;
        }
    }

    if (n == 1 || !cc->decoder_use_flash || !primary_is_gpu || transcribe::debug::enabled() || any_long_form) {
        return run_batch_serial(cc, pcm, n_samples, n, params);
    }

    transcribe::debug::init();
    const auto & hp      = cm->hparams;
    const int    hidden  = hp.dec_hidden;
    const int    n_layer = hp.dec_n_layers;

    // ----- Shared prompt (identical across the batch) -----
    const char *                   lang          = (params && params->language) ? params->language : "en";
    const std::string              lang_token    = std::string("<|") + lang + "|>";
    const std::vector<std::string> prompt_pieces = {
        "\xe2\x96\x81", "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>", lang_token, lang_token,
        "<|pnc|>",      "<|noitn|>",          "<|notimestamp|>",       "<|nodiarize|>",
    };
    std::vector<int32_t> prompt_ids;
    for (const auto & piece : prompt_pieces) {
        const int id = cm->tok.find(piece);
        if (id < 0) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        prompt_ids.push_back(id);
    }
    const int prompt_len = static_cast<int>(prompt_ids.size());

    // ----- Pass 0: parallel mel -----
    std::vector<char>               valid(n, 0);
    std::vector<std::vector<float>> mel_bufs(n);
    std::vector<int>                mel_nf(n, 0);
    int                             n_threads = cc->n_threads;
    if (n_threads <= 0) {
        n_threads = transcribe::default_n_threads();
    }
    int64_t       mel_us = 0, enc_us = 0;
    const int64_t t_mel0 = ggml_time_us();
    transcribe::parallel_for_all(n, n_threads, [&](int b) {
        if (pcm[b] == nullptr || n_samples[b] <= 0) {
            return true;
        }
        int nm = 0, nf = 0;
        if (cm->mel->compute(pcm[b], static_cast<size_t>(n_samples[b]), mel_bufs[b], nm, nf, /*n_threads=*/1) ==
                TRANSCRIBE_OK &&
            nf > 0) {
            mel_nf[b] = nf;
            valid[b]  = 1;
        }
        return true;
    });
    mel_us += ggml_time_us() - t_mel0;

    // ----- Pass 1: serial per-utterance encoder -----
    // Per-utterance input-length gate (same enc_pos_emb_max_len limit as
    // run()). An over-length utterance is marked invalid with INPUT_TOO_LONG
    // so the rest of the batch still decodes.
    std::vector<transcribe_status>  reject_status(n, TRANSCRIBE_ERR_INVALID_ARG);
    std::vector<std::vector<float>> enc_hosts(n);
    std::vector<int>                T_enc(n, 0);
    int                             T_enc_max = 0;
    for (int b = 0; b < n; ++b) {
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        if (!valid[b]) {
            continue;
        }
        if (cm->hparams.enc_pos_emb_max_len > 0) {
            const int t_enc_pred = cohere_predict_t_enc(mel_nf[b], cm->hparams.enc_subsampling_factor);
            if (t_enc_pred > cm->hparams.enc_pos_emb_max_len) {
                const double max_s = static_cast<double>(cohere_max_audio_ms(cm->hparams)) / 1000.0;
                transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                    "cohere run_batch: utterance %d too long — %d encoder frames "
                                    "exceed the %d the model supports (~%.0f s max). See "
                                    "transcribe_capabilities.max_audio_ms.",
                                    b, t_enc_pred, cm->hparams.enc_pos_emb_max_len, max_s);
                reject_status[b] = TRANSCRIBE_ERR_INPUT_TOO_LONG;
                valid[b]         = 0;
                continue;
            }
        }
        if (encode_one_to_host(cc, cm, mel_bufs[b], mel_nf[b], enc_hosts[b], T_enc[b], enc_us) != TRANSCRIBE_OK ||
            T_enc[b] <= 0) {
            valid[b] = 0;
            continue;
        }
        T_enc_max = std::max(T_enc_max, T_enc[b]);
    }
    if (T_enc_max <= 0) {
        for (int b = 0; b < n; ++b) {
            transcribe_session::ResultSet rs;
            rs.status = reject_status[b];
            cc->batch_results.push_back(std::move(rs));
        }
        return TRANSCRIBE_OK;
    }

    // ----- Allocate batched KV cache -----
    const int max_new  = std::min(512, /*budget*/ 4096);
    int       max_n_kv = 1024;
    while (max_n_kv < prompt_len + max_new) {
        max_n_kv *= 2;
    }
    // Honor the session context cap (same ceiling the single-shot path uses),
    // not the raw model max — so a lowered n_ctx bounds batch decoder KV too.
    const int n_ctx_cap = cohere_dec_ctx_ceiling(cc->n_ctx, hp);
    if (max_n_kv > n_ctx_cap) {
        max_n_kv = n_ctx_cap;
    }

    ggml_type kv_type = (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
    if (cc->kv_cache.buffer != nullptr &&
        (cc->kv_cache.n_batch != n || cc->kv_cache.T_enc != T_enc_max || cc->kv_cache.n_ctx != max_n_kv)) {
        cc->kv_cache.free();
    }
    if (cc->kv_cache.buffer == nullptr) {
        if (!kv_cache_init_batched(cc->kv_cache, cm->plan.primary, max_n_kv, T_enc_max, hidden, n_layer, n, kv_type)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "cohere run_batch: batched KV cache allocation failed — out of memory.");
            return TRANSCRIBE_ERR_OOM;
        }
    } else {
        ggml_backend_buffer_clear(cc->kv_cache.buffer, 0);
        cc->kv_cache.n               = 0;
        cc->kv_cache.head            = 0;
        cc->kv_cache.cross_populated = false;
    }

    auto new_compute_ctx = [&](size_t mem) -> bool {
        if (cc->compute_ctx != nullptr) {
            ggml_free(cc->compute_ctx);
            cc->compute_ctx = nullptr;
        }
        cc->encoder_out = nullptr;
        ggml_init_params ip{};
        ip.mem_size     = mem;
        ip.mem_buffer   = nullptr;
        ip.no_alloc     = true;
        cc->compute_ctx = ggml_init(ip);
        return cc->compute_ctx != nullptr;
    };

    const int64_t t_dec0 = ggml_time_us();

    // ----- Batched cross-attention K/V -----
    {
        if (!new_compute_ctx(8 * 1024 * 1024)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "cohere run_batch: ggml_init for cross_kv failed — out of memory.");
            return TRANSCRIBE_ERR_OOM;
        }
        DecoderBuild cross = build_cross_kv_graph_batched(cc->compute_ctx, cm->weights, hp, cc->kv_cache, T_enc_max, n);
        if (cross.graph == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, cross.graph)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "cohere run_batch: cross_kv graph allocation failed — out of memory.");
            return TRANSCRIBE_ERR_OOM;
        }
        // Pack encoder outputs [hidden, T_enc_max, B], zero-padded.
        std::vector<float> packed(static_cast<size_t>(hidden) * T_enc_max * n, 0.0f);
        for (int b = 0; b < n; ++b) {
            if (!valid[b]) {
                continue;
            }
            std::memcpy(packed.data() + static_cast<size_t>(b) * T_enc_max * hidden, enc_hosts[b].data(),
                        static_cast<size_t>(hidden) * T_enc[b] * sizeof(float));
        }
        ggml_backend_tensor_set(cross.encoder_out_in, packed.data(), 0, packed.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(cc->sched, cross.graph) != GGML_STATUS_SUCCESS) {
            return TRANSCRIBE_ERR_GGUF;
        }
        cc->kv_cache.cross_populated = true;
    }

    // ----- Batched step graph with a GROWING self-attention window -----
    // Self-KV holds only the short prompt + transcript (audio is in cross-KV),
    // so a static n_ctx-wide read is mostly empty for short clips. Start the
    // read window at 64 and double it as n_past advances (O(log) rebuilds; the
    // KV cache persists), keeping per-step self-KV bandwidth within 2x of real.
    const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);
    const int32_t     eos_id   = hp.eos_token_id;

    // Fixed cross-pad mask [T_enc_max, 1, 1, B] (re-uploaded after rebuilds).
    std::vector<ggml_fp16_t> cmask(static_cast<size_t>(T_enc_max) * n, f16_ninf);
    for (int b = 0; b < n; ++b) {
        const int     real = valid[b] ? T_enc[b] : 1;  // >=1 valid col avoids NaN
        ggml_fp16_t * base = cmask.data() + static_cast<size_t>(b) * T_enc_max;
        std::fill(base, base + std::min(real, T_enc_max), f16_zero);
    }

    int init_window = 64;
    while (init_window > max_n_kv) {
        init_window /= 2;
    }
    if (init_window < 1) {
        init_window = max_n_kv;
    }

    StepBuildBatched sb{};
    auto             rebuild = [&](int win, transcribe::EncDecStepIO & io) -> bool {
        if (!new_compute_ctx(16 * 1024 * 1024)) {
            return false;
        }
        sb = build_step_graph_batched(cc->compute_ctx, cm->weights, hp, cc->kv_cache, win, T_enc_max, n,
                                      cc->decoder_use_flash);
        if (sb.graph == nullptr || sb.argmax_out == nullptr) {
            return false;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
            return false;
        }
        ggml_backend_tensor_set(sb.cross_mask_in, cmask.data(), 0, cmask.size() * sizeof(ggml_fp16_t));
        io.token_ids = sb.token_ids_in;
        io.pos_ids   = sb.pos_ids_in;
        io.kv_idx    = sb.kv_idx_in;
        io.self_mask = sb.self_mask_in;
        io.argmax    = sb.argmax_out;
        io.graph     = sb.graph;
        return true;
    };

    std::vector<std::vector<int32_t>> generated(n);
    std::vector<char>                 truncated;
    if (const transcribe_status st = transcribe::run_batched_encdec_step_loop(
            cc, cc->sched, rebuild, prompt_ids, prompt_len, init_window, max_new, max_n_kv, eos_id, n, valid, generated,
            /*n_steps_out=*/nullptr, &truncated);
        st != TRANSCRIBE_OK) {
        return st;
    }
    const int64_t dec_us = ggml_time_us() - t_dec0;

    // ----- Capture -----
    const int valid_count = std::max(1, static_cast<int>(std::count(valid.begin(), valid.end(), char(1))));
    for (int b = 0; b < n; ++b) {
        transcribe_session::ResultSet rs;
        if (!valid[b]) {
            rs.status = reject_status[b];
            cc->batch_results.push_back(std::move(rs));
            continue;
        }
        std::string full = cm->tok.decode(generated[b].data(), static_cast<int>(generated[b].size()));
        rs.raw_text      = full;
        if (!full.empty() && full.front() == ' ') {
            full.erase(full.begin());
        }
        transcribe_session::SegmentEntry seg{};
        seg.t0_ms = 0;
        seg.t1_ms = 0;
        seg.text  = full;
        rs.segments.push_back(std::move(seg));
        rs.full_text   = full;
        rs.result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        rs.has_result  = true;
        rs.status      = TRANSCRIBE_OK;
        // Truncation parity with the single-shot path; only override an
        // otherwise-OK status, never a worse one.
        if (rs.status == TRANSCRIBE_OK && b < static_cast<int>(truncated.size()) && truncated[b]) {
            cc->was_truncated = true;
            rs.status         = TRANSCRIBE_ERR_OUTPUT_TRUNCATED;
        }
        rs.t_mel_us    = mel_us / valid_count;
        rs.t_encode_us = enc_us / valid_count;
        rs.t_decode_us = dec_us / valid_count;
        cc->batch_results.push_back(std::move(rs));
    }

    if (transcribe::env::flag("TRANSCRIBE_PERF_DEBUG")) {
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                "cohere run_batch: n=%d T_enc_max=%d kv_cap=%d prompt=%d\n"
                "  mel=%.1fms (parallel)  enc=%.1fms (serial x%d)  decode=%.1fms (batched)",
                n, T_enc_max, max_n_kv, prompt_len, mel_us / 1000.0, enc_us / 1000.0, n, dec_us / 1000.0);
    }
    return TRANSCRIBE_OK;
}

}  // namespace

extern const Arch arch = {
    /* .name             = */ "cohere_asr",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ run_batch,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
};

}  // namespace transcribe::cohere
