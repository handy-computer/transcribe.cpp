// arch/parakeet/multitalker.cpp - multitalker orchestration for bundle
// GGUFs (multitalker Parakeet + embedded streaming Sortformer).
//
// Mirrors NeMo's SpeakerTaggedASR parallel streaming pipeline
// (speech_to_text_multitalker_streaming_infer.py at the pinned rev),
// replayed offline inside one transcribe_run:
//
//   1. The embedded Sortformer runs its streaming AOSC/FIFO forward over
//      the whole clip at the multitalker operating point (chunk = ASR
//      chunk cadence att_context_right+1 = 14 diar frames, lc/rc = 0,
//      spkcache 188, fifo 188, update period 188 — the example's config
//      on the v2.1 checkpoint defaults). The accumulated per-chunk preds
//      equal the causal per-chunk supervision the reference loop feeds
//      each ASR instance.
//   2. Active speakers = columns whose activity exceeds 0.5 anywhere
//      (the offline analog of the reference's per-chunk cache gating,
//      which tops-k by max over the last 2 chunks).
//   3. One ASR decode pass per active speaker over the same audio, with
//      that speaker's supervision: kernel mode (masked_asr=False, the
//      default here) drives the layer-0 speaker-kernel injection; masked
//      mode (masked_asr=True, the NeMo example default, selected via
//      TRANSCRIBE_MULTITALKER_MODE=masked) masks mel features.
//   4. Per-speaker results merge into one speaker-tagged result: segments
//      ordered by start time with speaker_id set, words/tokens re-indexed
//      under them, full_text joined in segment order (SegLST rendering),
//      and the transcript-independent speaker_segment rows emitted from
//      the diarizer preds.
//
// Deliberate deviation from the reference (documented in the family doc):
// the ASR pass is the validated OFFLINE chunked-attention graph over the
// whole clip, not a replay of the cache-aware streaming loop, so encoder
// state at chunk boundaries can differ at numerical (not transcript)
// scale. The streaming API multitalker path will be chunk-faithful by
// construction.

#include "../sortformer/sortformer.h"
#include "ggml.h"
#include "parakeet.h"
#include "transcribe-debug.h"
#include "transcribe-log.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace transcribe::parakeet {

namespace sf = transcribe::sortformer;

namespace {

struct SpeakerDecode {
    int                                           speaker = 0;  // 0-based diar column
    std::vector<transcribe_session::TokenEntry>   tokens;
    std::vector<transcribe_session::WordEntry>    words;
    std::vector<transcribe_session::SegmentEntry> segments;
    std::string                                   full_text;
    std::string                                   raw_text;
};

}  // namespace

// ---- Streaming multitalker passes (bounded per-speaker state) ----
//
// Chunk-faithful mirror of the reference loop (conformer_stream_step +
// instance manager): one global 14-enc-frame chunk grid; per active
// speaker a private streaming cache + decoder-state instance that only
// advances on chunks where cache-gating selects that speaker. Per-speaker
// encoder/decoder working state is O(1) in clip length (~5 MB per speaker
// plus one chunk graph), unlike the offline replay's O(T^2) attention
// buffers. The enclosing one-shot call still retains O(T) input, mel, and
// diarizer-prediction buffers. Per-speaker instances are swapped into the
// session around each emit_streaming_chunk call so the single-speaker
// streaming driver runs unchanged.
static transcribe_status run_streaming_passes(ParakeetSession *             pc,
                                              ParakeetModel *               pm,
                                              const float *                 pcm,
                                              int                           n_samples,
                                              const transcribe_run_params * params,
                                              const std::vector<float> &    probs,
                                              int                           T_diar,
                                              int                           n_spk,
                                              const std::vector<int> &      active,
                                              bool                          use_kernel,
                                              std::vector<SpeakerDecode> &  per_spk,
                                              int64_t &                     mel_us,
                                              int64_t &                     enc_us,
                                              int64_t &                     dec_us) {
    const auto & hp = pm->hparams;

    // ASR mel over the whole clip (row-major [n_mels, n_frames]; each
    // mel row contiguous over frames). This is O(T), about 170 MB at 55 min;
    // unlike offline attention, no O(T^2) buffer is allocated.
    const int64_t      t_mel_start = ggml_time_us();
    std::vector<float> mel;
    int                n_mels = 0, n_frames = 0;
    if (const transcribe_status st = pm->mel->compute(pcm, static_cast<size_t>(n_samples), mel, n_mels, n_frames);
        st != TRANSCRIBE_OK) {
        return st;
    }
    mel_us += ggml_time_us() - t_mel_start;

    const int chunk_enc = hp.enc_att_context_right + 1;                                          // 14
    const int sub       = hp.enc_subsampling_factor;                                             // 8
    const int mel_first = hp.enc_stream_sampling_frames_first + sub * hp.enc_att_context_right;  // 105
    const int mel_chunk = chunk_enc * sub;                                                       // 112
    const int mel_cache = hp.enc_stream_pre_encode_cache_size;                                   // 9
    const int drop_sub  = hp.enc_stream_drop_extra_pre_encoded;                                  // 2

    // Reference cache_gating: speaker s is active in chunk k iff the max
    // of its accumulated preds over the last 2 chunks (incl. current)
    // exceeds 0.5. Applies to BOTH supervision modes (the reference
    // instance selection is mode-independent).
    auto chunk_active = [&](int s, int k) {
        const int begin = std::max(0, (k - 1) * chunk_enc);
        const int end   = std::min((k + 1) * chunk_enc, T_diar);
        float     mx    = 0.0f;
        for (int t = begin; t < end; ++t) {
            mx = std::max(mx, probs[static_cast<size_t>(t) * n_spk + s]);
        }
        return mx > 0.5f;
    };

    struct Inst {
        ParakeetStreamingCaches       caches;
        ParakeetStreamingDecoderState dec;
        std::vector<TdtToken>         raw;
        std::vector<int32_t>          fed;  // absolute enc frame per gathered frame
        std::vector<float>            spk;  // [T_diar] supervision values
        std::vector<float>            bg;
        bool                          started = false;
    };

    std::vector<Inst> inst(active.size());

    auto swap_in = [&](Inst & I) {
        std::swap(pc->stream_caches, I.caches);
        std::swap(pc->stream_dec_state, I.dec);
        std::swap(pc->raw_tokens, I.raw);
    };
    auto free_inst = [&](Inst & I) {
        if (I.caches.buffer != nullptr) {
            transcribe::safe_buffer_free(I.caches.buffer);
            I.caches.buffer = nullptr;
        }
        if (I.caches.ctx != nullptr) {
            ggml_free(I.caches.ctx);
            I.caches.ctx = nullptr;
        }
        if (I.caches.pos_proj_buf != nullptr) {
            transcribe::safe_buffer_free(I.caches.pos_proj_buf);
            I.caches.pos_proj_buf = nullptr;
        }
        if (I.caches.pos_proj_ctx != nullptr) {
            ggml_free(I.caches.pos_proj_ctx);
            I.caches.pos_proj_ctx = nullptr;
        }
    };

    // Per-speaker supervision value arrays (same construction as the
    // offline pass: raw sigmoids for the target, binarized union of the
    // other active speakers for the background).
    for (size_t i = 0; i < active.size(); ++i) {
        const int s = active[i];
        inst[i].spk.resize(static_cast<size_t>(T_diar));
        inst[i].bg.assign(static_cast<size_t>(T_diar), 0.0f);
        for (int t = 0; t < T_diar; ++t) {
            inst[i].spk[static_cast<size_t>(t)] = probs[static_cast<size_t>(t) * n_spk + s];
        }
        for (const int o : active) {
            if (o == s) {
                continue;
            }
            for (int t = 0; t < T_diar; ++t) {
                if (probs[static_cast<size_t>(t) * n_spk + o] > 0.5f) {
                    inst[i].bg[static_cast<size_t>(t)] = 1.0f;
                }
            }
        }
    }

    const int64_t      t_enc_start = ggml_time_us();
    std::vector<float> chunk_buf;
    transcribe_status  st = TRANSCRIBE_OK;

    for (int k = 0;; ++k) {
        const int c0_new = (k == 0) ? 0 : mel_first + (k - 1) * mel_chunk;  // first NEW mel col
        if (c0_new >= n_frames) {
            break;
        }
        const int c0      = (k == 0) ? 0 : c0_new - mel_cache;  // history prepend
        const int c1      = std::min(c0_new + ((k == 0) ? mel_first : mel_chunk), n_frames);
        const int n_feed  = c1 - c0;
        const int advance = c1 - c0_new;
        if (n_feed <= 0 || advance <= 0) {
            break;
        }

        for (size_t i = 0; i < active.size() && st == TRANSCRIBE_OK; ++i) {
            const int s = active[i];
            if (!chunk_active(s, k)) {
                continue;
            }
            Inst & I = inst[i];

            // Slice this chunk's mel columns (shared grid; the 9-column
            // history prepend reproduces the continuous pre_encode).
            chunk_buf.resize(static_cast<size_t>(n_mels) * static_cast<size_t>(n_feed));
            for (int m = 0; m < n_mels; ++m) {
                std::memcpy(chunk_buf.data() + static_cast<size_t>(m) * n_feed,
                            mel.data() + static_cast<size_t>(m) * n_frames + c0,
                            static_cast<size_t>(n_feed) * sizeof(float));
            }

            // Masked mode: reference mask_features semantics —
            //   masked = mel * mask (masked columns -> 0.0, NOT the
            //   log-silence value); cells whose ORIGINAL mel was exactly
            //   0 -> mask_value; the upsampled mask is left-aligned to
            //   the chunk's NEW columns (col j -> chunk frame j/8) and
            //   the short side is left-PADDED WITH 0, so the history
            //   prepend columns are always masked.
            if (!use_kernel) {
                const int hist = c0_new - c0;  // 9 (0 for the first chunk)
                for (int c = 0; c < n_feed; ++c) {
                    bool keep = false;
                    if (c >= hist) {
                        const int f = k * chunk_enc + (c - hist) / sub;
                        keep        = f < T_diar && I.spk[static_cast<size_t>(f)] > 0.5f;
                    }
                    for (int m = 0; m < n_mels; ++m) {
                        float & v = chunk_buf[static_cast<size_t>(m) * n_feed + c];
                        if (v == 0.0f) {
                            v = -16.6355f;
                        } else if (!keep) {
                            v = 0.0f;
                        }
                    }
                }
            }

            // Kernel mode: per-chunk supervision over the NEW enc frames.
            const int          t_enc0   = k * chunk_enc;
            const int          mask_len = std::max(0, std::min(chunk_enc, T_diar - t_enc0));
            const float *      spk_ptr  = nullptr;
            const float *      bg_ptr   = nullptr;
            std::vector<float> bg_compat;
            if (use_kernel && mask_len > 0) {
                spk_ptr                         = I.spk.data() + t_enc0;
                bg_ptr                          = I.bg.data() + t_enc0;
                // TRANSCRIBE_MULTITALKER_REF_BG_COMPAT=1 reproduces the
                // reference harness's background-slot indexing bug
                // (multispk_transcribe_utils.get_active_speakers_info
                // takes range(len(active)) minus the SLOT id, so with
                // active slots like [0,2] speaker 0's background tracks
                // the silent slot 1 instead of slot 2). Verified against
                // the reference's own bg_targets dumps. Parity tooling
                // only — the default is the intended union of the other
                // ACTIVE slots.
                static const bool ref_bg_compat = (std::getenv("TRANSCRIBE_MULTITALKER_REF_BG_COMPAT") != nullptr);
                if (ref_bg_compat) {
                    std::vector<int> act_k;
                    for (const int o : active) {
                        if (chunk_active(o, k)) {
                            act_k.push_back(o);
                        }
                    }
                    bg_compat.assign(static_cast<size_t>(mask_len), 0.0f);
                    for (int slot = 0; slot < static_cast<int>(act_k.size()); ++slot) {
                        if (slot == s) {
                            continue;  // reference compares POSITION to SLOT id
                        }
                        for (int t = 0; t < mask_len; ++t) {
                            if (probs[static_cast<size_t>(t_enc0 + t) * n_spk + slot] > 0.5f) {
                                bg_compat[static_cast<size_t>(t)] = 1.0f;
                            }
                        }
                    }
                    bg_ptr = bg_compat.data();
                }
            }

            swap_in(I);
            if (!I.started) {
                // (fields now live on pc after the swap)
                if ((st = init_streaming_caches(pc, pm)) == TRANSCRIBE_OK) {
                    reset_streaming_decoder_state(pc, pm);
                    pc->stream_caches.att_context_left  = hp.enc_att_context_left;
                    pc->stream_caches.att_context_right = hp.enc_att_context_right;
                    pc->stream_caches.channel_len       = 0;
                    pc->stream_caches.chunk_step        = 0;
                }
            }
            if (st == TRANSCRIBE_OK) {
                const int64_t off_before = pc->stream_dec_state.frame_offset;
                st = emit_streaming_chunk(pc, pm, chunk_buf.data(), n_feed, (k == 0) ? 0 : drop_sub, advance, spk_ptr,
                                          bg_ptr, use_kernel ? mask_len : 0);
                if (st == TRANSCRIBE_OK) {
                    const int n_new = static_cast<int>(pc->stream_dec_state.frame_offset - off_before);
                    for (int f = 0; f < n_new; ++f) {
                        I.fed.push_back(t_enc0 + f);
                    }
                }
            }
            swap_in(I);  // swap back
            I.started = true;
        }
        if (st != TRANSCRIBE_OK) {
            break;
        }
    }
    enc_us += ggml_time_us() - t_enc_start;

    // Build per-speaker results from the accumulated raw tokens, with
    // step_at_emit remapped from the gathered timeline to absolute
    // encoder frames through the fed-chunk list.
    const int64_t t_dec_start = ggml_time_us();
    for (size_t i = 0; i < active.size(); ++i) {
        Inst & I = inst[i];
        if (st == TRANSCRIBE_OK && !I.raw.empty()) {
            for (auto & rt : I.raw) {
                const size_t f  = std::min(static_cast<size_t>(std::max(rt.step_at_emit, 0)), I.fed.size() - 1);
                rt.step_at_emit = I.fed[f];
            }
            transcribe_run_params sp = *params;
            sp.timestamps            = TRANSCRIBE_TIMESTAMPS_AUTO;
            pc->clear_result();
            pc->raw_tokens              = std::move(I.raw);
            const transcribe_status bst = build_result_from_raw_tokens(pc, pm, &sp);
            if (bst != TRANSCRIBE_OK) {
                st = bst;
            } else {
                SpeakerDecode r;
                r.speaker   = active[i];
                r.tokens    = std::move(pc->tokens);
                r.words     = std::move(pc->words);
                r.segments  = std::move(pc->segments);
                r.full_text = std::move(pc->full_text);
                r.raw_text  = std::move(pc->raw_text);
                per_spk.push_back(std::move(r));
            }
        }
        free_inst(I);
    }
    dec_us += ggml_time_us() - t_dec_start;
    return st;
}

transcribe_status run_multitalker(ParakeetSession *             pc,
                                  ParakeetModel *               pm,
                                  const float *                 pcm,
                                  int                           n_samples,
                                  const transcribe_run_params * params) {
    if (pm->diar == nullptr || !pm->diar_mel.has_value() || !pm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (pc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    transcribe::debug::init();

    // Supervision mode. Default is the checkpoint's trained speaker-kernel
    // path: on ami-ihm-test it scores 19.35% cpWER vs masked's 23.73%
    // (and vs the NeMo reference's own 21.39% kernel / 24.00% masked).
    // TRANSCRIBE_MULTITALKER_MODE=masked selects mel feature masking
    // (the NeMo example default) until the family run-ext grows a knob.
    bool use_kernel = true;
    if (const char * mode = std::getenv("TRANSCRIBE_MULTITALKER_MODE")) {
        use_kernel = (std::strcmp(mode, "masked") != 0);
    }

    const sf::SortformerEmbedded & diar = *pm->diar;

    // ---- 1. Embedded diarizer streaming forward ----
    std::vector<float> diar_mel;
    int                dm_mels = 0, dm_frames = 0;
    if (const transcribe_status st =
            pm->diar_mel->compute(pcm, static_cast<size_t>(n_samples), diar_mel, dm_mels, dm_frames);
        st != TRANSCRIBE_OK) {
        return st;
    }

    if (pc->sched == nullptr) {
        pc->sched = ggml_backend_sched_new(pm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(pm->plan.scheduler_list.size()),
                                           /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/true);
        if (pc->sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet multitalker: ggml_backend_sched_new failed");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    sf::SortformerStreamParams P = sf::resolve_stream_params(diar.hp, TRANSCRIBE_SORTFORMER_PRESET_DEFAULT);
    P.chunk_len                  = pm->hparams.enc_att_context_right + 1;  // ASR chunk cadence (14 at [70,13])
    P.chunk_left_context         = 0;
    P.chunk_right_context        = 0;
    P.spkcache_len               = 188;
    P.fifo_len                   = 188;
    P.spkcache_update_period     = 188;
    // Reference-exact windowing: the NeMo loop feeds the diarizer the
    // ASR streaming buffer's chunks (chunk_size_first = sampling_frames +
    // sub * R mel frames first, then pre_encode_cache-frame overlapped
    // windows with drop_extra_pre_encoded outputs discarded).
    P.feat_first_chunk           = pm->hparams.enc_stream_sampling_frames_first +
                                   pm->hparams.enc_subsampling_factor * pm->hparams.enc_att_context_right;
    P.feat_cache                 = pm->hparams.enc_stream_pre_encode_cache_size;
    P.drop_pre_encode            = pm->hparams.enc_stream_drop_extra_pre_encoded;

    const int64_t t_diar_start = ggml_time_us();
    transcribe::debug::push_name_prefix("mt.diar");
    const transcribe_status dst = sf::run_diar_streaming_core(
        pc->diar_scratch, diar.hp, diar.conformer_hp, diar.conformer, diar.weights, pm->backend.c_str(), pc->sched,
        pc->n_threads, P, diar_mel.data(), dm_mels, dm_frames, pc);
    transcribe::debug::pop_name_prefix();
    if (dst != TRANSCRIBE_OK) {
        return dst;
    }
    const int64_t t_diar_us = ggml_time_us() - t_diar_start;

    const int                  sub    = diar.hp.enc_subsampling_factor > 0 ? diar.hp.enc_subsampling_factor : 8;
    const int                  n_spk  = diar.hp.max_speakers;
    const int                  T_diar = std::min(pc->diar_scratch.stream.total_n, (dm_frames + sub - 1) / sub);
    const std::vector<float> & probs  = pc->diar_scratch.stream.total_preds;  // row-major [total_n, n_spk]

    if (transcribe::debug::enabled()) {
        const long long shape[2] = { T_diar, n_spk };
        transcribe::debug::dump_host_f32("mt.diar.probs", probs.data(), static_cast<long long>(T_diar) * n_spk, shape,
                                         2, "diarize");
    }

    // ---- 2. Active speakers ----
    std::vector<int> active;
    for (int s = 0; s < n_spk; ++s) {
        float mx = 0.0f;
        for (int t = 0; t < T_diar; ++t) {
            mx = std::max(mx, probs[static_cast<size_t>(t) * n_spk + s]);
        }
        if (mx > 0.5f) {
            active.push_back(s);
        }
    }

    const double diar_ms_per_frame =
        1000.0 * static_cast<double>(diar.hp.frame_hop) / static_cast<double>(diar.hp.fe_sample_rate);

    if (active.empty()) {
        // No speech activity detected anywhere (reference behavior: every
        // chunk skipped). Empty-but-valid result.
        pc->t_mel_us    = 0;
        pc->t_encode_us = t_diar_us;
        pc->t_decode_us = 0;
        pc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        pc->has_result  = true;
        return TRANSCRIBE_OK;
    }

    // ---- 3. Per-speaker decode passes ----
    //
    // Default is the bounded-memory streaming pass (chunk-faithful to the
    // reference loop). TRANSCRIBE_MULTITALKER_OFFLINE=1 selects the
    // offline whole-clip replay (O(T^2) attention buffers; parity/dump
    // tooling only).
    std::vector<SpeakerDecode> per_spk;
    per_spk.reserve(active.size());
    int64_t mel_us = 0, enc_us = 0, dec_us = 0;

    const bool offline_replay = (std::getenv("TRANSCRIBE_MULTITALKER_OFFLINE") != nullptr);
    if (!offline_replay) {
        if (const transcribe_status st = run_streaming_passes(pc, pm, pcm, n_samples, params, probs, T_diar, n_spk,
                                                              active, use_kernel, per_spk, mel_us, enc_us, dec_us);
            st != TRANSCRIBE_OK) {
            return st;
        }
    } else {
        for (const int s : active) {
            MultitalkerPass pass;
            pass.use_kernel = use_kernel;
            pass.spk.resize(static_cast<size_t>(T_diar));
            pass.bg.assign(static_cast<size_t>(T_diar), 0.0f);
            for (int t = 0; t < T_diar; ++t) {
                pass.spk[static_cast<size_t>(t)] = probs[static_cast<size_t>(t) * n_spk + s];
            }
            for (const int o : active) {
                if (o == s) {
                    continue;
                }
                for (int t = 0; t < T_diar; ++t) {
                    if (probs[static_cast<size_t>(t) * n_spk + o] > 0.5f) {
                        pass.bg[static_cast<size_t>(t)] = 1.0f;
                    }
                }
            }

            // Kernel-mode chunk gating (reference cache_gating=true,
            // cache_gating_buffer_size=2): a speaker's conformer/decoder only
            // ever sees chunks where _find_active_speakers selected it — max
            // of the accumulated preds over the last 2 chunks (incl. current)
            // > 0.5. Without this, a mostly-inactive speaker's pass
            // transcribes everyone else's speech (the trained kernels only
            // ever saw gated chunks). Masked mode hard-silences non-target
            // features instead, so it stays ungated here.
            if (use_kernel) {
                const int chunk    = P.chunk_len;  // 14 = nframes_per_chunk
                const int n_chunks = (T_diar + chunk - 1) / chunk;
                // +1 covers the encoder length trailing the diar frame count
                // at the tail; run_one_shot_inner clamps to the true length.
                pass.keep_enc_frames.reserve(static_cast<size_t>(T_diar) + 1);
                for (int k = 0; k < n_chunks; ++k) {
                    const int end   = std::min((k + 1) * chunk, T_diar);
                    const int begin = std::max(0, (k - 1) * chunk);  // 2-chunk window
                    float     mx    = 0.0f;
                    for (int t = begin; t < end; ++t) {
                        mx = std::max(mx, probs[static_cast<size_t>(t) * n_spk + s]);
                    }
                    if (mx > 0.5f) {
                        const int enc_end = (k == n_chunks - 1) ? end + 1 : end;
                        for (int t = k * chunk; t < enc_end; ++t) {
                            pass.keep_enc_frames.push_back(t);
                        }
                    }
                }
                if (pass.keep_enc_frames.empty()) {
                    continue;  // globally active but never chunk-selected
                }
            }

            // Decode at full timestamp resolution; the merged result is elided
            // to the caller's request at the end.
            transcribe_run_params sp = *params;
            sp.timestamps            = TRANSCRIBE_TIMESTAMPS_AUTO;

            pc->clear_result();
            char prefix[32];
            std::snprintf(prefix, sizeof(prefix), "mt.spk%d", s);
            transcribe::debug::push_name_prefix(prefix);
            const transcribe_status st = run_one_shot_inner(pc, pm, pcm, n_samples, &sp, &pass);
            transcribe::debug::pop_name_prefix();
            if (st != TRANSCRIBE_OK) {
                return st;
            }
            mel_us += pc->t_mel_us;
            enc_us += pc->t_encode_us;
            dec_us += pc->t_decode_us;

            SpeakerDecode r;
            r.speaker   = s;
            r.tokens    = std::move(pc->tokens);
            r.words     = std::move(pc->words);
            r.segments  = std::move(pc->segments);
            r.full_text = std::move(pc->full_text);
            r.raw_text  = std::move(pc->raw_text);

            // Gated pass timestamps are in the gathered timeline; map each
            // frame back through the keep list to absolute clip time.
            if (!pass.keep_enc_frames.empty()) {
                const auto & keep     = pass.keep_enc_frames;
                const double f_ms     = diar_ms_per_frame;  // 80 ms encoder cadence
                auto         remap_t0 = [&](int64_t t_ms) {
                    size_t f = static_cast<size_t>(std::max<int64_t>(0, t_ms) / static_cast<int64_t>(f_ms));
                    f        = std::min(f, keep.size() - 1);
                    return static_cast<int64_t>(keep[f]) * static_cast<int64_t>(f_ms);
                };
                auto remap_t1 = [&](int64_t t_ms) {
                    size_t f = static_cast<size_t>(std::max<int64_t>(0, t_ms - 1) / static_cast<int64_t>(f_ms));
                    f        = std::min(f, keep.size() - 1);
                    return static_cast<int64_t>(keep[f] + 1) * static_cast<int64_t>(f_ms);
                };
                for (auto & tk : r.tokens) {
                    tk.t0_ms = remap_t0(tk.t0_ms);
                    tk.t1_ms = remap_t1(tk.t1_ms);
                }
                for (auto & wd : r.words) {
                    wd.t0_ms = remap_t0(wd.t0_ms);
                    wd.t1_ms = remap_t1(wd.t1_ms);
                }
                for (auto & sg : r.segments) {
                    sg.t0_ms = remap_t0(sg.t0_ms);
                    sg.t1_ms = remap_t1(sg.t1_ms);
                }
            }
            per_spk.push_back(std::move(r));
        }
    }

    // ---- 4. Merge into one speaker-tagged result ----
    pc->clear_result();

    struct SegRef {
        const SpeakerDecode * r;
        int                   seg_idx;
    };

    std::vector<SegRef> order;
    for (const SpeakerDecode & r : per_spk) {
        for (size_t i = 0; i < r.segments.size(); ++i) {
            order.push_back({ &r, static_cast<int>(i) });
        }
    }
    std::stable_sort(order.begin(), order.end(), [](const SegRef & a, const SegRef & b) {
        return a.r->segments[static_cast<size_t>(a.seg_idx)].t0_ms <
               b.r->segments[static_cast<size_t>(b.seg_idx)].t0_ms;
    });

    std::string merged_text;
    std::string merged_raw;
    for (const SegRef & ref : order) {
        const SpeakerDecode &                    r   = *ref.r;
        const transcribe_session::SegmentEntry & src = r.segments[static_cast<size_t>(ref.seg_idx)];

        const int new_seg_idx = static_cast<int>(pc->segments.size());
        const int word_base   = static_cast<int>(pc->words.size());
        const int token_base  = static_cast<int>(pc->tokens.size());

        for (int j = 0; j < src.n_tokens; ++j) {
            transcribe_session::TokenEntry te = r.tokens[static_cast<size_t>(src.first_token + j)];
            te.seg_index                      = new_seg_idx;
            te.word_index                     = te.word_index - src.first_word + word_base;
            pc->tokens.push_back(std::move(te));
        }
        for (int j = 0; j < src.n_words; ++j) {
            transcribe_session::WordEntry we = r.words[static_cast<size_t>(src.first_word + j)];
            we.seg_index                     = new_seg_idx;
            we.first_token                   = we.first_token - src.first_token + token_base;
            pc->words.push_back(std::move(we));
        }

        transcribe_session::SegmentEntry seg = src;
        seg.first_word                       = word_base;
        seg.first_token                      = token_base;
        seg.speaker_id                       = r.speaker + 1;  // 1-based
        pc->segments.push_back(std::move(seg));

        if (!src.text.empty()) {
            if (!merged_text.empty()) {
                merged_text += ' ';
            }
            merged_text += src.text;
        }
        // One raw-text contribution per speaker, in first-segment order.
        if (ref.seg_idx == 0 && !r.raw_text.empty()) {
            if (!merged_raw.empty()) {
                merged_raw += ' ';
            }
            merged_raw += r.raw_text;
        }
    }
    pc->full_text = std::move(merged_text);
    pc->raw_text  = std::move(merged_raw);

    // Transcript-independent "who spoke when" rows from the diarizer preds
    // (same emission as the standalone sortformer family).
    sf::probs_to_speaker_segments(pc, probs, T_diar, n_spk, diar_ms_per_frame, /*threshold=*/0.5f);

    // ---- Elide to the caller's requested granularity (same convention
    // as decode_and_populate). ----
    transcribe_timestamp_kind eff = params->timestamps;
    if (eff == TRANSCRIBE_TIMESTAMPS_AUTO) {
        eff = pm->caps.max_timestamp_kind;  // TOKEN for parakeet
    }
    if (eff == TRANSCRIBE_TIMESTAMPS_NONE) {
        pc->tokens.clear();
        pc->words.clear();
        for (auto & s : pc->segments) {
            s.t0_ms       = 0;
            s.t1_ms       = 0;
            s.first_word  = 0;
            s.n_words     = 0;
            s.first_token = 0;
            s.n_tokens    = 0;
        }
    } else if (eff == TRANSCRIBE_TIMESTAMPS_SEGMENT) {
        pc->tokens.clear();
        pc->words.clear();
        for (auto & s : pc->segments) {
            s.first_word  = 0;
            s.n_words     = 0;
            s.first_token = 0;
            s.n_tokens    = 0;
        }
    } else if (eff == TRANSCRIBE_TIMESTAMPS_WORD) {
        pc->tokens.clear();
        for (auto & w : pc->words) {
            w.first_token = 0;
            w.n_tokens    = 0;
        }
        for (auto & s : pc->segments) {
            s.first_token = 0;
            s.n_tokens    = 0;
        }
    }

    pc->t_mel_us    = mel_us;
    pc->t_encode_us = enc_us + t_diar_us;
    pc->t_decode_us = dec_us;
    pc->result_kind = eff;
    pc->has_result  = true;
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::parakeet
