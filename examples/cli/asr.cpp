// asr.cpp - transcribe-cli ASR driver: batch and single-file transcription
// through the public ASR API (transcribe_run, transcribe_run_batch,
// transcribe_stream_*), plus the JSONL / text output for them.

#include "cli.h"
#include "transcribe.h"
#include "transcribe/parakeet.h"
#include "transcribe/voxtral_realtime.h"
#include "transcribe/whisper.h"
#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using transcribe_cli::cli_args;

namespace {

// Minimal JSON string escape: covers the characters MUST be escaped by
// the JSON spec (quote, backslash, control chars). Transcribed text is
// short UTF-8 in practice; we don't need unicode escaping.
std::string json_escape(const char * s) {
    std::string out;
    for (const char * p = s ? s : ""; *p; ++p) {
        const unsigned char c = static_cast<unsigned char>(*p);
        if (c == '"') {
            out += "\\\"";
        } else if (c == '\\') {
            out += "\\\\";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else {
            out += static_cast<char>(c);
        }
    }
    return out;
}

// Shared row formatter for segments_json / batch_segments_json below.
std::string segment_row_json(const struct transcribe_segment & seg) {
    std::string out;
    char        head[128];
    if (seg.speaker_id > 0) {
        std::snprintf(head, sizeof(head), "{\"t0_ms\":%lld,\"t1_ms\":%lld,\"speaker_id\":%d,\"text\":\"",
                      static_cast<long long>(seg.t0_ms), static_cast<long long>(seg.t1_ms),
                      static_cast<int>(seg.speaker_id));
    } else {
        std::snprintf(head, sizeof(head), "{\"t0_ms\":%lld,\"t1_ms\":%lld,\"text\":\"",
                      static_cast<long long>(seg.t0_ms), static_cast<long long>(seg.t1_ms));
    }
    out += head;
    out += json_escape(seg.text != nullptr ? seg.text : "");
    out += "\"}";
    return out;
}

// Build the ",\"segments\":[...]" fragment when the context has real
// segment rows: segment timestamps, or speaker-attributed turns (which
// may carry no timing — granite's speaker-attribution task). Returns an
// empty string otherwise (the library still populates a single dummy
// entry when the kind is NONE, but with zero timing — don't pollute the
// JSON with it).
std::string segments_json(const transcribe_session * ctx) {
    if (transcribe_returned_timestamp_kind(ctx) == TRANSCRIBE_TIMESTAMPS_NONE &&
        transcribe_n_speaker_segments(ctx) <= 0) {
        return {};
    }
    const int n_seg = transcribe_n_segments(ctx);
    if (n_seg <= 0) {
        return {};
    }
    std::string out = ",\"segments\":[";
    for (int s = 0; s < n_seg; ++s) {
        if (s > 0) {
            out += ",";
        }
        struct transcribe_segment seg;
        transcribe_segment_init(&seg);
        (void) transcribe_get_segment(ctx, s, &seg);
        out += segment_row_json(seg);
    }
    out += "]";
    return out;
}

// Batch variant: segments JSON for utterance `i` of a transcribe_run_batch
// result, using the indexed transcribe_batch_* accessors.
std::string batch_segments_json(const transcribe_session * ctx, int i) {
    if (transcribe_batch_returned_timestamp_kind(ctx, i) == TRANSCRIBE_TIMESTAMPS_NONE &&
        transcribe_batch_n_speaker_segments(ctx, i) <= 0) {
        return {};
    }
    const int n_seg = transcribe_batch_n_segments(ctx, i);
    if (n_seg <= 0) {
        return {};
    }
    std::string out = ",\"segments\":[";
    for (int s = 0; s < n_seg; ++s) {
        if (s > 0) {
            out += ",";
        }
        struct transcribe_segment seg;
        transcribe_segment_init(&seg);
        (void) transcribe_batch_get_segment(ctx, i, s, &seg);
        out += segment_row_json(seg);
    }
    out += "]";
    return out;
}

// ",\"raw_text\":\"...\"" fragment: the model's pre-cleanup decode
// (transcribe_raw_text). Emitted only when it differs from the clean text,
// so JSONL stays compact for families with no post-processing.
std::string raw_text_json(const char * raw, const char * clean) {
    if (raw == nullptr || raw[0] == '\0' || (clean != nullptr && std::strcmp(raw, clean) == 0)) {
        return {};
    }
    std::string out = ",\"raw_text\":\"";
    out += json_escape(raw);
    out += "\"";
    return out;
}

// A decode cut short before end-of-stream (budget or repetition stop): non-OK,
// but the partial transcript is preserved (see docs/input-limits.md).
bool is_cut_short(transcribe_status st) {
    return st == TRANSCRIBE_ERR_OUTPUT_TRUNCATED || st == TRANSCRIBE_ERR_OUTPUT_REPETITION;
}

const char * cut_short_label(transcribe_status st) {
    return st == TRANSCRIBE_ERR_OUTPUT_REPETITION ? "stopped repeating" : "truncated";
}

// ",\"speakers\":[...]" fragment: the "who spoke when" rows. Emitted only
// when the run produced speaker segments. p is omitted unless finite
// (NaN — "model provides no confidence" — is not representable in JSON).
std::string speaker_row_json(const struct transcribe_speaker_segment & row) {
    char buf[160];
    if (std::isfinite(row.p)) {
        std::snprintf(buf, sizeof(buf), "{\"t0_ms\":%lld,\"t1_ms\":%lld,\"speaker_id\":%d,\"p\":%.4f}",
                      static_cast<long long>(row.t0_ms), static_cast<long long>(row.t1_ms),
                      static_cast<int>(row.speaker_id), static_cast<double>(row.p));
    } else {
        std::snprintf(buf, sizeof(buf), "{\"t0_ms\":%lld,\"t1_ms\":%lld,\"speaker_id\":%d}",
                      static_cast<long long>(row.t0_ms), static_cast<long long>(row.t1_ms),
                      static_cast<int>(row.speaker_id));
    }
    return buf;
}

std::string speakers_json(const transcribe_session * ctx) {
    const int n = transcribe_n_speaker_segments(ctx);
    if (n <= 0) {
        return {};
    }
    std::string out = ",\"speakers\":[";
    for (int s = 0; s < n; ++s) {
        if (s > 0) {
            out += ",";
        }
        struct transcribe_speaker_segment row;
        transcribe_speaker_segment_init(&row);
        (void) transcribe_get_speaker_segment(ctx, s, &row);
        out += speaker_row_json(row);
    }
    out += "]";
    return out;
}

std::string batch_speakers_json(const transcribe_session * ctx, int i) {
    const int n = transcribe_batch_n_speaker_segments(ctx, i);
    if (n <= 0) {
        return {};
    }
    std::string out = ",\"speakers\":[";
    for (int s = 0; s < n; ++s) {
        if (s > 0) {
            out += ",";
        }
        struct transcribe_speaker_segment row;
        transcribe_speaker_segment_init(&row);
        (void) transcribe_batch_get_speaker_segment(ctx, i, s, &row);
        out += speaker_row_json(row);
    }
    out += "]";
    return out;
}

// Point rp's generic prompting fields at args' storage; vocabulary_ptrs
// backs rp.vocabulary and must outlive the run.
void apply_prompting(const cli_args & args, transcribe_run_params & rp, std::vector<const char *> & vocabulary_ptrs) {
    if (args.instruct) {
        rp.task = TRANSCRIBE_TASK_INSTRUCT;
    }
    vocabulary_ptrs.clear();
    for (const std::string & term : args.vocabulary) {
        vocabulary_ptrs.push_back(term.c_str());
    }
    rp.vocabulary   = vocabulary_ptrs.empty() ? nullptr : vocabulary_ptrs.data();
    rp.n_vocabulary = static_cast<int32_t>(vocabulary_ptrs.size());
    rp.prompt       = args.prompt.empty() ? nullptr : args.prompt.c_str();
    rp.prefix       = args.prefix.empty() ? nullptr : args.prefix.c_str();
}

}  // namespace

namespace transcribe_cli {

bool write_output_file(std::ofstream * output, const std::string & path, const char * text) {
    if (output == nullptr) {
        return true;
    }
    const char * value = text != nullptr ? text : "";
    *output << value;
    if (value[0] == '\0' || value[std::strlen(value) - 1] != '\n') {
        *output << '\n';
    }
    output->flush();
    if (!*output) {
        std::fprintf(stderr, "error: cannot write %s\n", path.c_str());
        return false;
    }
    return true;
}

int run_asr_batch(const cli_args & args, std::ofstream * output) {
    bool output_ok = true;
    if (args.model_path.empty()) {
        std::fprintf(stderr, "error: --batch requires --model\n");
        return EXIT_FAILURE;
    }

    std::vector<std::string> wav_paths;
    {
        std::ifstream fin(args.batch_file);
        if (!fin) {
            std::fprintf(stderr, "error: cannot open batch file %s\n", args.batch_file.c_str());
            return EXIT_FAILURE;
        }
        std::string line;
        while (std::getline(fin, line)) {
            while (!line.empty() &&
                   (line.back() == '\n' || line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
            if (!line.empty()) {
                wav_paths.push_back(line);
            }
        }
    }
    if (wav_paths.empty()) {
        std::fprintf(stderr, "error: batch file is empty\n");
        return EXIT_FAILURE;
    }
    if (!args.batch_jsonl) {
        std::fprintf(stderr, "batch: %zu files\n", wav_paths.size());
    }

    struct transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend = args.backend;
    mp.device  = args.device_index >= 0 ? transcribe_device_get(args.device_index) : nullptr;
    if (args.device_index >= 0 && mp.device == nullptr) {
        std::fprintf(stderr, "error: --device index %d is not available\n", args.device_index);
        return EXIT_FAILURE;
    }
    struct transcribe_model * model   = nullptr;
    const transcribe_status   load_st = transcribe_model_load_file(args.model_path.c_str(), &mp, &model);
    if (load_st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "model load: %s\n", transcribe_status_string(load_st));
        return EXIT_FAILURE;
    }
    if ((transcribe_model_roles(model) & TRANSCRIBE_ROLE_ASR) == 0) {
        std::fprintf(stderr,
                     "error: --batch is ASR-only and this model has no ASR role; "
                     "run it on one file instead: transcribe-cli -m MODEL audio.wav\n");
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    struct transcribe_session_params cp;
    transcribe_session_params_init(&cp);
    cp.n_threads                        = args.n_threads;
    cp.n_ctx                            = args.n_ctx;
    cp.kv_type                          = args.kv_type;
    struct transcribe_session * ctx     = nullptr;
    const transcribe_status     init_st = transcribe_session_init(model, &cp, &ctx);
    if (init_st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "context init: %s\n", transcribe_status_string(init_st));
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    struct transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    if (args.translate) {
        rp.task = TRANSCRIBE_TASK_TRANSLATE;
    }
    std::vector<const char *> vocabulary_ptrs;
    apply_prompting(args, rp, vocabulary_ptrs);
    if (!args.language.empty()) {
        rp.language = args.language.c_str();
    }
    if (!args.target_language.empty()) {
        rp.target_language = args.target_language.c_str();
    }
    rp.timestamps    = args.timestamps;
    rp.spec_k_drafts = args.spec_k_drafts;

    if (args.itn_set) {
        rp.itn = args.use_itn ? TRANSCRIBE_ITN_MODE_ON : TRANSCRIBE_ITN_MODE_OFF;
    }
    if (args.canary_pnc_set) {
        rp.pnc = args.canary_pnc ? TRANSCRIBE_PNC_MODE_ON : TRANSCRIBE_PNC_MODE_OFF;
    }
    if (args.diarize_set) {
        rp.diarize = args.diarize ? TRANSCRIBE_DIARIZE_MODE_ON : TRANSCRIBE_DIARIZE_MODE_OFF;
    }

    // Whisper run extension. Allocated outside rp's scope so its
    // bytes outlive the per-file loop below; the library copies
    // initial_prompt/prompt_tokens before transcribe_run returns,
    // but rp aliases &wx.ext for the run call itself.
    struct transcribe_whisper_run_ext wx;
    transcribe_whisper_run_ext_init(&wx);
    if (args.whisper_set) {
        if (!args.initial_prompt.empty()) {
            wx.initial_prompt = args.initial_prompt.c_str();
        }
        wx.condition_on_prev_tokens = args.condition_on_prev_tokens;
        if (args.temperature_set) {
            wx.temperature = args.temperature;
        }
        wx.prompt_condition = args.prompt_condition;
        if (transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_WHISPER_RUN)) {
            rp.family = &wx.ext;
        }
    }

    if (args.keep_special_tags) {
        rp.keep_special_tags = true;
    }

    // Emit a batch header line once, before any per-file output. Carries
    // the one-shot load time so downstream WER tooling can record it
    // without parsing stderr. Per-file lines follow on subsequent lines.
    if (args.batch_jsonl) {
        struct transcribe_timings load_tm;
        transcribe_timings_init(&load_tm);
        (void) transcribe_get_timings(ctx, &load_tm);
        std::printf("{\"type\":\"batch_header\",\"load_ms\":%.1f}\n", (double) load_tm.load_ms);
        std::fflush(stdout);
    }

    int n_ok        = 0;
    int n_truncated = 0;  // result-bearing: hit the generation cap, partial hyp emitted
    int n_repeating = 0;  // result-bearing: stopped when the output looped, partial hyp emitted
    int n_fail      = 0;  // no usable result (wav load / backend / unsupported / whole-batch)

    // Offline batched path: group up to batch_size utterances into one
    // transcribe_run_batch call. Mutually exclusive with the streaming
    // path (--stream-chunk-ms), which is per-utterance by construction.
    // Per-file JSONL output is byte-identical in shape to the serial
    // path so the WER harness (scripts/wer/run.py) consumes either.
    const bool use_batched = args.batch_size > 1 && args.stream_chunk_ms <= 0;
    if (use_batched) {
        const size_t total = wav_paths.size();
        const size_t group = static_cast<size_t>(args.batch_size);
        for (size_t base = 0; base < total; base += group) {
            const size_t end = std::min(total, base + group);

            // Load each wav in the group. A wav that fails to load is
            // emitted as an error line and excluded from the batch, so
            // its failure cannot abort its neighbours.
            std::vector<std::vector<float>> pcms;
            std::vector<const float *>      pcm_ptrs;
            std::vector<int>                n_samps;
            std::vector<size_t>             src_index;  // -> wav_paths
            for (size_t i = base; i < end; ++i) {
                std::vector<float> pcm;
                std::string        wav_err;
                if (!transcribe_cli::load_wav_mono_16k(wav_paths[i], pcm, wav_err)) {
                    if (args.batch_jsonl) {
                        const std::string file_esc = json_escape(wav_paths[i].c_str());
                        std::printf(
                            "{\"file\":\"%s\",\"text\":\"\","
                            "\"error\":\"wav: %s\"}\n",
                            file_esc.c_str(), json_escape(wav_err.c_str()).c_str());
                    } else {
                        std::fprintf(stderr, "SKIP %s: %s\n", wav_paths[i].c_str(), wav_err.c_str());
                    }
                    ++n_fail;
                    std::fflush(stdout);
                    continue;
                }
                pcms.push_back(std::move(pcm));
                src_index.push_back(i);
            }
            for (auto & p : pcms) {
                pcm_ptrs.push_back(p.data());
                n_samps.push_back(static_cast<int>(p.size()));
            }
            if (pcms.empty()) {
                continue;
            }

            const transcribe_status bst =
                transcribe_run_batch(ctx, pcm_ptrs.data(), n_samps.data(), static_cast<int>(pcm_ptrs.size()), &rp);
            if (bst != TRANSCRIBE_OK && transcribe_batch_n_results(ctx) == 0) {
                // Whole-batch failure (no per-utterance results): emit an
                // error line per file in the group and continue.
                for (size_t k = 0; k < src_index.size(); ++k) {
                    if (args.batch_jsonl) {
                        const std::string file_esc = json_escape(wav_paths[src_index[k]].c_str());
                        std::printf(
                            "{\"file\":\"%s\",\"text\":\"\","
                            "\"error\":\"%s\"}\n",
                            file_esc.c_str(), json_escape(transcribe_status_string(bst)).c_str());
                    }
                    ++n_fail;
                }
                std::fflush(stdout);
                continue;
            }

            for (size_t k = 0; k < src_index.size(); ++k) {
                const std::string &     wav            = wav_paths[src_index[k]];
                const transcribe_status ust            = transcribe_batch_status(ctx, static_cast<int>(k));
                // OUTPUT_TRUNCATED / OUTPUT_REPETITION are result-bearing: the
                // partial transcript is preserved and readable via
                // transcribe_batch_full_text (see transcribe.h). Emit it as the hyp
                // so downstream tooling scores the partial rather than an empty
                // string; the error field below still tags it so the stop stays
                // visible.
                const bool              result_present = ust == TRANSCRIBE_OK || is_cut_short(ust);
                const char *            text           = "";
                if (result_present) {
                    const char * t = transcribe_batch_full_text(ctx, static_cast<int>(k));
                    if (t && *t) {
                        text = t;
                    }
                }
                if (ust == TRANSCRIBE_OK) {
                    ++n_ok;
                } else if (ust == TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
                    ++n_truncated;
                } else if (ust == TRANSCRIBE_ERR_OUTPUT_REPETITION) {
                    ++n_repeating;
                } else {
                    ++n_fail;
                }
                if (args.batch_jsonl) {
                    const std::string escaped  = json_escape(text);
                    std::string       segments = batch_segments_json(ctx, static_cast<int>(k));
                    segments += batch_speakers_json(ctx, static_cast<int>(k));
                    segments += raw_text_json(transcribe_batch_raw_text(ctx, static_cast<int>(k)), text);
                    std::string err_field;
                    if (ust != TRANSCRIBE_OK) {
                        err_field = ",\"error\":\"";
                        err_field += json_escape(transcribe_status_string(ust));
                        err_field += "\"";
                    }
                    // Per-utterance timings: the batched encoder time is
                    // amortized across the batch, decode is real per-utt,
                    // so summing across utterances gives the true encode vs
                    // decode split for the whole batched run.
                    struct transcribe_timings tm;
                    transcribe_timings_init(&tm);
                    (void) transcribe_batch_get_timings(ctx, static_cast<int>(k), &tm);
                    const std::string file_esc = json_escape(wav.c_str());
                    std::printf(
                        "{\"file\":\"%s\",\"text\":\"%s\"%s,"
                        "\"mel_ms\":%.1f,\"encode_ms\":%.1f,"
                        "\"decode_ms\":%.1f%s}\n",
                        file_esc.c_str(), escaped.c_str(), segments.c_str(), (double) tm.mel_ms, (double) tm.encode_ms,
                        (double) tm.decode_ms, err_field.c_str());
                } else {
                    std::printf("[%zu/%zu] %s", src_index[k] + 1, total, wav.c_str());
                    if (ust == TRANSCRIBE_OK) {
                        std::printf("\n  text: %s\n", text);
                    } else if (is_cut_short(ust)) {
                        std::printf("  (%s)\n  text: %s\n", cut_short_label(ust), text);
                    } else {
                        std::printf("  ERROR: %s\n", transcribe_status_string(ust));
                    }
                }
                output_ok = write_output_file(output, args.output_path, text) && output_ok;
                std::fflush(stdout);
            }
        }
    } else {
        for (size_t i = 0; i < wav_paths.size(); ++i) {
            const std::string & wav = wav_paths[i];

            std::vector<float> pcm;
            std::string        wav_err;
            if (!transcribe_cli::load_wav_mono_16k(wav, pcm, wav_err)) {
                if (args.batch_jsonl) {
                    const std::string file_esc = json_escape(wav.c_str());
                    std::printf(
                        "{\"file\":\"%s\",\"text\":\"\","
                        "\"error\":\"wav: %s\"}\n",
                        file_esc.c_str(), json_escape(wav_err.c_str()).c_str());
                } else {
                    std::fprintf(stderr, "SKIP %s: %s\n", wav.c_str(), wav_err.c_str());
                }
                ++n_fail;
                std::fflush(stdout);
                continue;
            }

            // Run. When --stream-chunk-ms > 0, drive the streaming API
            // for this utterance (begin/feed/finalize) so the WER
            // harness can measure cache-aware streaming output.
            transcribe_status run_st = TRANSCRIBE_OK;
            if (args.stream_chunk_ms > 0) {
                struct transcribe_stream_params sp;
                transcribe_stream_params_init(&sp);
                struct transcribe_parakeet_stream_ext pkt_sp;
                transcribe_parakeet_stream_ext_init(&pkt_sp);
                struct transcribe_parakeet_buffered_stream_ext pkt_buf_sp;
                transcribe_parakeet_buffered_stream_ext_init(&pkt_buf_sp);
                struct transcribe_voxtral_realtime_stream_ext vx_sp;
                transcribe_voxtral_realtime_stream_ext_init(&vx_sp);
                const bool want_cache_aware = (args.stream_att_right >= 0);
                const bool want_buffered =
                    args.stream_buf_left_ms >= 0 || args.stream_buf_chunk_ms >= 0 || args.stream_buf_right_ms >= 0;
                if (want_cache_aware && transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                                          TRANSCRIBE_EXT_KIND_PARAKEET_STREAM)) {
                    pkt_sp.att_context_right = args.stream_att_right;
                    sp.family                = &pkt_sp.ext;
                } else if (want_buffered &&
                           transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                             TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM)) {
                    pkt_buf_sp.left_ms  = args.stream_buf_left_ms;
                    pkt_buf_sp.chunk_ms = args.stream_buf_chunk_ms;
                    pkt_buf_sp.right_ms = args.stream_buf_right_ms;
                    sp.family           = &pkt_buf_sp.ext;
                } else if (args.stream_voxtral_delay != -1 &&
                           transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                             TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM)) {
                    vx_sp.num_delay_tokens = args.stream_voxtral_delay;
                    sp.family              = &vx_sp.ext;
                }
                run_st = transcribe_stream_begin(ctx, &rp, &sp);
                if (run_st == TRANSCRIBE_OK) {
                    const int chunk_samples = std::max(1, args.stream_chunk_ms * 16000 / 1000);
                    size_t    pos           = 0;
                    while (pos < pcm.size()) {
                        const size_t take = std::min<size_t>(static_cast<size_t>(chunk_samples), pcm.size() - pos);
                        struct transcribe_stream_update upd;
                        transcribe_stream_update_init(&upd);
                        run_st = transcribe_stream_feed(ctx, pcm.data() + pos, static_cast<int>(take), &upd);
                        if (run_st != TRANSCRIBE_OK) {
                            break;
                        }
                        pos += take;
                    }
                    if (run_st == TRANSCRIBE_OK) {
                        struct transcribe_stream_update fin_upd;
                        transcribe_stream_update_init(&fin_upd);
                        run_st = transcribe_stream_finalize(ctx, &fin_upd);
                    }
                }
            } else {
                run_st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
            }

            // OUTPUT_TRUNCATED / OUTPUT_REPETITION are result-bearing: the partial
            // transcript is preserved and readable via transcribe_full_text (see
            // transcribe.h). Emit it as the hyp so downstream tooling scores the
            // partial rather than an empty string; the error field below still
            // tags it so the stop stays visible.
            const bool   result_present = run_st == TRANSCRIBE_OK || is_cut_short(run_st);
            const char * text           = "";
            if (result_present) {
                const char * t = transcribe_full_text(ctx);
                if (t && *t) {
                    text = t;
                }
            }
            if (run_st == TRANSCRIBE_OK) {
                ++n_ok;
            } else if (run_st == TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
                ++n_truncated;
            } else if (run_st == TRANSCRIBE_ERR_OUTPUT_REPETITION) {
                ++n_repeating;
            } else {
                ++n_fail;
            }

            // Output. When run_st != OK we still emit a JSON line so
            // batch consumers see one record per input wav, but we tag
            // it with the error string. Without this the wav-load error
            // path (above) reports errors but transcribe_run failures
            // (e.g. unsupported language) produce empty hyp_text with no
            // error tag, so downstream tools count them as silent
            // successes.
            if (args.batch_jsonl) {
                struct transcribe_timings tm;
                transcribe_timings_init(&tm);
                (void) transcribe_get_timings(ctx, &tm);
                const std::string escaped  = json_escape(text);
                std::string       segments = segments_json(ctx);
                segments += speakers_json(ctx);
                segments += raw_text_json(transcribe_raw_text(ctx), text);
                std::string err_field;
                if (run_st != TRANSCRIBE_OK) {
                    err_field = ",\"error\":\"";
                    err_field += json_escape(transcribe_status_string(run_st));
                    err_field += "\"";
                }
                const std::string file_esc = json_escape(wav.c_str());
                std::printf(
                    "{\"file\":\"%s\",\"text\":\"%s\"%s,"
                    "\"mel_ms\":%.1f,\"encode_ms\":%.1f,"
                    "\"decode_ms\":%.1f%s}\n",
                    file_esc.c_str(), escaped.c_str(), segments.c_str(), (double) tm.mel_ms, (double) tm.encode_ms,
                    (double) tm.decode_ms, err_field.c_str());
            } else {
                std::printf("[%zu/%zu] %s", i + 1, wav_paths.size(), wav.c_str());
                if (run_st == TRANSCRIBE_OK) {
                    std::printf("\n  text: %s\n", text);
                } else if (is_cut_short(run_st)) {
                    std::printf("  (%s)\n  text: %s\n", cut_short_label(run_st), text);
                } else {
                    std::printf("  ERROR: %s\n", transcribe_status_string(run_st));
                }
            }
            output_ok = write_output_file(output, args.output_path, text) && output_ok;
            std::fflush(stdout);
        }
    }

    if (!args.batch_jsonl) {
        std::fprintf(stderr, "batch: %d ok, %d truncated, %d stopped repeating, %d failed out of %zu\n", n_ok,
                     n_truncated, n_repeating, n_fail, wav_paths.size());
    }

    transcribe_session_free(ctx);
    transcribe_model_free(model);
    // OUTPUT_TRUNCATED / OUTPUT_REPETITION are result-bearing and do not fail the batch, but
    // hard per-utterance failures must remain visible to automation.
    return n_fail > 0 || !output_ok ? EXIT_FAILURE : EXIT_SUCCESS;
}

int run_asr_file(const cli_args &           args,
                 transcribe_model *         model,
                 const std::vector<float> & pcm,
                 double                     duration_s,
                 std::ofstream *            output) {
    bool output_ok = true;

    struct transcribe_session_params cp;
    transcribe_session_params_init(&cp);
    cp.n_threads                        = args.n_threads;
    cp.n_ctx                            = args.n_ctx;
    cp.kv_type                          = args.kv_type;
    struct transcribe_session * ctx     = nullptr;
    const transcribe_status     init_st = transcribe_session_init(model, &cp, &ctx);
    if (init_st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "context init: %s\n", transcribe_status_string(init_st));
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    // Surface the effective input-length limit so it's obvious how much
    // audio this session accepts (reflects --n-ctx). 0 means "no practical
    // limit", which covers two different families: one that chunks long
    // audio internally (FEATURE_LONG_FORM) and one that is genuinely
    // unbounded and encodes the clip in a single pass. Distinguish them
    // rather than asserting the first. See docs/input-limits.md.
    {
        struct transcribe_session_limits lim;
        transcribe_session_limits_init(&lim);
        if (transcribe_session_get_limits(ctx, &lim) == TRANSCRIBE_OK) {
            if (lim.effective_max_audio_ms > 0) {
                std::printf("  max audio:  %.1f s", (double) lim.effective_max_audio_ms / 1000.0);
                if (lim.effective_n_ctx > 0) {
                    std::printf("  (context %d tok, ~%lld MiB KV max)", lim.effective_n_ctx,
                                (long long) (lim.max_kv_bytes >> 20));
                }
                std::printf("\n");
            } else if (lim.effective_n_ctx > 0) {
                // Capped family whose context is too small to fit any audio
                // plus a prompt (e.g. an aggressively low --n-ctx).
                std::printf(
                    "  max audio:  ~0 s (context %d tok too small for "
                    "audio + prompt)\n",
                    lim.effective_n_ctx);
            } else if (transcribe_model_supports(model, TRANSCRIBE_FEATURE_LONG_FORM)) {
                std::printf("  max audio:  unbounded (long audio chunked internally)\n");
            } else {
                // No context cap and no chunker: the family encodes the
                // whole clip in one pass (e.g. block-local attention,
                // where cost is linear in audio length).
                std::printf("  max audio:  unbounded (whole clip in one pass)\n");
            }
        }
    }

    struct transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    if (args.translate) {
        rp.task = TRANSCRIBE_TASK_TRANSLATE;
    }
    std::vector<const char *> vocabulary_ptrs;
    apply_prompting(args, rp, vocabulary_ptrs);
    if (!args.language.empty()) {
        rp.language = args.language.c_str();
    }
    if (!args.target_language.empty()) {
        rp.target_language = args.target_language.c_str();
    }
    rp.timestamps    = args.timestamps;
    rp.spec_k_drafts = args.spec_k_drafts;

    if (args.itn_set) {
        rp.itn = args.use_itn ? TRANSCRIBE_ITN_MODE_ON : TRANSCRIBE_ITN_MODE_OFF;
    }
    if (args.canary_pnc_set) {
        rp.pnc = args.canary_pnc ? TRANSCRIBE_PNC_MODE_ON : TRANSCRIBE_PNC_MODE_OFF;
    }
    if (args.diarize_set) {
        rp.diarize = args.diarize ? TRANSCRIBE_DIARIZE_MODE_ON : TRANSCRIBE_DIARIZE_MODE_OFF;
    }

    struct transcribe_whisper_run_ext wx;
    transcribe_whisper_run_ext_init(&wx);
    if (args.whisper_set) {
        if (!args.initial_prompt.empty()) {
            wx.initial_prompt = args.initial_prompt.c_str();
        }
        wx.condition_on_prev_tokens = args.condition_on_prev_tokens;
        if (args.temperature_set) {
            wx.temperature = args.temperature;
        }
        wx.prompt_condition = args.prompt_condition;
        if (transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_WHISPER_RUN)) {
            rp.family = &wx.ext;
        }
    }

    if (args.keep_special_tags) {
        rp.keep_special_tags = true;
    }

    // Streaming demo: drive transcribe_stream_begin/feed/finalize
    // with fixed-size PCM chunks. Families with true per-feed
    // partial decoding (moonshine_streaming) flip
    // update.result_changed whenever the transcript advances; the
    // CLI prints the live tentative text on each such feed.
    // Families that only commit at finalize keep result_changed
    // false until the finalize call.
    transcribe_status run_st = TRANSCRIBE_OK;
    if (args.stream_chunk_ms > 0) {
        struct transcribe_capabilities caps;
        transcribe_capabilities_init(&caps);
        const transcribe_status caps_st = transcribe_model_get_capabilities(model, &caps);
        if (caps_st != TRANSCRIBE_OK || !caps.supports_streaming) {
            std::fprintf(stderr,
                         "stream: model does not advertise "
                         "supports_streaming; use a streaming-capable "
                         "model or drop --stream-chunk-ms\n");
            transcribe_session_free(ctx);
            transcribe_model_free(model);
            return EXIT_FAILURE;
        }

        const int chunk_samples = std::max(1, args.stream_chunk_ms * 16000 / 1000);
        std::printf("stream: chunk=%d ms (%d samples)\n", args.stream_chunk_ms, chunk_samples);

        struct transcribe_stream_params sp;
        transcribe_stream_params_init(&sp);
        struct transcribe_parakeet_stream_ext pkt_sp;
        transcribe_parakeet_stream_ext_init(&pkt_sp);
        struct transcribe_parakeet_buffered_stream_ext pkt_buf_sp;
        transcribe_parakeet_buffered_stream_ext_init(&pkt_buf_sp);
        struct transcribe_voxtral_realtime_stream_ext vx_sp;
        transcribe_voxtral_realtime_stream_ext_init(&vx_sp);
        const bool want_cache_aware = (args.stream_att_right >= 0);
        const bool want_buffered =
            args.stream_buf_left_ms >= 0 || args.stream_buf_chunk_ms >= 0 || args.stream_buf_right_ms >= 0;
        if (want_cache_aware &&
            transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM, TRANSCRIBE_EXT_KIND_PARAKEET_STREAM)) {
            pkt_sp.att_context_right = args.stream_att_right;
            sp.family                = &pkt_sp.ext;
            std::printf("stream: att_context_right=%d\n", args.stream_att_right);
        } else if (want_buffered && transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                                      TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM)) {
            pkt_buf_sp.left_ms  = args.stream_buf_left_ms;
            pkt_buf_sp.chunk_ms = args.stream_buf_chunk_ms;
            pkt_buf_sp.right_ms = args.stream_buf_right_ms;
            sp.family           = &pkt_buf_sp.ext;
            std::printf("stream: buffered (L,C,R)_ms=(%d,%d,%d)\n", args.stream_buf_left_ms, args.stream_buf_chunk_ms,
                        args.stream_buf_right_ms);
        } else if (args.stream_voxtral_delay != -1 &&
                   transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                     TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM)) {
            vx_sp.num_delay_tokens = args.stream_voxtral_delay;
            sp.family              = &vx_sp.ext;
            std::printf("stream: voxtral num_delay_tokens=%d\n", args.stream_voxtral_delay);
        }
        run_st = transcribe_stream_begin(ctx, &rp, &sp);
        if (run_st != TRANSCRIBE_OK) {
            std::fprintf(stderr, "stream_begin: %s\n", transcribe_status_string(run_st));
        } else {
            size_t pos    = 0;
            int    feed_n = 0;
            while (pos < pcm.size()) {
                const size_t take = std::min<size_t>(static_cast<size_t>(chunk_samples), pcm.size() - pos);
                struct transcribe_stream_update upd;
                transcribe_stream_update_init(&upd);
                run_st = transcribe_stream_feed(ctx, pcm.data() + pos, static_cast<int>(take), &upd);
                if (run_st != TRANSCRIBE_OK) {
                    std::fprintf(stderr, "stream_feed[%d]: %s\n", feed_n, transcribe_status_string(run_st));
                    break;
                }
                pos += take;
                std::printf("  feed[%2d]: input=%lld ms buffered=%lld ms", feed_n, (long long) upd.input_received_ms,
                            (long long) upd.buffered_ms);
                if (upd.result_changed) {
                    const char * partial = transcribe_full_text(ctx);
                    std::printf("  partial=\"%s\"", (partial && *partial) ? partial : "");
                }
                std::printf("\n");
                ++feed_n;
            }
            if (run_st == TRANSCRIBE_OK) {
                struct transcribe_stream_update fin_upd;
                transcribe_stream_update_init(&fin_upd);
                run_st = transcribe_stream_finalize(ctx, &fin_upd);
                std::printf(
                    "  finalize: status=%s "
                    "revision=%d input=%lld ms committed=%lld ms\n",
                    transcribe_status_string(run_st), fin_upd.revision, (long long) fin_upd.input_received_ms,
                    (long long) fin_upd.audio_committed_ms);
            }
        }
    } else {
        // --repeat N runs transcribe_run() N times for steady-state
        // perf measurements.
        for (int r = 0; r < args.repeat; ++r) {
            run_st = transcribe_run(ctx, pcm.data(), static_cast<int>(pcm.size()), &rp);
            if (run_st != TRANSCRIBE_OK) {
                break;
            }
        }
    }
    std::printf("run: %s\n", transcribe_status_string(run_st));
    // OUTPUT_TRUNCATED, OUTPUT_REPETITION and ABORTED are non-OK but
    // preserve the partial transcript (see docs/input-limits.md), so show
    // the result for them too — just flagged.
    const bool result_present = run_st == TRANSCRIBE_OK || is_cut_short(run_st) || run_st == TRANSCRIBE_ERR_ABORTED;
    if (result_present) {
        const char * text = transcribe_full_text(ctx);
        std::printf("text: %s\n", (text && *text) ? text : "(empty)");
        output_ok = write_output_file(output, args.output_path, text) && output_ok;

        // A decode cut short before end-of-stream; the text above is
        // incomplete.
        if (run_st == TRANSCRIBE_ERR_OUTPUT_REPETITION) {
            std::printf(
                "  note:      decode stopped when the output began repeating "
                "itself (repeats dropped); transcript is incomplete\n");
        } else if (transcribe_was_truncated(ctx)) {
            std::printf(
                "  note:      output truncated (hit the model's "
                "context/generation cap before end-of-stream); "
                "transcript is incomplete\n");
        }

        const char * dl = transcribe_detected_language(ctx);
        if (dl && *dl) {
            std::printf("detected-language: %s\n", dl);
        }

        const transcribe_timestamp_kind ret_kind = transcribe_returned_timestamp_kind(ctx);
        const int                       n_seg    = transcribe_n_segments(ctx);
        const bool                      has_spk  = transcribe_n_speaker_segments(ctx) > 0;
        if (n_seg > 0 && (ret_kind != TRANSCRIBE_TIMESTAMPS_NONE || has_spk)) {
            std::printf("segments: %d\n", n_seg);
            for (int i = 0; i < n_seg; ++i) {
                struct transcribe_segment seg;
                transcribe_segment_init(&seg);
                (void) transcribe_get_segment(ctx, i, &seg);
                char spk[16] = "";
                if (seg.speaker_id > 0) {
                    std::snprintf(spk, sizeof(spk), "S%d: ", static_cast<int>(seg.speaker_id));
                }
                if (ret_kind != TRANSCRIBE_TIMESTAMPS_NONE) {
                    std::printf("  [%7.2f -> %7.2f] %s%s\n", seg.t0_ms / 1000.0, seg.t1_ms / 1000.0, spk,
                                (seg.text != nullptr) ? seg.text : "");
                } else {
                    // Speaker-attributed turns without timing (granite SAA).
                    std::printf("  %s%s\n", spk, (seg.text != nullptr) ? seg.text : "");
                }
            }
        }
        if (ret_kind == TRANSCRIBE_TIMESTAMPS_WORD || ret_kind == TRANSCRIBE_TIMESTAMPS_TOKEN) {
            const int n_wrd = transcribe_n_words(ctx);
            std::printf("words: %d\n", n_wrd);
            for (int i = 0; i < n_wrd; ++i) {
                struct transcribe_word wrd;
                transcribe_word_init(&wrd);
                (void) transcribe_get_word(ctx, i, &wrd);
                std::printf("  [%7.2f -> %7.2f] %s\n", wrd.t0_ms / 1000.0, wrd.t1_ms / 1000.0,
                            (wrd.text != nullptr) ? wrd.text : "");
            }
        }
        if (ret_kind == TRANSCRIBE_TIMESTAMPS_TOKEN) {
            const int n_tok = transcribe_n_tokens(ctx);
            std::printf("tokens: %d\n", n_tok);
            for (int i = 0; i < n_tok; ++i) {
                struct transcribe_token tok;
                transcribe_token_init(&tok);
                (void) transcribe_get_token(ctx, i, &tok);
                std::printf("  [%7.2f -> %7.2f] p=%.3f %s\n", tok.t0_ms / 1000.0, tok.t1_ms / 1000.0, tok.p,
                            (tok.text != nullptr) ? tok.text : "");
            }
        }
    }

    transcribe_print_timings(ctx);

    {
        struct transcribe_timings tm;
        transcribe_timings_init(&tm);
        (void) transcribe_get_timings(ctx, &tm);
        const double total_ms = tm.mel_ms + tm.encode_ms + tm.decode_ms;
        if (total_ms > 0.0 && duration_s > 0.0) {
            std::printf("  realtime:   %.0fx (%.1f ms for %.1f s)\n", (duration_s * 1000.0) / total_ms, total_ms,
                        duration_s);
        }
    }

    transcribe_session_free(ctx);
    transcribe_model_free(model);

    if (run_st != TRANSCRIBE_OK || !output_ok) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

}  // namespace transcribe_cli
