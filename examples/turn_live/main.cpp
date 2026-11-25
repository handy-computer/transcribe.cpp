// transcribe-turn-live - a live turn-taking loop over a WAV, fed in 20 ms
// chunks as a microphone would deliver it. Prints the events a voice service
// emits: `speech_start` (stop the TTS) and `turn` (the turn is over: start the
// LLM), with the turn's span.
//
//   transcribe-turn-live --vad silero-vad-v6.2-F32.gguf --turn smart-turn-v3.2-F32.gguf in.wav
//   transcribe-turn-live --turn sava-F32.gguf in.wav
//
// Setup A (a VAD model and a clip turn model), Pipecat's rule: a turn starts
// on the first speech frame (with 500 ms of audio before it); after 200 ms of
// silence the turn model scores the turn's audio, and `ended` closes the turn;
// 3 s of silence closes it anyway.
//
// Setup B (a turn model that also scores speech, one stream), Inworld's
// reference rule (sava.segment): a turn opens on the first speech frame,
// pre_roll_ms earlier; while it is open, each silent frame checks the turn
// score of the step that judged it, and max_silence_ms closes it anyway.
// The speech track lags the turn track by speech_lookahead_frames, so frame
// t's speech is paired with the turn score of frame t + lag.
//
// The library returns scores; these rules are the caller's (docs/turn.md).

#include "transcribe.h"
#include "transcribe/turn.h"
#include "transcribe/vad.h"
#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

namespace {

constexpr int    k_sr               = 16000;
constexpr int    k_chunk            = 320;  // 20 ms
constexpr double k_s                = 1.0 / k_sr;
// VAD speech threshold for setup A (transcribe_vad_params' default).
constexpr float  k_speech_threshold = 0.5f;

transcribe_model * load(const char * path) {
    transcribe_model * m = nullptr;
    if (transcribe_model_load_file(path, nullptr, &m) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "cannot load %s\n", path);
    }
    return m;
}

// ---- setup A: VAD stream + clip turn model --------------------------------

int run_vad_plus_clip(transcribe_model * vad_model, transcribe_model * turn_model, const std::vector<float> & pcm) {
    transcribe_vad_info vi;
    transcribe_vad_info_init(&vi);
    transcribe_vad_get_info(vad_model, &vi);
    transcribe_vad_session *  vad  = nullptr;
    transcribe_turn_session * turn = nullptr;
    if (transcribe_vad_session_init(vad_model, nullptr, &vad) != TRANSCRIBE_OK ||
        transcribe_turn_session_init(turn_model, nullptr, &turn) != TRANSCRIBE_OK) {
        return EXIT_FAILURE;
    }
    const int64_t pre_speech = k_sr / 2;  // 500 ms
    const int64_t stop       = k_sr / 5;  // 200 ms of silence: ask the turn model
    const int64_t give_up    = 3 * k_sr;  // 3 s of silence: end the turn anyway
    bool          open       = false;
    bool          asked      = false;     // the turn model was asked during this pause
    int64_t       start = 0, last_speech_end = 0;

    auto close = [&](int64_t end, const char * reason, float p) {
        std::printf("%8.3f turn       [%.3f, %.3f) %s p=%.3f\n", end * k_s, start * k_s, end * k_s, reason, p);
        open = false;
    };

    for (size_t off = 0; off < pcm.size(); off += k_chunk) {
        const int n = static_cast<int>(std::min<size_t>(k_chunk, pcm.size() - off));
        transcribe_vad_stream_feed(vad, pcm.data() + off, n);
        transcribe_vad_result r;
        transcribe_vad_result_init(&r);
        transcribe_vad_get_result(vad, &r);
        const float * p = transcribe_vad_probs(vad);
        for (int i = 0; i < r.n_probs; ++i) {
            const int64_t frame_end = (r.first_frame + i + 1) * vi.frame_samples;
            if (p[i] >= k_speech_threshold) {
                if (!open) {
                    open  = true;
                    start = std::max<int64_t>(0, (r.first_frame + i) * vi.frame_samples - pre_speech);
                    std::printf("%8.3f speech_start\n", frame_end * k_s);
                }
                last_speech_end = frame_end;
                asked           = false;
                continue;
            }
            if (!open) {
                continue;
            }
            const int64_t silence = frame_end - last_speech_end;
            if (silence >= give_up) {
                close(frame_end, "silence", 0.0f);
            } else if (silence >= stop && !asked) {
                asked = true;
                transcribe_turn_run(turn, pcm.data() + start, static_cast<int>(frame_end - start), nullptr);
                transcribe_turn_result tr;
                transcribe_turn_result_init(&tr);
                transcribe_turn_get_result(turn, &tr);
                if (tr.ended) {
                    close(frame_end, "model", tr.p);
                } else {
                    std::printf("%8.3f not yet     p=%.3f\n", frame_end * k_s, tr.p);
                }
            }
        }
    }
    if (open) {
        close(static_cast<int64_t>(pcm.size()), "audio", 0.0f);
    }
    transcribe_turn_session_free(turn);
    transcribe_vad_session_free(vad);
    return EXIT_SUCCESS;
}

// ---- setup B: one turn stream with a speech track -------------------------

int run_turn_stream(transcribe_model * model, const std::vector<float> & pcm) {
    transcribe_turn_info info;
    transcribe_turn_info_init(&info);
    transcribe_turn_get_info(model, &info);
    if (info.frame_samples == 0 || !info.has_speech) {
        std::fprintf(stderr, "setup B needs a turn model that streams and scores speech; pass --vad too\n");
        return EXIT_FAILURE;
    }
    transcribe_turn_session * s = nullptr;
    if (transcribe_turn_session_init(model, nullptr, &s) != TRANSCRIBE_OK) {
        return EXIT_FAILURE;
    }
    // Inworld's reference policy (sava.Policy defaults).
    const double  speech_threshold = 0.5, confident = 0.9, turn_threshold = info.threshold;
    const int64_t frame_ms = std::lround(1000.0 * info.frame_samples / k_sr);
    const int64_t pre_roll = std::lround(800.0 / frame_ms), max_silence_ms = 1600;
    const int64_t min_speech_ms = 400, min_confident_ms = 200;
    const int64_t lag = info.speech_lookahead_frames;

    std::deque<float> turn_hist;  // turn scores by frame, from turn_base
    int64_t           turn_base = 0;
    bool              open      = false;
    int64_t           start = 0, speech_ms = 0, confident_ms = 0, pause = 0, last_judged = 0;

    auto close = [&](int64_t end_frame, const char * reason, float p) {
        const bool    kept = speech_ms >= min_speech_ms || confident_ms >= min_confident_ms;
        const int64_t end  = std::min<int64_t>(end_frame * info.frame_samples, static_cast<int64_t>(pcm.size()));
        std::printf("%8.3f turn       [%.3f, %.3f) %s p=%.3f%s\n", end * k_s, start * info.frame_samples * k_s,
                    end * k_s, reason, p, kept ? "" : " (dropped: too little speech)");
        open = false;
    };

    auto consume = [&] {
        transcribe_turn_result r;
        transcribe_turn_result_init(&r);
        transcribe_turn_get_result(s, &r);
        const float * tp = transcribe_turn_probs(s);
        for (int i = 0; i < r.n_probs; ++i) {
            turn_hist.push_back(tp[i]);
        }
        const float * sp = transcribe_turn_speech_probs(s);
        for (int i = 0; i < r.n_speech_probs; ++i) {
            const int64_t t         = r.speech_first_frame + i;  // the frame judged
            const int64_t now       = t + lag;                   // the step that judged it
            const bool    have_turn = now - turn_base < static_cast<int64_t>(turn_hist.size());
            const float   turn_p    = have_turn ? turn_hist[static_cast<size_t>(now - turn_base)] : 0.0f;
            last_judged             = t;
            const bool is_speech    = sp[i] >= speech_threshold;
            if (!open) {
                if (!is_speech) {
                    continue;
                }
                open      = true;
                start     = std::max<int64_t>(0, t - pre_roll);
                speech_ms = confident_ms = pause = 0;
                // Known once frame t has been judged, lag frames after its audio.
                std::printf("%8.3f speech_start\n", (now + 1) * info.frame_samples * k_s);
            }
            if (is_speech) {
                pause = 0;
                speech_ms += frame_ms;
                confident_ms += sp[i] >= confident ? frame_ms : 0;
                continue;
            }
            ++pause;
            if (pause * frame_ms >= max_silence_ms) {
                close(now + 1, "silence", turn_p);
            } else if (have_turn && turn_p >= turn_threshold) {
                close(now + 1, "model", turn_p);
            }
        }
        // Keep the turn scores still to be paired (frames >= the next judged one + lag).
        while (!turn_hist.empty() && turn_base < last_judged + lag) {
            turn_hist.pop_front();
            ++turn_base;
        }
    };

    for (size_t off = 0; off < pcm.size(); off += k_chunk) {
        const int n = static_cast<int>(std::min<size_t>(k_chunk, pcm.size() - off));
        transcribe_turn_stream_feed(s, pcm.data() + off, n);
        consume();
    }
    transcribe_turn_stream_flush(s);
    consume();
    if (open) {
        close(last_judged + 1, "audio", 0.0f);
    }
    transcribe_turn_session_free(s);
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * vad_path  = nullptr;
    const char * turn_path = nullptr;
    const char * wav       = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--vad") == 0 && i + 1 < argc) {
            vad_path = argv[++i];
        } else if (std::strcmp(argv[i], "--turn") == 0 && i + 1 < argc) {
            turn_path = argv[++i];
        } else {
            wav = argv[i];
        }
    }
    if (turn_path == nullptr || wav == nullptr) {
        std::fprintf(stderr, "usage: %s [--vad VAD.gguf] --turn TURN.gguf in.wav\n", argv[0]);
        return 2;
    }
    std::vector<float> pcm;
    std::string        err;
    if (!transcribe_cli::load_wav_mono_16k(wav, pcm, err)) {
        std::fprintf(stderr, "%s: %s\n", wav, err.c_str());
        return EXIT_FAILURE;
    }
    transcribe_model * turn_model = load(turn_path);
    if (turn_model == nullptr) {
        return EXIT_FAILURE;
    }
    int rc;
    if (vad_path != nullptr) {
        transcribe_model * vad_model = load(vad_path);
        rc = vad_model != nullptr ? run_vad_plus_clip(vad_model, turn_model, pcm) : EXIT_FAILURE;
        transcribe_model_free(vad_model);
    } else {
        rc = run_turn_stream(turn_model, pcm);
    }
    transcribe_model_free(turn_model);
    return rc;
}
