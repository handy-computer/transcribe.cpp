// hello_stream.c — the smallest complete streaming program.
//
// Loads a model, feeds a WAV into the streaming API in fixed-size
// chunks, and prints the cumulative transcript after every feed
// (separating the committed prefix from the still-tentative suffix).
// This is the binding-author reference for what a real-time consumer
// looks like through the public C ABI; pair it with examples/hello/main.c
// for the offline equivalent.
//
// Lifecycle:
//
//     transcribe_open
//       └─ transcribe_stream_begin
//             └─ loop: transcribe_stream_feed → inspect update / accessors
//             └─ transcribe_stream_finalize
//       └─ transcribe_session_free   (frees the owned model too)
//
// The streaming model is poll-based. After every feed, the library
// has already updated the cumulative result on the session; the
// caller decides what to do with it. `update.result_changed` is the
// cheap "did anything move?" probe; the committed-count accessors
// identify the stable prefix.
//
// Run:
//     transcribe-hello-stream model.gguf audio.wav [chunk_ms]
//
// chunk_ms defaults to 500. Smaller values produce more frequent
// updates with higher per-update cost.
//
// Text-pointer lifetime caveat: every transcribe_stream_feed call may
// invalidate every text pointer previously returned by transcribe_full_text
// or transcribe_get_*(). This example uses the pointers and writes them
// to stdout in the same iteration, which is safe. A consumer that holds
// text past the next feed must copy bytes — see the "Result text-pointer
// lifetime" block at the top of include/transcribe.h.

#include "transcribe.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- audio loading (NOT part of the transcribe API) -----------------------
static float * load_wav_mono(const char * path, int * out_n_samples,
                             unsigned int * out_rate) {
    unsigned int channels = 0;
    unsigned int rate     = 0;
    drwav_uint64 frames   = 0;
    float * interleaved = drwav_open_file_and_read_pcm_frames_f32(
        path, &channels, &rate, &frames, NULL);
    if (interleaved == NULL) {
        return NULL;
    }
    *out_rate = rate;
    if (channels <= 1) {
        *out_n_samples = (int) frames;
        return interleaved;
    }
    float * mono = (float *) malloc((size_t) frames * sizeof(float));
    for (drwav_uint64 i = 0; i < frames; ++i) {
        float acc = 0.0f;
        for (unsigned int c = 0; c < channels; ++c) {
            acc += interleaved[i * channels + c];
        }
        mono[i] = acc / (float) channels;
    }
    drwav_free(interleaved, NULL);
    *out_n_samples = (int) frames;
    return mono;
}
// --------------------------------------------------------------------------

// Print the current UI-facing stream text split.
//
// IMPORTANT lifetime rule: the `text` pointer returned by
// transcribe_stream_get_text is valid only until the next
// transcribe_stream_feed or transcribe_stream_finalize call. Use it
// immediately, like the printf() below, and let it go out of scope. If
// you need to hold transcript text past the next feed — for a UI render
// loop on another thread, a queued message, a binding object — copy the
// bytes out before continuing.
//
static void print_partial(struct transcribe_session * session) {
    struct transcribe_stream_text text;
    transcribe_stream_text_init(&text);
    if (transcribe_stream_get_text(session, &text) != TRANSCRIBE_OK ||
        text.full_text == NULL || *text.full_text == '\0')
    {
        return;
    }
    printf("    committed=\"%s\" tentative=\"%s\" raw=\"%s\"\n",
           text.committed_text ? text.committed_text : "",
           text.tentative_text ? text.tentative_text : "",
           text.full_text ? text.full_text : "");
    fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr,
                "usage: %s model.gguf audio.wav [chunk_ms=500]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * wav_path   = argv[2];
    const int    chunk_ms   = (argc == 4) ? atoi(argv[3]) : 500;
    if (chunk_ms <= 0) {
        fprintf(stderr, "error: chunk_ms must be > 0\n");
        return 2;
    }

    // 1. Load audio (16 kHz mono float32 PCM).
    int          n_samples = 0;
    unsigned int rate      = 0;
    float *      pcm       = load_wav_mono(wav_path, &n_samples, &rate);
    if (pcm == NULL) {
        fprintf(stderr, "error: could not read WAV: %s\n", wav_path);
        return 1;
    }
    if (rate != 16000) {
        fprintf(stderr,
                "error: audio must be 16 kHz (got %u Hz); resample first, "
                "e.g. ffmpeg -i in.wav -ar 16000 -ac 1 out.wav\n", rate);
        free(pcm);
        return 1;
    }

    // 2. Open a session. NULL params == library defaults. transcribe_open
    //    returns a session that owns its model; transcribe_session_free
    //    later frees both.
    struct transcribe_session * session = NULL;
    transcribe_status st = transcribe_open(model_path, NULL, NULL, &session);
    if (st != TRANSCRIBE_OK) {
        fprintf(stderr, "error: transcribe_open: %s\n",
                transcribe_status_string(st));
        free(pcm);
        return 1;
    }

    // 3. Capability gate. Streaming is opt-in per model; pre-check so the
    //    failure mode is a clear message, not a NOT_IMPLEMENTED from
    //    transcribe_stream_begin.
    struct transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    if (transcribe_model_get_capabilities(
            transcribe_get_model(session), &caps) != TRANSCRIBE_OK ||
        !caps.supports_streaming)
    {
        fprintf(stderr,
                "error: model does not advertise supports_streaming; "
                "use a streaming-capable model.\n");
        transcribe_session_free(session);
        free(pcm);
        return 1;
    }

    // 4. Start the stream. NULL run/stream params == defaults
    //    (transcribe task, no timestamps, family default for the
    //    streaming knob). A family-specific knob would go on
    //    stream_params.family via transcribe_model_accepts_ext_kind
    //    + the family's init function; this example sticks to defaults.
    st = transcribe_stream_begin(session, NULL, NULL);
    if (st != TRANSCRIBE_OK) {
        fprintf(stderr, "error: stream_begin: %s\n",
                transcribe_status_string(st));
        transcribe_session_free(session);
        free(pcm);
        return 1;
    }

    printf("stream: chunk=%d ms\n", chunk_ms);

    // 5. Feed PCM in fixed-size chunks. After each feed, inspect the
    //    update struct. result_changed says "the cumulative transcript
    //    moved"; transcribe_stream_get_text then reads the UI-facing
    //    committed/tentative split.
    const int chunk_samples = chunk_ms * 16000 / 1000;
    int pos = 0;
    int feed_idx = 0;
    while (pos < n_samples) {
        int take = chunk_samples;
        if (pos + take > n_samples) take = n_samples - pos;

        struct transcribe_stream_update upd;
        transcribe_stream_update_init(&upd);
        st = transcribe_stream_feed(session, pcm + pos, take, &upd);
        if (st != TRANSCRIBE_OK) {
            fprintf(stderr, "error: stream_feed[%d]: %s\n",
                    feed_idx, transcribe_status_string(st));
            break;
        }

        printf("feed[%2d]: input=%lld ms buffered=%lld ms%s\n",
               feed_idx,
               (long long) upd.input_received_ms,
               (long long) upd.buffered_ms,
               upd.result_changed ? "  (result changed)" : "");
        if (upd.result_changed) {
            print_partial(session);
        }

        pos += take;
        ++feed_idx;
    }

    // 6. Flush. finalize forces a last decode if needed, satisfies
    //    right-context / lookahead, closes tentative_text, and transitions
    //    the stream to FINISHED. After finalize, full_text remains the raw
    //    model transcript while committed_text remains append-only.
    if (st == TRANSCRIBE_OK) {
        struct transcribe_stream_update fin;
        transcribe_stream_update_init(&fin);
        st = transcribe_stream_finalize(session, &fin);
        if (st != TRANSCRIBE_OK) {
            fprintf(stderr, "error: stream_finalize: %s\n",
                    transcribe_status_string(st));
        } else {
            printf("finalize: revision=%d input=%lld ms committed=%lld ms\n",
                   fin.revision,
                   (long long) fin.input_received_ms,
                   (long long) fin.audio_committed_ms);
            print_partial(session);
        }
    }

    // 7. Tear down. transcribe_session_free frees the session AND, since
    //    transcribe_open set owns_model=true, the model it loaded.
    transcribe_session_free(session);
    free(pcm);
    return (st == TRANSCRIBE_OK) ? 0 : 1;
}
