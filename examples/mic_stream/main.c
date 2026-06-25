// mic_stream.c — live microphone streaming demo.
//
// Reads raw S16_LE 16 kHz mono PCM from stdin and drives the streaming
// API in real time, rewriting one terminal line with the committed
// (white) and tentative (gray) transcript after every chunk. Pair it
// with arecord so the demo needs no audio-capture dependency:
//
//     arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -t raw -B 4000000 |
//         transcribe-mic-stream model.gguf \
//             --chunk-ms 1120 --att-right 13 --threads 4 --language en-US
//
// (-B 4000000 gives arecord a 4 s ring buffer so a slow chunk doesn't
// overrun the capture device; arecord prints "overrun!" on its stderr
// if audio was lost.)
//
// It also works from a file, just faster than real time:
//
//     ffmpeg -v 0 -i talk.wav -f s16le -ar 16000 -ac 1 - |
//         transcribe-mic-stream model.gguf
//
// The line prefix shows the running real-time factor (feed wall time /
// audio time; below 1.0 means the box keeps up). Ctrl-C stops arecord,
// stdin hits EOF, and the stream finalizes normally before exit.
//
// See examples/hello_stream/main.c for the fully-commented walk-through
// of the streaming lifecycle; this file only annotates what differs
// (live input, session threads, parakeet att-right extension).

#include "transcribe.h"
#include "transcribe/parakeet.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int sig) { (void) sig; g_stop = 1; }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

// Fill buf with n int16 samples from stdin; short count means EOF.
static size_t read_full(int16_t * buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        size_t r = fread(buf + got, sizeof(int16_t), n - got, stdin);
        if (r == 0) break;
        got += r;
    }
    return got;
}

// Rewrite the status line: "[rtf] committed tentative", tail-truncated
// so \r line-rewriting survives narrow terminals.
static void render_line(struct transcribe_session * session, double rtf) {
    struct transcribe_stream_text text;
    transcribe_stream_text_init(&text);
    if (transcribe_stream_get_text(session, &text) != TRANSCRIBE_OK) {
        return;
    }
    const char * com = text.committed_text ? text.committed_text : "";
    const char * ten = text.tentative_text ? text.tentative_text : "";

    enum { MAX_TAIL = 100 };
    size_t com_len = strlen(com);
    size_t ten_len = strlen(ten);
    size_t total   = com_len + ten_len;
    size_t skip    = total > MAX_TAIL ? total - MAX_TAIL : 0;

    printf("\r\033[2K[%4.2fx] ", rtf);
    if (skip < com_len) {
        printf("%s", com + skip);
        printf("\033[90m%s\033[0m", ten);
    } else {
        printf("\033[90m%s\033[0m", ten + (skip - com_len));
    }
    fflush(stdout);
}

int main(int argc, char ** argv) {
    const char * model_path = NULL;
    const char * language   = "en-US";
    int chunk_ms  = 1120;
    int att_right = -1;   // -1 = model default lookahead
    int n_threads = 0;    //  0 = library default

    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (strcmp(a, "--chunk-ms") == 0 && i + 1 < argc) {
            chunk_ms = atoi(argv[++i]);
        } else if (strcmp(a, "--att-right") == 0 && i + 1 < argc) {
            att_right = atoi(argv[++i]);
        } else if (strcmp(a, "--threads") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(a, "--language") == 0 && i + 1 < argc) {
            language = argv[++i];
        } else if (a[0] != '-' && model_path == NULL) {
            model_path = a;
        } else {
            fprintf(stderr,
                    "usage: %s model.gguf [--chunk-ms N] [--att-right R] "
                    "[--threads T] [--language L]\n"
                    "reads raw S16_LE 16 kHz mono PCM from stdin\n",
                    argv[0]);
            return 2;
        }
    }
    if (model_path == NULL || chunk_ms <= 0) {
        fprintf(stderr, "error: model path required, chunk_ms must be > 0\n");
        return 2;
    }

    signal(SIGINT, on_sigint);

    struct transcribe_session_params sess;
    transcribe_session_params_init(&sess);
    sess.n_threads = n_threads;

    struct transcribe_session * session = NULL;
    transcribe_status st = transcribe_open(model_path, NULL, &sess, &session);
    if (st != TRANSCRIBE_OK) {
        fprintf(stderr, "error: transcribe_open: %s\n",
                transcribe_status_string(st));
        return 1;
    }

    struct transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    if (transcribe_model_get_capabilities(
            transcribe_get_model(session), &caps) != TRANSCRIBE_OK ||
        !caps.supports_streaming)
    {
        fprintf(stderr, "error: model does not support streaming\n");
        transcribe_session_free(session);
        return 1;
    }

    struct transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    rp.language = language;

    // Cache-aware lookahead knob; only attached when requested AND the
    // model takes it, so the binary still runs other streaming families.
    struct transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);
    struct transcribe_parakeet_stream_ext pkt;
    transcribe_parakeet_stream_ext_init(&pkt);
    if (att_right >= 0 && transcribe_model_accepts_ext_kind(
            transcribe_get_model(session),
            TRANSCRIBE_EXT_SLOT_STREAM,
            TRANSCRIBE_EXT_KIND_PARAKEET_STREAM))
    {
        pkt.att_context_right = att_right;
        sp.family = &pkt.ext;
    }

    st = transcribe_stream_begin(session, &rp, &sp);
    if (st != TRANSCRIBE_OK) {
        fprintf(stderr, "error: stream_begin: %s\n",
                transcribe_status_string(st));
        transcribe_session_free(session);
        return 1;
    }

    const size_t chunk_samples = (size_t) chunk_ms * 16000 / 1000;
    int16_t * s16 = malloc(chunk_samples * sizeof(int16_t));
    float *   f32 = malloc(chunk_samples * sizeof(float));
    if (s16 == NULL || f32 == NULL) {
        fprintf(stderr, "error: out of memory\n");
        transcribe_session_free(session);
        return 1;
    }

    fprintf(stderr, "listening: chunk=%d ms, threads=%d, att_right=%d "
            "(Ctrl-C to stop)\n", chunk_ms, n_threads, att_right);

    double audio_ms_total = 0.0;
    double feed_ms_total  = 0.0;
    long   n_feeds        = 0;

    while (!g_stop) {
        size_t got = read_full(s16, chunk_samples);
        if (got == 0) break;
        for (size_t i = 0; i < got; ++i) {
            f32[i] = (float) s16[i] / 32768.0f;
        }

        struct transcribe_stream_update upd;
        transcribe_stream_update_init(&upd);
        double t0 = now_ms();
        st = transcribe_stream_feed(session, f32, (int) got, &upd);
        double t1 = now_ms();
        if (st != TRANSCRIBE_OK) {
            fprintf(stderr, "\nerror: stream_feed: %s\n",
                    transcribe_status_string(st));
            break;
        }

        audio_ms_total += (double) got / 16.0;
        feed_ms_total  += t1 - t0;
        ++n_feeds;
        render_line(session, feed_ms_total / audio_ms_total);

        if (got < chunk_samples) break;  // EOF mid-chunk
    }

    if (st == TRANSCRIBE_OK) {
        struct transcribe_stream_update fin;
        transcribe_stream_update_init(&fin);
        st = transcribe_stream_finalize(session, &fin);
        if (st != TRANSCRIBE_OK) {
            fprintf(stderr, "\nerror: stream_finalize: %s\n",
                    transcribe_status_string(st));
        }
    }

    printf("\r\033[2K");
    const char * final_text = transcribe_full_text(session);
    if (final_text && *final_text) {
        printf("%s\n", final_text);
    }
    if (n_feeds > 0) {
        fprintf(stderr,
                "fed %.1f s audio in %ld chunks; mean feed %.0f ms/chunk, "
                "%.2fx real time\n",
                audio_ms_total / 1e3, n_feeds, feed_ms_total / n_feeds,
                feed_ms_total / audio_ms_total);
    }

    free(s16);
    free(f32);
    transcribe_session_free(session);
    return (st == TRANSCRIBE_OK) ? 0 : 1;
}
