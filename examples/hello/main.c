// hello.c — the smallest complete transcribe.cpp program.
//
// Loads a model, transcribes one 16 kHz mono WAV, prints the text. This
// is the canonical "does the library work?" smoke and the reference for
// binding authors: every transcribe_* call below is part of the stable C
// ABI in <transcribe.h>, and the whole happy path is five calls:
//
//     transcribe_open  -> transcribe_run -> transcribe_full_text
//                      -> (read result) -> transcribe_close
//
// Pass NULL for any params pointer to get library defaults — that is the
// version-proof way to ask for defaults (a NULL carries no struct_size,
// so it always matches the running library; a zero-initialized struct is
// NOT the same and is rejected).
//
// Run:
//     transcribe-hello model.gguf audio.wav
//
// The WAV decoding here uses dr_wav and is deliberately fenced off below;
// it is NOT part of the library. A real application (or a binding) feeds
// float32 PCM from wherever it likes — the library only ever sees the
// samples array passed to transcribe_run().

#include "transcribe.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <stdio.h>
#include <stdlib.h>

// --- audio loading (NOT part of the transcribe API) -----------------------
// Reads a WAV into mono float32 samples at its native rate. Returns a
// malloc'd buffer the caller frees, or NULL on failure. Multi-channel
// input is downmixed by averaging.
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
        return interleaved;  // already mono
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

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.gguf audio.wav\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * wav_path   = argv[2];

    // 1. Load audio. The v1 runtime accepts only 16 kHz mono float32 PCM
    //    and does not resample; the caller is responsible for that.
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

    // 2. Open a session: load the model and bind a session to it in one
    //    call. NULL load/session params == library defaults. The returned
    //    session OWNS the model; transcribe_close frees both.
    struct transcribe_session * session = NULL;
    transcribe_status st = transcribe_open(model_path, NULL, NULL, &session);
    if (st != TRANSCRIBE_OK) {
        fprintf(stderr, "error: transcribe_open: %s\n",
                transcribe_status_string(st));
        free(pcm);
        return 1;
    }

    // 3. Transcribe. NULL run params == defaults (transcribe task, no
    //    timestamps). The result is stored on the session until the next
    //    run or close.
    st = transcribe_run(session, pcm, n_samples, NULL);
    if (st != TRANSCRIBE_OK) {
        fprintf(stderr, "error: transcribe_run: %s\n",
                transcribe_status_string(st));
        transcribe_close(session);
        free(pcm);
        return 1;
    }

    // 4. Read the transcript. The returned pointer is owned by the session
    //    and valid until the next run or transcribe_close.
    printf("%s\n", transcribe_full_text(session));

    // 5. Tear down. transcribe_close frees the session and the owned model.
    transcribe_close(session);
    free(pcm);
    return 0;
}
