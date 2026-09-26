/*
 * include/transcribe/nemotron3_diar.h - Nemotron-3-Diarization public
 * extensions.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. Holds the
 * streaming-operating-point extensions for both entry points, their kind
 * constants, and their init functions.
 *
 * Nemotron-3-Diarization is a diarization-only model (up to 8 speakers,
 * 10 ms output): a run produces no text; the product is the who-spoke-when
 * rows read back via transcribe_n_speaker_segments /
 * transcribe_get_speaker_segment (TRANSCRIBE_FEATURE_DIARIZATION). The
 * compute core is streaming (arrival-order speaker cache + FIFO) and is
 * reachable two ways:
 *
 *   transcribe_run            whole recording in one call. Operating point
 *                             via transcribe_nemotron3_diar_run_ext on the
 *                             RUN slot.
 *   transcribe_stream_*       push-audio live diarization. Operating point
 *                             via transcribe_nemotron3_diar_stream_ext on
 *                             the STREAM slot. After every feed the speaker
 *                             segments cover all audio processed so far: a
 *                             finished turn is final, a turn still in
 *                             progress is reported open-ended up to the
 *                             latest processed frame and may extend on later
 *                             feeds. Finalize flushes the tail; the final
 *                             segments equal a transcribe_run over the same
 *                             audio at the same preset.
 *
 * Probe via transcribe_model_accepts_ext_kind(model, slot, kind) before
 * pointing run_params::family / stream_params::family at the struct.
 *
 * FourCC kinds are reserved in docs/extension-kinds.md.
 */

#ifndef TRANSCRIBE_NEMOTRON3_DIAR_H
#define TRANSCRIBE_NEMOTRON3_DIAR_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'N3DR' little-endian = 0x5244334E (RUN slot) */
#define TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_RUN    0x5244334Eu
/* 'N3DS' little-endian = 0x5344334E (STREAM slot) */
#define TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_STREAM 0x5344334Eu

/*
 * Streaming operating point (latency / accuracy trade-off).
 *
 * The model processes audio in fixed chunks with a fixed lookahead and
 * carries speaker identity across chunks in a bounded cache. Each named
 * preset is a jointly-tuned bundle published on the upstream model card
 * (chunk length, lookahead, FIFO and speaker-cache geometry); the menu is
 * discrete, not a continuous latency dial. Latency = chunk + lookahead.
 *
 *   DEFAULT             The GGUF-shipped configuration (= VERY_HIGH_LATENCY).
 *   VERY_HIGH_LATENCY   30.4 s (chunk 340 + lookahead 40 frames @ 80 ms).
 *                       Highest accuracy; the offline operating point.
 *   LOW_LATENCY         1.04 s (chunk 9 + lookahead 4).
 *   VERY_LOW_LATENCY    0.64 s (chunk 6 + lookahead 2).
 *   ULTRA_LOW_LATENCY   0.32 s (chunk 3 + lookahead 1).
 *
 * Smaller chunks cost more compute per audio second (the whole speaker
 * cache is re-encoded every chunk). Values outside the enum range are
 * rejected with TRANSCRIBE_ERR_INVALID_ARG before the previous result is
 * cleared.
 */
typedef enum {
    TRANSCRIBE_NEMOTRON3_DIAR_PRESET_DEFAULT           = 0,
    TRANSCRIBE_NEMOTRON3_DIAR_PRESET_VERY_HIGH_LATENCY = 1,
    TRANSCRIBE_NEMOTRON3_DIAR_PRESET_LOW_LATENCY       = 2,
    TRANSCRIBE_NEMOTRON3_DIAR_PRESET_VERY_LOW_LATENCY  = 3,
    TRANSCRIBE_NEMOTRON3_DIAR_PRESET_ULTRA_LOW_LATENCY = 4,
} transcribe_nemotron3_diar_preset;

/* RUN slot (transcribe_run_params::family). */
struct transcribe_nemotron3_diar_run_ext {
    struct transcribe_ext            ext;
    transcribe_nemotron3_diar_preset preset;
};

/* STREAM slot (transcribe_stream_params::family). */
struct transcribe_nemotron3_diar_stream_ext {
    struct transcribe_ext            ext;
    transcribe_nemotron3_diar_preset preset;
};

/* Fill ext.size/kind and preset = DEFAULT. */
TRANSCRIBE_API void transcribe_nemotron3_diar_run_ext_init(struct transcribe_nemotron3_diar_run_ext * ext);
TRANSCRIBE_API void transcribe_nemotron3_diar_stream_ext_init(struct transcribe_nemotron3_diar_stream_ext * ext);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_NEMOTRON3_DIAR_H */
