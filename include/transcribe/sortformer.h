/*
 * include/transcribe/sortformer.h - Sortformer-family public extension.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. Holds the
 * streaming-operating-point run extension, its kind constant, and its
 * init function.
 *
 * Sortformer (diar_streaming_sortformer_4spk-v2.1) is a diarization-only model: a run
 * produces no text; the product is the who-spoke-when rows read back via
 * transcribe_n_speaker_segments / transcribe_get_speaker_segment
 * (TRANSCRIBE_FEATURE_DIARIZATION). The compute core is streaming
 * (AOSC speaker cache + FIFO) but the shipped entry point is the batch
 * transcribe_run over a whole recording, so this extension lives on the
 * RUN slot. A future push-audio entry point (transcribe_stream_begin/
 * feed) would register a separate STREAM-slot kind taking the same
 * preset enum.
 *
 * Probe via transcribe_model_accepts_ext_kind(model,
 * TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM)
 * before pointing transcribe_run_params::family at the struct.
 *
 * FourCC kinds are reserved in docs/extension-kinds.md.
 */

#ifndef TRANSCRIBE_SORTFORMER_H
#define TRANSCRIBE_SORTFORMER_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'SFST' little-endian = 0x54534653 */
#define TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM 0x54534653u

/*
 * Streaming operating point (latency / accuracy trade-off).
 *
 * The model processes audio in fixed chunks and carries speaker identity
 * across chunks in a bounded cache; the operating point sets the chunk
 * geometry. Each named preset is a jointly-tuned bundle published by the
 * upstream model (chunk length, lookahead, FIFO and speaker-cache
 * geometry) - the menu is discrete, not a continuous latency dial, and
 * only these bundles are accuracy-validated (AMI DER, see
 * docs/porting/families/sortformer.md).
 *
 *   DEFAULT             The GGUF-shipped checkpoint configuration.
 *   VERY_HIGH_LATENCY   ~30.4 s algorithmic lookahead (chunk 340 + rc 40
 *                       frames @ 80 ms). Highest accuracy; the published
 *                       operating point for offline file processing.
 *   HIGH_LATENCY        ~10.0 s lookahead (chunk 124 + rc 1).
 *   LOW_LATENCY         ~1.04 s lookahead (chunk 6 + rc 7). The
 *                       real-time operating point. Note: much higher
 *                       compute per audio second than the larger chunks
 *                       (many small windows); see the family doc for
 *                       measured throughput.
 *
 * Values outside the enum range are rejected by transcribe_run with
 * TRANSCRIBE_ERR_INVALID_ARG before the previous result is cleared.
 */
typedef enum {
    TRANSCRIBE_SORTFORMER_PRESET_DEFAULT           = 0,
    TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY = 1,
    TRANSCRIBE_SORTFORMER_PRESET_HIGH_LATENCY      = 2,
    TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY       = 3,
} transcribe_sortformer_preset;

struct transcribe_sortformer_stream_ext {
    struct transcribe_ext        ext;
    transcribe_sortformer_preset preset;
};

/* Fills ext.size/kind and preset = DEFAULT (GGUF-shipped cfg). */
TRANSCRIBE_API void transcribe_sortformer_stream_ext_init(struct transcribe_sortformer_stream_ext * ext);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_SORTFORMER_H */
