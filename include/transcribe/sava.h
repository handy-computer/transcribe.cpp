/*
 * include/transcribe/sava.h - SAVA-family public extension.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. SAVA (Inworld)
 * serves the VAD and TURN roles (include/transcribe/vad.h, turn.h). Its one
 * extension tunes the reference rule transcribe_turn_run uses to cut a clip
 * into turn segments (Inworld's sava.Policy); probe with
 * transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_TURN_RUN,
 * TRANSCRIBE_EXT_KIND_SAVA_TURN).
 *
 * FourCC kinds are reserved in docs/extension-kinds.md.
 */

#ifndef TRANSCRIBE_SAVA_H
#define TRANSCRIBE_SAVA_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'SVTN' little-endian = 0x4E545653 (TURN_RUN slot) */
#define TRANSCRIBE_EXT_KIND_SAVA_TURN 0x4E545653u

/*
 * transcribe_turn_params::family: the turn-segment rule. A turn opens on the
 * first frame judged speech, pre_roll_ms earlier; while it is open, a silent
 * frame closes it when the turn probability reaches
 * transcribe_turn_params::threshold (after min_silence_before_check_ms of
 * pause), or when the pause reaches max_silence_ms. A turn with less than
 * min_speech_ms of speech and less than min_confident_speech_ms of confident
 * speech is dropped. A negative field keeps the model's default (Inworld's,
 * in parentheses); transcribe_sava_turn_ext_init sets every field to -1.
 * Out-of-range values are INVALID_ARG before the previous result is touched.
 */
struct transcribe_sava_turn_ext {
    struct transcribe_ext ext;
    double                speech_threshold;           /* speech p >= this is speech, 0 to 1 (0.5) */
    double                confident_speech_threshold; /* speech p >= this is confident, 0 to 1 (0.9) */
    int32_t               pre_roll_ms;                /* (800) */
    int32_t               min_silence_before_check_ms; /* (0) */
    int32_t               max_silence_ms;             /* (1600) */
    int32_t               min_speech_ms;              /* (400) */
    int32_t               min_confident_speech_ms;    /* (200) */
};

/* Fills ext.size/kind and sets every field to -1 (the model's defaults). */
TRANSCRIBE_API void transcribe_sava_turn_ext_init(struct transcribe_sava_turn_ext * ext);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_SAVA_H */
