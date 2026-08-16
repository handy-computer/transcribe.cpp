/*
 * transcribe/crisperwhisper.h - CrisperWhisper 2.0 family run extension.
 *
 * CrisperWhisper decodes in one of two modes, selected by the prompt tag
 * block that opens the decoder input:
 *
 *   verbatim  [verbatim_1]…[verbatim_5]   keeps disfluencies, false starts,
 *                                          repetitions, and the 15 bracketed
 *                                          vocal-event tokens ([UM], [UH],
 *                                          [laughter], …). This is the
 *                                          publisher's default and the reason
 *                                          the family exists.
 *   intended  [intended_1]…[intended_5]   the cleaned-up transcript: no event
 *                                          tokens, disfluencies removed, but
 *                                          every content clause kept.
 *
 * Both are the same weights and the same graph; only the tag block differs.
 * The mode is therefore a per-run choice, not a model-load choice.
 *
 * The default when no extension is supplied comes from the GGUF
 * (stt.crisperwhisper.mode.default, "verbatim" for every shipped checkpoint),
 * so a caller that passes nothing gets the publisher's behaviour.
 */

#ifndef TRANSCRIBE_CRISPERWHISPER_H
#define TRANSCRIBE_CRISPERWHISPER_H

#include "../transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'CWRN' little-endian. */
#define TRANSCRIBE_EXT_KIND_CRISPERWHISPER_RUN 0x4E525743u

enum transcribe_crisperwhisper_mode {
    /* Use stt.crisperwhisper.mode.default from the GGUF. */
    TRANSCRIBE_CRISPERWHISPER_MODE_DEFAULT  = 0,
    TRANSCRIBE_CRISPERWHISPER_MODE_VERBATIM = 1,
    TRANSCRIBE_CRISPERWHISPER_MODE_INTENDED = 2,
};

struct transcribe_crisperwhisper_run_ext {
    struct transcribe_ext ext;

    /* Transcription mode for this run. Default = read from the GGUF. */
    enum transcribe_crisperwhisper_mode mode;
};

/*
 * Fill *ext with defaults (kind + size set, mode = DEFAULT). Call this
 * before setting fields so the struct stays forward-compatible.
 */
TRANSCRIBE_API void transcribe_crisperwhisper_run_ext_init(struct transcribe_crisperwhisper_run_ext * ext);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_CRISPERWHISPER_H */
