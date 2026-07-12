/*
 * include/transcribe/qwen3_asr.h - Qwen3-ASR-family public run extension.
 *
 * The extension supplies optional recognition context as the body of the
 * Qwen chat template's system message. Audio always remains in the user
 * message; callers must not construct or inject Qwen special tokens.
 */

#ifndef TRANSCRIBE_QWEN3_ASR_H
#define TRANSCRIBE_QWEN3_ASR_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'Q3RN' little-endian = 0x4E523351 */
#define TRANSCRIBE_EXT_KIND_QWEN3_ASR_RUN 0x4E523351u

/*
 * Optional Qwen3-ASR recognition context. Reached through
 * transcribe_run_params::family when the loaded model advertises this kind
 * on TRANSCRIBE_EXT_SLOT_RUN.
 *
 * NULL or an empty string preserves the historical empty system message.
 * The text is encoded by the loaded model tokenizer and inserted only between
 * the system role marker and its closing chat marker. Qwen control markers
 * (the <|...|> form) are rejected rather than being interpreted as template
 * structure. The library consumes the bytes synchronously; callers may free
 * context after transcribe_run() returns.
 */
struct transcribe_qwen3_asr_run_ext {
    struct transcribe_ext ext;
    const char *          context;
};

/* Fills ext.size/kind and leaves context NULL. */
TRANSCRIBE_API void transcribe_qwen3_asr_run_ext_init(struct transcribe_qwen3_asr_run_ext * ext);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_QWEN3_ASR_H */
