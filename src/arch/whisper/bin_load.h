// arch/whisper/bin_load.h - legacy whisper.cpp .bin adapter for the
// Whisper architecture.
//
// INTERNAL. Consumed by transcribe.cpp (magic-byte dispatch) and
// arch/whisper/bin_load.cpp.
//
// The .bin parser (transcribe-bin-loader.{h,cpp}) is architecture-
// neutral: it knows the byte layout but nothing about WhisperHParams,
// WhisperWeights, or the canonical tensor names this codebase uses.
// This module is the whisper-specific bridge: it consumes a parsed
// WhisperBinModel, synthesizes hparams + capabilities + special-token
// ids + language list, populates the Tokenizer via the decode-only
// raw-bytes path, builds ctx_meta with canonical-named tensors, allocs
// backend memory, and streams tensor bytes from the .bin into the
// allocated slots.
//
// Once load_from_bin returns, the resulting transcribe_model* is
// indistinguishable from one produced by the GGUF whisper_load path
// — same WhisperModel struct, same encoder/decoder runtime, same
// public ABI.

#pragma once

#include "transcribe.h"

namespace transcribe::whisper {

// Load a legacy whisper.cpp `.bin` (single magic 0x67676d6c) and
// produce a populated transcribe_model* compatible with the rest of
// the Whisper runtime.
//
// Preconditions: the caller has already sniffed the file's magic and
// determined it is `ggml`. The .bin parser inside this function
// validates the hparams against a known Whisper geometry; non-Whisper
// `ggml`-magic files (e.g. Silero VAD) are rejected with
// TRANSCRIBE_ERR_UNSUPPORTED_ARCH.
//
// Status returns mirror the GGUF whisper_load path:
//   TRANSCRIBE_OK                   - success
//   TRANSCRIBE_ERR_FILE_NOT_FOUND   - path missing
//   TRANSCRIBE_ERR_UNSUPPORTED_ARCH - magic ok but not whisper-shaped
//   TRANSCRIBE_ERR_GGUF             - structural / tensor failure
//   TRANSCRIBE_ERR_INVALID_ARG      - bad params
//   TRANSCRIBE_ERR_BACKEND          - backend init failure
transcribe_status load_from_bin(const char *                           path,
                                const struct transcribe_model_load_params * params,
                                struct transcribe_model **             out_model);

} // namespace transcribe::whisper
