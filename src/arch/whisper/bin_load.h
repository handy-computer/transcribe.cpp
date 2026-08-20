// arch/whisper/bin_load.h - legacy whisper.cpp .bin adapter. INTERNAL,
// consumed by transcribe.cpp (magic-byte dispatch) and bin_load.cpp.
//
// The .bin parser (transcribe-bin-loader.{h,cpp}) is architecture-neutral.
// This module is the whisper-specific bridge: it consumes a parsed
// WhisperBinModel, synthesizes hparams/capabilities/special-tokens/languages,
// populates the Tokenizer, builds ctx_meta with canonical-named tensors, and
// streams tensor bytes into the allocated slots. The resulting
// transcribe_model* is indistinguishable from the GGUF whisper_load path.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <vector>

namespace transcribe::whisper {

// Synthesize the generation-time suppression list omitted by the legacy
// format. English-only and multilingual .bin files carry different tokenizers,
// so their non-speech token ids differ. Special-token ids additionally shift
// with the multilingual language count.
std::vector<int32_t> synthesize_bin_suppress_tokens(bool is_multilingual, int n_vocab);

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
transcribe_status load_from_bin(const char *                                path,
                                const struct transcribe_model_load_params * params,
                                struct transcribe_model **                  out_model);

}  // namespace transcribe::whisper
