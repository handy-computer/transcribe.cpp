// arch/crisperwhisper/public.cpp - CrisperWhisper public C entry points.
//
// The run-extension init lives here rather than in the generic transcribe.cpp
// so the central dispatcher stays family-agnostic (matching whisper /
// parakeet / moonshine). Public ABI from include/transcribe/crisperwhisper.h.

#include "transcribe/crisperwhisper.h"

#include <cstring>

extern "C" void transcribe_crisperwhisper_run_ext_init(struct transcribe_crisperwhisper_run_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size = sizeof(*p);
    p->ext.kind = TRANSCRIBE_EXT_KIND_CRISPERWHISPER_RUN;
    // mode = TRANSCRIBE_CRISPERWHISPER_MODE_DEFAULT (0, covered by memset):
    // read stt.crisperwhisper.mode.default from the GGUF, which is "verbatim"
    // on every shipped checkpoint.
}
