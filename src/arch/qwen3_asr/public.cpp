// arch/qwen3_asr/public.cpp - Qwen3-ASR family public C entry points.

#include "transcribe/qwen3_asr.h"

#include <cstring>

extern "C" void transcribe_qwen3_asr_run_ext_init(struct transcribe_qwen3_asr_run_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size = sizeof(*p);
    p->ext.kind = TRANSCRIBE_EXT_KIND_QWEN3_ASR_RUN;
}
