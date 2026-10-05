// transcribe-arch.cpp - explicit registry of supported architectures.
//
// The registry is a function-local static array of pointers (no static-
// initializer-order dependency on the per-family TUs, grep-able from one
// file). Adding a family is a two-line edit plus the per-family TU.

#include "transcribe-arch.h"

#include "transcribe-log.h"
#include "transcribe-model.h"

#include <cstring>

namespace transcribe {

// Per-family Arch instances. Defined in src/arch/<family>/model.cpp.
namespace parakeet {
extern const Arch arch;
}

namespace cohere {
extern const Arch arch;
}

namespace canary {
extern const Arch arch;
}

namespace qwen3_asr {
extern const Arch arch;
}

namespace moss {
extern const Arch arch;
}

namespace voxtral {
extern const Arch arch;
}

namespace voxtral_realtime {
extern const Arch arch;
}

namespace canary_qwen {
extern const Arch arch;
}

namespace whisper {
extern const Arch arch;
}

namespace moonshine {
extern const Arch arch;
}

namespace moonshine_streaming {
extern const Arch arch;
}

namespace sensevoice {
extern const Arch arch;
}

namespace funasr_nano {
extern const Arch arch;
}

namespace gigaam {
extern const Arch arch;
}

namespace granite {
extern const Arch arch;
}

namespace granite_nar {
extern const Arch arch;
}

namespace granite5_ctc {
extern const Arch arch;
}

namespace medasr {
extern const Arch arch;
}

namespace sortformer {
extern const Arch arch;
}

namespace ecapa_tdnn {
extern const Arch arch;
}

const Arch * find_arch(const char * name) {
    if (name == nullptr) {
        return nullptr;
    }

    static const Arch * const k_archs[] = {
        &parakeet::arch,         &cohere::arch,      &canary::arch,     &qwen3_asr::arch,    &voxtral::arch,
        &voxtral_realtime::arch, &canary_qwen::arch, &whisper::arch,    &moonshine::arch,    &moonshine_streaming::arch,
        &sensevoice::arch,       &funasr_nano::arch, &gigaam::arch,     &granite::arch,      &granite_nar::arch,
        &medasr::arch,           &moss::arch,        &sortformer::arch, &granite5_ctc::arch, &ecapa_tdnn::arch,
    };
    constexpr size_t k_n = sizeof(k_archs) / sizeof(k_archs[0]);

    for (size_t i = 0; i < k_n; ++i) {
        const Arch * a = k_archs[i];
        if (a == nullptr || a->name == nullptr) {
            continue;
        }
        if (std::strcmp(a->name, name) == 0) {
            return a;
        }
    }
    return nullptr;
}

transcribe_status resolve_roles(transcribe_model * model) {
    if (model == nullptr || model->arch == nullptr) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    const Arch & arch        = *model->arch;
    const char * name        = arch.name != nullptr ? arch.name : "(unknown)";
    const bool   has_asr     = arch.init_context != nullptr && arch.run != nullptr;
    const bool   has_diarize = arch.diarize != nullptr;
    const bool   has_langid  = arch.langid != nullptr;

    if (model->roles == 0 && has_asr) {
        model->roles = TRANSCRIBE_ROLE_ASR;
    }

    const uint32_t known = TRANSCRIBE_ROLE_ASR | TRANSCRIBE_ROLE_DIARIZE | TRANSCRIBE_ROLE_LANGID;
    const char *   why   = nullptr;
    if (model->roles == 0) {
        why = "serves no role";
    } else if ((model->roles & ~known) != 0) {
        why = "sets an unknown role bit";
    } else if ((model->roles & TRANSCRIBE_ROLE_ASR) != 0 && !has_asr) {
        why = "sets the ASR role without init_context / run hooks";
    } else if ((model->roles & TRANSCRIBE_ROLE_DIARIZE) != 0 && !has_diarize) {
        why = "sets the DIARIZE role without a diarize ops table";
    } else if ((model->roles & TRANSCRIBE_ROLE_LANGID) != 0 && !has_langid) {
        why = "sets the LANGID role without a langid ops table";
    }
    if (why != nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "transcribe_model_load_file: arch '%s' %s (roles 0x%x)", name, why,
                static_cast<unsigned>(model->roles));
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    return TRANSCRIBE_OK;
}

}  // namespace transcribe
