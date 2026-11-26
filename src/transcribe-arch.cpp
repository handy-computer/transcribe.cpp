// transcribe-arch.cpp - explicit registry of supported architectures.
//
// The registry is a function-local static array of pointers (no static-
// initializer-order dependency on the per-family TUs, grep-able from one
// file). Adding a family is a two-line edit plus the per-family TU.

#include "transcribe-arch.h"

#include "transcribe-log.h"
#include "transcribe-model.h"
#include "transcribe-turn.h"

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

namespace silero_vad {
extern const Arch arch;
}

namespace smart_turn {
extern const Arch arch;
}

namespace sava {
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
        &silero_vad::arch,       &smart_turn::arch,  &sava::arch,
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

namespace {

// Why a TURN ops table does not fit the model it loaded, or nullptr. The
// dispatcher calls score_clip for a clip model (frame_samples == 0) and
// step / reset for a streaming one, unchecked.
const char * turn_ops_mismatch(const TurnOps & ops, const transcribe_model * model) {
    if (ops.info == nullptr || ops.new_session == nullptr) {
        return "sets the TURN role with a turn ops table missing info / new_session";
    }
    const TurnModelInfo & info = ops.info(model);
    if (info.frame_samples < 0 || info.max_audio_samples < 0 || info.speech_lookahead_frames < 0) {
        return "reports negative turn sizes";
    }
    if (info.frame_samples == 0) {
        if (ops.score_clip == nullptr) {
            return "is a clip TURN model without score_clip";
        }
        return nullptr;
    }
    if (ops.step == nullptr || ops.reset == nullptr) {
        return "is a streaming TURN model without step / reset";
    }
    // A streaming model has no context limit: run would crop the clip while
    // a stream of the same audio is not cropped, and the two would disagree.
    if (info.max_audio_samples > 0) {
        return "is a streaming TURN model with max_audio_samples > 0";
    }
    return nullptr;
}

}  // namespace

transcribe_status resolve_roles(transcribe_model * model) {
    if (model == nullptr || model->arch == nullptr) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    const Arch & arch        = *model->arch;
    const char * name        = arch.name != nullptr ? arch.name : "(unknown)";
    const bool   has_asr     = arch.init_context != nullptr && arch.run != nullptr;
    const bool   has_diarize = arch.diarize != nullptr;
    const bool   has_langid  = arch.langid != nullptr;
    const bool   has_vad     = arch.vad != nullptr;
    const bool   has_turn    = arch.turn != nullptr;

    if (model->roles == 0 && has_asr) {
        model->roles = TRANSCRIBE_ROLE_ASR;
    }

    const uint32_t known = TRANSCRIBE_ROLE_ASR | TRANSCRIBE_ROLE_DIARIZE | TRANSCRIBE_ROLE_LANGID |
                           TRANSCRIBE_ROLE_VAD | TRANSCRIBE_ROLE_TURN;
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
    } else if ((model->roles & TRANSCRIBE_ROLE_VAD) != 0 && !has_vad) {
        why = "sets the VAD role without a vad ops table";
    } else if ((model->roles & TRANSCRIBE_ROLE_TURN) != 0 && !has_turn) {
        why = "sets the TURN role without a turn ops table";
    } else if ((model->roles & TRANSCRIBE_ROLE_TURN) != 0) {
        why = turn_ops_mismatch(*arch.turn, model);
    }
    if (why != nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "transcribe_model_load_file: arch '%s' %s (roles 0x%x)", name, why,
                static_cast<unsigned>(model->roles));
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    return TRANSCRIBE_OK;
}

}  // namespace transcribe
