// arch/cohere/capabilities.cpp - Cohere ASR capability defaults.

#include "cohere.h"

namespace transcribe::cohere {

void apply_family_invariants(transcribe_capabilities & caps) {
    caps.native_sample_rate = 16000;
    caps.supports_translate = false;
}

} // namespace transcribe::cohere
