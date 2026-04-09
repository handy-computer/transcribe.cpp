// arch/parakeet/capabilities.cpp - Parakeet capability defaults.
//
// Per the 2B revision after the variant-dispatch review (see
// RESUME.md "Decisions still load-bearing" -> "stt.variant is
// descriptive metadata, not a behavioral discriminator"):
//
//   - The previous variant-driven resolver is gone. Capability
//     resolution is now KV-driven via transcribe::read_capability_kv.
//   - This file holds only the family-default population, applied
//     before the KV reader runs. KV present overrides the family
//     default; KV absent leaves the family default in place. PLAN.md
//     "GGUF KV is authoritative; if a key is absent, the architecture
//     supplies a default" — this function supplies those defaults.
//
// What stays here vs what goes to KV:
//
//   - native_sample_rate is set here, not exposed via KV. Parakeet's
//     feature extractor is a fixed 16 kHz mel bank for every
//     published variant; treating it as a KV-overridable field would
//     just create a footgun. If a future variant truly used a
//     different rate, the right answer would be to add a separate
//     parakeet handler subtree, not to flip a KV.
//
//   - supports_translate defaults to false here. Strictly speaking
//     the KV reader could overwrite it, but no published Parakeet
//     variant has a translate task head and the converter is not
//     expected to emit stt.capability.translate for this family. The
//     default sticks because nothing overrides it.
//
//   - supports_language_detect, supports_streaming come from KV. The
//     family default is false; v3 multilingual GGUFs are expected to
//     emit stt.capability.lang_detect = true and the KV reader will
//     pick it up.
//
//   - n_languages / languages come from general.languages via
//     read_languages_kv (called by the family handler after this
//     function runs). Absent leaves the model in the documented
//     "information gap, not a claim" state.

#include "parakeet.h"

namespace transcribe::parakeet {

void apply_family_invariants(transcribe_capabilities & caps) {
    // Parakeet's feature extractor is a fixed 16 kHz mel bank (NeMo
    // AudioToMelSpectrogramPreprocessor). Every published variant — v2
    // English and v3 multilingual — uses it. The CLI rejects non-
    // 16 kHz WAVs at load time so this number is the contract.
    caps.native_sample_rate = 16000;

    // Family default. No published Parakeet variant has a translate
    // task head; NeMo ships translation via Canary, not Parakeet.
    // KV is authoritative per PLAN.md, so a converter that explicitly
    // wrote stt.capability.translate could in principle override
    // this — but for Parakeet we never expect that to happen.
    caps.supports_translate = false;
}

} // namespace transcribe::parakeet
