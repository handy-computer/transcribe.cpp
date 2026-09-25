#include "arch/parakeet/decoder.h"
#include "arch/parakeet/parakeet.h"

#include <cstdint>
#include <vector>

int main() {
    transcribe::parakeet::ParakeetModel model;
    model.hparams.has_prompt = true;
    model.host_decoder.head_kind = transcribe::parakeet::HostHeadKind::RNNT;
    model.host_decoder.n_vocab = 6;
    model.set_languages({ "en-US", "de-DE", "fr-FR" });
    if (model.tok.load_decode_only_raw_bytes({ "word", "<en-US>", "<de-DE>", "<fr-FR>", "<xx-ZZ>", "<blank>" }) !=
        TRANSCRIBE_OK) return 10;

    transcribe_run_params params;
    transcribe_run_params_init(&params);
    std::vector<uint8_t> mask;
    using transcribe::parakeet::resolve_language_block_mask;
    if (resolve_language_block_mask(&model, &params, mask) != TRANSCRIBE_OK || !mask.empty()) return 11;
    const char * allowed[] = { "en-US", "de-DE" };
    params.allowed_languages = allowed;
    params.n_allowed_languages = 2;
    if (resolve_language_block_mask(&model, &params, mask) != TRANSCRIBE_OK ||
        mask.size() != 6 || mask[0] || mask[1] || mask[2] || !mask[3] || !mask[4] || mask[5]) return 12;
    params.language = "de-DE";
    if (resolve_language_block_mask(&model, &params, mask) != TRANSCRIBE_OK || !mask.empty()) return 13;
    params.language = "en-US";
    if (resolve_language_block_mask(&model, &params, mask) != TRANSCRIBE_OK || !mask.empty()) return 14;
    params.language = nullptr;
    const char * unknown[] = { "xx-ZZ" };
    params.allowed_languages = unknown;
    params.n_allowed_languages = 1;
    if (resolve_language_block_mask(&model, &params, mask) != TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE) return 15;
    params.n_allowed_languages = 0;
    if (resolve_language_block_mask(&model, &params, mask) != TRANSCRIBE_OK || !mask.empty()) return 16;

    params.allowed_languages = allowed;
    params.n_allowed_languages = 2;
    if (resolve_language_block_mask(&model, &params, mask) != TRANSCRIBE_OK) return 17;

    // 0 = ordinary token, 1 = English tag, 2 = German tag,
    // 3 = French tag, 4 = unadvertised language tag, 5 = blank.
    const float scores[] = { 1.0f, 6.0f, 7.0f, 9.0f, 0.0f, 0.0f };
    using transcribe::parakeet::argmax_language_masked;
    if (argmax_language_masked(scores, 6, {}) != 3) return 1;

    if (argmax_language_masked(scores, 6, mask) != 2) return 2;
    // Apply the same constraint on a later streaming decoder step.
    const float later_scores[] = { 2.0f, 8.0f, 7.0f, 10.0f, 0.0f, 0.0f };
    if (argmax_language_masked(later_scores, 6, mask) != 1) return 3;

    // An ordinary token or blank still wins on its own raw score.
    const float ordinary_scores[] = { 11.0f, 6.0f, 7.0f, 9.0f, 0.0f, 0.0f };
    if (argmax_language_masked(ordinary_scores, 6, mask) != 0) return 4;
    const float blank_scores[] = { 1.0f, 6.0f, 7.0f, 9.0f, 0.0f, 12.0f };
    if (argmax_language_masked(blank_scores, 6, mask) != 5) return 5;
    if (argmax_language_masked(scores, 6, { 1, 1, 1, 1, 1, 1 }) != -1) return 6;
    return 0;
}
