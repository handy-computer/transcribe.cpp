// transcribe-prompting.h - shared helpers for the generic prompting fields
// (transcribe_run_params::vocabulary / prompt / prefix, TASK_INSTRUCT).
//
// INTERNAL. The dispatcher validates the fields and removes the ones a model
// ignores, so a family acts on whatever is set in the params view it
// receives. Families own the rendering and their budget rules.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transcribe {

class Tokenizer;

namespace prompting {

inline bool has_text(const char * s) {
    return s != nullptr && s[0] != '\0';
}

inline bool is_instruct(const transcribe_run_params * p) {
    return p != nullptr && p->task == TRANSCRIBE_TASK_INSTRUCT;
}

// Non-empty vocabulary terms in caller order. Assumes the dispatcher's
// shape validation (non-negative count, no NULL entries) already passed.
std::vector<std::string> terms(const transcribe_run_params * p);

std::string join(const std::vector<std::string> & terms, const char * sep);

// `s` without leading / trailing whitespace (std::isspace).
std::string strip(const std::string & s);

// Returns INVALID_ARG when `text` contains a literal of one of the
// tokenizer's control tokens (e.g. "<|im_end|>", "[INST]"): the upstream
// tokenizers would encode it as a control id, ours never do. `what` names
// the field in the error log.
transcribe_status check_plain_text(const Tokenizer & tok, const std::string & text, const char * what);

// check_plain_text + encode.
transcribe_status encode_plain(const Tokenizer &      tok,
                               const std::string &    text,
                               std::vector<int32_t> & out_ids,
                               const char *           what);

// Tokenized vocabulary + context for one text slot, fitted to `budget`
// tokens (negative = 0). Terms render as `lead + join(terms, sep) + trail`;
// the context is encoded as given. Over budget, the context is trimmed
// first, keeping its most recent tokens, then terms are dropped from the end
// of the list, with one WARN. Both are control-token checked.
//
// A fit that comes in under a smaller budget is also the fit for that
// budget; batch paths rely on this to share one fit across rows.
struct TermsFormat {
    std::string lead;
    std::string sep;
    std::string trail;
};

struct FittedPrompt {
    std::vector<int32_t> term_ids;
    std::vector<int32_t> ctx_ids;
    size_t               n_terms = 0;  // terms kept
    std::string          terms_text;   // the kept terms rendered (lead + join + trail); empty if none

    size_t n_tokens() const { return term_ids.size() + ctx_ids.size(); }
};

transcribe_status fit_terms_and_context(const Tokenizer &                tok,
                                        const std::vector<std::string> & terms,
                                        const TermsFormat &              fmt,
                                        const std::string &              ctx,
                                        int                              budget,
                                        const char *                     family,
                                        FittedPrompt &                   out);

}  // namespace prompting
}  // namespace transcribe
