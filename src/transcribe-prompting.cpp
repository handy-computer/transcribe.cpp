// transcribe-prompting.cpp - shared helpers for the generic prompting fields.

#include "transcribe-prompting.h"

#include "transcribe-log.h"
#include "transcribe-tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace transcribe::prompting {

std::vector<std::string> terms(const transcribe_run_params * p) {
    std::vector<std::string> out;
    if (p == nullptr || p->vocabulary == nullptr || p->n_vocabulary <= 0) {
        return out;
    }
    out.reserve(static_cast<size_t>(p->n_vocabulary));
    for (int32_t i = 0; i < p->n_vocabulary; ++i) {
        if (has_text(p->vocabulary[i])) {
            out.emplace_back(p->vocabulary[i]);
        }
    }
    return out;
}

std::string strip(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) {
        ++a;
    }
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) {
        --b;
    }
    return s.substr(a, b - a);
}

std::string join(const std::vector<std::string> & terms, const char * sep) {
    std::string out;
    for (size_t i = 0; i < terms.size(); ++i) {
        if (i != 0) {
            out += sep;
        }
        out += terms[i];
    }
    return out;
}

transcribe_status check_plain_text(const Tokenizer & tok, const std::string & text, const char * what) {
    // Candidate literals are "<...>" and "[...]" spans up to a control
    // token's plausible length. "<|...|>" pieces are rejected whenever the
    // vocab has them (Whisper's rule: those are never plain text); other
    // shapes only when the vocab types them CONTROL or they are BOS/EOS.
    constexpr size_t k_max_literal = 48;
    for (size_t i = 0; i < text.size(); ++i) {
        const char open = text[i];
        if (open != '<' && open != '[') {
            continue;
        }
        const char   close = open == '<' ? '>' : ']';
        const size_t end   = text.find(close, i + 1);
        if (end == std::string::npos || end - i + 1 > k_max_literal) {
            continue;
        }
        const std::string piece = text.substr(i, end - i + 1);
        const int         id    = tok.find(piece);
        if (id < 0) {
            continue;
        }
        const bool pipe_form = piece.size() >= 4 && piece[1] == '|' && piece[piece.size() - 2] == '|';
        if (pipe_form || tok.is_control(id) || id == tok.bos_id() || id == tok.eos_id()) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "%s contains the control token \"%s\" (id %d); control tokens are "
                    "not accepted in prompting text",
                    what, piece.c_str(), id);
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }
    return TRANSCRIBE_OK;
}

transcribe_status encode_plain(const Tokenizer &      tok,
                               const std::string &    text,
                               std::vector<int32_t> & out_ids,
                               const char *           what) {
    if (const transcribe_status st = check_plain_text(tok, text, what); st != TRANSCRIBE_OK) {
        return st;
    }
    out_ids.clear();
    if (text.empty()) {
        return TRANSCRIBE_OK;
    }
    return tok.encode(text, out_ids);
}

transcribe_status fit_terms_and_context(const Tokenizer &                tok,
                                        const std::vector<std::string> & terms,
                                        const TermsFormat &              fmt,
                                        const std::string &              ctx,
                                        int                              budget,
                                        const char *                     family,
                                        FittedPrompt &                   out) {
    out                 = FittedPrompt{};
    const size_t cap    = static_cast<size_t>(std::max(budget, 0));
    // The first k terms as rendered text, and its ids.
    auto         render = [&](size_t k) {
        std::string text;
        if (k > 0) {
            text = fmt.lead;
            for (size_t i = 0; i < k; ++i) {
                text += (i != 0 ? fmt.sep : std::string()) + terms[i];
            }
            text += fmt.trail;
        }
        return text;
    };
    auto encode_terms = [&](size_t k, std::vector<int32_t> & ids) -> transcribe_status {
        ids.clear();
        return k == 0 ? TRANSCRIBE_OK : encode_plain(tok, render(k), ids, "vocabulary");
    };
    size_t kept = terms.size();
    if (const transcribe_status st = encode_terms(kept, out.term_ids); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = encode_plain(tok, ctx, out.ctx_ids, "prompt"); st != TRANSCRIBE_OK) {
        return st;
    }
    const size_t ctx_in = out.ctx_ids.size();
    if (out.n_tokens() > cap) {
        const size_t room = out.term_ids.size() < cap ? cap - out.term_ids.size() : 0;
        out.ctx_ids.erase(out.ctx_ids.begin(), out.ctx_ids.end() - std::min(room, out.ctx_ids.size()));
        if (out.term_ids.size() > cap) {
            // The most terms that fit, by binary search over the count (the
            // token count grows with it): 0 terms always fit, all do not.
            size_t               lo = 0, hi = kept;
            std::vector<int32_t> ids;
            while (hi - lo > 1) {
                const size_t mid = lo + (hi - lo) / 2;
                if (const transcribe_status st = encode_terms(mid, ids); st != TRANSCRIBE_OK) {
                    return st;
                }
                (ids.size() <= cap ? lo : hi) = mid;
            }
            kept = lo;
            if (const transcribe_status st = encode_terms(kept, out.term_ids); st != TRANSCRIBE_OK) {
                return st;
            }
        }
        char terms_note[96] = "";
        if (kept < terms.size()) {
            std::snprintf(terms_note, sizeof(terms_note), "dropped %zu of %zu vocabulary terms", terms.size() - kept,
                          terms.size());
        }
        char ctx_note[96] = "";
        if (out.ctx_ids.size() < ctx_in) {
            std::snprintf(ctx_note, sizeof(ctx_note), "kept the last %zu of %zu context tokens", out.ctx_ids.size(),
                          ctx_in);
        }
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "%s: %s%s%s (prompt budget: %zu tokens)", family, terms_note,
                (terms_note[0] != '\0' && ctx_note[0] != '\0') ? "; " : "", ctx_note, cap);
    }
    out.n_terms    = kept;
    out.terms_text = render(kept);
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::prompting
