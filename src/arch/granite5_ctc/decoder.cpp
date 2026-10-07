// arch/granite5_ctc/decoder.cpp - see decoder.h for the contract.

#include "decoder.h"

#include "transcribe-log.h"
#include "transcribe-session.h"
#include "transcribe-tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace transcribe::granite5_ctc {

namespace {

int argmax_row(const float * row, int n) {
    int   best   = 0;
    float best_v = row[0];
    for (int i = 1; i < n; ++i) {
        if (row[i] > best_v) {
            best_v = row[i];
            best   = i;
        }
    }
    return best;
}

// softmax probability of class `idx` in `row`, computed in one pass
// against the row max for stability.
float softmax_prob(const float * row, int n, int idx) {
    const float max_v = row[argmax_row(row, n)];
    double      sum   = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::exp(static_cast<double>(row[i] - max_v));
    }
    if (sum <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(std::exp(static_cast<double>(row[idx] - max_v)) / sum);
}

void collapse_whitespace(std::string & s) {
    std::string out;
    out.reserve(s.size());
    bool prev_space = true;  // also trims the leading space
    for (char c : s) {
        const bool is_space = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (is_space) {
            if (!prev_space) {
                out.push_back(' ');
            }
            prev_space = true;
        } else {
            out.push_back(c);
            prev_space = false;
        }
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    s = std::move(out);
}

}  // namespace

void ctc_greedy_collapse(const float * logits, int t_len, int vocab, int blank_id, std::vector<CtcToken> & out_tokens) {
    out_tokens.clear();
    if (logits == nullptr || t_len <= 0 || vocab <= 0) {
        return;
    }
    // groupby-then-drop-blank, in that order: `prev` tracks the raw
    // per-frame label including blanks, so [A, blank, A] keeps both A's
    // while [A, A] keeps one.
    int prev = -1;
    for (int t = 0; t < t_len; ++t) {
        const float * row   = logits + static_cast<size_t>(t) * static_cast<size_t>(vocab);
        const int     label = argmax_row(row, vocab);
        if (label == prev) {
            continue;
        }
        prev = label;
        if (label == blank_id) {
            continue;
        }
        CtcToken tok;
        tok.id    = label;
        tok.p     = softmax_prob(row, vocab, label);
        tok.frame = t;
        out_tokens.push_back(tok);
    }
}

double ms_per_encoder_frame(const Granite5CtcHParams & hp) {
    if (hp.fe_sample_rate <= 0) {
        return 0.0;
    }
    int64_t samples = static_cast<int64_t>(hp.fe_hop_length) * hp.fe_stack_factor;
    for (size_t i = 0; i < hp.enc_subsample_layers.size(); ++i) {
        samples *= 2;
    }
    return 1000.0 * static_cast<double>(samples) / static_cast<double>(hp.fe_sample_rate);
}

void build_result(transcribe_session &          cc,
                  const transcribe::Tokenizer & tok,
                  const Granite5CtcHParams &    hp,
                  const std::vector<CtcToken> & tokens,
                  int64_t                       clip_ms) {
    cc.tokens.clear();
    cc.words.clear();
    cc.segments.clear();
    cc.full_text.clear();
    cc.raw_text.clear();

    cc.tokens.reserve(tokens.size());
    for (const CtcToken & t : tokens) {
        transcribe_session::TokenEntry te;
        te.id   = t.id;
        te.p    = t.p;
        // No timestamps: t0_ms/t1_ms stay 0. The emitting frame is known
        // (t.frame) but a token's true extent is not -- CTC is peaky, so
        // the label occupies the single frame where it wins regardless of
        // how long the sound lasted. See capabilities.cpp.
        te.text = tok.decode(&te.id, 1);
        cc.tokens.push_back(std::move(te));
    }

    if (cc.tokens.empty()) {
        cc.has_result  = true;
        cc.result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        return;
    }

    // Exactly one segment per run, spanning the whole clip. No
    // segmentation policy is defined for this family and no timestamps
    // are published, so the segment is a container for the text rather
    // than a timing claim; t0/t1 are the clip bounds, not emission times.
    transcribe_session::SegmentEntry seg;
    seg.t0_ms       = 0;
    seg.t1_ms       = clip_ms > 0 ? clip_ms : 0;
    seg.first_token = 0;
    seg.n_tokens    = static_cast<int>(cc.tokens.size());
    seg.first_word  = 0;
    seg.n_words     = 0;

    for (size_t i = 0; i < cc.tokens.size(); ++i) {
        cc.tokens[i].seg_index  = 0;
        cc.tokens[i].word_index = -1;  // no words published
    }

    std::vector<int> all_ids;
    all_ids.reserve(cc.tokens.size());
    for (const auto & tk : cc.tokens) {
        all_ids.push_back(tk.id);
    }
    std::string full = tok.decode(all_ids.data(), static_cast<int>(all_ids.size()));
    cc.raw_text      = full;
    collapse_whitespace(full);
    seg.text     = full;
    cc.full_text = full;

    cc.segments.push_back(std::move(seg));
    cc.has_result  = true;
    cc.result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
}

}  // namespace transcribe::granite5_ctc
