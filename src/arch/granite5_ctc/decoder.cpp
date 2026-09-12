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

// Byte-level BPE word-boundary marker. GPT-2 byte encoding maps the
// space character (0x20) to U+0120 "Ġ", which is C4 A0 in UTF-8, so a
// raw piece starting with those two bytes opens a new word.
//
// This is the one place parakeet's CTC result builder does NOT carry
// over: it splits on the SentencePiece marker U+2581 (E2 96 81), which
// never appears in this vocabulary. Using it here would yield exactly
// one word spanning the whole clip.
constexpr const char k_bpe_space_marker[]  = "\xC4\xA0";
constexpr int        k_bpe_space_marker_len = 2;

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

void ctc_greedy_collapse(const float *           logits,
                         int                     t_len,
                         int                     vocab,
                         int                     blank_id,
                         std::vector<CtcToken> & out_tokens) {
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

    const double frame_ms = ms_per_encoder_frame(hp);

    auto clamp_ms = [&](int64_t v) {
        if (clip_ms > 0 && v > clip_ms) {
            return clip_ms;
        }
        return v;
    };

    cc.tokens.reserve(tokens.size());
    for (const CtcToken & t : tokens) {
        transcribe_session::TokenEntry te;
        te.id    = t.id;
        te.p     = t.p;
        // CTC alignments are peaky: a label is emitted on the single
        // frame where it wins, so the token spans exactly that frame.
        te.t0_ms = clamp_ms(static_cast<int64_t>(std::llround(frame_ms * static_cast<double>(t.frame))));
        te.t1_ms = clamp_ms(static_cast<int64_t>(std::llround(frame_ms * static_cast<double>(t.frame + 1))));
        te.text  = tok.decode(&te.id, 1);
        cc.tokens.push_back(std::move(te));
    }

    if (cc.tokens.empty()) {
        cc.has_result   = true;
        cc.result_kind  = TRANSCRIBE_TIMESTAMPS_NONE;
        return;
    }

    transcribe_session::SegmentEntry seg;
    seg.t0_ms       = cc.tokens.front().t0_ms;
    seg.t1_ms       = cc.tokens.back().t1_ms;
    seg.first_token = 0;
    seg.n_tokens    = static_cast<int>(cc.tokens.size());
    seg.first_word  = 0;

    transcribe_session::WordEntry cur_word;
    bool                          cur_word_open = false;

    auto close_word = [&](int end_token_index) {
        cur_word.t1_ms    = cc.tokens[static_cast<size_t>(end_token_index - 1)].t1_ms;
        cur_word.n_tokens = end_token_index - cur_word.first_token;
        cc.words.push_back(std::move(cur_word));
        cur_word = transcribe_session::WordEntry{};
    };

    for (size_t i = 0; i < cc.tokens.size(); ++i) {
        const auto &      tk        = cc.tokens[i];
        const std::string & raw_piece = tok.token(tk.id);
        // The first token always opens a word: a byte-level BPE
        // utterance carries no marker on its opening piece.
        const bool starts_word =
            (i == 0) || (raw_piece.size() >= static_cast<size_t>(k_bpe_space_marker_len) &&
                         std::memcmp(raw_piece.data(), k_bpe_space_marker, k_bpe_space_marker_len) == 0);
        if (starts_word) {
            if (cur_word_open) {
                close_word(static_cast<int>(i));
            }
            cur_word.t0_ms       = tk.t0_ms;
            cur_word.first_token = static_cast<int>(i);
            cur_word.seg_index   = 0;
            cur_word_open        = true;
        }
        cc.tokens[i].seg_index  = 0;
        cc.tokens[i].word_index = static_cast<int>(cc.words.size());
    }
    if (cur_word_open) {
        close_word(static_cast<int>(cc.tokens.size()));
    }

    // Word text comes from a single decode over the word's ids so the
    // ByteLevel decoder sees the whole span; the opener's leading space
    // is then trimmed.
    std::vector<int> id_buf;
    for (auto & wd : cc.words) {
        id_buf.clear();
        id_buf.reserve(static_cast<size_t>(wd.n_tokens));
        for (int j = 0; j < wd.n_tokens; ++j) {
            id_buf.push_back(cc.tokens[static_cast<size_t>(wd.first_token + j)].id);
        }
        std::string text = tok.decode(id_buf.data(), wd.n_tokens);
        if (!text.empty() && text.front() == ' ') {
            text.erase(text.begin());
        }
        wd.text = std::move(text);
    }
    seg.n_words = static_cast<int>(cc.words.size());

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
    cc.result_kind = TRANSCRIBE_TIMESTAMPS_WORD;
}

}  // namespace transcribe::granite5_ctc
