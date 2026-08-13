// arch/crisperwhisper/word_timing.cpp - port of crisperwhisper/word_timing.py.
//
// Pipeline, in reference order:
//   1. tokens -> words (explicit space token 220 is a separator, not content)
//   2. per-token log-prob over encoder frames, from sharpened attention
//   3. per-frame blank log-prob from mel-band energy
//   4. word-level Viterbi over [blank, word, blank, word, ..., blank]
//   5. split short inter-word gaps at their midpoint

#include "word_timing.h"

#include "crisperwhisper.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace transcribe::crisperwhisper {

namespace {

constexpr int   k_space_token_id = 220;    // explicit space token
constexpr float k_attn_eps       = 1e-8f;  // token_logp_from_attention eps
constexpr float k_blank_eps      = 1e-6f;  // blank_logp_from_mel_energy eps
constexpr float k_neg_inf        = -1e9f;  // Viterbi -inf sentinel

bool is_special_piece(const std::string & p) {
    if (p.size() >= 4 && p.compare(0, 2, "<|") == 0 && p.compare(p.size() - 2, 2, "|>") == 0) {
        return true;
    }
    if (p.rfind("[verbatim_", 0) == 0 || p.rfind("[intended_", 0) == 0) {
        return true;
    }
    return p == "<ctx>" || p == "<ectx>" || p == "<htx>" || p == "<ehtx>" || p == "<vtx>" || p == "<evtx>" ||
           p == "<sot>" || p == "<eot>";
}

bool is_space_piece(int id, const std::string & p) { return id == k_space_token_id || p == " "; }

std::string trim(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

// numpy.percentile with the default linear interpolation.
float percentile_linear(std::vector<float> sorted_vals, float q) {
    if (sorted_vals.empty()) {
        return 0.0f;
    }
    std::sort(sorted_vals.begin(), sorted_vals.end());
    const double pos = (static_cast<double>(sorted_vals.size()) - 1.0) * (static_cast<double>(q) / 100.0);
    const size_t lo  = static_cast<size_t>(std::floor(pos));
    const size_t hi  = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) {
        return sorted_vals[lo];
    }
    const double frac = pos - static_cast<double>(lo);
    return static_cast<float>(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac);
}

// numpy.interp onto a uniform grid: _resample_1d.
std::vector<float> resample_1d(const std::vector<float> & x, int target_len) {
    std::vector<float> out;
    if (target_len <= 0) {
        return out;
    }
    if (static_cast<int>(x.size()) == target_len) {
        return x;
    }
    out.assign(static_cast<size_t>(target_len), 0.0f);
    if (x.size() <= 1) {
        const float v = x.empty() ? 0.0f : x[0];
        std::fill(out.begin(), out.end(), v);
        return out;
    }
    const double n_src = static_cast<double>(x.size());
    for (int i = 0; i < target_len; ++i) {
        // src grid is linspace(0,1,x.size()), dst grid linspace(0,1,target_len)
        const double t   = (target_len == 1) ? 0.0 : static_cast<double>(i) / (target_len - 1.0);
        const double pos = t * (n_src - 1.0);
        const size_t lo  = static_cast<size_t>(std::floor(pos));
        const size_t hi  = std::min(lo + 1, x.size() - 1);
        const double f   = pos - static_cast<double>(lo);
        out[static_cast<size_t>(i)] = static_cast<float>(x[lo] * (1.0 - f) + x[hi] * f);
    }
    return out;
}

// log(sum(exp(.))) accumulate, matching np.logaddexp.reduce.
float logaddexp(float a, float b) {
    if (a == k_neg_inf) {
        return b;
    }
    if (b == k_neg_inf) {
        return a;
    }
    const float hi = std::max(a, b);
    const float lo = std::min(a, b);
    return hi + std::log1p(std::exp(lo - hi));
}

}  // namespace

void group_tokens_into_words(const std::vector<int> &         ids,
                             const std::vector<std::string> & pieces,
                             std::vector<std::vector<int>> &  out_word_token_idx,
                             std::vector<std::string> &       out_word_texts) {
    out_word_token_idx.clear();
    out_word_texts.clear();

    std::vector<int> cur_idx;
    std::string      cur_text;

    auto flush = [&]() {
        if (cur_idx.empty() || trim(cur_text).empty()) {
            cur_idx.clear();
            cur_text.clear();
            return;
        }
        out_word_token_idx.push_back(cur_idx);
        out_word_texts.push_back(trim(cur_text));
        cur_idx.clear();
        cur_text.clear();
    };

    const size_t n = std::min(ids.size(), pieces.size());
    for (size_t i = 0; i < n; ++i) {
        const int           id = ids[i];
        const std::string & p  = pieces[i];

        if (is_special_piece(p)) {
            flush();
            continue;
        }
        if (is_space_piece(id, p)) {
            flush();
            continue;
        }
        // A leading space starts a new word (byte-level BPE convention).
        const bool starts_new = !p.empty() && p[0] == ' ';
        if (starts_new && !cur_text.empty()) {
            flush();
        }
        cur_idx.push_back(static_cast<int>(i));
        cur_text += p;
    }
    flush();
}

std::vector<CwWord> extract_word_timings(const std::vector<int> &         ids,
                                         const std::vector<std::string> & pieces,
                                         const std::vector<float> &       attn,
                                         int                              n_rows,
                                         int                              n_frames,
                                         const std::vector<float> &       mel,
                                         int                              n_mels,
                                         int                              n_mel_frames,
                                         const WordTimingParams &         params) {
    std::vector<CwWord> out;
    if (ids.empty() || n_rows <= 0 || n_frames <= 0) {
        return out;
    }

    std::vector<std::vector<int>> word_token_idx;
    std::vector<std::string>      word_texts;
    group_tokens_into_words(ids, pieces, word_token_idx, word_texts);
    if (word_token_idx.empty()) {
        return out;
    }

    const int F = n_frames;
    const int W = static_cast<int>(word_token_idx.size());

    // ---- 2. token log-probs over frames (sharpen, row-normalize, log) ----
    // Kept in double like the reference (attn.astype(np.float64)); the
    // normalization denominator is a 1500-term sum and float32 loses the tail.
    std::vector<double> tok_logp(static_cast<size_t>(n_rows) * static_cast<size_t>(F), 0.0);
    for (int t = 0; t < n_rows; ++t) {
        double sum = 0.0;
        for (int f = 0; f < F; ++f) {
            double a = static_cast<double>(attn[static_cast<size_t>(t) * F + f]);
            if (a < 0.0) {
                a = 0.0;
            }
            if (params.sharpen != 1.0f) {
                a = std::pow(a, static_cast<double>(params.sharpen));
            }
            tok_logp[static_cast<size_t>(t) * F + f] = a;
            sum += a;
        }
        sum = std::max(sum, static_cast<double>(k_attn_eps));
        for (int f = 0; f < F; ++f) {
            const double p              = tok_logp[static_cast<size_t>(t) * F + f] / sum;
            tok_logp[static_cast<size_t>(t) * F + f] = std::log(p + k_attn_eps);
        }
    }

    // ---- 3. blank log-prob from mel-band energy ----
    // Full encoder window, no clipping to audio duration: clip_to_audio
    // defaults False upstream because recomputing the energy percentiles over
    // the speech region only makes quiet intra-word frames read as blank and
    // the Viterbi collapses words onto their attention peak.
    std::vector<float> blank_logp(static_cast<size_t>(F), 0.0f);
    {
        std::vector<float> energy(static_cast<size_t>(n_mel_frames), 0.0f);
        for (int f = 0; f < n_mel_frames; ++f) {
            double acc = 0.0;
            for (int m = 0; m < n_mels; ++m) {
                acc += static_cast<double>(mel[static_cast<size_t>(m) * n_mel_frames + f]);
            }
            energy[static_cast<size_t>(f)] = static_cast<float>(acc / std::max(n_mels, 1));
        }
        const float p10   = percentile_linear(energy, 10.0f);
        const float p90   = percentile_linear(energy, 90.0f);
        const float denom = std::max(1e-6f, p90 - p10);
        for (float & e : energy) {
            e = std::clamp((e - p10) / denom, 0.0f, 1.0f);
        }
        std::vector<float> energy_norm = resample_1d(energy, F);
        for (int f = 0; f < F; ++f) {
            float b = std::clamp(1.0f - energy_norm[static_cast<size_t>(f)], 1e-4f, 1.0f - 1e-4f);
            if (params.blank_gamma != 1.0f) {
                b = std::clamp(std::pow(b, params.blank_gamma), 1e-4f, 1.0f);
            }
            blank_logp[static_cast<size_t>(f)] = std::log(b + k_blank_eps) - params.blank_penalty;
        }
    }

    // ---- 4. collapse each word's token rows, then Viterbi with blanks ----
    std::vector<double> word_logp(static_cast<size_t>(W) * static_cast<size_t>(F), k_neg_inf);
    for (int w = 0; w < W; ++w) {
        bool any = false;
        for (const int ti : word_token_idx[static_cast<size_t>(w)]) {
            if (ti < 0 || ti >= n_rows) {
                continue;
            }
            for (int f = 0; f < F; ++f) {
                double & dst = word_logp[static_cast<size_t>(w) * F + f];
                const double src = tok_logp[static_cast<size_t>(ti) * F + f];
                dst = any ? static_cast<double>(logaddexp(static_cast<float>(dst), static_cast<float>(src))) : src;
            }
            any = true;
        }
    }

    const int S = 2 * W + 1;
    // emit[s][f]: even states are virtual blanks (all sharing blank_logp),
    // odd states emit their word's collapsed row.
    auto emit = [&](int s, int f) -> double {
        return (s % 2 == 0) ? static_cast<double>(blank_logp[static_cast<size_t>(f)])
                            : word_logp[static_cast<size_t>((s - 1) / 2) * F + f];
    };

    std::vector<double>  dp(static_cast<size_t>(S) * static_cast<size_t>(F), k_neg_inf);
    std::vector<uint8_t> back(static_cast<size_t>(S) * static_cast<size_t>(F), 0);
    dp[0] = emit(0, 0);

    for (int f = 1; f < F; ++f) {
        for (int s = 0; s < S; ++s) {
            const double stay = dp[static_cast<size_t>(s) * F + (f - 1)];
            const double adv  = (s >= 1) ? dp[static_cast<size_t>(s - 1) * F + (f - 1)] : k_neg_inf;
            const bool   take = adv > stay;
            dp[static_cast<size_t>(s) * F + f]   = (take ? adv : stay) + emit(s, f);
            back[static_cast<size_t>(s) * F + f] = take ? 1 : 0;
        }
    }

    // Tie-break toward later states, matching the reference's
    // argmax(dp[:, F-1] + arange(S) * 1e-4).
    int    end_state = 0;
    double best      = -std::numeric_limits<double>::infinity();
    for (int s = 0; s < S; ++s) {
        const double v = dp[static_cast<size_t>(s) * F + (F - 1)] + static_cast<double>(s) * 1e-4;
        if (v > best) {
            best      = v;
            end_state = s;
        }
    }

    std::vector<int> states(static_cast<size_t>(F), 0);
    {
        int s = end_state;
        for (int f = F - 1; f >= 0; --f) {
            states[static_cast<size_t>(f)] = s;
            if (f == 0) {
                break;
            }
            if (back[static_cast<size_t>(s) * F + f] == 1) {
                s -= 1;
            }
        }
    }

    out.reserve(static_cast<size_t>(W));
    for (int w = 0; w < W; ++w) {
        const int tok_state = 2 * w + 1;
        int       first = -1, last = -1;
        for (int f = 0; f < F; ++f) {
            if (states[static_cast<size_t>(f)] == tok_state) {
                if (first < 0) {
                    first = f;
                }
                last = f;
            }
        }
        CwWord cw;
        cw.text = word_texts[static_cast<size_t>(w)];
        if (first >= 0) {
            cw.start = static_cast<float>(first) * k_frame_duration_s;
            cw.end   = static_cast<float>(last) * k_frame_duration_s;
        }
        out.push_back(std::move(cw));
    }

    // ---- 5. split short inter-word gaps at their midpoint ----
    if (params.split_gap_max_s > 0.0f) {
        std::vector<CwWord *> placed;
        for (CwWord & w : out) {
            if (w.placed()) {
                placed.push_back(&w);
            }
        }
        for (size_t i = 0; i + 1 < placed.size(); ++i) {
            CwWord & a   = *placed[i];
            CwWord & b   = *placed[i + 1];
            const float gap = b.start - a.end;
            if (gap > 0.0f && gap <= params.split_gap_max_s) {
                const float mid = a.end + gap / 2.0f;
                a.end           = mid;
                b.start         = mid;
            }
        }
    }

    return out;
}

}  // namespace transcribe::crisperwhisper
