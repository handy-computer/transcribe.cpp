// arch/crisperwhisper/word_timing.h - Viterbi word alignment over supervised
// cross-attention. INTERNAL to src/arch/crisperwhisper/.
//
// Port of crisperwhisper/word_timing.py (`viterbi_mel_word`). NOT Whisper's
// median-filtered DTW: words are aligned by a Viterbi pass over a state chain
// of alternating virtual-blank and word states, where the word emission comes
// from the head-averaged cross-attention and the blank emission comes from
// mel-band energy (silent frames are more likely to be blanks).
//
// Every default here is a reference constant, and they are coupled: the
// upstream docstring is explicit that `sharpen` and the blank shaping must
// move together (gamma=3/penalty=3 at a softer sharpen=3 over-extends words
// and hurts clean speech). Do not tune one in isolation.

#pragma once

#include <string>
#include <vector>

namespace transcribe::crisperwhisper {

struct CwWord;

// Reference: FRAME_DURATION_S. One encoder frame after the stride-2 conv.
inline constexpr float k_frame_duration_s = 0.02f;

struct WordTimingParams {
    // extract_word_timings(sharpen=5.0): "empirically best for this verbatim
    // timing model (TIMIT 34.7 vs 39.3 at 3.0)".
    float sharpen = 5.0f;
    // blank_gamma / blank_penalty = 3.0 / 3.0. The raw mel blank rates ordinary
    // low-energy conversational speech as half-silent and the Viterbi collapses
    // words onto their attention peak (Buckeye words ~26% of true length).
    float blank_gamma   = 3.0f;
    float blank_penalty = 3.0f;
    // split_gap_max_s=0.1: the Viterbi parks inter-word silence in a blank
    // state, so tight boundaries end up non-contiguous. Splitting gaps up to
    // 100 ms at their midpoint is a uniform win upstream; genuine pauses are
    // left alone.
    float split_gap_max_s = 0.1f;
};

// Group generated tokens into words, mirroring group_tokens_into_words.
// `pieces[i]` is the decoded text of `ids[i]`. Special pieces and the explicit
// space token (id 220) flush the current word without joining it.
void group_tokens_into_words(const std::vector<int> &         ids,
                             const std::vector<std::string> & pieces,
                             std::vector<std::vector<int>> &  out_word_token_idx,
                             std::vector<std::string> &       out_word_texts);

// Align `word_token_idx` against `attn` ([n_rows, n_frames], post-softmax,
// head-averaged, row k predicting token k) using `mel` ([n_mels, n_mel_frames]
// at the 10 ms hop) for the blank signal.
//
// Returns one entry per word, in order. Words the Viterbi could not place keep
// start/end < 0 (the reference's None placeholders), which the long-form drop
// logic tolerates and the public API filters out.
std::vector<CwWord> extract_word_timings(const std::vector<int> &         ids,
                                         const std::vector<std::string> & pieces,
                                         const std::vector<float> &       attn,
                                         int                              n_rows,
                                         int                              n_frames,
                                         const std::vector<float> &       mel,
                                         int                              n_mels,
                                         int                              n_mel_frames,
                                         const WordTimingParams &         params = {});

}  // namespace transcribe::crisperwhisper
