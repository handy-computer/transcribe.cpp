#ifndef TRANSCRIBE_WHISPER_ALIGNMENT_H
#define TRANSCRIBE_WHISPER_ALIGNMENT_H

#include <string_view>
#include <vector>

namespace transcribe::whisper {

struct alignment_head {
    int layer;
    int head;
};

// Empty for variants without a known preset or with mismatched decoder geometry.
std::vector<alignment_head> alignment_heads_for_variant(std::string_view variant, int n_layers, int n_heads);

// Attention is contiguous [head][decoder token][encoder frame], with each row
// softmaxed over valid encoder frames. All decoder rows contribute to normalization.
// DTW uses the given subrange, normally no_timestamps followed by the text tokens.
// On success, token_frames holds the first frame reached in each DTW row. Text
// token i spans [token_frames[i], token_frames[i + 1]). Invalid input returns false
// and clears token_frames.
bool align_attention(const std::vector<float> & attention,
                     int                        n_heads,
                     int                        n_tokens,
                     int                        n_frames,
                     int                        first_alignment_row,
                     int                        n_alignment_rows,
                     std::vector<int> &         token_frames);

}  // namespace transcribe::whisper

#endif
