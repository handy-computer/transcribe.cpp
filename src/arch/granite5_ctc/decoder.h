// arch/granite5_ctc/decoder.h - CTC greedy decode and result assembly.
//
// There is no neural decoder in this family. `GraniteSpeech5ForCTC` is
// one tied linear over the encoder output; "decoding" is argmax per
// frame followed by the standard CTC collapse.
//
// Reference for the collapse is NOT the modeling file — `generate()`
// only does `logits.argmax(-1)` and leaves the frame sequence intact.
// The collapse lives in the tokenizer the checkpoint declares,
// `ParakeetTokenizer._decode` (tokenization_parakeet.py):
//
//     token_ids = [g[0] for g in itertools.groupby(token_ids)]
//     token_ids = [t for t in token_ids if t != self.pad_token_id]
//
// i.e. run-length collapse FIRST, then drop blanks, so a blank between
// two identical labels correctly preserves both. `tokenizer_class` in
// tokenizer_config.json is load-bearing, not cosmetic: decoding with a
// plain byte-level BPE tokenizer emits every repeated frame.

#pragma once

#include "transcribe.h"
#include "weights.h"

#include <cstdint>
#include <vector>

namespace transcribe {
class Tokenizer;
}

struct transcribe_session;

namespace transcribe::granite5_ctc {

// One surviving CTC emission.
struct CtcToken {
    int   id    = 0;
    float p     = 0.0f;  // softmax probability of the winning class
    int   frame = 0;     // encoder frame the label first appeared on
};

// Greedy argmax + CTC collapse over [vocab, T] logits in row-major
// [T, vocab] host order (i.e. ggml ne = [vocab, T] read contiguously).
void ctc_greedy_collapse(const float * logits, int t_len, int vocab, int blank_id, std::vector<CtcToken> & out_tokens);

// Milliseconds per encoder frame. Every subsampling block halves the
// rate on top of the frontend's hop and frame stacking:
//     hop * stack * 2^len(subsample_layers) / sample_rate
// which is 160 * 2 * 4 / 16000 = 80 ms on this variant.
double ms_per_encoder_frame(const Granite5CtcHParams & hp);

// Fill the session's token / word / segment / text fields from the
// collapsed tokens. `clip_ms` clamps the trailing timestamp: the
// frontend right-pads the waveform to fill the last stacking group, so
// the final frame can extend past the real audio.
void build_result(transcribe_session &          cc,
                  const transcribe::Tokenizer & tok,
                  const Granite5CtcHParams &    hp,
                  const std::vector<CtcToken> & tokens,
                  int64_t                       clip_ms);

}  // namespace transcribe::granite5_ctc
