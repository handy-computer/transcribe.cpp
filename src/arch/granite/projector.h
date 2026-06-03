// arch/granite/projector.h - Granite Speech projector (BLIP-2 Q-Former).
//
// Reference: GraniteSpeechEncoderProjector +
// Blip2QFormerModel/Encoder/Layer in transformers/models/blip_2/.
//
// Forward shape:
//
//   in     :  [hidden_enc, T_enc]  (encoder.out for 1b/2b; for -plus
//             this is the cat of block[3] and block[N-1] hidden along
//             the channel axis, doubling hidden_enc)
//   pad    :  zero-pad along T_enc → nblocks * window_size where
//             nblocks = ceil(T_enc / window_size).
//   window :  view as [hidden_enc, window_size, nblocks]
//   query  :  [hidden, num_queries=3, 1] broadcast → [hidden, 3, nblocks]
//   for layer in [0, n_layers):
//       q   = LN(self_attn.out(self_attn(q, q, q)) + q)
//       q   = LN(cross_attn.out(cross_attn(q, window, window)) + q)
//       q   = LN(ffn.down(GELU(ffn.up(q))) + q)
//   q       = LN(final_norm)(q)            # proj.qformer.out  [hidden, 3, nblocks]
//   out     = linear(q.reshape(nblocks*3, hidden))   # proj.out  [text_hidden, nblocks*3]
//
// BERT/BLIP-2 layer style: residual + LN happens AFTER each output
// projection ("post-LN" — different from the granite encoder's pre-LN
// macaron / pre-LN attention pattern).
//
// This header is INTERNAL to src/arch/granite/.

#pragma once

#include "weights.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace transcribe::granite {

struct ProjectorBuild {
    // Graph inputs.
    ggml_tensor * enc_in    = nullptr;  // [hidden_enc, T_enc]
    ggml_tensor * enc_pad   = nullptr;  // [hidden_enc, pad_n], or nullptr
                                        // when T_enc is aligned to
                                        // window_size

    // Graph output.
    ggml_tensor * out       = nullptr;  // [text_hidden, n_audio_tokens]

    // Dump points.
    struct {
        ggml_tensor * qformer_out = nullptr;  // [hidden, num_queries, nblocks]
        ggml_tensor * proj_out    = nullptr;  // == out, named "proj.out"
    } dumps;

    ggml_cgraph * graph        = nullptr;

    int nblocks                = 0;
    int n_audio_tokens         = 0;
    int t_enc_pad              = 0;   // pad_n = nblocks * window_size - T_enc
};

ProjectorBuild build_projector_graph(ggml_context *         ctx,
                                     const GraniteWeights & weights,
                                     const GraniteHParams & hp,
                                     int                    T_enc);

} // namespace transcribe::granite
