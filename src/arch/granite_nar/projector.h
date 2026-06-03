// arch/granite_nar/projector.h - NLE EncoderProjectorQFormer.
//
// Reference: EncoderProjectorQFormer in the NLE HF repo's modeling_nle.py.
// Strictly simpler than the AR granite BLIP-2 Q-Former: only cross-
// attention + MLP per layer, no self-attention. Windowed mean-pool of
// the encoder K/V plus a learned `window_positions` bias acts in place
// of positional encodings.
//
// Forward shape (per upstream):
//
//   in [4 * 1024, T_enc]      // 4 captured encoder layers concatenated
//   for j in [0..4):
//     in[j*1024:(j+1)*1024] = per_layer_LN_j(in[j*1024:(j+1)*1024])
//   in = layer_projector(in) + GELU                                  // [2048, T_enc]
//   pad along T_enc → nblocks * 15
//   reshape to [2048, 15, nblocks]
//   kv = mean over the window axis -> [2048, nblocks]
//   kv = kv + window_positions  (broadcast across nblocks)            // [2048, 15, nblocks]
//   query = learned [2048, 3, 1] repeated across nblocks -> [2048, 3, nblocks]
//   for layer in [0, prj_n_layers):
//     # cross-attention sublayer (post-LN, BLIP-2 style):
//     q = norm_attn(query)
//     q' = linear(cross_attn_q, q)
//     k' = linear(cross_attn_k, kv)
//     v' = linear(cross_attn_v, kv)
//     attn = softmax(q' k'^T / sqrt(head_dim)) v'
//     query = query + linear(cross_attn_o, attn)
//     # FFN sublayer:
//     ff_in = norm_ffn(query)
//     query = query + linear(ffn_fc2, SiLU(linear(ffn_fc1, ff_in)))
//   query = out_norm(query)
//   tokens = reshape(query, [2048, 3 * nblocks])
//   out = linear(out_linear, tokens)                                   // [2048, 3 * nblocks]
//
// Note that the upstream layer math is PRE-LN (not post-LN like the AR
// BLIP-2 layer): norm comes BEFORE each sublayer's linear cluster, and
// the residual closes after the second linear.
//
// This header is INTERNAL to src/arch/granite_nar/.

#pragma once

#include "weights.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace transcribe::granite_nar {

struct ProjectorBuild {
    ggml_tensor * enc_in    = nullptr;  // [num_layers * enc_hidden, T_enc]
    ggml_tensor * enc_pad   = nullptr;  // optional [num_layers * enc_hidden, pad_n]

    ggml_tensor * out       = nullptr;  // [llm_dim, n_audio_tokens]

    struct {
        ggml_tensor * qformer_out = nullptr;  // [llm_dim, n_query, nblocks]
        ggml_tensor * proj_out    = nullptr;  // == out
    } dumps;

    ggml_cgraph * graph = nullptr;
    int nblocks         = 0;
    int n_audio_tokens  = 0;
    int t_enc_pad       = 0;
};

ProjectorBuild build_projector_graph(ggml_context *            ctx,
                                     const GraniteNarWeights & weights,
                                     const GraniteNarHParams & hp,
                                     int                       T_enc);

} // namespace transcribe::granite_nar
