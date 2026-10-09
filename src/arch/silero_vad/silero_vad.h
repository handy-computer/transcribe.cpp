// arch/silero_vad/silero_vad.h - Silero VAD family (VAD role): hparams,
// weights, model and session types.
//
// INTERNAL to src/arch/silero_vad/.
//
// One frame of frame_samples (512) samples is scored as
//
//   chunk  = [context (64) | frame (512)]           context = previous frame's tail
//   padded = reflect_pad_right(chunk, 64)           640 samples
//   mag    = |basis . windows(padded, 256, hop 128)| [129, 4]
//   enc    = 4 x (conv1d k3 p1 + ReLU), strides 1, 2, 2, 1 -> [128, 1]
//   h, c   = LSTMCell(enc, (h, c))
//   p      = sigmoid(head . relu(h) + b)
//
// Everything up to the LSTM input projection is independent of the
// recurrent state, so it runs as one batched ggml graph over a block of
// frames (encoder.cpp). The recurrence runs on the host (decoder.cpp).

#pragma once

#include "transcribe-backend.h"
#include "transcribe-model.h"
#include "transcribe-vad.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct gguf_context;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace transcribe {
struct Arch;
}

namespace transcribe::silero_vad {

constexpr int kNumEncoder = 4;
constexpr int kKernel     = 3;  // every encoder conv: k3, pad 1, dilation 1

struct HParams {
    int32_t sample_rate     = 16000;
    int32_t n_fft           = 256;
    int32_t hop             = 128;
    int32_t frame_samples   = 512;
    int32_t context_samples = 64;
    int32_t reflect_pad     = 64;
    int32_t channels[kNumEncoder + 1]{};  // [n_bins, c1, c2, c3, c4]
    int32_t strides[kNumEncoder]{};
    int32_t hidden = 128;

    // Derived by read_hparams.
    int32_t n_bins     = 0;            // n_fft / 2 + 1
    int32_t n_bins_pad = 0;            // n_bins rounded up to a multiple of 8 (see stft_basis_pad)
    int32_t chunk      = 0;            // context + frame
    int32_t padded     = 0;            // chunk + reflect_pad
    int32_t t_len[kNumEncoder + 1]{};  // time steps entering each encoder layer; t_len[0] = STFT windows
};

transcribe_status read_hparams(const gguf_context * gguf, HParams & hp);

struct Weights {
    // From the file.
    ggml_tensor * stft_basis = nullptr;   // ne [n_fft, 2 * n_bins]
    ggml_tensor * conv_w[kNumEncoder]{};  // ne [K, IC, OC]
    ggml_tensor * conv_b[kNumEncoder]{};  // ne [OC]
    ggml_tensor * lstm_w_ih = nullptr;    // ne [in, 4H], gates i, f, g, o
    ggml_tensor * lstm_w_hh = nullptr;    // ne [H, 4H]
    ggml_tensor * lstm_b_ih = nullptr;    // ne [4H]
    ggml_tensor * lstm_b_hh = nullptr;    // ne [4H]
    ggml_tensor * head_w    = nullptr;    // ne [H]
    ggml_tensor * head_b    = nullptr;    // ne [1]

    // Derived at load (build_derived_weights). stft_basis_pad is the basis
    // with zero rows appended to each half, ne [n_fft, 2 * n_bins_pad], so
    // every matmul dimension is a multiple of the SIMD width and ggml's CPU
    // backend takes its tinyBLAS path; the padded bins are exact zeros and
    // dense_w[0] ignores them. Each conv over the frame's
    // few time steps is one dense matrix from the time-major input
    // [t_in * IC] to the time-major output [t_out * OC]; taps that land in
    // the zero padding are zero.
    ggml_tensor * stft_basis_pad = nullptr;
    ggml_tensor * dense_w[kNumEncoder]{};  // ne [t_in * IC, t_out * OC]; layer 0 IC = n_bins_pad
    ggml_tensor * dense_b[kNumEncoder]{};  // ne [t_out * OC], conv_b repeated
    ggml_tensor * lstm_b = nullptr;        // ne [4H], b_ih + b_hh
};

transcribe_status bind_weights(ggml_context * ctx_meta, const HParams & hp, Weights & w);

// Host copies for the recurrence: W_hh transposed to [H][4H] (row j holds
// the weights h[j] multiplies), the head weights and bias.
struct HostDecoder {
    std::vector<float> w_hh_t;
    std::vector<float> head_w;
    float              head_b = 0.0f;
};

struct Model final : public transcribe_model {
    HParams hparams;
    Weights weights;

    ggml_context *        ctx_meta       = nullptr;  // owns the file's tensor structs
    ggml_backend_buffer_t backend_buffer = nullptr;
    ggml_context *        ctx_derived    = nullptr;
    ggml_backend_buffer_t derived_buffer = nullptr;
    BackendPlan           plan;

    HostDecoder host;

    Model() = default;
    ~Model() override;
};

transcribe_status build_derived_weights(Model & m);

// Debug dump accumulators (TRANSCRIBE_DUMP_DIR only): every scored frame of
// the current stream, appended block by block.
struct DumpTrace {
    std::vector<float> stft_mag;          // [n, t_len[0], n_bins] (padding stripped)
    std::vector<float> enc[kNumEncoder];  // [n, t_len[i + 1], channels[i + 1]]
    std::vector<float> lstm_h;            // [n, H]
    std::vector<float> lstm_c;            // [n, H]
    std::vector<float> probs;             // [n]
    int64_t            n_frames = 0;

    void clear();
};

// The encoder graph of the current block size; tensors live in the session's
// compute_ctx and are valid while graph_n != 0.
struct GraphTensors {
    ggml_cgraph * graph    = nullptr;
    ggml_tensor * windows  = nullptr;  // input, ne [n_fft, t_len[0] * n]
    ggml_tensor * stft_mag = nullptr;  // ne [n_bins_pad, t_len[0] * n]
    ggml_tensor * enc[kNumEncoder]{};  // ne [t_len[i + 1] * channels[i + 1], n]
    ggml_tensor * gates_in = nullptr;  // output, ne [4H, n]
};

struct Session final : public transcribe_vad_session {
    // Recurrent state of the stream.
    std::vector<float> h;
    std::vector<float> c;
    std::vector<float> context;  // last context_samples of the previous frame

    // Host scratch, reused across calls.
    std::vector<float> chunk_buf;  // [padded]
    std::vector<float> windows;    // [block, t_len[0], n_fft]
    std::vector<float> gates_in;   // [block, 4H]: W_ih x + b
    std::vector<float> gates;      // [4H]

    GraphTensors g;
    int          graph_n = 0;  // frames the built graph takes; 0 = none

    DumpTrace trace;

  protected:
    void on_scratch_released() noexcept override {
        g       = GraphTensors{};
        graph_n = 0;
    }
};

// Encoder over a graph of n_graph frames (encoder.cpp), of which the first
// n_real are real and the rest zero padding. windows is [n_graph, t_len[0],
// n_fft] (see build_windows); gates_in receives [n_graph, 4H]. Uses the
// session's compute scratch, rebuilding the graph when n_graph changes.
transcribe_status encode_block(Session &     s,
                               const Model & m,
                               const float * windows,
                               int           n_graph,
                               int           n_real,
                               float *       gates_in);

// The STFT windows of n consecutive frames, advancing s.context.
void build_windows(Session & s, const HParams & hp, const float * pcm, int n, float * out);

// The recurrence over n frames (decoder.cpp): consumes gates_in [n, 4H],
// advances s.h / s.c, writes n probabilities.
void decode_block(Session & s, const Model & m, const float * gates_in, int n, float * probs);

extern const Arch arch;

}  // namespace transcribe::silero_vad
