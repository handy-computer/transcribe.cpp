// arch/silero_vad/weights.cpp - hparams, weight binding, and the load-time
// derived weights (dense per-frame conv matrices, folded LSTM bias, host
// copies for the recurrence).

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "silero_vad.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace transcribe::silero_vad {

namespace {

constexpr const char * kTag = "silero_vad";

// Bounds on metadata-derived sizes; the real model is far inside them.
constexpr int32_t kMaxDim = 4096;

transcribe_status read_i32_array(const gguf_context * gguf, const char * key, int32_t * out, size_t n) {
    std::vector<int32_t> v;
    if (read_int32_array_kv(gguf, key, v) != KvResult::Ok || v.size() != n) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: KV \"%s\" missing or not an int32 array of %zu", kTag, key, n);
        return TRANSCRIBE_ERR_GGUF;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = v[i];
    }
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status read_hparams(const gguf_context * gguf, HParams & hp) {
#define REQ(expr)                             \
    do {                                      \
        const transcribe_status _st = (expr); \
        if (_st != TRANSCRIBE_OK) {           \
            return _st;                       \
        }                                     \
    } while (0)

    REQ(read_required_u32_kv(gguf, "stt.frontend.sample_rate", kTag, hp.sample_rate));
    REQ(read_required_u32_kv(gguf, "stt.frontend.n_fft", kTag, hp.n_fft));
    REQ(read_required_u32_kv(gguf, "stt.frontend.hop_length", kTag, hp.hop));
    REQ(read_required_u32_kv(gguf, "stt.vad.frame_samples", kTag, hp.frame_samples));
    REQ(read_required_u32_kv(gguf, "stt.silero_vad.context_samples", kTag, hp.context_samples));
    REQ(read_required_u32_kv(gguf, "stt.silero_vad.reflect_pad", kTag, hp.reflect_pad));
    REQ(read_required_u32_kv(gguf, "stt.silero_vad.lstm_hidden", kTag, hp.hidden));
    REQ(read_i32_array(gguf, "stt.silero_vad.encoder_channels", hp.channels, kNumEncoder + 1));
    REQ(read_i32_array(gguf, "stt.silero_vad.encoder_strides", hp.strides, kNumEncoder));
#undef REQ

    auto bad = [](const char * what) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: invalid hparams: %s", kTag, what);
        return TRANSCRIBE_ERR_GGUF;
    };
    if (hp.sample_rate != 16000) {
        return bad("stt.frontend.sample_rate must be 16000");
    }
    if (hp.n_fft <= 0 || hp.n_fft % 2 != 0 || hp.n_fft > kMaxDim || hp.hop <= 0 || hp.hop > hp.n_fft) {
        return bad("n_fft must be even and positive, 0 < hop_length <= n_fft");
    }
    if (hp.frame_samples <= 0 || hp.frame_samples > 16000 || hp.context_samples < 0 ||
        hp.context_samples > hp.frame_samples) {
        return bad("0 < frame_samples <= 16000 and 0 <= context_samples <= frame_samples");
    }
    hp.n_bins     = hp.n_fft / 2 + 1;
    hp.n_bins_pad = (hp.n_bins + 7) / 8 * 8;
    hp.chunk      = hp.context_samples + hp.frame_samples;
    // torch "reflect" needs the pad strictly shorter than the signal.
    if (hp.reflect_pad < 0 || hp.reflect_pad >= hp.chunk) {
        return bad("reflect_pad must be in [0, context + frame)");
    }
    hp.padded = hp.chunk + hp.reflect_pad;
    if (hp.padded < hp.n_fft) {
        return bad("padded frame shorter than n_fft");
    }
    // The host recurrence tiles 4H gates by 32 (decoder.cpp).
    if (hp.hidden <= 0 || hp.hidden > kMaxDim || hp.hidden % 8 != 0) {
        return bad("lstm_hidden must be a positive multiple of 8");
    }
    if (hp.channels[0] != hp.n_bins) {
        return bad("encoder_channels[0] must equal n_fft / 2 + 1");
    }
    hp.t_len[0] = (hp.padded - hp.n_fft) / hp.hop + 1;
    for (int i = 0; i < kNumEncoder; ++i) {
        if (hp.channels[i + 1] <= 0 || hp.channels[i + 1] > kMaxDim || hp.strides[i] <= 0) {
            return bad("encoder channels and strides must be positive");
        }
        // conv1d, kernel 3, padding 1.
        hp.t_len[i + 1] = (hp.t_len[i] + 2 - kKernel) / hp.strides[i] + 1;
        if (hp.t_len[i + 1] <= 0) {
            return bad("encoder strides leave no time steps");
        }
    }
    // The decoder reads the encoder at time step 0 only (squeeze(-1)), so the
    // encoder must end on one step.
    if (hp.t_len[kNumEncoder] != 1) {
        return bad("the encoder must end on a single time step");
    }
    return TRANSCRIBE_OK;
}

transcribe_status bind_weights(ggml_context * ctx_meta, const HParams & hp, Weights & w) {
    using transcribe::weights::find_tensor;

#define GET(slot, name, ...)                                                                        \
    do {                                                                                            \
        ggml_tensor * _t = find_tensor(ctx_meta, (name), { GGML_TYPE_F32 }, { __VA_ARGS__ }, kTag); \
        if (_t == nullptr) {                                                                        \
            return TRANSCRIBE_ERR_GGUF;                                                             \
        }                                                                                           \
        (slot) = _t;                                                                                \
    } while (0)

    const int64_t H = hp.hidden;
    GET(w.stft_basis, "frontend.stft_basis", hp.n_fft, 2 * hp.n_bins);
    char name[64];
    for (int i = 0; i < kNumEncoder; ++i) {
        snprintf(name, sizeof(name), "enc.%d.conv.weight", i);
        GET(w.conv_w[i], name, kKernel, hp.channels[i], hp.channels[i + 1]);
        snprintf(name, sizeof(name), "enc.%d.conv.bias", i);
        GET(w.conv_b[i], name, hp.channels[i + 1]);
    }
    GET(w.lstm_w_ih, "lstm.weight_ih", hp.channels[kNumEncoder], 4 * H);
    GET(w.lstm_w_hh, "lstm.weight_hh", H, 4 * H);
    GET(w.lstm_b_ih, "lstm.bias_ih", 4 * H);
    GET(w.lstm_b_hh, "lstm.bias_hh", 4 * H);
    GET(w.head_w, "head.weight", H);
    GET(w.head_b, "head.bias", 1);
#undef GET
    return TRANSCRIBE_OK;
}

namespace {

std::vector<float> read_back(const ggml_tensor * t) {
    std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

}  // namespace

transcribe_status build_derived_weights(Model & m) {
    const HParams & hp = m.hparams;
    Weights &       w  = m.weights;
    const int64_t   H  = hp.hidden;

    ggml_init_params ip{};
    ip.mem_size   = (2 * kNumEncoder + 3) * ggml_tensor_overhead();
    ip.no_alloc   = true;
    m.ctx_derived = ggml_init(ip);
    if (m.ctx_derived == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    w.stft_basis_pad = ggml_new_tensor_2d(m.ctx_derived, GGML_TYPE_F32, hp.n_fft, 2 * hp.n_bins_pad);
    for (int i = 0; i < kNumEncoder; ++i) {
        const int64_t ic  = i == 0 ? hp.n_bins_pad : hp.channels[i];
        const int64_t in  = static_cast<int64_t>(hp.t_len[i]) * ic;
        const int64_t out = static_cast<int64_t>(hp.t_len[i + 1]) * hp.channels[i + 1];
        w.dense_w[i]      = ggml_new_tensor_2d(m.ctx_derived, GGML_TYPE_F32, in, out);
        w.dense_b[i]      = ggml_new_tensor_1d(m.ctx_derived, GGML_TYPE_F32, out);
    }
    w.lstm_b         = ggml_new_tensor_1d(m.ctx_derived, GGML_TYPE_F32, 4 * H);
    m.derived_buffer = ggml_backend_alloc_ctx_tensors(m.ctx_derived, m.plan.primary);
    if (m.derived_buffer == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_buffer_set_usage(m.derived_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Basis rows: [re 0..n_bins) zeros [im 0..n_bins) zeros, each half n_bins_pad.
    {
        const std::vector<float> basis = read_back(w.stft_basis);  // [2 * n_bins][n_fft]
        const size_t             nf    = static_cast<size_t>(hp.n_fft);
        std::vector<float>       pad(nf * 2 * static_cast<size_t>(hp.n_bins_pad), 0.0f);
        for (int half = 0; half < 2; ++half) {
            for (int k = 0; k < hp.n_bins; ++k) {
                const float * src = basis.data() + static_cast<size_t>(half * hp.n_bins + k) * nf;
                float *       dst = pad.data() + static_cast<size_t>(half * hp.n_bins_pad + k) * nf;
                std::memcpy(dst, src, nf * sizeof(float));
            }
        }
        ggml_backend_tensor_set(w.stft_basis_pad, pad.data(), 0, pad.size() * sizeof(float));
    }

    // dense[to * OC + oc][ti * ICP + ic] = conv_w[oc][ic][k] for the tap k with
    // ti = to * stride + k - 1 inside [0, t_in); padding taps contribute 0.
    // ICP is the input row stride: n_bins_pad for layer 0, IC otherwise.
    for (int i = 0; i < kNumEncoder; ++i) {
        const int64_t            IC   = hp.channels[i];
        const int64_t            ICP  = i == 0 ? hp.n_bins_pad : IC;
        const int64_t            OC   = hp.channels[i + 1];
        const int64_t            t_in = hp.t_len[i];
        const int64_t            t_o  = hp.t_len[i + 1];
        const std::vector<float> cw   = read_back(w.conv_w[i]);  // [OC][IC][K]
        const std::vector<float> cb   = read_back(w.conv_b[i]);
        std::vector<float>       dw(static_cast<size_t>(t_in * ICP * t_o * OC), 0.0f);
        std::vector<float>       db(static_cast<size_t>(t_o * OC));
        for (int64_t to = 0; to < t_o; ++to) {
            for (int64_t oc = 0; oc < OC; ++oc) {
                float * row = dw.data() + static_cast<size_t>((to * OC + oc) * t_in * ICP);
                for (int64_t k = 0; k < kKernel; ++k) {
                    const int64_t ti = to * hp.strides[i] + k - 1;
                    if (ti < 0 || ti >= t_in) {
                        continue;
                    }
                    for (int64_t ic = 0; ic < IC; ++ic) {
                        row[ti * ICP + ic] = cw[static_cast<size_t>((oc * IC + ic) * kKernel + k)];
                    }
                }
                db[static_cast<size_t>(to * OC + oc)] = cb[static_cast<size_t>(oc)];
            }
        }
        ggml_backend_tensor_set(w.dense_w[i], dw.data(), 0, dw.size() * sizeof(float));
        ggml_backend_tensor_set(w.dense_b[i], db.data(), 0, db.size() * sizeof(float));
    }

    std::vector<float> b = read_back(w.lstm_b_ih);
    {
        const std::vector<float> b_hh = read_back(w.lstm_b_hh);
        for (size_t j = 0; j < b.size(); ++j) {
            b[j] += b_hh[j];
        }
    }
    ggml_backend_tensor_set(w.lstm_b, b.data(), 0, b.size() * sizeof(float));

    // Host recurrence: W_hh [4H][H] transposed to [H][4H].
    const std::vector<float> w_hh = read_back(w.lstm_w_hh);
    m.host.w_hh_t.assign(static_cast<size_t>(H * 4 * H), 0.0f);
    for (int64_t g = 0; g < 4 * H; ++g) {
        for (int64_t j = 0; j < H; ++j) {
            m.host.w_hh_t[static_cast<size_t>(j * 4 * H + g)] = w_hh[static_cast<size_t>(g * H + j)];
        }
    }
    m.host.head_w = read_back(w.head_w);
    m.host.head_b = read_back(w.head_b)[0];
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::silero_vad
