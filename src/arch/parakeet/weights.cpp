// arch/parakeet/weights.cpp - implementation of read_parakeet_hparams
// and build_parakeet_weights.
//
// Pattern follows llama.cpp's per-arch load_tensors function:
//   - read every required hparam from KV explicitly
//   - then build the tensor catalog as a sequence of get_tensor()
//     calls with explicit expected shapes
//   - missing tensor or shape mismatch -> TRANSCRIBE_ERR_GGUF with a
//     log naming the offending tensor
//
// The shape contract documented in weights.h is enforced here. The
// converter has to write tensors in these shapes; the synthetic
// fixture in tests/fixtures/make_gguf_fixtures.py emits them in
// these shapes; phase 4's encoder builder will read them in these
// shapes.

#include "weights.h"

#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace transcribe::parakeet {

// ---------------------------------------------------------------------------
// Hparams
// ---------------------------------------------------------------------------

namespace {

// Read a uint32 KV into an int32 hparam slot. Required: Absent and
// BadType both surface as TRANSCRIBE_ERR_GGUF. The signed/unsigned
// transition is fine because every Parakeet hparam fits comfortably
// in int32.
transcribe_status read_required_u32(const gguf_context * gguf,
                                    const char *         key,
                                    int32_t &            out)
{
    uint32_t v = 0;
    switch (read_uint32_kv(gguf, key, v)) {
        case KvResult::Ok:
            out = static_cast<int32_t>(v);
            return TRANSCRIBE_OK;
        case KvResult::Absent:
        case KvResult::BadType:
            std::fprintf(stderr,
                         "parakeet: required KV \"%s\" missing or wrong type\n",
                         key);
            return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_ERR_GGUF; // unreachable
}

transcribe_status read_required_f32(const gguf_context * gguf,
                                    const char *         key,
                                    float &              out)
{
    float v = 0.0f;
    switch (read_float32_kv(gguf, key, v)) {
        case KvResult::Ok:
            out = v;
            return TRANSCRIBE_OK;
        case KvResult::Absent:
        case KvResult::BadType:
            std::fprintf(stderr,
                         "parakeet: required KV \"%s\" missing or wrong type\n",
                         key);
            return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_ERR_GGUF; // unreachable
}

transcribe_status read_required_string(const gguf_context * gguf,
                                       const char *         key,
                                       std::string &        out)
{
    std::string v;
    switch (read_string_kv(gguf, key, v)) {
        case KvResult::Ok:
            out = std::move(v);
            return TRANSCRIBE_OK;
        case KvResult::Absent:
        case KvResult::BadType:
            std::fprintf(stderr,
                         "parakeet: required KV \"%s\" missing or wrong type\n",
                         key);
            return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_ERR_GGUF; // unreachable
}

// Optional bool with a default. Absent leaves out at the supplied
// default; BadType is fatal because we control the converter.
transcribe_status read_optional_bool(const gguf_context * gguf,
                                     const char *         key,
                                     bool                 default_value,
                                     bool &               out)
{
    bool tmp = default_value;
    switch (read_bool_kv(gguf, key, tmp)) {
        case KvResult::Absent:
            out = default_value;
            return TRANSCRIBE_OK;
        case KvResult::Ok:
            out = tmp;
            return TRANSCRIBE_OK;
        case KvResult::BadType:
            std::fprintf(stderr,
                         "parakeet: optional KV \"%s\" has wrong type\n", key);
            return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_ERR_GGUF; // unreachable
}

} // namespace

transcribe_status read_parakeet_hparams(const gguf_context * gguf,
                                        ParakeetHParams &    hp)
{
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Encoder.
    if (auto st = read_required_u32(gguf, "stt.parakeet.encoder.n_layers",          hp.enc_n_layers);          st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.parakeet.encoder.d_model",           hp.enc_d_model);           st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.parakeet.encoder.n_heads",           hp.enc_n_heads);           st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.parakeet.encoder.d_ff",              hp.enc_d_ff);              st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.parakeet.encoder.conv_kernel",       hp.enc_conv_kernel);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.parakeet.encoder.subsampling_factor",hp.enc_subsampling_factor);st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.parakeet.encoder.subsampling_channels", hp.enc_subsampling_channels); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.parakeet.encoder.pos_emb_max_len",   hp.enc_pos_emb_max_len);   st != TRANSCRIBE_OK) return st;

    // Optional. NeMo Parakeet has use_bias=false on every published
    // variant we know about, so the default matches reality.
    if (auto st = read_optional_bool(gguf, "stt.parakeet.encoder.use_bias", false, hp.enc_use_bias); st != TRANSCRIBE_OK) return st;

    // Predictor.
    if (auto st = read_required_u32(gguf, "stt.parakeet.predictor.hidden",   hp.pred_hidden);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.parakeet.predictor.n_layers", hp.pred_n_layers); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.parakeet.predictor.vocab",    hp.pred_vocab);    st != TRANSCRIBE_OK) return st;

    // Joint.
    if (auto st = read_required_u32(gguf, "stt.parakeet.joint.hidden",            hp.joint_hidden);            st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32(gguf, "stt.parakeet.joint.num_extra_outputs", hp.joint_num_extra_outputs); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string(gguf, "stt.parakeet.joint.activation",     hp.joint_activation);        st != TRANSCRIBE_OK) return st;

    // TDT decoding parameters. `durations` is required (the decoder
    // can't run without it); `max_symbols` is optional with a default
    // matching every published v2/v3 config we've seen.
    switch (read_int32_array_kv(gguf, "stt.parakeet.tdt.durations", hp.tdt_durations)) {
        case KvResult::Ok:
            break;
        case KvResult::Absent:
        case KvResult::BadType:
            std::fprintf(stderr,
                         "parakeet: required KV \"stt.parakeet.tdt.durations\" "
                         "missing or wrong type\n");
            return TRANSCRIBE_ERR_GGUF;
    }
    {
        uint32_t max_sym = 10;
        switch (read_uint32_kv(gguf, "stt.parakeet.tdt.max_symbols", max_sym)) {
            case KvResult::Ok:
            case KvResult::Absent:
                hp.tdt_max_symbols = static_cast<int32_t>(max_sym);
                break;
            case KvResult::BadType:
                std::fprintf(stderr,
                             "parakeet: optional KV \"stt.parakeet.tdt.max_symbols\" "
                             "has wrong type\n");
                return TRANSCRIBE_ERR_GGUF;
        }
    }

    // Frontend. The full stt.frontend.* block PLAN.md declares as the
    // complete contract between converter and loader. Reading every
    // field at load time, even though phase 3's C++ frontend hasn't
    // landed yet, makes the loader the gate that catches a future
    // converter that drops a field. Required keys: type, num_mels,
    // sample_rate, n_fft, win_length, hop_length, window, normalize,
    // dither, pre_emphasis, f_min, f_max. CMVN/LFR fields are
    // intentionally not read here — Parakeet doesn't use them and
    // the converter doesn't emit them.
    if (auto st = read_required_string(gguf, "stt.frontend.type",         hp.fe_type);         st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32   (gguf, "stt.frontend.num_mels",     hp.fe_num_mels);     st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32   (gguf, "stt.frontend.sample_rate",  hp.fe_sample_rate);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32   (gguf, "stt.frontend.n_fft",        hp.fe_n_fft);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32   (gguf, "stt.frontend.win_length",   hp.fe_win_length);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32   (gguf, "stt.frontend.hop_length",   hp.fe_hop_length);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string(gguf, "stt.frontend.window",       hp.fe_window);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string(gguf, "stt.frontend.normalize",    hp.fe_normalize);    st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32   (gguf, "stt.frontend.dither",       hp.fe_dither);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32   (gguf, "stt.frontend.pre_emphasis", hp.fe_pre_emphasis); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32   (gguf, "stt.frontend.f_min",        hp.fe_f_min);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32   (gguf, "stt.frontend.f_max",        hp.fe_f_max);        st != TRANSCRIBE_OK) return st;

    // Cross-field invariants. These have to hold or the model is
    // unbuildable; we catch them here rather than letting them
    // surface as confusing shape mismatches downstream.
    if (hp.enc_n_layers <= 0 || hp.enc_d_model <= 0 || hp.enc_n_heads <= 0 ||
        hp.enc_d_ff <= 0 || hp.enc_conv_kernel <= 0 ||
        hp.enc_subsampling_factor <= 0 || hp.enc_subsampling_channels <= 0)
    {
        std::fprintf(stderr, "parakeet: encoder hparams must be positive\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_d_model % hp.enc_n_heads != 0) {
        std::fprintf(stderr,
                     "parakeet: encoder d_model (%d) not divisible by n_heads (%d)\n",
                     hp.enc_d_model, hp.enc_n_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.pred_hidden <= 0 || hp.pred_n_layers <= 0 || hp.pred_vocab <= 1) {
        std::fprintf(stderr, "parakeet: predictor hparams must be positive\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.joint_hidden <= 0 || hp.joint_num_extra_outputs < 0) {
        std::fprintf(stderr, "parakeet: joint hparams invalid\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    // Joint activation allow-list. The C++ joint forward implements
    // exactly the same three the parakeet-mlx rnnt.py JointNetwork
    // accepts; anything else is a converter mistake or a future model
    // we haven't ported yet, and producing wrong logits silently is
    // worse than failing loudly here.
    if (hp.joint_activation != "relu" &&
        hp.joint_activation != "sigmoid" &&
        hp.joint_activation != "tanh")
    {
        std::fprintf(stderr,
                     "parakeet: unsupported joint activation \"%s\" "
                     "(only relu, sigmoid, tanh are implemented)\n",
                     hp.joint_activation.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    // TDT durations: must be non-empty, must match num_extra_outputs
    // (each duration logit corresponds to one durations[i] choice),
    // and every entry must be non-negative (negative jumps would
    // walk backwards in time and break the decode loop's invariant).
    // Zero is allowed and is the standard "stay on this frame, just
    // emit a token without advancing" duration.
    if (hp.tdt_durations.empty()) {
        std::fprintf(stderr,
                     "parakeet: stt.parakeet.tdt.durations must be non-empty\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (static_cast<int32_t>(hp.tdt_durations.size()) != hp.joint_num_extra_outputs) {
        std::fprintf(stderr,
                     "parakeet: stt.parakeet.tdt.durations length (%zu) "
                     "must equal joint.num_extra_outputs (%d)\n",
                     hp.tdt_durations.size(), hp.joint_num_extra_outputs);
        return TRANSCRIBE_ERR_GGUF;
    }
    for (int32_t d : hp.tdt_durations) {
        if (d < 0) {
            std::fprintf(stderr,
                         "parakeet: stt.parakeet.tdt.durations contains "
                         "negative value %d\n", d);
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    if (hp.tdt_max_symbols < 0) {
        std::fprintf(stderr,
                     "parakeet: stt.parakeet.tdt.max_symbols must be >= 0 "
                     "(got %d)\n", hp.tdt_max_symbols);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_num_mels <= 0 || hp.fe_sample_rate <= 0 ||
        hp.fe_n_fft <= 0 || hp.fe_win_length <= 0 || hp.fe_hop_length <= 0)
    {
        std::fprintf(stderr, "parakeet: frontend dimensions must be positive\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_win_length > hp.fe_n_fft) {
        std::fprintf(stderr,
                     "parakeet: frontend win_length (%d) > n_fft (%d)\n",
                     hp.fe_win_length, hp.fe_n_fft);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_f_min < 0.0f || hp.fe_f_max <= hp.fe_f_min) {
        std::fprintf(stderr,
                     "parakeet: frontend mel band invalid: f_min=%f f_max=%f\n",
                     hp.fe_f_min, hp.fe_f_max);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_dither < 0.0f) {
        std::fprintf(stderr,
                     "parakeet: frontend dither must be >= 0 (got %f)\n",
                     hp.fe_dither);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.fe_type != "mel") {
        // Recognized-but-unsupported frontend type. Surfaces as
        // ERR_GGUF for now; if we add a non-mel STT family later,
        // this turns into a NOT_IMPLEMENTED branch.
        std::fprintf(stderr,
                     "parakeet: unsupported frontend type \"%s\" (only \"mel\")\n",
                     hp.fe_type.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }

    // Frontend value-domain checks. The loader reads several string
    // and integer fields that the C++ MelFrontend implementation
    // (src/transcribe-mel.cpp) does not actually parameterize on. If
    // a GGUF carries a value the implementation cannot honor, fail
    // here loud and early — letting it through would silently
    // substitute the hard-coded behavior at runtime and produce
    // wrong features without any error.
    //
    // When the C++ frontend grows support for more variants (e.g.
    // hamming/blackman windows or all_features normalize), the
    // corresponding check below should be loosened.

    // Window: only symmetric Hann is implemented
    // (build_hann_window_symmetric_padded in transcribe-mel.cpp).
    if (hp.fe_window != "hann") {
        std::fprintf(stderr,
                     "parakeet: unsupported frontend window \"%s\" "
                     "(only \"hann\" is implemented)\n",
                     hp.fe_window.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }

    // Normalize: only NeMo's "per_feature" (unbiased per-mel-bin
    // mean/std normalization) is implemented. NeMo also supports
    // "all_features" and CMVN ("fixed_mean"/"fixed_std") but neither
    // applies to Parakeet, and silently per-feature-normalizing a
    // GGUF that asked for something else would be a confusing bug.
    if (hp.fe_normalize != "per_feature") {
        std::fprintf(stderr,
                     "parakeet: unsupported frontend normalize \"%s\" "
                     "(only \"per_feature\" is implemented)\n",
                     hp.fe_normalize.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }

    // n_fft must be a power of two: the FFT in transcribe-mel.cpp is
    // a radix-2 Cooley-Tukey, and its bit-reversal loop produces
    // garbage for non-power-of-two sizes rather than failing
    // cleanly. Catch it here so a future converter that emits e.g.
    // n_fft=400 fails at load instead of producing nonsense mels.
    // (fe_n_fft > 0 was already enforced above.)
    if ((hp.fe_n_fft & (hp.fe_n_fft - 1)) != 0) {
        std::fprintf(stderr,
                     "parakeet: frontend n_fft (%d) must be a power of 2 "
                     "(radix-2 FFT requirement)\n",
                     hp.fe_n_fft);
        return TRANSCRIBE_ERR_GGUF;
    }

    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

namespace {

// Format a per-layer tensor name into a small stack buffer. The
// caller passes a fmt that contains exactly one %d; the layer index
// is substituted. Returns a pointer to a thread-local static buffer;
// the next call within the same thread invalidates the previous
// pointer.
const char * lname(const char * fmt, int layer_idx) {
    thread_local char buf[128];
    std::snprintf(buf, sizeof(buf), fmt, layer_idx);
    return buf;
}

// The canonical find_tensor() helper lives in
// src/transcribe-weights-util.{h,cpp}; see that header for rationale.
// It is shared between every per-family weights.cpp. The GET_*
// macros below still live here because their type allowlists encode
// a per-family quantization policy, not a shared convention.
constexpr const char * kTag = "parakeet";

// Sugar for the GET_OR_FAIL pattern. Variadic so the caller can pass
// the expected dims as a comma-separated list rather than wrapping
// them in extra braces (which would otherwise turn into a GNU
// statement expression when piped through a function-like macro).
//
// Three macros, one per tensor bucket. Each bucket has a fixed
// allowlist of acceptable ggml types; the bucket choice at the call
// site is part of the catalog and never changes per-converter:
//
//   GET_F32  - norms, biases, BatchNorm running stats, attn pos_bias.
//              These are precision-sensitive and tiny, so they stay
//              fp32 across every quant preset the converter ships.
//
//   GET_CONV - conv kernels (pre_encode + per-block conv module).
//              The local f32-friendly conv wrappers in encoder.cpp
//              im2col against the kernel's real type, which means
//              quantized kernels would require a quantized im2col
//              path that ggml does not provide. Allow F32+F16 only.
//              Cost is small (~5 MB across all conv kernels) so this
//              is a deliberate quality choice, not a workaround.
//
//   GET_LIN  - linear weights consumed by ggml_mul_mat (encoder FF +
//              attention projections, predictor LSTM gate matrices,
//              joint enc/pred/out projections, predictor embed table).
//              ggml_mul_mat handles a quantized W against an fp32 X
//              natively, so this list grows as quant types come
//              online. Today: F32 + F16. As Q8_0 / Q5_K / Q4_K land
//              they get appended to this allowlist.
#define GET_F32(slot, name, ...) \
    do { \
        ggml_tensor * _t = transcribe::weights::find_tensor( \
            gguf, ctx_meta, (name), \
            {GGML_TYPE_F32}, {__VA_ARGS__}, kTag); \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF; \
        (slot) = _t; \
    } while (0)

#define GET_CONV(slot, name, ...) \
    do { \
        ggml_tensor * _t = transcribe::weights::find_tensor( \
            gguf, ctx_meta, (name), \
            {GGML_TYPE_F32, GGML_TYPE_F16}, \
            {__VA_ARGS__}, kTag); \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF; \
        (slot) = _t; \
    } while (0)

#define GET_LIN(slot, name, ...) \
    do { \
        ggml_tensor * _t = transcribe::weights::find_tensor( \
            gguf, ctx_meta, (name), \
            {GGML_TYPE_F32, GGML_TYPE_F16, \
             GGML_TYPE_Q8_0, \
             GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, \
             GGML_TYPE_Q6_K}, \
            {__VA_ARGS__}, kTag); \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF; \
        (slot) = _t; \
    } while (0)

} // namespace

transcribe_status build_parakeet_weights(const gguf_context *    gguf,
                                         ggml_context *          ctx_meta,
                                         const ParakeetHParams & hp,
                                         ParakeetWeights &       weights)
{
    if (gguf == nullptr || ctx_meta == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int64_t channels = hp.enc_subsampling_channels;
    const int64_t d_model  = hp.enc_d_model;
    const int64_t d_ff     = hp.enc_d_ff;
    const int64_t n_heads  = hp.enc_n_heads;
    const int64_t head_dim = hp.enc_head_dim();
    const int64_t k        = hp.enc_conv_kernel;
    const int64_t pred_h   = hp.pred_hidden;
    const int64_t pred_v   = hp.pred_vocab;
    const int64_t joint_h  = hp.joint_hidden;
    const int64_t joint_n  = hp.joint_n_classes();

    // The flattened "freq" axis going into pre_encode.out is
    // (subsampling_channels * (num_mels / subsampling_factor)). NeMo's
    // dw_striding subsamples on the spatial axis, ending up with that
    // many features per time step.
    const int64_t pre_encode_freq = channels * (hp.fe_num_mels / hp.enc_subsampling_factor);
    const int64_t pre_encode_in   = pre_encode_freq;

    // ----- pre_encode -----
    // Conv shapes documented in weights.h. ggml_tensor::ne is fast-to-
    // slow dim order; for an OIHW conv kernel that's [kw, kh, in, out].
    // 1×1 pointwise convs collapse to [1, 1, in, out].
    GET_CONV(weights.pre_encode.conv0_w, "enc.pre_encode.conv.0.weight", 3, 3, 1,        channels);
    GET_F32 (weights.pre_encode.conv0_b, "enc.pre_encode.conv.0.bias", channels);
    GET_CONV(weights.pre_encode.conv2_w, "enc.pre_encode.conv.2.weight", 3, 3, 1,        channels);
    GET_F32 (weights.pre_encode.conv2_b, "enc.pre_encode.conv.2.bias", channels);
    GET_CONV(weights.pre_encode.conv3_w, "enc.pre_encode.conv.3.weight", 1, 1, channels, channels);
    GET_F32 (weights.pre_encode.conv3_b, "enc.pre_encode.conv.3.bias", channels);
    GET_CONV(weights.pre_encode.conv5_w, "enc.pre_encode.conv.5.weight", 3, 3, 1,        channels);
    GET_F32 (weights.pre_encode.conv5_b, "enc.pre_encode.conv.5.bias", channels);
    GET_CONV(weights.pre_encode.conv6_w, "enc.pre_encode.conv.6.weight", 1, 1, channels, channels);
    GET_F32 (weights.pre_encode.conv6_b, "enc.pre_encode.conv.6.bias", channels);
    GET_LIN (weights.pre_encode.out_w,   "enc.pre_encode.out.weight", pre_encode_in, d_model);
    GET_F32 (weights.pre_encode.out_b,   "enc.pre_encode.out.bias", d_model);

    // ----- encoder blocks -----
    weights.blocks.assign(hp.enc_n_layers, ParakeetBlock{});
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        auto & b = weights.blocks[i];

        // Macaron FF1.
        GET_F32(b.norm_ff1_w, lname("enc.blocks.%d.norm_ff1.weight", i), d_model);
        GET_F32(b.norm_ff1_b, lname("enc.blocks.%d.norm_ff1.bias",   i), d_model);
        GET_LIN(b.ff1_lin1_w, lname("enc.blocks.%d.ff1.linear1.weight", i), d_model, d_ff);
        GET_LIN(b.ff1_lin2_w, lname("enc.blocks.%d.ff1.linear2.weight", i), d_ff,    d_model);

        // Self-attention with relative position. Bias-free linears
        // (use_bias=false on every Parakeet variant we know about);
        // the per-head pos_bias_u/v live alongside.
        GET_F32(b.norm_attn_w, lname("enc.blocks.%d.norm_attn.weight", i), d_model);
        GET_F32(b.norm_attn_b, lname("enc.blocks.%d.norm_attn.bias",   i), d_model);
        GET_LIN(b.attn_q_w,    lname("enc.blocks.%d.attn.linear_q.weight",   i), d_model, d_model);
        GET_LIN(b.attn_k_w,    lname("enc.blocks.%d.attn.linear_k.weight",   i), d_model, d_model);
        GET_LIN(b.attn_v_w,    lname("enc.blocks.%d.attn.linear_v.weight",   i), d_model, d_model);
        GET_LIN(b.attn_out_w,  lname("enc.blocks.%d.attn.linear_out.weight", i), d_model, d_model);
        GET_LIN(b.attn_pos_w,  lname("enc.blocks.%d.attn.linear_pos.weight", i), d_model, d_model);
        // pos_bias_u/v are added directly to a fp32 q tensor via
        // ggml_add — must stay fp32 to avoid a mixed-type broadcast.
        GET_F32(b.attn_pos_u,  lname("enc.blocks.%d.attn.pos_bias_u",        i), head_dim, n_heads);
        GET_F32(b.attn_pos_v,  lname("enc.blocks.%d.attn.pos_bias_v",        i), head_dim, n_heads);

        // Conv module. pointwise1 doubles the channel count for the
        // GLU split; depthwise has groups=d_model so its weight is
        // [kernel, 1, d_model] in ggml ne order. pointwise2 collapses
        // back to d_model.
        GET_F32 (b.norm_conv_w, lname("enc.blocks.%d.norm_conv.weight", i), d_model);
        GET_F32 (b.norm_conv_b, lname("enc.blocks.%d.norm_conv.bias",   i), d_model);
        GET_CONV(b.conv_pw1_w,  lname("enc.blocks.%d.conv.pointwise1.weight", i), 1, d_model, 2 * d_model);
        GET_CONV(b.conv_dw_w,   lname("enc.blocks.%d.conv.depthwise.weight",  i), k, 1,       d_model);
        GET_CONV(b.conv_pw2_w,  lname("enc.blocks.%d.conv.pointwise2.weight", i), 1, d_model, d_model);
        GET_F32 (b.conv_bn_w,   lname("enc.blocks.%d.conv.bn.weight",       i), d_model);
        GET_F32 (b.conv_bn_b,   lname("enc.blocks.%d.conv.bn.bias",         i), d_model);
        GET_F32 (b.conv_bn_rm,  lname("enc.blocks.%d.conv.bn.running_mean", i), d_model);
        GET_F32 (b.conv_bn_rv,  lname("enc.blocks.%d.conv.bn.running_var",  i), d_model);

        // Macaron FF2.
        GET_F32(b.norm_ff2_w, lname("enc.blocks.%d.norm_ff2.weight", i), d_model);
        GET_F32(b.norm_ff2_b, lname("enc.blocks.%d.norm_ff2.bias",   i), d_model);
        GET_LIN(b.ff2_lin1_w, lname("enc.blocks.%d.ff2.linear1.weight", i), d_model, d_ff);
        GET_LIN(b.ff2_lin2_w, lname("enc.blocks.%d.ff2.linear2.weight", i), d_ff,    d_model);

        // Final per-block layer norm.
        GET_F32(b.norm_out_w, lname("enc.blocks.%d.norm_out.weight", i), d_model);
        GET_F32(b.norm_out_b, lname("enc.blocks.%d.norm_out.bias",   i), d_model);
    }

    // ----- predictor -----
    // The predictor + joint run on host CPU via decoder.cpp's
    // hand-rolled fp32 linear(); the host weight mirror dequantizes
    // through ggml_get_type_traits()->to_float on load, so any of the
    // GET_LIN-allowed types are safe here.
    GET_LIN(weights.predictor.embed_w, "pred.embed.weight", pred_h, pred_v);

    weights.predictor.lstm.assign(hp.pred_n_layers, ParakeetPredictor::LstmLayer{});
    for (int i = 0; i < hp.pred_n_layers; ++i) {
        auto & l = weights.predictor.lstm[i];
        // Both Wx and Wh project from pred_hidden (NeMo's prednet uses
        // a learned embedding of size pred_hidden; the embed table
        // shape is [pred_vocab, pred_hidden]). The 4*hidden output
        // dim is the concatenated (i, f, g, o) gates.
        const int64_t gates = 4 * pred_h;
        GET_LIN(l.Wx, lname("pred.lstm.%d.Wx",   i), pred_h, gates);
        GET_LIN(l.Wh, lname("pred.lstm.%d.Wh",   i), pred_h, gates);
        GET_F32(l.b,  lname("pred.lstm.%d.bias", i), gates);
    }

    // ----- joint -----
    GET_LIN(weights.joint.enc_w,  "joint.enc.weight", d_model, joint_h);
    GET_F32(weights.joint.enc_b,  "joint.enc.bias", joint_h);
    GET_LIN(weights.joint.pred_w, "joint.pred.weight", pred_h,  joint_h);
    GET_F32(weights.joint.pred_b, "joint.pred.bias", joint_h);
    GET_LIN(weights.joint.out_w,  "joint.out.weight", joint_h, joint_n);
    GET_F32(weights.joint.out_b,  "joint.out.bias", joint_n);

    return TRANSCRIBE_OK;
}

#undef GET_F32
#undef GET_CONV
#undef GET_LIN

} // namespace transcribe::parakeet
