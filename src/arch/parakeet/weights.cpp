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
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace transcribe::parakeet {

// ---------------------------------------------------------------------------
// Hparams
// ---------------------------------------------------------------------------

namespace {

constexpr const char * kFamilyTag = "parakeet";

} // namespace

transcribe_status read_parakeet_hparams(const gguf_context * gguf,
                                        ParakeetHParams &    hp)
{
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Encoder.
    if (auto st = read_required_u32_kv(gguf, "stt.parakeet.encoder.n_layers", kFamilyTag,          hp.enc_n_layers);          st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.parakeet.encoder.d_model", kFamilyTag,           hp.enc_d_model);           st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.parakeet.encoder.n_heads", kFamilyTag,           hp.enc_n_heads);           st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.parakeet.encoder.d_ff", kFamilyTag,              hp.enc_d_ff);              st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.parakeet.encoder.conv_kernel", kFamilyTag,       hp.enc_conv_kernel);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.parakeet.encoder.subsampling_factor", kFamilyTag,hp.enc_subsampling_factor);st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.parakeet.encoder.subsampling_channels", kFamilyTag, hp.enc_subsampling_channels); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.parakeet.encoder.pos_emb_max_len", kFamilyTag,   hp.enc_pos_emb_max_len);   st != TRANSCRIBE_OK) return st;

    // Optional. NeMo Parakeet v2/v3 ship with use_bias=false; the
    // 1.1B and rnnt/ctc variants ship with use_bias=true (11 biases per
    // block). Default matches v2/v3 so legacy GGUFs without the KV stay
    // on the bias-free path.
    if (auto st = read_optional_bool_kv(gguf, "stt.parakeet.encoder.use_bias", kFamilyTag, false, hp.enc_use_bias); st != TRANSCRIBE_OK) return st;

    // xscaling: NeMo's RelPositionalEncoding multiplies x by
    // sqrt(d_model) before generating the pos_emb when this is true.
    // v2/v3/tdt-* are False; ctc-*/rnnt-*/unified-en are True.
    // Default to false so legacy GGUFs (which don't ship this KV)
    // continue to work without scaling - they're all xscaling=False
    // models in the current set.
    if (auto st = read_optional_bool_kv(gguf, "stt.parakeet.encoder.xscaling", kFamilyTag, false, hp.enc_xscaling); st != TRANSCRIBE_OK) return st;

    // Local attention window. Default -1/-1 = full attention; matches
    // every variant except parakeet-tdt_ctc-1.1b ([128, 128] regular
    // local) and nemotron-speech-streaming-en-0.6b ([70, 13] chunked).
    if (auto st = read_optional_int32_kv(gguf, "stt.parakeet.encoder.att_context_left",  kFamilyTag, -1, hp.enc_att_context_left);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_int32_kv(gguf, "stt.parakeet.encoder.att_context_right", kFamilyTag, -1, hp.enc_att_context_right); st != TRANSCRIBE_OK) return st;

    // Multi-lookahead training menu (cache-aware streaming models). Flat
    // int32 array [L0,R0,L1,R1,...]; index 0 must equal (att_context_left,
    // att_context_right). Optional — absent for offline GGUFs, in which
    // case we synthesize a one-element list from the scalar fields so
    // call sites read uniformly.
    {
        std::vector<int32_t> flat;
        switch (read_int32_array_kv(gguf, "stt.parakeet.encoder.att_context_size_choices", flat)) {
            case KvResult::Ok:
                if (flat.empty() || (flat.size() % 2) != 0) {
                    std::fprintf(stderr,
                                 "parakeet: stt.parakeet.encoder.att_context_size_choices "
                                 "has odd length %zu (expected pairs)\n", flat.size());
                    return TRANSCRIBE_ERR_GGUF;
                }
                hp.enc_att_context_size_choices.clear();
                hp.enc_att_context_size_choices.reserve(flat.size() / 2);
                for (size_t i = 0; i + 1 < flat.size(); i += 2) {
                    hp.enc_att_context_size_choices.emplace_back(flat[i], flat[i + 1]);
                }
                if (hp.enc_att_context_size_choices.front().first  != hp.enc_att_context_left ||
                    hp.enc_att_context_size_choices.front().second != hp.enc_att_context_right)
                {
                    std::fprintf(stderr,
                                 "parakeet: att_context_size_choices[0] = (%d, %d) "
                                 "but att_context_left/right = (%d, %d); "
                                 "GGUF is inconsistent\n",
                                 hp.enc_att_context_size_choices.front().first,
                                 hp.enc_att_context_size_choices.front().second,
                                 hp.enc_att_context_left, hp.enc_att_context_right);
                    return TRANSCRIBE_ERR_GGUF;
                }
                break;
            case KvResult::Absent:
                // Synthesize from the scalar fields. For full-attention
                // GGUFs (-1, -1) this still produces a one-element list;
                // streaming code paths gate on att_context_style anyway.
                hp.enc_att_context_size_choices.clear();
                hp.enc_att_context_size_choices.emplace_back(
                    hp.enc_att_context_left, hp.enc_att_context_right);
                break;
            case KvResult::BadType:
                std::fprintf(stderr,
                             "parakeet: stt.parakeet.encoder.att_context_size_choices "
                             "has wrong type (expected int32 array)\n");
                return TRANSCRIBE_ERR_GGUF;
        }
    }

    // Chunked-limited-with-rc training menu (buffered streaming variants).
    // Three independent int32 arrays carrying the L / C / R choices the
    // model was trained against. Absent for non-buffered-streaming GGUFs;
    // when present they're a structured triple read into the hparams
    // and the runtime picks one entry from each at stream_begin time.
    {
        auto read_menu = [&](const char * key, std::vector<int32_t> & out) -> transcribe_status {
            std::vector<int32_t> raw;
            switch (read_int32_array_kv(gguf, key, raw)) {
                case KvResult::Ok:
                    out = std::move(raw);
                    return TRANSCRIBE_OK;
                case KvResult::Absent:
                    out.clear();
                    return TRANSCRIBE_OK;
                case KvResult::BadType:
                    std::fprintf(stderr,
                                 "parakeet: %s has wrong type "
                                 "(expected int32 array)\n", key);
                    return TRANSCRIBE_ERR_GGUF;
            }
            return TRANSCRIBE_ERR_GGUF;
        };
        if (auto st = read_menu("stt.parakeet.encoder.att_chunk_left_choices",
                                hp.enc_att_chunk_left_choices);
            st != TRANSCRIBE_OK) return st;
        if (auto st = read_menu("stt.parakeet.encoder.att_chunk_chunk_choices",
                                hp.enc_att_chunk_chunk_choices);
            st != TRANSCRIBE_OK) return st;
        if (auto st = read_menu("stt.parakeet.encoder.att_chunk_right_choices",
                                hp.enc_att_chunk_right_choices);
            st != TRANSCRIBE_OK) return st;
    }

    // Attention context style. Optional with default "regular" so legacy
    // GGUFs (every variant before nemotron-speech-streaming-en-0.6b)
    // stay on the existing local/full path.
    {
        std::string style;
        if (auto st = read_optional_string_kv(gguf, "stt.parakeet.encoder.att_context_style",
                                              kFamilyTag, "regular", style);
            st != TRANSCRIBE_OK)
        {
            return st;
        }
        if (style == "regular") {
            hp.enc_att_context_style = ParakeetHParams::AttContextStyle::Regular;
        } else if (style == "chunked_limited") {
            hp.enc_att_context_style = ParakeetHParams::AttContextStyle::ChunkedLimited;
        } else if (style == "chunked_limited_with_rc") {
            hp.enc_att_context_style = ParakeetHParams::AttContextStyle::ChunkedLimitedWithRc;
            if (hp.enc_att_chunk_left_choices.empty()  ||
                hp.enc_att_chunk_chunk_choices.empty() ||
                hp.enc_att_chunk_right_choices.empty())
            {
                std::fprintf(stderr,
                             "parakeet: att_context_style=chunked_limited_with_rc "
                             "requires non-empty att_chunk_{left,chunk,right}_choices "
                             "arrays in the GGUF; got %zu/%zu/%zu entries\n",
                             hp.enc_att_chunk_left_choices.size(),
                             hp.enc_att_chunk_chunk_choices.size(),
                             hp.enc_att_chunk_right_choices.size());
                return TRANSCRIBE_ERR_GGUF;
            }
        } else {
            std::fprintf(stderr,
                         "parakeet: unsupported att_context_style \"%s\" "
                         "(allowed: regular, chunked_limited, chunked_limited_with_rc)\n",
                         style.c_str());
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // Conv module: per-side depthwise padding and norm type. Both are
    // optional with defaults that match the offline-Conformer config —
    // -1 means "centred (k-1)/2"; "batch_norm" means the historic
    // fused BN path with running_mean / running_var.
    if (auto st = read_optional_int32_kv(gguf, "stt.parakeet.encoder.conv_context_left",  kFamilyTag, -1, hp.enc_conv_context_left);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_optional_int32_kv(gguf, "stt.parakeet.encoder.conv_context_right", kFamilyTag, -1, hp.enc_conv_context_right); st != TRANSCRIBE_OK) return st;
    {
        std::string norm_type;
        if (auto st = read_optional_string_kv(gguf, "stt.parakeet.encoder.conv_norm_type",
                                              kFamilyTag, "batch_norm", norm_type);
            st != TRANSCRIBE_OK)
        {
            return st;
        }
        if (norm_type == "batch_norm") {
            hp.enc_conv_norm_type = ParakeetHParams::ConvNormType::BatchNorm;
        } else if (norm_type == "layer_norm") {
            hp.enc_conv_norm_type = ParakeetHParams::ConvNormType::LayerNorm;
        } else {
            std::fprintf(stderr,
                         "parakeet: unsupported conv_norm_type \"%s\" "
                         "(allowed: batch_norm, layer_norm)\n",
                         norm_type.c_str());
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // Cache-aware streaming pre-encode constants. Optional — both KVs
    // are emitted only by streaming-trained variants. Default 0 (means
    // "non-streaming"); the streaming code paths gate on both > 0.
    if (auto st = read_optional_int32_kv(gguf, "stt.parakeet.encoder.streaming.pre_encode_cache_size",
                                         kFamilyTag, 0, hp.enc_stream_pre_encode_cache_size);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (auto st = read_optional_int32_kv(gguf, "stt.parakeet.encoder.streaming.drop_extra_pre_encoded",
                                         kFamilyTag, 0, hp.enc_stream_drop_extra_pre_encoded);
        st != TRANSCRIBE_OK)
    {
        return st;
    }
    if (auto st = read_optional_int32_kv(gguf, "stt.parakeet.encoder.streaming.sampling_frames_first",
                                         kFamilyTag, 0, hp.enc_stream_sampling_frames_first);
        st != TRANSCRIBE_OK)
    {
        return st;
    }

    // Head kind dispatch. Optional KV with default "tdt" so legacy
    // v2/v3 GGUFs (predate the KV) resolve to TDT, which is what they
    // are. The 8 new variants all ship the KV explicitly.
    {
        std::string head_kind_str;
        if (auto st = read_optional_string_kv(gguf, "stt.parakeet.head_kind",
                                              kFamilyTag, "tdt", head_kind_str);
            st != TRANSCRIBE_OK)
        {
            return st;
        }
        if (head_kind_str == "tdt") {
            hp.head_kind = HeadKind::TDT;
        } else if (head_kind_str == "rnnt") {
            hp.head_kind = HeadKind::RNNT;
        } else if (head_kind_str == "ctc") {
            hp.head_kind = HeadKind::CTC;
        } else {
            std::fprintf(stderr,
                         "parakeet: unsupported head_kind \"%s\" "
                         "(allowed: tdt, rnnt, ctc)\n",
                         head_kind_str.c_str());
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    if (hp.head_kind != HeadKind::CTC) {
        // Predictor (TDT and RNNT only; CTC has no predictor).
        if (auto st = read_required_u32_kv(gguf, "stt.parakeet.predictor.hidden", kFamilyTag,   hp.pred_hidden);   st != TRANSCRIBE_OK) return st;
        if (auto st = read_required_u32_kv(gguf, "stt.parakeet.predictor.n_layers", kFamilyTag, hp.pred_n_layers); st != TRANSCRIBE_OK) return st;
        if (auto st = read_required_u32_kv(gguf, "stt.parakeet.predictor.vocab", kFamilyTag,    hp.pred_vocab);    st != TRANSCRIBE_OK) return st;

        // Joint (TDT and RNNT only).
        if (auto st = read_required_u32_kv(gguf, "stt.parakeet.joint.hidden", kFamilyTag,            hp.joint_hidden);            st != TRANSCRIBE_OK) return st;
        if (auto st = read_required_u32_kv(gguf, "stt.parakeet.joint.num_extra_outputs", kFamilyTag, hp.joint_num_extra_outputs); st != TRANSCRIBE_OK) return st;
        if (auto st = read_required_string_kv(gguf, "stt.parakeet.joint.activation", kFamilyTag, hp.joint_activation);        st != TRANSCRIBE_OK) return st;
    }

    if (hp.head_kind == HeadKind::TDT) {
        // TDT decoding parameters. `durations` is required (the decoder
        // can't run without it); `max_symbols` is optional with a default
        // matching every published v2/v3 config we've seen. RNNT does
        // NOT carry these — its joint emits `vocab+1` (no extras) and
        // the decode loop has no duration choice.
        switch (read_int32_array_kv(gguf, "stt.parakeet.tdt.durations", hp.tdt_durations)) {
            case KvResult::Ok:
                break;
            case KvResult::Absent:
            case KvResult::BadType:
                std::fprintf(stderr,
                             "parakeet: required KV \"stt.parakeet.tdt.durations\" "
                             "missing or wrong type (head_kind=tdt)\n");
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
    } else if (hp.head_kind == HeadKind::RNNT) {
        // RNNT shares the predictor + joint code paths with TDT, but has
        // no durations and the joint emits exactly `vocab+1` logits.
        // Reuse `tdt_max_symbols` as the decode loop's per-frame stuck
        // cap (NeMo's RNNTGreedyDecodeBatched uses the same constant).
        hp.tdt_durations.clear();
        hp.tdt_max_symbols = 10;
    } else {
        // CTC: no predictor, no joint, no durations.
        hp.tdt_durations.clear();
        hp.tdt_max_symbols = 0;
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
    if (auto st = read_required_string_kv(gguf, "stt.frontend.type", kFamilyTag, hp.fe_type);         st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.frontend.num_mels", kFamilyTag, hp.fe_num_mels);     st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.frontend.sample_rate", kFamilyTag, hp.fe_sample_rate);  st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.frontend.n_fft", kFamilyTag, hp.fe_n_fft);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.frontend.win_length", kFamilyTag, hp.fe_win_length);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_u32_kv(gguf, "stt.frontend.hop_length", kFamilyTag, hp.fe_hop_length);   st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string_kv(gguf, "stt.frontend.window", kFamilyTag, hp.fe_window);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_string_kv(gguf, "stt.frontend.normalize", kFamilyTag, hp.fe_normalize);    st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32_kv(gguf, "stt.frontend.dither", kFamilyTag, hp.fe_dither);       st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32_kv(gguf, "stt.frontend.pre_emphasis", kFamilyTag, hp.fe_pre_emphasis); st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32_kv(gguf, "stt.frontend.f_min", kFamilyTag, hp.fe_f_min);        st != TRANSCRIBE_OK) return st;
    if (auto st = read_required_f32_kv(gguf, "stt.frontend.f_max", kFamilyTag, hp.fe_f_max);        st != TRANSCRIBE_OK) return st;

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
    if (hp.head_kind != HeadKind::CTC) {
        if (hp.pred_hidden <= 0 || hp.pred_n_layers <= 0 || hp.pred_vocab <= 1) {
            std::fprintf(stderr, "parakeet: predictor hparams must be positive\n");
            return TRANSCRIBE_ERR_GGUF;
        }
        if (hp.joint_hidden <= 0 || hp.joint_num_extra_outputs < 0) {
            std::fprintf(stderr, "parakeet: joint hparams invalid\n");
            return TRANSCRIBE_ERR_GGUF;
        }
        // Joint activation allow-list. The C++ joint forward implements
        // the same three the reference JointNetwork accepts; anything
        // else is a converter mistake or a future model we haven't
        // ported yet, and producing wrong logits silently is worse than
        // failing loudly here.
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
    }
    if (hp.head_kind == HeadKind::TDT) {
        // TDT durations: must be non-empty, must match num_extra_outputs
        // (each duration logit corresponds to one durations[i] choice),
        // and every entry must be non-negative (negative jumps would
        // walk backwards in time and break the decode loop's invariant).
        // Zero is allowed and is the standard "stay on this frame, just
        // emit a token without advancing" duration.
        if (hp.tdt_durations.empty()) {
            std::fprintf(stderr,
                         "parakeet: stt.parakeet.tdt.durations must be non-empty (head_kind=tdt)\n");
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
    } else if (hp.head_kind == HeadKind::RNNT) {
        // RNNT joint emits exactly vocab+1 — no duration extras.
        if (hp.joint_num_extra_outputs != 0) {
            std::fprintf(stderr,
                         "parakeet: head_kind=rnnt requires joint.num_extra_outputs=0 "
                         "(got %d)\n", hp.joint_num_extra_outputs);
            return TRANSCRIBE_ERR_GGUF;
        }
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

    // Normalize: NeMo's "per_feature" (unbiased per-mel-bin mean/std)
    // for offline Parakeet variants, and "none" for streaming variants
    // (nemotron-speech-streaming-en-0.6b) whose feature normalisation
    // is baked into training rather than applied at inference. NeMo
    // also supports "all_features" and CMVN ("fixed_mean"/"fixed_std")
    // but no published Parakeet variant uses them, and silently
    // substituting per-feature on a GGUF that asked for something else
    // would be a confusing bug.
    if (hp.fe_normalize != "per_feature" && hp.fe_normalize != "none") {
        std::fprintf(stderr,
                     "parakeet: unsupported frontend normalize \"%s\" "
                     "(only \"per_feature\" and \"none\" are implemented)\n",
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

using transcribe::weights::lname;

// The canonical find_tensor() + lname() helpers live in
// src/transcribe-weights-util.{h,cpp}; see that header for rationale.
// They are shared between every per-family weights.cpp. The GET_*
// macros below still live here because their type allowlists encode
// a per-family quantization policy, not a shared convention.
constexpr const char * kTag = kFamilyTag;

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
//              Uses the project-wide TRANSCRIBE_QUANT_CONV_TYPES
//              allowlist (F32 / F16). Quantized kernels would require a
//              quantized im2col path that ggml does not provide.
//
//   GET_LIN  - linear weights consumed by ggml_mul_mat (encoder FF +
//              attention projections, predictor LSTM gate matrices,
//              joint enc/pred/out projections, predictor embed table).
//              Uses the project-wide TRANSCRIBE_QUANT_LINEAR_TYPES
//              allowlist. Every family must accept the same set so a
//              tools/transcribe-quantize preset never produces a
//              tensor the loader refuses.
#define GET_F32(slot, name, ...) \
    do { \
        ggml_tensor * _t = transcribe::weights::find_tensor( \
            ctx_meta, (name), \
            {GGML_TYPE_F32}, {__VA_ARGS__}, kTag); \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF; \
        (slot) = _t; \
    } while (0)

#define GET_CONV(slot, name, ...) \
    do { \
        ggml_tensor * _t = transcribe::weights::find_tensor( \
            ctx_meta, (name), \
            {TRANSCRIBE_QUANT_CONV_TYPES}, \
            {__VA_ARGS__}, kTag); \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF; \
        (slot) = _t; \
    } while (0)

#define GET_LIN(slot, name, ...) \
    do { \
        ggml_tensor * _t = transcribe::weights::find_tensor( \
            ctx_meta, (name), \
            {TRANSCRIBE_QUANT_LINEAR_TYPES}, \
            {__VA_ARGS__}, kTag); \
        if (_t == nullptr) return TRANSCRIBE_ERR_GGUF; \
        (slot) = _t; \
    } while (0)

} // namespace

transcribe_status build_parakeet_weights(ggml_context *          ctx_meta,
                                         const ParakeetHParams & hp,
                                         ParakeetWeights &       weights)
{
    if (ctx_meta == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int64_t channels = hp.enc_subsampling_channels;
    const int64_t d_model  = hp.enc_d_model;
    const int64_t d_ff     = hp.enc_d_ff;
    const int64_t n_heads  = hp.enc_n_heads;
    const int64_t head_dim = hp.enc_head_dim();
    const int64_t k        = hp.enc_conv_kernel;
    const int64_t pred_h   = hp.pred_hidden;       // 0 for CTC
    const int64_t pred_v   = hp.pred_vocab;        // 0 for CTC
    const int64_t joint_h  = hp.joint_hidden;      // 0 for CTC
    const int64_t joint_n  = hp.joint_n_classes(); // meaningless for CTC; we read CTC head shape instead

    // The flattened "freq" axis going into pre_encode.out is
    // (subsampling_channels * F') where F' is the freq dim after the
    // three stride-2 convs. The pre-encode convs are k=3, s=2 (NeMo's
    // dw_striding stack); their padding tracks NeMo's
    // `encoder.causal_downsampling` flag, NOT the conformer's
    // conv_context_size (which is the kernel-9 conv-module's context
    // and lives in hp.enc_conv_context_{left,right}). The two are
    // unrelated.
    //
    //   causal_downsampling=false (every offline variant and the
    //     buffered-streaming parakeet-unified-en-0.6b): symmetric
    //     (k-1)/2 padding on both sides. 128 → 64 → 32 → 16; 256*16
    //     = 4096.
    //   causal_downsampling=true  (cache-aware streaming variants,
    //     i.e. nemotron-speech-streaming-en-0.6b): CausalConv2D with
    //     (left=k-1, right=stride-1) on both axes. 128 → 65 → 33 →
    //     17; 256*17 = 4352.
    //
    // We don't currently emit the boolean to GGUF; we infer it from
    // the attention style. Only ChunkedLimited (2-tuple, cache-aware)
    // implies causal pre-encode. Regular (offline full / local) and
    // ChunkedLimitedWithRc (3-tuple buffered) both use non-causal.
    auto pre_encode_F_prime = [&]() {
        const bool causal =
            (hp.enc_att_context_style ==
                 ParakeetHParams::AttContextStyle::ChunkedLimited);
        const int k = 3, s = 2;
        // Total per-axis pad: (k-1)/2 + (k-1)/2 = k-1 = 2 for non-causal;
        // (k-1) + (s-1) = 3 for causal.
        const int total_pad = causal ? ((k - 1) + (s - 1))
                                     : ((k - 1) / 2 + (k - 1) / 2);
        int dim = hp.fe_num_mels;
        for (int i = 0; i < 3; ++i) {
            dim = ((dim + total_pad - k) / s) + 1;
        }
        return static_cast<int64_t>(dim);
    };
    const int64_t pre_encode_freq = channels * pre_encode_F_prime();
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

        // Self-attention with relative position. Linears can carry
        // biases when `enc.use_bias=true` (1.1B / rnnt / ctc variants);
        // v2/v3 ship without (Q/K/V/out + pos = bias-free). The
        // per-head pos_bias_u/v live alongside regardless. NeMo's
        // `linear_pos` is bias-free even when use_bias=true.
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
        if (hp.enc_conv_norm_type == ParakeetHParams::ConvNormType::BatchNorm) {
            GET_F32 (b.conv_bn_rm, lname("enc.blocks.%d.conv.bn.running_mean", i), d_model);
            GET_F32 (b.conv_bn_rv, lname("enc.blocks.%d.conv.bn.running_var",  i), d_model);
        }

        // Macaron FF2.
        GET_F32(b.norm_ff2_w, lname("enc.blocks.%d.norm_ff2.weight", i), d_model);
        GET_F32(b.norm_ff2_b, lname("enc.blocks.%d.norm_ff2.bias",   i), d_model);
        GET_LIN(b.ff2_lin1_w, lname("enc.blocks.%d.ff2.linear1.weight", i), d_model, d_ff);
        GET_LIN(b.ff2_lin2_w, lname("enc.blocks.%d.ff2.linear2.weight", i), d_ff,    d_model);

        // Final per-block layer norm.
        GET_F32(b.norm_out_w, lname("enc.blocks.%d.norm_out.weight", i), d_model);
        GET_F32(b.norm_out_b, lname("enc.blocks.%d.norm_out.bias",   i), d_model);

        // Optional encoder linear/conv biases. Loaded only when
        // hp.enc_use_bias=true (1.1B FastConformer-XL, rnnt-*, ctc-*).
        // The 11 names mirror the converter's ENCODER_BLOCK_BIAS_TABLE
        // exactly. NeMo's `linear_pos` is bias-free even when
        // use_bias=true so attn_pos has no bias slot.
        if (hp.enc_use_bias) {
            GET_F32(b.ff1_lin1_b,  lname("enc.blocks.%d.ff1.linear1.bias",      i), d_ff);
            GET_F32(b.ff1_lin2_b,  lname("enc.blocks.%d.ff1.linear2.bias",      i), d_model);
            GET_F32(b.attn_q_b,    lname("enc.blocks.%d.attn.linear_q.bias",    i), d_model);
            GET_F32(b.attn_k_b,    lname("enc.blocks.%d.attn.linear_k.bias",    i), d_model);
            GET_F32(b.attn_v_b,    lname("enc.blocks.%d.attn.linear_v.bias",    i), d_model);
            GET_F32(b.attn_out_b,  lname("enc.blocks.%d.attn.linear_out.bias",  i), d_model);
            GET_F32(b.conv_pw1_b,  lname("enc.blocks.%d.conv.pointwise1.bias", i), 2 * d_model);
            GET_F32(b.conv_dw_b,   lname("enc.blocks.%d.conv.depthwise.bias",  i), d_model);
            GET_F32(b.conv_pw2_b,  lname("enc.blocks.%d.conv.pointwise2.bias", i), d_model);
            GET_F32(b.ff2_lin1_b,  lname("enc.blocks.%d.ff2.linear1.bias",      i), d_ff);
            GET_F32(b.ff2_lin2_b,  lname("enc.blocks.%d.ff2.linear2.bias",      i), d_model);
        }
    }

    if (hp.head_kind == HeadKind::CTC) {
        // ----- CTC head -----
        //
        // Single 1×1 Conv1d projecting d_model -> vocab+1. NeMo's
        // ConvASRDecoder stores it as `decoder.decoder_layers.0`; the
        // converter flattens to `head.ctc.weight` /
        // `head.ctc.bias`.
        //
        // Tensor shape in PyTorch is [vocab+1, d_model, 1]
        // (out_channels, in_channels, kernel). In ggml ne (fast-to-slow)
        // that's [1, d_model, vocab+1]. We don't know `vocab+1` from the
        // hparams (CTC GGUFs ship neither predictor.vocab nor
        // joint_n_classes), so peek at the raw tensor to capture the
        // trailing dim, then run the standard find_tensor with a fully
        // specified expected shape so the type+rank checks still fire.
        ggml_tensor * ctc_peek = ggml_get_tensor(ctx_meta, "head.ctc.weight");
        if (ctc_peek == nullptr) {
            std::fprintf(stderr, "parakeet: missing tensor \"head.ctc.weight\"\n");
            return TRANSCRIBE_ERR_GGUF;
        }
        const int64_t n_classes = ctc_peek->ne[2];
        if (n_classes <= 1) {
            std::fprintf(stderr,
                         "parakeet: head.ctc.weight vocab+1 (=%lld) must be > 1\n",
                         (long long)n_classes);
            return TRANSCRIBE_ERR_GGUF;
        }
        GET_LIN(weights.ctc_head.weight, "head.ctc.weight", 1, d_model, n_classes);
        GET_F32(weights.ctc_head.bias,   "head.ctc.bias",   n_classes);
        // Stash the resolved vocab+1 into hp for downstream consumers.
        // We're given a const &; we relax that on the way out via a
        // const_cast — safe because read_parakeet_hparams /
        // build_parakeet_weights are the single producer of hp and its
        // only contract is "writable while loading".
        const_cast<ParakeetHParams &>(hp).head_ctc_n_classes =
            static_cast<int32_t>(n_classes);
    } else {
        // ----- predictor (TDT, RNNT) -----
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

        // ----- joint (TDT, RNNT) -----
        GET_LIN(weights.joint.enc_w,  "joint.enc.weight", d_model, joint_h);
        GET_F32(weights.joint.enc_b,  "joint.enc.bias", joint_h);
        GET_LIN(weights.joint.pred_w, "joint.pred.weight", pred_h,  joint_h);
        GET_F32(weights.joint.pred_b, "joint.pred.bias", joint_h);
        GET_LIN(weights.joint.out_w,  "joint.out.weight", joint_h, joint_n);
        GET_F32(weights.joint.out_b,  "joint.out.bias", joint_n);
    }

    return TRANSCRIBE_OK;
}

#undef GET_F32
#undef GET_CONV
#undef GET_LIN

} // namespace transcribe::parakeet
