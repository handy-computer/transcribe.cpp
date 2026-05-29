// arch/canary_qwen/model.cpp - SALM (FastConformer + Qwen3-1.7B) family handler.
//
// Forward shape:
//   pcm
//     -> mel preprocessor (NeMo AudioToMelSpectrogramPreprocessor;
//        per_feature normalize; trailing-frame masked to zero)
//     -> FastConformer encoder (32 blocks; identical to canary-1b-flash)
//     -> perception projection (Linear(1024, 2048) + bias)
//   prompt = HF chat template applied to
//            "Transcribe the following: <|audioplaceholder|>"
//     -> token ids (15 ids for the JFK case)
//   audio scatter: replace single <|audioplaceholder|> position with
//                  T_enc=138 perception rows -> input_embeds (T_prompt, hidden=2048)
//     -> Qwen3-1.7B causal LM (28 blocks)
//     -> tied lm_head -> greedy autoregressive loop until EOS or max_new.
//
// BF16 weight promotion: the canary-1b-flash perception encoder was
// validated F32-only; the SALM checkpoint stores its weights as BF16.
// On CPU primary backend we promote ALL BF16 linear weights (encoder
// and decoder) to F32 at load time so ggml's CPU matmul hits the F32
// path. This matches the reference's `model_dtype = f32` regime.

#include "canary_qwen.h"

#include "decoder.h"
#include "encoder.h"
#include "weights.h"

#include "qwen3_lm/qwen3_lm.h"
#include "transcribe-arch.h"
#include "transcribe-debug.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-mel.h"
#include "transcribe-meta.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace transcribe::canary_qwen {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   CanaryQwenModel>);
static_assert(std::is_base_of_v<transcribe_session, CanaryQwenSession>);

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

CanaryQwenSession::~CanaryQwenSession() {
    kv_cache.free();
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
}

CanaryQwenModel::~CanaryQwenModel() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (backend_buffer != nullptr) {
        ggml_backend_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
    if (bn_fused_buffer != nullptr) {
        ggml_backend_buffer_free(bn_fused_buffer);
        bn_fused_buffer = nullptr;
    }
    if (bn_fused_ctx != nullptr) {
        ggml_free(bn_fused_ctx);
        bn_fused_ctx = nullptr;
    }
    if (conv_pw_f32_buffer != nullptr) {
        ggml_backend_buffer_free(conv_pw_f32_buffer);
        conv_pw_f32_buffer = nullptr;
    }
    if (conv_pw_f32_ctx != nullptr) {
        ggml_free(conv_pw_f32_ctx);
        conv_pw_f32_ctx = nullptr;
    }
    if (linear_f32_buffer != nullptr) {
        ggml_backend_buffer_free(linear_f32_buffer);
        linear_f32_buffer = nullptr;
    }
    if (linear_f32_ctx != nullptr) {
        ggml_free(linear_f32_ctx);
        linear_f32_ctx = nullptr;
    }
    packed_gate_up.free();
    for (auto it = plan.scheduler_list.rbegin();
         it != plan.scheduler_list.rend(); ++it)
    {
        ggml_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary      = nullptr;
    plan.primary_kind = transcribe::BackendKind::Unknown;
}

namespace {

constexpr const char k_default_variant[] = "canary-qwen-2.5b";
constexpr float kBnEps = 1e-5f;

// ---------------------------------------------------------------------------
// Chat-template prompt construction
// ---------------------------------------------------------------------------
//
// Per the reference SALM trace, HF's chat template renders the JFK
// request as the literal token sequence:
//   <|im_start|>  user  \n  Trans  cribe   the   following  :   <space>
//   <|audioplaceholder|>
//   <|im_end|>  \n  <|im_start|>  assistant  \n
//
// Verified empirically: tok.encode("user\nTranscribe the following: ")
// -> [872, 198, 3167, 3114, 279, 2701, 25, 220].
//
// We pre-encode the prefix ("<|im_start|>" + "user\nTranscribe the
// following: ") and suffix ("<|im_end|>\n<|im_start|>assistant\n") at
// load time. The audio_locator id is replicated T_enc times between
// them at run time.
//
// The ChatTokens struct holds the few special-token ids we need.

transcribe_status resolve_chat_tokens(const transcribe::Tokenizer & tok,
                                      ChatTokens &                  out)
{
    struct PieceSlot { const char * piece; int32_t * slot; };
    const PieceSlot pieces[] = {
        { "<|im_start|>", &out.im_start       },
        { "<|im_end|>",   &out.im_end         },
        { "user",         &out.role_user      },
        { "assistant",    &out.role_assistant },
    };
    for (const auto & p : pieces) {
        const int id = tok.find(p.piece);
        if (id < 0) {
            std::fprintf(stderr,
                         "canary_qwen: chat-template piece \"%s\" not in tokenizer\n",
                         p.piece);
            return TRANSCRIBE_ERR_GGUF;
        }
        *p.slot = id;
    }
    return TRANSCRIBE_OK;
}

transcribe_status build_static_prompt_segments(CanaryQwenModel & m) {
    if (!m.tok.has_encoder()) {
        std::fprintf(stderr,
                     "canary_qwen: tokenizer has no BPE encoder — cannot "
                     "tokenize chat template at load time\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // prefix = [<|im_start|>] + bpe("user\nTranscribe the following: ")
    m.prompt_prefix_ids.clear();
    m.prompt_prefix_ids.push_back(m.chat_tokens.im_start);
    {
        std::vector<int32_t> ids;
        if (auto st = m.tok.encode("user\nTranscribe the following: ", ids);
            st != TRANSCRIBE_OK)
        {
            std::fprintf(stderr, "canary_qwen: prefix BPE encode failed\n");
            return st;
        }
        m.prompt_prefix_ids.insert(m.prompt_prefix_ids.end(),
                                   ids.begin(), ids.end());
    }

    // suffix = [<|im_end|>] + bpe("\n") + [<|im_start|>] + bpe("assistant\n")
    m.prompt_suffix_ids.clear();
    m.prompt_suffix_ids.push_back(m.chat_tokens.im_end);
    {
        std::vector<int32_t> ids;
        if (auto st = m.tok.encode("\n", ids); st != TRANSCRIBE_OK) return st;
        m.prompt_suffix_ids.insert(m.prompt_suffix_ids.end(),
                                   ids.begin(), ids.end());
    }
    m.prompt_suffix_ids.push_back(m.chat_tokens.im_start);
    {
        std::vector<int32_t> ids;
        if (auto st = m.tok.encode("assistant\n", ids); st != TRANSCRIBE_OK) return st;
        m.prompt_suffix_ids.insert(m.prompt_suffix_ids.end(),
                                   ids.begin(), ids.end());
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// BatchNorm fusion (same math as canary). Pre-fused (scale, bias) fall
// out of (running_mean, running_var, weight, bias, eps).
// ---------------------------------------------------------------------------

transcribe_status fuse_batch_norm(CanaryQwenModel & m) {
    const size_t n_blocks = m.weights.blocks.size();
    if (n_blocks == 0) return TRANSCRIBE_OK;

    const int64_t d = m.hparams.enc_d_model;
    const size_t  tensor_bytes = static_cast<size_t>(d) * sizeof(float);

    const size_t ctx_size = n_blocks * 2 * ggml_tensor_overhead() + 256;
    ggml_init_params params = {ctx_size, nullptr, true};
    m.bn_fused_ctx = ggml_init(params);
    if (m.bn_fused_ctx == nullptr) return TRANSCRIBE_ERR_BACKEND;

    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.weights.blocks[i];
        b.conv_bn_fused_scale = ggml_new_tensor_1d(m.bn_fused_ctx, GGML_TYPE_F32, d);
        b.conv_bn_fused_bias  = ggml_new_tensor_1d(m.bn_fused_ctx, GGML_TYPE_F32, d);
    }

    m.bn_fused_buffer = ggml_backend_alloc_ctx_tensors(
        m.bn_fused_ctx, m.plan.scheduler_list.back());
    if (m.bn_fused_buffer == nullptr) return TRANSCRIBE_ERR_BACKEND;

    std::vector<float> bn_w(d), bn_b(d), rm(d), rv(d);
    std::vector<float> fused_s(d), fused_b(d);

    for (size_t i = 0; i < n_blocks; ++i) {
        auto & b = m.weights.blocks[i];
        ggml_backend_tensor_get(b.conv_bn_w,  bn_w.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_b,  bn_b.data(), 0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_rm, rm.data(),   0, tensor_bytes);
        ggml_backend_tensor_get(b.conv_bn_rv, rv.data(),   0, tensor_bytes);

        for (int64_t c = 0; c < d; ++c) {
            const float s = bn_w[c] / std::sqrt(rv[c] + kBnEps);
            fused_s[c] = s;
            fused_b[c] = bn_b[c] - rm[c] * s;
        }

        ggml_backend_tensor_set(b.conv_bn_fused_scale, fused_s.data(), 0, tensor_bytes);
        ggml_backend_tensor_set(b.conv_bn_fused_bias,  fused_b.data(), 0, tensor_bytes);
    }

    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// F16 → F32 promotion for pointwise AND depthwise conv kernels (any backend).
//
// canary-1b-flash (the closest sibling) ships F32 conv kernels because its
// reference dtype is F32. canary-qwen-2.5b is BF16-reference, so the
// converter stores conv kernels at F16 (the loader rejects BF16 conv).
//
// `ggml_conv_2d_dw_direct` silently produces wildly wrong output for F16
// kernels (verified on both CPU and Metal): the conv module's output
// magnitude blows up by ~1500x on the first block, destroying every
// downstream value (decoder collapses to a single `!` token). Promoting
// the depthwise kernel to F32 at load time bypasses that path's dtype
// handling and matches NeMo to single-percent drift.
//
// The pointwise kernels were originally promoted only on CPU (a perf fix
// — the F16 path is slow there because of dequant inside matmul) but
// promoting them on Metal too is harmless and simplifies the policy.
//
// We do NOT call `load_common::promote_conv_pw_f16_to_f32_on_cpu` because
// that helper has a hard `if (primary_kind != Cpu) return OK` early-out.
// Refactoring the shared helper to accept a force-flag is the right
// long-term move; for now we duplicate the logic here so the canary_qwen
// fix does not depend on cross-family churn.
// ---------------------------------------------------------------------------

transcribe_status promote_conv_pw_to_f32_on_cpu(CanaryQwenModel & m) {
    if (m.plan.primary == nullptr) return TRANSCRIBE_OK;

    struct Slot { ggml_tensor ** dst_slot; ggml_tensor * src; };
    std::vector<Slot> slots;
    slots.reserve(m.weights.blocks.size() * 3);
    auto add = [&](ggml_tensor ** s) {
        if (s != nullptr && *s != nullptr && (*s)->type == GGML_TYPE_F16) {
            slots.push_back({s, *s});
        }
    };
    for (auto & b : m.weights.blocks) {
        add(&b.conv_pw1_w);
        add(&b.conv_dw_w);
        add(&b.conv_pw2_w);
    }
    if (slots.empty()) return TRANSCRIBE_OK;

    const size_t ctx_size = slots.size() * ggml_tensor_overhead() + 256;
    ggml_init_params init_params = {ctx_size, nullptr, true};
    m.conv_pw_f32_ctx = ggml_init(init_params);
    if (m.conv_pw_f32_ctx == nullptr) return TRANSCRIBE_ERR_BACKEND;

    std::vector<ggml_tensor *> replacements;
    replacements.reserve(slots.size());
    for (const auto & s : slots) {
        ggml_tensor * r = ggml_new_tensor(
            m.conv_pw_f32_ctx, GGML_TYPE_F32,
            ggml_n_dims(s.src), s.src->ne);
        if (r == nullptr) {
            ggml_free(m.conv_pw_f32_ctx);
            m.conv_pw_f32_ctx = nullptr;
            return TRANSCRIBE_ERR_BACKEND;
        }
        ggml_set_name(r, s.src->name);
        replacements.push_back(r);
    }

    m.conv_pw_f32_buffer = ggml_backend_alloc_ctx_tensors(
        m.conv_pw_f32_ctx, m.plan.primary);
    if (m.conv_pw_f32_buffer == nullptr) {
        std::fprintf(stderr,
            "canary_qwen: conv F16->F32 promotion buffer alloc failed\n");
        ggml_free(m.conv_pw_f32_ctx);
        m.conv_pw_f32_ctx = nullptr;
        return TRANSCRIBE_ERR_BACKEND;
    }
    ggml_backend_buffer_set_usage(
        m.conv_pw_f32_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const auto * f16_traits = ggml_get_type_traits(GGML_TYPE_F16);
    if (f16_traits == nullptr || f16_traits->to_float == nullptr) {
        std::fprintf(stderr,
            "canary_qwen: no f16 to_float trait — skipping conv promotion\n");
        ggml_backend_buffer_free(m.conv_pw_f32_buffer);
        m.conv_pw_f32_buffer = nullptr;
        ggml_free(m.conv_pw_f32_ctx);
        m.conv_pw_f32_ctx = nullptr;
        return TRANSCRIBE_OK;
    }

    std::vector<uint8_t> f16_staging;
    std::vector<float>   f32_staging;
    for (size_t i = 0; i < slots.size(); ++i) {
        ggml_tensor * src = slots[i].src;
        ggml_tensor * dst = replacements[i];
        const int64_t n_elem    = ggml_nelements(src);
        const size_t  f16_bytes = ggml_nbytes(src);
        const size_t  f32_bytes = static_cast<size_t>(n_elem) * sizeof(float);

        if (f16_staging.size() < f16_bytes) f16_staging.resize(f16_bytes);
        if (f32_staging.size() < static_cast<size_t>(n_elem)) {
            f32_staging.resize(n_elem);
        }

        ggml_backend_tensor_get(src, f16_staging.data(), 0, f16_bytes);
        f16_traits->to_float(f16_staging.data(), f32_staging.data(), n_elem);
        ggml_backend_tensor_set(dst, f32_staging.data(), 0, f32_bytes);

        *slots[i].dst_slot = dst;
    }

    std::fprintf(stderr,
        "canary_qwen: promoted %zu F16 conv weights to F32\n", slots.size());
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// CPU-only BF16 → F32 promotion of all linear weights.
//
// The reference (NeMo SALM dumper) loads the model in F32 and runs F32
// inference. The canary-1b-flash encoder (whose weights canary_qwen
// reuses) was validated F32-only when ported; the BF16 matmul path
// through ggml's CPU backend is unverified for this graph topology.
//
// We promote ALL BF16 linear weights to F32 once at load time. This
// stays inside the existing F32 weight buffer (separate from
// backend_buffer, so we can free it independently). The graph builders
// don't need to change — they read whatever dtype each tensor pointer
// reports.
//
// On non-CPU primary backend (Metal, Vulkan), we skip — those backends
// have their own well-tested BF16 paths.
// ---------------------------------------------------------------------------

transcribe_status promote_linears_bf16_to_f32_on_cpu(CanaryQwenModel & m) {
    if (m.plan.primary_kind != transcribe::BackendKind::Cpu ||
        m.plan.primary == nullptr)
    {
        return TRANSCRIBE_OK;
    }

    // Collect (slot_pointer, source_tensor) for every BF16 linear we own.
    struct Slot { ggml_tensor ** dst_slot; ggml_tensor * src; };
    std::vector<Slot> slots;
    auto add = [&](ggml_tensor ** s) {
        if (s != nullptr && *s != nullptr && (*s)->type == GGML_TYPE_BF16) {
            slots.push_back({s, *s});
        }
    };

    // Encoder pre-encode out projection.
    add(&m.weights.pre_encode.out_w);

    // Encoder blocks — every linear (FFs, Q/K/V/O/pos).
    for (auto & b : m.weights.blocks) {
        add(&b.ff1_lin1_w); add(&b.ff1_lin2_w);
        add(&b.attn_q_w);   add(&b.attn_k_w);
        add(&b.attn_v_w);   add(&b.attn_out_w);
        add(&b.attn_pos_w);
        add(&b.ff2_lin1_w); add(&b.ff2_lin2_w);
    }

    // Perception projection.
    add(&m.weights.perception_proj.weight);

    // Decoder embedding (also serves as tied lm_head).
    add(&m.weights.dec_embed.token_w);

    // Decoder blocks — Q/K/V/O + gate/up/down.
    for (auto & b : m.weights.dec_blocks) {
        add(&b.attn_q_w); add(&b.attn_k_w);
        add(&b.attn_v_w); add(&b.attn_o_w);
        add(&b.ffn_gate_w); add(&b.ffn_up_w); add(&b.ffn_down_w);
    }

    if (slots.empty()) return TRANSCRIBE_OK;

    // Allocate a dedicated ctx + buffer for the F32 replacements.
    const size_t ctx_size = slots.size() * ggml_tensor_overhead() + 256;
    ggml_init_params init_params = {ctx_size, nullptr, true};
    m.linear_f32_ctx = ggml_init(init_params);
    if (m.linear_f32_ctx == nullptr) return TRANSCRIBE_ERR_BACKEND;

    std::vector<ggml_tensor *> replacements;
    replacements.reserve(slots.size());
    for (const auto & s : slots) {
        ggml_tensor * r = ggml_new_tensor(
            m.linear_f32_ctx, GGML_TYPE_F32,
            ggml_n_dims(s.src), s.src->ne);
        if (r == nullptr) {
            ggml_free(m.linear_f32_ctx);
            m.linear_f32_ctx = nullptr;
            return TRANSCRIBE_ERR_BACKEND;
        }
        ggml_set_name(r, s.src->name);
        replacements.push_back(r);
    }

    m.linear_f32_buffer = ggml_backend_alloc_ctx_tensors(
        m.linear_f32_ctx, m.plan.primary);
    if (m.linear_f32_buffer == nullptr) {
        std::fprintf(stderr,
            "canary_qwen: BF16→F32 linear promotion buffer alloc failed\n");
        ggml_free(m.linear_f32_ctx);
        m.linear_f32_ctx = nullptr;
        return TRANSCRIBE_ERR_BACKEND;
    }
    ggml_backend_buffer_set_usage(
        m.linear_f32_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const auto * bf16_traits = ggml_get_type_traits(GGML_TYPE_BF16);
    if (bf16_traits == nullptr || bf16_traits->to_float == nullptr) {
        std::fprintf(stderr,
            "canary_qwen: no bf16 to_float trait — skipping linear promotion\n");
        ggml_backend_buffer_free(m.linear_f32_buffer);
        m.linear_f32_buffer = nullptr;
        ggml_free(m.linear_f32_ctx);
        m.linear_f32_ctx = nullptr;
        return TRANSCRIBE_OK;
    }

    std::vector<uint8_t> bf16_staging;
    std::vector<float>   f32_staging;
    for (size_t i = 0; i < slots.size(); ++i) {
        ggml_tensor * src = slots[i].src;
        ggml_tensor * dst = replacements[i];
        const int64_t n_elem    = ggml_nelements(src);
        const size_t  bf16_bytes = ggml_nbytes(src);
        const size_t  f32_bytes  = static_cast<size_t>(n_elem) * sizeof(float);

        if (bf16_staging.size() < bf16_bytes) bf16_staging.resize(bf16_bytes);
        if (f32_staging.size() < static_cast<size_t>(n_elem)) {
            f32_staging.resize(n_elem);
        }

        ggml_backend_tensor_get(src, bf16_staging.data(), 0, bf16_bytes);
        bf16_traits->to_float(bf16_staging.data(), f32_staging.data(), n_elem);
        ggml_backend_tensor_set(dst, f32_staging.data(), 0, f32_bytes);

        *slots[i].dst_slot = dst;
    }

    std::fprintf(stderr,
        "canary_qwen: promoted %zu BF16 linear weights to F32 for CPU backend\n",
        slots.size());
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Loader entry points (forward-declared so we can register `arch` after).
// ---------------------------------------------------------------------------

extern transcribe_status load        (Loader &, const transcribe_model_load_params *,
                                      transcribe_model **);
extern transcribe_status init_context(transcribe_model *, const transcribe_session_params *,
                                      transcribe_session **);
extern transcribe_status run         (transcribe_session *, const float *, int,
                                      const transcribe_run_params *);

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

transcribe_status load(
    Loader &                          loader,
    const transcribe_model_load_params *   params,
    transcribe_model **               out_model)
{
    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<CanaryQwenModel>();
    m->arch      = &arch;
    m->t_load_us = 0;
    m->variant   = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (auto st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) return st;
    if (auto st = read_languages_kv (loader.gguf(), *m);       st != TRANSCRIBE_OK) return st;

    if (auto st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) return st;

    if (auto st = read_canary_qwen_hparams(loader.gguf(), m->hparams);
        st != TRANSCRIBE_OK) return st;

    m->hparams.vocab_size   = m->tok.n_tokens();
    m->hparams.bos_token_id = m->tok.bos_id();
    m->hparams.eos_token_id = m->tok.eos_id();

    if (m->hparams.vocab_size != m->hparams.dec_vocab_size) {
        std::fprintf(stderr,
                     "canary_qwen: tokenizer vocab (%d) != decoder vocab_size (%d)\n",
                     m->hparams.vocab_size, m->hparams.dec_vocab_size);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (m->hparams.eos_token_id < 0) {
        std::fprintf(stderr,
                     "canary_qwen: GGUF tokenizer has no eos_token_id\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    if (auto st = resolve_chat_tokens(m->tok, m->chat_tokens);   st != TRANSCRIBE_OK) return st;
    if (auto st = build_static_prompt_segments(*m);              st != TRANSCRIBE_OK) return st;

    // ---- Mel frontend ----
    {
        transcribe::MelConfig cfg {};
        cfg.sample_rate  = m->hparams.fe_sample_rate;
        cfg.num_mels     = m->hparams.fe_num_mels;
        cfg.n_fft        = m->hparams.fe_n_fft;
        cfg.win_length   = m->hparams.fe_win_length;
        cfg.hop_length   = m->hparams.fe_hop_length;
        cfg.pre_emphasis = m->hparams.fe_pre_emphasis;
        cfg.f_min        = m->hparams.fe_f_min;
        cfg.f_max        = m->hparams.fe_f_max;
        cfg.pad_mode     = m->hparams.fe_pad_mode;
        // NeMo's AudioToMelSpectrogramPreprocessor uses periodic=False
        // (symmetric Hann); the GGUF KV stt.frontend.window only names
        // the window family.
        cfg.window_type  = "hann_symmetric";
        cfg.normalize    = m->hparams.fe_normalize;

        // Pull baked filterbank/window from GGUF if present.
        using R = load_common::ReadF32Result;
        const size_t fb_elems = static_cast<size_t>(cfg.num_mels) *
                                static_cast<size_t>(cfg.n_fft / 2 + 1);
        const auto fb_rc = load_common::read_f32_tensor_checked(
            loader.gguf(), loader.path(),
            "frontend.mel_filterbank", fb_elems,
            "canary_qwen", cfg.filterbank);
        if (fb_rc != R::Ok && fb_rc != R::Absent) {
            return TRANSCRIBE_ERR_GGUF;
        }
        const size_t win_elems = static_cast<size_t>(cfg.win_length);
        const auto win_rc = load_common::read_f32_tensor_checked(
            loader.gguf(), loader.path(),
            "frontend.window", win_elems,
            "canary_qwen", cfg.window);
        if (win_rc != R::Ok && win_rc != R::Absent) {
            return TRANSCRIBE_ERR_GGUF;
        }

        m->mel.emplace(cfg);
    }

    // ---- Reopen GGUF with no_alloc to bind the tensor catalog ----
    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) return TRANSCRIBE_ERR_GGUF;

    if (auto st = build_canary_qwen_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    // ---- Backend plan + alloc + stream tensor data ----
    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (auto st = load_common::init_backends(backend_req, "canary_qwen", m->plan);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }
    m->backend = ggml_backend_name(m->plan.primary);

    ggml_backend_buffer_t weights_buffer =
        ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (weights_buffer == nullptr) {
        gguf_free(gguf_data);
        std::fprintf(stderr,
                     "canary_qwen: ggml_backend_alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (auto st = load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "canary_qwen");
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    // ---- Post-load weight transformations ----
    if (auto st = fuse_batch_norm(*m); st != TRANSCRIBE_OK) return st;

    // F16 conv pointwise → F32 (CPU only). Same as canary.
    if (auto st = promote_conv_pw_to_f32_on_cpu(*m); st != TRANSCRIBE_OK) return st;

    // BF16 linears → F32 (CPU only). Mitigates the canary-1b-flash
    // F32-only validation gap for this BF16 GGUF.
    if (auto st = promote_linears_bf16_to_f32_on_cpu(*m); st != TRANSCRIBE_OK) return st;

    // Pack gate + up into one tensor per layer (one mul_mat instead of two).
    {
        std::vector<transcribe::qwen3_lm::GateUpEntry> entries;
        entries.reserve(m->weights.dec_blocks.size());
        for (auto & b : m->weights.dec_blocks) {
            entries.push_back({b.ffn_gate_w, b.ffn_up_w, &b.ffn_gate_up_w});
        }
        if (!transcribe::qwen3_lm::pack_gate_up(
                m->plan.primary,
                m->hparams.dec_hidden,
                m->hparams.dec_intermediate,
                entries,
                m->packed_gate_up,
                "canary_qwen"))
        {
            m->packed_gate_up.free();
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model = m.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// init_context
// ---------------------------------------------------------------------------

transcribe_status init_context(
    transcribe_model *                model,
    const transcribe_session_params * params,
    transcribe_session **             out_ctx)
{
    if (model->arch != &arch) return TRANSCRIBE_ERR_INVALID_ARG;

    auto cc = std::make_unique<CanaryQwenSession>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    // Reference runs without flash; default to off for tightest tensor
    // parity. TRANSCRIBE_FLASH_DECODER=1 (or =encoder) overrides.
    // Flash defaults: encoder off (the FastConformer rel-pos path has a
    // manual rel_shift trick that ggml_flash_attn_ext doesn't subsume),
    // decoder ON (the Qwen3 LM step graph is dispatch-bound on Metal —
    // turning flash on takes per-step kernel count from ~22 ops/layer to
    // ~14 and yields ~2.4x decode speedup measured on jfk.wav).
    cc->encoder_use_flash = false;
    cc->decoder_use_flash = true;
    transcribe::flash::apply_env_overrides(cc->encoder_use_flash,
                                           cc->decoder_use_flash);

    auto * cm = static_cast<CanaryQwenModel *>(model);
    {
        ggml_type kv_type = GGML_TYPE_F16;
        if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) kv_type = GGML_TYPE_F32;
        if (!transcribe::qwen3_lm::kv_init(cc->kv_cache, cm->plan.primary,
                                           /*n_ctx=*/2048,
                                           cm->hparams.dec_n_kv_heads,
                                           cm->hparams.dec_head_dim,
                                           cm->hparams.dec_n_layers,
                                           kv_type))
        {
            std::fprintf(stderr,
                         "canary_qwen init_context: kv_init failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

void apply_thread_policy(CanaryQwenSession * cc) {
    int n_threads = cc->n_threads;
    if (n_threads <= 0) {
        n_threads = std::min(8, std::max(1, static_cast<int>(
            std::thread::hardware_concurrency())));
    }
    for (int i = 0; i < ggml_backend_sched_get_n_backends(cc->sched); ++i) {
        ggml_backend_t be       = ggml_backend_sched_get_backend(cc->sched, i);
        ggml_backend_dev_t dev  = ggml_backend_get_device(be);
        ggml_backend_reg_t reg  = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg == nullptr) continue;
        auto * fn = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (fn != nullptr) fn(be, n_threads);
    }
}

void build_relpos_emb_host(std::vector<float> & pos_buf,
                           std::vector<float> & div_term,
                           int                  d_model,
                           int                  T_enc)
{
    // RelPositionalEncoding produces pos_emb of shape (1, 2*T_enc - 1, d_model).
    // The position values descend from (T_enc - 1) at index 0 down to
    // -(T_enc - 1) at index 2*T_enc - 2 (per NeMo's
    // `positions = torch.arange(length - 1, -length, -1, dtype=torch.float32)`).
    const int pos_len = 2 * T_enc - 1;
    pos_buf.assign(static_cast<size_t>(pos_len) * d_model, 0.0f);
    div_term.resize(static_cast<size_t>(d_model / 2));

    // div_term[k] = exp(-2k * ln(10000) / d_model)
    const float ln_10000 = std::log(10000.0f);
    for (int k = 0; k < d_model / 2; ++k) {
        div_term[static_cast<size_t>(k)] =
            std::exp(static_cast<float>(2 * k) *
                     (-ln_10000 / static_cast<float>(d_model)));
    }
    for (int i = 0; i < pos_len; ++i) {
        const float pos = static_cast<float>((T_enc - 1) - i);
        float * row = pos_buf.data() + static_cast<size_t>(i) * d_model;
        for (int k = 0; k < d_model / 2; ++k) {
            const float div = div_term[static_cast<size_t>(k)];
            row[2 * k]     = std::sin(pos * div);
            row[2 * k + 1] = std::cos(pos * div);
        }
    }
}

transcribe_status run(transcribe_session *      context,
                      const float *             pcm,
                      int                       n_samples,
                      const transcribe_run_params * params)
{
    if (context == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<CanaryQwenSession *>(context);
    auto * cm = static_cast<CanaryQwenModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() || !cm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    (void)params;

    if (cc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

    transcribe::debug::init();
    cc->clear_result();

    const auto & hp = cm->hparams;

    // ---- Mel frontend ----
    const int64_t t_mel_start = ggml_time_us();
    int mel_n_mels   = 0;
    int mel_n_frames = 0;
    if (auto mst = cm->mel->compute(pcm, static_cast<size_t>(n_samples),
                                    cc->mel_buf, mel_n_mels, mel_n_frames,
                                    cc->n_threads);
        mst != TRANSCRIBE_OK)
    {
        std::fprintf(stderr,
                     "canary_qwen run: MelFrontend::compute failed (%s)\n",
                     transcribe_status_string(mst));
        return mst;
    }
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    // ---- Encoder ----
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    {
        ggml_init_params ip {};
        ip.mem_size   = 32 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx == nullptr) {
            std::fprintf(stderr, "canary_qwen run: ggml_init failed (encoder)\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    EncoderBuild eb = build_encoder_graph(
        cc->compute_ctx, cm->weights, hp, mel_n_frames,
        /*kv_type=*/GGML_TYPE_COUNT,
        cc->encoder_use_flash, cm->backend.c_str());
    if (eb.graph == nullptr || eb.out == nullptr) return TRANSCRIBE_ERR_GGUF;

    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(
            cm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(cm->plan.scheduler_list.size()),
            16384, /*parallel=*/false, /*op_offload=*/true);
        if (cc->sched == nullptr) {
            std::fprintf(stderr,
                         "canary_qwen run: ggml_backend_sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        std::fprintf(stderr,
                     "canary_qwen run: sched_alloc_graph failed (encoder)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload mel.
    ggml_backend_tensor_set(eb.mel_in, cc->mel_buf.data(),
                            0, cc->mel_buf.size() * sizeof(float));
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { mel_n_mels, mel_n_frames };
        transcribe::debug::dump_host_f32(
            "enc.mel.in", cc->mel_buf.data(),
            static_cast<long long>(cc->mel_buf.size()),
            shape, 2, "frontend.mel.norm");
    }

    // Upload relative-position embedding.
    if (eb.pos_emb_in != nullptr) {
        const int d_model = hp.enc_d_model;
        const int pos_len = static_cast<int>(eb.pos_emb_in->ne[1]);
        const int T_enc   = (pos_len + 1) / 2;
        build_relpos_emb_host(cc->pos_buf, cc->pos_div_term, d_model, T_enc);
        ggml_backend_tensor_set(eb.pos_emb_in, cc->pos_buf.data(),
                                0, cc->pos_buf.size() * sizeof(float));
        if (transcribe::debug::enabled()) {
            transcribe::debug::dump_tensor("enc.pos_emb", eb.pos_emb_in,
                                           "encoder.pos_emb");
        }
    }

    apply_thread_policy(cc);

    const int64_t t_enc_start = ggml_time_us();
    if (const ggml_status gs =
            ggml_backend_sched_graph_compute(cc->sched, eb.graph);
        gs != GGML_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "canary_qwen run: encoder compute failed (%d)\n",
                     static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->t_encode_us = ggml_time_us() - t_enc_start;

    auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
        if (t != nullptr) transcribe::debug::dump_tensor(name, t, stage);
    };
    try_dump("enc.pre_encode.out",  eb.dumps.pre_encode_out, "encoder.pre_encode");
    try_dump("enc.block.0.out",     eb.dumps.block0_out,     "encoder.block0");
    if (eb.dumps.block_mid_out != nullptr) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "enc.block.%d.out", hp.enc_n_layers / 2);
        try_dump(nm, eb.dumps.block_mid_out, "encoder.block_mid");
    }
    if (eb.dumps.block_last_out != nullptr) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "enc.block.%d.out", hp.enc_n_layers - 1);
        try_dump(nm, eb.dumps.block_last_out, "encoder.block_last");
    }
    try_dump("enc.final",           eb.dumps.final_out,      "encoder.final");
    try_dump("perception.proj.out", eb.dumps.perception_out, "perception.proj.out");

    // ---- Read perception output to host ----
    const int hidden = static_cast<int>(eb.out->ne[0]);
    const int T_enc  = static_cast<int>(eb.out->ne[1]);
    cc->enc_host.resize(static_cast<size_t>(hidden) *
                        static_cast<size_t>(T_enc));
    ggml_backend_tensor_get(eb.out, cc->enc_host.data(), 0,
                            cc->enc_host.size() * sizeof(float));

    // ---- Build prompt token-id list ----
    std::vector<int32_t> prompt_ids;
    prompt_ids.reserve(cm->prompt_prefix_ids.size() + T_enc +
                       cm->prompt_suffix_ids.size());
    prompt_ids.insert(prompt_ids.end(),
                      cm->prompt_prefix_ids.begin(),
                      cm->prompt_prefix_ids.end());
    const int prefix_len = static_cast<int>(prompt_ids.size());
    for (int i = 0; i < T_enc; ++i) {
        prompt_ids.push_back(hp.audio_locator_id);
    }
    prompt_ids.insert(prompt_ids.end(),
                      cm->prompt_suffix_ids.begin(),
                      cm->prompt_suffix_ids.end());
    const int T_prompt   = static_cast<int>(prompt_ids.size());
    const int suffix_len = T_prompt - prefix_len - T_enc;

    if (T_prompt > cc->kv_cache.n_ctx) {
        std::fprintf(stderr,
                     "canary_qwen run: prompt len %d exceeds kv_n_ctx %d\n",
                     T_prompt, cc->kv_cache.n_ctx);
        return TRANSCRIBE_ERR_GGUF;
    }

    // ---- Reset KV cache, build prefill graph ----
    if (cc->kv_cache.buffer != nullptr) {
        ggml_backend_buffer_clear(cc->kv_cache.buffer, 0);
    }
    cc->kv_cache.n    = 0;
    cc->kv_cache.head = 0;

    const int64_t t_dec_start = ggml_time_us();

    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    {
        ggml_init_params ip {};
        ip.mem_size   = 32 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx == nullptr) {
            std::fprintf(stderr, "canary_qwen run: ggml_init failed (prefill)\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    const bool dumps_on   = transcribe::debug::enabled();
    const bool slice_last = !dumps_on;
    PrefillBuild pb = build_prefill_graph(
        cc->compute_ctx, cm->weights, hp,
        cc->kv_cache, T_prompt, T_enc, prefix_len, suffix_len,
        cc->decoder_use_flash, slice_last);
    if (pb.graph == nullptr || pb.out == nullptr) return TRANSCRIBE_ERR_GGUF;

    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, pb.graph)) {
        std::fprintf(stderr,
                     "canary_qwen run: sched_alloc_graph failed (prefill)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_tensor_set(pb.input_ids_in, prompt_ids.data(),
                            0, prompt_ids.size() * sizeof(int32_t));
    if (T_enc > 0 && pb.audio_in != nullptr) {
        ggml_backend_tensor_set(pb.audio_in, cc->enc_host.data(),
                                0, cc->enc_host.size() * sizeof(float));
    }
    {
        std::vector<int32_t> positions(T_prompt);
        for (int i = 0; i < T_prompt; ++i) positions[i] = i;
        ggml_backend_tensor_set(pb.positions_in, positions.data(),
                                0, positions.size() * sizeof(int32_t));
    }
    {
        const ggml_fp16_t mask_zero    = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask(
            static_cast<size_t>(T_prompt) * T_prompt, mask_neg_inf);
        for (int r = 0; r < T_prompt; ++r) {
            for (int c = 0; c <= r; ++c) {
                mask[static_cast<size_t>(r) * T_prompt + c] = mask_zero;
            }
        }
        ggml_backend_tensor_set(pb.mask_in, mask.data(),
                                0, mask.size() * sizeof(ggml_fp16_t));
    }

    const bool profile_decode = (std::getenv("TRANSCRIBE_PROFILE_DECODE") != nullptr);
    int64_t t_prefill_us = 0;
    int64_t t_prefill_compute_us = 0;
    {
        const int64_t t0 = ggml_time_us();
        if (const ggml_status gs =
                ggml_backend_sched_graph_compute(cc->sched, pb.graph);
            gs != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "canary_qwen run: prefill compute failed (%d)\n",
                         static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }
        t_prefill_compute_us = ggml_time_us() - t0;
        t_prefill_us = t_prefill_compute_us;
    }

    cc->kv_cache.n    = T_prompt;
    cc->kv_cache.head = T_prompt;

    try_dump("dec.token_emb",       pb.dumps.token_emb,       "decoder.token_emb");
    try_dump("dec.audio_injected",  pb.dumps.audio_injected,  "decoder.audio_injected");
    try_dump("dec.block.0.out",     pb.dumps.block_0_out,     "decoder.block0");
    if (pb.dumps.block_mid_out != nullptr) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "dec.block.%d.out", hp.dec_n_layers / 2);
        try_dump(nm, pb.dumps.block_mid_out, "decoder.block_mid");
    }
    if (pb.dumps.block_last_out != nullptr) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "dec.block.%d.out", hp.dec_n_layers - 1);
        try_dump(nm, pb.dumps.block_last_out, "decoder.block_last");
    }
    try_dump("dec.out_before_head", pb.dumps.out_before_head, "decoder.out_before_head");
    try_dump("dec.logits_raw.gen0", pb.dumps.logits_raw,      "decoder.logits.gen0");

    // ---- Read prefill logits + first argmax ----
    const int vocab = hp.dec_vocab_size;
    std::vector<float> logits(vocab);
    ggml_backend_tensor_get(pb.out, logits.data(), 0,
                            logits.size() * sizeof(float));

    auto argmax = [&](const std::vector<float> & v) -> int32_t {
        int32_t best = 0;
        float   best_v = v[0];
        for (int32_t i = 1; i < static_cast<int32_t>(v.size()); ++i) {
            if (v[i] > best_v) { best_v = v[i]; best = i; }
        }
        return best;
    };

    std::vector<int32_t> generated_ids;
    int32_t next_tok = argmax(logits);
    generated_ids.push_back(next_tok);

    // ---- Step loop ----
    const int32_t eos_id = hp.eos_token_id;
    const int max_new = 256;
    int cur_past = T_prompt;

    int max_n_kv = 1024;
    while (max_n_kv < T_prompt + max_new) max_n_kv *= 2;
    if (max_n_kv > cc->kv_cache.n_ctx) max_n_kv = cc->kv_cache.n_ctx;

    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    {
        ggml_init_params ip {};
        ip.mem_size   = 16 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx == nullptr) {
            std::fprintf(stderr, "canary_qwen run: ggml_init failed (step)\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    StepBuild sb = build_step_graph(cc->compute_ctx, cm->weights, hp,
                                    cc->kv_cache, max_n_kv,
                                    cc->decoder_use_flash);

    if (profile_decode && sb.graph != nullptr) {
        const int n_nodes = ggml_graph_n_nodes(sb.graph);
        int op_counts[GGML_OP_COUNT] = {0};
        for (int ni = 0; ni < n_nodes; ++ni) {
            ggml_tensor * t = ggml_graph_node(sb.graph, ni);
            if (t != nullptr) op_counts[t->op] += 1;
        }
        std::fprintf(stderr, "[profile_decode] step graph: n_nodes=%d\n", n_nodes);
        int shown = 0;
        for (int o = 0; o < GGML_OP_COUNT && shown < 12; ++o) {
            if (op_counts[o] > 0) {
                std::fprintf(stderr,
                    "[profile_decode]   op %-22s x %4d\n",
                    ggml_op_name(static_cast<ggml_op>(o)), op_counts[o]);
                ++shown;
            }
        }
    }
    if (sb.graph == nullptr || sb.out == nullptr) return TRANSCRIBE_ERR_GGUF;

    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
        std::fprintf(stderr,
                     "canary_qwen run: sched_alloc_graph failed (step)\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    const ggml_fp16_t mask_zero    = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> step_mask(max_n_kv, mask_neg_inf);

    // Mid-generation tensor coverage: `dec.logits_raw.gen8` is the
    // logits for the 9th lm_head call (= step iter 7 of our step loop;
    // prefill = 1st call, iter K = (K+2)th call). Same semantics as
    // funasr_nano's gen_dump_step.
    constexpr int gen_dump_step = 7;
    int n_steps = 0;

    int64_t t_step_input_set_us = 0;
    int64_t t_step_compute_us   = 0;
    int64_t t_step_argmax_get_us = 0;
    std::vector<int64_t> per_step_compute_us;
    if (profile_decode) per_step_compute_us.reserve(64);

    while (next_tok != eos_id &&
           static_cast<int32_t>(generated_ids.size()) < max_new &&
           cur_past + 1 <= max_n_kv)
    {
        const int64_t t_in0 = profile_decode ? ggml_time_us() : 0;
        ggml_backend_tensor_set(sb.input_id_in, &next_tok, 0, sizeof(int32_t));
        const int32_t pos_val = cur_past;
        ggml_backend_tensor_set(sb.position_in, &pos_val, 0, sizeof(int32_t));
        const int64_t kv_idx_val = cur_past;
        ggml_backend_tensor_set(sb.kv_idx_in, &kv_idx_val, 0, sizeof(int64_t));

        if (cur_past == T_prompt) {
            std::fill(step_mask.begin(),
                      step_mask.begin() + cur_past + 1, mask_zero);
        } else {
            step_mask[cur_past] = mask_zero;
        }
        ggml_backend_tensor_set(sb.mask_in, step_mask.data(), 0,
                                static_cast<size_t>(max_n_kv) *
                                sizeof(ggml_fp16_t));
        if (profile_decode) t_step_input_set_us += ggml_time_us() - t_in0;

        const int64_t t_c0 = profile_decode ? ggml_time_us() : 0;
        if (const ggml_status gs =
                ggml_backend_sched_graph_compute(cc->sched, sb.graph);
            gs != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "canary_qwen run: step compute failed (%d)\n",
                         static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }
        if (profile_decode) {
            const int64_t dt = ggml_time_us() - t_c0;
            t_step_compute_us += dt;
            per_step_compute_us.push_back(dt);
        }

        const int64_t t_a0 = profile_decode ? ggml_time_us() : 0;
        int32_t argmax_tok = 0;
        ggml_backend_tensor_get(sb.out, &argmax_tok, 0, sizeof(int32_t));
        if (profile_decode) t_step_argmax_get_us += ggml_time_us() - t_a0;
        next_tok = argmax_tok;
        generated_ids.push_back(next_tok);

        if (n_steps == gen_dump_step && transcribe::debug::enabled()) {
            try_dump("dec.logits_raw.gen8", sb.logits, "decoder.logits.gen8");
        }

        cur_past += 1;
        n_steps  += 1;
    }

    if (profile_decode) {
        const int n = n_steps;
        std::fprintf(stderr,
            "[profile_decode] T_prompt=%d max_n_kv=%d steps=%d use_flash=%d\n",
            T_prompt, max_n_kv, n, cc->decoder_use_flash ? 1 : 0);
        std::fprintf(stderr,
            "[profile_decode] prefill_compute=%.2f ms\n",
            t_prefill_compute_us / 1000.0);
        std::fprintf(stderr,
            "[profile_decode] step totals: input_set=%.2f ms compute=%.2f ms argmax_get=%.2f ms\n",
            t_step_input_set_us / 1000.0, t_step_compute_us / 1000.0,
            t_step_argmax_get_us / 1000.0);
        if (n > 0) {
            std::vector<int64_t> sorted = per_step_compute_us;
            std::sort(sorted.begin(), sorted.end());
            const auto pct = [&](double p) {
                size_t idx = static_cast<size_t>(p * (n - 1));
                if (idx >= sorted.size()) idx = sorted.size() - 1;
                return sorted[idx] / 1000.0;
            };
            std::fprintf(stderr,
                "[profile_decode] per-step compute ms: mean=%.2f p50=%.2f p90=%.2f p99=%.2f min=%.2f max=%.2f\n",
                (t_step_compute_us / 1000.0) / n,
                pct(0.50), pct(0.90), pct(0.99),
                sorted.front() / 1000.0, sorted.back() / 1000.0);
            std::fprintf(stderr,
                "[profile_decode] per-step input_set ms: mean=%.3f\n",
                (t_step_input_set_us / 1000.0) / n);
            std::fprintf(stderr,
                "[profile_decode] per-step argmax_get ms: mean=%.3f\n",
                (t_step_argmax_get_us / 1000.0) / n);
        }
    }
    (void)n_steps;

    if (!generated_ids.empty() && generated_ids.back() == eos_id) {
        generated_ids.pop_back();
    }

    std::string transcript = cm->tok.decode(
        generated_ids.data(), static_cast<int>(generated_ids.size()));

    cc->t_decode_us = ggml_time_us() - t_dec_start;

    cc->full_text   = transcript;
    cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    cc->has_result  = true;

    transcribe_session::SegmentEntry seg {};
    seg.text  = transcript;
    seg.t0_ms = 0;
    seg.t1_ms = static_cast<int64_t>(n_samples) * 1000
              / static_cast<int64_t>(hp.fe_sample_rate);
    cc->segments.push_back(std::move(seg));

    return TRANSCRIBE_OK;
}

} // namespace

extern const Arch arch = {
    /* .name             = */ "canary_qwen",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
};

} // namespace transcribe::canary_qwen
