// qwen3_asr_real_smoke_0_6b.cpp - real-model gated loader structural
// test for the Qwen3-ASR 0.6B variant.
//
// Loads a real Qwen3-ASR-0.6B GGUF (output of
// scripts/convert-qwen3_asr.py) and asserts the loader produces a
// fully populated QwenAsrModel matching the published 0.6B
// architecture. The 1.7B variant has its own paired test in
// qwen3_asr_real_smoke_1_7b.cpp with variant-specific dims.
//
// Gating:
//   - TRANSCRIBE_BUILD_REAL_MODEL_TESTS (CMake, default OFF) controls
//     whether this binary is built.
//   - At runtime, TRANSCRIBE_QWEN3_ASR_0_6B_GGUF points at the GGUF.
//     If unset, exits 77 (CTest "skipped").
//
// What we assert:
//   1. Load returns OK; arch string is "qwen3_asr"; variant string is
//      "qwen3-asr-0.6b"; backend non-empty.
//   2. QwenAsrHParams match the 0.6B variant: 18-layer encoder at
//      d_model=896 / n_heads=14 / ffn=3584, 28-layer LM at hidden=1024
//      with GQA 16/8 heads and head_dim=128, SwiGLU (silu),
//      RMSNorm eps 1e-6, RoPE theta 1e6, MRoPE [24,20,20], vocab
//      151936, tie_word_embeddings=true.
//   3. Frontend is Whisper-style: 128 mels / 16 kHz / n_fft=400 /
//      hop=160 / periodic hann / per_utterance normalize / reflect pad.
//   4. enc.conv.0.weight ne=[3, 3, 1, 480]; enc.proj2.weight ne=
//      [d_model=896, output_dim=1024]; dec.blocks.0 attn Q/K/V/O shapes
//      match GQA 16/8 with head_dim=128.
//   5. No lm_head tensor slot (tied to dec.token_embd.weight).
//   6. Capabilities: native_sample_rate=16000, timestamps=NONE,
//      n_languages matches the 30-language roster the converter
//      writes, supports_language_detect true, supports_translate false.

#include "arch/qwen3_asr/qwen3_asr.h"
#include "arch/qwen3_asr/weights.h"
#include "ggml.h"
#include "transcribe-model.h"
#include "transcribe.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_EQ_INT(actual, expected)                                                                           \
    do {                                                                                                         \
        const long long _a = static_cast<long long>(actual);                                                     \
        const long long _e = static_cast<long long>(expected);                                                   \
        if (_a != _e) {                                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s = %lld, expected %lld\n", __FILE__, __LINE__, #actual, _a, _e); \
            ++g_failures;                                                                                        \
        }                                                                                                        \
    } while (0)

#define CHECK_STR_EQ(a, b)                                                                                        \
    do {                                                                                                          \
        const std::string _av = (a);                                                                              \
        const std::string _bv = (b);                                                                              \
        if (_av != _bv) {                                                                                         \
            std::fprintf(stderr, "FAIL %s:%d: \"%s\" != \"%s\"\n", __FILE__, __LINE__, _av.c_str(), _bv.c_str()); \
            ++g_failures;                                                                                         \
        }                                                                                                         \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

const transcribe::qwen3_asr::QwenAsrModel * qwen_view(const struct transcribe_model * m) {
    return static_cast<const transcribe::qwen3_asr::QwenAsrModel *>(m);
}

}  // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_QWEN3_ASR_0_6B_GGUF");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "qwen3_asr_real_smoke_0_6b: TRANSCRIBE_QWEN3_ASR_0_6B_GGUF "
                     "not set; skipping.\n");
        return 77;
    }
    const std::string fixture = env;
    if (!file_exists(fixture)) {
        std::fprintf(stderr, "qwen3_asr_real_smoke_0_6b: file not found: %s\n", fixture.c_str());
        return 77;
    }

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    struct transcribe_model * model = nullptr;
    const transcribe_status   st    = transcribe_model_load_file(fixture.c_str(), &mp, &model);
    if (st != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL load: %s\n", transcribe_status_string(st));
        return EXIT_FAILURE;
    }

    CHECK_STR_EQ(transcribe_model_arch_string(model), "qwen3_asr");
    CHECK_STR_EQ(transcribe_model_variant_string(model), "qwen3-asr-0.6b");
    {
        const char * backend = transcribe_model_backend(model);
        CHECK(backend != nullptr && backend[0] != '\0');
        if (backend) {
            std::fprintf(stderr, "qwen3_asr_real_smoke_0_6b: backend=%s\n", backend);
        }
    }

    transcribe_capabilities caps_buf;
    transcribe_capabilities_init(&caps_buf);
    const bool                      caps_ok = transcribe_model_get_capabilities(model, &caps_buf) == TRANSCRIBE_OK;
    const transcribe_capabilities * caps    = caps_ok ? &caps_buf : nullptr;
    CHECK(caps != nullptr);
    if (caps != nullptr) {
        CHECK_EQ_INT(caps->native_sample_rate, 16000);
        CHECK(caps->supports_translate == false);
        CHECK(caps->supports_language_detect == true);
        // 30-language roster the Qwen3-ASR converter writes (see
        // scripts/convert-qwen3_asr.py / intake.json). Matches the
        // manifest.
        CHECK_EQ_INT(caps->n_languages, 30);
        CHECK(caps->languages != nullptr);
        CHECK_EQ_INT(caps->max_timestamp_kind, TRANSCRIBE_TIMESTAMPS_NONE);
    }

    const auto * qm = qwen_view(model);
    CHECK(qm != nullptr);
    const auto & hp = qm->hparams;

    // Encoder (Qwen3ASRAudioEncoder, 18 layers, d_model 896).
    CHECK_EQ_INT(hp.enc_n_layers, 18);
    CHECK_EQ_INT(hp.enc_d_model, 896);
    CHECK_EQ_INT(hp.enc_n_heads, 14);
    CHECK_EQ_INT(hp.enc_ffn_dim, 3584);
    CHECK_EQ_INT(hp.enc_num_mel_bins, 128);
    CHECK_EQ_INT(hp.enc_downsample_hidden, 480);
    CHECK_EQ_INT(hp.enc_output_dim, 1024);
    CHECK_EQ_INT(hp.enc_max_source_positions, 1500);
    CHECK_EQ_INT(hp.enc_n_window, 50);
    CHECK_EQ_INT(hp.enc_n_window_infer, 800);
    CHECK_STR_EQ(hp.enc_activation, "gelu");

    // LM (28-layer Qwen3 causal LM, GQA 16/8).
    CHECK_EQ_INT(hp.dec_n_layers, 28);
    CHECK_EQ_INT(hp.dec_hidden, 1024);
    CHECK_EQ_INT(hp.dec_intermediate, 3072);
    CHECK_EQ_INT(hp.dec_n_heads, 16);
    CHECK_EQ_INT(hp.dec_n_kv_heads, 8);
    CHECK_EQ_INT(hp.dec_head_dim, 128);
    CHECK_STR_EQ(hp.dec_hidden_act, "silu");
    CHECK(hp.dec_rms_norm_eps == 1e-6f);
    CHECK(hp.dec_rope_theta == 1000000.0f);
    CHECK_EQ_INT(hp.dec_rope_mrope_section_t, 24);
    CHECK_EQ_INT(hp.dec_rope_mrope_section_h, 20);
    CHECK_EQ_INT(hp.dec_rope_mrope_section_w, 20);
    CHECK(hp.dec_rope_mrope_interleaved);
    CHECK(hp.dec_tie_word_embeddings);
    CHECK_EQ_INT(hp.dec_vocab_size, 151936);

    // Audio-token injection ids.
    CHECK_EQ_INT(hp.audio_start_token_id, 151669);
    CHECK_EQ_INT(hp.audio_end_token_id, 151670);
    CHECK_EQ_INT(hp.audio_token_id, 151676);

    // Frontend (Whisper).
    CHECK_STR_EQ(hp.fe_type, "mel");
    CHECK_EQ_INT(hp.fe_num_mels, 128);
    CHECK_EQ_INT(hp.fe_sample_rate, 16000);
    CHECK_EQ_INT(hp.fe_n_fft, 400);
    CHECK_EQ_INT(hp.fe_win_length, 400);
    CHECK_EQ_INT(hp.fe_hop_length, 160);
    CHECK_STR_EQ(hp.fe_window, "hann_periodic");
    CHECK_STR_EQ(hp.fe_normalize, "per_utterance");
    CHECK_STR_EQ(hp.fe_pad_mode, "reflect");
    CHECK(hp.fe_dither == 0.0f);
    CHECK(hp.fe_pre_emphasis == 0.0f);

    CHECK_EQ_INT(qm->weights.enc_blocks.size(), 18);
    CHECK_EQ_INT(qm->weights.dec_blocks.size(), 28);

    // Representative tensor shapes.
    {
        const auto * t = qm->weights.enc_subsample.conv0_w;
        CHECK(t != nullptr);
        if (t) {
            CHECK_EQ_INT(t->ne[0], 3);
            CHECK_EQ_INT(t->ne[1], 3);
            CHECK_EQ_INT(t->ne[2], 1);
            CHECK_EQ_INT(t->ne[3], 480);
        }
    }
    {
        // Post-encoder proj2: [d_model, output_dim].
        const auto * t = qm->weights.enc_head.proj2_w;
        CHECK(t != nullptr);
        if (t) {
            CHECK_EQ_INT(t->ne[0], hp.enc_d_model);
            CHECK_EQ_INT(t->ne[1], hp.enc_output_dim);
        }
    }
    {
        // Decoder embedding table: [hidden, vocab].
        const auto * t = qm->weights.dec_embed.token_w;
        CHECK(t != nullptr);
        if (t) {
            CHECK_EQ_INT(t->ne[0], hp.dec_hidden);
            CHECK_EQ_INT(t->ne[1], hp.dec_vocab_size);
        }
    }
    {
        // GQA Q proj: [hidden, n_heads * head_dim = 2048].
        const auto * t = qm->weights.dec_blocks[0].attn_q_w;
        CHECK(t != nullptr);
        if (t) {
            CHECK_EQ_INT(t->ne[0], hp.dec_hidden);
            CHECK_EQ_INT(t->ne[1], hp.dec_n_heads * hp.dec_head_dim);
        }
    }
    {
        // GQA K proj: [hidden, n_kv_heads * head_dim = 1024].
        const auto * t = qm->weights.dec_blocks[0].attn_k_w;
        CHECK(t != nullptr);
        if (t) {
            CHECK_EQ_INT(t->ne[0], hp.dec_hidden);
            CHECK_EQ_INT(t->ne[1], hp.dec_n_kv_heads * hp.dec_head_dim);
        }
    }
    {
        // Per-head q_norm / k_norm weights: [head_dim].
        const auto * q = qm->weights.dec_blocks[0].attn_q_norm;
        const auto * k = qm->weights.dec_blocks[0].attn_k_norm;
        CHECK(q != nullptr && k != nullptr);
        if (q) {
            CHECK_EQ_INT(q->ne[0], hp.dec_head_dim);
        }
        if (k) {
            CHECK_EQ_INT(k->ne[0], hp.dec_head_dim);
        }
    }

    // ---- Input-length contract (docs/input-limits.md) ----
    // qwen3_asr is a hard-context-cap family: it publishes a finite audio
    // bound, the session limits query tracks the n_ctx cap, and an over-cap
    // run is rejected with TRANSCRIBE_ERR_INPUT_TOO_LONG before decoding.
    CHECK(caps != nullptr && caps->max_audio_ms > 0);
    {
        transcribe_session_params sp;
        transcribe_session_params_init(&sp);
        struct transcribe_session * s = nullptr;

        // Default session: effective limits come from the full model context.
        CHECK(transcribe_session_init(model, &sp, &s) == TRANSCRIBE_OK);
        transcribe_session_limits lim;
        transcribe_session_limits_init(&lim);
        CHECK(transcribe_session_get_limits(s, &lim) == TRANSCRIBE_OK);
        CHECK_EQ_INT(lim.effective_n_ctx, hp.dec_max_position_embeddings);
        CHECK(lim.effective_max_audio_ms > 0);
        CHECK(lim.max_kv_bytes > 0);
        const long long default_audio_ms = (long long) lim.effective_max_audio_ms;
        const long long default_kv_bytes = (long long) lim.max_kv_bytes;
        transcribe_session_free(s);

        // A lowered n_ctx lowers the effective audio bound and KV ceiling.
        sp.n_ctx = 4096;
        s        = nullptr;
        CHECK(transcribe_session_init(model, &sp, &s) == TRANSCRIBE_OK);
        transcribe_session_limits_init(&lim);
        CHECK(transcribe_session_get_limits(s, &lim) == TRANSCRIBE_OK);
        CHECK_EQ_INT(lim.effective_n_ctx, 4096);
        CHECK(lim.effective_max_audio_ms > 0);
        CHECK((long long) lim.effective_max_audio_ms < default_audio_ms);
        CHECK((long long) lim.max_kv_bytes < default_kv_bytes);
        transcribe_session_free(s);

        // With a tiny n_ctx, any audio is too long: transcribe_run rejects it
        // up front with INPUT_TOO_LONG (synthetic silence — the gate keys on
        // sample count, not content). 1 s of 16 kHz mono silence.
        sp.n_ctx = 64;
        s        = nullptr;
        CHECK(transcribe_session_init(model, &sp, &s) == TRANSCRIBE_OK);
        std::vector<float>      silence(16000, 0.0f);
        const transcribe_status rst = transcribe_run(s, silence.data(), (int) silence.size(), nullptr);
        CHECK(rst == TRANSCRIBE_ERR_INPUT_TOO_LONG);
        CHECK(transcribe_was_truncated(s) == false);  // rejected, not truncated

        // Batch parity: run_batch enforces the same contract PER UTTERANCE —
        // the whole-batch call still returns OK, and each over-cap utterance
        // carries INPUT_TOO_LONG in its per-utterance status.
        std::vector<float> sil2    = silence;
        const float *      pcms[2] = { silence.data(), sil2.data() };
        const int          lens[2] = { (int) silence.size(), (int) sil2.size() };
        CHECK(transcribe_run_batch(s, pcms, lens, 2, nullptr) == TRANSCRIBE_OK);
        CHECK_EQ_INT(transcribe_batch_n_results(s), 2);
        CHECK(transcribe_batch_status(s, 0) == TRANSCRIBE_ERR_INPUT_TOO_LONG);
        CHECK(transcribe_batch_status(s, 1) == TRANSCRIBE_ERR_INPUT_TOO_LONG);
        transcribe_session_free(s);

        // max_kv_bytes tracks the session's kv_type exactly: AUTO and F16 both
        // resolve to an f16 KV cache (2 bytes/element); explicit F32 is 4
        // bytes/element, so its byte ceiling is exactly double. The generic
        // transcribe_session_get_limits() reads session->kv_type to compute
        // this — this locks that in so a regression can't silently report the
        // f16 size for an f32 session (or vice-versa).
        {
            auto kv_bytes_for = [&](transcribe_kv_type t) -> long long {
                transcribe_session_params spk;
                transcribe_session_params_init(&spk);
                spk.kv_type                    = t;
                struct transcribe_session * sk = nullptr;
                CHECK(transcribe_session_init(model, &spk, &sk) == TRANSCRIBE_OK);
                transcribe_session_limits lk;
                transcribe_session_limits_init(&lk);
                CHECK(transcribe_session_get_limits(sk, &lk) == TRANSCRIBE_OK);
                const long long b = (long long) lk.max_kv_bytes;
                transcribe_session_free(sk);
                return b;
            };
            const long long kv_f16  = kv_bytes_for(TRANSCRIBE_KV_TYPE_F16);
            const long long kv_auto = kv_bytes_for(TRANSCRIBE_KV_TYPE_AUTO);
            const long long kv_f32  = kv_bytes_for(TRANSCRIBE_KV_TYPE_F32);
            CHECK(kv_f16 > 0);
            CHECK_EQ_INT(kv_auto, kv_f16);     // AUTO == f16 for the KV cache
            CHECK_EQ_INT(kv_f32, kv_f16 * 2);  // f32 is exactly 2x f16
        }
    }

    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "qwen3_asr_real_smoke_0_6b: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "qwen3_asr_real_smoke_0_6b: ok\n");
    return EXIT_SUCCESS;
}
