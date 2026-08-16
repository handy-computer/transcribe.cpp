// arch/crisperwhisper/model.cpp - CrisperWhisper 2.0 family handler.
//
// The graph is stock Whisper and comes from src/whisper_graph/ unchanged.
// Everything in this file is the decode harness above it, which shares
// nothing with the `whisper` family:
//
//   * the prompt is "[verbatim_1]…[verbatim_5]" (or [intended_N]) BEFORE the
//     Whisper prefix, with no <|startofprev|> wrapper;
//   * <|notimestamps|> is always forced and no timestamp token is ever
//     decoded, so none of whisper's timestamp rules exist here;
//   * the 31 added tokens are all `special: false`, so prompt artifacts are
//     stripped by explicit id while the 15 vocal-event tokens are kept —
//     they are the verbatim output this family exists to produce;
//   * long-form is <ctx> conditional continuation over 30 s windows at a
//     26 s stride, not timestamp-token stitching.
//
// Reference: crisperwhisper 2.0.2, transformers backend. See
// reports/porting/crisperwhisper/forward-map.md for the row-by-row map.

#include "crisperwhisper.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "transcribe-arch.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-env.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe/crisperwhisper.h"
#include "whisper_graph/decoder.h"
#include "whisper_graph/encoder.h"
#include "word_timing.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace transcribe::crisperwhisper {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model, CwModel>);
static_assert(std::is_base_of_v<transcribe_session, CwSession>);

namespace wg = transcribe::whisper_graph;

CwModel::~CwModel() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (backend_buffer != nullptr) {
        safe_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary      = nullptr;
    plan.primary_kind = transcribe::BackendKind::Unknown;
}

CwSession::~CwSession() {
    kv_cache.free();
    enc_out.free();
    if (sched != nullptr) {
        safe_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
    compute_ctx_size = 0;
}

namespace {

constexpr const char k_default_variant[] = "crisperwhisper-2.0-small";

// crisperwhisper/model.py::transcribe(max_new_tokens=256).
constexpr int k_default_max_new_tokens = 256;

constexpr int k_sample_rate = 16000;

bool ensure_compute_ctx(CwSession * cc, size_t mem) {
    if (cc->compute_ctx != nullptr) {
        if (cc->compute_ctx_size >= mem) {
            ggml_reset(cc->compute_ctx);
            return true;
        }
        ggml_free(cc->compute_ctx);
        cc->compute_ctx      = nullptr;
        cc->compute_ctx_size = 0;
    }
    ggml_init_params p{};
    p.mem_size      = mem;
    p.mem_buffer    = nullptr;
    p.no_alloc      = true;
    cc->compute_ctx = ggml_init(p);
    if (cc->compute_ctx == nullptr) {
        return false;
    }
    cc->compute_ctx_size = mem;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

transcribe_status cw_load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = ggml_time_us();

    auto m       = std::make_unique<CwModel>();
    m->arch      = &arch;
    m->t_load_us = 0;
    m->variant   = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    if (const transcribe_status st = read_cw_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = read_cw_contract(loader.gguf(), m->hparams, m->contract); st != TRANSCRIBE_OK) {
        return st;
    }

    // Tokenizer: GPT-2 byte-level BPE, base vocab byte-identical to
    // openai/whisper-small's. DO NOT overwrite vocab_size from the tokenizer;
    // the decoder vocab (51896) intentionally exceeds the tokens array.
    if (const transcribe_status st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) {
        return st;
    }
    if (m->tok.pretokenizer() != "gpt2") {
        m->tok.set_pretokenizer("gpt2");
    }

    // Mel frontend: filterbank + Hann window baked into the GGUF. Unlike
    // whisper tiny..large-v2, CrisperWhisper's preprocessor_config.json ships
    // no mel_filters array, so the converter rebuilt it exactly as
    // WhisperFeatureExtractor would and serialized the result.
    {
        using R = transcribe::load_common::ReadF32Result;
        std::vector<float> fb_buf;
        std::vector<float> win_buf;

        const size_t fb_elems =
            static_cast<size_t>(m->hparams.fe_num_mels) * static_cast<size_t>(m->hparams.fe_n_fft / 2 + 1);
        const auto fb_rc = transcribe::load_common::read_f32_tensor_checked(
            loader.gguf(), loader.path(), "frontend.mel_filterbank", fb_elems, k_family_tag, fb_buf);
        if (fb_rc != R::Ok && fb_rc != R::Absent) {
            return TRANSCRIBE_ERR_GGUF;
        }
        const size_t win_elems = static_cast<size_t>(m->hparams.fe_n_fft);
        const auto   win_rc    = transcribe::load_common::read_f32_tensor_checked(
            loader.gguf(), loader.path(), "frontend.window", win_elems, k_family_tag, win_buf);
        if (win_rc != R::Ok && win_rc != R::Absent) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (const transcribe_status st = install_cw_mel(m->hparams, std::move(fb_buf), std::move(win_buf), m->mel);
            st != TRANSCRIBE_OK) {
            return st;
        }
    }

    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (const transcribe_status st = build_cw_weights(m->ctx_meta, m->hparams, m->weights); st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req = (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, (params != nullptr) ? params->gpu_device : 0, k_family_tag, m->plan);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (weights_buffer == nullptr) {
        gguf_free(gguf_data);
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: ggml_backend_alloc_ctx_tensors failed", k_family_tag);
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (const transcribe_status st =
            transcribe::load_common::stream_tensor_data(loader.path(), gguf_data, m->ctx_meta, k_family_tag);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m); st != TRANSCRIBE_OK) {
        return st;
    }

    // Resolve <|lang|> token ids up front so a vocab drift surfaces at load.
    // Every shipped 2.0 checkpoint is multilingual (there is no .en variant),
    // but gate on the capability anyway rather than assuming.
    m->lang_codes.clear();
    m->lang_token_ids.clear();
    if (m->caps.languages != nullptr && m->caps.n_languages > 0) {
        m->lang_codes.reserve(static_cast<size_t>(m->caps.n_languages));
        m->lang_token_ids.reserve(static_cast<size_t>(m->caps.n_languages));
        for (int i = 0; i < m->caps.n_languages; ++i) {
            const char * code = m->caps.languages[i];
            if (code == nullptr || code[0] == '\0') {
                continue;
            }
            m->lang_codes.emplace_back(code);
            if (!m->caps.supports_language_detect) {
                continue;
            }
            const std::string piece = std::string("<|") + code + "|>";
            const int         id    = m->tok.find(piece);
            if (id < 0) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: language '%s' has no '%s' token in tokenizer vocab",
                        k_family_tag, code, piece.c_str());
                return TRANSCRIBE_ERR_GGUF;
            }
            m->lang_token_ids.push_back(static_cast<int32_t>(id));
        }
    }

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status cw_init_context(transcribe_model *                model,
                                  const transcribe_session_params * params,
                                  transcribe_session **             out_ctx) {
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto cc       = std::make_unique<CwSession>();
    cc->model     = model;
    cc->n_threads = params->n_threads;
    cc->kv_type   = params->kv_type;

    cc->encoder_use_flash = true;
    cc->decoder_use_flash = true;
    transcribe::flash::apply_env_overrides(cc->encoder_use_flash, cc->decoder_use_flash);

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// run helpers
// ---------------------------------------------------------------------------

namespace {

// Reference-mel injection for numerical validation, mirroring whisper's
// TRANSCRIBE_MEL_FROM_REF. Isolates encoder/decoder drift from frontend drift;
// production inference always uses the C++ mel.
transcribe_status load_mel_from_ref(const char * ref_dir, int n_mels, int n_mel_frames, std::vector<float> & out) {
    std::string path = std::string(ref_dir) + "/enc.mel.in.f32";
    std::FILE * f    = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: TRANSCRIBE_MEL_FROM_REF set but %s is unreadable", k_family_tag,
                path.c_str());
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const size_t want = static_cast<size_t>(n_mels) * static_cast<size_t>(n_mel_frames);
    out.assign(want, 0.0f);
    const size_t got = std::fread(out.data(), sizeof(float), want, f);
    std::fclose(f);
    if (got != want) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: reference mel %s has %zu floats, expected %zu", k_family_tag,
                path.c_str(), got, want);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    return TRANSCRIBE_OK;
}

// Run the encoder on one mel window; output lands in the backend-resident
// cc->enc_out.tensor that the cross-KV graph reads via a view.
transcribe_status run_encoder_on_window(CwSession *   cc,
                                        CwModel *     cm,
                                        const float * mel_data,
                                        int           n_mels,
                                        int           n_mel_frames,
                                        bool          allow_dumps,
                                        int &         out_T_enc) {
    if (!ensure_compute_ctx(cc, 8 * 1024 * 1024)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: ensure_compute_ctx (encoder) failed", k_family_tag);
        return TRANSCRIBE_ERR_GGUF;
    }

    wg::EncoderBuild eb = wg::build_encoder_graph(cc->compute_ctx, cm->weights, cm->hparams, n_mel_frames,
                                                  cc->encoder_use_flash, cm->backend.c_str());
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    {
        const int d_enc_g = static_cast<int>(eb.out->ne[0]);
        const int T_enc_g = static_cast<int>(eb.out->ne[1]);
        if (cc->enc_out.tensor == nullptr || cc->enc_out.d_model != d_enc_g || cc->enc_out.T_enc != T_enc_g) {
            if (!enc_out_init(cc->enc_out, cm->plan.primary, d_enc_g, T_enc_g)) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: enc_out_init failed", k_family_tag);
                return TRANSCRIBE_ERR_GGUF;
            }
        }
        ggml_tensor * enc_out_view =
            ggml_view_2d(cc->compute_ctx, cc->enc_out.tensor, d_enc_g, T_enc_g, cc->enc_out.tensor->nb[1], 0);
        ggml_build_forward_expand(eb.graph, ggml_cpy(cc->compute_ctx, eb.out, enc_out_view));
    }

    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(cm->plan.scheduler_list.size()), 16384, false, true);
        if (cc->sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: ggml_backend_sched_new failed", k_family_tag);
            return TRANSCRIBE_ERR_GGUF;
        }
        transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, eb.graph)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: sched_alloc_graph failed (encoder)", k_family_tag);
        return TRANSCRIBE_ERR_GGUF;
    }

    const size_t mel_bytes = static_cast<size_t>(n_mels) * static_cast<size_t>(n_mel_frames) * sizeof(float);
    ggml_backend_tensor_set(eb.mel_in, mel_data, 0, mel_bytes);

    if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, eb.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: encoder graph compute failed (%d)", k_family_tag,
                static_cast<int>(gs));
        return TRANSCRIBE_ERR_GGUF;
    }

    if (allow_dumps) {
        auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
            if (t != nullptr) {
                transcribe::debug::dump_tensor(name, t, stage);
            }
        };
        try_dump("enc.mel.in", eb.dumps.mel_in, "encoder.mel");
        try_dump("enc.conv1.out", eb.dumps.conv1_out, "encoder.conv1");
        try_dump("enc.conv2.out", eb.dumps.conv2_out, "encoder.conv2");
        try_dump("enc.pos_emb", eb.dumps.pos_emb, "encoder.pos_emb");
        try_dump("enc.embed.out", eb.dumps.embed_out, "encoder.embed");
        for (size_t i = 0; i < eb.dumps.block_outs.size(); ++i) {
            char bname[64], stage[64];
            std::snprintf(bname, sizeof(bname), "enc.block.%zu.out", i);
            std::snprintf(stage, sizeof(stage), "encoder.block%zu.out", i);
            try_dump(bname, eb.dumps.block_outs[i], stage);
        }
        try_dump("enc.final", eb.dumps.final_out, "encoder.final");
    }

    out_T_enc = static_cast<int>(eb.out->ne[1]);
    cc->enc_T = out_T_enc;
    return TRANSCRIBE_OK;
}

// Build the decoder prompt exactly as crisperwhisper/prompt.py does:
//
//   encode("[verbatim_1]…[verbatim_N]" [+ " <ctx> " + context + " <ectx>"])
//   + [<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>]
//
// The mode tags lead, with no <|startofprev|> wrapper — this is the single
// biggest divergence from the whisper family, and getting it wrong produces
// plausible-looking output from the wrong conditioning.
//
// `context_text` is the last N confirmed words at a long-form seam, or empty.
// Hotwords are deliberately unsupported (Pro-checkpoint feature; the open
// weights were never trained with them and the reference warns of degradation).
std::vector<int32_t> build_prompt(const CwModel & cm, CwMode mode, int32_t lang_id, const std::string & context_text) {
    const CwDecodeContract & ct = cm.contract;

    std::vector<int32_t>         ids;
    const std::vector<int32_t> & tags = ct.tags_for(mode);
    ids.insert(ids.end(), tags.begin(), tags.end());

    if (!context_text.empty()) {
        // The reference builds ONE string,
        //     "{mode tags} <ctx> {context} <ectx>"
        // and lets the HF tokenizer split it on the added tokens. The spaces
        // around the markers therefore survive as their own space tokens:
        //
        //   [verbatim_1..5] ' ' <ctx> ' was' ' sort' ... ' Google' ' ' <ectx>
        //
        // Our tokenizer does not split on added tokens, so the sequence is
        // assembled by hand. Emitting <ctx> straight after the tag block (no
        // space token) makes the prompt two tokens shorter than the
        // reference's, which is enough to change what the model does: on
        // whole-earth.wav it moved the window-1 decode off the reference's
        // early-EOT stop and produced a different, longer transcript.
        std::vector<int32_t> space_ids;
        (void) cm.tok.encode(" ", space_ids);

        ids.insert(ids.end(), space_ids.begin(), space_ids.end());
        ids.push_back(ct.ctx_id);

        std::vector<int32_t> ctx_ids;
        if (cm.tok.encode(" " + context_text, ctx_ids) == TRANSCRIBE_OK) {
            ids.insert(ids.end(), ctx_ids.begin(), ctx_ids.end());
        }

        ids.insert(ids.end(), space_ids.begin(), space_ids.end());
        ids.push_back(ct.ectx_id);
    }

    // Whisper prefix. <|notimestamps|> is unconditional for this family.
    ids.push_back(cm.hparams.sot_token_id >= 0 ? cm.hparams.sot_token_id : cm.hparams.decoder_start_token_id);
    if (lang_id >= 0) {
        ids.push_back(lang_id);
    }
    if (cm.hparams.transcribe_token_id >= 0) {
        ids.push_back(cm.hparams.transcribe_token_id);
    }
    ids.push_back(cm.hparams.no_timestamps_token_id);
    return ids;
}

// Strip prompt artifacts at the id level, mirroring
// crisperwhisper/prompt.py::strip_prompt_artifacts. The paired markers remove
// their whole span (the reference uses non-greedy regex over the decoded
// text), mode tags remove just themselves, and the 15 vocal-event tokens are
// deliberately untouched: they ARE the verbatim output.
std::vector<int32_t> strip_prompt_artifacts(const CwDecodeContract & ct, const std::vector<int32_t> & ids) {
    struct Span {
        int32_t open;
        int32_t close;
    };

    const Span spans[] = {
        { ct.htx_id, ct.ehtx_id },
        { ct.vtx_id, ct.evtx_id },
        { ct.ctx_id, ct.ectx_id },
    };

    std::vector<int32_t> out;
    out.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t id = ids[i];

        bool consumed_span = false;
        for (const Span & s : spans) {
            if (id != s.open) {
                continue;
            }
            // Drop through the matching close (inclusive). An unmatched open
            // swallows the rest, matching the reference's `<sot>.*` fallback
            // behaviour of dropping a dangling opener's tail.
            size_t j = i + 1;
            while (j < ids.size() && ids[j] != s.close) {
                ++j;
            }
            i             = (j < ids.size()) ? j : ids.size() - 1;
            consumed_span = true;
            break;
        }
        if (consumed_span) {
            continue;
        }

        if (ct.is_prompt_artifact(id)) {
            continue;  // mode tags and stray markers
        }
        out.push_back(id);
    }
    return out;
}

// Collapse runs of whitespace and trim, matching the reference's
// `" ".join(text.split()).strip()` tail.
std::string normalize_ws(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    bool pending_space = false;
    bool wrote_any     = false;
    for (const char c : s) {
        const bool is_ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v');
        if (is_ws) {
            pending_space = wrote_any;
            continue;
        }
        if (pending_space) {
            out.push_back(' ');
            pending_space = false;
        }
        out.push_back(c);
        wrote_any = true;
    }
    return out;
}

// Greedy argmax over a logits row, honouring the suppression list.
//
// Suppression semantics follow the reference exactly:
//   * `suppress_tokens` from generation_config is applied at EVERY step;
//   * `begin_suppress_tokens` is NOT applied at all. TransformersEngine
//     explicitly clears it at construction ("Match the CT2 backend's
//     first-token behaviour... leaving HF's begin-suppression active would
//     forbid EOT as the first token and skew the first-step distribution").
//     Applying it here would silently diverge from the oracle on step 0.
int argmax_with_suppression(const float * logits, int vocab, const std::vector<int32_t> & suppress) {
    std::vector<bool> banned(static_cast<size_t>(vocab), false);
    for (const int32_t t : suppress) {
        if (t >= 0 && t < vocab) {
            banned[static_cast<size_t>(t)] = true;
        }
    }
    int   best    = -1;
    float best_lp = -INFINITY;
    for (int i = 0; i < vocab; ++i) {
        if (banned[static_cast<size_t>(i)]) {
            continue;
        }
        if (logits[i] > best_lp) {
            best_lp = logits[i];
            best    = i;
        }
    }
    return best;
}

// One decode pass over the already-encoded window: prompt prefill, then greedy
// steps until EOT or max_new_tokens. Returns the generated ids (prompt
// excluded), matching the reference's `_run_generate` contract.
transcribe_status decode_window(CwSession *                  cc,
                                CwModel *                    cm,
                                const std::vector<int32_t> & prompt_ids,
                                int                          T_enc,
                                int                          max_new_tokens,
                                bool                         emit_dumps,
                                bool                         capture_align,
                                std::vector<int32_t> &       out_gen,
                                std::vector<int32_t> &       out_align_ids) {
    out_gen.clear();
    out_align_ids.clear();
    cc->cross_attn.clear();
    cc->cross_attn_rows   = 0;
    cc->cross_attn_frames = 0;

    // Alignment-head capture forces the listed cross-attention layers onto the
    // manual softmax path. Row k of cc->cross_attn is the head-averaged
    // attention that predicted generated token k, matching the reference's
    // _stack_step_attention 1-to-1 convention (prompt pass -> row 0, then one
    // row per step, including the step that emits EOT).
    const wg::AlignHeads * heads = nullptr;
    wg::AlignHeads         heads_storage;
    if (capture_align) {
        for (const auto & lh : cm->contract.alignment_heads) {
            heads_storage.emplace_back(lh.first, lh.second);
        }
        heads = &heads_storage;
    }
    // One captured row per PREDICTED token, pushed together so the two stay
    // 1-to-1 by construction. The reference's attention pass ends on the row
    // that predicts EOT, so its matrix is one row longer than the generated
    // text; keeping that row (its piece is special, so word grouping ignores
    // it) is what makes dec.xattn.align shape-match the oracle.
    auto capture_row = [&](ggml_tensor * row, int32_t predicted) {
        out_align_ids.push_back(predicted);
        if (row == nullptr) {
            return;
        }
        const int    n   = static_cast<int>(row->ne[0]);
        const size_t off = cc->cross_attn.size();
        cc->cross_attn.resize(off + static_cast<size_t>(n));
        ggml_backend_tensor_get(row, cc->cross_attn.data() + off, 0, static_cast<size_t>(n) * sizeof(float));
        cc->cross_attn_rows += 1;
        cc->cross_attn_frames = n;
    };

    const CwHParams & hp         = cm->hparams;
    const int         vocab_size = hp.dec_vocab_size;
    const int         n_layers   = hp.dec_n_layers;
    const int         seq_len    = static_cast<int>(prompt_ids.size());
    const int         eos_id     = hp.decoder_start_token_id >= 0 ? cm->tok.find("<|endoftext|>") : -1;

    if (seq_len <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Decoder budget: the learned positional table is the hard ceiling, and
    // the prompt + <ctx> block already consumed part of it (intake risk 7).
    const int n_ctx_decoder = hp.dec_max_target_positions;
    if (seq_len >= n_ctx_decoder) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: prompt (%d tokens) exceeds decoder context %d", k_family_tag,
                seq_len, n_ctx_decoder);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int budget = std::min(max_new_tokens, n_ctx_decoder - seq_len);

    // ---- KV cache ----
    {
        ggml_type kv_type_g;
        if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) {
            kv_type_g = GGML_TYPE_F32;
        } else if (cc->kv_type == TRANSCRIBE_KV_TYPE_F16) {
            kv_type_g = GGML_TYPE_F16;
        } else {
            // AUTO: match the cache dtype to the weight dtype. CrisperWhisper
            // ships BF16 weights, which take the F16 branch; an F32 reference
            // GGUF would get a lossless F32 cache.
            const ggml_tensor * probe = !cm->weights.dec_blocks.empty() ? cm->weights.dec_blocks[0].self_q_w : nullptr;
            kv_type_g = (probe != nullptr && probe->type == GGML_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
        }
        cc->kv_cache.free();
        if (!kv_cache_init(cc->kv_cache, cm->plan.primary, n_ctx_decoder, T_enc, hp.dec_d_model, n_layers, kv_type_g)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: KV cache init failed", k_family_tag);
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    // ---- cross-attention K/V precompute (once per window) ----
    {
        if (!ensure_compute_ctx(cc, 8 * 1024 * 1024)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        wg::DecoderBuild cross_db =
            wg::build_cross_kv_graph(cc->compute_ctx, cm->weights, hp, cc->kv_cache, cc->enc_out.tensor, T_enc);
        if (cross_db.graph == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, cross_db.graph)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: alloc_graph failed (cross_kv)", k_family_tag);
            return TRANSCRIBE_ERR_GGUF;
        }
        if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, cross_db.graph);
            gs != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: cross_kv compute failed (%d)", k_family_tag,
                    static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }
        cc->kv_cache.cross_populated = true;
    }

    cc->kv_cache.n    = 0;
    cc->kv_cache.head = 0;

    std::vector<float> logits_row(static_cast<size_t>(vocab_size), 0.0f);
    int                next_id = -1;

    // ---- prompt prefill (n_past = 0) ----
    {
        if (!ensure_compute_ctx(cc, 16 * 1024 * 1024)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        const int        kv_pad = kv_pad_self_attn(cm->plan.primary_kind, cc->decoder_use_flash);
        wg::DecoderBuild db     = wg::build_decoder_graph_kv(cc->compute_ctx, cm->weights, hp, cc->kv_cache,
                                                             /*n_tokens=*/seq_len, /*n_past=*/0, T_enc, kv_pad,
                                                             /*skip_log_softmax=*/false, cc->decoder_use_flash, heads);
        if (db.out == nullptr || db.graph == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, db.graph)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: alloc_graph failed (prompt)", k_family_tag);
            return TRANSCRIBE_ERR_GGUF;
        }

        ggml_backend_tensor_set(db.token_ids_in, prompt_ids.data(), 0, prompt_ids.size() * sizeof(int32_t));

        std::vector<int32_t> pos_ids(static_cast<size_t>(seq_len));
        for (int i = 0; i < seq_len; ++i) {
            pos_ids[static_cast<size_t>(i)] = i;
        }
        if (ggml_tensor * pos_in = ggml_graph_get_tensor(db.graph, "dec.pos_ids"); pos_in != nullptr) {
            ggml_backend_tensor_set(pos_in, pos_ids.data(), 0, pos_ids.size() * sizeof(int32_t));
        }

        if (db.causal_mask_in != nullptr) {
            const int          n_kv_mask = static_cast<int>(db.causal_mask_in->ne[0]);
            std::vector<float> mask(static_cast<size_t>(n_kv_mask) * static_cast<size_t>(seq_len));
            for (int q = 0; q < seq_len; ++q) {
                for (int k = 0; k < n_kv_mask; ++k) {
                    mask[static_cast<size_t>(q) * n_kv_mask + k] = (k < seq_len && k <= q) ? 0.0f : -1e9f;
                }
            }
            ggml_backend_tensor_set(db.causal_mask_in, mask.data(), 0, mask.size() * sizeof(float));
        }
        if (db.cross_mask_in != nullptr) {
            const int          n_kv_cross = static_cast<int>(db.cross_mask_in->ne[0]);
            std::vector<float> mask(static_cast<size_t>(n_kv_cross) * static_cast<size_t>(seq_len));
            for (int q = 0; q < seq_len; ++q) {
                for (int k = 0; k < n_kv_cross; ++k) {
                    mask[static_cast<size_t>(q) * n_kv_cross + k] = (k < T_enc) ? 0.0f : -1e9f;
                }
            }
            ggml_backend_tensor_set(db.cross_mask_in, mask.data(), 0, mask.size() * sizeof(float));
        }

        if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, db.graph); gs != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: prompt compute failed (%d)", k_family_tag,
                    static_cast<int>(gs));
            return TRANSCRIBE_ERR_GGUF;
        }
        cc->kv_cache.n    = seq_len;
        cc->kv_cache.head = seq_len;

        if (emit_dumps) {
            auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
                if (t != nullptr) {
                    transcribe::debug::dump_tensor(name, t, stage);
                }
            };
            try_dump("dec.token_emb", db.dumps.token_emb, "decoder.embedding");
            try_dump("dec.pos_emb", db.dumps.pos_emb, "decoder.position_embedding");
            try_dump("dec.embed_sum", db.dumps.embed_sum, "decoder.embed_sum");
            for (size_t i = 0; i < db.dumps.block_outs.size(); ++i) {
                char bname[64], stage[64];
                std::snprintf(bname, sizeof(bname), "dec.block.%zu.out", i);
                std::snprintf(stage, sizeof(stage), "decoder.block%zu.out", i);
                try_dump(bname, db.dumps.block_outs[i], stage);
            }
            try_dump("dec.out_before_head", db.dumps.out_before_head, "decoder.output_before_head");
            try_dump("dec.logits_raw", db.dumps.logits_raw, "decoder.logits_raw");
            try_dump("dec.logits", db.dumps.logits, "decoder.logits");
        }

        // Last prompt row predicts the first generated token.
        ggml_tensor * src        = db.dumps.logits_raw != nullptr ? db.dumps.logits_raw : db.out;
        const size_t  row_bytes  = static_cast<size_t>(vocab_size) * sizeof(float);
        const size_t  row_offset = static_cast<size_t>(seq_len - 1) * row_bytes;
        ggml_backend_tensor_get(src, logits_row.data(), row_offset, row_bytes);
        next_id = argmax_with_suppression(logits_row.data(), vocab_size, hp.suppress_tokens);
        if (capture_align) {
            capture_row(db.cross_align, next_id);
        }
    }

    // ---- greedy step loop ----
    // Two variants, mirroring the whisper family (which shares these graph
    // builders): the static-topology step graph is a GPU optimization, and the
    // per-step build_decoder_graph_kv path is what CPU and dump runs use. The
    // static graph's ggml_set_rows KV writes + fixed max_n_kv window are not
    // exercised correctly by the CPU backend here, and debug runs need the
    // per-step graph anyway so `dec.logits_raw.gen20` is a real graph output.
    const bool primary_is_gpu = cm->plan.primary_kind != transcribe::BackendKind::Cpu &&
                                cm->plan.primary_kind != transcribe::BackendKind::Accel &&
                                cm->plan.primary_kind != transcribe::BackendKind::Unknown;
    const bool use_step_graph = primary_is_gpu && !transcribe::debug::enabled();

    // Sized to fit prompt + generated tail, padded to the next power of two
    // and capped at the decoder context.
    int max_n_kv = 256;
    while (max_n_kv < seq_len + budget) {
        max_n_kv *= 2;
    }
    max_n_kv = std::min(max_n_kv, n_ctx_decoder);

    wg::StepBuild            sb{};
    const ggml_fp16_t        mask_zero    = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t        mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> step_mask;

    if (use_step_graph) {
        if (!ensure_compute_ctx(cc, 8 * 1024 * 1024)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        sb = wg::build_step_graph(cc->compute_ctx, cm->weights, hp, cc->kv_cache, max_n_kv, T_enc,
                                  cc->decoder_use_flash, heads);
        if (sb.graph == nullptr || sb.logits_out == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: build_step_graph failed", k_family_tag);
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: sched_alloc_graph failed (step)", k_family_tag);
            return TRANSCRIBE_ERR_GGUF;
        }

        step_mask.assign(static_cast<size_t>(max_n_kv), mask_neg_inf);
        for (int p = 0; p < seq_len; ++p) {
            step_mask[static_cast<size_t>(p)] = mask_zero;
        }
        if (sb.cross_mask_in != nullptr) {
            const int          n_kv_cross = static_cast<int>(sb.cross_mask_in->ne[0]);
            std::vector<float> cross_mask(static_cast<size_t>(n_kv_cross), 0.0f);
            for (int k = T_enc; k < n_kv_cross; ++k) {
                cross_mask[static_cast<size_t>(k)] = -1e9f;
            }
            ggml_backend_tensor_set(sb.cross_mask_in, cross_mask.data(), 0, cross_mask.size() * sizeof(float));
        }
    }

    int           n_past          = seq_len;
    ggml_tensor * last_step_align = nullptr;
    for (int step = 0; step < budget; ++step) {
        if (next_id < 0 || next_id == eos_id) {
            break;
        }
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        out_gen.push_back(next_id);

        if (n_past + 1 > max_n_kv) {
            break;
        }

        const size_t row_bytes = static_cast<size_t>(vocab_size) * sizeof(float);

        if (use_step_graph) {
            int32_t tok     = next_id;
            int32_t pos_val = n_past;
            int64_t kv_val  = n_past;
            ggml_backend_tensor_set(sb.token_id_in, &tok, 0, sizeof(int32_t));
            ggml_backend_tensor_set(sb.pos_id_in, &pos_val, 0, sizeof(int32_t));
            ggml_backend_tensor_set(sb.kv_idx_in, &kv_val, 0, sizeof(int64_t));
            step_mask[static_cast<size_t>(n_past)] = mask_zero;
            ggml_backend_tensor_set(sb.mask_in, step_mask.data(), 0,
                                    static_cast<size_t>(max_n_kv) * sizeof(ggml_fp16_t));

            if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, sb.graph);
                gs != GGML_STATUS_SUCCESS) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: step compute failed (%d)", k_family_tag,
                        static_cast<int>(gs));
                return TRANSCRIBE_ERR_GGUF;
            }
            ggml_backend_tensor_get(sb.logits_out, logits_row.data(), 0, row_bytes);
            n_past += 1;
            cc->kv_cache.n    = n_past;
            cc->kv_cache.head = n_past;
        } else {
            if (!ensure_compute_ctx(cc, 4 * 1024 * 1024)) {
                return TRANSCRIBE_ERR_GGUF;
            }
            const int        kv_pad = kv_pad_self_attn(cm->plan.primary_kind, cc->decoder_use_flash);
            wg::DecoderBuild step_db =
                wg::build_decoder_graph_kv(cc->compute_ctx, cm->weights, hp, cc->kv_cache,
                                           /*n_tokens=*/1, /*n_past=*/n_past, T_enc, kv_pad,
                                           /*skip_log_softmax=*/true, cc->decoder_use_flash, heads);
            if (step_db.out == nullptr || step_db.graph == nullptr) {
                return TRANSCRIBE_ERR_GGUF;
            }
            ggml_backend_sched_reset(cc->sched);
            if (!ggml_backend_sched_alloc_graph(cc->sched, step_db.graph)) {
                return TRANSCRIBE_ERR_GGUF;
            }

            int32_t tok = next_id;
            int32_t pos = n_past;
            ggml_backend_tensor_set(step_db.token_ids_in, &tok, 0, sizeof(int32_t));
            if (ggml_tensor * pos_in = ggml_graph_get_tensor(step_db.graph, "dec.pos_ids"); pos_in != nullptr) {
                ggml_backend_tensor_set(pos_in, &pos, 0, sizeof(int32_t));
            }
            if (step_db.causal_mask_in != nullptr) {
                const int          n_kv_mask = static_cast<int>(step_db.causal_mask_in->ne[0]);
                std::vector<float> mask(static_cast<size_t>(n_kv_mask), 0.0f);
                for (int k = n_past + 1; k < n_kv_mask; ++k) {
                    mask[static_cast<size_t>(k)] = -1e9f;
                }
                ggml_backend_tensor_set(step_db.causal_mask_in, mask.data(), 0, mask.size() * sizeof(float));
            }
            if (step_db.cross_mask_in != nullptr) {
                const int          n_kv_cross = static_cast<int>(step_db.cross_mask_in->ne[0]);
                std::vector<float> mask(static_cast<size_t>(n_kv_cross), 0.0f);
                for (int k = T_enc; k < n_kv_cross; ++k) {
                    mask[static_cast<size_t>(k)] = -1e9f;
                }
                ggml_backend_tensor_set(step_db.cross_mask_in, mask.data(), 0, mask.size() * sizeof(float));
            }

            if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, step_db.graph);
                gs != GGML_STATUS_SUCCESS) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: step compute failed (%d)", k_family_tag,
                        static_cast<int>(gs));
                return TRANSCRIBE_ERR_GGUF;
            }

            n_past += 1;
            cc->kv_cache.n    = n_past;
            cc->kv_cache.head = n_past;

            // Mid-generation gate tensor: step 19 emits the logits for the
            // 20th generated token, exercising the n_past>0 KV read/write path
            // the prompt pass cannot reach. Same index as the reference dumper.
            if (emit_dumps && step == 19) {
                transcribe::debug::dump_tensor("dec.logits_raw.gen20", step_db.out, "decoder.logits_raw.gen20");
            }
            ggml_backend_tensor_get(step_db.out, logits_row.data(), 0, row_bytes);
            last_step_align = step_db.cross_align;
        }

        next_id = argmax_with_suppression(logits_row.data(), vocab_size, hp.suppress_tokens);
        if (capture_align) {
            capture_row(use_step_graph ? sb.cross_align : last_step_align, next_id);
        }
    }

    return TRANSCRIBE_OK;
}

// Populate the cross-attention K/V cache for the current window. Split out of
// decode_window so the language-detection forward can run before the real
// decode without recomputing the encoder.
transcribe_status prime_cross_kv(CwSession * cc, CwModel * cm, int T_enc) {
    const CwHParams & hp = cm->hparams;

    ggml_type kv_type_g;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) {
        kv_type_g = GGML_TYPE_F32;
    } else if (cc->kv_type == TRANSCRIBE_KV_TYPE_F16) {
        kv_type_g = GGML_TYPE_F16;
    } else {
        const ggml_tensor * probe = !cm->weights.dec_blocks.empty() ? cm->weights.dec_blocks[0].self_q_w : nullptr;
        kv_type_g                 = (probe != nullptr && probe->type == GGML_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
    }
    cc->kv_cache.free();
    if (!kv_cache_init(cc->kv_cache, cm->plan.primary, hp.dec_max_target_positions, T_enc, hp.dec_d_model,
                       hp.dec_n_layers, kv_type_g)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: KV cache init failed", k_family_tag);
        return TRANSCRIBE_ERR_BACKEND;
    }
    if (!ensure_compute_ctx(cc, 8 * 1024 * 1024)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    wg::DecoderBuild cross =
        wg::build_cross_kv_graph(cc->compute_ctx, cm->weights, hp, cc->kv_cache, cc->enc_out.tensor, T_enc);
    if (cross.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, cross.graph)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (ggml_backend_sched_graph_compute(cc->sched, cross.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->kv_cache.cross_populated = true;
    cc->kv_cache.n               = 0;
    cc->kv_cache.head            = 0;
    return TRANSCRIBE_OK;
}

// Detect the spoken language from a single decoder forward.
//
// The publisher's transcribe() cannot do this: `language` defaults to "en" and
// a language token is always forced, so there is no auto path in the author
// harness at all (see the family doc's Bridge validation note). The reference
// for this row is therefore stock transformers `generate(language=None)`,
// which runs one forward on the prefix and argmaxes over the 99 <|lang|>
// tokens. That is what this reproduces.
//
// Forcing <|en|> instead would be wrong in a way that hides itself: on
// samples/german.wav a forced <|en|> still returns correct German, so the
// no-hint path looks fine right up until it silently isn't.
//
// Requires the cross-attention cache to be populated for the current window.
int32_t detect_language(CwSession * cc, CwModel * cm, int T_enc) {
    const CwHParams & hp = cm->hparams;
    if (cm->lang_token_ids.empty() || !cc->kv_cache.cross_populated) {
        return -1;
    }

    // Detection prefix is a BARE <|startoftranscript|> — deliberately NOT the
    // mode-tag block that every other decode on this family carries. Stock
    // Whisper (and the HF bridge that is this row's reference) detects from a
    // bare <|sot|>, and the mode tags wreck the language head: measured on
    // samples/jfk.wav, `[verbatim_1..5] <|sot|>` yields en=0.207 over a
    // 0.28-logit margin against de/it/nl/sv, so Q6_K and Q5_K_M flip to `de`
    // and the run then decodes to empty text. A bare <|sot|> yields en=0.998
    // over an 8.05-logit margin, stable on every quant tier down to Q4_K_M.
    // Both prefixes agree on all nine language samples; only the margin
    // differs, and only the tagged one is close enough for noise to flip.
    std::vector<int32_t> ids;
    ids.push_back(hp.sot_token_id >= 0 ? hp.sot_token_id : hp.decoder_start_token_id);
    const int seq_len = static_cast<int>(ids.size());

    cc->kv_cache.n    = 0;
    cc->kv_cache.head = 0;

    if (!ensure_compute_ctx(cc, 16 * 1024 * 1024)) {
        return -1;
    }
    const int        kv_pad = kv_pad_self_attn(cm->plan.primary_kind, cc->decoder_use_flash);
    wg::DecoderBuild db     = wg::build_decoder_graph_kv(cc->compute_ctx, cm->weights, hp, cc->kv_cache,
                                                         /*n_tokens=*/seq_len, /*n_past=*/0, T_enc, kv_pad,
                                                         /*skip_log_softmax=*/true, cc->decoder_use_flash);
    if (db.graph == nullptr || db.out == nullptr) {
        return -1;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, db.graph)) {
        return -1;
    }
    ggml_backend_tensor_set(db.token_ids_in, ids.data(), 0, ids.size() * sizeof(int32_t));
    std::vector<int32_t> pos_ids(static_cast<size_t>(seq_len));
    for (int i = 0; i < seq_len; ++i) {
        pos_ids[static_cast<size_t>(i)] = i;
    }
    if (ggml_tensor * pos_in = ggml_graph_get_tensor(db.graph, "dec.pos_ids"); pos_in != nullptr) {
        ggml_backend_tensor_set(pos_in, pos_ids.data(), 0, pos_ids.size() * sizeof(int32_t));
    }
    if (db.causal_mask_in != nullptr) {
        const int          n_kv_mask = static_cast<int>(db.causal_mask_in->ne[0]);
        std::vector<float> mask(static_cast<size_t>(n_kv_mask) * static_cast<size_t>(seq_len));
        for (int q = 0; q < seq_len; ++q) {
            for (int k = 0; k < n_kv_mask; ++k) {
                mask[static_cast<size_t>(q) * n_kv_mask + k] = (k < seq_len && k <= q) ? 0.0f : -1e9f;
            }
        }
        ggml_backend_tensor_set(db.causal_mask_in, mask.data(), 0, mask.size() * sizeof(float));
    }
    if (db.cross_mask_in != nullptr) {
        const int          n_kv_cross = static_cast<int>(db.cross_mask_in->ne[0]);
        std::vector<float> mask(static_cast<size_t>(n_kv_cross) * static_cast<size_t>(seq_len));
        for (int q = 0; q < seq_len; ++q) {
            for (int k = 0; k < n_kv_cross; ++k) {
                mask[static_cast<size_t>(q) * n_kv_cross + k] = (k < T_enc) ? 0.0f : -1e9f;
            }
        }
        ggml_backend_tensor_set(db.cross_mask_in, mask.data(), 0, mask.size() * sizeof(float));
    }
    if (ggml_backend_sched_graph_compute(cc->sched, db.graph) != GGML_STATUS_SUCCESS) {
        return -1;
    }

    const int          vocab = hp.dec_vocab_size;
    std::vector<float> row(static_cast<size_t>(vocab), 0.0f);
    ggml_backend_tensor_get(db.out, row.data(), static_cast<size_t>(seq_len - 1) * vocab * sizeof(float),
                            static_cast<size_t>(vocab) * sizeof(float));

    // Argmax restricted to the language tokens.
    int32_t best_id = -1;
    float   best_lp = -INFINITY;
    for (const int32_t lid : cm->lang_token_ids) {
        if (lid >= 0 && lid < vocab && row[static_cast<size_t>(lid)] > best_lp) {
            best_lp = row[static_cast<size_t>(lid)];
            best_id = lid;
        }
    }

    if (transcribe::env::flag("TRANSCRIBE_CW_DETECT_TOPK")) {
        // lang_token_ids is ordered by language-token id and the converter
        // writes general.languages in that same order, so index i lines up.
        std::vector<std::pair<float, size_t>> lp;
        for (size_t i = 0; i < cm->lang_token_ids.size(); ++i) {
            const int32_t lid = cm->lang_token_ids[i];
            if (lid >= 0 && lid < vocab) {
                lp.emplace_back(row[static_cast<size_t>(lid)], i);
            }
        }
        std::sort(lp.begin(), lp.end(), [](auto & a, auto & b) { return a.first > b.first; });
        double sum = 0.0;
        for (const auto & e : lp) {
            sum += std::exp(static_cast<double>(e.first - lp[0].first));
        }
        std::fprintf(stderr, "cw-detect:");
        for (size_t i = 0; i < lp.size() && i < 5; ++i) {
            const double pr   = std::exp(static_cast<double>(lp[i].first - lp[0].first)) / sum;
            const size_t idx  = lp[i].second;
            const char * code = (idx < static_cast<size_t>(cm->caps.n_languages)) ? cm->caps.languages[idx] : "?";
            std::fprintf(stderr, " %s=%.4f", code, pr);
        }
        std::fprintf(stderr, "  margin=%.4f\n", lp.size() > 1 ? static_cast<double>(lp[0].first - lp[1].first) : 0.0);
    }

    cc->kv_cache.n    = 0;
    cc->kv_cache.head = 0;
    return best_id;
}

// Run one extra prompt pass in the OTHER mode and dump its raw logits as
// dec.intended.logits_raw.
//
// This mirrors the reference dumper, which builds both tag blocks and reports
// the sibling pass so the gate can prove the mode tags are a real control
// surface rather than decoration (verbatim vs intended prompt logits differ by
// max 26.0 / mean 5.6 on the same audio). Validation-only: it runs after the
// real decode, reuses the already-populated cross-attention cache, and its
// output is discarded.
transcribe_status dump_sibling_mode_logits(CwSession * cc, CwModel * cm, CwMode main_mode, int32_t lang_id, int T_enc) {
    const CwHParams & hp        = cm->hparams;
    const CwMode      sibling   = (main_mode == CwMode::Verbatim) ? CwMode::Intended : CwMode::Verbatim;
    const char *      dump_name = (sibling == CwMode::Intended) ? "dec.intended.logits_raw" : "dec.verbatim.logits_raw";

    const std::vector<int32_t> prompt_ids = build_prompt(*cm, sibling, lang_id, /*context_text=*/"");
    const int                  seq_len    = static_cast<int>(prompt_ids.size());
    if (seq_len <= 0 || !cc->kv_cache.cross_populated) {
        return TRANSCRIBE_OK;
    }

    // Rewind the self-attention cache only; cross-K/V is a function of the
    // encoder output and stays valid for this window.
    cc->kv_cache.n    = 0;
    cc->kv_cache.head = 0;

    if (!ensure_compute_ctx(cc, 16 * 1024 * 1024)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    const int        kv_pad = kv_pad_self_attn(cm->plan.primary_kind, cc->decoder_use_flash);
    wg::DecoderBuild db     = wg::build_decoder_graph_kv(cc->compute_ctx, cm->weights, hp, cc->kv_cache,
                                                         /*n_tokens=*/seq_len, /*n_past=*/0, T_enc, kv_pad,
                                                         /*skip_log_softmax=*/true, cc->decoder_use_flash);
    if (db.graph == nullptr || db.dumps.logits_raw == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!ggml_backend_sched_alloc_graph(cc->sched, db.graph)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_tensor_set(db.token_ids_in, prompt_ids.data(), 0, prompt_ids.size() * sizeof(int32_t));
    std::vector<int32_t> pos_ids(static_cast<size_t>(seq_len));
    for (int i = 0; i < seq_len; ++i) {
        pos_ids[static_cast<size_t>(i)] = i;
    }
    if (ggml_tensor * pos_in = ggml_graph_get_tensor(db.graph, "dec.pos_ids"); pos_in != nullptr) {
        ggml_backend_tensor_set(pos_in, pos_ids.data(), 0, pos_ids.size() * sizeof(int32_t));
    }
    if (db.causal_mask_in != nullptr) {
        const int          n_kv_mask = static_cast<int>(db.causal_mask_in->ne[0]);
        std::vector<float> mask(static_cast<size_t>(n_kv_mask) * static_cast<size_t>(seq_len));
        for (int q = 0; q < seq_len; ++q) {
            for (int k = 0; k < n_kv_mask; ++k) {
                mask[static_cast<size_t>(q) * n_kv_mask + k] = (k < seq_len && k <= q) ? 0.0f : -1e9f;
            }
        }
        ggml_backend_tensor_set(db.causal_mask_in, mask.data(), 0, mask.size() * sizeof(float));
    }
    if (db.cross_mask_in != nullptr) {
        const int          n_kv_cross = static_cast<int>(db.cross_mask_in->ne[0]);
        std::vector<float> mask(static_cast<size_t>(n_kv_cross) * static_cast<size_t>(seq_len));
        for (int q = 0; q < seq_len; ++q) {
            for (int k = 0; k < n_kv_cross; ++k) {
                mask[static_cast<size_t>(q) * n_kv_cross + k] = (k < T_enc) ? 0.0f : -1e9f;
            }
        }
        ggml_backend_tensor_set(db.cross_mask_in, mask.data(), 0, mask.size() * sizeof(float));
    }
    if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, db.graph); gs != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_GGUF;
    }
    transcribe::debug::dump_tensor(dump_name, db.dumps.logits_raw, "decoder.sibling_mode.logits_raw");
    return TRANSCRIBE_OK;
}

}  // namespace

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

transcribe_status cw_run(transcribe_session *          session,
                         const float *                 pcm,
                         int                           n_samples,
                         const transcribe_run_params * params) {
    auto * cc = static_cast<CwSession *>(session);
    auto * cm = static_cast<CwModel *>(cc->model);
    if (cc == nullptr || cm == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Arms the tensor dumper when TRANSCRIBE_DUMP_DIR is set; a no-op
    // otherwise. Must run before the first graph build, because
    // debug::enabled() also selects the per-step decoder graph below (the
    // static step graph has no per-step tensor to dump).
    transcribe::debug::init();

    const CwHParams & hp = cm->hparams;

    // Language. The reference's transcribe() always forces a language token
    // (defaulting to "en"); an explicit hint resolves through the same table.
    int32_t lang_id = -1;
    if (params != nullptr && params->language != nullptr && params->language[0] != '\0') {
        const std::string want(params->language);
        for (size_t i = 0; i < cm->lang_codes.size() && i < cm->lang_token_ids.size(); ++i) {
            if (cm->lang_codes[i] == want) {
                lang_id = cm->lang_token_ids[i];
                break;
            }
        }
        if (lang_id < 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: unsupported language '%s'", k_family_tag, want.c_str());
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }
    // lang_id < 0 here means "no hint": detect it from the audio once the
    // first window's cross-attention cache exists (see detect_language).
    const bool want_lang_detect = lang_id < 0;

    // Mode: the run ext wins, otherwise the GGUF default. Both tag blocks are
    // the same weights and the same graph; only the prompt prefix differs.
    CwMode mode = cm->contract.default_mode;
    if (params != nullptr && params->family != nullptr &&
        params->family->kind == TRANSCRIBE_EXT_KIND_CRISPERWHISPER_RUN) {
        const auto * ext = reinterpret_cast<const transcribe_crisperwhisper_run_ext *>(params->family);
        if (ext->mode == TRANSCRIBE_CRISPERWHISPER_MODE_VERBATIM) {
            mode = CwMode::Verbatim;
        } else if (ext->mode == TRANSCRIBE_CRISPERWHISPER_MODE_INTENDED) {
            mode = CwMode::Intended;
        }
    }

    // ---- window plan (crisperwhisper/longform/base.py::make_chunks) ----
    // <= 30 s decodes as one window. Longer audio walks 30 s windows at a 26 s
    // stride, and each window after the first is conditioned on the last N
    // confirmed words via the <ctx> block. This is NOT whisper's timestamp
    // stitching and shares no code with it.
    const CwDecodeContract & ct             = cm->contract;
    const int                window_samples = static_cast<int>(ct.longform_chunk_duration * k_sample_rate);
    const int                stride_samples = static_cast<int>(ct.longform_stride * k_sample_rate);

    std::vector<std::pair<int, int>> windows;  // (offset, length)
    if (n_samples <= window_samples) {
        windows.emplace_back(0, n_samples);
    } else {
        int start = 0;
        while (start < n_samples) {
            const int len = std::min(window_samples, n_samples - start);
            windows.emplace_back(start, len);
            if (start + len >= n_samples) {
                break;
            }
            start += stride_samples;
        }
    }
    const bool is_longform = windows.size() > 1;

    // Word timing is not optional on the long-form path: the overlap-aware
    // boundary drop is driven by per-word start times (timestamp_aware_drop
    // defaults true upstream), so a multi-window run always aligns even when
    // the caller did not ask for word timestamps.
    // Requested granularity with AUTO resolved to this family's ceiling.
    // include/transcribe.h: AUTO means "equal to the model's
    // max_timestamp_kind" (WORD here), and a coarser-or-equal explicit
    // request must be answered at that granularity rather than dropped
    // to NONE. Only TOKEN is finer than the ceiling, and the dispatcher
    // has already rejected it by the time we get here.
    const transcribe_timestamp_kind req_ts = params != nullptr ? params->timestamps : TRANSCRIBE_TIMESTAMPS_NONE;
    const transcribe_timestamp_kind eff_ts = req_ts == TRANSCRIBE_TIMESTAMPS_AUTO ? TRANSCRIBE_TIMESTAMPS_WORD : req_ts;

    const bool want_words = eff_ts == TRANSCRIBE_TIMESTAMPS_WORD || transcribe::debug::enabled() || is_longform;

    const int    n_mels       = hp.fe_num_mels;
    const int    n_mel_frames = hp.fe_nb_max_frames > 0 ? hp.fe_nb_max_frames : 3000;
    const char * ref_mel_dir  = transcribe::env::str("TRANSCRIBE_MEL_FROM_REF");

    std::vector<std::string> confirmed_words;
    std::vector<CwWord>      global_words;

    for (size_t wi = 0; wi < windows.size(); ++wi) {
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const auto [off, len] = windows[wi];
        const bool  is_last   = (wi + 1 == windows.size());
        const float start_sec = static_cast<float>(wi) * ct.longform_stride;
        const bool  first_win = (wi == 0);

        // ---- frontend ----
        if (ref_mel_dir != nullptr && ref_mel_dir[0] != '\0') {
            if (const transcribe_status st = load_mel_from_ref(ref_mel_dir, n_mels, n_mel_frames, cc->mel_buf);
                st != TRANSCRIBE_OK) {
                return st;
            }
            cc->mel_10ms.clear();
            cc->mel_10ms_frames = 0;
        } else {
            if (!cm->mel.has_value()) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: mel frontend unavailable", k_family_tag);
                return TRANSCRIBE_ERR_GGUF;
            }
            std::vector<float> padded(static_cast<size_t>(window_samples), 0.0f);
            std::copy(pcm + off, pcm + off + len, padded.begin());

            int                mel_n_mels = 0, mel_n_frames = 0;
            std::vector<float> mel_mn;  // MelFrontend layout: [n_mels, n_frames]
            if (const transcribe_status st =
                    cm->mel->compute(padded.data(), padded.size(), mel_mn, mel_n_mels, mel_n_frames, cc->n_threads);
                st != TRANSCRIBE_OK) {
                return st;
            }
            if (mel_n_mels != n_mels) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run: mel n_mels %d != expected %d", k_family_tag, mel_n_mels,
                        n_mels);
                return TRANSCRIBE_ERR_GGUF;
            }
            // Transpose to encoder layout [n_frames, n_mels] (ggml ne=[n_mels,
            // T] means n_mels is INNERMOST). Feeding the untransposed buffer
            // does not error: it decodes as silence.
            const int frames_to_use = std::min(mel_n_frames, n_mel_frames);
            cc->mel_buf.assign(static_cast<size_t>(n_mels) * static_cast<size_t>(n_mel_frames), 0.0f);
            for (int t = 0; t < frames_to_use; ++t) {
                for (int m = 0; m < mel_n_mels; ++m) {
                    cc->mel_buf[static_cast<size_t>(t) * mel_n_mels + m] =
                        mel_mn[static_cast<size_t>(m) * mel_n_frames + t];
                }
            }
            // The 10 ms mel is the Viterbi aligner's blank-energy source.
            cc->mel_10ms        = std::move(mel_mn);
            cc->mel_10ms_frames = mel_n_frames;
        }

        int T_enc = 0;
        if (const transcribe_status st = run_encoder_on_window(cc, cm, cc->mel_buf.data(), n_mels, n_mel_frames,
                                                               /*allow_dumps=*/first_win, T_enc);
            st != TRANSCRIBE_OK) {
            return st;
        }

        // ---- language detection (first window only, no hint given) ----
        if (first_win && want_lang_detect) {
            // detect_language needs the cross-attention cache, which
            // decode_window populates. Prime it here with a throwaway
            // cross-KV pass so detection sees this window's encoder output.
            if (const transcribe_status st = prime_cross_kv(cc, cm, T_enc); st != TRANSCRIBE_OK) {
                return st;
            }
            const int32_t detected = detect_language(cc, cm, T_enc);
            if (detected >= 0) {
                lang_id = detected;
                for (size_t i = 0; i < cm->lang_token_ids.size() && i < cm->lang_codes.size(); ++i) {
                    if (cm->lang_token_ids[i] == detected) {
                        cc->detected_language = cm->lang_codes[i];
                        break;
                    }
                }
            }
        }

        // ---- decode ----
        // Context is the last `context_words` confirmed words; empty on the
        // first window. Upstream builds it BEFORE any hotword block, which is
        // the training order.
        std::string context_text;
        if (!first_win && !confirmed_words.empty()) {
            const size_t take =
                std::min<size_t>(confirmed_words.size(), static_cast<size_t>(std::max(ct.longform_context_words, 0)));
            const size_t begin = confirmed_words.size() - take;
            for (size_t k = begin; k < confirmed_words.size(); ++k) {
                if (!context_text.empty()) {
                    context_text += ' ';
                }
                context_text += confirmed_words[k];
            }
        }

        const std::vector<int32_t> prompt_ids = build_prompt(*cm, mode, lang_id, context_text);
        std::vector<int32_t>       gen_ids;
        std::vector<int32_t>       align_ids;
        if (const transcribe_status st =
                decode_window(cc, cm, prompt_ids, T_enc, k_default_max_new_tokens,
                              /*emit_dumps=*/first_win, /*capture_align=*/want_words, gen_ids, align_ids);
            st != TRANSCRIBE_OK) {
            return st;
        }

        // ---- word segmentation + timing ----
        // Upstream treats the timing pipeline's segmentation as the single
        // canonical word source, so text and timings cannot drift apart at the
        // boundary drop. Unplaceable words are kept as placeholders so the
        // list stays 1-to-1 with the text.
        std::vector<CwWord> words;
        if (want_words && cc->cross_attn_rows > 0 && !cc->mel_10ms.empty()) {
            const int rows   = std::min(cc->cross_attn_rows, static_cast<int>(align_ids.size()));
            const int frames = std::min(cc->cross_attn_frames, T_enc);

            std::vector<float> attn(static_cast<size_t>(rows) * static_cast<size_t>(frames), 0.0f);
            for (int r = 0; r < rows; ++r) {
                const float * src = cc->cross_attn.data() + static_cast<size_t>(r) * cc->cross_attn_frames;
                std::copy(src, src + frames, attn.begin() + static_cast<size_t>(r) * frames);
            }
            if (first_win && transcribe::debug::enabled()) {
                const long long shape[2] = { rows, frames };
                transcribe::debug::dump_host_f32("dec.xattn.align", attn.data(), static_cast<long long>(attn.size()),
                                                 shape, 2, "decoder.xattn.align");
            }
            std::vector<int>         ids(align_ids.begin(), align_ids.begin() + rows);
            std::vector<std::string> pieces;
            pieces.reserve(ids.size());
            for (const int id : ids) {
                pieces.push_back(cm->tok.decode(&id, 1));
            }
            words = extract_word_timings(ids, pieces, attn, rows, frames, cc->mel_10ms, hp.fe_num_mels,
                                         cc->mel_10ms_frames);
        }

        if (first_win && transcribe::debug::enabled()) {
            (void) dump_sibling_mode_logits(cc, cm, mode, lang_id, T_enc);
        }

        if (!is_longform) {
            // Single window: the decoded token stream is the transcript.
            const std::vector<int32_t> kept = strip_prompt_artifacts(ct, gen_ids);
            std::string                text;
            if (!kept.empty()) {
                text = cm->tok.decode(kept.data(), static_cast<int>(kept.size()));
            }
            cc->full_text = normalize_ws(text);
            for (const CwWord & w : words) {
                if (w.placed()) {
                    global_words.push_back(w);
                }
            }
            break;
        }

        if (words.empty()) {
            log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "%s run: long-form window %zu/%zu produced no words", k_family_tag,
                    wi + 1, windows.size());
            continue;
        }

        // ---- overlap-aware boundary drop (_overlap_drop_index) ----
        // A trailing word may be dropped only when the NEXT window re-covers
        // it, i.e. its audio starts at or after the stride boundary.
        // drop_words caps how many may go; words starting before the overlap
        // are always kept, because nothing downstream would re-transcribe them.
        const int n_words = static_cast<int>(words.size());
        int       keep    = n_words;
        if (!is_last) {
            const int legacy     = std::max(n_words - std::max(ct.longform_drop_words, 0), 0);
            bool      any_placed = false;
            for (const CwWord & w : words) {
                if (w.placed()) {
                    any_placed = true;
                    break;
                }
            }
            if (!any_placed) {
                keep = legacy;
            } else {
                int first_overlap = n_words;
                for (int k = 0; k < n_words; ++k) {
                    if (words[static_cast<size_t>(k)].placed() &&
                        words[static_cast<size_t>(k)].start >= ct.longform_stride) {
                        first_overlap = k;
                        break;
                    }
                }
                keep = std::max(legacy, first_overlap);
            }
        }

        for (int k = 0; k < keep; ++k) {
            const CwWord & w = words[static_cast<size_t>(k)];
            confirmed_words.push_back(w.text);
            if (!w.placed()) {
                continue;  // no timestamp to lift
            }
            CwWord g = w;
            g.start += start_sec;
            g.end += start_sec;
            global_words.push_back(g);
        }
    }

    if (is_longform) {
        std::string text;
        for (const std::string & w : confirmed_words) {
            if (!text.empty()) {
                text += ' ';
            }
            text += w;
        }
        cc->full_text = normalize_ws(text);

        // Clamp starts forward at the seams: the Viterbi can place a word in
        // window i+1 slightly before its predecessor's end in window i.
        for (size_t i = 1; i < global_words.size(); ++i) {
            const float prev_end = global_words[i - 1].end;
            if (global_words[i].start < prev_end) {
                global_words[i].start = prev_end;
                global_words[i].end   = std::max(prev_end, global_words[i].end);
            }
        }
    }

    cc->has_result = true;
    cc->words.clear();
    for (const CwWord & w : global_words) {
        transcribe_session::WordEntry we;
        we.text  = w.text;
        we.t0_ms = static_cast<int64_t>(w.start * 1000.0f);
        we.t1_ms = static_cast<int64_t>(w.end * 1000.0f);
        cc->words.push_back(we);
    }
    if (eff_ts == TRANSCRIBE_TIMESTAMPS_WORD && !cc->words.empty()) {
        cc->result_kind = TRANSCRIBE_TIMESTAMPS_WORD;
    } else if (eff_ts == TRANSCRIBE_TIMESTAMPS_SEGMENT) {
        cc->result_kind = TRANSCRIBE_TIMESTAMPS_SEGMENT;
    } else {
        cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    }

    // One segment spanning the audio. This family emits no timestamp tokens,
    // so there is no real boundary to recover; a SEGMENT request is answered
    // with the whole-audio span, which is the only honest answer available.
    transcribe_session::SegmentEntry seg;
    seg.text       = cc->full_text;
    seg.t0_ms      = 0;
    seg.t1_ms      = static_cast<int64_t>(1000.0 * static_cast<double>(n_samples) / k_sample_rate);
    seg.first_word = 0;
    seg.n_words    = static_cast<int>(cc->words.size());
    cc->segments.clear();
    cc->segments.push_back(seg);

    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// run_batch
// ---------------------------------------------------------------------------

namespace {

transcribe_status cw_run_batch_serial(CwSession *                   cc,
                                      const float * const *         pcm,
                                      const int *                   n_samples,
                                      int                           n,
                                      const transcribe_run_params * params) {
    for (int i = 0; i < n; ++i) {
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const transcribe_status st = (pcm[i] == nullptr || n_samples[i] <= 0) ?
                                         TRANSCRIBE_ERR_INVALID_ARG :
                                         cw_run(cc, pcm[i], n_samples[i], params);
        if (st == TRANSCRIBE_OK) {
            cc->batch_results.push_back(cc->capture_result(st));
        } else {
            transcribe_session::ResultSet rs;
            rs.status = st;
            cc->batch_results.push_back(std::move(rs));
        }
    }
    return TRANSCRIBE_OK;
}

}  // namespace

// Offline batched decode: B utterances stepped in lockstep through the shared
// batched cross-KV + step graphs.
//
// The prompt is uniform across the batch here, which is what makes this
// simpler than whisper's equivalent: short-form CrisperWhisper has no <ctx>
// block, so every utterance gets the same mode tags + Whisper prefix and
// therefore the same prompt length. The prompt is fed one position at a time
// through the same batched step graph rather than needing a separate
// prompt-pass graph.
//
// Peeled to serial (each for a reason the batched graph genuinely cannot
// serve, not for convenience):
//   * n == 1                     nothing to batch.
//   * no flash / non-GPU backend the batched view layout requires
//                                ggml_flash_attn_ext.
//   * debug dumping              per-utterance dumps would overwrite each
//                                other under one TRANSCRIBE_DUMP_DIR.
//   * word timestamps            cross-attention capture forces the manual
//                                softmax path, which the batched graph does
//                                not build.
//   * any utterance > 30 s       long-form is <ctx> continuation, and each
//                                window's prompt depends on the PREVIOUS
//                                window's confirmed words. That is sequential
//                                by construction and cannot run in lockstep.
transcribe_status cw_run_batch(transcribe_session *          session,
                               const float * const *         pcm,
                               const int *                   n_samples,
                               int                           n,
                               const transcribe_run_params * params) {
    if (session == nullptr || pcm == nullptr || n_samples == nullptr || n <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<CwSession *>(session);
    auto * cm = static_cast<CwModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty() || !cm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const CwHParams &        hp = cm->hparams;
    const CwDecodeContract & ct = cm->contract;

    const bool primary_is_gpu = cm->plan.primary_kind != transcribe::BackendKind::Cpu &&
                                cm->plan.primary_kind != transcribe::BackendKind::Accel &&
                                cm->plan.primary_kind != transcribe::BackendKind::Unknown;
    const int  window_samples = hp.fe_n_samples > 0 ? hp.fe_n_samples : 30 * k_sample_rate;

    bool any_longform = false;
    for (int b = 0; b < n; ++b) {
        if (pcm[b] == nullptr || n_samples[b] <= 0 || n_samples[b] > window_samples) {
            any_longform = true;
            break;
        }
    }
    // Same AUTO resolution as cw_run: AUTO means WORD for this family, and
    // the batched graph has no cross-attention capture, so both route to the
    // serial path that runs the Viterbi aligner.
    const transcribe_timestamp_kind req_ts = params != nullptr ? params->timestamps : TRANSCRIBE_TIMESTAMPS_NONE;
    const transcribe_timestamp_kind eff_ts = req_ts == TRANSCRIBE_TIMESTAMPS_AUTO ? TRANSCRIBE_TIMESTAMPS_WORD : req_ts;
    const bool                      want_word_ts = eff_ts == TRANSCRIBE_TIMESTAMPS_WORD;
    if (n == 1 || !cc->decoder_use_flash || !primary_is_gpu || transcribe::debug::enabled() || want_word_ts ||
        any_longform) {
        return cw_run_batch_serial(cc, pcm, n_samples, n, params);
    }

    transcribe::debug::init();

    // Mode + language are uniform for the batch (one run_params).
    CwMode mode = ct.default_mode;
    if (params != nullptr && params->family != nullptr &&
        params->family->kind == TRANSCRIBE_EXT_KIND_CRISPERWHISPER_RUN) {
        const auto * ext = reinterpret_cast<const transcribe_crisperwhisper_run_ext *>(params->family);
        if (ext->mode == TRANSCRIBE_CRISPERWHISPER_MODE_VERBATIM) {
            mode = CwMode::Verbatim;
        } else if (ext->mode == TRANSCRIBE_CRISPERWHISPER_MODE_INTENDED) {
            mode = CwMode::Intended;
        }
    }
    int32_t lang_id = -1;
    {
        std::string want = "en";
        if (params != nullptr && params->language != nullptr && params->language[0] != '\0') {
            want = params->language;
        }
        for (size_t i = 0; i < cm->lang_codes.size() && i < cm->lang_token_ids.size(); ++i) {
            if (cm->lang_codes[i] == want) {
                lang_id = cm->lang_token_ids[i];
                break;
            }
        }
        if (lang_id < 0) {
            return cw_run_batch_serial(cc, pcm, n_samples, n, params);
        }
    }

    const std::vector<int32_t> prompt_ids   = build_prompt(*cm, mode, lang_id, /*context_text=*/"");
    const int                  prompt_len   = static_cast<int>(prompt_ids.size());
    const int                  B            = n;
    const int                  d_model      = hp.dec_d_model;
    const int                  n_layer      = hp.dec_n_layers;
    const int                  vocab_size   = hp.dec_vocab_size;
    const int                  eos_id       = cm->tok.find("<|endoftext|>");
    const int                  n_mels       = hp.fe_num_mels;
    const int                  n_mel_frames = hp.fe_nb_max_frames > 0 ? hp.fe_nb_max_frames : 3000;

    // ---- per-utterance encoder ----
    std::vector<std::vector<float>> enc_hosts(static_cast<size_t>(B));
    int                             T_enc_max = 0;
    for (int b = 0; b < B; ++b) {
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        std::vector<float> padded(static_cast<size_t>(window_samples), 0.0f);
        std::copy(pcm[b], pcm[b] + std::min(n_samples[b], window_samples), padded.begin());

        int                mel_n_mels = 0, mel_n_frames = 0;
        std::vector<float> mel_mn;
        if (cm->mel->compute(padded.data(), padded.size(), mel_mn, mel_n_mels, mel_n_frames, cc->n_threads) !=
            TRANSCRIBE_OK) {
            return cw_run_batch_serial(cc, pcm, n_samples, n, params);
        }
        const int frames_to_use = std::min(mel_n_frames, n_mel_frames);
        cc->mel_buf.assign(static_cast<size_t>(n_mels) * static_cast<size_t>(n_mel_frames), 0.0f);
        for (int t = 0; t < frames_to_use; ++t) {
            for (int m = 0; m < mel_n_mels; ++m) {
                cc->mel_buf[static_cast<size_t>(t) * mel_n_mels + m] =
                    mel_mn[static_cast<size_t>(m) * mel_n_frames + t];
            }
        }
        int T_enc = 0;
        if (const transcribe_status st = run_encoder_on_window(cc, cm, cc->mel_buf.data(), n_mels, n_mel_frames,
                                                               /*allow_dumps=*/false, T_enc);
            st != TRANSCRIBE_OK) {
            return st;
        }
        enc_hosts[static_cast<size_t>(b)].assign(static_cast<size_t>(d_model) * static_cast<size_t>(T_enc), 0.0f);
        ggml_backend_tensor_get(cc->enc_out.tensor, enc_hosts[static_cast<size_t>(b)].data(), 0,
                                enc_hosts[static_cast<size_t>(b)].size() * sizeof(float));
        T_enc_max = std::max(T_enc_max, T_enc);
    }

    // ---- batched KV cache ----
    int max_n_kv = 256;
    while (max_n_kv < prompt_len + k_default_max_new_tokens) {
        max_n_kv *= 2;
    }
    max_n_kv = std::min(max_n_kv, hp.dec_max_target_positions);

    ggml_type kv_type_g;
    if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) {
        kv_type_g = GGML_TYPE_F32;
    } else if (cc->kv_type == TRANSCRIBE_KV_TYPE_F16) {
        kv_type_g = GGML_TYPE_F16;
    } else {
        const ggml_tensor * probe = !cm->weights.dec_blocks.empty() ? cm->weights.dec_blocks[0].self_q_w : nullptr;
        kv_type_g                 = (probe != nullptr && probe->type == GGML_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
    }
    cc->kv_cache.free();
    if (!kv_cache_init_batched(cc->kv_cache, cm->plan.primary, max_n_kv, T_enc_max, d_model, n_layer, B, kv_type_g)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s run_batch: kv_cache_init_batched failed", k_family_tag);
        return TRANSCRIBE_ERR_BACKEND;
    }

    auto new_compute_ctx = [&](size_t mem) -> bool {
        if (cc->compute_ctx != nullptr) {
            ggml_free(cc->compute_ctx);
            cc->compute_ctx = nullptr;
        }
        cc->compute_ctx_size = 0;
        ggml_init_params ip{};
        ip.mem_size     = mem;
        ip.mem_buffer   = nullptr;
        ip.no_alloc     = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx != nullptr) {
            cc->compute_ctx_size = mem;
        }
        return cc->compute_ctx != nullptr;
    };

    // ---- batched cross-attention K/V ----
    {
        if (!new_compute_ctx(16 * 1024 * 1024)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        wg::DecoderBuild cross =
            wg::build_cross_kv_graph_batched(cc->compute_ctx, cm->weights, hp, cc->kv_cache, T_enc_max, B);
        if (cross.graph == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, cross.graph)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        std::vector<float> packed(static_cast<size_t>(d_model) * T_enc_max * B, 0.0f);
        for (int b = 0; b < B; ++b) {
            std::memcpy(packed.data() + static_cast<size_t>(b) * T_enc_max * d_model,
                        enc_hosts[static_cast<size_t>(b)].data(),
                        enc_hosts[static_cast<size_t>(b)].size() * sizeof(float));
        }
        ggml_backend_tensor_set(cross.encoder_out_in, packed.data(), 0, packed.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(cc->sched, cross.graph) != GGML_STATUS_SUCCESS) {
            return TRANSCRIBE_ERR_GGUF;
        }
        cc->kv_cache.cross_populated = true;
    }

    // ---- batched step graph, growing self-attention window ----
    const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);

    std::vector<ggml_fp16_t> cmask(static_cast<size_t>(T_enc_max) * B, f16_zero);

    int kv_window = 256;
    while (kv_window > max_n_kv) {
        kv_window /= 2;
    }
    if (kv_window < 1) {
        kv_window = max_n_kv;
    }

    wg::StepBuildBatched sb{};
    auto                 rebuild_step = [&](int win) -> bool {
        if (!new_compute_ctx(32 * 1024 * 1024)) {
            return false;
        }
        sb = wg::build_step_graph_batched(cc->compute_ctx, cm->weights, hp, cc->kv_cache, win, T_enc_max, B,
                                          cc->decoder_use_flash);
        if (sb.graph == nullptr || sb.logits_out == nullptr) {
            return false;
        }
        ggml_backend_sched_reset(cc->sched);
        if (!ggml_backend_sched_alloc_graph(cc->sched, sb.graph)) {
            return false;
        }
        ggml_backend_tensor_set(sb.cross_mask_in, cmask.data(), 0, cmask.size() * sizeof(ggml_fp16_t));
        return true;
    };
    if (!rebuild_step(kv_window)) {
        return TRANSCRIBE_ERR_GGUF;
    }

    std::vector<ggml_fp16_t> smask(static_cast<size_t>(kv_window) * B, f16_ninf);
    std::vector<int32_t>     tok_buf(static_cast<size_t>(B), 0), pos_buf(static_cast<size_t>(B), 0);
    std::vector<int64_t>     kvidx_buf(static_cast<size_t>(B), 0);
    std::vector<float>       logits_host(static_cast<size_t>(vocab_size) * B);

    auto ensure_window = [&](int posv) -> bool {
        if (posv + 1 <= kv_window) {
            return true;
        }
        int win = kv_window;
        while (win < posv + 1 && win < max_n_kv) {
            win *= 2;
        }
        win = std::min(win, max_n_kv);
        if (win == kv_window) {
            return true;
        }
        std::vector<ggml_fp16_t> wider(static_cast<size_t>(win) * B, f16_ninf);
        for (int b = 0; b < B; ++b) {
            std::fill(wider.data() + static_cast<size_t>(b) * win, wider.data() + static_cast<size_t>(b) * win + posv,
                      f16_zero);
        }
        smask.swap(wider);
        kv_window = win;
        return rebuild_step(kv_window);
    };

    auto run_step = [&](int posv) -> transcribe_status {
        for (int b = 0; b < B; ++b) {
            pos_buf[static_cast<size_t>(b)]                  = posv;
            kvidx_buf[static_cast<size_t>(b)]                = posv;
            smask[static_cast<size_t>(b) * kv_window + posv] = f16_zero;
        }
        ggml_backend_tensor_set(sb.token_ids_in, tok_buf.data(), 0, B * sizeof(int32_t));
        ggml_backend_tensor_set(sb.pos_ids_in, pos_buf.data(), 0, B * sizeof(int32_t));
        ggml_backend_tensor_set(sb.kv_idx_in, kvidx_buf.data(), 0, B * sizeof(int64_t));
        ggml_backend_tensor_set(sb.self_mask_in, smask.data(), 0, smask.size() * sizeof(ggml_fp16_t));
        if (ggml_backend_sched_graph_compute(cc->sched, sb.graph) != GGML_STATUS_SUCCESS) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_get(sb.logits_out, logits_host.data(), 0,
                                static_cast<size_t>(vocab_size) * B * sizeof(float));
        return TRANSCRIBE_OK;
    };

    std::vector<std::vector<int32_t>> gen(static_cast<size_t>(B));
    std::vector<char>                 fin(static_cast<size_t>(B), 0);

    // Prompt pass: one position per step, identical across the batch.
    for (int pos = 0; pos < prompt_len; ++pos) {
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        if (!ensure_window(pos)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        for (int b = 0; b < B; ++b) {
            tok_buf[static_cast<size_t>(b)] = prompt_ids[static_cast<size_t>(pos)];
        }
        if (const transcribe_status st = run_step(pos); st != TRANSCRIBE_OK) {
            return st;
        }
    }

    // The last prompt position's logits predict each utterance's first token.
    for (int b = 0; b < B; ++b) {
        const float * row = logits_host.data() + static_cast<size_t>(b) * vocab_size;
        const int     id  = argmax_with_suppression(row, vocab_size, hp.suppress_tokens);
        if (id < 0 || id == eos_id) {
            fin[static_cast<size_t>(b)] = 1;
        } else {
            gen[static_cast<size_t>(b)].push_back(id);
        }
        tok_buf[static_cast<size_t>(b)] = (id >= 0) ? id : eos_id;
    }

    const int budget = std::min(k_default_max_new_tokens, hp.dec_max_target_positions - prompt_len);
    for (int step = 0; step < budget; ++step) {
        bool all_done = true;
        for (int b = 0; b < B; ++b) {
            if (!fin[static_cast<size_t>(b)]) {
                all_done = false;
                break;
            }
        }
        if (all_done) {
            break;
        }
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const int pos = prompt_len + step;
        if (pos + 1 > max_n_kv || !ensure_window(pos)) {
            break;
        }
        if (const transcribe_status st = run_step(pos); st != TRANSCRIBE_OK) {
            return st;
        }
        for (int b = 0; b < B; ++b) {
            if (fin[static_cast<size_t>(b)]) {
                continue;
            }
            const float * row = logits_host.data() + static_cast<size_t>(b) * vocab_size;
            const int     id  = argmax_with_suppression(row, vocab_size, hp.suppress_tokens);
            if (id < 0 || id == eos_id) {
                fin[static_cast<size_t>(b)]     = 1;
                tok_buf[static_cast<size_t>(b)] = eos_id;
                continue;
            }
            gen[static_cast<size_t>(b)].push_back(id);
            tok_buf[static_cast<size_t>(b)] = id;
        }
    }

    // ---- per-utterance results ----
    for (int b = 0; b < B; ++b) {
        const std::vector<int32_t> kept = strip_prompt_artifacts(ct, gen[static_cast<size_t>(b)]);
        std::string                text;
        if (!kept.empty()) {
            text = cm->tok.decode(kept.data(), static_cast<int>(kept.size()));
        }
        transcribe_session::ResultSet rs;
        rs.full_text = normalize_ws(text);
        // WORD (and AUTO, which resolves to it) never reaches this path.
        rs.result_kind =
            eff_ts == TRANSCRIBE_TIMESTAMPS_SEGMENT ? TRANSCRIBE_TIMESTAMPS_SEGMENT : TRANSCRIBE_TIMESTAMPS_NONE;
        rs.has_result = true;
        rs.status     = TRANSCRIBE_OK;

        transcribe_session::SegmentEntry seg;
        seg.text  = rs.full_text;
        seg.t0_ms = 0;
        seg.t1_ms = static_cast<int64_t>(1000.0 * static_cast<double>(n_samples[b]) / k_sample_rate);
        rs.segments.push_back(seg);

        cc->batch_results.push_back(std::move(rs));
    }

    // Mirror utterance 0 into the scratch slot so the single-shot accessors
    // stay coherent (the dispatcher's documented contract).
    if (!cc->batch_results.empty()) {
        cc->full_text   = cc->batch_results[0].full_text;
        cc->segments    = cc->batch_results[0].segments;
        cc->words       = cc->batch_results[0].words;
        cc->result_kind = cc->batch_results[0].result_kind;
        cc->has_result  = cc->batch_results[0].has_result;
    }

    return TRANSCRIBE_OK;
}

namespace {

bool cw_accepts_ext_kind(const transcribe_model * model, transcribe_ext_slot slot, uint32_t kind) {
    (void) model;
    if (slot != TRANSCRIBE_EXT_SLOT_RUN) {
        return false;
    }
    return kind == TRANSCRIBE_EXT_KIND_CRISPERWHISPER_RUN;
}

// Shape-only pre-clear validation for the _RUN slot: reject a too-small run
// ext before the prior result snapshot is wiped.
transcribe_status cw_run_validate(const transcribe_session * ctx, const transcribe_run_params * params) {
    (void) ctx;
    return transcribe_ext_check(params != nullptr ? params->family : nullptr, TRANSCRIBE_EXT_KIND_CRISPERWHISPER_RUN,
                                sizeof(struct transcribe_crisperwhisper_run_ext));
}

}  // namespace

const Arch arch = {
    /* name             */ "crisperwhisper",
    /* load             */ cw_load,
    /* init_context     */ cw_init_context,
    /* run              */ cw_run,
    /* run_batch        */ cw_run_batch,
    /* stream_validate  */ nullptr,
    /* stream_begin     */ nullptr,
    /* stream_feed      */ nullptr,
    /* stream_finalize  */ nullptr,
    /* stream_reset     */ nullptr,
    /* accepts_ext_kind */ cw_accepts_ext_kind,
    /* run_validate     */ cw_run_validate,
};

}  // namespace transcribe::crisperwhisper
