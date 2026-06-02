// arch/gigaam/model.cpp - GigaAM family handler.
//
// M1 (this turn): load() reads hparams, validates the tensor catalog,
// streams weight bytes into a backend buffer, ingests the tokenizer
// and languages KV. init_context() works. run() returns NOT_IMPLEMENTED
// gracefully — the encoder/decoder forward lands in M2/M3.
//
// Stage 4 follow-ups:
//   M2: encoder.cpp build_encoder_graph + run() wires it up.
//   M3: per-family mel frontend (center=false, htk, log-clamp) and
//       RNN-T predictor + joint + greedy decode on host. host_decoder
//       populated here at load() time.
//   M4: CTC head + greedy collapse. Charwise tokenizer extension lands
//       on the shared Tokenizer (not in this file).

#include "gigaam.h"

#include "decoder.h"
#include "encoder.h"
#include "weights.h"

#include "transcribe-arch.h"
#include "transcribe-debug.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-meta.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace transcribe::gigaam {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model,   GigaamModel>);
static_assert(std::is_base_of_v<transcribe_session, GigaamSession>);

GigaamSession::~GigaamSession() {
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
    encoder_out = nullptr;
}

GigaamModel::~GigaamModel() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (backend_buffer != nullptr) {
        ggml_backend_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
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

constexpr const char k_default_variant[] = "gigaam-v3-e2e-rnnt";

transcribe_status load(Loader &                         loader,
                       const transcribe_model_load_params *  params,
                       transcribe_model **              out_model)
{
    const int64_t t_load_start = ggml_time_us();

    auto m = std::make_unique<GigaamModel>();
    m->arch      = &arch;
    m->t_load_us = 0;

    m->variant = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    if (auto st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK)
        return st;
    if (auto st = read_languages_kv(loader.gguf(), *m); st != TRANSCRIBE_OK)
        return st;
    if (auto st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK)
        return st;
    if (auto st = read_gigaam_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK)
        return st;

    // Stage 2: reopen for tensor metadata.
    gguf_init_params init_params {};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;
    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(),
                                                   init_params);
    if (gguf_data == nullptr) return TRANSCRIBE_ERR_GGUF;

    if (auto st = build_gigaam_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req =
        (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (auto st = transcribe::load_common::init_backends(
            backend_req, "gigaam", m->plan);
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
                     "gigaam: ggml_backend_alloc_ctx_tensors failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (auto st = transcribe::load_common::stream_tensor_data(
            loader.path(), gguf_data, m->ctx_meta, "gigaam");
        st != TRANSCRIBE_OK)
    {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    // MelFrontend construction: deferred to M3. GigaAM's center=false +
    // htk + log-clamp combination doesn't fit the existing MelConfig
    // shape; the per-family path lands with the encoder bring-up.

    // Build host mirror of the predictor + joint (RNN-T) or CTC head (CTC).
    if (auto st = build_host_decoder_weights(m->weights, m->hparams,
                                             m->host_decoder);
        st != TRANSCRIBE_OK) return st;

    // Initialize the family mel frontend. The Hann window and HTK
    // filterbank are baked into the GGUF by the converter; mel.init
    // copies them out of backend memory rather than recomputing.
    m->mel.init(m->hparams, m->weights);

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(transcribe_model *                model,
                               const transcribe_session_params * params,
                               transcribe_session **             out_ctx)
{
    if (model->arch != &arch) return TRANSCRIBE_ERR_INVALID_ARG;

    auto gc = std::make_unique<GigaamSession>();
    gc->model     = model;
    gc->n_threads = params->n_threads;
    gc->kv_type   = params->kv_type;

    *out_ctx = gc.release();
    return TRANSCRIBE_OK;
}

namespace {

// Load reference mel sidecar from <dir>/frontend.mel.out.f32. Disk
// layout is row-major (n_mels, T_mel) — byte i is at numpy
// (m=i/T_mel, t=i%T_mel). The encoder mel_in tensor has
// ne=[T_mel, n_mels] (fast=T_mel, slow=n_mels), and ggml's
// fast-to-slow layout maps byte i to ne_index (t=i%T_mel, m=i/T_mel).
// Both byte orders are identical — just a memcpy is needed.
transcribe_status load_ref_mel(const std::string & dir,
                               int                 n_mels,
                               int &               n_mel_frames,
                               std::vector<float> & out)
{
    std::string path = dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += "frontend.mel.out.f32";

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr,
                     "gigaam: cannot open mel ref '%s'\n", path.c_str());
        return TRANSCRIBE_ERR_FILE_NOT_FOUND;
    }
    const std::streamsize total_bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (total_bytes <= 0 ||
        (static_cast<size_t>(total_bytes) % (sizeof(float) * n_mels)) != 0)
    {
        std::fprintf(stderr,
                     "gigaam: mel ref '%s' size %lld not divisible by "
                     "n_mels=%d * 4\n",
                     path.c_str(), static_cast<long long>(total_bytes), n_mels);
        return TRANSCRIBE_ERR_GGUF;
    }

    const size_t n_elems = static_cast<size_t>(total_bytes) / sizeof(float);
    n_mel_frames = static_cast<int>(n_elems / n_mels);
    out.assign(n_elems, 0.0f);
    f.read(reinterpret_cast<char *>(out.data()),
           static_cast<std::streamsize>(total_bytes));
    if (static_cast<std::streamsize>(f.gcount()) != total_bytes) {
        std::fprintf(stderr, "gigaam: short read on mel ref\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

} // namespace

transcribe_status run(transcribe_session *      session,
                      const float *             pcm,
                      int                       n_samples,
                      const transcribe_run_params *)
{
    if (session == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * gc = static_cast<GigaamSession *>(session);
    auto * gm = static_cast<GigaamModel *>(gc->model);
    if (gm == nullptr || gm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    if (gc->poll_abort()) return TRANSCRIBE_ERR_ABORTED;

    transcribe::debug::init();

    // ----- Mel acquisition -----------------------------------------------
    // Production path: the family mel (HTK + power=2 + log-clamp +
    // center=False + periodic Hann). The env-var injection stays as a
    // debug knob for tensor-parity isolation.
    int mel_n_frames = 0;
    const int64_t t_mel_start = ggml_time_us();
    if (const char * ref_dir = std::getenv("TRANSCRIBE_GIGAAM_MEL_FROM_REF");
        ref_dir != nullptr && ref_dir[0] != '\0')
    {
        if (auto st = load_ref_mel(ref_dir, gm->hparams.fe_num_mels,
                                   mel_n_frames, gc->mel_buf);
            st != TRANSCRIBE_OK) return st;
    } else {
        if (auto st = gm->mel.compute(pcm, static_cast<size_t>(n_samples),
                                       gc->mel_buf, mel_n_frames);
            st != TRANSCRIBE_OK) return st;
    }
    gc->t_mel_us = ggml_time_us() - t_mel_start;

    {
        const long long mel_shape[2] = {gm->hparams.fe_num_mels, mel_n_frames};
        transcribe::debug::dump_host_f32(
            "frontend.mel.out",
            gc->mel_buf.data(),
            static_cast<long long>(gc->mel_buf.size()),
            mel_shape, 2, "frontend.mel");
    }

    if (mel_n_frames <= 0) return TRANSCRIBE_ERR_GGUF;

    // ----- Reset per-call compute state ---------------------------------
    if (gc->compute_ctx != nullptr) {
        ggml_free(gc->compute_ctx);
        gc->compute_ctx = nullptr;
    }
    gc->encoder_out = nullptr;

    {
        ggml_init_params ip {};
        ip.mem_size = 4 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        gc->compute_ctx = ggml_init(ip);
        if (gc->compute_ctx == nullptr) {
            std::fprintf(stderr, "gigaam: ggml_init failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // ----- Build encoder graph ------------------------------------------
    EncoderBuild eb = build_encoder_graph(gc->compute_ctx,
                                          gm->weights, gm->hparams,
                                          mel_n_frames,
                                          /*kv_type=*/GGML_TYPE_F32,
                                          gm->backend.c_str());
    if (eb.mel_in == nullptr || eb.out == nullptr || eb.graph == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // ----- Scheduler ----------------------------------------------------
    if (gc->sched == nullptr) {
        gc->sched = ggml_backend_sched_new(
            gm->plan.scheduler_list.data(), nullptr,
            static_cast<int>(gm->plan.scheduler_list.size()),
            /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/true);
        if (gc->sched == nullptr) {
            std::fprintf(stderr, "gigaam: ggml_backend_sched_new failed\n");
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(gc->sched);
    if (!ggml_backend_sched_alloc_graph(gc->sched, eb.graph)) {
        std::fprintf(stderr, "gigaam: sched_alloc_graph failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Upload mel.
    ggml_backend_tensor_set(eb.mel_in, gc->mel_buf.data(),
                            0, gc->mel_buf.size() * sizeof(float));
    transcribe::debug::dump_tensor("enc.mel.in", eb.mel_in, "encoder.mel");

    // Upload positions [0, 1, ..., T_enc-1] as int32.
    if (eb.dumps.pos_emb != nullptr) {
        const int64_t T_enc = eb.dumps.pos_emb->ne[0];
        std::vector<int32_t> pos(T_enc);
        for (int64_t i = 0; i < T_enc; ++i) pos[i] = static_cast<int32_t>(i);
        ggml_backend_tensor_set(eb.dumps.pos_emb, pos.data(), 0,
                                pos.size() * sizeof(int32_t));
    }

    // ----- Compute -------------------------------------------------------
    const int64_t t_enc_start = ggml_time_us();
    if (ggml_backend_sched_graph_compute(gc->sched, eb.graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "gigaam: graph_compute failed\n");
        return TRANSCRIBE_ERR_GGUF;
    }
    gc->t_encode_us = ggml_time_us() - t_enc_start;

    // ----- Dump intermediates -------------------------------------------
    if (eb.dumps.pre_encode_out)
        transcribe::debug::dump_tensor("enc.subsample.out",
                                       eb.dumps.pre_encode_out, "encoder.subsample");
    if (eb.dumps.block0_out)
        transcribe::debug::dump_tensor("enc.block.0.out",
                                       eb.dumps.block0_out, "encoder.block.0");
    if (eb.dumps.mid_block_out && eb.dumps.mid_block_idx >= 0) {
        char nm[32];
        std::snprintf(nm, sizeof(nm), "enc.block.%d.out", eb.dumps.mid_block_idx);
        transcribe::debug::dump_tensor(nm, eb.dumps.mid_block_out,
                                       "encoder.block.mid");
    }
    if (eb.dumps.last_block_out && eb.dumps.last_block_idx >= 0) {
        char nm[32];
        std::snprintf(nm, sizeof(nm), "enc.block.%d.out", eb.dumps.last_block_idx);
        transcribe::debug::dump_tensor(nm, eb.dumps.last_block_out,
                                       "encoder.block.last");
    }
    if (eb.dumps.final_out)
        transcribe::debug::dump_tensor("enc.out", eb.dumps.final_out,
                                       "encoder.out");

    gc->encoder_out = eb.out;

    // ----- Decoder (RNN-T or CTC greedy) -------------------------------
    // Both heads consume the pre-final-transpose encoder tensor
    // (ne=[d_model, T_enc, 1]), which is named `rnnt.encoded` on the
    // graph for historical reasons. For CTC variants we read it back
    // but skip the `rnnt.encoded` debug dump (the reference dumper does
    // not emit it for CTC variants, so emitting one on the C++ side
    // would surface as a MISSING-right gate failure in compare_tensors).
    {
        const int64_t t_dec_start = ggml_time_us();

        ggml_tensor * enc_t = eb.dumps.rnnt_encoded;
        if (enc_t == nullptr) {
            std::fprintf(stderr, "gigaam: rnnt.encoded missing\n");
            return TRANSCRIBE_ERR_GGUF;
        }
        const int T_enc = static_cast<int>(enc_t->ne[1]);
        const int D     = static_cast<int>(enc_t->ne[0]);
        gc->enc_host.assign(static_cast<size_t>(T_enc) * D, 0.0f);
        ggml_backend_tensor_get(enc_t, gc->enc_host.data(), 0,
                                gc->enc_host.size() * sizeof(float));

        std::vector<int> tokens;
        std::vector<int> frames;

        if (gm->hparams.head_kind == HeadKind::RNNT) {
            const long long enc_shape[2] = {T_enc, D};
            transcribe::debug::dump_host_f32(
                "rnnt.encoded",
                gc->enc_host.data(),
                static_cast<long long>(gc->enc_host.size()),
                enc_shape, 2, "decoder.rnnt.encoded");

            if (auto st = decode_rnnt_greedy(gm->host_decoder, gm->hparams,
                                              gc->enc_host.data(),
                                              T_enc,
                                              /*max_symbols=*/10,
                                              tokens, frames);
                st != TRANSCRIBE_OK) return st;
        } else { // CTC
            if (auto st = decode_ctc_greedy(gm->host_decoder, gm->hparams,
                                             gc->enc_host.data(),
                                             T_enc,
                                             tokens, frames);
                st != TRANSCRIBE_OK) return st;
        }
        gc->t_decode_us = ggml_time_us() - t_dec_start;

        // Detokenize via the family-agnostic Tokenizer.
        std::string text = gm->tok.decode(tokens.data(),
                                          static_cast<int>(tokens.size()));
        // SentencePiece convention: the first token's leading ▁ is
        // stripped on decode (`sp.decode([▁В, ...])` → "В...", not
        // " В..."). Our shared SP detokenizer maps every ▁ to a space
        // unconditionally, so we trim a single leading space here.
        // Mirrors parakeet's post-decode in model.cpp:1093.
        if (!text.empty() && text.front() == ' ') {
            text.erase(text.begin());
        }

        // Publish on the base context.
        gc->full_text   = text;
        gc->has_result  = true;
        gc->result_kind = TRANSCRIBE_TIMESTAMPS_TOKEN;

        // Tokens. Subsampling factor 4 + hop 160 + sample_rate 16000
        // gives 40 ms per encoder frame. Token spans one frame.
        gc->tokens.clear();
        gc->tokens.reserve(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            transcribe_session::TokenEntry te{};
            te.id    = tokens[i];
            te.text  = gm->tok.token(tokens[i]);
            te.t0_ms = static_cast<int64_t>(frames[i]) * 40;
            te.t1_ms = te.t0_ms + 40;
            gc->tokens.push_back(te);
        }
    }

    return TRANSCRIBE_OK;
}

} // namespace

const Arch arch = {
    /*.name             =*/ "gigaam",
    /*.load             =*/ load,
    /*.init_context     =*/ init_context,
    /*.run              =*/ run,
    /*.stream_validate  =*/ nullptr,
    /*.stream_begin     =*/ nullptr,
    /*.stream_feed      =*/ nullptr,
    /*.stream_finalize  =*/ nullptr,
    /*.stream_reset     =*/ nullptr,
    /*.accepts_ext_kind =*/ nullptr,
};

} // namespace transcribe::gigaam
