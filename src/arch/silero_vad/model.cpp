// arch/silero_vad/model.cpp - Silero VAD family handler (VAD role): load,
// the VAD ops table, and the validation dumps.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "silero_vad.h"
#include "transcribe-arch.h"
#include "transcribe-backend.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace transcribe::silero_vad {

namespace {

constexpr const char * kTag = "silero_vad";

// Default when stt.variant is absent.
constexpr const char k_default_variant[] = "silero-vad-v6.2";

// Frames per encoder graph. Long inputs are scored in blocks of this many
// frames (about 8 s of audio); the graph is rebuilt only when the block size
// changes, so a stream fed at a steady rate reuses it.
constexpr int64_t k_block_frames = 256;

// Smallest graph, in frames. Shorter blocks (a stream fed 32 ms at a time)
// are zero padded to this many columns: ggml's CPU matmul takes its tinyBLAS
// path only from 4 columns (NEON; 2 on x86) and otherwise falls back to a
// kernel with a different reduction order. With the padding every frame is
// computed by the same kernel in the same order whatever the block size, so
// a stream's probabilities are bit-identical to an offline run's.
constexpr int k_min_graph_frames = 4;

struct GgufGuard {
    gguf_context * ctx = nullptr;

    GgufGuard() = default;

    ~GgufGuard() {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }

    GgufGuard(const GgufGuard &)             = delete;
    GgufGuard & operator=(const GgufGuard &) = delete;
};

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = ggml_time_us();

    auto m     = std::make_unique<Model>();
    m->arch    = &arch;
    m->variant = loader.variant().empty() ? k_default_variant : loader.variant();

    if (auto st = read_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }

    GgufGuard guard;
    {
        gguf_init_params init_params{};
        init_params.no_alloc = true;
        init_params.ctx      = &m->ctx_meta;
        guard.ctx            = gguf_init_from_file(loader.path().c_str(), init_params);
        if (guard.ctx == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: failed to reopen \"%s\" for tensor data", kTag,
                    loader.path().c_str());
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    if (auto st = bind_weights(m->ctx_meta, m->hparams, m->weights); st != TRANSCRIBE_OK) {
        return st;
    }

    // AUTO resolves to the strict CPU backend. The model is tiny and the CPU
    // path matches the reference to ~1e-6 in probability, while ggml-metal's
    // F32 mul_mm stages operands as half (about 2e-3). A specific GPU request
    // or an explicit device is still honored.
    transcribe_device_t        device      = params != nullptr ? params->device : nullptr;
    transcribe_backend_request backend_req = params != nullptr ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    int                        req_raw     = TRANSCRIBE_BACKEND_AUTO;
    std::memcpy(&req_raw, &backend_req, sizeof(req_raw));
    if (req_raw == TRANSCRIBE_BACKEND_AUTO && device == nullptr) {
        backend_req = TRANSCRIBE_BACKEND_CPU;
    }
    if (auto st = load_common::init_backends(backend_req, device, kTag, m->plan); st != TRANSCRIBE_OK) {
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    m->backend_buffer = ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (m->backend_buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: ggml_backend_alloc_ctx_tensors failed", kTag);
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_buffer_set_usage(m->backend_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (auto st = load_common::stream_tensor_data(loader.path(), guard.ctx, m->ctx_meta, kTag); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = build_derived_weights(*m); st != TRANSCRIBE_OK) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: derived weight allocation failed", kTag);
        return st;
    }

    m->roles = TRANSCRIBE_ROLE_VAD;
    set_feature(m.get(), TRANSCRIBE_FEATURE_CANCELLATION, true);
    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

// Write the stream's accumulated stage tensors, then drop them. Called when a
// stream ends (reset). No-op unless TRANSCRIBE_DUMP_DIR is set.
void dump_trace(Session & s, const HParams & hp) {
    DumpTrace & t = s.trace;
    if (!debug::enabled() || t.n_frames == 0) {
        t.clear();
        return;
    }
    const long long n = t.n_frames;
    {
        const long long shape[3] = { n, hp.t_len[0], hp.n_bins };
        debug::dump_host_f32("fe.stft_mag", t.stft_mag.data(), static_cast<long long>(t.stft_mag.size()), shape, 3,
                             "encoder");
    }
    for (int i = 0; i < kNumEncoder; ++i) {
        char            name[32];
        const long long shape[3] = { n, hp.t_len[i + 1], hp.channels[i + 1] };
        snprintf(name, sizeof(name), "enc.%d.out", i);
        debug::dump_host_f32(name, t.enc[i].data(), static_cast<long long>(t.enc[i].size()), shape, 3, "encoder");
    }
    const long long shape_h[2] = { n, hp.hidden };
    debug::dump_host_f32("dec.lstm_h", t.lstm_h.data(), static_cast<long long>(t.lstm_h.size()), shape_h, 2, "decoder");
    debug::dump_host_f32("dec.lstm_c", t.lstm_c.data(), static_cast<long long>(t.lstm_c.size()), shape_h, 2, "decoder");
    const long long shape_p[1] = { n };
    debug::dump_host_f32("vad.probs", t.probs.data(), static_cast<long long>(t.probs.size()), shape_p, 1, "decoder");
    t.clear();
}

// ---------------------------------------------------------------------------
// VAD ops
// ---------------------------------------------------------------------------

int32_t vad_frame_samples(const transcribe_model * model) {
    return static_cast<const Model *>(model)->hparams.frame_samples;
}

transcribe_vad_session * vad_new_session() {
    return new Session();
}

void vad_reset(transcribe_vad_session * session) {
    auto &          s  = static_cast<Session &>(*session);
    const HParams & hp = static_cast<const Model &>(*session->model).hparams;
    dump_trace(s, hp);
    s.h.assign(static_cast<size_t>(hp.hidden), 0.0f);
    s.c.assign(static_cast<size_t>(hp.hidden), 0.0f);
    s.context.assign(static_cast<size_t>(hp.context_samples), 0.0f);
    s.chunk_buf.assign(static_cast<size_t>(hp.padded), 0.0f);
    s.gates.assign(4 * static_cast<size_t>(hp.hidden), 0.0f);
}

transcribe_status vad_score(transcribe_vad_session * session, const float * pcm, int64_t n_frames, float * probs) {
    auto &          s  = static_cast<Session &>(*session);
    const Model &   m  = static_cast<const Model &>(*session->model);
    const HParams & hp = m.hparams;
    debug::init();
    if (s.n_threads <= 0) {
        s.n_threads = default_n_threads();
    }

    for (int64_t f0 = 0; f0 < n_frames; f0 += k_block_frames) {
        if (s.poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const int n = static_cast<int>(std::min(k_block_frames, n_frames - f0));

        const int     n_graph = std::max(n, k_min_graph_frames);
        const size_t  per_win = static_cast<size_t>(hp.t_len[0]) * static_cast<size_t>(hp.n_fft);
        const int64_t t0      = ggml_time_us();
        s.windows.assign(static_cast<size_t>(n_graph) * per_win, 0.0f);
        s.gates_in.resize(static_cast<size_t>(n_graph) * 4 * static_cast<size_t>(hp.hidden));
        build_windows(s, hp, pcm + f0 * hp.frame_samples, n, s.windows.data());
        if (auto st = encode_block(s, m, s.windows.data(), n_graph, n, s.gates_in.data()); st != TRANSCRIBE_OK) {
            return st;
        }
        const int64_t t1 = ggml_time_us();
        decode_block(s, m, s.gates_in.data(), n, probs + f0);
        s.t_encode_us += t1 - t0;
        s.t_decode_us += ggml_time_us() - t1;
        s.trace.n_frames += debug::enabled() ? n : 0;
    }
    return TRANSCRIBE_OK;
}

const VadOps k_vad_ops = { vad_frame_samples, vad_new_session, vad_reset, vad_score };

}  // namespace

void DumpTrace::clear() {
    stft_mag.clear();
    for (auto & e : enc) {
        e.clear();
    }
    lstm_h.clear();
    lstm_c.clear();
    probs.clear();
    n_frames = 0;
}

Model::~Model() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
    }
    if (backend_buffer != nullptr) {
        safe_buffer_free(backend_buffer);
    }
    if (ctx_derived != nullptr) {
        ggml_free(ctx_derived);
    }
    if (derived_buffer != nullptr) {
        safe_buffer_free(derived_buffer);
    }
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary = nullptr;
}

const Arch arch = {
    /* .name             = */ "silero_vad",
    /* .load             = */ load,
    /* .init_context     = */ nullptr,
    /* .run              = */ nullptr,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
    /* .run_validate     = */ nullptr,
    /* .diarize          = */ nullptr,
    /* .langid           = */ nullptr,
    /* .vad              = */ &k_vad_ops,
};

}  // namespace transcribe::silero_vad
