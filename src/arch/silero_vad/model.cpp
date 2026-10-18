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
#include "transcribe-path.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
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

// Shared only by the two Silero file adapters.
transcribe_status init_storage(Model & m, const transcribe_model_load_params * params) {
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
    if (auto st = load_common::init_backends(backend_req, device, kTag, m.plan); st != TRANSCRIBE_OK) {
        return st;
    }
    m.backend         = ggml_backend_name(m.plan.primary);
    m.primary_backend = m.plan.primary;

    m.backend_buffer = ggml_backend_alloc_ctx_tensors(m.ctx_meta, m.plan.primary);
    if (m.backend_buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: ggml_backend_alloc_ctx_tensors failed", kTag);
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_buffer_set_usage(m.backend_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return TRANSCRIBE_OK;
}

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

    if (auto st = init_storage(*m, params); st != TRANSCRIBE_OK) {
        return st;
    }
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

namespace {

// Wire layout documented by whisper.cpp models/convert-silero-vad-to-ggml.py.
// This is a native, narrowly scoped reader, not the whisper.cpp inference graph.
struct BinTensor {
    const char * source;
    const char * target;
    int          rank;
    int32_t      ne[3];
};

constexpr BinTensor k_bin_tensors[] = {
    { "_model.encoder.0.reparam_conv.weight", "enc.0.conv.weight",   3, { 3, 129, 128 } },
    { "_model.encoder.0.reparam_conv.bias",   "enc.0.conv.bias",     1, { 128 }         },
    { "_model.encoder.1.reparam_conv.weight", "enc.1.conv.weight",   3, { 3, 128, 64 }  },
    { "_model.encoder.1.reparam_conv.bias",   "enc.1.conv.bias",     1, { 64 }          },
    { "_model.encoder.2.reparam_conv.weight", "enc.2.conv.weight",   3, { 3, 64, 64 }   },
    { "_model.encoder.2.reparam_conv.bias",   "enc.2.conv.bias",     1, { 64 }          },
    { "_model.encoder.3.reparam_conv.weight", "enc.3.conv.weight",   3, { 3, 64, 128 }  },
    { "_model.encoder.3.reparam_conv.bias",   "enc.3.conv.bias",     1, { 128 }         },
    { "_model.decoder.rnn.weight_ih",         "lstm.weight_ih",      2, { 128, 512 }    },
    { "_model.decoder.rnn.weight_hh",         "lstm.weight_hh",      2, { 128, 512 }    },
    { "_model.decoder.rnn.bias_ih",           "lstm.bias_ih",        1, { 512 }         },
    { "_model.decoder.rnn.bias_hh",           "lstm.bias_hh",        1, { 512 }         },
    { "_model.decoder.decoder.2.weight",      "head.weight",         1, { 128 }         },
    { "_model.decoder.decoder.2.bias",        "head.bias",           0, {}              },
    { "_model.stft.forward_basis_buffer",     "frontend.stft_basis", 3, { 256, 1, 258 } },
};
constexpr size_t k_bin_count = sizeof(k_bin_tensors) / sizeof(k_bin_tensors[0]);

struct BinReader {
    std::ifstream file;
    int64_t       remaining;

    explicit BinReader(const char * path) :
        file(path_from_utf8(path), std::ios::binary | std::ios::ate),
        remaining(-1) {
        if (file) {
            remaining = static_cast<int64_t>(file.tellg());
            file.seekg(0);
        }
    }

    bool bytes(void * out, size_t n) {
        if (remaining < 0 || n > static_cast<uint64_t>(remaining)) {
            return false;
        }
        if (!file.read(static_cast<char *>(out), static_cast<std::streamsize>(n))) {
            return false;
        }
        remaining -= static_cast<int64_t>(n);
        return true;
    }

    bool u32(uint32_t & out) {
        uint8_t b[4];
        if (!bytes(b, sizeof(b))) {
            return false;
        }
        out = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
        return true;
    }

    bool expect(uint32_t value) {
        uint32_t got;
        return u32(got) && got == value;
    }
};

}  // namespace

transcribe_status load_from_bin(const char *                         path,
                                const transcribe_model_load_params * params,
                                transcribe_model **                  out_model) {
    *out_model                 = nullptr;
    const int64_t t_load_start = ggml_time_us();
    BinReader     r(path);
    auto          bad = []() {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "%s: invalid or unsupported Silero .bin", kTag);
        return TRANSCRIBE_ERR_GGUF;
    };
    char     tag[sizeof("silero-16k") - 1];
    uint32_t major, minor, patch;
    if (!r.expect(0x67676d6c) || !r.expect(sizeof(tag)) || !r.bytes(tag, sizeof(tag)) ||
        std::memcmp(tag, "silero-16k", sizeof(tag)) != 0 || !r.u32(major) || !r.u32(minor) || !r.u32(patch)) {
        return bad();
    }
    if (!((major == 5 && minor == 1 && patch == 2) || (major == 6 && minor == 2 && patch == 0))) {
        return bad();
    }
    // Validate the complete fixed geometry before any model/backend allocation.
    constexpr uint32_t geometry[] = { 512, 64, 4, 129, 128, 3, 128, 64, 3, 64, 64, 3, 64, 128, 3, 128, 128, 128, 1 };
    for (uint32_t dim : geometry) {
        if (!r.expect(dim)) {
            return bad();
        }
    }

    struct Record {
        std::streampos offset{};
        size_t         count = 1;
        uint32_t       type  = 0;
        bool           seen  = false;
    } records[k_bin_count];

    // First pass checks exact ranks/shapes and all payload bounds. Multiplication
    // uses only trusted dimensions, never attacker-supplied dimensions.
    while (r.remaining > 0) {
        uint32_t rank, name_len, type;
        if (!r.u32(rank) || !r.u32(name_len) || !r.u32(type) || rank > 3 || name_len == 0 || name_len > 128 ||
            type > 1) {
            return bad();
        }
        uint32_t ne[3]{};
        for (uint32_t i = 0; i < rank; ++i) {
            if (!r.u32(ne[i])) {
                return bad();
            }
        }
        char name[128];
        if (!r.bytes(name, name_len)) {
            return bad();
        }
        size_t index = 0;
        for (; index < k_bin_count; ++index) {
            if (std::strlen(k_bin_tensors[index].source) == name_len &&
                std::memcmp(name, k_bin_tensors[index].source, name_len) == 0) {
                break;
            }
        }
        if (index == k_bin_count || records[index].seen) {
            return bad();
        }
        const BinTensor & spec = k_bin_tensors[index];
        Record &          rec  = records[index];
        if (rank != static_cast<uint32_t>(spec.rank)) {
            return bad();
        }
        for (uint32_t i = 0; i < rank; ++i) {
            if (ne[i] != static_cast<uint32_t>(spec.ne[i])) {
                return bad();
            }
            rec.count *= static_cast<size_t>(spec.ne[i]);
        }
        const size_t bytes = rec.count * (type == 0 ? 4 : 2);
        if (bytes > static_cast<uint64_t>(r.remaining)) {
            return bad();
        }
        rec.seen   = true;
        rec.type   = type;
        rec.offset = r.file.tellg();
        r.file.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
        if (!r.file) {
            return bad();
        }
        r.remaining -= static_cast<int64_t>(bytes);
    }
    for (const Record & rec : records) {
        if (!rec.seen) {
            return bad();
        }
    }

    auto m                          = std::make_unique<Model>();
    m->arch                         = &arch;
    m->variant                      = major == 5 ? "silero-vad-v5.1.2" : "silero-vad-v6.2.0";
    m->meta["general.architecture"] = "silero_vad";
    m->meta["general.version"]      = major == 5 ? "5.1.2" : "6.2.0";
    m->meta["stt.variant"]          = m->variant;
    HParams &     hp                = m->hparams;
    const int32_t channels[]        = { 129, 128, 64, 64, 128 };
    const int32_t strides[]         = { 1, 2, 2, 1 };
    const int32_t t_len[]           = { 4, 4, 2, 1, 1 };
    std::copy(channels, channels + 5, hp.channels);
    std::copy(strides, strides + 4, hp.strides);
    std::copy(t_len, t_len + 5, hp.t_len);
    hp.n_bins     = 129;
    hp.n_bins_pad = 136;
    hp.chunk      = 576;
    hp.padded     = 640;

    ggml_init_params ip{};
    ip.no_alloc = true;
    ip.mem_size = k_bin_count * ggml_tensor_overhead();
    m->ctx_meta = ggml_init(ip);
    if (m->ctx_meta == nullptr) {
        return TRANSCRIBE_ERR_OOM;
    }
    for (size_t i = 0; i < k_bin_count; ++i) {
        const BinTensor & spec = k_bin_tensors[i];
        int64_t           ne[] = { 1, 1, 1 };
        for (int j = 0; j < spec.rank; ++j) {
            ne[j] = spec.ne[j];
        }
        // The singleton STFT axis and scalar bias collapse to canonical slots.
        if (i == k_bin_count - 1) {
            ne[1] = ne[2];
            ne[2] = 1;
        }
        ggml_tensor * t = ggml_new_tensor(m->ctx_meta, GGML_TYPE_F32, 3, ne);
        ggml_set_name(t, spec.target);
    }
    if (auto st = bind_weights(m->ctx_meta, hp, m->weights); st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = init_storage(*m, params); st != TRANSCRIBE_OK) {
        return st;
    }
    for (size_t i = 0; i < k_bin_count; ++i) {
        const Record & rec   = records[i];
        const size_t   bytes = rec.count * (rec.type == 0 ? 4 : 2);
        r.file.seekg(rec.offset);
        r.remaining = static_cast<int64_t>(bytes);
        std::vector<uint8_t> raw(bytes);
        std::vector<float>   values(rec.count);
        if (!r.bytes(raw.data(), bytes)) {
            return bad();
        }
        for (size_t j = 0; j < rec.count; ++j) {
            if (rec.type == 1) {
                const ggml_fp16_t half = static_cast<ggml_fp16_t>(uint16_t(raw[2 * j]) | uint16_t(raw[2 * j + 1]) << 8);
                values[j]              = ggml_fp16_to_fp32(half);
            } else {
                const uint8_t * b = raw.data() + 4 * j;
                const uint32_t  bits =
                    uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
                std::memcpy(&values[j], &bits, sizeof(bits));
            }
            if (!std::isfinite(values[j])) {
                return bad();
            }
        }
        ggml_tensor * t = ggml_get_tensor(m->ctx_meta, k_bin_tensors[i].target);
        ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
    }
    if (auto st = build_derived_weights(*m); st != TRANSCRIBE_OK) {
        return st;
    }
    m->roles = TRANSCRIBE_ROLE_VAD;
    set_feature(m.get(), TRANSCRIBE_FEATURE_CANCELLATION, true);
    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

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
