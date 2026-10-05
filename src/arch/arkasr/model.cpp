#include "arkasr.h"
#include "decoder.h"
#include "encoder.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "transcribe-arch.h"
#include "transcribe-batch-util.h"
#include "transcribe-flash-policy.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace transcribe::arkasr {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model, ArkAsrModel>);
static_assert(std::is_base_of_v<transcribe_session, ArkAsrSession>);

ArkAsrSession::~ArkAsrSession() {
    kv_cache.free();
    if (sched != nullptr) {
        safe_sched_free(sched);
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
    }
}

ArkAsrModel::~ArkAsrModel() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
    }
    if (backend_buffer != nullptr) {
        safe_buffer_free(backend_buffer);
    }
    packed_gate_up.free();
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        safe_backend_free(*it);
    }
}

namespace {

constexpr int kFirstSpecialToken = 151643;
constexpr int kMaxAudioSamples   = 30 * 16000;
constexpr int kMinAudioSamples   = 20 * 16000;
constexpr int kOverlapSamples    = 2 * 16000;

int context_ceiling(const ArkAsrSession & session, const ArkAsrHParams & hp) {
    return session.n_ctx > 0 ? std::min(session.n_ctx, hp.dec_max_position) : hp.dec_max_position;
}

bool reset_compute_context(ArkAsrSession & session) {
    if (session.sched != nullptr) {
        ggml_backend_sched_reset(session.sched);
    }
    if (session.compute_ctx != nullptr) {
        ggml_free(session.compute_ctx);
    }
    ggml_init_params params{};
    params.mem_size     = 32 * 1024 * 1024;
    params.no_alloc     = true;
    session.compute_ctx = ggml_init(params);
    return session.compute_ctx != nullptr;
}

transcribe_status resolve_special_tokens(ArkAsrModel & model) {
    struct Entry {
        const char * text;
        int32_t *    id;
    };

    const Entry entries[] = {
        { "<|user|>",           &model.special.user        },
        { "<|begin_of_audio|>", &model.special.begin_audio },
        { "<|end_of_audio|>",   &model.special.end_audio   },
        { "<|assistant|>",      &model.special.assistant   },
    };
    for (const auto & entry : entries) {
        *entry.id = model.tok.find(entry.text);
        if (*entry.id < 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "arkasr: tokenizer missing %s", entry.text);
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    model.special.audio      = model.hparams.audio_token_id;
    model.special.eos        = model.hparams.eos_token_id;
    const int audio_vocab_id = model.tok.find("<|audio|>");
    if (audio_vocab_id >= 0 && audio_vocab_id != model.special.audio) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "arkasr: audio token metadata does not match tokenizer");
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

void build_prompt(const ArkAsrModel & model, int audio_tokens, std::vector<int32_t> & ids, int & audio_start) {
    ids.clear();
    ids.reserve(static_cast<size_t>(audio_tokens) + model.instruction_tokens.size() + 4);
    ids.push_back(model.special.user);
    ids.push_back(model.special.begin_audio);
    audio_start = static_cast<int>(ids.size());
    ids.insert(ids.end(), audio_tokens, model.special.audio);
    ids.push_back(model.special.end_audio);
    ids.insert(ids.end(), model.instruction_tokens.begin(), model.instruction_tokens.end());
    ids.push_back(model.special.assistant);
}

std::vector<ggml_fp16_t> causal_mask(int tokens) {
    std::vector<ggml_fp16_t> mask(static_cast<size_t>(tokens) * tokens);
    for (int q = 0; q < tokens; ++q) {
        for (int k = 0; k < tokens; ++k) {
            const float value                         = k <= q ? 0.0f : -std::numeric_limits<float>::infinity();
            mask[static_cast<size_t>(q) * tokens + k] = ggml_fp32_to_fp16(value);
        }
    }
    return mask;
}

std::vector<ggml_fp16_t> step_mask(int width, int position) {
    std::vector<ggml_fp16_t> mask(width);
    for (int i = 0; i < width; ++i) {
        const float value = i <= position ? 0.0f : -std::numeric_limits<float>::infinity();
        mask[i]           = ggml_fp32_to_fp16(value);
    }
    return mask;
}

int argmax_initial_visible(const std::vector<float> & logits) {
    int       best    = -1;
    const int visible = std::min(kFirstSpecialToken, static_cast<int>(logits.size()));
    for (int i = 0; i < visible; ++i) {
        if (best < 0 || logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

void normalize_transcript(std::string & text) {
    const auto trim_left = [&] {
        const auto pos = text.find_first_not_of(" \t\r\n");
        text.erase(0, pos == std::string::npos ? text.size() : pos);
    };
    trim_left();
    if (!text.empty() && text.front() == '.' &&
        (text.size() == 1 || text[1] == ' ' || text[1] == '\t' || text[1] == '\n')) {
        text.erase(0, 1);
        trim_left();
    }
}

struct WindowResult {
    std::string raw_text;
    std::string text;
    bool        truncated = false;
};

struct WordSpan {
    size_t      begin = 0;
    size_t      end   = 0;
    std::string key;
};

std::vector<WordSpan> word_spans(const std::string & text) {
    std::vector<WordSpan> spans;
    for (size_t i = 0; i < text.size();) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) {
            ++i;
        }
        const size_t begin = i;
        while (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i]))) {
            ++i;
        }
        if (begin == i) {
            continue;
        }
        WordSpan span{ begin, i, {} };
        for (size_t j = begin; j < i; ++j) {
            const unsigned char ch = static_cast<unsigned char>(text[j]);
            if (ch >= 0x80 || std::isalnum(ch)) {
                span.key.push_back(ch < 0x80 ? static_cast<char>(std::tolower(ch)) : text[j]);
            }
        }
        spans.push_back(std::move(span));
    }
    return spans;
}

std::string append_with_word_overlap(std::string & full, const std::string & segment) {
    if (segment.empty()) {
        return {};
    }
    if (full.empty()) {
        full = segment;
        return segment;
    }

    const auto   previous = word_spans(full);
    const auto   current  = word_spans(segment);
    const size_t limit    = std::min<size_t>({ 16, previous.size(), current.size() });
    size_t       overlap  = 0;
    for (size_t count = 1; count <= limit; ++count) {
        bool equal = true;
        for (size_t i = 0; i < count; ++i) {
            if (previous[previous.size() - count + i].key != current[i].key) {
                equal = false;
                break;
            }
        }
        if (equal) {
            overlap = count;
        }
    }

    size_t begin = overlap == 0 ? 0 : current[overlap - 1].end;
    while (begin < segment.size() && std::isspace(static_cast<unsigned char>(segment[begin]))) {
        ++begin;
    }
    std::string appended = segment.substr(begin);
    if (!appended.empty()) {
        if (!std::isspace(static_cast<unsigned char>(full.back()))) {
            full.push_back(' ');
        }
        full += appended;
    }
    return appended;
}

int choose_window_samples(const float * pcm, int remaining) {
    if (remaining <= kMaxAudioSamples) {
        return remaining;
    }

    constexpr int kFrameSamples = 80 * 16000 / 1000;
    constexpr int kHopSamples   = 20 * 16000 / 1000;
    double        total_energy  = 0.0;
    for (int i = 0; i < kMaxAudioSamples; ++i) {
        total_energy += static_cast<double>(pcm[i]) * pcm[i];
    }

    double frame_energy = 0.0;
    for (int i = kMinAudioSamples; i < kMinAudioSamples + kFrameSamples; ++i) {
        frame_energy += static_cast<double>(pcm[i]) * pcm[i];
    }
    double best_energy = frame_energy;
    int    best_start  = kMinAudioSamples;
    for (int start = kMinAudioSamples + kHopSamples; start + kFrameSamples <= kMaxAudioSamples; start += kHopSamples) {
        for (int i = start - kHopSamples; i < start; ++i) {
            frame_energy -= static_cast<double>(pcm[i]) * pcm[i];
        }
        for (int i = start + kFrameSamples - kHopSamples; i < start + kFrameSamples; ++i) {
            frame_energy += static_cast<double>(pcm[i]) * pcm[i];
        }
        if (frame_energy < best_energy) {
            best_energy = frame_energy;
            best_start  = start;
        }
    }

    const double window_mean_square = total_energy / kMaxAudioSamples;
    const double best_mean_square   = best_energy / kFrameSamples;
    if (window_mean_square > 0.0 && best_mean_square <= window_mean_square * 0.01) {
        return best_start + kFrameSamples / 2;
    }
    return kMaxAudioSamples;
}

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t start = ggml_time_us();
    auto          model = std::make_unique<ArkAsrModel>();
    model->arch         = &arch;
    model->variant      = loader.variant().empty() ? "ark-asr-3b" : loader.variant();
    apply_family_invariants(*model);

    if (const auto st = read_capability_kv(loader.gguf(), model->caps); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const auto st = read_languages_kv(loader.gguf(), *model); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const auto st = model->tok.load(loader.gguf()); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const auto st = read_arkasr_hparams(loader.gguf(), model->hparams); st != TRANSCRIBE_OK) {
        return st;
    }
    if (model->tok.n_tokens() != model->hparams.dec_vocab_size) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "arkasr: tokenizer and decoder vocabulary sizes differ");
        return TRANSCRIBE_ERR_GGUF;
    }
    if (const auto st = resolve_special_tokens(*model); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const auto st = model->tok.encode("Please transcribe this audio.", model->instruction_tokens);
        st != TRANSCRIBE_OK) {
        return st;
    }

    model->caps.max_audio_ms = 0;
    model->limits            = {};

    MelConfig mel_cfg{};
    mel_cfg.sample_rate  = model->hparams.sample_rate;
    mel_cfg.num_mels     = model->hparams.enc_num_mels;
    mel_cfg.n_fft        = model->hparams.n_fft;
    mel_cfg.win_length   = model->hparams.n_fft;
    mel_cfg.hop_length   = model->hparams.hop_length;
    mel_cfg.pre_emphasis = 0.0f;
    mel_cfg.f_min        = 0.0f;
    mel_cfg.f_max        = static_cast<float>(model->hparams.sample_rate) / 2.0f;
    mel_cfg.pad_mode     = "reflect";
    mel_cfg.window_type  = "hann_periodic";
    mel_cfg.normalize    = "per_utterance";
    {
        using R = load_common::ReadF32Result;
        std::vector<float> freq_major;
        const size_t       n_freq = static_cast<size_t>(mel_cfg.n_fft / 2 + 1);
        const size_t       n_mels = static_cast<size_t>(mel_cfg.num_mels);
        const auto         fb  = load_common::read_f32_tensor_checked(loader.gguf(), loader.path(), "enc.mel_filters",
                                                                      n_freq * n_mels, "arkasr", freq_major);
        const auto         win = load_common::read_f32_tensor_checked(loader.gguf(), loader.path(), "enc.mel_window",
                                                                      mel_cfg.win_length, "arkasr", mel_cfg.window);
        if (fb != R::Ok || win != R::Ok) {
            return TRANSCRIBE_ERR_GGUF;
        }
        mel_cfg.filterbank.resize(freq_major.size());
        for (size_t f = 0; f < n_freq; ++f) {
            for (size_t m = 0; m < n_mels; ++m) {
                mel_cfg.filterbank[m * n_freq + f] = freq_major[f * n_mels + m];
            }
        }
    }
    model->mel.emplace(mel_cfg);

    gguf_init_params init{};
    init.no_alloc       = true;
    init.ctx            = &model->ctx_meta;
    gguf_context * gguf = gguf_init_from_file(loader.path().c_str(), init);
    if (gguf == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (const auto st = build_arkasr_weights(model->ctx_meta, model->hparams, model->weights); st != TRANSCRIBE_OK) {
        gguf_free(gguf);
        return st;
    }

    const auto backend = params != nullptr ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    const auto device  = params != nullptr ? params->device : nullptr;
    if (const auto st = load_common::init_backends(backend, device, "arkasr", model->plan); st != TRANSCRIBE_OK) {
        gguf_free(gguf);
        return st;
    }
    model->backend         = ggml_backend_name(model->plan.primary);
    model->primary_backend = model->plan.primary;
    model->backend_buffer  = ggml_backend_alloc_ctx_tensors(model->ctx_meta, model->plan.primary);
    if (model->backend_buffer == nullptr) {
        gguf_free(gguf);
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_buffer_set_usage(model->backend_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (const auto st = load_common::stream_tensor_data(loader.path(), gguf, model->ctx_meta, "arkasr");
        st != TRANSCRIBE_OK) {
        gguf_free(gguf);
        return st;
    }
    gguf_free(gguf);

    std::vector<causal_lm::GateUpEntry> entries;
    entries.reserve(model->weights.dec_blocks.size());
    for (auto & block : model->weights.dec_blocks) {
        entries.push_back({ block.ffn_gate_w, block.ffn_up_w, &block.ffn_gate_up_w });
    }
    const auto pack_status =
        causal_lm::pack_gate_up(model->plan.primary, model->hparams.dec_hidden, model->hparams.dec_intermediate,
                                entries, model->packed_gate_up, "arkasr");
    if (pack_status != TRANSCRIBE_OK) {
        return pack_status;
    }

    model->t_load_us = ggml_time_us() - start;
    *out_model       = model.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(transcribe_model *                model,
                               const transcribe_session_params * params,
                               transcribe_session **             out_session) {
    if (model == nullptr || model->arch != &arch || out_session == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto session       = std::make_unique<ArkAsrSession>();
    session->model     = model;
    session->n_threads = params != nullptr ? params->n_threads : 0;
    session->kv_type   = params != nullptr ? params->kv_type : TRANSCRIBE_KV_TYPE_AUTO;
    session->n_ctx     = transcribe_session_params_n_ctx(params);
    bool encoder_flash = true;
    flash::apply_env_overrides(encoder_flash, session->decoder_use_flash);
    *out_session = session.release();
    return TRANSCRIBE_OK;
}

transcribe_status run_window(ArkAsrSession & session, const float * pcm, int n_samples, WindowResult & result) {
    if (pcm == nullptr || n_samples <= 0 || n_samples > kMaxAudioSamples) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto & model = *static_cast<ArkAsrModel *>(session.model);
    if (!model.mel.has_value() || model.plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (session.poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    const int64_t mel_start = ggml_time_us();
    int           n_mels    = 0;
    int           n_frames  = 0;
    if (const auto st = model.mel->compute(pcm, static_cast<size_t>(n_samples), session.mel_buf, n_mels, n_frames,
                                           session.n_threads);
        st != TRANSCRIBE_OK) {
        return st;
    }
    session.t_mel_us += ggml_time_us() - mel_start;

    if (session.sched == nullptr) {
        session.sched = ggml_backend_sched_new(model.plan.scheduler_list.data(), nullptr,
                                               static_cast<int>(model.plan.scheduler_list.size()), 16384, false, true);
        if (session.sched == nullptr) {
            return TRANSCRIBE_ERR_OOM;
        }
    }
    transcribe::configure_sched_n_threads(session.sched, session.n_threads);

    if (!reset_compute_context(session)) {
        return TRANSCRIBE_ERR_OOM;
    }
    const int64_t enc_start = ggml_time_us();
    auto          encoder   = build_encoder_graph(session.compute_ctx, model.weights, model.hparams, n_frames);
    if (encoder.graph == nullptr || encoder.output == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(session.sched);
    if (!ggml_backend_sched_alloc_graph(session.sched, encoder.graph)) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_tensor_set(encoder.mel, session.mel_buf.data(), 0, session.mel_buf.size() * sizeof(float));
    std::vector<int32_t> enc_positions(static_cast<size_t>(encoder.positions->ne[0]));
    for (int i = 0; i < static_cast<int>(enc_positions.size()); ++i) {
        enc_positions[i] = i;
    }
    ggml_backend_tensor_set(encoder.positions, enc_positions.data(), 0, enc_positions.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(session.sched, encoder.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_GGUF;
    }
    const int audio_tokens = encoder.n_audio;
    session.audio_buf.resize(static_cast<size_t>(model.hparams.dec_hidden) * audio_tokens);
    ggml_backend_tensor_get(encoder.output, session.audio_buf.data(), 0, session.audio_buf.size() * sizeof(float));
    session.t_encode_us += ggml_time_us() - enc_start;

    const int64_t        decode_start = ggml_time_us();
    std::vector<int32_t> prompt;
    int                  audio_start = 0;
    build_prompt(model, audio_tokens, prompt, audio_start);
    const int prompt_len = static_cast<int>(prompt.size());
    const int max_new    = std::clamp(n_samples / 1280 + 64, 256, 4096);
    const int ceiling    = context_ceiling(session, model.hparams);
    if (prompt_len + max_new > ceiling) {
        return TRANSCRIBE_ERR_INPUT_TOO_LONG;
    }

    int kv_width = 1;
    while (kv_width < prompt_len + max_new) {
        kv_width *= 2;
    }
    if (session.kv_cache.ctx == nullptr || session.kv_cache.n_ctx < kv_width) {
        session.kv_cache.free();
        const ggml_type kv_type = session.kv_type == TRANSCRIBE_KV_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16;
        if (!causal_lm::kv_init(session.kv_cache, model.plan.primary, kv_width, model.hparams.dec_n_kv_heads,
                                model.hparams.dec_head_dim, model.hparams.dec_n_layers, kv_type)) {
            return TRANSCRIBE_ERR_OOM;
        }
    } else {
        ggml_backend_buffer_clear(session.kv_cache.buffer, 0);
    }
    session.kv_cache.n    = 0;
    session.kv_cache.head = 0;

    if (!reset_compute_context(session)) {
        return TRANSCRIBE_ERR_OOM;
    }
    auto prefill = build_prefill_graph(session.compute_ctx, model.weights, model.hparams, session.kv_cache, prompt_len,
                                       audio_tokens, audio_start, session.decoder_use_flash);
    if (prefill.graph == nullptr || prefill.logits == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(session.sched);
    if (!ggml_backend_sched_alloc_graph(session.sched, prefill.graph)) {
        return TRANSCRIBE_ERR_OOM;
    }
    ggml_backend_tensor_set(prefill.input_ids, prompt.data(), 0, prompt.size() * sizeof(int32_t));
    ggml_backend_tensor_set(prefill.audio, session.audio_buf.data(), 0, session.audio_buf.size() * sizeof(float));
    std::vector<int32_t> positions(prompt.size());
    for (int i = 0; i < prompt_len; ++i) {
        positions[i] = i;
    }
    const auto mask = causal_mask(prompt_len);
    ggml_backend_tensor_set(prefill.positions, positions.data(), 0, positions.size() * sizeof(int32_t));
    ggml_backend_tensor_set(prefill.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_sched_graph_compute(session.sched, prefill.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_GGUF;
    }
    std::vector<float> logits(model.hparams.dec_vocab_size);
    ggml_backend_tensor_get(prefill.logits, logits.data(), 0, logits.size() * sizeof(float));

    std::vector<int32_t> generated;
    generated.reserve(max_new);
    int next = argmax_initial_visible(logits);
    if (next < 0) {
        return TRANSCRIBE_ERR_GGUF;
    }
    generated.push_back(next);

    std::vector<float> logit_mask(static_cast<size_t>(model.hparams.dec_vocab_size), 0.0f);
    for (int id = kFirstSpecialToken; id < model.hparams.dec_vocab_size; ++id) {
        logit_mask[id] = -std::numeric_limits<float>::infinity();
    }
    if (model.special.eos >= 0 && model.special.eos < model.hparams.dec_vocab_size) {
        logit_mask[model.special.eos] = 0.0f;
    }
    bool reached_eos = false;
    for (int count = 1; count < max_new; ++count) {
        if (next == model.special.eos) {
            reached_eos = true;
            break;
        }
        if (session.poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        if (!reset_compute_context(session)) {
            return TRANSCRIBE_ERR_OOM;
        }
        auto step = build_step_graph(session.compute_ctx, model.weights, model.hparams, session.kv_cache, kv_width,
                                     session.decoder_use_flash);
        if (step.graph == nullptr || step.token == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_sched_reset(session.sched);
        if (!ggml_backend_sched_alloc_graph(session.sched, step.graph)) {
            return TRANSCRIBE_ERR_OOM;
        }
        ggml_backend_tensor_set(step.logit_mask, logit_mask.data(), 0, logit_mask.size() * sizeof(float));
        const int32_t input    = next;
        const int32_t position = prompt_len + count - 1;
        const int64_t kv_index = position;
        const auto    smask    = step_mask(kv_width, position);
        ggml_backend_tensor_set(step.input_id, &input, 0, sizeof(input));
        ggml_backend_tensor_set(step.position, &position, 0, sizeof(position));
        ggml_backend_tensor_set(step.kv_index, &kv_index, 0, sizeof(kv_index));
        ggml_backend_tensor_set(step.mask, smask.data(), 0, smask.size() * sizeof(ggml_fp16_t));
        if (ggml_backend_sched_graph_compute(session.sched, step.graph) != GGML_STATUS_SUCCESS) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_get(step.token, &next, 0, sizeof(next));
        generated.push_back(next);
    }
    if (!generated.empty() && generated.back() == model.special.eos) {
        generated.pop_back();
        reached_eos = true;
    }
    result.truncated = !reached_eos;
    session.t_decode_us += ggml_time_us() - decode_start;

    result.raw_text = model.tok.decode(generated.data(), static_cast<int>(generated.size()));
    if (std::getenv("TRANSCRIBE_ARKASR_DEBUG_GEN") != nullptr) {
        std::fprintf(stderr, "arkasr tokens:");
        for (const int32_t id : generated) {
            std::fprintf(stderr, " %d", id);
        }
        std::fprintf(stderr, "\n");
    }
    std::vector<int32_t> visible;
    visible.reserve(generated.size());
    for (const int32_t id : generated) {
        if (id >= 0 && id < kFirstSpecialToken) {
            visible.push_back(id);
        }
    }
    result.text = model.tok.decode(visible.data(), static_cast<int>(visible.size()));
    normalize_transcript(result.text);
    return result.truncated ? TRANSCRIBE_ERR_OUTPUT_TRUNCATED : TRANSCRIBE_OK;
}

transcribe_status run(transcribe_session * base, const float * pcm, int n_samples, const transcribe_run_params *) {
    if (base == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto & session = *static_cast<ArkAsrSession *>(base);
    auto & model   = *static_cast<ArkAsrModel *>(session.model);
    if (!model.mel.has_value() || model.plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    session.t_mel_us      = 0;
    session.t_encode_us   = 0;
    session.t_decode_us   = 0;
    session.was_truncated = false;
    session.raw_text.clear();
    session.full_text.clear();
    session.segments.clear();

    for (int offset = 0; offset < n_samples;) {
        if (session.poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const int    window_samples = choose_window_samples(pcm + offset, n_samples - offset);
        WindowResult window;
        const auto   st = run_window(session, pcm + offset, window_samples, window);
        if (st != TRANSCRIBE_OK && st != TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
            return st;
        }
        session.was_truncated = session.was_truncated || window.truncated;
        if (!window.raw_text.empty()) {
            if (!session.raw_text.empty()) {
                session.raw_text.push_back(' ');
            }
            session.raw_text += window.raw_text;
        }

        const std::string appended = append_with_word_overlap(session.full_text, window.text);
        if (!appended.empty()) {
            transcribe_session::SegmentEntry segment{};
            segment.text  = appended;
            segment.t0_ms = static_cast<int64_t>(offset) * 1000 / model.hparams.sample_rate;
            segment.t1_ms = static_cast<int64_t>(offset + window_samples) * 1000 / model.hparams.sample_rate;
            session.segments.push_back(std::move(segment));
        }

        if (offset + window_samples >= n_samples) {
            break;
        }
        offset += window_samples - kOverlapSamples;
    }

    session.result_kind = TRANSCRIBE_TIMESTAMPS_SEGMENT;
    session.has_result  = true;
    return session.was_truncated ? TRANSCRIBE_ERR_OUTPUT_TRUNCATED : TRANSCRIBE_OK;
}

}  // namespace

extern const Arch arch = {
    "arkasr", load, init_context, run, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
};

}  // namespace transcribe::arkasr
