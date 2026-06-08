// transcribe-batch-util.cpp - see transcribe-batch-util.h.

#include "transcribe-batch-util.h"
#include "transcribe-log.h"
#include "transcribe-session.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <thread>
#include <vector>

namespace transcribe {

bool parallel_for_all(int n, int n_threads,
                      const std::function<bool(int)> & work)
{
    if (n <= 0) return true;
    if (n_threads <= 0) {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
    }
    n_threads = std::max(1, std::min(n, n_threads));

    std::atomic<int>  next{0};
    std::atomic<bool> all_ok{true};
    auto worker = [&]() {
        int i;
        while ((i = next.fetch_add(1, std::memory_order_relaxed)) < n) {
            if (!work(i)) all_ok.store(false, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(n_threads - 1));
    for (int w = 0; w < n_threads - 1; ++w) pool.emplace_back(worker);
    worker();  // the calling thread participates
    for (auto & th : pool) th.join();

    return all_ok.load(std::memory_order_relaxed);
}

void pack_pad_channel_major(
    std::vector<float> &                    dst,
    const std::vector<std::vector<float>> & src,
    const std::vector<int> &                lens,
    int                                     n_ch,
    int                                     T_max)
{
    const int    n   = static_cast<int>(src.size());
    const size_t per = static_cast<size_t>(n_ch) * static_cast<size_t>(T_max);
    dst.assign(per * static_cast<size_t>(n), 0.0f);
    for (int b = 0; b < n; ++b) {
        const int     nb  = lens[static_cast<size_t>(b)];
        const float * s   = src[static_cast<size_t>(b)].data();
        float *       d   = dst.data() + static_cast<size_t>(b) * per;
        for (int c = 0; c < n_ch; ++c) {
            std::copy(s + static_cast<size_t>(c) * nb,
                      s + static_cast<size_t>(c) * nb + nb,
                      d + static_cast<size_t>(c) * T_max);
        }
    }
}

void fill_keypad_mask(ggml_tensor *            mask,
                      const std::vector<int> & real_lens,
                      int                      T,
                      int                      n)
{
    if (mask == nullptr) return;
    const float ninf = -std::numeric_limits<float>::infinity();
    std::vector<float> buf(static_cast<size_t>(T) * n);
    for (int b = 0; b < n; ++b) {
        const int real = real_lens[static_cast<size_t>(b)];
        for (int k = 0; k < T; ++k) {
            buf[static_cast<size_t>(b) * T + k] = (k < real) ? 0.0f : ninf;
        }
    }
    ggml_backend_tensor_set(mask, buf.data(), 0, buf.size() * sizeof(float));
}

void fill_valid_frame_mask(ggml_tensor *            mask,
                           const std::vector<int> & real_lens,
                           int                      T,
                           int                      n)
{
    if (mask == nullptr) return;
    std::vector<float> buf(static_cast<size_t>(T) * n);
    for (int b = 0; b < n; ++b) {
        const int real = real_lens[static_cast<size_t>(b)];
        for (int t = 0; t < T; ++t) {
            buf[static_cast<size_t>(b) * T + t] = (t < real) ? 1.0f : 0.0f;
        }
    }
    ggml_backend_tensor_set(mask, buf.data(), 0, buf.size() * sizeof(float));
}

transcribe_status decode_batch_slices(
    transcribe_session *  session,
    int                   n,
    const float *         host_buf,
    std::size_t           utt_elems,
    int64_t               total_encode_us,
    int64_t               total_mel_us,
    const std::function<transcribe_status(int b, const float * slice)> & decode_fn)
{
    const int64_t enc_per_utt = total_encode_us / std::max(1, n);
    const int64_t mel_per_utt = total_mel_us    / std::max(1, n);
    for (int b = 0; b < n; ++b) {
        if (session->poll_abort()) return TRANSCRIBE_ERR_ABORTED;
        session->clear_result();
        const float * slice = host_buf + static_cast<size_t>(b) * utt_elems;
        const transcribe_status st = decode_fn(b, slice);
        auto rs = session->capture_result(st);
        rs.t_mel_us    = mel_per_utt;
        rs.t_encode_us = enc_per_utt;
        session->batch_results.push_back(std::move(rs));
    }
    return TRANSCRIBE_OK;
}

transcribe_status run_batched_encdec_step_loop(
    transcribe_session *                session,
    ggml_backend_sched_t                sched,
    const EncDecRebuildFn &             rebuild,
    const std::vector<int32_t> &        prompt_ids,
    int                                 prompt_len,
    int                                 init_window,
    int                                 max_new,
    int                                 max_n_kv,
    int32_t                             eos_id,
    int                                 n_batch,
    const std::vector<char> &           valid,
    std::vector<std::vector<int32_t>> & generated,
    int *                               n_steps_out,
    std::vector<char> *                 truncated_out)
{
    const int         n        = n_batch;
    const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());

    int          kv_window = init_window;
    EncDecStepIO io {};
    if (!rebuild(kv_window, io)) return TRANSCRIBE_ERR_GGUF;

    std::vector<ggml_fp16_t> smask(static_cast<size_t>(kv_window) * n, f16_ninf);
    std::vector<int32_t> tok_buf(n, 0), pos_buf(n, 0), argmax_buf(n, 0);
    std::vector<int64_t> kvidx_buf(n, 0);
    std::vector<char>    finished(n, 0);
    std::vector<int32_t> next_tok(n, 0);
    for (int b = 0; b < n; ++b) if (!valid[b]) finished[b] = 1;

    int n_steps = 0;
    auto run_step = [&](int posv) -> transcribe_status {
        for (int b = 0; b < n; ++b) {
            pos_buf[b] = posv; kvidx_buf[b] = posv;
            smask[static_cast<size_t>(b) * kv_window + posv] = f16_zero;
        }
        ggml_backend_tensor_set(io.token_ids, tok_buf.data(), 0, n * sizeof(int32_t));
        ggml_backend_tensor_set(io.pos_ids,   pos_buf.data(), 0, n * sizeof(int32_t));
        ggml_backend_tensor_set(io.kv_idx,    kvidx_buf.data(), 0, n * sizeof(int64_t));
        ggml_backend_tensor_set(io.self_mask, smask.data(), 0,
                                smask.size() * sizeof(ggml_fp16_t));
        if (ggml_backend_sched_graph_compute(sched, io.graph) != GGML_STATUS_SUCCESS)
            return TRANSCRIBE_ERR_GGUF;
        ggml_backend_tensor_get(io.argmax, argmax_buf.data(), 0, n * sizeof(int32_t));
        ++n_steps;
        return TRANSCRIBE_OK;
    };

    // Grow the read window (rebuild graph + widen mask) so position `posv` fits.
    auto ensure_window = [&](int posv) -> bool {
        if (posv + 1 <= kv_window) return true;
        int win = kv_window;
        while (win < posv + 1 && win < max_n_kv) win *= 2;
        if (win > max_n_kv) win = max_n_kv;
        if (win == kv_window) return true;
        std::vector<ggml_fp16_t> wider(static_cast<size_t>(win) * n, f16_ninf);
        for (int b = 0; b < n; ++b)
            std::fill(wider.data() + static_cast<size_t>(b) * win,
                      wider.data() + static_cast<size_t>(b) * win + posv, f16_zero);
        smask.swap(wider);
        kv_window = win;
        return rebuild(kv_window, io);
    };

    // Prompt feed: prompt_len sequential steps (uniform tokens across rows).
    int pos = 0;
    for (; pos < prompt_len; ++pos) {
        if (session->poll_abort()) return TRANSCRIBE_ERR_ABORTED;
        if (!ensure_window(pos)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "batched decode: step graph allocation failed — out of memory. "
                "Lower transcribe_session_params.n_ctx or the batch size.");
            return TRANSCRIBE_ERR_OOM;
        }
        for (int b = 0; b < n; ++b) tok_buf[b] = prompt_ids[static_cast<size_t>(pos)];
        if (run_step(pos) != TRANSCRIBE_OK) return TRANSCRIBE_ERR_GGUF;
    }
    // argmax from the last prompt position = first generated token.
    for (int b = 0; b < n; ++b) {
        if (finished[b]) continue;
        next_tok[b] = argmax_buf[b];
        if (next_tok[b] == eos_id) finished[b] = 1;
        else generated[b].push_back(next_tok[b]);
    }

    // Generation.
    for (int produced = 1; produced < max_new; ++produced, ++pos) {
        if (session->poll_abort()) return TRANSCRIBE_ERR_ABORTED;
        bool all_done = true;
        for (int b = 0; b < n; ++b) if (!finished[b]) { all_done = false; break; }
        if (all_done || pos + 1 > max_n_kv) break;
        if (!ensure_window(pos)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "batched decode: step graph allocation failed — out of memory. "
                "Lower transcribe_session_params.n_ctx or the batch size.");
            return TRANSCRIBE_ERR_OOM;
        }
        for (int b = 0; b < n; ++b) tok_buf[b] = finished[b] ? eos_id : next_tok[b];
        if (run_step(pos) != TRANSCRIBE_OK) return TRANSCRIBE_ERR_GGUF;
        for (int b = 0; b < n; ++b) {
            if (finished[b]) continue;
            next_tok[b] = argmax_buf[b];
            if (next_tok[b] == eos_id) finished[b] = 1;
            else generated[b].push_back(next_tok[b]);
        }
    }

    if (n_steps_out != nullptr) *n_steps_out = n_steps;

    // A valid row that never reached eos was cut off at the generation budget
    // or the context window — report it as truncated so the family can return
    // per-utterance TRANSCRIBE_ERR_OUTPUT_TRUNCATED. See docs/input-limits.md.
    if (truncated_out != nullptr) {
        truncated_out->assign(n, 0);
        for (int b = 0; b < n; ++b) {
            (*truncated_out)[b] = (valid[b] && !finished[b]) ? 1 : 0;
        }
    }
    return TRANSCRIBE_OK;
}

} // namespace transcribe
