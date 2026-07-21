// arch/sortformer/stream.cpp - Host-side sync streaming state machine for
// the Sortformer AOSC speaker-cache + FIFO path. Exact ports of NeMo's
// SortformerModules (sortformer_modules.py): streaming_update (sync branch),
// _get_silence_profile, and the exactness-critical _compress_spkcache stack
// (_get_log_pred_scores -> _disable_low_scores -> scores_boost_latest ->
// _boost_topk_scores x2 -> silence pad -> _get_topk_indices -> gather).
//
// Batch size is always 1 here (the CLI path), so every per-batch loop in the
// reference collapses and spk_perm is always None (inference, permute_spk=
// self.training=False). See reports/porting/sortformer/forward-map.md and the
// approved plan for the traced reference call graph.

#include "sortformer.h"
#include "transcribe-debug.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace transcribe::sortformer {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();

// ---- env helpers ----

const char * env_get(const char * name) {
    const char * v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

bool env_int(const char * name, int & out) {
    const char * v = env_get(name);
    if (v == nullptr) {
        return false;
    }
    char *    end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

// One card / validation preset. lc stays at the model default (1); only the
// chunking geometry changes. Mirrors scripts/diar/run_reference_sortformer_nemo.py
// PRESETS plus a "small" preset (forces multi-chunk + compression on the short
// oracle clip for diar.probs parity).
struct Preset {
    const char * name;
    int          chunk_len;
    int          chunk_right_context;
    int          fifo_len;
    int          spkcache_update_period;
    int          spkcache_len;
};

const Preset k_presets[] = {
    { "very_high_latency", 340, 40, 40, 300, 188 },
    { "high_latency",      124, 1, 124, 124, 188 },
    { "low_latency",       6, 7, 188, 144, 188 },
    { "small",             20, 1, 10, 20, 24 },
};

}  // namespace

SortformerStreamParams resolve_stream_params(const SortformerHParams & hp, transcribe_sortformer_preset preset) {
    SortformerStreamParams p;  // score / boost params keep their (verified) defaults
    // GGUF-shipped streaming cfg is the baseline.
    p.chunk_len              = hp.stream_chunk_len > 0 ? hp.stream_chunk_len : p.chunk_len;
    p.spkcache_len           = hp.stream_spkcache_len > 0 ? hp.stream_spkcache_len : p.spkcache_len;
    p.fifo_len               = hp.stream_fifo_len;  // 0 is a valid shipped value
    p.spkcache_update_period = hp.stream_spkcache_update_period > 0 ? hp.stream_spkcache_update_period
                                                                    : p.spkcache_update_period;

    // Public run-ext preset (DEFAULT keeps the GGUF cfg). Range-checked by
    // run_validate before the dispatcher clears the previous result.
    const char * ext_name = nullptr;
    switch (preset) {
        case TRANSCRIBE_SORTFORMER_PRESET_DEFAULT:
            break;
        case TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY:
            ext_name = "very_high_latency";
            break;
        case TRANSCRIBE_SORTFORMER_PRESET_HIGH_LATENCY:
            ext_name = "high_latency";
            break;
        case TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY:
            ext_name = "low_latency";
            break;
    }
    if (ext_name != nullptr) {
        for (const Preset & pr : k_presets) {
            if (std::string(ext_name) == pr.name) {
                p.chunk_len              = pr.chunk_len;
                p.chunk_right_context    = pr.chunk_right_context;
                p.fifo_len               = pr.fifo_len;
                p.spkcache_update_period = pr.spkcache_update_period;
                p.spkcache_len           = pr.spkcache_len;
                break;
            }
        }
    }

    // Named preset env override (validation / DER operating point).
    if (const char * name = env_get("TRANSCRIBE_SORTFORMER_STREAM_PRESET")) {
        for (const Preset & pr : k_presets) {
            if (std::string(name) == pr.name) {
                p.chunk_len              = pr.chunk_len;
                p.chunk_right_context    = pr.chunk_right_context;
                p.fifo_len               = pr.fifo_len;
                p.spkcache_update_period = pr.spkcache_update_period;
                p.spkcache_len           = pr.spkcache_len;
                break;
            }
        }
    }

    // Per-field env overrides (highest precedence).
    env_int("TRANSCRIBE_SORTFORMER_STREAM_CHUNK_LEN", p.chunk_len);
    env_int("TRANSCRIBE_SORTFORMER_STREAM_FIFO_LEN", p.fifo_len);
    env_int("TRANSCRIBE_SORTFORMER_STREAM_SPKCACHE_LEN", p.spkcache_len);
    env_int("TRANSCRIBE_SORTFORMER_STREAM_UPDATE_PERIOD", p.spkcache_update_period);
    env_int("TRANSCRIBE_SORTFORMER_STREAM_RC", p.chunk_right_context);
    env_int("TRANSCRIBE_SORTFORMER_STREAM_LC", p.chunk_left_context);
    return p;
}

// ---- _get_silence_profile (sortformer_modules.py:636-667) ----

// File-local (only streaming_update_sync calls it).
static void get_silence_profile(SortformerStreamState & st, float sil_threshold, const float * pop_embs,
                                const float * pop_preds, int n, int n_spk, int emb_dim) {
    if (n <= 0) {
        return;
    }
    // is_sil[i] = (sum_s preds[i,s]) < sil_threshold ; sil_count = sum is_sil
    std::vector<uint8_t> is_sil(static_cast<size_t>(n));
    int64_t              sil_count = 0;
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int c = 0; c < n_spk; ++c) {
            s += pop_preds[static_cast<size_t>(i) * n_spk + c];
        }
        is_sil[static_cast<size_t>(i)] = (s < sil_threshold) ? 1 : 0;
        sil_count += is_sil[static_cast<size_t>(i)];
    }
    if (sil_count == 0) {
        return;  // has_new_sil false -> unchanged
    }
    const int64_t old_n = st.n_sil_frames;
    const int64_t new_n = old_n + sil_count;
    const float   denom = static_cast<float>(std::max<int64_t>(new_n, 1));
    for (int d = 0; d < emb_dim; ++d) {
        float sil_sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            if (is_sil[static_cast<size_t>(i)]) {
                sil_sum += pop_embs[static_cast<size_t>(i) * emb_dim + d];
            }
        }
        const float old_sum = st.mean_sil_emb[static_cast<size_t>(d)] * static_cast<float>(old_n);
        st.mean_sil_emb[static_cast<size_t>(d)] = (old_sum + sil_sum) / denom;
    }
    st.n_sil_frames = new_n;
}

namespace {

// _get_log_pred_scores (sortformer_modules.py:669-686). scores[i,s] =
// log(clamp(p,thr)) - log(clamp(1-p,thr)) + sum_s log(clamp(1-p,thr)) - log(0.5).
std::vector<float> get_log_pred_scores(const float * preds, int n, int n_spk, float thr) {
    std::vector<float> scores(static_cast<size_t>(n) * n_spk);
    const float        log_half = std::log(0.5f);
    for (int i = 0; i < n; ++i) {
        float l1sum = 0.0f;
        for (int s = 0; s < n_spk; ++s) {
            const float p = preds[static_cast<size_t>(i) * n_spk + s];
            l1sum += std::log(std::max(1.0f - p, thr));
        }
        for (int s = 0; s < n_spk; ++s) {
            const float p  = preds[static_cast<size_t>(i) * n_spk + s];
            const float lp = std::log(std::max(p, thr));
            const float l1 = std::log(std::max(1.0f - p, thr));
            scores[static_cast<size_t>(i) * n_spk + s] = lp - l1 + l1sum - log_half;
        }
    }
    return scores;
}

// _disable_low_scores (sortformer_modules.py:782-808).
void disable_low_scores(const float * preds, std::vector<float> & scores, int n, int n_spk, int min_pos) {
    // Non-speech (preds <= 0.5) -> -inf.
    for (int i = 0; i < n; ++i) {
        for (int s = 0; s < n_spk; ++s) {
            const size_t idx = static_cast<size_t>(i) * n_spk + s;
            if (!(preds[idx] > 0.5f)) {
                scores[idx] = kNegInf;
            }
        }
    }
    // Per speaker: if it has >= min_pos positive-scored frames, disable its
    // non-positive (but speech) frames.
    for (int s = 0; s < n_spk; ++s) {
        int pos_count = 0;
        for (int i = 0; i < n; ++i) {
            if (scores[static_cast<size_t>(i) * n_spk + s] > 0.0f) {
                ++pos_count;
            }
        }
        if (pos_count >= min_pos) {
            for (int i = 0; i < n; ++i) {
                const size_t idx      = static_cast<size_t>(i) * n_spk + s;
                const bool   is_speech = preds[idx] > 0.5f;
                const bool   is_pos    = scores[idx] > 0.0f;
                if (is_speech && !is_pos) {
                    scores[idx] = kNegInf;
                }
            }
        }
    }
}

// _boost_topk_scores (sortformer_modules.py:611-634). For each speaker column,
// the k highest-scored frames get scores -= scale*log(0.5) (log(0.5)<0 =>
// boosts). Tie-break matches ATen CPU topk: value desc, frame index asc.
void boost_topk_scores(std::vector<float> & scores, int n, int n_spk, int k, float scale) {
    if (k <= 0) {
        return;
    }
    // Reference: scores[topk] -= scale * log(0.5). log(0.5) < 0, so the top-k
    // scores are increased (boosted).
    const float delta = scale * std::log(0.5f);  // negative
    std::vector<int> order(static_cast<size_t>(n));
    for (int s = 0; s < n_spk; ++s) {
        std::iota(order.begin(), order.end(), 0);
        const int kk = std::min(k, n);
        std::partial_sort(order.begin(), order.begin() + kk, order.end(),
                          [&](int a, int b) {
                              const float va = scores[static_cast<size_t>(a) * n_spk + s];
                              const float vb = scores[static_cast<size_t>(b) * n_spk + s];
                              if (va != vb) {
                                  return va > vb;  // descending value
                              }
                              return a < b;  // tie -> lower index first
                          });
        for (int j = 0; j < kk; ++j) {
            scores[static_cast<size_t>(order[static_cast<size_t>(j)]) * n_spk + s] -= delta;
        }
    }
}

}  // namespace

// ---- _compress_spkcache (sortformer_modules.py:838-896) ----

void compress_spkcache(SortformerStreamState & st, const SortformerStreamParams & p, int n_spk, int emb_dim) {
    const int N   = st.spkcache_n;          // n_frames (> spkcache_len)
    const int L   = p.spkcache_len;         // target frames
    const int sil = p.spkcache_sil_frames_per_spk;

    const int per_spk = L / n_spk - sil;
    const int strong  = static_cast<int>(std::floor(per_spk * p.strong_boost_rate));
    const int weak    = static_cast<int>(std::floor(per_spk * p.weak_boost_rate));
    const int min_pos = static_cast<int>(std::floor(per_spk * p.min_pos_scores_rate));

    const float * preds = st.spkcache_preds.data();  // [N, n_spk]

    // 1. log-pred scores, disable low scores.
    std::vector<float> scores = get_log_pred_scores(preds, N, n_spk, p.pred_score_threshold);
    disable_low_scores(preds, scores, N, n_spk, min_pos);

    // 2. boost latest frames (rows >= spkcache_len). += on -inf stays -inf.
    if (p.scores_boost_latest > 0.0f) {
        for (int i = L; i < N; ++i) {
            for (int s = 0; s < n_spk; ++s) {
                scores[static_cast<size_t>(i) * n_spk + s] += p.scores_boost_latest;
            }
        }
    }

    // 3. strong then weak top-k boosting.
    boost_topk_scores(scores, N, n_spk, strong, /*scale=*/2.0f);
    boost_topk_scores(scores, N, n_spk, weak, /*scale=*/1.0f);

    // 4. append `sil` rows of +inf per speaker; n_frames = N + sil.
    const int n_frames        = N + sil;
    const int n_frames_no_sil = N;
    std::vector<float> scores_ext(static_cast<size_t>(n_frames) * n_spk);
    std::copy(scores.begin(), scores.end(), scores_ext.begin());
    for (int i = N; i < n_frames; ++i) {
        for (int s = 0; s < n_spk; ++s) {
            scores_ext[static_cast<size_t>(i) * n_spk + s] = kPosInf;
        }
    }

    // 5. _get_topk_indices: flatten permute(0,2,1) -> flat[s*n_frames + i];
    // top-L largest (value desc, flat-idx asc), -inf picks -> max_index, sort
    // ascending, derive frame idx + is_disabled.
    const int64_t M = static_cast<int64_t>(n_spk) * n_frames;
    std::vector<int64_t> flat(static_cast<size_t>(M));
    std::iota(flat.begin(), flat.end(), int64_t{ 0 });
    auto flat_val = [&](int64_t f) -> float {
        const int s = static_cast<int>(f / n_frames);
        const int i = static_cast<int>(f % n_frames);
        return scores_ext[static_cast<size_t>(i) * n_spk + s];
    };
    const int kk = static_cast<int>(std::min<int64_t>(L, M));
    std::partial_sort(flat.begin(), flat.begin() + kk, flat.end(),
                      [&](int64_t a, int64_t b) {
                          const float va = flat_val(a);
                          const float vb = flat_val(b);
                          if (va != vb) {
                              return va > vb;
                          }
                          return a < b;
                      });
    std::vector<int64_t> picks(static_cast<size_t>(L));
    for (int j = 0; j < L; ++j) {
        if (j < kk) {
            const int64_t f = flat[static_cast<size_t>(j)];
            picks[static_cast<size_t>(j)] = (flat_val(f) == kNegInf) ? p.max_index : f;
        } else {
            picks[static_cast<size_t>(j)] = p.max_index;  // fewer than L finite picks
        }
    }
    std::sort(picks.begin(), picks.end());

    std::vector<int>  frame_idx(static_cast<size_t>(L));
    std::vector<char> is_disabled(static_cast<size_t>(L));
    for (int j = 0; j < L; ++j) {
        const int64_t idx      = picks[static_cast<size_t>(j)];
        bool          disabled = (idx == p.max_index);
        int           f        = disabled ? 0 : static_cast<int>(idx % n_frames);
        if (!disabled && f >= n_frames_no_sil) {
            disabled = true;  // came from a +inf silence pad row
        }
        if (disabled) {
            f = 0;
        }
        frame_idx[static_cast<size_t>(j)]   = f;
        is_disabled[static_cast<size_t>(j)] = disabled ? 1 : 0;
    }

    // 6. gather embeddings + preds; disabled -> mean_sil_emb / 0.
    std::vector<float> new_emb(static_cast<size_t>(L) * emb_dim);
    std::vector<float> new_preds(static_cast<size_t>(L) * n_spk);
    for (int j = 0; j < L; ++j) {
        const int f = frame_idx[static_cast<size_t>(j)];
        if (is_disabled[static_cast<size_t>(j)]) {
            std::copy(st.mean_sil_emb.begin(), st.mean_sil_emb.end(),
                      new_emb.begin() + static_cast<size_t>(j) * emb_dim);
            // preds row already zero-initialized.
        } else {
            std::copy(st.spkcache.begin() + static_cast<size_t>(f) * emb_dim,
                      st.spkcache.begin() + static_cast<size_t>(f + 1) * emb_dim,
                      new_emb.begin() + static_cast<size_t>(j) * emb_dim);
            std::copy(st.spkcache_preds.begin() + static_cast<size_t>(f) * n_spk,
                      st.spkcache_preds.begin() + static_cast<size_t>(f + 1) * n_spk,
                      new_preds.begin() + static_cast<size_t>(j) * n_spk);
        }
    }

    // Parity dump (residual-uncertainty #1): emit the per-compression selected
    // frame indices + is_disabled mask + gathered preds so a matching NeMo dump
    // can be compared index-for-index. Gated behind an explicit env on top of
    // the normal dump-enabled flag so DER/validate runs are unaffected.
    if (transcribe::debug::enabled() && env_get("TRANSCRIBE_SORTFORMER_COMPRESS_DUMP") != nullptr) {
        const int k = st.compress_count;
        char name[64];
        std::vector<float> fidx(static_cast<size_t>(L)), fdis(static_cast<size_t>(L));
        for (int j = 0; j < L; ++j) {
            fidx[static_cast<size_t>(j)] = static_cast<float>(frame_idx[static_cast<size_t>(j)]);
            fdis[static_cast<size_t>(j)] = is_disabled[static_cast<size_t>(j)] ? 1.0f : 0.0f;
        }
        const long long li = L;
        const long long shp_in[2] = { N, n_spk };
        std::snprintf(name, sizeof(name), "compress.%03d.input_preds", k);
        transcribe::debug::dump_host_f32(name, st.spkcache_preds.data(), static_cast<long long>(N) * n_spk,
                                         shp_in, 2, "compress");
        std::snprintf(name, sizeof(name), "compress.%03d.frame_idx", k);
        transcribe::debug::dump_host_f32(name, fidx.data(), li, &li, 1, "compress");
        std::snprintf(name, sizeof(name), "compress.%03d.is_disabled", k);
        transcribe::debug::dump_host_f32(name, fdis.data(), li, &li, 1, "compress");
        const long long shp[2] = { L, n_spk };
        std::snprintf(name, sizeof(name), "compress.%03d.spkcache_preds", k);
        transcribe::debug::dump_host_f32(name, new_preds.data(), static_cast<long long>(L) * n_spk, shp, 2,
                                         "compress");
    }
    ++st.compress_count;

    st.spkcache       = std::move(new_emb);
    st.spkcache_preds = std::move(new_preds);
    st.spkcache_n     = L;
}

// ---- streaming_update (sync branch, sortformer_modules.py:526-609) ----

void streaming_update_sync(SortformerStreamState & st, const SortformerStreamParams & p, int n_spk, int emb_dim,
                           const std::vector<float> & chunk_embs, int T_diar,
                           const std::vector<float> & preds, int /*T_concat*/, int lc, int rc) {
    const int S = st.spkcache_n;         // spkcache_len (local, pre-update)
    const int F = st.fifo_n;             // fifo_len (local, pre-update)
    const int C = T_diar - lc - rc;      // chunk_len (middle frames)

    // fifo_preds = preds[S : S+F]  (rebuilt from the concat preds each call).
    st.fifo_preds.assign(preds.begin() + static_cast<size_t>(S) * n_spk,
                         preds.begin() + static_cast<size_t>(S + F) * n_spk);
    // chunk_preds = preds[S+F+lc : S+F+lc+C].
    std::vector<float> chunk_preds(preds.begin() + static_cast<size_t>(S + F + lc) * n_spk,
                                   preds.begin() + static_cast<size_t>(S + F + lc + C) * n_spk);

    // Append the middle chunk (embs + preds) to the FIFO.
    st.fifo.insert(st.fifo.end(), chunk_embs.begin() + static_cast<size_t>(lc) * emb_dim,
                   chunk_embs.begin() + static_cast<size_t>(lc + C) * emb_dim);
    st.fifo_preds.insert(st.fifo_preds.end(), chunk_preds.begin(), chunk_preds.end());
    st.fifo_n = F + C;

    if (F + C > p.fifo_len) {
        int pop = p.spkcache_update_period;
        pop     = std::max(pop, C - p.fifo_len + F);
        pop     = std::min(pop, F + C);

        std::vector<float> pop_embs(st.fifo.begin(), st.fifo.begin() + static_cast<size_t>(pop) * emb_dim);
        std::vector<float> pop_preds(st.fifo_preds.begin(),
                                     st.fifo_preds.begin() + static_cast<size_t>(pop) * n_spk);

        get_silence_profile(st, p.sil_threshold, pop_embs.data(), pop_preds.data(), pop, n_spk, emb_dim);

        // fifo = fifo[pop:].
        st.fifo.erase(st.fifo.begin(), st.fifo.begin() + static_cast<size_t>(pop) * emb_dim);
        st.fifo_preds.erase(st.fifo_preds.begin(), st.fifo_preds.begin() + static_cast<size_t>(pop) * n_spk);
        st.fifo_n = (F + C) - pop;

        // Append pop-out to speaker cache.
        st.spkcache.insert(st.spkcache.end(), pop_embs.begin(), pop_embs.end());
        st.spkcache_n = S + pop;
        if (st.spkcache_preds_init) {
            st.spkcache_preds.insert(st.spkcache_preds.end(), pop_preds.begin(), pop_preds.end());
        }

        if (st.spkcache_n > p.spkcache_len) {
            if (!st.spkcache_preds_init) {
                // First compress: seed spkcache_preds = preds[:S] ++ pop_preds.
                st.spkcache_preds.assign(preds.begin(), preds.begin() + static_cast<size_t>(S) * n_spk);
                st.spkcache_preds.insert(st.spkcache_preds.end(), pop_preds.begin(), pop_preds.end());
                st.spkcache_preds_init = true;
            }
            compress_spkcache(st, p, n_spk, emb_dim);
        }
    }

    // Accumulate the chunk's predictions.
    st.total_preds.insert(st.total_preds.end(), chunk_preds.begin(), chunk_preds.end());
    st.total_n += C;
}

}  // namespace transcribe::sortformer
