// arch/nemotron3_diar/stream.cpp - host-side sync streaming state machine
// for the AOSC speaker cache + FIFO, plus the incremental speaker-segment
// builder. Exact ports of NeMo Speech sortformer_modules.py @ cf724ac337d1:
// streaming_update (sync branch) and the _compress_spkcache stack
// (_get_log_pred_scores -> _disable_low_scores -> scores_boost_latest ->
// _boost_topk_scores x2 -> silence pad -> _get_topk_indices -> gather).
//
// Forked from src/arch/sortformer/stream.cpp (NeMo 2.x port). Differences
// for this checkpoint (use_learnable_sil_emb=True):
//   - no _get_silence_profile running mean: disabled cache slots are filled
//     with the learned `learnable_sil_emb` (GGUF diar.sil_emb);
//   - the cache runs on the x8 average-pooled 80 ms probabilities.
// Batch size is always 1 per state (spk_perm is None at inference).

#include "nemotron3_diar.h"
#include "torch_logf.h"
#include "transcribe-debug.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace transcribe::nemotron3_diar {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();

const char * env_get(const char * name) {
    const char * v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

// Card presets (NVIDIA model card) + "small" (validation only: tiny cache so
// the short oracle clip exercises many compressions). lc is 0 for all.
struct Preset {
    const char * name;
    int          spkcache_len;
    int          fifo_len;
    int          chunk_len;
    int          chunk_right_context;
    int          spkcache_update_period;
};

const Preset k_presets[] = {
    { "very_high_latency", 264, 40,  340, 40, 300 },
    { "low_latency",       264, 264, 9,   4,  222 },
    { "very_low_latency",  264, 264, 6,   2,  222 },
    { "ultra_low_latency", 264, 264, 3,   1,  222 },
    { "small",             24,  10,  20,  2,  20  },
};

void apply_preset(StreamParams & p, const Preset & pr) {
    p.spkcache_len           = pr.spkcache_len;
    p.fifo_len               = pr.fifo_len;
    p.chunk_len              = pr.chunk_len;
    p.chunk_right_context    = pr.chunk_right_context;
    p.spkcache_update_period = pr.spkcache_update_period;
}

bool apply_named_preset(StreamParams & p, const char * name) {
    for (const Preset & pr : k_presets) {
        if (std::string(name) == pr.name) {
            apply_preset(p, pr);
            return true;
        }
    }
    return false;
}

}  // namespace

bool preset_is_valid(transcribe_nemotron3_diar_preset preset) {
    switch (preset) {
        case TRANSCRIBE_NEMOTRON3_DIAR_PRESET_DEFAULT:
        case TRANSCRIBE_NEMOTRON3_DIAR_PRESET_VERY_HIGH_LATENCY:
        case TRANSCRIBE_NEMOTRON3_DIAR_PRESET_LOW_LATENCY:
        case TRANSCRIBE_NEMOTRON3_DIAR_PRESET_VERY_LOW_LATENCY:
        case TRANSCRIBE_NEMOTRON3_DIAR_PRESET_ULTRA_LOW_LATENCY:
            return true;
    }
    return false;
}

StreamParams resolve_stream_params(const Nemotron3DiarHParams & hp, transcribe_nemotron3_diar_preset preset) {
    StreamParams p;
    // GGUF-shipped operating point is the baseline.
    p.spkcache_len                = hp.stream_spkcache_len;
    p.fifo_len                    = hp.stream_fifo_len;
    p.chunk_len                   = hp.stream_chunk_len;
    p.chunk_right_context         = hp.stream_chunk_right_context;
    p.spkcache_update_period      = hp.stream_spkcache_update_period;
    p.spkcache_sil_frames_per_spk = hp.spkcache_sil_frames_per_spk;
    p.pred_score_threshold        = hp.pred_score_threshold;
    p.scores_boost_latest         = hp.scores_boost_latest;
    p.strong_boost_rate           = hp.strong_boost_rate;
    p.weak_boost_rate             = hp.weak_boost_rate;
    p.min_pos_scores_rate         = hp.min_pos_scores_rate;
    p.max_index                   = hp.max_index;

    const char * name = nullptr;
    switch (preset) {
        case TRANSCRIBE_NEMOTRON3_DIAR_PRESET_DEFAULT:
            break;
        case TRANSCRIBE_NEMOTRON3_DIAR_PRESET_VERY_HIGH_LATENCY:
            name = "very_high_latency";
            break;
        case TRANSCRIBE_NEMOTRON3_DIAR_PRESET_LOW_LATENCY:
            name = "low_latency";
            break;
        case TRANSCRIBE_NEMOTRON3_DIAR_PRESET_VERY_LOW_LATENCY:
            name = "very_low_latency";
            break;
        case TRANSCRIBE_NEMOTRON3_DIAR_PRESET_ULTRA_LOW_LATENCY:
            name = "ultra_low_latency";
            break;
    }
    if (name != nullptr) {
        apply_named_preset(p, name);
    }
    // Validation / DER operating-point override (highest precedence).
    if (const char * env = env_get("TRANSCRIBE_NEMOTRON3_DIAR_PRESET")) {
        apply_named_preset(p, env);
    }
    return p;
}

namespace {

// _get_log_pred_scores: log(clamp(p,thr)) - log(clamp(1-p,thr)) + sum_s log(clamp(1-p,thr)) - log(0.5).
// Bit-exact with the torch CPU (aarch64) evaluation: the boundary scores the
// compression top-k separates are ~1e-7 apart, so log uses torch's Sleef
// algorithm (torch_logf.h) and the per-frame sum reproduces ATen's reduction
// order (4 accumulators acc[s % 4], combined ((a0 + a1) + a2) + a3; verified
// bit-identical on 1812 reference rows).
std::vector<float> get_log_pred_scores(const float * preds, int n, int n_spk, float thr) {
    std::vector<float> scores(static_cast<size_t>(n) * n_spk);
    std::vector<float> l1(static_cast<size_t>(n_spk));
    const float        log_half = static_cast<float>(std::log(0.5));  // math.log(0.5) cast to the float op
    for (int i = 0; i < n; ++i) {
        float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int s = 0; s < n_spk; ++s) {
            const float p              = preds[static_cast<size_t>(i) * n_spk + s];
            l1[static_cast<size_t>(s)] = torch_logf::logf_u10(std::max(1.0f - p, thr));
            acc[s % 4]                 = acc[s % 4] + l1[static_cast<size_t>(s)];
        }
        const float l1sum = ((acc[0] + acc[1]) + acc[2]) + acc[3];
        for (int s = 0; s < n_spk; ++s) {
            const float p                              = preds[static_cast<size_t>(i) * n_spk + s];
            const float lp                             = torch_logf::logf_u10(std::max(p, thr));
            scores[static_cast<size_t>(i) * n_spk + s] = ((lp - l1[static_cast<size_t>(s)]) + l1sum) - log_half;
        }
    }
    return scores;
}

// _disable_low_scores.
void disable_low_scores(const float * preds, std::vector<float> & scores, int n, int n_spk, int min_pos) {
    for (int i = 0; i < n; ++i) {
        for (int s = 0; s < n_spk; ++s) {
            const size_t idx = static_cast<size_t>(i) * n_spk + s;
            if (!(preds[idx] > 0.5f)) {
                scores[idx] = kNegInf;
            }
        }
    }
    for (int s = 0; s < n_spk; ++s) {
        int pos_count = 0;
        for (int i = 0; i < n; ++i) {
            if (scores[static_cast<size_t>(i) * n_spk + s] > 0.0f) {
                ++pos_count;
            }
        }
        if (pos_count >= min_pos) {
            for (int i = 0; i < n; ++i) {
                const size_t idx = static_cast<size_t>(i) * n_spk + s;
                if (preds[idx] > 0.5f && !(scores[idx] > 0.0f)) {
                    scores[idx] = kNegInf;
                }
            }
        }
    }
}

// torch.topk(values, k, largest=True, sorted=False) on CPU: the exact
// selection ATen makes (aten/src/ATen/native/TopKImpl.h topk_impl_loop).
// The queue of (value, index) pairs is filled in index order, then
// std::partial_sort (k * 64 <= n) or std::nth_element with a strict `>`
// comparator. Which of several EQUAL values is kept is decided by that
// algorithm's swap sequence, not by index order, and exact ties are common
// here (saturated sigmoids give identical scores), so a "value desc, index
// asc" rule selects different cache frames than NeMo. Same libc++
// algorithms, same comparator, same call => same selection. Returns the
// selected indices (in queue order; callers treat them as a set).
std::vector<int64_t> aten_topk_indices(const std::vector<double> & values, int64_t k) {
    using elem_t          = std::pair<double, int64_t>;
    const int64_t       n = static_cast<int64_t>(values.size());
    std::vector<elem_t> queue(static_cast<size_t>(n));
    for (int64_t j = 0; j < n; ++j) {
        queue[static_cast<size_t>(j)] = { values[static_cast<size_t>(j)], j };
    }
    auto gt = [](const elem_t & x, const elem_t & y) -> bool {
        return (std::isnan(x.first) && !std::isnan(y.first)) || (x.first > y.first);
    };
    if (k * 64 <= n) {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(), gt);
    } else {
        std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(), gt);
    }
    std::vector<int64_t> out(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
        out[static_cast<size_t>(j)] = queue[static_cast<size_t>(j)].second;
    }
    return out;
}

// _boost_topk_scores: the k highest frames per speaker get -= scale*log(0.5)
// (torch.topk over the frame axis, sorted=False).
void boost_topk_scores(std::vector<float> & scores, int n, int n_spk, int k, float scale) {
    if (k <= 0) {
        return;
    }
    const float         delta = scale * std::log(0.5f);
    std::vector<double> col(static_cast<size_t>(n));
    for (int s = 0; s < n_spk; ++s) {
        for (int i = 0; i < n; ++i) {
            col[static_cast<size_t>(i)] = scores[static_cast<size_t>(i) * n_spk + s];
        }
        for (const int64_t i : aten_topk_indices(col, std::min(k, n))) {
            scores[static_cast<size_t>(i) * n_spk + s] -= delta;
        }
    }
}

// Minimal .npy reader for the reference dumper's 1-D compression dumps
// (descr '<i8' or '|b1'). Returns false on any format surprise.
bool read_npy_1d(const std::string & path, std::vector<int64_t> & out) {
    std::ifstream f(path, std::ios::binary);
    char          magic[8];
    if (!f.read(magic, 8) || std::string(magic + 1, 5) != "NUMPY") {
        return false;
    }
    uint32_t hlen = 0;
    if (magic[6] == 1) {
        uint16_t h16 = 0;
        f.read(reinterpret_cast<char *>(&h16), 2);
        hlen = h16;
    } else {
        f.read(reinterpret_cast<char *>(&hlen), 4);
    }
    std::string header(hlen, '\0');
    if (!f.read(header.data(), hlen)) {
        return false;
    }
    const size_t sp = header.find("'shape': (");
    if (sp == std::string::npos) {
        return false;
    }
    const long long n = std::strtoll(header.c_str() + sp + 10, nullptr, 10);
    out.assign(static_cast<size_t>(n), 0);
    if (header.find("'<i8'") != std::string::npos) {
        f.read(reinterpret_cast<char *>(out.data()), n * 8);
    } else if (header.find("'|b1'") != std::string::npos) {
        std::vector<uint8_t> b(static_cast<size_t>(n));
        f.read(reinterpret_cast<char *>(b.data()), n);
        for (long long i = 0; i < n; ++i) {
            out[static_cast<size_t>(i)] = b[static_cast<size_t>(i)];
        }
    } else {
        return false;
    }
    return static_cast<bool>(f);
}

// _compress_spkcache (permute_spk=False, use_learnable_sil_emb=True).
void compress_spkcache(StreamState &              st,
                       const StreamParams &       p,
                       int                        n_spk,
                       int                        emb_dim,
                       const std::vector<float> & sil_emb) {
    const int N   = st.spkcache_n;
    const int L   = p.spkcache_len;
    const int sil = p.spkcache_sil_frames_per_spk;

    const int per_spk = L / n_spk - sil;
    const int strong  = static_cast<int>(std::floor(per_spk * p.strong_boost_rate));
    const int weak    = static_cast<int>(std::floor(per_spk * p.weak_boost_rate));
    const int min_pos = static_cast<int>(std::floor(per_spk * p.min_pos_scores_rate));

    const float * preds = st.spkcache_preds.data();

    std::vector<float> scores = get_log_pred_scores(preds, N, n_spk, p.pred_score_threshold);
    disable_low_scores(preds, scores, N, n_spk, min_pos);

    if (p.scores_boost_latest > 0.0f) {
        for (int i = L; i < N; ++i) {
            for (int s = 0; s < n_spk; ++s) {
                scores[static_cast<size_t>(i) * n_spk + s] += p.scores_boost_latest;
            }
        }
    }

    boost_topk_scores(scores, N, n_spk, strong, /*scale=*/2.0f);
    boost_topk_scores(scores, N, n_spk, weak, /*scale=*/1.0f);

    const int          n_frames = N + sil;
    std::vector<float> scores_ext(static_cast<size_t>(n_frames) * n_spk);
    std::copy(scores.begin(), scores.end(), scores_ext.begin());
    for (int i = N; i < n_frames; ++i) {
        for (int s = 0; s < n_spk; ++s) {
            scores_ext[static_cast<size_t>(i) * n_spk + s] = kPosInf;
        }
    }

    // _get_topk_indices over flat[s*n_frames + i].
    const int64_t       M = static_cast<int64_t>(n_spk) * n_frames;
    std::vector<double> flat(static_cast<size_t>(M));
    for (int s = 0; s < n_spk; ++s) {
        for (int i = 0; i < n_frames; ++i) {
            flat[static_cast<size_t>(s) * n_frames + i] = scores_ext[static_cast<size_t>(i) * n_spk + s];
        }
    }
    const int                  kk  = static_cast<int>(std::min<int64_t>(L, M));
    const std::vector<int64_t> sel = aten_topk_indices(flat, kk);
    std::vector<int64_t>       picks(static_cast<size_t>(L), p.max_index);
    for (int j = 0; j < kk; ++j) {
        const int64_t f               = sel[static_cast<size_t>(j)];
        picks[static_cast<size_t>(j)] = (flat[static_cast<size_t>(f)] == kNegInf) ? p.max_index : f;
    }
    std::sort(picks.begin(), picks.end());

    std::vector<int>  frame_idx(static_cast<size_t>(L));
    std::vector<char> is_disabled(static_cast<size_t>(L));
    for (int j = 0; j < L; ++j) {
        const int64_t idx      = picks[static_cast<size_t>(j)];
        bool          disabled = (idx == p.max_index);
        int           f        = static_cast<int>(idx % n_frames);
        if (!disabled && f >= N) {
            disabled = true;  // a +inf silence pad row
        }
        if (disabled) {
            f = 0;
        }
        frame_idx[static_cast<size_t>(j)]   = f;
        is_disabled[static_cast<size_t>(j)] = disabled ? 1 : 0;
    }

    // Validation isolation (like reference-mel injection): with
    // TRANSCRIBE_NEMOTRON3_DIAR_COMPRESS_FROM_REF=<dir> (and the dumper
    // enabled), take compression k's selected frames from the reference
    // dumper's compress.NNN.{topk_indices,is_disabled}.npy and gather from
    // this port's own cache. The top-k selection is discontinuous: a ~1e-6
    // GEMM difference can move a boundary score across a ~1e-5 gap and flip a
    // pick, after which every later chunk differs. Forcing the reference's
    // picks lets diar.probs gate the graph + bookkeeping at tight tolerance;
    // the selection itself is gated separately (scripts/diar/
    // check_nemotron3_diar_compress.py replays NeMo on this port's inputs).
    if (const char * ref_dir = env_get("TRANSCRIBE_NEMOTRON3_DIAR_COMPRESS_FROM_REF");
        ref_dir != nullptr && transcribe::debug::enabled()) {
        char base[64];
        std::snprintf(base, sizeof(base), "/compress.%03d.", st.compress_count);
        std::vector<int64_t> ref_idx, ref_dis;
        if (read_npy_1d(std::string(ref_dir) + base + "topk_indices.npy", ref_idx) &&
            read_npy_1d(std::string(ref_dir) + base + "is_disabled.npy", ref_dis) &&
            static_cast<int>(ref_idx.size()) == L && static_cast<int>(ref_dis.size()) == L) {
            for (int j = 0; j < L; ++j) {
                frame_idx[static_cast<size_t>(j)]   = static_cast<int>(ref_idx[static_cast<size_t>(j)]);
                is_disabled[static_cast<size_t>(j)] = ref_dis[static_cast<size_t>(j)] != 0 ? 1 : 0;
            }
        } else {
            std::fprintf(stderr, "nemotron3_diar: COMPRESS_FROM_REF: no usable reference picks for compression %d\n",
                         st.compress_count);
        }
    }

    // _gather_spkcache_and_preds: disabled -> learned silence embedding / 0.
    std::vector<float> new_emb(static_cast<size_t>(L) * emb_dim);
    std::vector<float> new_preds(static_cast<size_t>(L) * n_spk, 0.0f);
    for (int j = 0; j < L; ++j) {
        const int f = frame_idx[static_cast<size_t>(j)];
        if (is_disabled[static_cast<size_t>(j)]) {
            std::copy(sil_emb.begin(), sil_emb.end(), new_emb.begin() + static_cast<size_t>(j) * emb_dim);
        } else {
            std::copy(st.spkcache.begin() + static_cast<size_t>(f) * emb_dim,
                      st.spkcache.begin() + static_cast<size_t>(f + 1) * emb_dim,
                      new_emb.begin() + static_cast<size_t>(j) * emb_dim);
            std::copy(st.spkcache_preds.begin() + static_cast<size_t>(f) * n_spk,
                      st.spkcache_preds.begin() + static_cast<size_t>(f + 1) * n_spk,
                      new_preds.begin() + static_cast<size_t>(j) * n_spk);
        }
    }

    // Parity dump, index-for-index with the reference's --dump-compress
    // (compress.NNN.{input_preds,topk_indices,is_disabled,spkcache_preds}).
    if (transcribe::debug::enabled() && env_get("TRANSCRIBE_NEMOTRON3_DIAR_COMPRESS_DUMP") != nullptr) {
        const int          k = st.compress_count;
        char               name[64];
        std::vector<float> fidx(static_cast<size_t>(L)), fdis(static_cast<size_t>(L));
        for (int j = 0; j < L; ++j) {
            fidx[static_cast<size_t>(j)] = static_cast<float>(frame_idx[static_cast<size_t>(j)]);
            fdis[static_cast<size_t>(j)] = is_disabled[static_cast<size_t>(j)] ? 1.0f : 0.0f;
        }
        const long long li        = L;
        const long long shp_in[2] = { N, n_spk };
        std::snprintf(name, sizeof(name), "compress.%03d.input_preds", k);
        transcribe::debug::dump_host_f32(name, st.spkcache_preds.data(), static_cast<long long>(N) * n_spk, shp_in, 2,
                                         "compress");
        std::snprintf(name, sizeof(name), "compress.%03d.topk_indices", k);
        transcribe::debug::dump_host_f32(name, fidx.data(), li, &li, 1, "compress");
        std::snprintf(name, sizeof(name), "compress.%03d.is_disabled", k);
        transcribe::debug::dump_host_f32(name, fdis.data(), li, &li, 1, "compress");
        const long long shp[2] = { L, n_spk };
        std::snprintf(name, sizeof(name), "compress.%03d.spkcache_preds", k);
        transcribe::debug::dump_host_f32(name, new_preds.data(), static_cast<long long>(L) * n_spk, shp, 2, "compress");
    }
    ++st.compress_count;

    st.spkcache       = std::move(new_emb);
    st.spkcache_preds = std::move(new_preds);
    st.spkcache_n     = L;
}

}  // namespace

// streaming_update (sync branch). preds80 = [spkcache | fifo | chunk] preds.
void streaming_update(StreamState &              st,
                      const StreamParams &       p,
                      int                        n_spk,
                      int                        emb_dim,
                      const std::vector<float> & chunk_embs,
                      int                        T_diar,
                      const std::vector<float> & preds80,
                      int                        lc,
                      int                        rc,
                      const std::vector<float> & sil_emb) {
    const int S = st.spkcache_n;
    const int F = st.fifo_n;
    const int C = T_diar - lc - rc;

    // fifo_preds = preds[S : S+F]; chunk_preds = preds[S+F+lc : S+F+lc+C].
    st.fifo_preds.assign(preds80.begin() + static_cast<size_t>(S) * n_spk,
                         preds80.begin() + static_cast<size_t>(S + F) * n_spk);
    st.fifo.insert(st.fifo.end(), chunk_embs.begin() + static_cast<size_t>(lc) * emb_dim,
                   chunk_embs.begin() + static_cast<size_t>(lc + C) * emb_dim);
    st.fifo_preds.insert(st.fifo_preds.end(), preds80.begin() + static_cast<size_t>(S + F + lc) * n_spk,
                         preds80.begin() + static_cast<size_t>(S + F + lc + C) * n_spk);
    st.fifo_n = F + C;

    if (F + C > p.fifo_len) {
        int pop = p.spkcache_update_period;
        pop     = std::max(pop, C - p.fifo_len + F);
        pop     = std::min(pop, F + C);

        std::vector<float> pop_embs(st.fifo.begin(), st.fifo.begin() + static_cast<size_t>(pop) * emb_dim);
        std::vector<float> pop_preds(st.fifo_preds.begin(), st.fifo_preds.begin() + static_cast<size_t>(pop) * n_spk);

        st.fifo.erase(st.fifo.begin(), st.fifo.begin() + static_cast<size_t>(pop) * emb_dim);
        st.fifo_preds.erase(st.fifo_preds.begin(), st.fifo_preds.begin() + static_cast<size_t>(pop) * n_spk);
        st.fifo_n = (F + C) - pop;

        st.spkcache.insert(st.spkcache.end(), pop_embs.begin(), pop_embs.end());
        st.spkcache_n = S + pop;
        if (st.compressed) {
            st.spkcache_preds.insert(st.spkcache_preds.end(), pop_preds.begin(), pop_preds.end());
        } else {
            // Before the first compression, fresh predictions for every cache frame.
            st.spkcache_preds.assign(preds80.begin(), preds80.begin() + static_cast<size_t>(S) * n_spk);
            st.spkcache_preds.insert(st.spkcache_preds.end(), pop_preds.begin(), pop_preds.end());
        }
        if (st.spkcache_n > p.spkcache_len) {
            compress_spkcache(st, p, n_spk, emb_dim, sil_emb);
            st.compressed = true;
        }
    }
}

// ---- incremental speaker segments ----

void segments_advance(SegmentTracker & tr, const float * probs, int T, int n_spk, double ms_per_frame) {
    for (int t = tr.n_seen; t < T; ++t) {
        for (int s = 0; s < n_spk; ++s) {
            const bool active = probs[static_cast<size_t>(t) * n_spk + s] > 0.5f;
            int &      rs     = tr.run_start[static_cast<size_t>(s)];
            if (active && rs < 0) {
                rs = t;
            } else if (!active && rs >= 0) {
                transcribe_session::SpeakerSegmentEntry row;
                row.t0_ms      = static_cast<int64_t>(std::llround(rs * ms_per_frame));
                row.t1_ms      = static_cast<int64_t>(std::llround(t * ms_per_frame));
                row.speaker_id = s + 1;
                row.p          = std::numeric_limits<float>::quiet_NaN();
                tr.closed.push_back(row);
                rs = -1;
            }
        }
    }
    tr.n_seen = std::max(tr.n_seen, T);
}

void segments_close_all(SegmentTracker & tr, double ms_per_frame) {
    for (size_t s = 0; s < tr.run_start.size(); ++s) {
        int & rs = tr.run_start[s];
        if (rs >= 0) {
            transcribe_session::SpeakerSegmentEntry row;
            row.t0_ms      = static_cast<int64_t>(std::llround(rs * ms_per_frame));
            row.t1_ms      = static_cast<int64_t>(std::llround(tr.n_seen * ms_per_frame));
            row.speaker_id = static_cast<int32_t>(s) + 1;
            row.p          = std::numeric_limits<float>::quiet_NaN();
            tr.closed.push_back(row);
            rs = -1;
        }
    }
}

void segments_publish(const SegmentTracker & tr, transcribe_session * session, double ms_per_frame) {
    auto & out = session->speaker_segments;
    out        = tr.closed;
    for (size_t s = 0; s < tr.run_start.size(); ++s) {
        const int rs = tr.run_start[s];
        if (rs >= 0 && tr.n_seen > rs) {
            transcribe_session::SpeakerSegmentEntry row;
            row.t0_ms      = static_cast<int64_t>(std::llround(rs * ms_per_frame));
            row.t1_ms      = static_cast<int64_t>(std::llround(tr.n_seen * ms_per_frame));
            row.speaker_id = static_cast<int32_t>(s) + 1;
            row.p          = std::numeric_limits<float>::quiet_NaN();
            out.push_back(row);
        }
    }
    // Speaker-major, time-ordered within a speaker (the offline ordering).
    std::stable_sort(out.begin(), out.end(), [](const auto & a, const auto & b) {
        if (a.speaker_id != b.speaker_id) {
            return a.speaker_id < b.speaker_id;
        }
        return a.t0_ms < b.t0_ms;
    });
}

}  // namespace transcribe::nemotron3_diar
