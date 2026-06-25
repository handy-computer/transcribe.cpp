// arch/parakeet/parakeet.h - Parakeet-family internal model and context
// types.
//
// This header is INTERNAL to the Parakeet sources. It defines the
// concrete classes that derive from transcribe_model / transcribe_session
// for this family. It is also visible to tests/transcribe_tokenizer_smoke
// (via the test target's PRIVATE include path) so the test can pull a
// const Tokenizer * out of a model loaded through the public C ABI
// without inventing a temporary public accessor.
//
// Other source files outside src/arch/parakeet/ should not include this
// header. The central dispatcher in transcribe.cpp talks to the family
// only through the Arch trait and the base classes.

#pragma once

#include "decoder.h"
#include "transcribe-backend.h"
#include "transcribe-session.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend *        ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::parakeet {

// ---------------------------------------------------------------------------
// Attention masks
// ---------------------------------------------------------------------------
//
// Host-side helpers that materialize the 2D attention masks used by the
// streaming paths. Declared here (in the family-internal header) so the
// per-mask unit tests can call them directly without needing a ggml
// backend or a real GGUF on disk.
//
// chunked_limited_with_rc mask (buffered streaming, parakeet-unified-en-0.6b)
//
//   Mirrors NeMo `ConformerEncoder._create_masks` lines 843-869 for the
//   chunked_limited_with_rc branch:
//
//     c_q = q / C
//     window_start = max(0, c_q * C - L)
//     window_end   = min(T - 1, c_q * C + C - 1 + R)
//     mask[q, k]   = 0  if  window_start <= k <= window_end  else -INF
//
//   The output buffer is row-major [T_q, T_k] with T_q == T_k == T —
//   the buffered driver always runs the encoder over a power-of-T
//   window, so query and key axes match. out_buf must point to at
//   least T*T floats. The mask broadcasts over heads in rel_pos_mhsa.
//
//   Asserts (debug-only): T >= 1, C >= 1, L >= 0, R >= 0.
//
//   `pad_length` (optional) folds in NeMo's conv-overhang pad_mask:
//   cells where row q >= pad_length OR col k >= pad_length are masked
//   regardless of the chunked window. This matches NeMo's
//   ConformerEncoder.forward, which ANDs pad_mask onto the chunked
//   mask before MHA. Set pad_length = T to disable padding masking
//   (the offline / no-pad-overhang case). For buffered streaming,
//   pass effective_T = ctx.total / encoder_frame2audio_samples; the
//   conv frontend may produce T_enc > effective_T due to padding
//   overhang, and those trailing frames must be masked out or they
//   contaminate the attention scores of valid frames.
void compute_chunked_limited_with_rc_mask(
    float * out_buf,
    int     T,
    int     left_context_frames,
    int     chunk_size_frames,
    int     right_context_frames,
    int     pad_length);


// Family defaults — applied before transcribe::read_capability_kv runs,
// so capability KV present overrides the default and capability KV
// absent leaves the default in place. Defined in capabilities.cpp.
//
// There is no per-variant resolver: variant identity is descriptive
// metadata, not a behavioral discriminator. Differences between v2
// and v3 (or any future Parakeet variant) are expressed as
// stt.capability.* / general.languages KV that the converter writes
// and the family-agnostic readers in transcribe-meta.h consume.
// RESUME.md "Decisions still load-bearing" has the rationale and the
// llama.cpp / whisper.cpp comparison that drove it.
void apply_family_invariants(transcribe_model & model);

// Concrete model. Owns the ggml_context that holds every weight
// tensor's data buffer (allocated by gguf_init_from_file with
// no_alloc=false during load). The destructor frees the ggml_context;
// every borrowed ggml_tensor* in `weights` is invalidated at that
// moment. The first gguf_context (the loader's header-only inspection
// pass) is no longer kept alive past load() — its only consumer is
// the tokenizer ingest, which copies all the data it needs. The
// second gguf_context (the one created by load() to read tensor
// metadata) is freed inside load() once the weights have been
// catalogued.
struct ParakeetModel final : public transcribe_model {
    Tokenizer       tok;
    ParakeetHParams hparams;
    ParakeetWeights weights;
    ggml_context *  ctx_meta = nullptr;

    // Runtime backend plan. Resolved at load() time from
    // transcribe_model_load_params::backend via
    // transcribe::load_common::init_backends. See
    // transcribe-backend.h for the plan semantics. The plan's
    // `scheduler_list` owns the backends; the destructor frees
    // them in reverse order. Helpers that key off "is this CPU?"
    // check `plan.primary_kind == BackendKind::Cpu` directly.
    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // On a CPU primary backend, the encoder's 2D Q8_0 matmul weights are
    // allocated in the CPU_REPACK extra buffer (interleaved blocked-GEMM
    // layout for the AVX-512 VNNI kernels); this holds that buffer. Decoder
    // weights stay in backend_buffer. Null on GPU primary / when disabled.
    ggml_backend_buffer_t   repack_buffer = nullptr;

    // Fused BN parameters live in a separate ggml context + buffer,
    // computed at load time from the raw BN tensors. Freed in dtor.
    ggml_context *          bn_fused_ctx    = nullptr;
    ggml_backend_buffer_t   bn_fused_buffer = nullptr;

    // On a CPU primary backend the conformer 1×1 pointwise conv weights
    // are dequantized from their on-disk F16 form to F32 at load time.
    // The current quantizer (transcribe-quantize, post commit 6fee9b9)
    // routes every Conformer ConvPw to F16 across all presets so that
    // GPU backends with native F16 compute (Metal, Vulkan, CUDA) get
    // the bandwidth halving for free; on CPUs without native F16
    // arithmetic (Zen 2 and earlier) the per-matmul F16→F32 upconvert
    // erases the bandwidth win and dwarfs the original cost. Promoting
    // back to F32 at load time pays the conversion exactly once.
    //
    // Today's parakeet GGUFs predate the universal F16 conv_pw policy
    // and ship F32 conv_pw, so the promotion is a no-op for them. The
    // wiring is in place so the next regen does the right thing
    // automatically. The promotion is also a no-op on every non-CPU
    // primary backend.
    //
    // The CohereBlock comment about ~235 MB of "wasted" originals
    // applies symmetrically: ggml backend buffers don't support
    // freeing individual tensor slots, so the F16 source tensors stay
    // resident in the main weight buffer after promotion. The wired
    // weight slot is repointed at the F32 copy and the encoder graph
    // never touches the originals again.
    ggml_context *          conv_pw_f32_ctx    = nullptr;
    ggml_backend_buffer_t   conv_pw_f32_buffer = nullptr;

    // Mel front-end. Constructed once at load() time from the
    // hparams (precomputes the periodic Hann window + Slaney mel
    // filterbank). MelFrontend is documented as const-after-ctor
    // and thread-safe across compute() calls, so it's shared by
    // every context derived from this model. std::optional because
    // MelFrontend has no default constructor; emplace() in load()
    // after the hparams are read.
    std::optional<transcribe::MelFrontend> mel;

    // Host mirror of the predictor + joint weights, populated at
    // load() time from the backend tensors via
    // build_host_decoder_weights. The decoder runs on host (see
    // decoder.h for the rationale); having a const-after-load
    // mirror lets every context share the same arrays without
    // each transcribe_run paying a backend readback cost. The
    // memory cost is small relative to the encoder weights
    // (~35 MB on v2, ~73 MB on v3 vs ~2.4 GB total).
    HostDecoderWeights host_decoder;

    ParakeetModel() = default;
    ~ParakeetModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

// Per-context streaming encoder state. Allocated lazily on the first
// stream_begin for ChunkedLimited variants. Mirrors NeMo's
// cache_last_channel + cache_last_time + cache_last_channel_len tuple
// returned by get_initial_cache_state. Layout cheat sheet:
//
//   cache_last_channel  [d_model, T_cache, n_layer] f32
//     Per-layer post-attn-LN input cache. Concat-prepended to x_norm
//     before Q/K/V projection in the streaming MHSA path. NeMo stores
//     the pre-projection tensor (not split K/V), so a single per-layer
//     slot suffices.
//
//   cache_last_time     [k_minus_1, d_model, n_layer] f32
//     Per-layer post-pw1+GLU conv-module input cache. Replaces the
//     zero-pad on the left of the depthwise conv in the streaming
//     conv_module path.
//
//   cache_last_channel_len   scalar
//     Valid-frame count in cache_last_channel (0..T_cache). Drives
//     the streaming-mask offset so unfilled prefix rows are masked
//     out until the cache fills.
//
// Sizes for nemotron-speech-streaming-en-0.6b at att_context_size=[70,13]:
//   T_cache         = 70           (att_context_left)
//   k_minus_1       = 8            (conv_kernel - 1)
//   n_layer         = 24, d_model = 1024
//   Total cache buffer ~7.5 MB.
struct ParakeetStreamingCaches {
    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    // Per-layer cache tensors. Shapes:
    //   last_channel[i] = ne[d_model, T_cache, 1, 1]
    //   last_time[i]    = ne[k_minus_1, d_model, 1, 1]
    std::vector<ggml_tensor *> last_channel;
    std::vector<ggml_tensor *> last_time;

    int32_t channel_len = 0;

    // Absolute mel-frame cursor across the whole stream — used to
    // anchor token / segment timestamps to original-audio time even
    // when chunks vary in size. Counts mel frames consumed (i.e. fed
    // to the encoder, minus history prepend).
    int64_t mel_frames_consumed = 0;

    // Absolute sample index of stream_pcm_buffer[0]. The buffer is a
    // sliding window: after each emit, prefix samples no longer needed
    // by any unemitted frame's STFT window are trimmed (hop-aligned).
    // Bounded per-feed mel cost: O(new audio) instead of O(total).
    // Always a multiple of hop_length so mel-frame indexing translates
    // cleanly between absolute and buffer-relative.
    int64_t pcm_start_sample = 0;

    // Streaming geometry for the active stream. Resolved at
    // stream_begin from ParakeetHParams + the caller-selected
    // att_context_right (transcribe_parakeet_stream_ext). All four
    // fields are constant across chunks within one stream; they
    // change only when stream_begin is called again with a different
    // att_context_right selection.
    //
    //   att_context_right: chosen R from the model's training menu
    //                      (default = enc_att_context_right).
    //   att_context_left:  matching L (currently always the model
    //                      default; multi-L selection isn't in the API).
    //
    //   First-chunk and subsequent-chunk geometry differ. NeMo's
    //   setup_streaming_params packs the (first, subsequent) pair into
    //   `streaming_cfg.chunk_size` and `shift_size`; we mirror that
    //   with parallel fields below. The driver in stream_feed picks
    //   the right pair based on the (now-tracked) is_first_chunk flag.
    //
    //   For nemotron-speech-streaming-en-0.6b at att_context_right=13:
    //     chunk_size_first      = 105  (sampling_frames_first + 8*R)
    //     chunk_size_subsequent = 112  (subsampling_factor * (1 + R))
    //     mel_fed_first         = 105  (no pre_encode_cache prepend)
    //     mel_fed_subsequent    = 121  (9 cache + 112 new)
    //     drop_extra_first      = 0
    //     drop_extra_subsequent = 2
    int att_context_right            = 0;
    int att_context_left             = 0;
    int chunk_size_first             = 0;
    int chunk_size_subsequent        = 0;
    int mel_fed_first                = 0;
    int mel_fed_subsequent           = 0;
    int drop_extra_first             = 0;
    int drop_extra_subsequent        = 0;
    // Whether the next emit_streaming_chunk call is the FIRST in this
    // stream. Flips to false after the first chunk is processed.
    bool is_first_chunk              = true;

    // Per-stream chunk counter (== "step_num" in NeMo's
    // perform_streaming reference). Resets to 0 on stream_begin, then
    // increments once per emit_streaming_chunk. Used for indexing dump
    // filenames against the Python reference dumps when
    // TRANSCRIBE_DUMP_DIR is set.
    int chunk_step              = 0;

    bool initialized = false;
};

// Per-context streaming RNN-T decoder state. The offline path
// allocates LstmState locally; for streaming we own the state on the
// context so it carries across stream_feed calls. Reset on each
// stream_begin to a fresh "start of sequence" state.
struct ParakeetStreamingDecoderState {
    // LSTM (h, c) per layer + scratch buffers. Sized once on
    // stream_begin from host_decoder.predictor.n_layers /
    // pred_hidden.
    LstmState lstm_state;

    // Last emitted non-blank token id; the predictor input. Starts at
    // a sentinel (-1) meaning "predictor seeded with SOS embedding".
    int32_t prev_token_id = -1;

    // Absolute encoder-frame offset for converting per-chunk
    // step_at_emit indices into stream-wide frame indices.
    int64_t frame_offset = 0;

    bool initialized = false;
};

// Concrete context. Owns a per-call compute context and a persistent
// multi-backend scheduler that dispatches encoder graph ops to the
// best available backend (GPU for matmuls, BLAS for CPU matmuls,
// CPU for everything else).
struct ParakeetSession final : public transcribe_session {
    // Compute context: holds the cgraph and intermediate tensor
    // metadata. Created with no_alloc=true; data lives in the
    // sched-managed buffers. Reset at the start of every run() call.
    ggml_context *        compute_ctx    = nullptr;

    // Multi-backend scheduler. Persists across calls; manages compute
    // buffer allocation and dispatches ops to the best backend.
    // Reuses buffers when the graph topology is unchanged.
    ggml_backend_sched_t  sched          = nullptr;

    // Encoder forward output, borrowed pointer into compute_ctx
    // tensors. Invalidated when compute_ctx is reset on the next
    // run() call.
    ggml_tensor * encoder_out = nullptr;

    // Per-context scratch reused across runs. Keeping these on the
    // context avoids re-allocating the same medium-sized host buffers
    // every call while preserving the exact arithmetic.
    std::vector<float> mel_buf;
    std::vector<float> pos_buf;
    std::vector<float> pos_div_term;
    std::vector<float> enc_host;
    std::vector<TdtToken> raw_tokens;

    // Per-call timings (t_mel_us, t_encode_us, t_decode_us) live
    // on the base transcribe_session now and are surfaced via the
    // public transcribe_get_timings API.

    // Phase 5: TDT decode result storage lives on the base
    // (transcribe_session::tokens / words / segments / full_text /
    // result_kind / has_result), populated by Parakeet::run during
    // decode. No per-family result fields here.

    // KV type for flash attention, resolved from the context params.

    // ---- streaming-of-whole state (Phase 4a) ----
    //
    // Parakeet exposes the streaming API surface for cache-aware
    // streaming variants (today: nemotron-speech-streaming-en-0.6b,
    // which has enc_att_context_style=ChunkedLimited). Phase 4a is a
    // stream-of-whole stub: feed appends PCM, finalize runs the
    // existing one-shot inference helper on the accumulated buffer.
    // This makes the dispatcher / lifecycle / result-snapshot path
    // observable end-to-end and trivially parity-equivalent to
    // transcribe_run on the same audio; real per-chunk encoder
    // feeding (cache_last_channel / cache_last_time / RNNT predictor
    // carry) is Phase 4b.
    //
    // Lifecycle:
    //   stream_begin:    stream_pcm_buffer.clear(); stream_run_params = *run_params;
    //   stream_feed:     append samples
    //   stream_finalize: drain buffer through run_one_shot_inner()
    //   stream_reset:    stream_pcm_buffer.clear() (keep capacity)
    //
    // These fields are NOT touched by clear_result; the dispatcher
    // wipes lifecycle-agnostic snapshot state there, and the family
    // owns its per-utterance audio scratch.
    std::vector<float>   stream_pcm_buffer;
    transcribe_run_params    stream_run_params {};

    // ---- M2 incremental streaming state (cache-aware) ----
    //
    // Allocated on the first stream_begin for ChunkedLimited variants
    // (today: nemotron-speech-streaming-en-0.6b). Zeroed at each
    // stream_begin so each utterance starts from a clean cache.
    // Persists across feed/finalize within an utterance; freed in the
    // context destructor.
    ParakeetStreamingCaches        stream_caches;
    ParakeetStreamingDecoderState  stream_dec_state;

    // ---- Buffered streaming state (parakeet-unified-en-0.6b) ----
    //
    // Mirrors NeMo's `StreamingBatchedAudioBuffer` + the reference
    // inference loop at
    // `examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py`.
    // Variable-stride: step 0 (initial fill) consumes
    // `samples_chunk + samples_right` of new audio; subsequent feed
    // steps consume `samples_chunk`; finalize's single last step
    // consumes whatever's left (chunk slot absorbs the trailing right
    // context, no zero-pad).
    //
    // The expected geometry (buf_left_frames / buf_samples_left etc.)
    // is constant per stream and reflects the active (L, C, R) tuple.
    // The ctx_* fields track the buffer's INTERNAL `ContextSize`
    // (left/chunk/right slot sizes in samples), updated per-chunk via
    // `buf_ctx_add_frames` to mirror NeMo's `add_frames_get_removed_`.
    // The encoder output sliced off this chunk = (ctx_left /
    // frame_samples) leading frames; the decoder decodes
    // (ctx_chunk / frame_samples) frames on non-last, or all remaining
    // frames on last.
    //
    // The RNN-T LSTM state + last-emitted token id ride on the
    // existing stream_dec_state slot — buffered streaming reuses the
    // same predictor/joint code path the cache-aware driver does,
    // just without a per-layer encoder cache.
    int32_t buf_left_frames   = 0;  // L (expected)
    int32_t buf_chunk_frames  = 0;  // C
    int32_t buf_right_frames  = 0;  // R
    int32_t buf_samples_left  = 0;  // L * encoder_frame2audio_samples
    int32_t buf_samples_chunk = 0;
    int32_t buf_samples_right = 0;
    // Next sample index in stream_pcm_buffer to feed (= NeMo's
    // `left_sample` at the start of the next step). 0 at stream_begin;
    // advances by num_new_samples after each emit.
    int64_t buf_next_audio_read = 0;
    // Buffer's internal ContextSize. (0,0,0) at stream_begin; ramps up
    // as audio arrives until it saturates at (samples_left, chunk, right).
    int64_t buf_ctx_left  = 0;
    int64_t buf_ctx_chunk = 0;
    int64_t buf_ctx_right = 0;
    // Whether step 0 (the initial fill of samples_chunk+samples_right)
    // has run. False at stream_begin; first emit flips it.
    bool    buf_initialized = false;
    // Per-stream chunk step counter (== NeMo's `step_num`). Used for
    // per-chunk dump file naming.
    int32_t buf_chunk_step = 0;
    bool    buf_active = false;

    ParakeetSession() = default;
    ~ParakeetSession() override;
};

} // namespace transcribe::parakeet
