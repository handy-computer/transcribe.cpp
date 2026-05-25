// arch/moonshine_streaming/capabilities.cpp - Moonshine-Streaming
// capability defaults.

#include "moonshine_streaming.h"

namespace transcribe::moonshine_streaming {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;

    // English-only model. Translation and language detection are off; the
    // family doc's Capability Validation table marks both SKIP — not
    // exposed by runtime.
    caps.supports_translate = false;

    // No timestamp tokens in the vocab; runtime returns a single
    // text-only segment with zeroed timings (same policy as moonshine).
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE;

    // Phase 4b-encoder streaming: stream_feed runs the encoder
    // incrementally over a sliding window, appending stable frames to
    // a per-utterance committed buffer; stream_finalize runs adapter +
    // cross_kv + AR decode once over the full committed buffer.
    caps.supports_streaming            = true;

    // Streaming timing hints. Cumulative right-context across the
    // encoder layer stack is 12 conv-stack output frames (the four
    // R=4 layers each contribute R-1=3 frames; the two R=0 layers
    // contribute 0). At 20 ms per encoder output frame (frame_ms=5,
    // two stride-2 convs → 4× downsample on top of 200 Hz) that is
    // 240 ms of lookahead before a frame is fully stable.
    //
    // The natural emit unit at the encoder output rate is one
    // frame = 20 ms; the family doc points callers at 80 ms (4 frames)
    // as a balance between API overhead and per-feed work — small
    // enough to surface progress promptly, large enough that the
    // encoder graph allocation amortizes.
    //
    // The model has no inference-time lookahead/latency knob (unlike
    // nemotron's att_context_right menu), so min == default.
    caps.streaming_lookahead_ms        = 240;
    caps.streaming_chunk_ms            = 80;
    caps.streaming_lookahead_ms_min    = 240;

    // Cancellation is wired at the per-run + per-feed level. No PNC/ITN
    // runtime toggle. Whisper-style fallback / long-form / prompt
    // features do not apply.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

} // namespace transcribe::moonshine_streaming
