//! Typed family-extension options.
//!
//! Each family exposes model-specific knobs via a kind-tagged extension struct
//! (`include/transcribe/<family>.h`). These typed options mirror those structs:
//! every field is an `Option`, and only the fields you set override the
//! defaults the C `*_init()` stamps. A [`RunExtension`] attaches to
//! [`RunOptions`](crate::RunOptions); a [`StreamExtension`] attaches to
//! [`StreamOptions`](crate::StreamOptions); a [`DiarizeExtension`] attaches to
//! [`DiarizeOptions`](crate::DiarizeOptions).
//!
//! Probe `transcribe_cpp::Model::accepts_ext` to learn whether a loaded model
//! accepts a given kind on a slot; an unaccepted extension is rejected by
//! `run`/`stream` with `transcribe_cpp::Error::InvalidArgument`.

/// Whisper run-extension knobs (run slot): initial prompt, temperature
/// fallback, and decode thresholds. `None` keeps the family default.
/// `initial_prompt` cannot be combined with `RunOptions::vocabulary` / `prompt`.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct WhisperRunOptions {
    pub initial_prompt: Option<String>,
    pub condition_on_prev_tokens: Option<bool>,
    pub temperature: Option<f32>,
    pub temperature_inc: Option<f32>,
    pub compression_ratio_thold: Option<f32>,
    pub logprob_thold: Option<f32>,
    pub no_speech_thold: Option<f32>,
    pub max_prev_context_tokens: Option<i32>,
    pub seed: Option<u32>,
    pub max_initial_timestamp: Option<f32>,
}

/// Moonshine-streaming stream-extension knobs (stream slot).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct MoonshineStreamingOptions {
    pub min_decode_interval_ms: Option<i32>,
}

/// Parakeet cache-aware streaming knobs (stream slot).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ParakeetStreamOptions {
    pub att_context_right: Option<i32>,
}

/// Parakeet chunked-attention buffered streaming knobs (stream slot).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ParakeetBufferedStreamOptions {
    pub left_ms: Option<i32>,
    pub chunk_ms: Option<i32>,
    pub right_ms: Option<i32>,
}

/// Voxtral-realtime streaming knobs (stream slot).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VoxtralRealtimeStreamOptions {
    pub num_delay_tokens: Option<i32>,
    pub min_decode_interval_ms: Option<i32>,
}

/// Sortformer streaming operating point (latency / accuracy trade-off).
/// The menu is discrete (jointly-tuned bundles), not a latency dial;
/// `Default` keeps the GGUF-shipped checkpoint configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SortformerPreset {
    #[default]
    Default,
    /// ~30.4 s algorithmic lookahead; the offline-file operating point.
    VeryHighLatency,
    /// ~10.0 s lookahead.
    HighLatency,
    /// ~1.04 s lookahead; the real-time point (compute-heavy per audio
    /// second — many small windows).
    LowLatency,
}

/// Sortformer diarize-extension knobs (diarize-run slot). `None` keeps the
/// family default (the GGUF-shipped cfg).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct SortformerDiarizeOptions {
    pub preset: Option<SortformerPreset>,
}

/// A family extension for the diarize-run slot
/// (`transcribe_cpp::DiarizeSession::run`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DiarizeExtension {
    Sortformer(SortformerDiarizeOptions),
}

/// A family extension for the run slot (offline `run`/`run_batch`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RunExtension {
    Whisper(WhisperRunOptions),
}

/// A family extension for the stream slot.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StreamExtension {
    ParakeetStream(ParakeetStreamOptions),
    ParakeetBuffered(ParakeetBufferedStreamOptions),
    MoonshineStreaming(MoonshineStreamingOptions),
    VoxtralRealtime(VoxtralRealtimeStreamOptions),
}
