//! The plain data of the VAD role: info, options and results. The
//! `transcribe_cpp::VadSession` handle stays in transcribe-cpp.

/// Static facts about a VAD model (`transcribe_cpp::Model::vad_info`).
///
/// `Default` is all zeros; outside this crate (it is `#[non_exhaustive]`),
/// build one from `default()` and set the fields.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VadInfo {
    /// Input PCM rate (16000).
    pub sample_rate: i32,
    /// Samples per probability (512 = 32 ms).
    pub frame_samples: i32,
}

/// Options for creating a VAD session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VadSessionOptions {
    /// CPU threads for ops that run on CPU; 0 = library default.
    pub n_threads: i32,
}

/// Probabilities -> speech segments (`transcribe_cpp::VadSession::run` only). Fields and
/// defaults are those of Silero's `get_speech_timestamps`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VadOptions {
    /// A frame with `p >= threshold` is speech (default 0.5). `[0, 1]`.
    pub threshold: f64,
    /// Inside speech, a frame with `p < neg_threshold` is silence. Negative
    /// (the default, -1) means `max(threshold - 0.15, 0.01)`; otherwise
    /// `[0, threshold]`.
    pub neg_threshold: f64,
    /// Segments of at most this length are dropped (default 250), except
    /// those split by `max_speech_ms`. Durations must be nonnegative.
    pub min_speech_ms: i32,
    /// Silence this long ends a segment (default 100).
    pub min_silence_ms: i32,
    /// Padding on both sides of each segment (default 30).
    pub speech_pad_ms: i32,
    /// Segments longer than this are split; 0 (the default) = no limit.
    pub max_speech_ms: i32,
}

impl Default for VadOptions {
    fn default() -> Self {
        VadOptions {
            threshold: 0.5,
            neg_threshold: -1.0,
            min_speech_ms: 250,
            min_silence_ms: 100,
            speech_pad_ms: 30,
            max_speech_ms: 0,
        }
    }
}

/// One speech segment, in samples of the run's input: `[start, end)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VadSegment {
    /// First sample of speech.
    pub start_sample: i64,
    /// First sample after speech.
    pub end_sample: i64,
}

/// The result of one `transcribe_cpp::VadSession::run`, `transcribe_cpp::VadSession::stream_feed` or
/// `transcribe_cpp::VadSession::stream_flush`.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VadResult {
    /// Speech probability per frame produced by the call.
    pub probs: Vec<f32>,
    /// Stream position of `probs[0]`, in frames: 0 after a run; for a feed
    /// or flush, the frames the stream had scored before the call.
    pub first_frame: i64,
    /// Speech segments, in time order; run only (empty after a feed or flush).
    pub segments: Vec<VadSegment>,
}
