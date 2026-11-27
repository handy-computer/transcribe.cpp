//! The plain data of the TURN role: info, options and results. The
//! `transcribe_cpp::TurnSession` handle stays in transcribe-cpp.

/// Static facts about an end-of-turn model (`transcribe_cpp::Model::turn_info`).
///
/// `Default` is all zeros; outside this crate (it is `#[non_exhaustive]`),
/// build one from `default()` and set the fields.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TurnInfo {
    /// Input PCM rate (16000).
    pub sample_rate: i32,
    /// The longest audio `transcribe_cpp::TurnSession::run` looks at; longer audio is scored
    /// on its last `max_audio_ms`. 0 = no limit.
    pub max_audio_ms: i32,
    /// Samples per frame for streaming; 0 = the model cannot stream (the
    /// `stream_*` calls return `transcribe_cpp::Error::NotImplemented`).
    pub frame_samples: i32,
    /// Whether the model also returns a speech probability per frame
    /// ([`TurnResult::speech_probs`]).
    pub has_speech: bool,
    /// The model's recommended cut-off: the default for
    /// [`TurnOptions::threshold`].
    pub threshold: f64,
}

/// Options for creating a turn session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct TurnSessionOptions {
    /// CPU threads for ops that run on CPU; 0 = library default.
    pub n_threads: i32,
}

/// Settings for `transcribe_cpp::TurnSession::run`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct TurnOptions {
    /// Probability at or above which the turn counts as ended
    /// ([`TurnResult::ended`]). Negative (the default, -1) uses the model's
    /// [`TurnInfo::threshold`]; otherwise `[0, 1]`.
    pub threshold: f64,
}

impl Default for TurnOptions {
    fn default() -> Self {
        TurnOptions { threshold: -1.0 }
    }
}

/// The result of one `transcribe_cpp::TurnSession::run`, `transcribe_cpp::TurnSession::stream_feed`,
/// `transcribe_cpp::TurnSession::stream_feed_batch` or `transcribe_cpp::TurnSession::stream_flush`.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct TurnResult {
    /// After a run: the probability that the turn has ended at the end of the
    /// clip. 0 after a feed or flush.
    pub p: f32,
    /// After a run: `p >= threshold`. False after a feed or flush.
    pub ended: bool,
    /// Per-frame turn probabilities from the call: `probs[i]` says whether the
    /// turn has ended at the end of frame `first_frame + i`. Always empty for
    /// models that cannot stream.
    pub probs: Vec<f32>,
    /// Stream position of `probs[0]`, in frames: 0 after a run; for a feed or
    /// flush, the frames the stream had processed before the call.
    pub first_frame: i64,
    /// Per-frame speech probabilities from the call (empty unless
    /// [`TurnInfo::has_speech`]). During a stream they can trail the turn
    /// probabilities; after a run every frame has both.
    pub speech_probs: Vec<f32>,
    /// Stream position of `speech_probs[0]`, in frames.
    pub speech_first_frame: i64,
}
