//! The plain data of the DIARIZE role: info and options. Results are
//! [`SpeakerSegment`](crate::SpeakerSegment)s. The
//! `transcribe_cpp::DiarizeSession` handle stays in transcribe-cpp.

use crate::family::DiarizeExtension;

/// Static facts about a diarization model (`transcribe_cpp::Model::diarize_info`).
///
/// `Default` is all zeros; outside this crate (it is `#[non_exhaustive]`),
/// build one from `default()` and set the fields.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiarizeInfo {
    /// Input PCM rate (16000).
    pub sample_rate: i32,
    /// `speaker_id` is in `[1, max_speakers]`.
    pub max_speakers: i32,
}

/// Options for creating a diarization session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct DiarizeSessionOptions {
    /// CPU threads for ops that run on CPU; 0 = library default.
    pub n_threads: i32,
}

/// Per-run diarization parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct DiarizeOptions {
    /// Optional family-specific extension (e.g. the Sortformer preset).
    pub family: Option<DiarizeExtension>,
}
