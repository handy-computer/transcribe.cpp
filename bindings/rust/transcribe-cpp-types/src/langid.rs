//! The plain data of the LANGID role: info, options and results. The
//! `transcribe_cpp::LangIdSession` handle stays in transcribe-cpp.

/// Static facts about a language ID model (`transcribe_cpp::Model::langid_info`).
///
/// `Default` is all zeros; outside this crate (it is `#[non_exhaustive]`),
/// build one from `default()` and set the fields.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LangIdInfo {
    /// Input PCM rate (16000).
    pub sample_rate: i32,
    /// Label indices are `[0, n_labels)`.
    pub n_labels: i32,
    /// Shorter scored audio is `transcribe_cpp::Error::InputTooShort`.
    pub min_audio_ms: i32,
    /// Longer input is scored on its first `max_audio_ms` (30000).
    pub max_audio_ms: i32,
}

/// Options for creating a language ID session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct LangIdSessionOptions {
    /// CPU threads for ops that run on CPU; 0 = library default.
    pub n_threads: i32,
}

/// Per-run language ID parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct LangIdOptions {
    /// Restrict the decision to these codes or aliases. `None` means every
    /// label; `Some` of an empty list is `transcribe_cpp::Error::InvalidArgument`. An
    /// unknown code is `transcribe_cpp::Error::Unsupported`.
    pub allowed: Option<Vec<String>>,
}

/// One ranked label. `code` is the model's own label (`"en"`, `"iw"`).
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct LangIdCandidate {
    /// Label index.
    pub index: i32,
    pub code: String,
    pub name: String,
    /// Softmax renormalized over the allowed set.
    pub p: f32,
    pub logit: f32,
}

/// The result of one `transcribe_cpp::LangIdSession::run`.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct LangIdResult {
    /// Every allowed label (duplicates once; all labels when unrestricted),
    /// ranked by `p`, descending; ties keep label order.
    pub candidates: Vec<LangIdCandidate>,
    /// Unrestricted probability inside the allowed set (1.0 when
    /// unrestricted); a low value means the speech is probably outside it.
    pub allowed_mass: f32,
}

impl LangIdResult {
    /// The top candidate's code, if any.
    pub fn code(&self) -> Option<&str> {
        self.candidates.first().map(|c| c.code.as_str())
    }
}
