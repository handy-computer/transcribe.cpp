//! Streaming data types: the owned option/result types a
//! `transcribe_cpp::Stream` exchanges.

use crate::family::StreamExtension;
use crate::types::CommitPolicy;

/// Options for beginning a stream.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct StreamOptions {
    /// When committed text grows. Default [`CommitPolicy::Auto`].
    pub commit_policy: CommitPolicy,
    /// Consecutive agreeing hypotheses required before a prefix commits
    /// (stable-prefix policies). 0 selects the library default (3).
    pub stable_prefix_agreement_n: u32,
    /// Optional family-specific stream extension.
    pub family: Option<StreamExtension>,
}

/// Per-call change metadata from `feed`/`finalize`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct StreamUpdate {
    /// Any observable property of the snapshot changed this call.
    pub result_changed: bool,
    /// True only on the `finalize` call's update.
    pub is_final: bool,
    /// Monotonic snapshot revision; diff against the previous value.
    pub revision: i32,
    /// Total audio received since begin (ms).
    pub input_received_ms: i64,
    /// Family-reported audio progress / drain hint (ms).
    pub audio_committed_ms: i64,
    /// Audio still buffered inside the family's streaming state (ms).
    pub buffered_ms: i64,
    /// `committed_text` changed this call.
    pub committed_changed: bool,
    /// `tentative_text` changed this call.
    pub tentative_changed: bool,
}

/// A UI-facing snapshot of the stream's text, fully owned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct StreamText {
    /// The raw current model hypothesis (authoritative; may rewrite anywhere).
    pub full: String,
    /// The append-only, flicker-free display/input prefix.
    pub committed: String,
    /// The volatile raw suffix after the committed prefix.
    pub tentative: String,
}

impl StreamText {
    /// `committed + tentative` — the flicker-free render most UIs want.
    pub fn display(&self) -> String {
        format!("{}{}", self.committed, self.tentative)
    }
}
