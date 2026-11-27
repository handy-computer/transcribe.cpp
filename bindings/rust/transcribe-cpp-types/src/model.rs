//! Model- and session-level plain data: [`Capabilities`], [`SessionOptions`]
//! and [`SessionLimits`].

use crate::types::{KvType, TimestampKind};

/// Immutable, model-level capabilities read from GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Capabilities {
    pub native_sample_rate: i32,
    /// Supported language codes (empty if the model is language-agnostic).
    pub languages: Vec<String>,
    /// Supported translation target language codes (empty if not advertised).
    pub translate_target_languages: Vec<String>,
    /// The finest timestamp granularity the model can produce.
    pub max_timestamp_kind: TimestampKind,
    pub supports_language_detect: bool,
    pub supports_translate: bool,
    pub supports_streaming: bool,
    pub supports_spec_decode: bool,
    /// Longest accepted audio in ms (0 = no practical limit).
    pub max_audio_ms: i64,
}

/// Options for creating a session.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct SessionOptions {
    /// CPU threads for ops that run on CPU; 0 = library default.
    pub n_threads: i32,
    /// K/V activation precision.
    pub kv_type: KvType,
    /// Optional decoder context cap in tokens; 0 = model maximum.
    pub n_ctx: i32,
}

impl Default for SessionOptions {
    fn default() -> Self {
        SessionOptions {
            n_threads: 0,
            kv_type: KvType::Auto,
            n_ctx: 0,
        }
    }
}

/// Per-session effective limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SessionLimits {
    /// Decoder context cap in force for this session (0 = unbounded family).
    pub effective_n_ctx: i32,
    /// Longest accepted audio in ms (0 = no practical limit).
    pub effective_max_audio_ms: i64,
    /// Worst-case decoder KV bytes for one utterance (0 = no decoder KV).
    pub max_kv_bytes: i64,
}
