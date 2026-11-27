//! [`RunOptions`]: the per-run parameters of `transcribe_cpp::Session::run`,
//! `run_batch` and `stream`.

use crate::family::RunExtension;
use crate::types::{Diarize, Itn, Pnc, Task, TimestampKind};

/// Per-run parameters.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct RunOptions {
    pub task: Task,
    pub timestamps: TimestampKind,
    pub pnc: Pnc,
    pub itn: Itn,
    pub diarize: Diarize,
    /// Source language hint (ISO code), or `None` to autodetect.
    pub language: Option<String>,
    /// Target language for translation, or `None`.
    pub target_language: Option<String>,
    /// Keep special vocab tags (e.g. `<|...|>`) in the returned text.
    pub keep_special_tags: bool,
    /// Speculative-decode draft length. `-1` = family default, `0` = disabled.
    pub spec_k_drafts: i32,
    /// Optional family-specific run extension (e.g. whisper decode knobs).
    pub family: Option<RunExtension>,
    /// Custom terms in priority order, formatted per family
    /// (`Feature::Vocabulary`; ignored with a warning elsewhere).
    pub vocabulary: Vec<String>,
    /// Context text under `Task::Transcribe`/`Translate`
    /// (`Feature::ContextPrompt`); the required instruction under
    /// `Task::Instruct`.
    pub prompt: Option<String>,
    /// Transcript text the model continues from (`Feature::TranscriptPrefix`;
    /// an error elsewhere, and in batch and streaming runs). `text` holds only
    /// the continuation; `raw_text` leads with the prefix.
    pub prefix: Option<String>,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions {
            task: Task::Transcribe,
            timestamps: TimestampKind::Auto,
            pnc: Pnc::Default,
            itn: Itn::Default,
            diarize: Diarize::Default,
            language: None,
            target_language: None,
            keep_special_tags: false,
            spec_k_drafts: -1,
            family: None,
            vocabulary: Vec::new(),
            prompt: None,
            prefix: None,
        }
    }
}
