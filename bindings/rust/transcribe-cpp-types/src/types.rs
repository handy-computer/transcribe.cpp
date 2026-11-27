//! Idiomatic Rust enums mirroring the C API's parameter enums.
//!
//! transcribe-cpp maps each to/from its `transcribe-cpp-sys` NewType.
//! Conversions INTO C are total; conversions OUT of C validate the raw integer
//! and fall back to a safe default for an unknown value (enum hygiene — we
//! never assume the library only ever hands back values this version knows
//! about).

/// The task a run performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Task {
    /// Transcribe speech in its source language.
    #[default]
    Transcribe,
    /// Translate speech into the target language (model must support it).
    Translate,
    /// `RunOptions::prompt` replaces the task instruction; the output is free
    /// text in `text` / `raw_text` (model must support `Feature::Instruct`;
    /// offline only).
    Instruct,
}

impl Task {
    /// The C enum value (`transcribe_task`), for transcribe-cpp's conversion. Not public
    /// API: an exhaustive match here makes a new variant fail to compile until
    /// it has a value, and transcribe-cpp tests every value against the header.
    #[doc(hidden)]
    pub const fn c_value(self) -> u32 {
        match self {
            Task::Transcribe => 0,
            Task::Translate => 1,
            Task::Instruct => 2,
        }
    }
}

/// Requested (or returned) timestamp granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TimestampKind {
    /// Text only, no alignment data.
    None,
    /// "Richest the model supports" — resolved per-family at run time.
    #[default]
    Auto,
    /// Segment-level start/end times.
    Segment,
    /// Word-level start/end times.
    Word,
    /// Token-level start/end times.
    Token,
}

/// K/V activation precision for the decoder's flash-attention path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KvType {
    /// f16 for quantized weights, f32 for f32 weights. The best default.
    #[default]
    Auto,
    /// Full-precision K/V.
    F32,
    /// Half-precision K/V.
    F16,
}

/// Punctuation + capitalization runtime toggle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Pnc {
    /// The family's shipped default (what its published WER was measured at).
    #[default]
    Default,
    /// Explicitly disable runtime PNC (supporting families only).
    Off,
    /// Explicitly enable runtime PNC (supporting families only).
    On,
}

/// Inverse-text-normalization runtime toggle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Itn {
    /// The family's shipped default.
    #[default]
    Default,
    /// Explicitly disable runtime ITN (supporting families only).
    Off,
    /// Explicitly enable runtime ITN (supporting families only).
    On,
}

/// Speaker-attribution runtime toggle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Diarize {
    /// Library default: speaker attribution is disabled for every family.
    #[default]
    Default,
    /// Explicitly disable speaker attribution.
    Off,
    /// Enable speaker attribution on supporting models.
    On,
}

/// Which compute backend to request when loading a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Backend {
    /// Best available device; CPU is the always-present fallback. The default.
    #[default]
    Auto,
    /// Strict CPU (no GPU, no host accelerators) — the determinism choice.
    Cpu,
    /// CPU plus host-memory accelerators (BLAS/AMX) when present.
    CpuAccel,
    /// Require Metal; errors if this build has no Metal.
    Metal,
    /// Require Vulkan; errors if this build has no Vulkan.
    Vulkan,
    /// Require CUDA; errors if this build has no CUDA.
    Cuda,
    /// Require ROCm; errors if this build has no ROCm.
    Rocm,
}

/// A yes/no model capability probe (`transcribe_model_supports`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Feature {
    /// The whisper run extension's initial prompt / prompt tokens.
    InitialPrompt,
    /// Runs a multi-tier temperature fallback loop (whisper today).
    TemperatureFallback,
    /// Handles audio longer than its native window in one run (whisper today).
    LongForm,
    /// Honors the abort callback between chunks/decode steps.
    Cancellation,
    /// Exposes a runtime PNC toggle.
    Pnc,
    /// Exposes a runtime ITN toggle.
    Itn,
    /// Produces structured speaker attribution.
    Diarization,
    /// `RunOptions::vocabulary` is formatted for this model.
    Vocabulary,
    /// `RunOptions::prompt` reaches a transcription-conditioning slot.
    ContextPrompt,
    /// Supports `Task::Instruct`.
    Instruct,
    /// Honors `RunOptions::prefix` as forced decoder text.
    TranscriptPrefix,
}

impl Feature {
    /// The C enum value (`transcribe_feature`), for transcribe-cpp's conversion. Not public
    /// API: an exhaustive match here makes a new variant fail to compile until
    /// it has a value, and transcribe-cpp tests every value against the header.
    #[doc(hidden)]
    pub const fn c_value(self) -> u32 {
        match self {
            Feature::InitialPrompt => 0,
            Feature::TemperatureFallback => 1,
            Feature::LongForm => 2,
            Feature::Cancellation => 3,
            Feature::Pnc => 4,
            Feature::Itn => 5,
            Feature::Diarization => 6,
            Feature::Vocabulary => 7,
            Feature::ContextPrompt => 8,
            Feature::Instruct => 9,
            Feature::TranscriptPrefix => 10,
        }
    }
}

/// When the UI-facing committed text grows during a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CommitPolicy {
    /// The family's stable-prefix implementation. The default.
    #[default]
    Auto,
    /// Commit nothing during feed; finalize commits the final text.
    OnFinalize,
    /// Commit a raw prefix accepted by the stable-prefix implementation.
    StablePrefix,
}

/// Stream lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StreamState {
    Idle,
    Active,
    Finished,
    Failed,
}

/// The slot a family extension is pointed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExtSlot {
    /// `transcribe_run_params::family`.
    Run,
    /// `transcribe_stream_params::family`.
    Stream,
    /// `transcribe_diarize_params::family`.
    DiarizeRun,
}

impl ExtSlot {
    /// The C enum value (`transcribe_ext_slot`), for transcribe-cpp's conversion. Not public
    /// API: an exhaustive match here makes a new variant fail to compile until
    /// it has a value, and transcribe-cpp tests every value against the header.
    #[doc(hidden)]
    pub const fn c_value(self) -> u32 {
        match self {
            ExtSlot::Run => 0,
            ExtSlot::Stream => 1,
            ExtSlot::DiarizeRun => 2,
        }
    }
}

/// A kind of work a model can do (`transcribe_role`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Role {
    /// Transcription: `transcribe_cpp::Session`.
    Asr,
    /// Speaker diarization: `transcribe_cpp::DiarizeSession`.
    Diarize,
    /// Language identification: `transcribe_cpp::LangIdSession`.
    LangId,
    /// Voice activity detection: `transcribe_cpp::VadSession`.
    Vad,
    /// End-of-turn detection: `transcribe_cpp::TurnSession`.
    Turn,
}

impl Role {
    /// This role's `TRANSCRIBE_ROLE_*` bit. transcribe-cpp tests these against
    /// the C header's values.
    const fn bit(self) -> u32 {
        match self {
            Role::Asr => 1,
            Role::Diarize => 2,
            Role::LangId => 4,
            Role::Vad => 8,
            Role::Turn => 16,
        }
    }
}

/// The set of [`Role`]s a model serves (`transcribe_cpp::Model::roles`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Roles(u32);

impl Roles {
    /// A set from a raw `transcribe_role` bitmask (`TRANSCRIBE_ROLE_*` bits).
    /// Bits this version does not know are kept.
    pub const fn from_bits(bits: u32) -> Roles {
        Roles(bits)
    }

    /// Whether the model serves `role`.
    pub fn contains(self, role: Role) -> bool {
        self.0 & role.bit() != 0
    }

    /// The raw `transcribe_role` bitmask (`TRANSCRIBE_ROLE_*` bits).
    pub fn bits(self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_contains_exactly_its_bits() {
        let all = [
            Role::Asr,
            Role::Diarize,
            Role::LangId,
            Role::Vad,
            Role::Turn,
        ];
        for (i, role) in all.iter().enumerate() {
            let one = Roles::from_bits(role.bit());
            assert_eq!(one.bits(), role.bit());
            for (j, other) in all.iter().enumerate() {
                assert_eq!(one.contains(*other), i == j, "{role:?} vs {other:?}");
            }
        }
        let asr_vad = Roles::from_bits(Role::Asr.bit() | Role::Vad.bit() | 1 << 30);
        assert!(asr_vad.contains(Role::Asr) && asr_vad.contains(Role::Vad));
        assert!(!asr_vad.contains(Role::Turn));
        assert_eq!(asr_vad.bits(), 1 | 8 | 1 << 30);
    }
}
