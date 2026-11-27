//! Conversions between the plain-data enums (from `transcribe-cpp-types`) and
//! the generated `transcribe-cpp-sys` NewTypes, plus [`AbiStruct`].
//!
//! Conversions INTO C are total for the exhaustive enums; conversions OUT of C
//! validate the raw integer and fall back to a safe default for an unknown
//! value (enum hygiene — we never assume the library only ever hands back
//! values this version knows about).
//!
//! The enums live in another crate, so the conversions are trait impls here
//! (inherent methods cannot be added to a foreign type). A `#[non_exhaustive]`
//! enum ([`Task`], [`Feature`], [`ExtSlot`]) could only be matched here with a
//! `_` arm the compiler cannot check, so instead each carries its C value in an
//! exhaustive (hidden) `c_value()` in transcribe-cpp-types: a new variant does
//! not compile without one, and the tests below check every value against the
//! header.

use transcribe_cpp_sys as sys;
use transcribe_cpp_types::{
    Backend, CommitPolicy, Diarize, ExtSlot, Feature, Itn, KvType, Pnc, StreamState, Task,
    TimestampKind,
};

/// Convert a plain-data value into its C representation.
pub(crate) trait ToRaw {
    type Raw;
    fn to_raw(self) -> Self::Raw;
}

/// Build a plain-data value from its C representation (copying any strings).
pub(crate) trait FromRaw<R> {
    fn from_raw(raw: R) -> Self;
}

impl ToRaw for Task {
    type Raw = sys::transcribe_task;
    fn to_raw(self) -> Self::Raw {
        sys::transcribe_task(self.c_value())
    }
}

impl ToRaw for TimestampKind {
    type Raw = sys::transcribe_timestamp_kind;
    fn to_raw(self) -> Self::Raw {
        use sys::transcribe_timestamp_kind as K;
        match self {
            TimestampKind::None => K::TRANSCRIBE_TIMESTAMPS_NONE,
            TimestampKind::Auto => K::TRANSCRIBE_TIMESTAMPS_AUTO,
            TimestampKind::Segment => K::TRANSCRIBE_TIMESTAMPS_SEGMENT,
            TimestampKind::Word => K::TRANSCRIBE_TIMESTAMPS_WORD,
            TimestampKind::Token => K::TRANSCRIBE_TIMESTAMPS_TOKEN,
        }
    }
}

impl FromRaw<sys::transcribe_timestamp_kind> for TimestampKind {
    fn from_raw(raw: sys::transcribe_timestamp_kind) -> Self {
        use sys::transcribe_timestamp_kind as K;
        match raw {
            K::TRANSCRIBE_TIMESTAMPS_AUTO => TimestampKind::Auto,
            K::TRANSCRIBE_TIMESTAMPS_SEGMENT => TimestampKind::Segment,
            K::TRANSCRIBE_TIMESTAMPS_WORD => TimestampKind::Word,
            K::TRANSCRIBE_TIMESTAMPS_TOKEN => TimestampKind::Token,
            // includes TRANSCRIBE_TIMESTAMPS_NONE and any unknown value
            _ => TimestampKind::None,
        }
    }
}

impl ToRaw for KvType {
    type Raw = sys::transcribe_kv_type;
    fn to_raw(self) -> Self::Raw {
        use sys::transcribe_kv_type as K;
        match self {
            KvType::Auto => K::TRANSCRIBE_KV_TYPE_AUTO,
            KvType::F32 => K::TRANSCRIBE_KV_TYPE_F32,
            KvType::F16 => K::TRANSCRIBE_KV_TYPE_F16,
        }
    }
}

impl ToRaw for Pnc {
    type Raw = sys::transcribe_pnc_mode;
    fn to_raw(self) -> Self::Raw {
        use sys::transcribe_pnc_mode as P;
        match self {
            Pnc::Default => P::TRANSCRIBE_PNC_MODE_DEFAULT,
            Pnc::Off => P::TRANSCRIBE_PNC_MODE_OFF,
            Pnc::On => P::TRANSCRIBE_PNC_MODE_ON,
        }
    }
}

impl ToRaw for Itn {
    type Raw = sys::transcribe_itn_mode;
    fn to_raw(self) -> Self::Raw {
        use sys::transcribe_itn_mode as I;
        match self {
            Itn::Default => I::TRANSCRIBE_ITN_MODE_DEFAULT,
            Itn::Off => I::TRANSCRIBE_ITN_MODE_OFF,
            Itn::On => I::TRANSCRIBE_ITN_MODE_ON,
        }
    }
}

impl ToRaw for Diarize {
    type Raw = sys::transcribe_diarize_mode;
    fn to_raw(self) -> Self::Raw {
        use sys::transcribe_diarize_mode as D;
        match self {
            Diarize::Default => D::TRANSCRIBE_DIARIZE_MODE_DEFAULT,
            Diarize::Off => D::TRANSCRIBE_DIARIZE_MODE_OFF,
            Diarize::On => D::TRANSCRIBE_DIARIZE_MODE_ON,
        }
    }
}

impl ToRaw for Backend {
    type Raw = sys::transcribe_backend_request;
    fn to_raw(self) -> Self::Raw {
        use sys::transcribe_backend_request as B;
        match self {
            Backend::Auto => B::TRANSCRIBE_BACKEND_AUTO,
            Backend::Cpu => B::TRANSCRIBE_BACKEND_CPU,
            Backend::CpuAccel => B::TRANSCRIBE_BACKEND_CPU_ACCEL,
            Backend::Metal => B::TRANSCRIBE_BACKEND_METAL,
            Backend::Vulkan => B::TRANSCRIBE_BACKEND_VULKAN,
            Backend::Cuda => B::TRANSCRIBE_BACKEND_CUDA,
            Backend::Rocm => B::TRANSCRIBE_BACKEND_ROCM,
        }
    }
}

impl ToRaw for Feature {
    type Raw = sys::transcribe_feature;
    fn to_raw(self) -> Self::Raw {
        sys::transcribe_feature(self.c_value())
    }
}

impl ToRaw for CommitPolicy {
    type Raw = sys::transcribe_stream_commit_policy;
    fn to_raw(self) -> Self::Raw {
        use sys::transcribe_stream_commit_policy as C;
        match self {
            CommitPolicy::Auto => C::TRANSCRIBE_STREAM_COMMIT_AUTO,
            CommitPolicy::OnFinalize => C::TRANSCRIBE_STREAM_COMMIT_ON_FINALIZE,
            CommitPolicy::StablePrefix => C::TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX,
        }
    }
}

impl FromRaw<sys::transcribe_stream_state> for StreamState {
    fn from_raw(raw: sys::transcribe_stream_state) -> Self {
        use sys::transcribe_stream_state as S;
        match raw {
            S::TRANSCRIBE_STREAM_ACTIVE => StreamState::Active,
            S::TRANSCRIBE_STREAM_FINISHED => StreamState::Finished,
            S::TRANSCRIBE_STREAM_FAILED => StreamState::Failed,
            _ => StreamState::Idle,
        }
    }
}

impl ToRaw for ExtSlot {
    type Raw = sys::transcribe_ext_slot;
    fn to_raw(self) -> Self::Raw {
        sys::transcribe_ext_slot(self.c_value())
    }
}

/// A public ABI struct, for size/alignment introspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AbiStruct {
    ModelLoadParams,
    SessionParams,
    RunParams,
    StreamParams,
    Capabilities,
    Timings,
    Segment,
    Word,
    Token,
    StreamUpdate,
    StreamText,
    SessionLimits,
    Ext,
    DeviceInfo,
    SpeakerSegment,
    BackendInitParams,
    DiarizeInfo,
    DiarizeSessionParams,
    DiarizeParams,
    LangIdInfo,
    LangIdSessionParams,
    LangIdParams,
    LangIdResult,
    LangIdCandidate,
    VadInfo,
    VadSessionParams,
    VadParams,
    VadResult,
    VadSegment,
    TurnInfo,
    TurnSessionParams,
    TurnParams,
    TurnResult,
}

impl AbiStruct {
    pub(crate) fn to_raw(self) -> sys::transcribe_abi_struct {
        use sys::transcribe_abi_struct as A;
        match self {
            AbiStruct::ModelLoadParams => A::TRANSCRIBE_ABI_MODEL_LOAD_PARAMS,
            AbiStruct::SessionParams => A::TRANSCRIBE_ABI_SESSION_PARAMS,
            AbiStruct::RunParams => A::TRANSCRIBE_ABI_RUN_PARAMS,
            AbiStruct::StreamParams => A::TRANSCRIBE_ABI_STREAM_PARAMS,
            AbiStruct::Capabilities => A::TRANSCRIBE_ABI_CAPABILITIES,
            AbiStruct::Timings => A::TRANSCRIBE_ABI_TIMINGS,
            AbiStruct::Segment => A::TRANSCRIBE_ABI_SEGMENT,
            AbiStruct::Word => A::TRANSCRIBE_ABI_WORD,
            AbiStruct::Token => A::TRANSCRIBE_ABI_TOKEN,
            AbiStruct::StreamUpdate => A::TRANSCRIBE_ABI_STREAM_UPDATE,
            AbiStruct::StreamText => A::TRANSCRIBE_ABI_STREAM_TEXT,
            AbiStruct::SessionLimits => A::TRANSCRIBE_ABI_SESSION_LIMITS,
            AbiStruct::Ext => A::TRANSCRIBE_ABI_EXT,
            AbiStruct::DeviceInfo => A::TRANSCRIBE_ABI_DEVICE_INFO,
            AbiStruct::SpeakerSegment => A::TRANSCRIBE_ABI_SPEAKER_SEGMENT,
            AbiStruct::BackendInitParams => A::TRANSCRIBE_ABI_BACKEND_INIT_PARAMS,
            AbiStruct::DiarizeInfo => A::TRANSCRIBE_ABI_DIARIZE_INFO,
            AbiStruct::DiarizeSessionParams => A::TRANSCRIBE_ABI_DIARIZE_SESSION_PARAMS,
            AbiStruct::DiarizeParams => A::TRANSCRIBE_ABI_DIARIZE_PARAMS,
            AbiStruct::LangIdInfo => A::TRANSCRIBE_ABI_LANGID_INFO,
            AbiStruct::LangIdSessionParams => A::TRANSCRIBE_ABI_LANGID_SESSION_PARAMS,
            AbiStruct::LangIdParams => A::TRANSCRIBE_ABI_LANGID_PARAMS,
            AbiStruct::LangIdResult => A::TRANSCRIBE_ABI_LANGID_RESULT,
            AbiStruct::LangIdCandidate => A::TRANSCRIBE_ABI_LANGID_CANDIDATE,
            AbiStruct::VadInfo => A::TRANSCRIBE_ABI_VAD_INFO,
            AbiStruct::VadSessionParams => A::TRANSCRIBE_ABI_VAD_SESSION_PARAMS,
            AbiStruct::VadParams => A::TRANSCRIBE_ABI_VAD_PARAMS,
            AbiStruct::VadResult => A::TRANSCRIBE_ABI_VAD_RESULT,
            AbiStruct::VadSegment => A::TRANSCRIBE_ABI_VAD_SEGMENT,
            AbiStruct::TurnInfo => A::TRANSCRIBE_ABI_TURN_INFO,
            AbiStruct::TurnSessionParams => A::TRANSCRIBE_ABI_TURN_SESSION_PARAMS,
            AbiStruct::TurnParams => A::TRANSCRIBE_ABI_TURN_PARAMS,
            AbiStruct::TurnResult => A::TRANSCRIBE_ABI_TURN_RESULT,
        }
    }
}

#[cfg(test)]
mod tests {
    //! The `#[non_exhaustive]` enums take their C value from transcribe-cpp-types
    //! (`c_value()`, see the module docs); each variant is checked here against
    //! the header's constant, both ways. A variant added there needs a row here.
    use super::*;
    use transcribe_cpp_types::{BackendMask, Role, Roles};

    /// Every listed variant maps to its C value and back, and the C values are
    /// exactly `0..n` (no two variants share one, none is skipped).
    fn round_trip<V, R>(rows: &[(V, R)], to_raw: impl Fn(V) -> R, raw_u32: impl Fn(R) -> u32)
    where
        V: Copy + PartialEq + std::fmt::Debug,
        R: Copy + PartialEq + std::fmt::Debug,
    {
        let back = |raw: R| rows.iter().find(|(_, r)| *r == raw).map(|(v, _)| *v);
        for &(variant, raw) in rows {
            assert_eq!(to_raw(variant), raw, "{variant:?} -> C");
            assert_eq!(back(raw), Some(variant), "C {raw:?} -> back");
        }
        let mut values: Vec<u32> = rows.iter().map(|(_, r)| raw_u32(*r)).collect();
        values.sort_unstable();
        assert_eq!(values, (0..rows.len() as u32).collect::<Vec<_>>());
    }

    #[test]
    fn task_maps_every_variant() {
        use sys::transcribe_task as T;
        round_trip(
            &[
                (Task::Transcribe, T::TRANSCRIBE_TASK_TRANSCRIBE),
                (Task::Translate, T::TRANSCRIBE_TASK_TRANSLATE),
                (Task::Instruct, T::TRANSCRIBE_TASK_INSTRUCT),
            ],
            Task::to_raw,
            |r| r.0,
        );
    }

    #[test]
    fn feature_maps_every_variant() {
        use sys::transcribe_feature as F;
        round_trip(
            &[
                (Feature::InitialPrompt, F::TRANSCRIBE_FEATURE_INITIAL_PROMPT),
                (
                    Feature::TemperatureFallback,
                    F::TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK,
                ),
                (Feature::LongForm, F::TRANSCRIBE_FEATURE_LONG_FORM),
                (Feature::Cancellation, F::TRANSCRIBE_FEATURE_CANCELLATION),
                (Feature::Pnc, F::TRANSCRIBE_FEATURE_PNC),
                (Feature::Itn, F::TRANSCRIBE_FEATURE_ITN),
                (Feature::Diarization, F::TRANSCRIBE_FEATURE_DIARIZATION),
                (Feature::Vocabulary, F::TRANSCRIBE_FEATURE_VOCABULARY),
                (Feature::ContextPrompt, F::TRANSCRIBE_FEATURE_CONTEXT_PROMPT),
                (Feature::Instruct, F::TRANSCRIBE_FEATURE_INSTRUCT),
                (
                    Feature::TranscriptPrefix,
                    F::TRANSCRIBE_FEATURE_TRANSCRIPT_PREFIX,
                ),
            ],
            Feature::to_raw,
            |r| r.0,
        );
    }

    #[test]
    fn ext_slot_maps_every_variant() {
        use sys::transcribe_ext_slot as E;
        round_trip(
            &[
                (ExtSlot::Run, E::TRANSCRIBE_EXT_SLOT_RUN),
                (ExtSlot::Stream, E::TRANSCRIBE_EXT_SLOT_STREAM),
                (ExtSlot::DiarizeRun, E::TRANSCRIBE_EXT_SLOT_DIARIZE_RUN),
            ],
            ExtSlot::to_raw,
            |r| r.0,
        );
    }

    /// `Roles::contains` uses role bits hard-coded in transcribe-cpp-types.
    #[test]
    fn role_bits_match_the_header() {
        use sys::transcribe_role as R;
        let rows = [
            (Role::Asr, R::TRANSCRIBE_ROLE_ASR),
            (Role::Diarize, R::TRANSCRIBE_ROLE_DIARIZE),
            (Role::LangId, R::TRANSCRIBE_ROLE_LANGID),
            (Role::Vad, R::TRANSCRIBE_ROLE_VAD),
            (Role::Turn, R::TRANSCRIBE_ROLE_TURN),
        ];
        for (role, bit) in rows {
            let roles = Roles::from_bits(bit.0);
            for (other, _) in rows {
                assert_eq!(roles.contains(other), other == role, "{bit:?} vs {other:?}");
            }
        }
    }

    /// The `BackendMask` constants are hard-coded in transcribe-cpp-types.
    #[test]
    fn backend_mask_bits_match_the_header() {
        for (mask, bits) in [
            (BackendMask::CPU, sys::TRANSCRIBE_BACKEND_MASK_CPU),
            (BackendMask::METAL, sys::TRANSCRIBE_BACKEND_MASK_METAL),
            (BackendMask::VULKAN, sys::TRANSCRIBE_BACKEND_MASK_VULKAN),
            (BackendMask::CUDA, sys::TRANSCRIBE_BACKEND_MASK_CUDA),
            (BackendMask::ROCM, sys::TRANSCRIBE_BACKEND_MASK_ROCM),
            (BackendMask::OTHER, sys::TRANSCRIBE_BACKEND_MASK_OTHER),
            (BackendMask::ALL, sys::TRANSCRIBE_BACKEND_MASK_ALL),
        ] {
            assert_eq!(mask.bits(), bits, "{mask:?}");
        }
    }
}
