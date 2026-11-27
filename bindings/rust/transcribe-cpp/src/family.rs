//! Materializing the typed family-extension options (from
//! `transcribe-cpp-types`) into the C extension structs.
//!
//! Each family exposes model-specific knobs via a kind-tagged extension struct
//! (`include/transcribe/<family>.h`). The typed options mirror those structs:
//! every field is an `Option`, and only the fields you set override the
//! defaults the C `*_init()` stamps.
//!
//! The extension enums are `#[non_exhaustive]` and live in another crate, so
//! each match below needs a `_` arm; the exact `=` pin on transcribe-cpp-types
//! keeps it unreachable, and it errors rather than send C a half-built struct.

use std::ffi::CString;

use transcribe_cpp_sys as sys;
use transcribe_cpp_types::{DiarizeExtension, RunExtension, SortformerPreset, StreamExtension};

use crate::error::{Error, Result};
use crate::types::ToRaw;

/// Build the owned C extension struct for a typed family extension.
pub(crate) trait Materialize {
    type Raw;
    fn materialize(&self) -> Self::Raw;
}

/// An extension variant this transcribe-cpp has no C mapping for. Unreachable
/// while transcribe-cpp-types is pinned to this crate's exact version.
fn unmapped(ext: &impl std::fmt::Debug) -> Error {
    Error::InvalidArgument(format!(
        "family extension {ext:?} is not supported by this transcribe-cpp build"
    ))
}

impl ToRaw for SortformerPreset {
    type Raw = sys::transcribe_sortformer_preset;
    fn to_raw(self) -> Self::Raw {
        use sys::transcribe_sortformer_preset as P;
        match self {
            SortformerPreset::Default => P::TRANSCRIBE_SORTFORMER_PRESET_DEFAULT,
            SortformerPreset::VeryHighLatency => P::TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY,
            SortformerPreset::HighLatency => P::TRANSCRIBE_SORTFORMER_PRESET_HIGH_LATENCY,
            SortformerPreset::LowLatency => P::TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY,
        }
    }
}

/// Owns a materialized diarize-slot C extension struct.
pub(crate) enum DiarizeExtRaw {
    Sortformer(Box<sys::transcribe_sortformer_diarize_ext>),
}

impl DiarizeExtRaw {
    pub(crate) fn ext_ptr(&self) -> *const sys::transcribe_ext {
        match self {
            DiarizeExtRaw::Sortformer(e) => {
                (&**e) as *const sys::transcribe_sortformer_diarize_ext
                    as *const sys::transcribe_ext
            }
        }
    }
}

impl Materialize for DiarizeExtension {
    type Raw = Result<DiarizeExtRaw>;
    fn materialize(&self) -> Self::Raw {
        Ok(match self {
            DiarizeExtension::Sortformer(o) => {
                let mut e: sys::transcribe_sortformer_diarize_ext = unsafe { std::mem::zeroed() };
                unsafe { sys::transcribe_sortformer_diarize_ext_init(&mut e) };
                set(&mut e.preset, o.preset.map(ToRaw::to_raw));
                DiarizeExtRaw::Sortformer(Box::new(e))
            }
            other => return Err(unmapped(other)),
        })
    }
}

/// Owns a materialized run-slot C extension struct (and any strings it points
/// at) so its address — and the `transcribe_ext` pointer handed to the C API —
/// stays valid for the duration of the native call. Boxed for a stable address
/// across moves of the holder.
pub(crate) enum RunExtRaw {
    Whisper {
        ext: Box<sys::transcribe_whisper_run_ext>,
        _prompt: Option<CString>,
    },
}

impl RunExtRaw {
    pub(crate) fn ext_ptr(&self) -> *const sys::transcribe_ext {
        match self {
            // `ext` is field 0, so &ext == &the family struct.
            RunExtRaw::Whisper { ext, .. } => {
                (&**ext) as *const sys::transcribe_whisper_run_ext as *const sys::transcribe_ext
            }
        }
    }
}

impl Materialize for RunExtension {
    type Raw = Result<RunExtRaw>;
    fn materialize(&self) -> Self::Raw {
        match self {
            RunExtension::Whisper(o) => {
                let mut ext: sys::transcribe_whisper_run_ext = unsafe { std::mem::zeroed() };
                unsafe { sys::transcribe_whisper_run_ext_init(&mut ext) };
                // Propagate an interior NUL as Error::Nul (like language /
                // target_language) instead of silently dropping the prompt.
                let prompt = o.initial_prompt.as_deref().map(CString::new).transpose()?;
                if let Some(c) = prompt.as_ref() {
                    ext.initial_prompt = c.as_ptr();
                }
                set(
                    &mut ext.condition_on_prev_tokens,
                    o.condition_on_prev_tokens,
                );
                set(&mut ext.temperature, o.temperature);
                set(&mut ext.temperature_inc, o.temperature_inc);
                set(&mut ext.compression_ratio_thold, o.compression_ratio_thold);
                set(&mut ext.logprob_thold, o.logprob_thold);
                set(&mut ext.no_speech_thold, o.no_speech_thold);
                set(&mut ext.max_prev_context_tokens, o.max_prev_context_tokens);
                set(&mut ext.seed, o.seed);
                set(&mut ext.max_initial_timestamp, o.max_initial_timestamp);
                Ok(RunExtRaw::Whisper {
                    ext: Box::new(ext),
                    _prompt: prompt,
                })
            }
            other => Err(unmapped(other)),
        }
    }
}

/// Owns a materialized stream-slot C extension struct (none carry strings).
pub(crate) enum StreamExtRaw {
    ParakeetStream(Box<sys::transcribe_parakeet_stream_ext>),
    ParakeetBuffered(Box<sys::transcribe_parakeet_buffered_stream_ext>),
    MoonshineStreaming(Box<sys::transcribe_moonshine_streaming_stream_ext>),
    VoxtralRealtime(Box<sys::transcribe_voxtral_realtime_stream_ext>),
}

impl StreamExtRaw {
    pub(crate) fn ext_ptr(&self) -> *const sys::transcribe_ext {
        match self {
            StreamExtRaw::ParakeetStream(e) => {
                (&**e) as *const sys::transcribe_parakeet_stream_ext as *const sys::transcribe_ext
            }
            StreamExtRaw::ParakeetBuffered(e) => {
                (&**e) as *const sys::transcribe_parakeet_buffered_stream_ext
                    as *const sys::transcribe_ext
            }
            StreamExtRaw::MoonshineStreaming(e) => {
                (&**e) as *const sys::transcribe_moonshine_streaming_stream_ext
                    as *const sys::transcribe_ext
            }
            StreamExtRaw::VoxtralRealtime(e) => {
                (&**e) as *const sys::transcribe_voxtral_realtime_stream_ext
                    as *const sys::transcribe_ext
            }
        }
    }
}

impl Materialize for StreamExtension {
    type Raw = Result<StreamExtRaw>;
    fn materialize(&self) -> Self::Raw {
        Ok(match self {
            StreamExtension::ParakeetStream(o) => {
                let mut e: sys::transcribe_parakeet_stream_ext = unsafe { std::mem::zeroed() };
                unsafe { sys::transcribe_parakeet_stream_ext_init(&mut e) };
                set(&mut e.att_context_right, o.att_context_right);
                StreamExtRaw::ParakeetStream(Box::new(e))
            }
            StreamExtension::ParakeetBuffered(o) => {
                let mut e: sys::transcribe_parakeet_buffered_stream_ext =
                    unsafe { std::mem::zeroed() };
                unsafe { sys::transcribe_parakeet_buffered_stream_ext_init(&mut e) };
                set(&mut e.left_ms, o.left_ms);
                set(&mut e.chunk_ms, o.chunk_ms);
                set(&mut e.right_ms, o.right_ms);
                StreamExtRaw::ParakeetBuffered(Box::new(e))
            }
            StreamExtension::MoonshineStreaming(o) => {
                let mut e: sys::transcribe_moonshine_streaming_stream_ext =
                    unsafe { std::mem::zeroed() };
                unsafe { sys::transcribe_moonshine_streaming_stream_ext_init(&mut e) };
                set(&mut e.min_decode_interval_ms, o.min_decode_interval_ms);
                StreamExtRaw::MoonshineStreaming(Box::new(e))
            }
            StreamExtension::VoxtralRealtime(o) => {
                let mut e: sys::transcribe_voxtral_realtime_stream_ext =
                    unsafe { std::mem::zeroed() };
                unsafe { sys::transcribe_voxtral_realtime_stream_ext_init(&mut e) };
                set(&mut e.num_delay_tokens, o.num_delay_tokens);
                set(&mut e.min_decode_interval_ms, o.min_decode_interval_ms);
                StreamExtRaw::VoxtralRealtime(Box::new(e))
            }
            other => return Err(unmapped(other)),
        })
    }
}

/// Overwrite `slot` only when the caller provided a value.
fn set<T>(slot: &mut T, value: Option<T>) {
    if let Some(v) = value {
        *slot = v;
    }
}
