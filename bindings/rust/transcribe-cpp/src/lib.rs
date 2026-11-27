//! Safe, idiomatic Rust bindings for
//! [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), a C/C++
//! speech-to-text library built on ggml.
//!
//! The raw FFI lives in the [`transcribe-cpp-sys`](transcribe_cpp_sys) crate
//! (re-exported as [`sys`]); this crate wraps it with owned result types, an
//! error enum, RAII handles, and a thread-safety model that matches the C
//! contract.
//!
//! The plain-data types (options, results, enums, [`ErrorReport`], ...) are
//! defined in [`transcribe-cpp-types`](transcribe_cpp_types), a pure-Rust crate
//! with no native dependency, and re-exported here: a program that only needs
//! to name them (one talking to transcribe.cpp worker processes, say) can
//! depend on that crate alone and keep the engine out of its build.
//!
//! # Quickstart
//!
//! ```no_run
//! use transcribe_cpp::{Model, RunOptions};
//!
//! let mut session = Model::load("model.gguf")?.session()?;
//! let pcm: Vec<f32> = vec![/* 16 kHz mono samples in [-1, 1] */];
//! let result = session.run(&pcm, &RunOptions::default())?;
//! println!("{}", result.text);
//! # Ok::<(), transcribe_cpp::Error>(())
//! ```
//!
//! # Threading
//!
//! - [`Model`] is `Send + Sync` and cheap to clone (`Arc`-backed). The native
//!   model is freed only after the last [`Model`] handle and every [`Session`]
//!   derived from it are dropped — in any order.
//! - [`Session`] is `Send` but not `Sync`; its mutating calls take `&mut self`,
//!   so the type system enforces one-call-at-a-time.
//! - **0.x concurrency:** the C library permits at most one in-flight compute
//!   across all sessions of a model, [`DiarizeSession`]s, [`LangIdSession`]s,
//!   [`VadSession`]s and [`TurnSession`]s included. This crate enforces that
//!   with a per-model mutex (held only for the native compute call), so
//!   concurrent calls from many sessions queue rather than race. An active ASR
//!   stream holds the model's stream lease; other runs, including VAD and TURN
//!   `run` / `stream_feed` / `stream_flush`, return [`Error::Busy`] meanwhile.
//!   VAD and TURN streams do not take the lease. For real parallelism, use one
//!   [`Model`] per worker today.
//!
//! # ABI verification
//!
//! The per-field struct-layout check that the ctypes binding performs is
//! **waived** here: bindgen takes every struct's layout from a real compiler at
//! generation time, so the generated FFI cannot disagree with the headers it
//! was built against. The load-time base-version lock (see [`Model::load`]) is
//! retained.

#![doc(html_root_url = "https://docs.rs/transcribe-cpp")]
#![warn(missing_debug_implementations)]

pub use transcribe_cpp_sys as sys;

mod backend;
mod cancel;
mod diarize;
mod error;
mod family;
mod langid;
mod logging;
mod model;
mod result;
mod session;
mod streaming;
mod turn;
mod types;
mod vad;
mod version;

// The plain-data types live in transcribe-cpp-types (pure Rust, no engine), at
// the exact same version; they are re-exported here at their pre-0.5.1 paths.
// `doc(inline)` keeps them documented on this crate's pages, as before.
#[doc(inline)]
pub use transcribe_cpp_types::{
    Backend, BackendMask, Capabilities, CommitPolicy, DeviceType, Diarize, DiarizeExtension,
    DiarizeInfo, DiarizeOptions, DiarizeSessionOptions, ErrorKind, ErrorReport, ExtSlot, Feature,
    Itn, KvType, LangIdCandidate, LangIdInfo, LangIdOptions, LangIdResult, LangIdSessionOptions,
    MoonshineStreamingOptions, ParakeetBufferedStreamOptions, ParakeetStreamOptions, Pnc, Role,
    Roles, RunExtension, RunOptions, Segment, SessionLimits, SessionOptions,
    SortformerDiarizeOptions, SortformerPreset, SpeakerSegment, StreamExtension, StreamOptions,
    StreamState, StreamText, StreamUpdate, Task, TimestampKind, Timings, Token, Transcript,
    TurnInfo, TurnOptions, TurnResult, TurnSessionOptions, VadInfo, VadOptions, VadResult,
    VadSegment, VadSessionOptions, VoxtralRealtimeStreamOptions, WhisperRunOptions, Word,
};

pub use backend::{
    allowed_backends, backend_available, device_count, devices, init_backends,
    init_backends_default, init_backends_with, Device,
};
pub use cancel::CancelToken;
pub use diarize::DiarizeSession;
pub use error::{Error, Result};
pub use langid::LangIdSession;
pub use logging::{disable_logging, init_logging};
pub use model::{Model, ModelOptions};
pub use session::{Session, Stream};
pub use turn::TurnSession;
pub use types::AbiStruct;
pub use vad::VadSession;
pub use version::{
    abi_struct_align, abi_struct_size, compiled_version, header_hash, version, version_commit,
};

/// Convenience: load a model, transcribe one PCM buffer, and return the result.
///
/// For repeated transcription, hold a [`Session`] instead of paying model load
/// on every call.
pub fn transcribe(
    model_path: impl AsRef<std::path::Path>,
    pcm: &[f32],
    options: &RunOptions,
) -> Result<Transcript> {
    let mut session = Model::load(model_path)?.session()?;
    session.run(pcm, options)
}
