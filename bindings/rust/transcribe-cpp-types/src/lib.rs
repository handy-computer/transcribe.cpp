//! The plain-data types of
//! [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)'s Rust
//! binding: options, results, enums, errors, abilities, stream updates and
//! family extensions. No handles, no FFI, nothing native.
//!
//! [`transcribe-cpp`](https://docs.rs/transcribe-cpp) depends on this crate at
//! the exact same version and re-exports every type at its own root, so
//! `transcribe_cpp::RunOptions` and `transcribe_cpp_types::RunOptions` are the
//! same type. Depend on this crate alone to name those types without the engine
//! (`transcribe-cpp-sys`, CMake, ggml, GPU frameworks) in your dependency graph:
//! a process that talks to transcribe.cpp workers over IPC, say, in the
//! workers' own types.
//!
//! # Features
//!
//! - `serde`: `Serialize` / `Deserialize` on every type.
//!   `transcribe-cpp/serde` turns it on.

#![doc(html_root_url = "https://docs.rs/transcribe-cpp-types")]
#![warn(missing_debug_implementations)]

mod backend;
mod diarize;
mod error;
mod family;
mod langid;
mod model;
mod result;
mod session;
mod streaming;
mod turn;
mod types;
mod vad;

pub use backend::{BackendMask, DeviceType};
pub use diarize::{DiarizeInfo, DiarizeOptions, DiarizeSessionOptions};
pub use error::{ErrorKind, ErrorReport};
pub use family::{
    DiarizeExtension, MoonshineStreamingOptions, ParakeetBufferedStreamOptions,
    ParakeetStreamOptions, RunExtension, SortformerDiarizeOptions, SortformerPreset,
    StreamExtension, VoxtralRealtimeStreamOptions, WhisperRunOptions,
};
pub use langid::{LangIdCandidate, LangIdInfo, LangIdOptions, LangIdResult, LangIdSessionOptions};
pub use model::{Capabilities, SessionLimits, SessionOptions};
pub use result::{Segment, SpeakerSegment, Timings, Token, Transcript, Word};
pub use session::RunOptions;
pub use streaming::{StreamOptions, StreamText, StreamUpdate};
pub use turn::{TurnInfo, TurnOptions, TurnResult, TurnSessionOptions};
pub use types::{
    Backend, CommitPolicy, Diarize, ExtSlot, Feature, Itn, KvType, Pnc, Role, Roles, StreamState,
    Task, TimestampKind,
};
pub use vad::{VadInfo, VadOptions, VadResult, VadSegment, VadSessionOptions};
