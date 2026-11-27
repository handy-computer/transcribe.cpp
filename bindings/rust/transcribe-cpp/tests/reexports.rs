//! Every plain-data type of transcribe-cpp-types is re-exported at this
//! crate's root as the very same type.
//!
//! cargo semver-checks does not follow re-exports from another crate, so it
//! would not notice a name dropped from the `pub use transcribe_cpp_types::{..}`
//! in lib.rs; the compile-time identities below do. The runtime test keeps this
//! list complete: a type added to transcribe-cpp-types must be added here (and
//! re-exported).

use std::collections::BTreeSet;

macro_rules! reexported {
    ($($name:ident),* $(,)?) => {
        /// Compiles only if `transcribe_cpp::X` exists and is `transcribe_cpp_types::X`.
        #[allow(dead_code)]
        fn identities() {
            $( let _: fn(transcribe_cpp::$name) -> transcribe_cpp_types::$name = |x| x; )*
        }
        const NAMES: &[&str] = &[$(stringify!($name)),*];
    };
}

reexported!(
    Backend,
    BackendMask,
    Capabilities,
    CommitPolicy,
    DeviceType,
    Diarize,
    DiarizeExtension,
    DiarizeInfo,
    DiarizeOptions,
    DiarizeSessionOptions,
    ErrorKind,
    ErrorReport,
    ExtSlot,
    Feature,
    Itn,
    KvType,
    LangIdCandidate,
    LangIdInfo,
    LangIdOptions,
    LangIdResult,
    LangIdSessionOptions,
    MoonshineStreamingOptions,
    ParakeetBufferedStreamOptions,
    ParakeetStreamOptions,
    Pnc,
    Role,
    Roles,
    RunExtension,
    RunOptions,
    Segment,
    SessionLimits,
    SessionOptions,
    SortformerDiarizeOptions,
    SortformerPreset,
    SpeakerSegment,
    StreamExtension,
    StreamOptions,
    StreamState,
    StreamText,
    StreamUpdate,
    Task,
    TimestampKind,
    Timings,
    Token,
    Transcript,
    TurnInfo,
    TurnOptions,
    TurnResult,
    TurnSessionOptions,
    VadInfo,
    VadOptions,
    VadResult,
    VadSegment,
    VadSessionOptions,
    VoxtralRealtimeStreamOptions,
    WhisperRunOptions,
    Word,
);

/// The names transcribe-cpp-types exports at its root, read from its lib.rs
/// `pub use` items. None outside the repository (a packaged crate has no
/// sibling source).
fn types_crate_exports() -> Option<BTreeSet<String>> {
    let lib =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../transcribe-cpp-types/src/lib.rs");
    let src = std::fs::read_to_string(lib).ok()?;
    let mut names = BTreeSet::new();
    let mut rest = src.as_str();
    while let Some(start) = rest.find("pub use ") {
        let item = &rest[start..];
        let end = item.find(';').expect("unterminated pub use");
        // `pub use module::{A, B as C};` or `pub use module::A;`: the exported
        // names follow the last `::`, one per comma (the last word: `C`).
        let body = &item["pub use ".len()..end];
        let tail = body.rsplit_once("::").map_or(body, |(_, t)| t);
        for entry in tail.trim_matches(|c| c == '{' || c == '}').split(',') {
            if let Some(name) = entry.split_whitespace().last() {
                names.insert(name.to_string());
            }
        }
        rest = &item[end..];
    }
    Some(names)
}

#[test]
fn every_types_crate_export_is_reexported() {
    let Some(exported) = types_crate_exports() else {
        eprintln!("transcribe-cpp-types sources not found; skipped");
        return;
    };
    let listed: BTreeSet<String> = NAMES.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        exported, listed,
        "transcribe-cpp-types' root exports differ from this list; re-export the new type \
         from transcribe-cpp's lib.rs and add it here"
    );
}
