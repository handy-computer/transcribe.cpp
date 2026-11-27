// transcribe-cpp's `Error` serializes as an `ErrorReport`. It stays in
// transcribe-cpp, so only the generator and transcribe-cpp's `tests/serde.rs`
// include this file (next to `values.rs`, as `crate::values`).

use crate::tc::Error;
use crate::values::transcript;

/// `(file stem, JSON, postcard)` per error value. Serialize-only: an `Error`
/// cannot be deserialized (the receiving side reads an `ErrorReport`).
pub fn error_goldens() -> Vec<(&'static str, String, Vec<u8>)> {
    let errors = [
        (
            "Error.aborted",
            Error::Aborted {
                message: "run: aborted (status 13)".to_string(),
                partial: Some(Box::new(transcript())),
            },
        ),
        (
            "Error.truncated",
            Error::OutputTruncated {
                message: "run: truncated (status 15)".to_string(),
                partial: None,
            },
        ),
        ("Error.busy", Error::Busy("a stream is active".to_string())),
        (
            "Error.nul",
            Error::Nul(std::ffi::CString::new("a\0b").unwrap_err()),
        ),
    ];
    errors
        .into_iter()
        .map(|(name, e)| {
            (
                name,
                serde_json::to_string(&e).unwrap(),
                postcard::to_allocvec(&e).unwrap(),
            )
        })
        .collect()
}
