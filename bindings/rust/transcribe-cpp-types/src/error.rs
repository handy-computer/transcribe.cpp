//! The plain-data side of transcribe-cpp's errors: [`ErrorKind`] and the
//! [`ErrorReport`] snapshot. `transcribe_cpp::Error` itself (with its native
//! status mapping) stays in transcribe-cpp, which converts it into an
//! [`ErrorReport`].

use crate::result::Transcript;

/// The category of a `transcribe_cpp::Error`, without its payload. One kind
/// per variant.
///
/// Stable for matching and, under the `serde` feature, serialized as a
/// snake_case string (`"aborted"`, `"model_file_not_found"`, ...). A kind
/// this version does not know deserializes as [`ErrorKind::Other`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum ErrorKind {
    InvalidArgument,
    NotImplemented,
    ModelFileNotFound,
    ModelLoad,
    OutOfMemory,
    Backend,
    Unsupported,
    BadStructSize,
    InputTooLong,
    Aborted,
    OutputTruncated,
    OutputRepetition,
    UnsupportedRole,
    VersionMismatch,
    Nul,
    Busy,
    InputTooShort,
    #[cfg_attr(feature = "serde", serde(other))]
    Other,
}

/// An owned, plain-data snapshot of a `transcribe_cpp::Error`: what crosses a
/// process or IPC boundary (e.g. a Tauri command's error).
///
/// `transcribe_cpp::Error` itself serializes to exactly this shape under the
/// `serde` feature, and converts into it with `From`; deserialize into
/// `ErrorReport` on the receiving side (an `Error` cannot be rebuilt from it).
/// JSON looks like:
///
/// ```json
/// { "kind": "aborted", "message": "operation aborted: ...", "status": 13, "partial": { "text": "..." } }
/// ```
///
/// `partial` is `null` unless the error carried a partial transcript. It is
/// always written (never skipped) so non-self-describing formats such as
/// postcard round-trip.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ErrorReport {
    /// The error category.
    pub kind: ErrorKind,
    /// The error's `Display` text.
    pub message: String,
    /// `transcribe_cpp::Error::raw_status`: the `transcribe_status`, or 0 for
    /// Rust-side errors.
    #[cfg_attr(feature = "serde", serde(default))]
    pub status: i32,
    /// `transcribe_cpp::Error::partial`, if any.
    #[cfg_attr(feature = "serde", serde(default))]
    pub partial: Option<Transcript>,
}

impl ErrorReport {
    /// A report of `kind` with `message`, no native status (0) and no partial
    /// transcript: for errors raised outside the engine, such as a host
    /// rejecting a request before it reaches a worker. Set
    /// [`status`](Self::status) and [`partial`](Self::partial) afterwards if
    /// you have them.
    ///
    /// ```
    /// use transcribe_cpp_types::{ErrorKind, ErrorReport};
    /// let report = ErrorReport::new(ErrorKind::InputTooLong, "frame of 70000 samples exceeds 65536");
    /// assert_eq!((report.status, report.partial), (0, None));
    /// ```
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        ErrorReport {
            kind,
            message: message.into(),
            status: 0,
            partial: None,
        }
    }
}
