//! Materializing the owned result types (from `transcribe-cpp-types`).
//!
//! Every result is fully materialized into owned types — all text is copied at
//! the FFI boundary (`CStr -> String`), so no borrowed session pointer ever
//! escapes to user code (the "copy at the FFI boundary" rule). Holding a
//! [`Transcript`](crate::Transcript) across the next `run`/`stream` call is
//! therefore always safe.

use std::ffi::CStr;
use std::os::raw::c_char;

use transcribe_cpp_sys as sys;
use transcribe_cpp_types::{Segment, SpeakerSegment, Timings, Token, Word};

use crate::types::FromRaw;

/// Copy a borrowed C string into an owned `String` (empty for NULL).
pub(crate) fn owned_str(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// Copy a borrowed C string into `Some(String)`, or `None` for NULL/empty.
pub(crate) fn owned_opt_str(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

impl FromRaw<&sys::transcribe_segment> for Segment {
    fn from_raw(raw: &sys::transcribe_segment) -> Self {
        Segment {
            t0_ms: raw.t0_ms,
            t1_ms: raw.t1_ms,
            first_word: raw.first_word,
            n_words: raw.n_words,
            first_token: raw.first_token,
            n_tokens: raw.n_tokens,
            text: owned_str(raw.text),
            speaker_id: raw.speaker_id,
        }
    }
}

impl FromRaw<&sys::transcribe_speaker_segment> for SpeakerSegment {
    fn from_raw(raw: &sys::transcribe_speaker_segment) -> Self {
        SpeakerSegment {
            t0_ms: raw.t0_ms,
            t1_ms: raw.t1_ms,
            speaker_id: raw.speaker_id,
            p: raw.p,
        }
    }
}

impl FromRaw<&sys::transcribe_word> for Word {
    fn from_raw(raw: &sys::transcribe_word) -> Self {
        Word {
            t0_ms: raw.t0_ms,
            t1_ms: raw.t1_ms,
            seg_index: raw.seg_index,
            first_token: raw.first_token,
            n_tokens: raw.n_tokens,
            text: owned_str(raw.text),
        }
    }
}

impl FromRaw<&sys::transcribe_token> for Token {
    fn from_raw(raw: &sys::transcribe_token) -> Self {
        Token {
            id: raw.id,
            p: raw.p,
            t0_ms: raw.t0_ms,
            t1_ms: raw.t1_ms,
            seg_index: raw.seg_index,
            word_index: raw.word_index,
            text: owned_str(raw.text),
        }
    }
}

impl FromRaw<&sys::transcribe_timings> for Timings {
    fn from_raw(raw: &sys::transcribe_timings) -> Self {
        Timings {
            load_ms: raw.load_ms,
            mel_ms: raw.mel_ms,
            encode_ms: raw.encode_ms,
            decode_ms: raw.decode_ms,
        }
    }
}
