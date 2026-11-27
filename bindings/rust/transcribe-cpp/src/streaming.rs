//! The stream-params builder and the conversions of the streaming data types
//! (from `transcribe-cpp-types`).
//!
//! The [`Stream`](crate::Stream) handle itself lives in `session.rs` (it needs
//! the session internals).

use transcribe_cpp_sys as sys;
use transcribe_cpp_types::{StreamOptions, StreamText, StreamUpdate};

use crate::error::Result;
use crate::family::{Materialize, StreamExtRaw};
use crate::result::owned_str;
use crate::types::{FromRaw, ToRaw};

impl FromRaw<&sys::transcribe_stream_update> for StreamUpdate {
    fn from_raw(raw: &sys::transcribe_stream_update) -> Self {
        StreamUpdate {
            result_changed: raw.result_changed,
            is_final: raw.is_final,
            revision: raw.revision,
            input_received_ms: raw.input_received_ms,
            audio_committed_ms: raw.audio_committed_ms,
            buffered_ms: raw.buffered_ms,
            committed_changed: raw.committed_changed,
            tentative_changed: raw.tentative_changed,
        }
    }
}

impl FromRaw<&sys::transcribe_stream_text> for StreamText {
    fn from_raw(raw: &sys::transcribe_stream_text) -> Self {
        StreamText {
            full: owned_str(raw.full_text),
            committed: owned_str(raw.committed_text),
            tentative: owned_str(raw.tentative_text),
        }
    }
}

/// Build `transcribe_stream_params`, returning the family-ext keepalive so its
/// backing struct outlives the `transcribe_stream_begin` call.
pub(crate) fn build_stream_params(
    o: &StreamOptions,
) -> Result<(sys::transcribe_stream_params, Option<StreamExtRaw>)> {
    let mut params: sys::transcribe_stream_params = unsafe { std::mem::zeroed() };
    unsafe { sys::transcribe_stream_params_init(&mut params) };
    params.commit_policy = o.commit_policy.to_raw();
    params.stable_prefix_agreement_n = o.stable_prefix_agreement_n;

    let family = o
        .family
        .as_ref()
        .map(Materialize::materialize)
        .transpose()?;
    params.family = family.as_ref().map_or(std::ptr::null(), |f| f.ext_ptr());
    Ok((params, family))
}
