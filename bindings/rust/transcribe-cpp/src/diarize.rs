//! [`DiarizeSession`] — the DIARIZE role (who spoke when), from [`Model::diarize_session`].
//! Threading, lifetime, and compute-lock rules are those of [`Session`](crate::Session).

use std::os::raw::c_void;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use transcribe_cpp_sys as sys;

use crate::cancel::{abort_trampoline, CancelToken};
use crate::error::{check, Result};
use crate::family::DiarizeExtension;
use crate::model::{Model, ModelInner};
use crate::result::{SpeakerSegment, Timings};
use crate::session::clamp_len;

/// Static facts about a diarization model ([`Model::diarize_info`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiarizeInfo {
    /// Input PCM rate (16000).
    pub sample_rate: i32,
    /// `speaker_id` is in `[1, max_speakers]`.
    pub max_speakers: i32,
}

/// Options for creating a diarization session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct DiarizeSessionOptions {
    /// CPU threads for ops that run on CPU; 0 = library default.
    pub n_threads: i32,
}

/// Per-run diarization parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct DiarizeOptions {
    /// Optional family-specific extension (e.g. the Sortformer preset).
    pub family: Option<DiarizeExtension>,
}

/// A diarization session.
pub struct DiarizeSession {
    ptr: *mut sys::transcribe_diarize_session,
    // Keeps the native model alive and carries the per-model compute lock.
    model: Arc<ModelInner>,
    // Keeps the abort callback's userdata alive while installed.
    cancel: Option<Arc<AtomicBool>>,
}

impl std::fmt::Debug for DiarizeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiarizeSession").finish_non_exhaustive()
    }
}

// SAFETY: as for `Session` — `&mut self` on the mutating calls keeps use to
// one thread at a time; deliberately NOT Sync.
unsafe impl Send for DiarizeSession {}

impl Drop for DiarizeSession {
    fn drop(&mut self) {
        unsafe { sys::transcribe_diarize_session_free(self.ptr) };
    }
}

impl DiarizeSession {
    pub(crate) fn new(model: &Model, options: &DiarizeSessionOptions) -> Result<DiarizeSession> {
        let mut params: sys::transcribe_diarize_session_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_diarize_session_params_init(&mut params) };
        params.n_threads = options.n_threads;

        let mut out: *mut sys::transcribe_diarize_session = std::ptr::null_mut();
        let status =
            unsafe { sys::transcribe_diarize_session_init(model.inner.ptr, &params, &mut out) };
        check(status, "diarize session init")?;
        debug_assert!(!out.is_null());

        Ok(DiarizeSession {
            ptr: out,
            model: Arc::clone(&model.inner),
            cancel: None,
        })
    }

    /// Install a [`CancelToken`] so an in-flight run can be aborted from
    /// another thread (the run then returns [`Error::Aborted`](crate::Error)).
    /// Replaces any previously installed token.
    pub fn set_cancel_token(&mut self, token: &CancelToken) {
        let flag = Arc::clone(&token.flag);
        let userdata = Arc::as_ptr(&flag) as *mut c_void;
        unsafe {
            sys::transcribe_diarize_set_abort_callback(self.ptr, Some(abort_trampoline), userdata)
        };
        self.cancel = Some(flag);
    }

    /// Remove any installed cancel token.
    pub fn clear_cancel_token(&mut self) {
        unsafe { sys::transcribe_diarize_set_abort_callback(self.ptr, None, std::ptr::null_mut()) };
        self.cancel = None;
    }

    /// Diarize one recording of 16 kHz mono float32 PCM. Returns the speaker
    /// segments, grouped by speaker and time-ordered within a speaker;
    /// segments of different speakers may overlap.
    pub fn run(&mut self, pcm: &[f32], options: &DiarizeOptions) -> Result<Vec<SpeakerSegment>> {
        let n = clamp_len(pcm.len())?;
        let family = options.family.as_ref().map(DiarizeExtension::materialize);
        let mut params: sys::transcribe_diarize_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_diarize_params_init(&mut params) };
        params.family = family.as_ref().map_or(std::ptr::null(), |f| f.ext_ptr());

        let status = self.model.with_compute(
            Some("a stream is active on this model; finish or drop it before diarize run()"),
            |_| unsafe { sys::transcribe_diarize_run(self.ptr, pcm.as_ptr(), n, &params) },
        )?;
        check(status, "diarize run")?;

        let count = unsafe { sys::transcribe_diarize_n_segments(self.ptr) };
        Ok((0..count)
            .map(|i| {
                let mut raw: sys::transcribe_speaker_segment = unsafe { std::mem::zeroed() };
                unsafe { sys::transcribe_speaker_segment_init(&mut raw) };
                let _ = unsafe { sys::transcribe_diarize_get_segment(self.ptr, i, &mut raw) };
                SpeakerSegment::from_raw(&raw)
            })
            .collect())
    }

    /// Model load time plus the last run's stage timings (`decode_ms` is 0).
    pub fn timings(&self) -> Timings {
        let mut raw: sys::transcribe_timings = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_timings_init(&mut raw) };
        let _ = unsafe { sys::transcribe_diarize_get_timings(self.ptr, &mut raw) };
        Timings::from_raw(&raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use std::sync::Mutex;

    #[test]
    fn run_is_busy_while_a_stream_holds_the_lease() {
        // Null native handles (both frees are NULL no-ops): the Busy check runs
        // under the lock before any native call.
        let mut session = DiarizeSession {
            ptr: std::ptr::null_mut(),
            model: Arc::new(ModelInner {
                ptr: std::ptr::null_mut(),
                compute_lock: Mutex::new(true),
            }),
            cancel: None,
        };
        let err = session
            .run(&[0.0; 160], &DiarizeOptions::default())
            .unwrap_err();
        let Error::Busy(msg) = err else {
            panic!("expected Busy, got {err:?}");
        };
        assert_eq!(
            msg,
            "a stream is active on this model; finish or drop it before diarize run()"
        );
    }
}
