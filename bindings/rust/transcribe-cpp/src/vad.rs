//! [`VadSession`] — the VAD role (where there is speech), from
//! [`Model::vad_session`]. Threading and lifetime rules are those of
//! [`Session`](crate::Session). Each call takes the compute lock for that call
//! only; `run`, `stream_feed` and `stream_flush` return
//! [`Error::Busy`](crate::Error) while an ASR stream is active. A VAD stream is
//! session state and does not take the stream lease.

use std::os::raw::c_void;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use transcribe_cpp_sys as sys;

use crate::cancel::{abort_trampoline, CancelToken};
use crate::error::{check, Result};
use crate::model::{Model, ModelInner};
use crate::result::Timings;
use crate::session::clamp_len;

/// Static facts about a VAD model ([`Model::vad_info`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VadInfo {
    /// Input PCM rate (16000).
    pub sample_rate: i32,
    /// Samples per probability (512 = 32 ms).
    pub frame_samples: i32,
}

/// Options for creating a VAD session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VadSessionOptions {
    /// CPU threads for ops that run on CPU; 0 = library default.
    pub n_threads: i32,
}

/// Probabilities -> speech segments ([`VadSession::run`] only). Fields and
/// defaults are those of Silero's `get_speech_timestamps`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VadOptions {
    /// A frame with `p >= threshold` is speech (default 0.5). `[0, 1]`.
    pub threshold: f64,
    /// Inside speech, a frame with `p < neg_threshold` is silence. Negative
    /// (the default, -1) means `max(threshold - 0.15, 0.01)`; otherwise
    /// `[0, threshold]`.
    pub neg_threshold: f64,
    /// Segments of at most this length are dropped (default 250), except
    /// those split by `max_speech_ms`. Durations must be nonnegative.
    pub min_speech_ms: i32,
    /// Silence this long ends a segment (default 100).
    pub min_silence_ms: i32,
    /// Padding on both sides of each segment (default 30).
    pub speech_pad_ms: i32,
    /// Segments longer than this are split; 0 (the default) = no limit.
    pub max_speech_ms: i32,
}

impl Default for VadOptions {
    fn default() -> Self {
        VadOptions {
            threshold: 0.5,
            neg_threshold: -1.0,
            min_speech_ms: 250,
            min_silence_ms: 100,
            speech_pad_ms: 30,
            max_speech_ms: 0,
        }
    }
}

/// One speech segment, in samples of the run's input: `[start, end)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VadSegment {
    /// First sample of speech.
    pub start_sample: i64,
    /// First sample after speech.
    pub end_sample: i64,
}

/// The result of one [`VadSession::run`], [`VadSession::stream_feed`] or
/// [`VadSession::stream_flush`].
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VadResult {
    /// Speech probability per frame produced by the call.
    pub probs: Vec<f32>,
    /// Stream position of `probs[0]`, in frames: 0 after a run; for a feed
    /// or flush, the frames the stream had scored before the call.
    pub first_frame: i64,
    /// Speech segments, in time order; run only (empty after a feed or flush).
    pub segments: Vec<VadSegment>,
}

/// A VAD session.
pub struct VadSession {
    ptr: *mut sys::transcribe_vad_session,
    // Keeps the native model alive and carries the per-model compute lock.
    model: Arc<ModelInner>,
    // Keeps the abort callback's userdata alive while installed.
    cancel: Option<Arc<AtomicBool>>,
}

impl std::fmt::Debug for VadSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VadSession").finish_non_exhaustive()
    }
}

// SAFETY: as for `Session` — `&mut self` on the mutating calls keeps use to
// one thread at a time; deliberately NOT Sync.
unsafe impl Send for VadSession {}

impl Drop for VadSession {
    fn drop(&mut self) {
        unsafe { sys::transcribe_vad_session_free(self.ptr) };
    }
}

const BUSY_RUN: &str = "a stream is active on this model; finish or drop it before vad run()";
const BUSY_FEED: &str =
    "a stream is active on this model; finish or drop it before vad stream_feed()";
const BUSY_FLUSH: &str =
    "a stream is active on this model; finish or drop it before vad stream_flush()";

/// Copy the last call's result out; called under the compute lock.
fn copy_result(ptr: *mut sys::transcribe_vad_session) -> Result<VadResult> {
    let mut res: sys::transcribe_vad_result = unsafe { std::mem::zeroed() };
    unsafe { sys::transcribe_vad_result_init(&mut res) };
    check(
        unsafe { sys::transcribe_vad_get_result(ptr, &mut res) },
        "vad result",
    )?;
    let p = unsafe { sys::transcribe_vad_probs(ptr) };
    let probs = if p.is_null() || res.n_probs <= 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(p, res.n_probs as usize) }.to_vec()
    };
    let segments = (0..res.n_segments)
        .map(|i| {
            let mut raw: sys::transcribe_vad_segment = unsafe { std::mem::zeroed() };
            unsafe { sys::transcribe_vad_segment_init(&mut raw) };
            let _ = unsafe { sys::transcribe_vad_get_segment(ptr, i, &mut raw) };
            VadSegment {
                start_sample: raw.start_sample,
                end_sample: raw.end_sample,
            }
        })
        .collect();
    Ok(VadResult {
        probs,
        first_frame: res.first_frame,
        segments,
    })
}

impl VadSession {
    pub(crate) fn new(model: &Model, options: &VadSessionOptions) -> Result<VadSession> {
        let mut params: sys::transcribe_vad_session_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_vad_session_params_init(&mut params) };
        params.n_threads = options.n_threads;

        let mut out: *mut sys::transcribe_vad_session = std::ptr::null_mut();
        let status =
            unsafe { sys::transcribe_vad_session_init(model.inner.ptr, &params, &mut out) };
        check(status, "vad session init")?;
        debug_assert!(!out.is_null());

        Ok(VadSession {
            ptr: out,
            model: Arc::clone(&model.inner),
            cancel: None,
        })
    }

    /// Install a [`CancelToken`] so an in-flight run, feed or flush can be aborted
    /// from another thread (the call then returns
    /// [`Error::Aborted`](crate::Error)). Replaces any previously installed token.
    pub fn set_cancel_token(&mut self, token: &CancelToken) {
        let flag = Arc::clone(&token.flag);
        let userdata = Arc::as_ptr(&flag) as *mut c_void;
        unsafe {
            sys::transcribe_vad_set_abort_callback(self.ptr, Some(abort_trampoline), userdata)
        };
        self.cancel = Some(flag);
    }

    /// Remove any installed cancel token.
    pub fn clear_cancel_token(&mut self) {
        unsafe { sys::transcribe_vad_set_abort_callback(self.ptr, None, std::ptr::null_mut()) };
        self.cancel = None;
    }

    /// Score one clip of 16 kHz mono float32 PCM from a fresh state and
    /// segment it. Resets any stream in progress.
    pub fn run(&mut self, pcm: &[f32], options: &VadOptions) -> Result<VadResult> {
        let n = clamp_len(pcm.len())?;
        let mut params: sys::transcribe_vad_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_vad_params_init(&mut params) };
        params.threshold = options.threshold;
        params.neg_threshold = options.neg_threshold;
        params.min_speech_ms = options.min_speech_ms;
        params.min_silence_ms = options.min_silence_ms;
        params.speech_pad_ms = options.speech_pad_ms;
        params.max_speech_ms = options.max_speech_ms;

        // Results are copied out under the compute lock, like every other
        // binding, so a concurrent call on the same model cannot interleave.
        let ptr = self.ptr;
        self.model
            .with_compute(Some(BUSY_RUN), |_| -> Result<VadResult> {
                check(
                    unsafe { sys::transcribe_vad_run(ptr, pcm.as_ptr(), n, &params) },
                    "vad run",
                )?;
                copy_result(ptr)
            })?
    }

    /// Append audio to the stream and return the probabilities of the frames
    /// it completed; a partial frame waits for the next call. On CPU the
    /// stream's probabilities equal a [`run`](Self::run) over the same audio.
    /// [`Error::InvalidArgument`](crate::Error) (e.g. NaN input) leaves the
    /// stream intact; an abort or processing error resets it to frame 0.
    pub fn stream_feed(&mut self, pcm: &[f32]) -> Result<VadResult> {
        let n = clamp_len(pcm.len())?;
        let ptr = self.ptr;
        self.model
            .with_compute(Some(BUSY_FEED), |_| -> Result<VadResult> {
                check(
                    unsafe { sys::transcribe_vad_stream_feed(ptr, pcm.as_ptr(), n) },
                    "vad stream feed",
                )?;
                copy_result(ptr)
            })?
    }

    /// End the stream: score the buffered partial frame, zero padded (one
    /// probability, or none), and reset. The next feed starts at frame 0; a
    /// processing error also resets the stream.
    pub fn stream_flush(&mut self) -> Result<VadResult> {
        let ptr = self.ptr;
        self.model
            .with_compute(Some(BUSY_FLUSH), |_| -> Result<VadResult> {
                check(
                    unsafe { sys::transcribe_vad_stream_flush(ptr) },
                    "vad stream flush",
                )?;
                copy_result(ptr)
            })?
    }

    /// Drop the stream without scoring; the next feed starts at frame 0.
    pub fn stream_reset(&mut self) {
        let ptr = self.ptr;
        let _ = self
            .model
            .with_compute(None, |_| unsafe { sys::transcribe_vad_stream_reset(ptr) });
    }

    /// Model load time plus the last call's stage timings (`encode_ms`: front
    /// end and encoder; `decode_ms`: the recurrent decoder).
    pub fn timings(&self) -> Timings {
        let mut raw: sys::transcribe_timings = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_timings_init(&mut raw) };
        let _ = unsafe { sys::transcribe_vad_get_timings(self.ptr, &mut raw) };
        Timings::from_raw(&raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use std::sync::Mutex;

    #[test]
    fn calls_are_busy_while_a_stream_holds_the_lease() {
        // Null native handles (both frees are NULL no-ops): the Busy check runs
        // under the lock before any native call.
        let mut session = VadSession {
            ptr: std::ptr::null_mut(),
            model: Arc::new(ModelInner {
                ptr: std::ptr::null_mut(),
                compute_lock: Mutex::new(true),
            }),
            cancel: None,
        };
        for (r, msg) in [
            (session.run(&[0.0; 160], &VadOptions::default()), BUSY_RUN),
            (session.stream_feed(&[0.0; 160]), BUSY_FEED),
            (session.stream_flush(), BUSY_FLUSH),
        ] {
            assert!(matches!(r, Err(Error::Busy(ref m)) if m == msg), "{r:?}");
        }
    }

    #[test]
    fn defaults_match_the_c_init() {
        let mut p: sys::transcribe_vad_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_vad_params_init(&mut p) };
        let d = VadOptions::default();
        assert_eq!(
            (
                d.threshold,
                d.neg_threshold,
                d.min_speech_ms,
                d.min_silence_ms,
                d.speech_pad_ms,
                d.max_speech_ms
            ),
            (
                p.threshold,
                p.neg_threshold,
                p.min_speech_ms,
                p.min_silence_ms,
                p.speech_pad_ms,
                p.max_speech_ms
            )
        );
    }
}
