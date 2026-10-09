//! [`VadSession`] — the VAD role (where there is speech), from
//! [`Model::vad_session`], plus the model-independent live policy
//! [`VadIterator`]. Threading, lifetime, and compute-lock rules are those of
//! [`Session`](crate::Session).

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
    /// (the default, -1) means `max(threshold - 0.15, 0.01)`.
    pub neg_threshold: f64,
    /// Segments of at most this length are dropped (default 250).
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
    pub start_sample: i64,
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

const BUSY: &str = "a stream is active on this model; finish or drop it before vad calls";

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

    /// Install a [`CancelToken`] so an in-flight run or feed can be aborted
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
            .with_compute(Some(BUSY), |_| -> Result<VadResult> {
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
    pub fn stream_feed(&mut self, pcm: &[f32]) -> Result<VadResult> {
        let n = clamp_len(pcm.len())?;
        let ptr = self.ptr;
        self.model
            .with_compute(Some(BUSY), |_| -> Result<VadResult> {
                check(
                    unsafe { sys::transcribe_vad_stream_feed(ptr, pcm.as_ptr(), n) },
                    "vad stream feed",
                )?;
                copy_result(ptr)
            })?
    }

    /// End the stream: score the buffered partial frame, zero padded (one
    /// probability, or none), and reset. The next feed starts at frame 0.
    pub fn stream_flush(&mut self) -> Result<VadResult> {
        let ptr = self.ptr;
        self.model
            .with_compute(Some(BUSY), |_| -> Result<VadResult> {
                check(
                    unsafe { sys::transcribe_vad_stream_flush(ptr) },
                    "vad stream flush",
                )?;
                copy_result(ptr)
            })?
    }

    /// Drop the stream without scoring.
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

/// Options for a [`VadIterator`] (Silero `VADIterator`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VadIteratorOptions {
    /// Speech starts at `p >= threshold` (default 0.5). `[0, 1]`.
    pub threshold: f64,
    /// Inside speech, `p < neg_threshold` is silence. Negative (default -1)
    /// means `threshold - 0.15`.
    pub neg_threshold: f64,
    /// Default 100.
    pub min_silence_ms: i32,
    /// Default 30.
    pub speech_pad_ms: i32,
}

impl Default for VadIteratorOptions {
    fn default() -> Self {
        VadIteratorOptions {
            threshold: 0.5,
            neg_threshold: -1.0,
            min_silence_ms: 100,
            speech_pad_ms: 30,
        }
    }
}

/// Whether a [`VadEvent`] opens or closes speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VadEventKind {
    Start,
    End,
}

/// A speech boundary from [`VadIterator::feed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VadEvent {
    pub kind: VadEventKind,
    /// Boundary in 16 kHz samples (padded), not the detection time.
    pub sample: i64,
}

/// The live speech/silence policy over per-frame probabilities (e.g. from
/// [`VadSession::stream_feed`]). Owns no model. End of input does NOT emit an
/// END event.
pub struct VadIterator {
    ptr: *mut sys::transcribe_vad_iterator,
}

impl std::fmt::Debug for VadIterator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VadIterator").finish_non_exhaustive()
    }
}

// SAFETY: plain heap state, no model; `&mut self` on the mutating calls keeps
// use to one thread at a time. Deliberately NOT Sync.
unsafe impl Send for VadIterator {}

impl Drop for VadIterator {
    fn drop(&mut self) {
        unsafe { sys::transcribe_vad_iterator_free(self.ptr) };
    }
}

impl VadIterator {
    /// `frame_samples` comes from [`VadInfo::frame_samples`].
    pub fn new(frame_samples: i32, options: &VadIteratorOptions) -> Result<VadIterator> {
        let mut params: sys::transcribe_vad_iterator_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_vad_iterator_params_init(&mut params) };
        params.threshold = options.threshold;
        params.neg_threshold = options.neg_threshold;
        params.min_silence_ms = options.min_silence_ms;
        params.speech_pad_ms = options.speech_pad_ms;

        let mut out: *mut sys::transcribe_vad_iterator = std::ptr::null_mut();
        check(
            unsafe { sys::transcribe_vad_iterator_init(frame_samples, &params, &mut out) },
            "vad iterator init",
        )?;
        debug_assert!(!out.is_null());
        Ok(VadIterator { ptr: out })
    }

    /// Consume sequential probabilities (each in `[0, 1]`) and return the
    /// events they produced, in detection order. A failure leaves the state
    /// intact.
    pub fn feed(&mut self, probs: &[f32]) -> Result<Vec<VadEvent>> {
        let n = clamp_len(probs.len())?;
        check(
            unsafe { sys::transcribe_vad_iterator_feed(self.ptr, probs.as_ptr(), n) },
            "vad iterator feed",
        )?;
        let n_events = self.raw_result().n_events;
        Ok((0..n_events)
            .map(|i| {
                let mut raw: sys::transcribe_vad_event = unsafe { std::mem::zeroed() };
                unsafe { sys::transcribe_vad_event_init(&mut raw) };
                let _ = unsafe { sys::transcribe_vad_iterator_get_event(self.ptr, i, &mut raw) };
                let kind = match raw.type_ {
                    sys::transcribe_vad_event_type::TRANSCRIBE_VAD_EVENT_START => {
                        VadEventKind::Start
                    }
                    // The library emits only START and END.
                    _ => VadEventKind::End,
                };
                VadEvent {
                    kind,
                    sample: raw.sample,
                }
            })
            .collect())
    }

    /// Reset time and hysteresis; active speech is dropped without an END.
    pub fn reset(&mut self) {
        unsafe { sys::transcribe_vad_iterator_reset(self.ptr) };
    }

    /// Probabilities consumed so far times `frame_samples`.
    pub fn current_sample(&self) -> i64 {
        self.raw_result().current_sample
    }

    /// Whether speech is active (possibly awaiting silence).
    pub fn triggered(&self) -> bool {
        self.raw_result().triggered
    }

    fn raw_result(&self) -> sys::transcribe_vad_iterator_result {
        let mut raw: sys::transcribe_vad_iterator_result = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_vad_iterator_result_init(&mut raw) };
        let _ = unsafe { sys::transcribe_vad_iterator_get_result(self.ptr, &mut raw) };
        raw
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
        let err = session
            .run(&[0.0; 160], &VadOptions::default())
            .unwrap_err();
        assert!(matches!(err, Error::Busy(ref m) if m == BUSY), "{err:?}");
        assert!(matches!(
            session.stream_feed(&[0.0; 160]),
            Err(Error::Busy(_))
        ));
        assert!(matches!(session.stream_flush(), Err(Error::Busy(_))));
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
        let mut p: sys::transcribe_vad_iterator_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_vad_iterator_params_init(&mut p) };
        let d = VadIteratorOptions::default();
        assert_eq!(
            (
                d.threshold,
                d.neg_threshold,
                d.min_silence_ms,
                d.speech_pad_ms
            ),
            (
                p.threshold,
                p.neg_threshold,
                p.min_silence_ms,
                p.speech_pad_ms
            )
        );
    }
}
