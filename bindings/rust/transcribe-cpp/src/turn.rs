//! [`TurnSession`] — the TURN role (has the speaker finished talking?), from
//! [`Model::turn_session`]. Threading and lifetime rules are those of
//! [`Session`](crate::Session). Each call takes the compute lock for that call
//! only; `run`, `stream_feed`, `stream_feed_batch` and `stream_flush` return
//! [`Error::Busy`](crate::Error) while an ASR stream is active. A turn stream is
//! session state and does not take the stream lease.

use std::os::raw::c_void;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use transcribe_cpp_sys as sys;

use crate::cancel::{abort_trampoline, CancelToken};
use crate::error::{check, Error, Result};
use crate::model::{Model, ModelInner};
use crate::result::Timings;
use crate::session::clamp_len;

/// Static facts about an end-of-turn model ([`Model::turn_info`]).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TurnInfo {
    /// Input PCM rate (16000).
    pub sample_rate: i32,
    /// The longest audio [`TurnSession::run`] looks at; longer audio is scored
    /// on its last `max_audio_ms`. 0 = no limit.
    pub max_audio_ms: i32,
    /// Samples per frame for streaming; 0 = the model cannot stream (the
    /// `stream_*` calls return [`Error::NotImplemented`](crate::Error)).
    pub frame_samples: i32,
    /// Whether the model also returns a speech probability per frame
    /// ([`TurnResult::speech_probs`]).
    pub has_speech: bool,
    /// The model's recommended cut-off: the default for
    /// [`TurnOptions::threshold`].
    pub threshold: f64,
}

/// Options for creating a turn session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct TurnSessionOptions {
    /// CPU threads for ops that run on CPU; 0 = library default.
    pub n_threads: i32,
}

/// Settings for [`TurnSession::run`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct TurnOptions {
    /// Probability at or above which the turn counts as ended
    /// ([`TurnResult::ended`]). Negative (the default, -1) uses the model's
    /// [`TurnInfo::threshold`]; otherwise `[0, 1]`.
    pub threshold: f64,
}

impl Default for TurnOptions {
    fn default() -> Self {
        TurnOptions { threshold: -1.0 }
    }
}

/// The result of one [`TurnSession::run`], [`TurnSession::stream_feed`],
/// [`TurnSession::stream_feed_batch`] or [`TurnSession::stream_flush`].
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct TurnResult {
    /// After a run: the probability that the turn has ended at the end of the
    /// clip. 0 after a feed or flush.
    pub p: f32,
    /// After a run: `p >= threshold`. False after a feed or flush.
    pub ended: bool,
    /// Per-frame turn probabilities from the call: `probs[i]` says whether the
    /// turn has ended at the end of frame `first_frame + i`. Always empty for
    /// models that cannot stream.
    pub probs: Vec<f32>,
    /// Stream position of `probs[0]`, in frames: 0 after a run; for a feed or
    /// flush, the frames the stream had processed before the call.
    pub first_frame: i64,
    /// Per-frame speech probabilities from the call (empty unless
    /// [`TurnInfo::has_speech`]). During a stream they can trail the turn
    /// probabilities; after a run every frame has both.
    pub speech_probs: Vec<f32>,
    /// Stream position of `speech_probs[0]`, in frames.
    pub speech_first_frame: i64,
}

/// An end-of-turn session.
pub struct TurnSession {
    ptr: *mut sys::transcribe_turn_session,
    // Keeps the native model alive and carries the per-model compute lock.
    model: Arc<ModelInner>,
    // Keeps the abort callback's userdata alive while installed.
    cancel: Option<Arc<AtomicBool>>,
}

impl std::fmt::Debug for TurnSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurnSession").finish_non_exhaustive()
    }
}

// SAFETY: as for `Session` — `&mut self` on the mutating calls keeps use to
// one thread at a time; deliberately NOT Sync.
unsafe impl Send for TurnSession {}

impl Drop for TurnSession {
    fn drop(&mut self) {
        unsafe { sys::transcribe_turn_session_free(self.ptr) };
    }
}

const BUSY_RUN: &str = "a stream is active on this model; finish or drop it before turn run()";
const BUSY_FEED: &str =
    "a stream is active on this model; finish or drop it before turn stream_feed()";
const BUSY_FEED_BATCH: &str =
    "a stream is active on this model; finish or drop it before turn stream_feed_batch()";
const BUSY_FLUSH: &str =
    "a stream is active on this model; finish or drop it before turn stream_flush()";

/// Copy `n` floats out of a native buffer (empty for NULL or `n <= 0`).
fn copy_probs(p: *const f32, n: i32) -> Vec<f32> {
    if p.is_null() || n <= 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(p, n as usize) }.to_vec()
    }
}

/// Copy the last call's result out; called under the compute lock.
fn copy_result(ptr: *mut sys::transcribe_turn_session) -> Result<TurnResult> {
    let mut res: sys::transcribe_turn_result = unsafe { std::mem::zeroed() };
    unsafe { sys::transcribe_turn_result_init(&mut res) };
    check(
        unsafe { sys::transcribe_turn_get_result(ptr, &mut res) },
        "turn result",
    )?;
    Ok(TurnResult {
        p: res.p,
        ended: res.ended,
        probs: copy_probs(unsafe { sys::transcribe_turn_probs(ptr) }, res.n_probs),
        first_frame: res.first_frame,
        speech_probs: copy_probs(
            unsafe { sys::transcribe_turn_speech_probs(ptr) },
            res.n_speech_probs,
        ),
        speech_first_frame: res.speech_first_frame,
    })
}

impl TurnSession {
    pub(crate) fn new(model: &Model, options: &TurnSessionOptions) -> Result<TurnSession> {
        let mut params: sys::transcribe_turn_session_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_turn_session_params_init(&mut params) };
        params.n_threads = options.n_threads;

        let mut out: *mut sys::transcribe_turn_session = std::ptr::null_mut();
        let status =
            unsafe { sys::transcribe_turn_session_init(model.inner.ptr, &params, &mut out) };
        check(status, "turn session init")?;
        debug_assert!(!out.is_null());

        Ok(TurnSession {
            ptr: out,
            model: Arc::clone(&model.inner),
            cancel: None,
        })
    }

    /// Install a [`CancelToken`] so an in-flight run, feed or flush can be aborted
    /// from another thread (the call then returns
    /// [`Error::Aborted`](crate::Error)). Replaces any previously installed token.
    /// A batch feed checks the token of every session in the batch.
    pub fn set_cancel_token(&mut self, token: &CancelToken) {
        let flag = Arc::clone(&token.flag);
        let userdata = Arc::as_ptr(&flag) as *mut c_void;
        unsafe {
            sys::transcribe_turn_set_abort_callback(self.ptr, Some(abort_trampoline), userdata)
        };
        self.cancel = Some(flag);
    }

    /// Remove any installed cancel token.
    pub fn clear_cancel_token(&mut self) {
        unsafe { sys::transcribe_turn_set_abort_callback(self.ptr, None, std::ptr::null_mut()) };
        self.cancel = None;
    }

    /// Score one clip of 16 kHz mono float32 PCM (the audio of the current
    /// turn, from where the speaker started): has the turn ended at its end?
    /// Starts fresh and discards any stream in progress. Audio longer than
    /// [`TurnInfo::max_audio_ms`] is scored on its tail. Models that can stream
    /// also return every frame's turn (and speech) probability.
    /// [`Error::InvalidArgument`](crate::Error) (empty or non-finite input, a
    /// threshold above 1) leaves the stream and last result unchanged.
    pub fn run(&mut self, pcm: &[f32], options: &TurnOptions) -> Result<TurnResult> {
        let n = clamp_len(pcm.len())?;
        let mut params: sys::transcribe_turn_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_turn_params_init(&mut params) };
        params.threshold = options.threshold;

        // Results are copied out under the compute lock, like every other
        // binding, so a concurrent call on the same model cannot interleave.
        let ptr = self.ptr;
        self.model
            .with_compute(Some(BUSY_RUN), |_| -> Result<TurnResult> {
                check(
                    unsafe { sys::transcribe_turn_run(ptr, pcm.as_ptr(), n, &params) },
                    "turn run",
                )?;
                copy_result(ptr)
            })?
    }

    /// Append audio to the stream and return the turn probabilities of the
    /// frames it completed and the speech probabilities of the frames the model
    /// judged; a partial frame waits for the next call. On CPU the stream's
    /// probabilities equal a [`run`](Self::run) over the same audio.
    /// [`Error::NotImplemented`](crate::Error) if the model cannot stream
    /// ([`TurnInfo::frame_samples`] is 0). [`Error::InvalidArgument`](crate::Error)
    /// (e.g. NaN input) leaves the stream intact; an abort or processing error
    /// resets it to frame 0.
    pub fn stream_feed(&mut self, pcm: &[f32]) -> Result<TurnResult> {
        let n = clamp_len(pcm.len())?;
        let ptr = self.ptr;
        self.model
            .with_compute(Some(BUSY_FEED), |_| -> Result<TurnResult> {
                check(
                    unsafe { sys::transcribe_turn_stream_feed(ptr, pcm.as_ptr(), n) },
                    "turn stream feed",
                )?;
                copy_result(ptr)
            })?
    }

    /// Feed several streams of one model in one compute: each `(session, pcm)`
    /// pair ends up exactly as after `session.stream_feed(pcm)`, and the result
    /// at index `i` is that session's. Much faster than one feed per session
    /// when many users share a model. Uses the first session's thread count and
    /// checks every session's cancel token. Takes the model's compute lock once.
    ///
    /// Errors with [`Error::InvalidArgument`](crate::Error) for an empty batch or
    /// sessions of different models (before any native call; every stream
    /// unchanged), and as [`stream_feed`](Self::stream_feed) otherwise; an
    /// abort or processing error resets every listed stream.
    pub fn stream_feed_batch<'s, 'p, I>(batch: I) -> Result<Vec<TurnResult>>
    where
        I: IntoIterator<Item = (&'s mut TurnSession, &'p [f32])>,
    {
        // `&mut` borrows make the sessions distinct by construction.
        let batch: Vec<(&'s mut TurnSession, &'p [f32])> = batch.into_iter().collect();
        let Some((first, _)) = batch.first() else {
            return Err(Error::InvalidArgument(
                "turn stream_feed_batch: empty batch".into(),
            ));
        };
        let model = Arc::clone(&first.model);
        if batch.iter().any(|(s, _)| !Arc::ptr_eq(&s.model, &model)) {
            return Err(Error::InvalidArgument(
                "turn stream_feed_batch: sessions belong to different models".into(),
            ));
        }
        let n = clamp_len(batch.len())?;
        let ptrs: Vec<*mut sys::transcribe_turn_session> =
            batch.iter().map(|(s, _)| s.ptr).collect();
        let pcms: Vec<*const f32> = batch.iter().map(|(_, p)| p.as_ptr()).collect();
        let lens = batch
            .iter()
            .map(|(_, p)| clamp_len(p.len()))
            .collect::<Result<Vec<i32>>>()?;

        // `batch` (and so every PCM slice) and the pointer arrays outlive the
        // native call; results are copied out under the same lock.
        model.with_compute(Some(BUSY_FEED_BATCH), |_| -> Result<Vec<TurnResult>> {
            check(
                unsafe {
                    sys::transcribe_turn_stream_feed_batch(
                        ptrs.as_ptr(),
                        pcms.as_ptr(),
                        lens.as_ptr(),
                        n,
                    )
                },
                "turn stream feed batch",
            )?;
            ptrs.iter().map(|&p| copy_result(p)).collect()
        })?
    }

    /// End the stream: process the buffered partial frame, zero padded (one
    /// turn probability, or none), and return the speech probabilities of
    /// every frame not yet judged. The next feed starts at frame 0; a
    /// processing error also resets the stream.
    pub fn stream_flush(&mut self) -> Result<TurnResult> {
        let ptr = self.ptr;
        self.model
            .with_compute(Some(BUSY_FLUSH), |_| -> Result<TurnResult> {
                check(
                    unsafe { sys::transcribe_turn_stream_flush(ptr) },
                    "turn stream flush",
                )?;
                copy_result(ptr)
            })?
    }

    /// Drop the stream without processing it; the next feed starts at frame 0.
    /// Keeps the last result.
    pub fn stream_reset(&mut self) {
        let ptr = self.ptr;
        let _ = self
            .model
            .with_compute(None, |_| unsafe { sys::transcribe_turn_stream_reset(ptr) });
    }

    /// Model load time plus the last call's stage timings.
    pub fn timings(&self) -> Timings {
        let mut raw: sys::transcribe_timings = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_timings_init(&mut raw) };
        let _ = unsafe { sys::transcribe_turn_get_timings(self.ptr, &mut raw) };
        Timings::from_raw(&raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn null_session(model: &Arc<ModelInner>) -> TurnSession {
        // Null native handles (the frees are NULL no-ops).
        TurnSession {
            ptr: std::ptr::null_mut(),
            model: Arc::clone(model),
            cancel: None,
        }
    }

    #[test]
    fn calls_are_busy_while_a_stream_holds_the_lease() {
        // The Busy check runs under the lock before any native call.
        let model = Arc::new(ModelInner {
            ptr: std::ptr::null_mut(),
            compute_lock: Mutex::new(true),
        });
        let mut session = null_session(&model);
        let mut other = null_session(&model);
        let pcm = [0.0; 160];
        for (r, msg) in [
            (session.run(&pcm, &TurnOptions::default()), BUSY_RUN),
            (session.stream_feed(&pcm), BUSY_FEED),
            (session.stream_flush(), BUSY_FLUSH),
        ] {
            assert!(matches!(r, Err(Error::Busy(ref m)) if m == msg), "{r:?}");
        }
        let r = TurnSession::stream_feed_batch([(&mut session, &pcm[..]), (&mut other, &pcm[..])]);
        assert!(
            matches!(r, Err(Error::Busy(ref m)) if m == BUSY_FEED_BATCH),
            "{r:?}"
        );
    }

    #[test]
    fn batch_rejects_empty_and_mixed_models() {
        let a = Arc::new(ModelInner {
            ptr: std::ptr::null_mut(),
            compute_lock: Mutex::new(false),
        });
        let b = Arc::new(ModelInner {
            ptr: std::ptr::null_mut(),
            compute_lock: Mutex::new(false),
        });
        let r = TurnSession::stream_feed_batch(std::iter::empty());
        assert!(matches!(r, Err(Error::InvalidArgument(_))), "{r:?}");
        let (mut sa, mut sb) = (null_session(&a), null_session(&b));
        let pcm = [0.0; 16];
        let r = TurnSession::stream_feed_batch([(&mut sa, &pcm[..]), (&mut sb, &pcm[..])]);
        assert!(matches!(r, Err(Error::InvalidArgument(_))), "{r:?}");
    }

    #[test]
    fn defaults_match_the_c_init() {
        let mut p: sys::transcribe_turn_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_turn_params_init(&mut p) };
        assert_eq!(TurnOptions::default().threshold, p.threshold);
        let mut s: sys::transcribe_turn_session_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_turn_session_params_init(&mut s) };
        assert_eq!(TurnSessionOptions::default().n_threads, s.n_threads);
    }
}
