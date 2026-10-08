//! [`LangIdSession`] — the LANGID role (which language is spoken), from
//! [`Model::langid_session`]. Threading, lifetime, and compute-lock rules are
//! those of [`Session`](crate::Session).

use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use transcribe_cpp_sys as sys;

use crate::cancel::{abort_trampoline, CancelToken};
use crate::error::{check, Error, Result};
use crate::model::{Model, ModelInner};
use crate::result::{owned_str, Timings};
use crate::session::clamp_len;

/// Static facts about a language ID model ([`Model::langid_info`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LangIdInfo {
    /// Input PCM rate (16000).
    pub sample_rate: i32,
    /// Label indices are `[0, n_labels)`.
    pub n_labels: i32,
    /// Shorter scored audio is [`Error::InputTooShort`].
    pub min_audio_ms: i32,
    /// Longer input is scored on its first `max_audio_ms` (30000).
    pub max_audio_ms: i32,
}

/// Options for creating a language ID session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct LangIdSessionOptions {
    /// CPU threads for ops that run on CPU; 0 = library default.
    pub n_threads: i32,
}

/// Per-run language ID parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct LangIdOptions {
    /// Restrict the decision to these codes or aliases. `None` means every
    /// label; `Some` of an empty list is [`Error::InvalidArgument`]. An
    /// unknown code is [`Error::Unsupported`].
    pub allowed: Option<Vec<String>>,
}

/// One ranked label. `code` is the model's own label (`"en"`, `"iw"`).
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct LangIdCandidate {
    /// Label index.
    pub index: i32,
    pub code: String,
    pub name: String,
    /// Softmax renormalized over the allowed set.
    pub p: f32,
    pub logit: f32,
}

/// The result of one [`LangIdSession::run`].
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct LangIdResult {
    /// Every allowed label (duplicates once; all labels when unrestricted),
    /// ranked by `p`, descending; ties keep label order.
    pub candidates: Vec<LangIdCandidate>,
    /// Unrestricted probability inside the allowed set (1.0 when
    /// unrestricted); a low value means the speech is probably outside it.
    pub allowed_mass: f32,
}

impl LangIdResult {
    /// The top candidate's code, if any.
    pub fn code(&self) -> Option<&str> {
        self.candidates.first().map(|c| c.code.as_str())
    }
}

/// A language ID session.
pub struct LangIdSession {
    ptr: *mut sys::transcribe_langid_session,
    // Keeps the native model alive and carries the per-model compute lock.
    model: Arc<ModelInner>,
    // Keeps the abort callback's userdata alive while installed.
    cancel: Option<Arc<AtomicBool>>,
}

impl std::fmt::Debug for LangIdSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangIdSession").finish_non_exhaustive()
    }
}

// SAFETY: as for `Session` — `&mut self` on the mutating calls keeps use to
// one thread at a time; deliberately NOT Sync.
unsafe impl Send for LangIdSession {}

impl Drop for LangIdSession {
    fn drop(&mut self) {
        unsafe { sys::transcribe_langid_session_free(self.ptr) };
    }
}

impl LangIdSession {
    pub(crate) fn new(model: &Model, options: &LangIdSessionOptions) -> Result<LangIdSession> {
        let mut params: sys::transcribe_langid_session_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_langid_session_params_init(&mut params) };
        params.n_threads = options.n_threads;

        let mut out: *mut sys::transcribe_langid_session = std::ptr::null_mut();
        let status =
            unsafe { sys::transcribe_langid_session_init(model.inner.ptr, &params, &mut out) };
        check(status, "langid session init")?;
        debug_assert!(!out.is_null());

        Ok(LangIdSession {
            ptr: out,
            model: Arc::clone(&model.inner),
            cancel: None,
        })
    }

    /// Install a [`CancelToken`] so an in-flight run can be aborted from
    /// another thread (the run then returns [`Error::Aborted`]).
    /// Replaces any previously installed token.
    pub fn set_cancel_token(&mut self, token: &CancelToken) {
        let flag = Arc::clone(&token.flag);
        let userdata = Arc::as_ptr(&flag) as *mut c_void;
        unsafe {
            sys::transcribe_langid_set_abort_callback(self.ptr, Some(abort_trampoline), userdata)
        };
        self.cancel = Some(flag);
    }

    /// Remove any installed cancel token.
    pub fn clear_cancel_token(&mut self) {
        unsafe { sys::transcribe_langid_set_abort_callback(self.ptr, None, std::ptr::null_mut()) };
        self.cancel = None;
    }

    /// Identify the language of one clip of 16 kHz mono float32 PCM. Input
    /// longer than [`LangIdInfo::max_audio_ms`] is scored on its first `max_audio_ms`.
    pub fn run(&mut self, pcm: &[f32], options: &LangIdOptions) -> Result<LangIdResult> {
        let n = clamp_len(pcm.len())?;
        // The CStrings and the pointer array stay alive until the end of this
        // call, past the native run.
        let codes: Option<Vec<CString>> = match &options.allowed {
            None => None,
            Some(list) if list.is_empty() => {
                // NULL would mean "every label", the opposite of an empty list.
                return Err(Error::InvalidArgument(
                    "allowed is empty; pass None for every label".into(),
                ));
            }
            Some(list) => Some(
                list.iter()
                    .map(|c| CString::new(c.as_str()))
                    .collect::<std::result::Result<_, _>>()?,
            ),
        };
        let ptrs: Option<Vec<*const c_char>> = codes
            .as_ref()
            .map(|cs| cs.iter().map(|c| c.as_ptr()).collect());
        let mut params: sys::transcribe_langid_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_langid_params_init(&mut params) };
        if let Some(p) = &ptrs {
            params.allowed = p.as_ptr();
            params.n_allowed = i32::try_from(p.len())
                .map_err(|_| Error::InvalidArgument("allowed list too long".into()))?;
        }

        // Results are copied out under the compute lock, like every other
        // binding, so a concurrent run on the same model cannot interleave.
        let ptr = self.ptr;
        self.model.with_compute(
            Some("a stream is active on this model; finish or drop it before langid run()"),
            |_| -> Result<LangIdResult> {
                check(
                    unsafe { sys::transcribe_langid_run(ptr, pcm.as_ptr(), n, &params) },
                    "langid run",
                )?;
                let mut res: sys::transcribe_langid_result = unsafe { std::mem::zeroed() };
                unsafe { sys::transcribe_langid_result_init(&mut res) };
                check(
                    unsafe { sys::transcribe_langid_get_result(ptr, &mut res) },
                    "langid result",
                )?;
                let candidates = (0..res.n_candidates)
                    .map(|i| {
                        let mut raw: sys::transcribe_langid_candidate =
                            unsafe { std::mem::zeroed() };
                        unsafe { sys::transcribe_langid_candidate_init(&mut raw) };
                        let _ = unsafe { sys::transcribe_langid_get_candidate(ptr, i, &mut raw) };
                        LangIdCandidate {
                            index: raw.index,
                            code: owned_str(raw.code),
                            name: owned_str(raw.name),
                            p: raw.p,
                            logit: raw.logit,
                        }
                    })
                    .collect();
                Ok(LangIdResult {
                    candidates,
                    allowed_mass: res.allowed_mass,
                })
            },
        )?
    }

    /// Model load time plus the last run's stage timings (`decode_ms` is 0).
    pub fn timings(&self) -> Timings {
        let mut raw: sys::transcribe_timings = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_timings_init(&mut raw) };
        let _ = unsafe { sys::transcribe_langid_get_timings(self.ptr, &mut raw) };
        Timings::from_raw(&raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn null_session(streaming: bool) -> LangIdSession {
        // Null native handles (both frees are NULL no-ops): every check below
        // fires before any native call.
        LangIdSession {
            ptr: std::ptr::null_mut(),
            model: Arc::new(ModelInner {
                ptr: std::ptr::null_mut(),
                compute_lock: Mutex::new(streaming),
            }),
            cancel: None,
        }
    }

    #[test]
    fn run_is_busy_while_a_stream_holds_the_lease() {
        let err = null_session(true)
            .run(&[0.0; 160], &LangIdOptions::default())
            .unwrap_err();
        let Error::Busy(msg) = err else {
            panic!("expected Busy, got {err:?}");
        };
        assert_eq!(
            msg,
            "a stream is active on this model; finish or drop it before langid run()"
        );
    }

    #[test]
    fn empty_allowed_is_rejected_not_all() {
        let opts = LangIdOptions {
            allowed: Some(vec![]),
        };
        let err = null_session(false).run(&[0.0; 160], &opts).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn interior_nul_in_allowed_is_rejected() {
        let opts = LangIdOptions {
            allowed: Some(vec!["e\0n".into()]),
        };
        let err = null_session(false).run(&[0.0; 160], &opts).unwrap_err();
        assert!(matches!(err, Error::Nul(_)), "{err:?}");
    }
}
