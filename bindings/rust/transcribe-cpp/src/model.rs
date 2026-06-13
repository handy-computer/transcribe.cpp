//! [`Model`] — a loaded GGUF, shareable across threads.
//!
//! A `Model` is `Arc`-backed and `Send + Sync`: cloning it is cheap and hands
//! out another handle to the same native model. The native model is freed when
//! the last handle AND every [`Session`](crate::Session) derived from it have
//! been dropped — Rust ownership gives the C "model must outlive its sessions"
//! contract (and close-ordering safety) for free, in any drop order.
//!
//! The `Arc` also carries the per-model run mutex that serializes the compute
//! path: the C library allows at most one in-flight `run`/stream across all
//! sessions of a model (the 0.x concurrency limitation), so the safe wrapper
//! makes concurrent calls queue rather than race.

use std::ffi::CString;
use std::path::Path;
use std::sync::{Arc, Mutex};

use transcribe_cpp_sys as sys;

use crate::error::{check, Result};
use crate::result::owned_str;
use crate::session::Session;
use crate::types::{Backend, ExtSlot, Feature, TimestampKind};
use crate::version;

/// Options for loading a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelOptions {
    /// Which backend to request. Default [`Backend::Auto`].
    pub backend: Backend,
    /// Reserved for multi-device selection; must be 0 in 0.x.
    pub gpu_device: i32,
}

impl Default for ModelOptions {
    fn default() -> Self {
        ModelOptions {
            backend: Backend::Auto,
            gpu_device: 0,
        }
    }
}

/// Immutable, model-level capabilities read from GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Capabilities {
    pub native_sample_rate: i32,
    /// Supported language codes (empty if the model is language-agnostic).
    pub languages: Vec<String>,
    /// The finest timestamp granularity the model can produce.
    pub max_timestamp_kind: TimestampKind,
    pub supports_language_detect: bool,
    pub supports_translate: bool,
    pub supports_streaming: bool,
    pub supports_spec_decode: bool,
    /// Longest accepted audio in ms (0 = no practical limit).
    pub max_audio_ms: i64,
}

/// The native model handle plus the per-model run lock, shared via `Arc`.
pub(crate) struct ModelInner {
    pub(crate) ptr: *mut sys::transcribe_model,
    /// Serializes the compute path across all sessions of this model.
    pub(crate) run_lock: Mutex<()>,
}

// SAFETY: the native model is documented thread-safe for query + session
// creation; the compute path is serialized by `run_lock`. The raw pointer is
// only ever used behind `&ModelInner`, never mutated through aliasing.
unsafe impl Send for ModelInner {}
unsafe impl Sync for ModelInner {}

impl Drop for ModelInner {
    fn drop(&mut self) {
        // Only reached once every Session (each holding an Arc clone) is gone.
        unsafe { sys::transcribe_model_free(self.ptr) };
    }
}

/// A loaded transcription model.
#[derive(Clone)]
pub struct Model {
    pub(crate) inner: Arc<ModelInner>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("arch", &self.arch())
            .field("variant", &self.variant())
            .field("backend", &self.backend())
            .finish()
    }
}

impl Model {
    /// Load a GGUF model from disk with default options.
    pub fn load(path: impl AsRef<Path>) -> Result<Model> {
        Model::load_with(path, &ModelOptions::default())
    }

    /// Load a GGUF model from disk with explicit options.
    pub fn load_with(path: impl AsRef<Path>, options: &ModelOptions) -> Result<Model> {
        // Pre-1.0 base-version lock against the loaded library (once).
        version::ensure_compatible()?;

        let path = path.as_ref();
        let c_path = CString::new(path_bytes(path)?)?;

        let mut params: sys::transcribe_model_load_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_model_load_params_init(&mut params) };
        params.backend = options.backend.to_raw();
        params.gpu_device = options.gpu_device;

        let mut out: *mut sys::transcribe_model = std::ptr::null_mut();
        let status = unsafe { sys::transcribe_model_load_file(c_path.as_ptr(), &params, &mut out) };
        check(status, &format!("load {}", path.display()))?;
        debug_assert!(!out.is_null());

        Ok(Model {
            inner: Arc::new(ModelInner {
                ptr: out,
                run_lock: Mutex::new(()),
            }),
        })
    }

    /// Open a session bound to this model with default session options.
    pub fn session(&self) -> Result<Session> {
        self.session_with(&SessionOptions::default())
    }

    /// Open a session bound to this model with explicit options.
    pub fn session_with(&self, options: &SessionOptions) -> Result<Session> {
        Session::new(self, options)
    }

    /// The model's immutable capabilities.
    pub fn capabilities(&self) -> Capabilities {
        let mut caps: sys::transcribe_capabilities = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_capabilities_init(&mut caps) };
        // A NULL/struct-size fault cannot happen here (we own a valid model
        // and an init'd struct), so a non-OK status leaves zeroed defaults.
        let _ = unsafe { sys::transcribe_model_get_capabilities(self.inner.ptr, &mut caps) };

        let mut languages = Vec::new();
        if !caps.languages.is_null() && caps.n_languages > 0 {
            let slice =
                unsafe { std::slice::from_raw_parts(caps.languages, caps.n_languages as usize) };
            for &lang in slice {
                languages.push(owned_str(lang));
            }
        }

        Capabilities {
            native_sample_rate: caps.native_sample_rate,
            languages,
            max_timestamp_kind: TimestampKind::from_raw(caps.max_timestamp_kind),
            supports_language_detect: caps.supports_language_detect,
            supports_translate: caps.supports_translate,
            supports_streaming: caps.supports_streaming,
            supports_spec_decode: caps.supports_spec_decode,
            max_audio_ms: caps.max_audio_ms,
        }
    }

    /// Probe a yes/no feature.
    pub fn supports(&self, feature: Feature) -> bool {
        unsafe { sys::transcribe_model_supports(self.inner.ptr, feature.to_raw()) }
    }

    /// Whether this model accepts the given family-extension `kind` in `slot`.
    pub fn accepts_ext(&self, slot: ExtSlot, kind: u32) -> bool {
        unsafe { sys::transcribe_model_accepts_ext_kind(self.inner.ptr, slot.to_raw(), kind) }
    }

    /// The `general.architecture` string, e.g. "parakeet".
    pub fn arch(&self) -> String {
        owned_str(unsafe { sys::transcribe_model_arch_string(self.inner.ptr) })
    }

    /// The `stt.variant` string, e.g. "tdt-0.6b-v2" (may be empty).
    pub fn variant(&self) -> String {
        owned_str(unsafe { sys::transcribe_model_variant_string(self.inner.ptr) })
    }

    /// The runtime backend bound to this model, e.g. "cpu", "metal" — the way
    /// to detect CPU fallback after requesting a GPU.
    pub fn backend(&self) -> String {
        owned_str(unsafe { sys::transcribe_model_backend(self.inner.ptr) })
    }

    /// Tokenize plain UTF-8 text into the model's vocabulary (no BOS/EOS, no
    /// special tags). Errors with [`Error::NotImplemented`](crate::Error) for
    /// families whose tokenizer has no encode path (e.g. SentencePiece today).
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        let c_text = CString::new(text)?;
        // Grow-and-retry on the negative-N "buffer too small" signal.
        let mut buf = vec![0i32; text.len().max(16)];
        loop {
            let n = unsafe {
                sys::transcribe_tokenize(
                    self.inner.ptr,
                    c_text.as_ptr(),
                    buf.as_mut_ptr(),
                    buf.len(),
                )
            };
            if n == i32::MIN {
                return Err(crate::error::Error::NotImplemented(
                    "model tokenizer has no encode path".to_string(),
                ));
            }
            if n < 0 {
                buf.resize((-n) as usize, 0);
                continue;
            }
            buf.truncate(n as usize);
            return Ok(buf);
        }
    }
}

/// Options for creating a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionOptions {
    /// CPU threads for ops that run on CPU; 0 = library default.
    pub n_threads: i32,
    /// K/V activation precision.
    pub kv_type: crate::types::KvType,
    /// Optional decoder context cap in tokens; 0 = model maximum.
    pub n_ctx: i32,
}

impl Default for SessionOptions {
    fn default() -> Self {
        SessionOptions {
            n_threads: 0,
            kv_type: crate::types::KvType::Auto,
            n_ctx: 0,
        }
    }
}

/// Per-session effective limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionLimits {
    /// Decoder context cap in force for this session (0 = unbounded family).
    pub effective_n_ctx: i32,
    /// Longest accepted audio in ms (0 = no practical limit).
    pub effective_max_audio_ms: i64,
    /// Worst-case decoder KV bytes for one utterance (0 = no decoder KV).
    pub max_kv_bytes: i64,
}

#[cfg(unix)]
fn path_bytes(path: &Path) -> Result<Vec<u8>> {
    use std::os::unix::ffi::OsStrExt;
    Ok(path.as_os_str().as_bytes().to_vec())
}

#[cfg(not(unix))]
fn path_bytes(path: &Path) -> Result<Vec<u8>> {
    // The C API takes a UTF-8 `const char *`; require a UTF-8-representable path.
    path.to_str().map(|s| s.as_bytes().to_vec()).ok_or_else(|| {
        crate::error::Error::InvalidArgument(format!("non-UTF-8 model path: {}", path.display()))
    })
}
