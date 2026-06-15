//! Shared fixtures for the integration tests, mirroring the Python conftest.
//!
//! Two tiers: no-model tests always run; model-gated tests resolve a local
//! whisper-tiny.en + jfk.wav and return `None` (clean skip) when absent. The
//! "country" content assertion is specific to jfk.wav and holds for any English
//! ASR model.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Once;

/// Register backend modules before the first model load. Idempotent and a no-op
/// in compiled-in (static / plain `shared`) builds; in a `dynamic-backends`
/// build it loads the per-ISA CPU / GPU modules the native build installed —
/// without it a model load in that posture finds zero devices. Folded into the
/// fixture resolvers below so every model-gated test gets it for free.
fn ensure_backends() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        transcribe_cpp::init_backends_default().expect("init_backends_default");
    });
}

/// Repo root: walk up from this crate until we find the marker that only the
/// repo root carries (immune to how deep the crate dir is nested).
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join("include/transcribe.h").is_file() && p.join("ggml").is_dir())
        .expect("could not locate repo root from CARGO_MANIFEST_DIR")
        .to_path_buf()
}

/// The smoke model path, or `None` when the GGUF is not present.
pub fn smoke_model() -> Option<PathBuf> {
    ensure_backends();
    let path = std::env::var_os("TRANSCRIBE_SMOKE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("models/whisper-tiny.en/whisper-tiny.en-Q5_K_M.gguf"));
    path.is_file().then_some(path)
}

/// The smoke audio as 16 kHz mono f32 PCM, or `None` when the WAV is absent.
pub fn smoke_audio() -> Option<Vec<f32>> {
    let path = std::env::var_os("TRANSCRIBE_SMOKE_AUDIO")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("samples/jfk.wav"));
    path.is_file().then(|| load_wav(&path))
}

/// The streaming smoke model (moonshine-streaming-tiny), or `None` if absent.
pub fn smoke_streaming_model() -> Option<PathBuf> {
    ensure_backends();
    let path = std::env::var_os("TRANSCRIBE_SMOKE_STREAMING_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            repo_root().join("models/moonshine-streaming-tiny/moonshine-streaming-tiny-Q8_0.gguf")
        });
    path.is_file().then_some(path)
}

/// Both fixtures together; prints a skip note and returns `None` if either is
/// missing (so the caller can `return` early — the Rust equivalent of skip).
pub fn smoke_fixtures(test: &str) -> Option<(PathBuf, Vec<f32>)> {
    match (smoke_model(), smoke_audio()) {
        (Some(m), Some(a)) => Some((m, a)),
        _ => {
            eprintln!(
                "skip {test}: smoke model/audio not present (set TRANSCRIBE_SMOKE_MODEL / _AUDIO)"
            );
            None
        }
    }
}

fn load_wav(path: &std::path::Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "{path:?} must be mono");
    assert_eq!(spec.sample_rate, 16_000, "{path:?} must be 16 kHz");
    assert_eq!(spec.bits_per_sample, 16, "{path:?} must be 16-bit");
    reader
        .samples::<i16>()
        .map(|s| s.expect("wav sample") as f32 / 32768.0)
        .collect()
}
