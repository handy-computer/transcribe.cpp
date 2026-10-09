//! VAD role: contract checks on the toy silero_vad fixture
//! (tests/fixtures/arch_silero_vad.gguf, built by the C++ test fixtures) and
//! the model-free VadIterator. Busy and the option defaults are covered in
//! `src/vad.rs`.

mod common;

use transcribe_cpp::{
    Backend, CancelToken, Error, Model, ModelOptions, Role, VadEventKind, VadIterator,
    VadIteratorOptions, VadOptions,
};

/// Bursts of a chirp over low noise (the C++ silero_vad_smoke signal).
fn signal(n: usize) -> Vec<f32> {
    let mut s: u32 = 12345;
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = ((s >> 8) as f32 / 16_777_216.0 - 0.5) * 0.01;
            let t = i as f32 / 16000.0;
            let burst = (i / 8000) % 3 != 0;
            noise
                + if burst {
                    0.3 * (std::f32::consts::TAU * (200.0 + 400.0 * t) * t).sin()
                } else {
                    0.0
                }
        })
        .collect()
}

fn load_cpu(path: &std::path::Path) -> Model {
    let opts = ModelOptions {
        backend: Backend::Cpu,
        ..Default::default()
    };
    Model::load_with(path, &opts).unwrap()
}

#[test]
fn toy_roles_info() {
    let Some(path) = common::vad_toy_model("toy_roles_info") else {
        return;
    };
    let model = load_cpu(&path);
    let roles = model.roles();
    assert!(roles.contains(Role::Vad) && !roles.contains(Role::Asr));
    let info = model.vad_info().unwrap();
    assert_eq!((info.sample_rate, info.frame_samples), (16000, 512));
    assert!(matches!(model.session(), Err(Error::UnsupportedRole(_))));
    assert!(matches!(
        model.langid_info(),
        Err(Error::UnsupportedRole(_))
    ));
}

#[test]
fn toy_run_and_stream_match() {
    let Some(path) = common::vad_toy_model("toy_run_and_stream_match") else {
        return;
    };
    let model = load_cpu(&path);
    let mut vad = model.vad_session().unwrap();
    // 2.5 s, not a whole number of frames.
    let pcm = signal(40000 + 123);

    let off = vad.run(&pcm, &VadOptions::default()).unwrap();
    assert_eq!(off.probs.len(), pcm.len().div_ceil(512));
    assert_eq!(off.first_frame, 0);
    assert!(off.probs.iter().all(|p| (0.0..=1.0).contains(p)));
    assert!(off
        .segments
        .iter()
        .all(|s| 0 <= s.start_sample && s.start_sample < s.end_sample));
    assert!(vad.timings().encode_ms > 0.0);
    assert!(matches!(
        vad.run(
            &pcm,
            &VadOptions {
                threshold: 2.0,
                ..Default::default()
            }
        ),
        Err(Error::InvalidArgument(_))
    ));

    // Streaming in ragged chunks equals the offline run, bit for bit.
    let mut probs = Vec::new();
    for chunk in pcm.chunks(777) {
        let r = vad.stream_feed(chunk).unwrap();
        assert_eq!(r.first_frame, probs.len() as i64);
        assert!(r.segments.is_empty());
        probs.extend_from_slice(&r.probs);
    }
    let r = vad.stream_flush().unwrap();
    assert_eq!(r.first_frame, probs.len() as i64);
    probs.extend_from_slice(&r.probs);
    assert_eq!(probs, off.probs);

    // Reset drops buffered audio; the next stream starts at frame 0.
    vad.stream_feed(&pcm[..1000]).unwrap();
    vad.stream_reset();
    assert_eq!(vad.stream_feed(&pcm[..512]).unwrap().first_frame, 0);
}

#[test]
fn toy_cancel_aborts() {
    let Some(path) = common::vad_toy_model("toy_cancel_aborts") else {
        return;
    };
    let model = load_cpu(&path);
    let mut vad = model.vad_session().unwrap();
    let token = CancelToken::new();
    vad.set_cancel_token(&token);
    token.cancel();
    assert!(matches!(
        vad.run(&signal(16000), &VadOptions::default()),
        Err(Error::Aborted { .. })
    ));
    vad.clear_cancel_token();
    assert!(vad.run(&signal(16000), &VadOptions::default()).is_ok());
}

#[test]
fn iterator_basics() {
    // 512-sample frames, threshold 0.5, 100 ms silence (1600 samples), 30 ms pad.
    let mut it = VadIterator::new(512, &VadIteratorOptions::default()).unwrap();
    let events = it.feed(&[0.9]).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!((events[0].kind, events[0].sample), (VadEventKind::Start, 0));
    assert!(it.triggered());
    // Silence from sample 1024 reaches 1600 samples at 3072; END = 1024 + 480 - 512.
    let events = it.feed(&[0.0; 5]).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!((events[0].kind, events[0].sample), (VadEventKind::End, 992));
    assert!(!it.triggered());
    assert_eq!(it.current_sample(), 6 * 512);

    // A bad probability fails and leaves the state intact.
    assert!(matches!(it.feed(&[1.5]), Err(Error::InvalidArgument(_))));
    assert_eq!(it.current_sample(), 6 * 512);

    it.reset();
    assert_eq!(it.current_sample(), 0);
    assert!(it.feed(&[]).unwrap().is_empty());
    assert!(matches!(
        VadIterator::new(0, &VadIteratorOptions::default()),
        Err(Error::InvalidArgument(_))
    ));
    // neg_threshold is passed through: 0.3 is silence by default, speech at 0.2.
    let opts = VadIteratorOptions {
        neg_threshold: 0.2,
        min_silence_ms: 0,
        ..Default::default()
    };
    let mut it = VadIterator::new(512, &opts).unwrap();
    it.feed(&[0.9, 0.3, 0.3]).unwrap();
    assert!(it.triggered());
    let bad = VadIteratorOptions {
        neg_threshold: 0.6,
        ..Default::default()
    };
    assert!(matches!(
        VadIterator::new(512, &bad),
        Err(Error::InvalidArgument(_))
    ));
}
