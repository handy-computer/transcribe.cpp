//! VAD role on the toy silero_vad fixture (tests/fixtures/arch_silero_vad.gguf,
//! built by the C++ test fixtures): run, stream parity with run, and
//! stream positions.

mod common;

use transcribe_cpp::{Backend, Model, ModelOptions, Role, VadOptions};

#[test]
fn toy_run_and_stream_match() {
    let Some(path) = common::vad_toy_model("toy_run_and_stream_match") else {
        return;
    };
    let opts = ModelOptions {
        backend: Backend::Cpu,
        ..Default::default()
    };
    let model = Model::load_with(&path, &opts).unwrap();
    assert!(model.roles().contains(Role::Vad));
    assert_eq!(model.vad_info().unwrap().frame_samples, 512);

    let mut s: u32 = 7;
    let pcm: Vec<f32> = (0..16000 + 100)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect();
    let mut vad = model.vad_session().unwrap();
    let off = vad.run(&pcm, &VadOptions::default()).unwrap();
    assert_eq!(off.probs.len(), 32);

    let mut probs = Vec::new();
    for chunk in pcm.chunks(777) {
        let r = vad.stream_feed(chunk).unwrap();
        assert_eq!(r.first_frame, probs.len() as i64);
        probs.extend_from_slice(&r.probs);
    }
    probs.extend_from_slice(&vad.stream_flush().unwrap().probs);
    assert_eq!(probs, off.probs);

    // flush and reset both restart the stream at frame 0.
    assert_eq!(vad.stream_feed(&pcm[..1024]).unwrap().first_frame, 0);
    assert_eq!(vad.stream_feed(&pcm[..1024]).unwrap().first_frame, 2);
    vad.stream_reset();
    assert_eq!(vad.stream_feed(&pcm[..1024]).unwrap().first_frame, 0);
}
