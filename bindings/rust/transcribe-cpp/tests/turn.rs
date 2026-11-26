//! TURN role on the toy fixtures (tests/fixtures/arch_smart_turn.gguf: clip
//! model; tests/fixtures/arch_sava.gguf: streaming, speech track; both built by
//! the C++ test fixtures): info, run, stream parity with run and stream
//! positions, threshold override, batch == separate feeds, invalid input. Plus
//! local-only real-model checks when models/smart-turn-v3.2 and models/sava
//! are present.

mod common;

use transcribe_cpp::{
    Backend, CancelToken, Error, Model, ModelOptions, Role, TurnOptions, TurnResult, TurnSession,
};

fn cpu() -> ModelOptions {
    ModelOptions {
        backend: Backend::Cpu,
        ..Default::default()
    }
}

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

/// Concatenate the probabilities of a feed / flush sequence, checking each
/// call's `first_frame` / `speech_first_frame` continue where the last ended.
#[derive(Default)]
struct Track {
    probs: Vec<f32>,
    speech: Vec<f32>,
}

impl Track {
    fn push(&mut self, r: &TurnResult) {
        assert_eq!(r.first_frame, self.probs.len() as i64);
        assert_eq!(r.speech_first_frame, self.speech.len() as i64);
        assert_eq!((r.p, r.ended), (0.0, false));
        self.probs.extend_from_slice(&r.probs);
        self.speech.extend_from_slice(&r.speech_probs);
    }
}

fn threshold(t: f64) -> TurnOptions {
    TurnOptions { threshold: t }
}

#[test]
fn toy_smart_turn_clip() {
    let Some(path) = common::turn_toy_model("arch_smart_turn.gguf", "toy_smart_turn_clip") else {
        return;
    };
    let model = Model::load_with(&path, &cpu()).unwrap();
    assert!(model.roles().contains(Role::Turn));
    assert!(!model.roles().contains(Role::Vad));
    let info = model.turn_info().unwrap();
    assert_eq!(info.sample_rate, 16000);
    assert_eq!(info.max_audio_ms, 200);
    assert_eq!(info.frame_samples, 0);
    assert!(!info.has_speech);
    assert_eq!(info.threshold, 0.5);

    let pcm = noise(5000, 3);
    let mut turn = model.turn_session().unwrap();
    let r = turn.run(&pcm, &TurnOptions::default()).unwrap();
    assert!((0.0..=1.0).contains(&r.p), "{r:?}");
    assert_eq!(r.ended, f64::from(r.p) >= info.threshold);
    assert!(r.probs.is_empty() && r.speech_probs.is_empty());
    assert_eq!((r.first_frame, r.speech_first_frame), (0, 0));
    // Longer than max_audio_ms: scored on its last 3200 samples.
    let tail = turn.run(&pcm[pcm.len() - 3200..], &TurnOptions::default());
    assert_eq!(tail.unwrap().p, r.p);

    // A clip model cannot stream.
    for e in [
        turn.stream_feed(&pcm[..640]).unwrap_err(),
        turn.stream_flush().unwrap_err(),
        TurnSession::stream_feed_batch([(&mut turn, &pcm[..640])]).unwrap_err(),
    ] {
        assert!(matches!(e, Error::NotImplemented(_)), "{e:?}");
    }
    turn.stream_reset();
    assert!(turn.timings().load_ms >= 0.0);
}

#[test]
fn toy_sava_run_and_stream_match() {
    let Some(path) = common::turn_toy_model("arch_sava.gguf", "toy_sava_run_and_stream_match")
    else {
        return;
    };
    let model = Model::load_with(&path, &cpu()).unwrap();
    assert!(model.roles().contains(Role::Turn));
    assert!(model.roles().contains(Role::Vad));
    let info = model.turn_info().unwrap();
    assert_eq!((info.sample_rate, info.frame_samples), (16000, 640));
    assert!(info.has_speech);
    assert_eq!(info.threshold, 0.9);

    let pcm = noise(16000 + 100, 7);
    let frames = pcm.len().div_ceil(640);
    let mut turn = model.turn_session().unwrap();
    let off = turn.run(&pcm, &TurnOptions::default()).unwrap();
    assert_eq!(off.probs.len(), frames);
    assert_eq!(off.speech_probs.len(), frames);
    assert_eq!((off.first_frame, off.speech_first_frame), (0, 0));
    assert_eq!(off.p, *off.probs.last().unwrap());
    assert_eq!(off.ended, f64::from(off.p) >= info.threshold);

    // Any chunking (including empty feeds) reproduces the run bit for bit.
    let mut track = Track::default();
    track.push(&turn.stream_feed(&[]).unwrap());
    for chunk in pcm.chunks(777) {
        track.push(&turn.stream_feed(chunk).unwrap());
    }
    track.push(&turn.stream_flush().unwrap());
    assert_eq!(track.probs, off.probs);
    assert_eq!(track.speech, off.speech_probs);

    // flush and reset both restart the stream at frame 0.
    assert_eq!(turn.stream_feed(&pcm[..1280]).unwrap().first_frame, 0);
    assert_eq!(turn.stream_feed(&pcm[..1280]).unwrap().first_frame, 2);
    turn.stream_reset();
    assert_eq!(turn.stream_feed(&pcm[..1280]).unwrap().first_frame, 0);
    // A run discards the stream.
    turn.run(&pcm[..1280], &TurnOptions::default()).unwrap();
    assert_eq!(turn.stream_feed(&pcm[..1280]).unwrap().first_frame, 0);
}

#[test]
fn toy_threshold_override_changes_ended() {
    for file in ["arch_smart_turn.gguf", "arch_sava.gguf"] {
        let Some(path) = common::turn_toy_model(file, "toy_threshold_override_changes_ended")
        else {
            return;
        };
        let model = Model::load_with(&path, &cpu()).unwrap();
        let mut turn = model.turn_session().unwrap();
        let pcm = noise(3000, 5);
        let base = turn.run(&pcm, &TurnOptions::default()).unwrap();
        let p = f64::from(base.p);
        // At its own probability the turn has ended; just above, it has not.
        let at = turn.run(&pcm, &threshold(p)).unwrap();
        assert_eq!((at.p, at.ended), (base.p, true), "{file}");
        assert!(turn.run(&pcm, &threshold(0.0)).unwrap().ended, "{file}");
        if p < 1.0 {
            let above = turn.run(&pcm, &threshold((p + 1.0) / 2.0)).unwrap();
            assert_eq!((above.p, above.ended), (base.p, false), "{file}");
        }
        // Any negative threshold is the model's.
        assert_eq!(turn.run(&pcm, &threshold(-5.0)).unwrap(), base, "{file}");
    }
}

#[test]
fn toy_sava_batch_equals_separate_feeds() {
    let Some(path) =
        common::turn_toy_model("arch_sava.gguf", "toy_sava_batch_equals_separate_feeds")
    else {
        return;
    };
    let model = Model::load_with(&path, &cpu()).unwrap();
    let users: Vec<Vec<f32>> = (0..3)
        .map(|i| noise(9000 + 517 * i, 11 + i as u32))
        .collect();
    let mut batched: Vec<TurnSession> = (0..3).map(|_| model.turn_session().unwrap()).collect();
    let mut alone: Vec<TurnSession> = (0..3).map(|_| model.turn_session().unwrap()).collect();

    // Different chunk sizes per user, so the users' frame counts drift apart;
    // a user with nothing left feeds an empty slice.
    let sizes = [320usize, 777, 1500];
    let mut pos = [0usize; 3];
    let mut tracks: Vec<Track> = (0..3).map(|_| Track::default()).collect();
    while pos.iter().zip(&users).any(|(&p, u)| p < u.len()) {
        let chunks: Vec<&[f32]> = (0..3)
            .map(|i| {
                let end = (pos[i] + sizes[i]).min(users[i].len());
                let c = &users[i][pos[i]..end];
                pos[i] = end;
                c
            })
            .collect();
        let got =
            TurnSession::stream_feed_batch(batched.iter_mut().zip(chunks.iter().copied())).unwrap();
        assert_eq!(got.len(), 3);
        for i in 0..3 {
            let want = alone[i].stream_feed(chunks[i]).unwrap();
            assert_eq!(got[i], want, "user {i}");
            tracks[i].push(&got[i]);
        }
    }
    for i in 0..3 {
        let a = batched[i].stream_flush().unwrap();
        assert_eq!(a, alone[i].stream_flush().unwrap());
        tracks[i].push(&a);
        // And the batched stream equals a run over the same audio.
        let off = alone[i].run(&users[i], &TurnOptions::default()).unwrap();
        assert_eq!(tracks[i].probs, off.probs);
        assert_eq!(tracks[i].speech, off.speech_probs);
    }

    // Invalid input (NaN) leaves every stream unchanged.
    let pcm = noise(640, 1);
    batched[0].stream_feed(&pcm).unwrap();
    let mut nan = pcm.clone();
    nan[3] = f32::NAN;
    let (b0, rest) = batched.split_at_mut(1);
    let e = TurnSession::stream_feed_batch([(&mut b0[0], &pcm[..]), (&mut rest[0], &nan[..])])
        .unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e:?}");
    assert_eq!(batched[0].stream_feed(&pcm).unwrap().first_frame, 1);
    // So do sessions of different models (the library rejects them).
    let other = Model::load_with(&path, &cpu()).unwrap();
    let mut foreign = other.turn_session().unwrap();
    let e = TurnSession::stream_feed_batch([(&mut batched[0], &pcm[..]), (&mut foreign, &pcm[..])])
        .unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e:?}");
    assert_eq!(batched[0].stream_feed(&pcm).unwrap().first_frame, 2);
    assert_eq!(foreign.stream_feed(&pcm).unwrap().first_frame, 0);
}

#[test]
fn toy_invalid_input_and_roles() {
    let Some(path) = common::turn_toy_model("arch_sava.gguf", "toy_invalid_input_and_roles") else {
        return;
    };
    let model = Model::load_with(&path, &cpu()).unwrap();
    let mut turn = model.turn_session().unwrap();
    let pcm = noise(1280, 9);
    let mut nan = pcm.clone();
    nan[100] = f32::NAN;
    let mut inf = pcm.clone();
    inf[5] = f32::INFINITY;
    for e in [
        turn.run(&[], &TurnOptions::default()).unwrap_err(),
        turn.run(&nan, &TurnOptions::default()).unwrap_err(),
        turn.run(&inf, &TurnOptions::default()).unwrap_err(),
        turn.run(&pcm, &threshold(1.5)).unwrap_err(),
        turn.run(&pcm, &threshold(f64::NAN)).unwrap_err(),
    ] {
        assert!(matches!(e, Error::InvalidArgument(_)), "{e:?}");
    }
    // A rejected feed keeps the stream and its position.
    assert_eq!(turn.stream_feed(&pcm).unwrap().first_frame, 0);
    let e = turn.stream_feed(&nan).unwrap_err();
    assert!(matches!(e, Error::InvalidArgument(_)), "{e:?}");
    assert_eq!(turn.stream_feed(&pcm).unwrap().first_frame, 2);

    // A model without the TURN role.
    if let Some(vad) = common::vad_toy_model("toy_invalid_input_and_roles") {
        let vad = Model::load_with(&vad, &cpu()).unwrap();
        assert!(!vad.roles().contains(Role::Turn));
        assert!(matches!(vad.turn_info(), Err(Error::UnsupportedRole(_))));
        assert!(matches!(vad.turn_session(), Err(Error::UnsupportedRole(_))));
    }
}

#[test]
fn toy_cancel_aborts_and_resets() {
    let Some(path) = common::turn_toy_model("arch_sava.gguf", "toy_cancel_aborts_and_resets")
    else {
        return;
    };
    let model = Model::load_with(&path, &cpu()).unwrap();
    let mut a = model.turn_session().unwrap();
    let mut b = model.turn_session().unwrap();
    let pcm = noise(1280, 4);
    let token = CancelToken::new();
    b.set_cancel_token(&token);
    token.cancel();
    let e = b.run(&pcm, &TurnOptions::default()).unwrap_err();
    assert!(matches!(e, Error::Aborted { .. }), "{e:?}");
    // A batch checks every session's token (here only the second's), and
    // the abort resets every listed stream.
    assert_eq!(a.stream_feed(&pcm).unwrap().first_frame, 0);
    let e = TurnSession::stream_feed_batch([(&mut a, &pcm[..]), (&mut b, &pcm[..])]).unwrap_err();
    assert!(matches!(e, Error::Aborted { .. }), "{e:?}");
    b.clear_cancel_token();
    assert_eq!(a.stream_feed(&pcm).unwrap().first_frame, 0);
    assert_eq!(b.stream_feed(&pcm).unwrap().first_frame, 0);
}

#[test]
fn real_smart_turn_and_sava() {
    let Some(audio) = common::smoke_audio() else {
        eprintln!("skip real_smart_turn_and_sava: samples/jfk.wav absent");
        return;
    };
    if let Some(path) = common::smoke_smart_turn_model() {
        let model = Model::load_with(&path, &cpu()).unwrap();
        let info = model.turn_info().unwrap();
        assert_eq!((info.max_audio_ms, info.frame_samples), (8000, 0));
        let r = model
            .turn_session()
            .unwrap()
            .run(&audio, &TurnOptions::default())
            .unwrap();
        assert!((0.0..=1.0).contains(&r.p), "{r:?}");
        assert_eq!(r.ended, f64::from(r.p) >= info.threshold);
    } else {
        eprintln!("skip real smart turn: model absent");
    }
    if let Some(path) = common::smoke_sava_model() {
        let model = Model::load_with(&path, &cpu()).unwrap();
        let info = model.turn_info().unwrap();
        assert!(info.frame_samples > 0 && info.has_speech);
        let mut turn = model.turn_session().unwrap();
        let off = turn.run(&audio, &TurnOptions::default()).unwrap();
        assert_eq!(
            off.probs.len(),
            audio.len().div_ceil(info.frame_samples as usize)
        );
        let mut track = Track::default();
        for chunk in audio.chunks(320) {
            track.push(&turn.stream_feed(chunk).unwrap());
        }
        track.push(&turn.stream_flush().unwrap());
        assert_eq!(track.probs, off.probs);
        assert_eq!(track.speech, off.speech_probs);
    } else {
        eprintln!("skip real sava: model absent");
    }
}
