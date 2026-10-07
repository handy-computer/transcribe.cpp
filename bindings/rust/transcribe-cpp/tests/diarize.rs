//! DIARIZE role on Sortformer (TRANSCRIBE_SMOKE_SORTFORMER_MODEL) and whisper
//! (TRANSCRIBE_SMOKE_MODEL); the Busy path is covered in `src/diarize.rs`.

mod common;

use transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_SORTFORMER_DIARIZE as SFDR;
use transcribe_cpp::{
    Backend, CancelToken, DiarizeExtension, DiarizeOptions, Error, ExtSlot, Model, ModelOptions,
    Role, SortformerDiarizeOptions, SortformerPreset,
};

#[test]
fn sortformer_is_diarize_only() {
    let Some((model_path, _)) = common::smoke_sortformer_fixtures("sortformer_is_diarize_only")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let roles = model.roles();
    assert!(!roles.contains(Role::Asr) && roles.contains(Role::Diarize));
    let info = model.diarize_info().unwrap();
    assert_eq!((info.sample_rate, info.max_speakers), (16000, 4));
    assert!(model.accepts_ext(ExtSlot::DiarizeRun, SFDR));
    assert!(!model.accepts_ext(ExtSlot::Run, SFDR));
    assert!(matches!(
        model.capabilities(),
        Err(Error::UnsupportedRole(_))
    ));
    assert!(matches!(model.session(), Err(Error::UnsupportedRole(_))));
}

#[test]
fn diarize_run_returns_turns_and_applies_preset() {
    let Some((model_path, pcm)) =
        common::smoke_sortformer_fixtures("diarize_run_returns_turns_and_applies_preset")
    else {
        return;
    };
    // Exact segments are pinned in C (tests/sortformer_diarize_unit.cpp); here
    // the binding only has to return well-formed turns and pass the preset on.
    let cpu = ModelOptions {
        backend: Backend::Cpu,
        ..Default::default()
    };
    let model = Model::load_with(&model_path, &cpu).unwrap();
    let max_speakers = model.diarize_info().unwrap().max_speakers;
    let mut diarize = model.diarize_session().unwrap();
    let rows = |turns: Vec<transcribe_cpp::SpeakerSegment>| -> Vec<_> {
        turns
            .iter()
            .map(|s| (s.t0_ms, s.t1_ms, s.speaker_id))
            .collect()
    };

    let default = rows(diarize.run(&pcm, &DiarizeOptions::default()).unwrap());
    assert!(!default.is_empty());
    for &(t0, t1, speaker) in &default {
        assert!(t0 < t1, "{t0} >= {t1}");
        assert!((1..=max_speakers).contains(&speaker), "speaker {speaker}");
    }
    assert!(diarize.timings().encode_ms > 0.0);

    let low_latency = DiarizeOptions {
        family: Some(DiarizeExtension::Sortformer(SortformerDiarizeOptions {
            preset: Some(SortformerPreset::LowLatency),
        })),
    };
    assert_ne!(rows(diarize.run(&pcm, &low_latency).unwrap()), default);
}

#[test]
fn diarize_session_rejects_bad_input_and_cancels() {
    let Some((model_path, pcm)) =
        common::smoke_sortformer_fixtures("diarize_session_rejects_bad_input_and_cancels")
    else {
        return;
    };
    // The session keeps its model alive after the last Model handle drops.
    let mut diarize = Model::load(&model_path).unwrap().diarize_session().unwrap();
    let opts = DiarizeOptions::default();
    for bad in [&[][..], &[f32::NAN; 16][..]] {
        let err = diarize.run(bad, &opts).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }
    let token = CancelToken::new();
    diarize.set_cancel_token(&token);
    token.cancel();
    let err = diarize.run(&pcm, &opts).unwrap_err();
    assert!(matches!(err, Error::Aborted { .. }), "{err:?}");
    diarize.clear_cancel_token();
    assert!(!diarize.run(&pcm, &opts).unwrap().is_empty());
}

#[test]
fn asr_only_model_refuses_diarize() {
    let Some(model_path) = common::smoke_model() else {
        eprintln!("skip asr_only_model_refuses_diarize: smoke model absent");
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let roles = model.roles();
    assert!(roles.contains(Role::Asr) && !roles.contains(Role::Diarize));
    assert!(matches!(
        model.diarize_info(),
        Err(Error::UnsupportedRole(_))
    ));
    assert!(matches!(
        model.diarize_session(),
        Err(Error::UnsupportedRole(_))
    ));
    assert!(!model.accepts_ext(ExtSlot::DiarizeRun, SFDR));
}
