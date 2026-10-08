//! LANGID role: contract checks on the toy ecapa_tdnn fixture
//! (tests/fixtures/arch_ecapa_tdnn_minimal.gguf, built by the C++ test
//! fixtures) and top-1 on the real VoxLingua107 model
//! (TRANSCRIBE_SMOKE_LANGID_MODEL). Busy / empty-allowed are covered in
//! `src/langid.rs`.

mod common;

use transcribe_cpp::{Backend, CancelToken, Error, LangIdOptions, Model, ModelOptions, Role};

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) & 0xFF_FFFF) as f32 / 16_777_216.0 - 0.5
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
fn toy_roles_info_labels() {
    let Some(path) = common::langid_toy_model("toy_roles_info_labels") else {
        return;
    };
    let model = load_cpu(&path);
    let roles = model.roles();
    assert!(roles.contains(Role::LangId) && !roles.contains(Role::Asr));
    let info = model.langid_info().unwrap();
    assert_eq!(
        (
            info.sample_rate,
            info.n_labels,
            info.min_audio_ms,
            info.max_audio_ms
        ),
        (16000, 5, 500, 30000)
    );
    assert_eq!(
        model.langid_labels().unwrap()[2],
        ("cc".into(), "Charlie".into())
    );
    assert_eq!(model.langid_label_index("xx"), Some(0));
    assert_eq!(model.langid_label_index("zz"), None);
    assert!(matches!(
        model.capabilities(),
        Err(Error::UnsupportedRole(_))
    ));
    assert!(matches!(model.session(), Err(Error::UnsupportedRole(_))));
}

#[test]
fn toy_run_contract() {
    let Some(path) = common::langid_toy_model("toy_run_contract") else {
        return;
    };
    let model = load_cpu(&path);
    let mut lid = model.langid_session().unwrap();
    let pcm = noise(16000, 7);

    let r = lid.run(&pcm, &LangIdOptions::default()).unwrap();
    assert_eq!(r.candidates.len(), 5);
    assert_eq!(r.allowed_mass, 1.0);
    assert_eq!(r.code(), Some(r.candidates[0].code.as_str()));
    let sum: f32 = r.candidates.iter().map(|c| c.p).sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(r.candidates.windows(2).all(|w| w[0].p >= w[1].p));

    let opts = LangIdOptions {
        allowed: Some(vec!["bb".into(), "dd".into()]),
    };
    let r = lid.run(&pcm, &opts).unwrap();
    assert_eq!(r.candidates.len(), 2);
    assert!(r
        .candidates
        .iter()
        .all(|c| c.code == "bb" || c.code == "dd"));
    assert!(r.allowed_mass > 0.0 && r.allowed_mass < 1.0);

    let opts = LangIdOptions {
        allowed: Some(vec!["zz".into()]),
    };
    assert!(matches!(lid.run(&pcm, &opts), Err(Error::Unsupported(_))));
    assert!(matches!(
        lid.run(&pcm[..6400], &LangIdOptions::default()),
        Err(Error::InputTooShort(_))
    ));
    assert!(lid.timings().encode_ms > 0.0);
}

#[test]
fn toy_cancel_aborts() {
    let Some(path) = common::langid_toy_model("toy_cancel_aborts") else {
        return;
    };
    let model = load_cpu(&path);
    let mut lid = model.langid_session().unwrap();
    let token = CancelToken::new();
    lid.set_cancel_token(&token);
    token.cancel();
    assert!(matches!(
        lid.run(&noise(16000, 3), &LangIdOptions::default()),
        Err(Error::Aborted { .. })
    ));
    lid.clear_cancel_token();
    assert!(lid.run(&noise(16000, 3), &LangIdOptions::default()).is_ok());
}

#[test]
fn real_fleurs_top1() {
    let Some(path) = common::smoke_langid_model("real_fleurs_top1") else {
        return;
    };
    let model = load_cpu(&path);
    let info = model.langid_info().unwrap();
    assert_eq!((info.n_labels, info.max_audio_ms), (107, 30000));
    assert_eq!(
        model.langid_label_index("he"),
        model.langid_label_index("iw")
    );
    let mut lid = model.langid_session().unwrap();
    for code in ["en", "de", "fr", "es", "ja", "zh", "ru", "id"] {
        let pcm = common::load_wav(&common::repo_root().join(format!("samples/fleurs-{code}.wav")));
        let r = lid.run(&pcm, &LangIdOptions::default()).unwrap();
        assert_eq!(r.candidates.len(), 107);
        assert_eq!(r.code(), Some(code), "{code}: {:?}", r.candidates);
        assert!(r.candidates[0].p >= 0.5);
    }
}
