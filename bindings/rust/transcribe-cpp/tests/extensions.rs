//! Family-extension (whisper run slot) and tokenizer tests against the whisper
//! canary.

mod common;

use transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_WHISPER_RUN;
use transcribe_cpp::{ExtSlot, Model, RunExtension, RunOptions, WhisperRunOptions};

#[test]
fn whisper_accepts_run_extension() {
    let Some((model_path, _)) = common::smoke_fixtures("whisper_accepts_run_extension") else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    assert!(
        model.accepts_ext(ExtSlot::Run, TRANSCRIBE_EXT_KIND_WHISPER_RUN),
        "whisper should accept its run extension"
    );
    // A whisper run ext on the STREAM slot is not accepted.
    assert!(!model.accepts_ext(ExtSlot::Stream, TRANSCRIBE_EXT_KIND_WHISPER_RUN));
}

#[test]
fn whisper_run_with_initial_prompt() {
    let Some((model_path, pcm)) = common::smoke_fixtures("whisper_run_with_initial_prompt") else {
        return;
    };
    let mut session = Model::load(&model_path).unwrap().session().unwrap();
    let options = RunOptions {
        family: Some(RunExtension::Whisper(WhisperRunOptions {
            initial_prompt: Some("This is a presidential speech.".to_string()),
            temperature: Some(0.0),
            ..Default::default()
        })),
        ..Default::default()
    };
    let result = session.run(&pcm, &options).unwrap();
    assert!(
        result.text.to_lowercase().contains("country"),
        "{:?}",
        result.text
    );
}

#[test]
fn tokenize_round_trips_nonempty() {
    let Some((model_path, _)) = common::smoke_fixtures("tokenize_round_trips_nonempty") else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    // Whisper uses GPT-2 byte-level BPE; encode is plumbed.
    let tokens = model
        .tokenize("ask not what your country can do for you")
        .unwrap();
    assert!(!tokens.is_empty(), "no tokens produced");
    // A longer string should not produce fewer tokens than a short prefix.
    let short = model.tokenize("ask").unwrap();
    assert!(tokens.len() >= short.len());
}

#[test]
fn nemotron3_diar_run_and_stream_extensions() {
    use transcribe_cpp::sys::{
        TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_RUN, TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_STREAM,
    };
    use transcribe_cpp::{
        Nemotron3DiarOptions, Nemotron3DiarPreset, StreamExtension, StreamOptions,
    };
    let (Some(model_path), Some(pcm)) =
        (common::smoke_nemotron3_diar_model(), common::smoke_audio())
    else {
        eprintln!("skip nemotron3_diar_run_and_stream_extensions: model not present");
        return;
    };
    let model = Model::load(&model_path).unwrap();
    assert!(model.accepts_ext(ExtSlot::Run, TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_RUN));
    assert!(model.accepts_ext(ExtSlot::Stream, TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_STREAM));
    assert!(!model.accepts_ext(ExtSlot::Stream, TRANSCRIBE_EXT_KIND_NEMOTRON3_DIAR_RUN));
    let opts = Nemotron3DiarOptions {
        preset: Some(Nemotron3DiarPreset::LowLatency),
    };
    let mut session = model.session().unwrap();
    let run = RunOptions {
        family: Some(RunExtension::Nemotron3Diar(opts.clone())),
        ..Default::default()
    };
    let result = session.run(&pcm, &run).unwrap();
    assert!(!result.speaker_segments.is_empty());

    let stream_opts = StreamOptions {
        family: Some(StreamExtension::Nemotron3Diar(opts)),
        ..Default::default()
    };
    let mut stream = session
        .stream(&RunOptions::default(), &stream_opts)
        .unwrap();
    for piece in pcm.chunks(2768) {
        stream.feed(piece).unwrap();
    }
    let fin = stream.finalize().unwrap();
    assert!(fin.is_final);
}
