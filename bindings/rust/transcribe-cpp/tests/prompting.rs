//! Generic prompting (vocabulary / prompt / prefix) against the whisper and
//! streaming canaries. Mirrors bindings/python/tests/test_prompting.py.

mod common;

use transcribe_cpp::{Error, Feature, Model, RunOptions, StreamOptions, Task, TimestampKind};

fn terms() -> Vec<String> {
    ["Kennedy", "Americans", "GGUF"].map(String::from).to_vec()
}

#[test]
fn nul_in_prompting_text_is_an_error() {
    let Some((model_path, pcm)) = common::smoke_fixtures("nul_in_prompting_text_is_an_error")
    else {
        return;
    };
    let mut session = Model::load(&model_path).unwrap().session().unwrap();
    let options = RunOptions {
        vocabulary: vec!["a\0b".into()],
        ..Default::default()
    };
    assert!(matches!(session.run(&pcm, &options), Err(Error::Nul(_))));
}

#[test]
fn vocabulary_and_prompt_reach_run_and_batch() {
    let Some((model_path, pcm)) =
        common::smoke_fixtures("vocabulary_and_prompt_reach_run_and_batch")
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let mut session = model.session().unwrap();
    let options = RunOptions {
        vocabulary: terms(),
        prompt: Some("A speech.".into()),
        ..Default::default()
    };
    let result = session.run(&pcm, &options).unwrap();
    assert!(result.text.to_lowercase().contains("country"));
    let batch = session.run_batch(&[&pcm, &pcm], &options).unwrap();
    assert!(batch.iter().all(|r| r.is_ok()));
    // A control-token literal is rejected; a prefix cannot apply to a batch.
    let control = RunOptions {
        prompt: Some("hi <|endoftext|>".into()),
        ..Default::default()
    };
    assert!(matches!(
        session.run(&pcm, &control),
        Err(Error::InvalidArgument(_))
    ));
    let prefix = RunOptions {
        prefix: Some("And so".into()),
        ..Default::default()
    };
    assert!(matches!(
        session.run_batch(&[&pcm], &prefix),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn prefix_text_is_the_continuation() {
    let Some((model_path, pcm)) = common::smoke_fixtures("prefix_text_is_the_continuation") else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    if !model.supports(Feature::TranscriptPrefix) {
        eprintln!("skip: model does not support a transcript prefix");
        return;
    }
    let mut session = model.session().unwrap();
    let options = RunOptions {
        prefix: Some("And so my fellow Americans".into()),
        timestamps: TimestampKind::None,
        ..Default::default()
    };
    let result = session.run(&pcm, &options).unwrap();
    assert!(result
        .raw_text
        .trim_start()
        .starts_with("And so my fellow Americans"));
    assert!(!result.text.contains("fellow Americans"));
    assert!(result.text.to_lowercase().contains("ask not"));
}

#[test]
fn stream_accepts_vocabulary_rejects_instruct() {
    let Some(model_path) = common::smoke_streaming_model() else {
        eprintln!("skip: streaming canary unavailable");
        return;
    };
    let mut session = Model::load(&model_path).unwrap().session().unwrap();
    let instruct = RunOptions {
        task: Task::Instruct,
        prompt: Some("Summarize.".into()),
        ..Default::default()
    };
    assert!(matches!(
        session.stream(&instruct, &StreamOptions::default()),
        Err(Error::Unsupported(_))
    ));
    let options = RunOptions {
        vocabulary: terms(),
        prompt: Some("A speech.".into()),
        ..Default::default()
    };
    assert!(session.stream(&options, &StreamOptions::default()).is_ok());
}
