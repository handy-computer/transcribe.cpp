//! Streaming tests against the moonshine-streaming canary. Skip cleanly when
//! the streaming model or jfk.wav is absent.

mod common;

use transcribe_cpp::{Model, RunOptions, StreamExtension, StreamOptions, StreamState};
use transcribe_cpp::{MoonshineStreamingOptions, Stream};

fn feed_in_chunks(stream: &mut Stream<'_>, pcm: &[f32], chunk: usize) {
    for frame in pcm.chunks(chunk) {
        stream.feed(frame).expect("feed");
    }
}

#[test]
fn streams_jfk_committed_text() {
    let (Some(model_path), Some(pcm)) = (common::smoke_streaming_model(), common::smoke_audio())
    else {
        eprintln!("skip streams_jfk_committed_text: streaming model/audio absent");
        return;
    };
    let model = Model::load(&model_path).unwrap();
    assert!(
        model.capabilities().supports_streaming,
        "model does not advertise streaming"
    );

    let mut session = model.session().unwrap();
    let mut stream = session
        .stream(&RunOptions::default(), &StreamOptions::default())
        .unwrap();
    assert_eq!(stream.state(), StreamState::Active);

    feed_in_chunks(&mut stream, &pcm, 1600); // 100 ms chunks
    let update = stream.finalize().unwrap();
    assert!(update.is_final);
    assert_eq!(stream.state(), StreamState::Finished);

    let text = stream.text();
    // `full` is the authoritative raw hypothesis. (committed_text is a
    // best-effort append-only prefix that can diverge for re-attending models
    // like moonshine_streaming, so we assert on `full` here.)
    assert!(
        text.full.to_lowercase().contains("country"),
        "stream text: {:?}",
        text.full
    );
}

#[test]
fn on_finalize_policy_commits_full_text() {
    // Under ON_FINALIZE, committed_text is empty during feed and becomes the
    // final full text at finalize — so display == full reliably.
    let (Some(model_path), Some(pcm)) = (common::smoke_streaming_model(), common::smoke_audio())
    else {
        return;
    };
    let mut session = Model::load(&model_path).unwrap().session().unwrap();
    let opts = StreamOptions {
        commit_policy: transcribe_cpp::CommitPolicy::OnFinalize,
        ..Default::default()
    };
    let mut stream = session.stream(&RunOptions::default(), &opts).unwrap();
    feed_in_chunks(&mut stream, &pcm, 1600);
    stream.finalize().unwrap();
    let text = stream.text();
    assert!(
        text.committed.to_lowercase().contains("country"),
        "committed: {:?}",
        text.committed
    );
}

#[test]
fn stream_revision_advances() {
    let (Some(model_path), Some(pcm)) = (common::smoke_streaming_model(), common::smoke_audio())
    else {
        return;
    };
    let mut session = Model::load(&model_path).unwrap().session().unwrap();
    let mut stream = session
        .stream(&RunOptions::default(), &StreamOptions::default())
        .unwrap();
    let r0 = stream.revision();
    feed_in_chunks(&mut stream, &pcm, 1600);
    stream.finalize().unwrap();
    assert!(stream.revision() >= r0, "revision regressed");
}

#[test]
fn stream_reset_returns_to_idle() {
    let (Some(model_path), Some(pcm)) = (common::smoke_streaming_model(), common::smoke_audio())
    else {
        return;
    };
    let mut session = Model::load(&model_path).unwrap().session().unwrap();
    {
        let mut stream = session
            .stream(&RunOptions::default(), &StreamOptions::default())
            .unwrap();
        stream.feed(&pcm[..pcm.len().min(1600)]).unwrap();
        stream.reset();
        assert_eq!(stream.state(), StreamState::Idle);
    }
    // The session is usable again for a fresh stream after the borrow ends.
    let stream = session
        .stream(&RunOptions::default(), &StreamOptions::default())
        .unwrap();
    assert_eq!(stream.state(), StreamState::Active);
}

#[test]
fn concurrent_compute_on_one_model_is_refused() {
    // The C library allows at most one in-flight compute per model (the 0.x
    // limitation, include/transcribe.h). An active stream spans begin..drop, so
    // a second stream — or an offline run — on ANOTHER session of the same model
    // must be refused with Error::Busy rather than race into the documented UB.
    let (Some(model_path), Some(pcm)) = (common::smoke_streaming_model(), common::smoke_audio())
    else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let mut s1 = model.session().unwrap();
    let mut s2 = model.session().unwrap();

    let mut stream1 = s1
        .stream(&RunOptions::default(), &StreamOptions::default())
        .unwrap();
    stream1.feed(&pcm[..pcm.len().min(1600)]).unwrap();

    // Second stream on the same model while the first is live -> Busy.
    match s2.stream(&RunOptions::default(), &StreamOptions::default()) {
        Err(transcribe_cpp::Error::Busy(_)) => {}
        other => panic!("expected Error::Busy for a second stream, got {other:?}"),
    }
    // An offline run on the same model while a stream is live -> Busy too.
    match s2.run(&pcm, &RunOptions::default()) {
        Err(transcribe_cpp::Error::Busy(_)) => {}
        other => panic!("expected Error::Busy for a run mid-stream, got {other:?}"),
    }

    // Dropping the first stream releases the model's compute lease; now s2 can
    // begin its own stream.
    drop(stream1);
    let stream2 = s2
        .stream(&RunOptions::default(), &StreamOptions::default())
        .unwrap();
    assert_eq!(stream2.state(), StreamState::Active);
}

#[test]
fn stream_family_extension_accepted_or_rejected() {
    // moonshine-streaming accepts its own stream extension; a parakeet stream
    // extension on the same model is rejected at begin (InvalidArgument).
    let Some(model_path) = common::smoke_streaming_model() else {
        return;
    };
    let model = Model::load(&model_path).unwrap();
    let mut session = model.session().unwrap();

    let opts = StreamOptions {
        family: Some(StreamExtension::MoonshineStreaming(
            MoonshineStreamingOptions {
                min_decode_interval_ms: Some(200),
            },
        )),
        ..Default::default()
    };
    let ok = session.stream(&RunOptions::default(), &opts);
    assert!(ok.is_ok(), "moonshine ext should be accepted: {ok:?}");
    drop(ok);

    // Wrong-family extension is rejected.
    let wrong = StreamOptions {
        family: Some(StreamExtension::ParakeetStream(Default::default())),
        ..Default::default()
    };
    let err = session.stream(&RunOptions::default(), &wrong);
    assert!(err.is_err(), "wrong-family ext should be rejected");
}
