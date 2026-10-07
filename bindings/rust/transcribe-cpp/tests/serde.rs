//! `serde` feature: derive coverage plus the crate's custom serde behavior
//! (NaN confidences, sparse-document defaults).

#![cfg(feature = "serde")]

use transcribe_cpp::{
    Backend, Capabilities, CommitPolicy, DeviceType, Diarize, DiarizeExtension, DiarizeInfo,
    DiarizeOptions, DiarizeSessionOptions, ExtSlot, Feature, Itn, KvType,
    MoonshineStreamingOptions, ParakeetBufferedStreamOptions, ParakeetStreamOptions, Pnc, Role,
    Roles, RunExtension, RunOptions, Segment, SessionLimits, SessionOptions,
    SortformerDiarizeOptions, SortformerPreset, SpeakerSegment, StreamExtension, StreamOptions,
    StreamState, StreamText, StreamUpdate, Task, TimestampKind, Timings, Token, Transcript,
    VoxtralRealtimeStreamOptions, WhisperRunOptions, Word,
};

fn assert_serde<T: serde::Serialize + serde::de::DeserializeOwned>() {}

/// Fails to compile if any plain-data type loses its derives.
#[test]
fn plain_data_types_are_serializable() {
    // Options.
    assert_serde::<RunOptions>();
    assert_serde::<StreamOptions>();
    assert_serde::<SessionOptions>();
    assert_serde::<RunExtension>();
    assert_serde::<StreamExtension>();
    assert_serde::<WhisperRunOptions>();
    assert_serde::<MoonshineStreamingOptions>();
    assert_serde::<ParakeetStreamOptions>();
    assert_serde::<ParakeetBufferedStreamOptions>();
    assert_serde::<VoxtralRealtimeStreamOptions>();
    assert_serde::<SortformerPreset>();
    assert_serde::<DiarizeOptions>();
    assert_serde::<DiarizeSessionOptions>();
    assert_serde::<DiarizeExtension>();
    assert_serde::<SortformerDiarizeOptions>();
    // Results.
    assert_serde::<DiarizeInfo>();
    assert_serde::<Transcript>();
    assert_serde::<Segment>();
    assert_serde::<SpeakerSegment>();
    assert_serde::<Word>();
    assert_serde::<Token>();
    assert_serde::<Timings>();
    assert_serde::<StreamUpdate>();
    assert_serde::<StreamText>();
    assert_serde::<Capabilities>();
    assert_serde::<SessionLimits>();
    // Enums.
    assert_serde::<Task>();
    assert_serde::<TimestampKind>();
    assert_serde::<KvType>();
    assert_serde::<Pnc>();
    assert_serde::<Itn>();
    assert_serde::<Diarize>();
    assert_serde::<Backend>();
    assert_serde::<Feature>();
    assert_serde::<CommitPolicy>();
    assert_serde::<StreamState>();
    assert_serde::<DeviceType>();
    assert_serde::<ExtSlot>();
    assert_serde::<Role>();
    assert_serde::<Roles>();
}

fn nan_transcript() -> Transcript {
    Transcript {
        text: "and so".into(),
        speaker_segments: vec![SpeakerSegment {
            t1_ms: 900,
            speaker_id: 1,
            p: f32::NAN,
            ..Default::default()
        }],
        tokens: vec![
            Token {
                id: 42,
                p: f32::NAN,
                text: " and".into(),
                ..Default::default()
            },
            Token {
                id: 43,
                p: 0.75,
                text: " so".into(),
                ..Default::default()
            },
        ],
        ..Default::default()
    }
}

/// NaN != NaN, so compare by postcard bytes.
fn same_transcript(a: &Transcript, b: &Transcript) -> bool {
    postcard::to_allocvec(a).unwrap() == postcard::to_allocvec(b).unwrap()
}

#[test]
fn nan_confidence_round_trips() {
    let original = nan_transcript();

    let json = serde_json::to_string(&original).unwrap();
    assert!(
        json.contains("\"p\":null"),
        "NaN should encode as null: {json}"
    );
    let from_json: Transcript = serde_json::from_str(&json).unwrap();
    assert!(from_json.tokens[0].p.is_nan());
    assert!(from_json.speaker_segments[0].p.is_nan());
    assert_eq!(from_json.tokens[1].p, 0.75);
    assert!(same_transcript(&from_json, &original), "json");

    let bytes = postcard::to_allocvec(&original).unwrap();
    let from_postcard: Transcript = postcard::from_bytes(&bytes).unwrap();
    assert!(from_postcard.tokens[0].p.is_nan());
    assert!(from_postcard.speaker_segments[0].p.is_nan());
    assert!(same_transcript(&from_postcard, &original), "postcard");
}

#[test]
fn missing_fields_take_defaults() {
    // `RunOptions::default()` is hand-written; its -1 sentinel must survive.
    let run: RunOptions = serde_json::from_str(r#"{"language":"fr"}"#).unwrap();
    assert_eq!(run.spec_k_drafts, -1);
    assert_eq!(
        run,
        RunOptions {
            language: Some("fr".into()),
            ..Default::default()
        }
    );

    let token: Token = serde_json::from_str(r#"{"id":5}"#).unwrap();
    assert_eq!(token.id, 5);
    assert!(token.p.is_nan(), "missing p decoded as {}", token.p);
    let speaker: SpeakerSegment = serde_json::from_str(r#"{"speaker_id":2}"#).unwrap();
    assert!(speaker.p.is_nan(), "missing p decoded as {}", speaker.p);
}
