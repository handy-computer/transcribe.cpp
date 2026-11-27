//! The custom serde behavior, tested without the engine: NaN confidences,
//! sparse documents taking the hand-written defaults, unknown error kinds.
//! (transcribe-cpp's `tests/serde.rs` checks the same through its paths.)

#![cfg(feature = "serde")]

use transcribe_cpp_types::{
    ErrorKind, ErrorReport, LangIdResult, RunOptions, SpeakerSegment, Token, Transcript,
    TurnOptions, VadOptions,
};

#[test]
fn nan_confidence_is_null_in_json_and_raw_in_postcard() {
    let t = Transcript {
        tokens: vec![Token {
            id: 1,
            p: f32::NAN,
            ..Default::default()
        }],
        speaker_segments: vec![SpeakerSegment {
            p: f32::NAN,
            ..Default::default()
        }],
        ..Default::default()
    };
    let json = serde_json::to_string(&t).unwrap();
    assert!(json.contains("\"p\":null"), "{json}");
    let back: Transcript = serde_json::from_str(&json).unwrap();
    assert!(back.tokens[0].p.is_nan() && back.speaker_segments[0].p.is_nan());
    let bytes = postcard::to_allocvec(&t).unwrap();
    let back: Transcript = postcard::from_bytes(&bytes).unwrap();
    assert!(back.tokens[0].p.is_nan() && back.speaker_segments[0].p.is_nan());
}

#[test]
fn missing_fields_take_defaults() {
    let run: RunOptions = serde_json::from_str(r#"{"language":"fr"}"#).unwrap();
    assert_eq!(run.spec_k_drafts, -1);
    let token: Token = serde_json::from_str(r#"{"id":5}"#).unwrap();
    assert!(token.p.is_nan());
    let langid: LangIdResult = serde_json::from_str(r#"{"allowed_mass":0.5}"#).unwrap();
    assert!(langid.candidates.is_empty());
    let vad: VadOptions = serde_json::from_str(r#"{"threshold":0.6}"#).unwrap();
    assert_eq!((vad.neg_threshold, vad.min_speech_ms), (-1.0, 250));
    let turn: TurnOptions = serde_json::from_str("{}").unwrap();
    assert_eq!(turn.threshold, -1.0);
}

#[test]
fn error_report_reads_unknown_kinds_and_sparse_documents() {
    let report: ErrorReport =
        serde_json::from_str(r#"{"kind":"from_the_future","message":"m"}"#).unwrap();
    assert_eq!(report, ErrorReport::new(ErrorKind::Other, "m"));
}
