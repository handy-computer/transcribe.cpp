// Representative values of every plain-data type, shared by:
//
// - the generator that wrote `json/` and `postcard/` from transcribe-cpp 0.5.0
//   off crates.io (see README.md here),
// - `tests/serde_golden.rs` in this crate (the types alone), and
// - `tests/serde.rs` in transcribe-cpp (the same types at their old paths).
//
// It names the types through `crate::tc`, which each includer aliases to the
// crate under test. It only uses API both 0.5.0 and this crate have: struct
// literals for exhaustive structs, `Default` plus field assignment for the
// `#[non_exhaustive]` ones that implement `Default`, and JSON for the rest.

#![allow(dead_code)]

use crate::tc::*;

/// One golden: the value's JSON (`serde_json::to_string`) and postcard bytes,
/// and functions that deserialize a golden back and serialize it again.
pub struct Golden {
    pub name: &'static str,
    pub json: String,
    pub postcard: Vec<u8>,
    pub json_again: fn(&str) -> String,
    pub postcard_again: fn(&[u8]) -> Vec<u8>,
}

fn json_again<T: serde::Serialize + serde::de::DeserializeOwned>(s: &str) -> String {
    let v: T = serde_json::from_str(s).expect("golden JSON deserializes");
    serde_json::to_string(&v).unwrap()
}

fn postcard_again<T: serde::Serialize + serde::de::DeserializeOwned>(b: &[u8]) -> Vec<u8> {
    let v: T = postcard::from_bytes(b).expect("golden postcard deserializes");
    postcard::to_allocvec(&v).unwrap()
}

fn g<T: serde::Serialize + serde::de::DeserializeOwned>(name: &'static str, v: T) -> Golden {
    Golden {
        name,
        json: serde_json::to_string(&v).unwrap(),
        postcard: postcard::to_allocvec(&v).unwrap(),
        json_again: json_again::<T>,
        postcard_again: postcard_again::<T>,
    }
}

fn from_json<T: serde::de::DeserializeOwned>(s: &str) -> T {
    serde_json::from_str(s).unwrap()
}

fn s(v: &str) -> String {
    v.to_string()
}

pub fn whisper() -> WhisperRunOptions {
    WhisperRunOptions {
        initial_prompt: Some(s("Glossary: ggml, GGUF.")),
        condition_on_prev_tokens: Some(false),
        temperature: Some(0.25),
        temperature_inc: Some(0.125),
        compression_ratio_thold: Some(2.375),
        logprob_thold: Some(-1.0),
        no_speech_thold: Some(0.625),
        max_prev_context_tokens: Some(224),
        seed: Some(7),
        max_initial_timestamp: Some(1.5),
    }
}

pub fn stream_extensions() -> Vec<StreamExtension> {
    vec![
        StreamExtension::ParakeetStream(ParakeetStreamOptions {
            att_context_right: Some(13),
        }),
        StreamExtension::ParakeetBuffered(ParakeetBufferedStreamOptions {
            left_ms: Some(10000),
            chunk_ms: Some(2000),
            right_ms: Some(1500),
        }),
        StreamExtension::MoonshineStreaming(MoonshineStreamingOptions {
            min_decode_interval_ms: Some(240),
        }),
        StreamExtension::VoxtralRealtime(VoxtralRealtimeStreamOptions {
            num_delay_tokens: Some(6),
            min_decode_interval_ms: Some(480),
        }),
    ]
}

pub fn segment() -> Segment {
    Segment {
        t0_ms: 120,
        t1_ms: 1500,
        first_word: 0,
        n_words: 2,
        first_token: 0,
        n_tokens: 3,
        text: s(" Hello world"),
        speaker_id: 1,
    }
}

pub fn speaker_segments() -> Vec<SpeakerSegment> {
    vec![
        SpeakerSegment {
            t0_ms: 120,
            t1_ms: 1500,
            speaker_id: 1,
            p: 0.875,
        },
        SpeakerSegment {
            t0_ms: 0,
            t1_ms: 0,
            speaker_id: 2,
            p: f32::NAN,
        },
    ]
}

pub fn words() -> Vec<Word> {
    vec![
        Word {
            t0_ms: 120,
            t1_ms: 640,
            seg_index: 0,
            first_token: 0,
            n_tokens: 2,
            text: s(" Hello"),
        },
        Word {
            t0_ms: 700,
            t1_ms: 1500,
            seg_index: 0,
            first_token: 2,
            n_tokens: 1,
            text: s(" world"),
        },
    ]
}

pub fn tokens() -> Vec<Token> {
    vec![
        Token {
            id: 15947,
            p: 0.9375,
            t0_ms: 120,
            t1_ms: 400,
            seg_index: 0,
            word_index: 0,
            text: s(" Hel"),
        },
        Token {
            id: 75,
            p: f32::NAN,
            t0_ms: 400,
            t1_ms: 640,
            seg_index: 0,
            word_index: 0,
            text: s("lo"),
        },
        Token {
            id: 1002,
            p: 0.5,
            t0_ms: 700,
            t1_ms: 1500,
            seg_index: 0,
            word_index: 1,
            text: s(" world"),
        },
    ]
}

pub fn timings() -> Timings {
    Timings {
        load_ms: 12.5,
        mel_ms: 1.25,
        encode_ms: 30.0,
        decode_ms: 7.75,
    }
}

pub fn transcript() -> Transcript {
    Transcript {
        text: s("Hello world"),
        raw_text: s("<|en|> Hello world"),
        language: Some(s("en")),
        timestamp_kind: TimestampKind::Token,
        segments: vec![segment()],
        speaker_segments: speaker_segments(),
        words: words(),
        tokens: tokens(),
        timings: timings(),
    }
}

pub fn vad_segment() -> VadSegment {
    let mut v = VadSegment::default();
    v.start_sample = 512;
    v.end_sample = 16384;
    v
}

pub fn langid_candidates() -> Vec<LangIdCandidate> {
    let mut a = LangIdCandidate::default();
    a.index = 41;
    a.code = s("iw");
    a.name = s("Hebrew");
    a.p = 0.8125;
    a.logit = 4.5;
    let mut b = LangIdCandidate::default();
    b.index = 20;
    b.code = s("en");
    b.name = s("English");
    b.p = 0.1875;
    b.logit = 3.0;
    vec![a, b]
}

/// Every golden, in a stable order. Names are file stems.
pub fn goldens() -> Vec<Golden> {
    let mut vad_result = VadResult::default();
    vad_result.probs = vec![0.125, 0.875, 0.75];
    vad_result.first_frame = 3;
    vad_result.segments = vec![vad_segment()];

    let mut langid_result = LangIdResult::default();
    langid_result.candidates = langid_candidates();
    langid_result.allowed_mass = 0.96875;

    let mut turn_result = TurnResult::default();
    turn_result.p = 0.875;
    turn_result.ended = true;
    turn_result.probs = vec![0.25, 0.875];
    turn_result.first_frame = 2;
    turn_result.speech_probs = vec![0.5, 0.9375];
    turn_result.speech_first_frame = 1;

    vec![
        // Enums: every variant.
        g(
            "Task",
            vec![Task::Transcribe, Task::Translate, Task::Instruct],
        ),
        g(
            "TimestampKind",
            vec![
                TimestampKind::None,
                TimestampKind::Auto,
                TimestampKind::Segment,
                TimestampKind::Word,
                TimestampKind::Token,
            ],
        ),
        g("KvType", vec![KvType::Auto, KvType::F32, KvType::F16]),
        g("Pnc", vec![Pnc::Default, Pnc::Off, Pnc::On]),
        g("Itn", vec![Itn::Default, Itn::Off, Itn::On]),
        g("Diarize", vec![Diarize::Default, Diarize::Off, Diarize::On]),
        g(
            "Backend",
            vec![
                Backend::Auto,
                Backend::Cpu,
                Backend::CpuAccel,
                Backend::Metal,
                Backend::Vulkan,
                Backend::Cuda,
                Backend::Rocm,
            ],
        ),
        g(
            "Feature",
            vec![
                Feature::InitialPrompt,
                Feature::TemperatureFallback,
                Feature::LongForm,
                Feature::Cancellation,
                Feature::Pnc,
                Feature::Itn,
                Feature::Diarization,
                Feature::Vocabulary,
                Feature::ContextPrompt,
                Feature::Instruct,
                Feature::TranscriptPrefix,
            ],
        ),
        g(
            "CommitPolicy",
            vec![
                CommitPolicy::Auto,
                CommitPolicy::OnFinalize,
                CommitPolicy::StablePrefix,
            ],
        ),
        g(
            "StreamState",
            vec![
                StreamState::Idle,
                StreamState::Active,
                StreamState::Finished,
                StreamState::Failed,
            ],
        ),
        g(
            "ExtSlot",
            vec![ExtSlot::Run, ExtSlot::Stream, ExtSlot::DiarizeRun],
        ),
        g(
            "Role",
            vec![
                Role::Asr,
                Role::Diarize,
                Role::LangId,
                Role::Vad,
                Role::Turn,
            ],
        ),
        g("Roles", from_json::<Roles>("21")),
        g(
            "DeviceType",
            vec![
                DeviceType::Cpu,
                DeviceType::Gpu,
                DeviceType::Igpu,
                DeviceType::Accel,
                DeviceType::Unknown,
            ],
        ),
        g(
            "BackendMask",
            vec![
                BackendMask::CPU,
                BackendMask::METAL,
                BackendMask::VULKAN,
                BackendMask::CUDA,
                BackendMask::ROCM,
                BackendMask::OTHER,
                BackendMask::ALL,
                BackendMask::for_backend(Backend::Metal),
            ],
        ),
        g(
            "SortformerPreset",
            vec![
                SortformerPreset::Default,
                SortformerPreset::VeryHighLatency,
                SortformerPreset::HighLatency,
                SortformerPreset::LowLatency,
            ],
        ),
        g(
            "ErrorKind",
            vec![
                ErrorKind::InvalidArgument,
                ErrorKind::NotImplemented,
                ErrorKind::ModelFileNotFound,
                ErrorKind::ModelLoad,
                ErrorKind::OutOfMemory,
                ErrorKind::Backend,
                ErrorKind::Unsupported,
                ErrorKind::BadStructSize,
                ErrorKind::InputTooLong,
                ErrorKind::Aborted,
                ErrorKind::OutputTruncated,
                ErrorKind::OutputRepetition,
                ErrorKind::UnsupportedRole,
                ErrorKind::VersionMismatch,
                ErrorKind::Nul,
                ErrorKind::Busy,
                ErrorKind::InputTooShort,
                ErrorKind::Other,
            ],
        ),
        // Errors.
        g(
            "ErrorReport.partial",
            from_json::<ErrorReport>(
                r#"{"kind":"aborted","message":"operation aborted: run: aborted (status 13)","status":13,"partial":{"text":"Hello","raw_text":"Hello","timestamp_kind":"Segment"}}"#,
            ),
        ),
        g(
            "ErrorReport.plain",
            from_json::<ErrorReport>(
                r#"{"kind":"busy","message":"model busy: a stream is active","status":0,"partial":null}"#,
            ),
        ),
        // Options.
        g(
            "RunOptions",
            RunOptions {
                task: Task::Translate,
                timestamps: TimestampKind::Word,
                pnc: Pnc::On,
                itn: Itn::Off,
                diarize: Diarize::On,
                language: Some(s("en")),
                target_language: Some(s("de")),
                keep_special_tags: true,
                spec_k_drafts: 4,
                family: Some(RunExtension::Whisper(whisper())),
                vocabulary: vec![s("Kubernetes"), s("ggml")],
                prompt: Some(s("A talk about inference.")),
                prefix: Some(s("And so")),
            },
        ),
        g("RunOptions.default", RunOptions::default()),
        g("WhisperRunOptions", whisper()),
        g("WhisperRunOptions.default", WhisperRunOptions::default()),
        g("RunExtension", vec![RunExtension::Whisper(whisper())]),
        g(
            "StreamOptions",
            StreamOptions {
                commit_policy: CommitPolicy::StablePrefix,
                stable_prefix_agreement_n: 4,
                family: Some(stream_extensions().remove(1)),
            },
        ),
        g("StreamOptions.default", StreamOptions::default()),
        g("StreamExtension", stream_extensions()),
        g(
            "SessionOptions",
            SessionOptions {
                n_threads: 8,
                kv_type: KvType::F16,
                n_ctx: 448,
            },
        ),
        g("SessionOptions.default", SessionOptions::default()),
        g(
            "SortformerDiarizeOptions",
            SortformerDiarizeOptions {
                preset: Some(SortformerPreset::LowLatency),
            },
        ),
        g(
            "DiarizeExtension",
            vec![DiarizeExtension::Sortformer(SortformerDiarizeOptions {
                preset: Some(SortformerPreset::HighLatency),
            })],
        ),
        g(
            "DiarizeOptions",
            DiarizeOptions {
                family: Some(DiarizeExtension::Sortformer(
                    SortformerDiarizeOptions::default(),
                )),
            },
        ),
        g("DiarizeOptions.default", DiarizeOptions::default()),
        g(
            "DiarizeSessionOptions",
            DiarizeSessionOptions { n_threads: 4 },
        ),
        g(
            "LangIdOptions",
            LangIdOptions {
                allowed: Some(vec![s("en"), s("he")]),
            },
        ),
        g("LangIdOptions.default", LangIdOptions::default()),
        g(
            "LangIdSessionOptions",
            LangIdSessionOptions { n_threads: 2 },
        ),
        g(
            "VadOptions",
            VadOptions {
                threshold: 0.625,
                neg_threshold: 0.375,
                min_speech_ms: 200,
                min_silence_ms: 80,
                speech_pad_ms: 20,
                max_speech_ms: 15000,
            },
        ),
        g("VadOptions.default", VadOptions::default()),
        g("VadSessionOptions", VadSessionOptions { n_threads: 3 }),
        g("TurnOptions", TurnOptions { threshold: 0.75 }),
        g("TurnOptions.default", TurnOptions::default()),
        g("TurnSessionOptions", TurnSessionOptions { n_threads: 1 }),
        // Results and abilities.
        g("Transcript", transcript()),
        g("Transcript.default", Transcript::default()),
        g("Segment", segment()),
        g("SpeakerSegment", speaker_segments()),
        g("Word", words()),
        g("Token", tokens()),
        g("Timings", timings()),
        g(
            "StreamUpdate",
            StreamUpdate {
                result_changed: true,
                is_final: false,
                revision: 3,
                input_received_ms: 1600,
                audio_committed_ms: 1200,
                buffered_ms: 400,
                committed_changed: true,
                tentative_changed: false,
            },
        ),
        g(
            "StreamText",
            StreamText {
                full: s("Hello wor"),
                committed: s("Hello"),
                tentative: s(" wor"),
            },
        ),
        g(
            "Capabilities",
            Capabilities {
                native_sample_rate: 16000,
                languages: vec![s("en"), s("de")],
                translate_target_languages: vec![s("en")],
                max_timestamp_kind: TimestampKind::Word,
                supports_language_detect: true,
                supports_translate: true,
                supports_streaming: false,
                supports_spec_decode: true,
                max_audio_ms: 30000,
            },
        ),
        g(
            "SessionLimits",
            SessionLimits {
                effective_n_ctx: 448,
                effective_max_audio_ms: 30000,
                max_kv_bytes: 1 << 33,
            },
        ),
        g(
            "VadInfo",
            from_json::<VadInfo>(r#"{"sample_rate":16000,"frame_samples":512}"#),
        ),
        g("VadSegment", vad_segment()),
        g("VadResult", vad_result),
        g(
            "LangIdInfo",
            from_json::<LangIdInfo>(
                r#"{"sample_rate":16000,"n_labels":107,"min_audio_ms":1000,"max_audio_ms":30000}"#,
            ),
        ),
        g("LangIdCandidate", langid_candidates()),
        g("LangIdResult", langid_result),
        g(
            "DiarizeInfo",
            from_json::<DiarizeInfo>(r#"{"sample_rate":16000,"max_speakers":4}"#),
        ),
        g(
            "TurnInfo",
            from_json::<TurnInfo>(
                r#"{"sample_rate":16000,"max_audio_ms":8000,"frame_samples":1280,"has_speech":true,"threshold":0.5}"#,
            ),
        ),
        g("TurnResult", turn_result),
    ]
}
