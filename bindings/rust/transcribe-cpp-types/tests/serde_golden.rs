//! Same wire format as transcribe-cpp 0.5.0: every plain-data type serializes
//! to the JSON (and postcard bytes) 0.5.0 wrote for the same value, and reads
//! those bytes back. The goldens under `tests/golden/` were written by 0.5.0
//! off crates.io (see the README there); never regenerate them from this tree.

#![cfg(feature = "serde")]

use std::path::{Path, PathBuf};

use transcribe_cpp_types as tc;

#[path = "golden/values.rs"]
mod values;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn read(kind: &str, name: &str, ext: &str) -> String {
    let path = golden_dir().join(kind).join(format!("{name}.{ext}"));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .trim_end()
        .to_string()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn serializes_like_0_5_0() {
    let goldens = values::goldens();
    assert_eq!(goldens.len(), 63, "a golden was added or dropped");
    let mut failures = Vec::new();
    for g in &goldens {
        let json = read("json", g.name, "json");
        if g.json != json {
            failures.push(format!(
                "{} JSON\n  0.5.0: {json}\n  now:   {}",
                g.name, g.json
            ));
        }
        let pc = read("postcard", g.name, "hex");
        if hex(&g.postcard) != pc {
            failures.push(format!(
                "{} postcard\n  0.5.0: {pc}\n  now:   {}",
                g.name,
                hex(&g.postcard)
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn reads_what_0_5_0_wrote() {
    for g in values::goldens() {
        let json = read("json", g.name, "json");
        assert_eq!((g.json_again)(&json), json, "{} JSON", g.name);
        let pc = unhex(&read("postcard", g.name, "hex"));
        assert_eq!((g.postcard_again)(&pc), pc, "{} postcard", g.name);
    }
}

/// The values the goldens build from JSON (0.5.0 had no way to construct these
/// `#[non_exhaustive]` types outside the crate) equal the ones this crate's
/// public constructors build.
#[test]
fn constructors_build_the_golden_values() {
    let json = |name: &str| read("json", name, "json");

    let mut report = tc::ErrorReport::new(
        tc::ErrorKind::Aborted,
        "operation aborted: run: aborted (status 13)",
    );
    report.status = 13;
    report.partial = Some(tc::Transcript {
        text: "Hello".into(),
        raw_text: "Hello".into(),
        timestamp_kind: tc::TimestampKind::Segment,
        ..Default::default()
    });
    assert_eq!(
        serde_json::to_string(&report).unwrap(),
        json("ErrorReport.partial")
    );

    let busy = tc::ErrorReport::new(tc::ErrorKind::Busy, "model busy: a stream is active");
    assert_eq!(
        serde_json::to_string(&busy).unwrap(),
        json("ErrorReport.plain")
    );

    let roles = tc::Roles::from_bits(21);
    assert_eq!(serde_json::to_string(&roles).unwrap(), json("Roles"));

    let mut vad = tc::VadInfo::default();
    vad.sample_rate = 16000;
    vad.frame_samples = 512;
    assert_eq!(serde_json::to_string(&vad).unwrap(), json("VadInfo"));

    let mut lid = tc::LangIdInfo::default();
    lid.sample_rate = 16000;
    lid.n_labels = 107;
    lid.min_audio_ms = 1000;
    lid.max_audio_ms = 30000;
    assert_eq!(serde_json::to_string(&lid).unwrap(), json("LangIdInfo"));

    let mut dia = tc::DiarizeInfo::default();
    dia.sample_rate = 16000;
    dia.max_speakers = 4;
    assert_eq!(serde_json::to_string(&dia).unwrap(), json("DiarizeInfo"));

    let mut turn = tc::TurnInfo::default();
    turn.sample_rate = 16000;
    turn.max_audio_ms = 8000;
    turn.frame_samples = 1280;
    turn.has_speech = true;
    turn.threshold = 0.5;
    assert_eq!(serde_json::to_string(&turn).unwrap(), json("TurnInfo"));
}

/// Every golden on disk is checked (none orphaned by a rename).
#[test]
fn no_orphan_goldens() {
    let names: Vec<&str> = values::goldens().iter().map(|g| g.name).collect();
    for kind in ["json", "postcard"] {
        for entry in std::fs::read_dir(golden_dir().join(kind)).unwrap() {
            let path = entry.unwrap().path();
            let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
            // `Error.*` belongs to transcribe-cpp's tests/serde.rs.
            assert!(
                stem.starts_with("Error.") || names.contains(&stem.as_str()),
                "orphan golden {}",
                path.display()
            );
        }
    }
}
