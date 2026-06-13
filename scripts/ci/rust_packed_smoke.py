#!/usr/bin/env python3
"""Build the shipped `transcribe-cpp-sys` tarball and prove it transcribes.

    python3 scripts/ci/rust_packed_smoke.py [--model M.gguf] [--audio A.wav]

The Rust twin of wheel_smoke.py and the sdist compile-and-transcribe gate
(requirements §4, "test the shipped artifact"): never the build tree, always
the packed crate. `transcribe-cpp-sys` is the load-bearing artifact — it carries
the whole C++ tree and builds libtranscribe from source via build.rs — so this
packs it, extracts it, and builds a throwaway consumer that compiles the native
library from NOTHING but the packed sources and then transcribes through the
real `transcribe-cpp` safe API. A tarball missing a vendored source, a broken
build.rs, or a bad link manifest is a red gate here, not a user's first build.

Why only the sys crate is packed: the safe `transcribe-cpp` crate depends on
`transcribe-cpp-sys = "0.0.1"`, and pre-publish the registry only has the 0.0.0
name-reservation placeholder, so `cargo package -p transcribe-cpp` cannot
resolve it (this is the inherent sys/safe publish ordering — sys lands first).
We therefore STAGE the safe crate (its source with the path stripped from the
sys dep, exactly what `cargo package` would emit) and redirect its sys dep to
the packed tarball via `[patch.crates-io]`. The safe crate's own registry
packaging is verified at release time, once the published sys crate exists.

With no model (`--model` / `$TRANSCRIBE_SMOKE_MODEL` absent) it degrades to an
install-and-link check: the consumer just prints `transcribe_cpp::version()`.

Stdlib only. Exit 0 on success, non-zero on any failure.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SAFE_CRATE_DIR = REPO / "bindings" / "rust" / "transcribe-cpp"

CONSUMER_MAIN = """\
//! Throwaway consumer of the PACKED transcribe-cpp-sys crate — what
//! `cargo add transcribe-cpp` + a five-line transcription does, but with the
//! native library built from the shipped tarball's vendored sources. Generated
//! by scripts/ci/rust_packed_smoke.py.
use transcribe_cpp::{Model, RunOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let version = transcribe_cpp::version();
    println!("packed transcribe-cpp {version}");
    assert!(!version.is_empty(), "empty version from the packed crate");

    let (Ok(model), Ok(audio)) = (std::env::var("SMOKE_MODEL"), std::env::var("SMOKE_AUDIO"))
    else {
        println!("no model -> install + link check only (no transcription)");
        return Ok(());
    };

    let mut reader = hound::WavReader::open(&audio)?;
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<Result<_, _>>()?;

    let mut session = Model::load(&model)?.session()?;
    let text = session.run(&pcm, &RunOptions::default())?.text;
    println!("transcript: {}", text.trim());
    assert!(
        text.to_lowercase().contains("country"),
        "unexpected transcript: {text:?}"
    );
    println!("ok: native lib built from the packed sys tarball and transcribed");
    Ok(())
}
"""


def pack_sys() -> Path:
    """Pack transcribe-cpp-sys (packaging only) and return its .crate path."""
    subprocess.run(
        ["cargo", "package", "--no-verify", "--allow-dirty", "-p", "transcribe-cpp-sys"],
        cwd=REPO,
        check=True,
    )
    crates = sorted((REPO / "target" / "package").glob("transcribe-cpp-sys-*.crate"))
    if not crates:
        sys.exit("packed-smoke FAILED: no transcribe-cpp-sys .crate produced")
    return crates[-1]


def extract(crate: Path, dest: Path) -> Path:
    """Extract a .crate (a .tar.gz with one top-level pkg-version/ dir)."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(crate) as tar:
        tar.extractall(dest, filter="data")  # filter arg: Python >= 3.12
    return next(d for d in dest.iterdir() if d.is_dir())


def stage_safe_crate(dest: Path) -> Path:
    """Copy the safe crate and strip the `path` from its sys dependency so the
    consumer's [patch.crates-io] can redirect it to the packed tarball — exactly
    the transform `cargo package` applies to a path+version dependency."""
    dest.mkdir(parents=True, exist_ok=True)
    for item in ["src", "build.rs", "Cargo.toml", "README.md", "LICENSE"]:
        src = SAFE_CRATE_DIR / item
        if src.is_dir():
            shutil.copytree(src, dest / item)
        elif src.is_file():
            shutil.copy2(src, dest / item)
    cargo_toml = dest / "Cargo.toml"
    text = cargo_toml.read_text()
    # `transcribe-cpp-sys = { version = "0.0.1", path = "../../.." }`
    #   -> `transcribe-cpp-sys = { version = "0.0.1" }`
    patched = re.sub(
        r'(transcribe-cpp-sys\s*=\s*\{[^}]*?),\s*path\s*=\s*"[^"]*"',
        r"\1",
        text,
    )
    if patched == text:
        sys.exit("packed-smoke FAILED: could not strip the sys path dep from the safe Cargo.toml")
    cargo_toml.write_text(patched)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("TRANSCRIBE_SMOKE_MODEL"))
    ap.add_argument(
        "--audio",
        default=os.environ.get("TRANSCRIBE_SMOKE_AUDIO") or str(REPO / "samples/jfk.wav"),
    )
    args = ap.parse_args()

    sys_crate = pack_sys()
    print(f"packed: {sys_crate.name}")

    staging = Path(tempfile.mkdtemp(prefix="rust-packed-smoke-"))
    sys_dir = extract(sys_crate, staging / "sys")
    safe_dir = stage_safe_crate(staging / "safe")

    consumer = staging / "consumer"
    (consumer / "src").mkdir(parents=True)
    (consumer / "Cargo.toml").write_text(
        textwrap.dedent(
            f"""\
            [package]
            name = "packed-smoke"
            version = "0.0.0"
            edition = "2021"

            [dependencies]
            transcribe-cpp = {{ path = "{safe_dir.as_posix()}" }}
            hound = "3"

            # The sys crate is not on the registry at this version yet; redirect
            # the safe crate's registry dep to the PACKED tarball under test.
            [patch.crates-io]
            transcribe-cpp-sys = {{ path = "{sys_dir.as_posix()}" }}

            [workspace]
            """
        )
    )
    (consumer / "src" / "main.rs").write_text(CONSUMER_MAIN)

    env = os.environ.copy()
    model = args.model
    if model and Path(model).is_file() and Path(args.audio).is_file():
        env["SMOKE_MODEL"] = model
        env["SMOKE_AUDIO"] = args.audio
        print(f"transcribing {Path(args.audio).name} with {Path(model).name}")
    else:
        env.pop("SMOKE_MODEL", None)
        env.pop("SMOKE_AUDIO", None)
        print("no model on disk -> install + link check only")

    # The native lib is always built in Release by build.rs; the consumer's own
    # (debug) build is just glue, so no --release needed for a fast smoke.
    subprocess.run(
        ["cargo", "run", "--manifest-path", str(consumer / "Cargo.toml")],
        check=True,
        env=env,
    )
    print("rust packed smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
