#!/usr/bin/env python3
"""Co-import smoke: prove transcribe-cpp coexists with numpy/torch in-process.

The wheel-hygiene posture (no vendored OpenMP/BLAS) exists exactly so the
native library can share a process with PyTorch/NumPy/MKL. This proves it by
transcribing real audio with the framework loaded, in both import orders —
runtime collisions (duplicate OpenMP runtimes, BLAS symbol clashes) typically
explode at first compute, not at import.

    python co_import_smoke.py <framework> <repo-dir>     # framework: numpy|torch

Runs two child processes:
  framework-first: import <framework>  -> import transcribe_cpp -> transcribe
  library-first:   import transcribe_cpp -> transcribe -> import <framework>
                   -> transcribe again (catches late-loaded runtime clashes)

A real model comes from TRANSCRIBE_SMOKE_MODEL or the HF canary (HF_TOKEN).
Without either, the smoke still co-imports and initializes backends but says
loudly that no transcription ran.
"""

import array
import os
import subprocess
import sys
import wave
from pathlib import Path


def resolve_model() -> "str | None":
    if os.environ.get("TRANSCRIBE_SMOKE_MODEL"):
        return os.environ["TRANSCRIBE_SMOKE_MODEL"]
    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            "handy-computer/whisper-tiny-gguf", "whisper-tiny-Q5_K_M.gguf"
        )
    return None


def load_jfk(repo: Path) -> "array.array":
    with wave.open(str(repo / "samples" / "jfk.wav"), "rb") as w:
        pcm16 = array.array("h")
        pcm16.frombytes(w.readframes(w.getnframes()))
    return array.array("f", (s / 32768.0 for s in pcm16))


def transcribe_once(model: str, repo: Path, tag: str) -> None:
    import transcribe_cpp as t

    with t.Model(model) as m:
        with m.session() as s:
            text = s.run(load_jfk(repo)).text
    print(f"[{tag}] text:", text.strip())
    assert "country" in text.lower(), text


def child(order: str, framework: str, repo: Path) -> None:
    model = resolve_model()

    if order == "framework-first":
        fw = __import__(framework)
        print(f"[{order}] {framework} {fw.__version__} imported first")
        import transcribe_cpp as t

        print(f"[{order}] devices:", [(d.name, d.kind) for d in t.backends()])
        if model:
            transcribe_once(model, repo, order)
    else:  # library-first
        import transcribe_cpp as t

        print(f"[{order}] devices:", [(d.name, d.kind) for d in t.backends()])
        if model:
            transcribe_once(model, repo, f"{order}/pre")
        fw = __import__(framework)
        print(f"[{order}] {framework} {fw.__version__} imported after")
        if model:
            transcribe_once(model, repo, f"{order}/post")

    if not model:
        print(f"[{order}] !! no model available (no TRANSCRIBE_SMOKE_MODEL / "
              "HF_TOKEN) — co-import only, no transcription ran")


def main() -> int:
    if len(sys.argv) == 4:  # child invocation: <framework> <repo> <order>
        child(sys.argv[3], sys.argv[1], Path(sys.argv[2]).resolve())
        return 0

    framework, repo = sys.argv[1], sys.argv[2]
    for order in ("framework-first", "library-first"):
        print(f"=== {framework} / {order} ===", flush=True)
        subprocess.check_call(
            [sys.executable, __file__, framework, repo, order]
        )
    print(f"OK: {framework} co-import smoke passed (both orders)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
