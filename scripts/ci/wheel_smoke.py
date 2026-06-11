#!/usr/bin/env python3
"""cibuildwheel test command for the transcribe-cpp-native provider wheel.

cibuildwheel installs the *repaired* native wheel into a fresh test venv and
runs this script (see [tool.cibuildwheel] in the repo-root pyproject.toml), so
everything here exercises the artifact that would ship, not the build output.

    python wheel_smoke.py <project-dir>

Steps:
  1. pip-install the pure API package from the repo — its
     transcribe-cpp-native==X.Y.Z.* pin must resolve against the installed
     provider wheel (this is the resolver-level base-version contract).
  2. Fetch the canary GGUFs when HF_TOKEN is present (skipped cleanly on
     forks), wiring them into the pytest suite's TRANSCRIBE_SMOKE_* env.
  3. Assert provider identity and per-platform backend posture.
  4. Run the full pytest suite against the installed pair.

Plain `python`/`pip` on purpose: this runs inside cibuildwheel's managed test
venv where uv does not exist.
"""

import os
import subprocess
import sys
from pathlib import Path


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(list(cmd))


def main() -> int:
    project = Path(sys.argv[1]).resolve()

    # 1. The API package. The native provider wheel is already installed, so
    #    the == pin resolves locally without an index round-trip for it.
    run(sys.executable, "-m", "pip", "install", str(project / "bindings" / "python"))

    # 2. Canary models (private HF repos; needs HF_TOKEN). Pre-set
    #    TRANSCRIBE_SMOKE_* env (e.g. local runs against on-disk models) wins.
    if os.environ.get("HF_TOKEN") and not os.environ.get("TRANSCRIBE_SMOKE_MODEL"):
        from huggingface_hub import hf_hub_download

        os.environ["TRANSCRIBE_SMOKE_MODEL"] = hf_hub_download(
            "handy-computer/whisper-tiny-gguf", "whisper-tiny-Q5_K_M.gguf"
        )
        os.environ["TRANSCRIBE_SMOKE_STREAMING_MODEL"] = hf_hub_download(
            "handy-computer/moonshine-streaming-tiny-gguf",
            "moonshine-streaming-tiny-Q8_0.gguf",
        )

    # 3. Provider identity + backend posture of the repaired artifact.
    import transcribe_cpp as t

    print("provider:    ", t.native_provider())
    print("library:     ", t.library_path())
    print("devices:     ", [(d.name, d.kind) for d in t.backends()])
    assert t.native_provider() == "transcribe-cpp-native", t.native_provider()
    assert "transcribe_cpp_native" in t.library_path(), t.library_path()
    assert t.backend_available("cpu")
    if sys.platform == "darwin":
        assert t.backend_available("metal"), "Metal must be available on arm64 CI"
    elif os.environ.get("CI"):
        # Linux containers / Windows runners have no GPU and no Vulkan driver:
        # the bundled Vulkan module must degrade quietly to CPU (whether the
        # loader is present-but-deviceless or absent entirely).
        assert not t.backend_available("vulkan"), (
            "vulkan unexpectedly available on a GPU-less CI runner"
        )

    # 4. Metal compute probe (darwin). Runner GPUs are not all trustworthy:
    #    GitHub-hosted macOS runners expose an "Apple Paravirtual device"
    #    whose Metal compute is broken (whisper decodes to garbage with
    #    runaway repetition; llama.cpp's CI disables Metal there too). On a
    #    lane whose runner has a known-broken GPU, set
    #    TRANSCRIBE_SMOKE_BACKEND=cpu: the probe is skipped and the model
    #    tests run CPU-steered. Anywhere else this transcribes on the GPU
    #    and fails loud if the device lies.
    smoke_backend = os.environ.get("TRANSCRIBE_SMOKE_BACKEND")
    model = os.environ.get("TRANSCRIBE_SMOKE_MODEL")
    if sys.platform == "darwin" and model and not smoke_backend:
        import array
        import wave

        with wave.open(str(project / "samples" / "jfk.wav"), "rb") as w:
            pcm16 = array.array("h")
            pcm16.frombytes(w.readframes(w.getnframes()))
        pcm = array.array("f", (s / 32768.0 for s in pcm16))
        with t.Model(model, backend="metal") as m, m.session() as s:
            text = s.run(pcm).text
        print("metal probe:  ", text.strip())
        assert "country" in text.lower(), (
            f"Metal compute produced garbage on this runner: {text[:120]!r}. "
            "If this runner's GPU is known-broken (e.g. GitHub's Apple "
            "Paravirtual device), set TRANSCRIBE_SMOKE_BACKEND=cpu for the lane."
        )

    # 5. Full suite against the installed pair (model tests run when the
    #    canary env above is set; otherwise they skip and say so via -rs).
    env = os.environ.copy()
    if smoke_backend:
        env["TRANSCRIBE_BACKEND"] = smoke_backend
    return subprocess.call(
        [sys.executable, "-m", "pytest",
         str(project / "bindings" / "python" / "tests"), "-q", "-rs"],
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
