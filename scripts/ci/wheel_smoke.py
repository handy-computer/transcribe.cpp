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
    #    TRANSCRIBE_SMOKE_PIP_NO_DEPS=1 skips dependency resolution for envs
    #    where the DEFAULT provider is deliberately absent (the cu12-only
    #    packaging check: its pin can't resolve until release artifacts
    #    exist on an index; the dual-provider interplay is TestPyPI-
    #    rehearsal territory).
    pip_args = ["-m", "pip", "install"]
    if os.environ.get("TRANSCRIBE_SMOKE_PIP_NO_DEPS"):
        pip_args.append("--no-deps")
    run(sys.executable, *pip_args, str(project / "bindings" / "python"))

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
    #    TRANSCRIBE_SMOKE_PROVIDER names the provider under test (default:
    #    the default provider; the cu12 lane sets its own name).
    import transcribe_cpp as t

    expected_provider = os.environ.get(
        "TRANSCRIBE_SMOKE_PROVIDER", "transcribe-cpp-native"
    )
    print("provider:    ", t.native_provider())
    print("library:     ", t.library_path())
    print("devices:     ", [(d.name, d.kind) for d in t.backends()])
    assert t.native_provider() == expected_provider, t.native_provider()
    assert "transcribe_cpp_native" in t.library_path(), t.library_path()
    assert t.backend_available("cpu")

    # 3b. DECLARED artifact capabilities (the descriptor's build-time stamp),
    #     not runtime availability. Every availability probe below answers
    #     "False" identically for "no device on this runner" and "backend not
    #     compiled into the wheel" — so a wheel built with the wrong lane
    #     posture sails through green. Observed 2026-06-11: an unanchored
    #     scikit-build override regex ("cpu" matching "cpu-vulkan") shipped
    #     Vulkan-less linux/windows/arm64 wheels through an all-green matrix;
    #     only clean-install tier B (a real lavapipe transcription) caught it.
    #     TRANSCRIBE_WHEEL_LANE is present in cibuildwheel's build AND test
    #     phases; sdist/dev installs have no lane and skip the equality check.
    from importlib.metadata import entry_points

    declared = None
    for ep in entry_points(group="transcribe_cpp.native"):
        desc = ep.load()
        if callable(desc):  # the contract: a zero-arg callable returning the descriptor
            desc = desc()
        get = desc.get if isinstance(desc, dict) else lambda k: getattr(desc, k, None)
        if get("name") == expected_provider:
            declared = set(get("backends") or ())
            break
    assert declared is not None, f"{expected_provider} not discoverable via entry points"
    print("declared:    ", sorted(declared))
    lane = os.environ.get("TRANSCRIBE_WHEEL_LANE")
    expected_caps = None
    if expected_provider == "transcribe-cpp-native-cu12":
        expected_caps = {"cuda", "cpu"}
    elif lane:
        expected_caps = {
            "cpu-vulkan": {"vulkan", "cpu"},
            "metal": {"metal", "cpu"},
            "cpu": {"cpu"},
        }.get(lane)
    if expected_caps is not None:
        assert declared == expected_caps, (
            f"wheel declares backends {sorted(declared)} but lane "
            f"{lane or expected_provider!r} requires {sorted(expected_caps)} — "
            "the wheel was built with the wrong lane posture"
        )
    if sys.platform == "darwin":
        import platform as _platform

        if _platform.machine() == "arm64":
            # Apple Silicon ships Metal embedded in the wheel.
            assert t.backend_available("metal"), "Metal must be available on arm64 CI"
        else:
            # Intel macOS ships a CPU-only wheel (no Metal — Intel-Mac GPUs are
            # out of scope; Metal / tuned CPU is reachable via the sdist). The
            # bundled artifact must therefore expose only CPU.
            for accel in ("metal", "vulkan", "cuda"):
                assert not t.backend_available(accel), (
                    f"{accel} unexpectedly available in the CPU-only x86 macOS wheel"
                )
    elif os.environ.get("CI"):
        # Linux containers / Windows runners have no GPU (no Vulkan driver,
        # no NVIDIA driver): the bundled accelerator modules must degrade
        # quietly to CPU (loader/driver present-but-deviceless or absent).
        for accel in ("vulkan", "cuda"):
            assert not t.backend_available(accel), (
                f"{accel} unexpectedly available on a GPU-less CI runner"
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
