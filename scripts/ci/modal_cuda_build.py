# Build + smoke the transcribe-cpp-native-cu12 wheel on Modal.
#
#     modal run scripts/ci/modal_cuda_build.py            # build + T4 smoke
#     modal run scripts/ci/modal_cuda_build.py --skip-gpu # packaging only
#
# Why Modal instead of cibuildwheel-on-CI: the manylinux_2_28 + CUDA-toolkit
# environment is a CACHED Modal image layer (installed once, not per run), the
# 16-core build bills by the second, and the same flow ends with a REAL GPU
# runtime smoke on a T4 — the cheapest NVIDIA part Modal offers and exactly
# the oldest arch (sm_75) in the wheel's fatbin list. CI just drives
# `modal run` from a 2-vCPU runner (.github/workflows/python-wheels.yml).
#
# Flow:
#   build_and_check (cpu=16, manylinux_2_28 + cuda image):
#     python -m build bindings/python-native-cu12 -> auditwheel repair with
#     the CUDA excludes (driver + nvidia-wheel-provided runtimes stay OUT of
#     the wheel) -> GPU-less packaging smoke via wheel_smoke.py (provider
#     selected, cuda quietly unavailable without a driver, CPU transcribes,
#     full pytest suite) -> repaired wheel + pure API wheel onto the volume.
#   gpu_smoke (gpu=T4): installs from the volume, transcribes the canary
#     with backend="cuda" on the real GPU.
#
# The wheel lands on the "transcribe-cu12-wheelhouse" volume; CI fetches it
# with `modal volume get` and uploads it as a workflow artifact.

import os
import subprocess
import sys

import modal

app = modal.App("transcribe-cpp-cu12")

VOLUME = modal.Volume.from_name("transcribe-cu12-wheelhouse", create_if_missing=True)
#: Compiler cache across runs: most pushes touch neither ggml nor the CUDA
#: backend, so warm rebuilds skip nearly all nvcc work.
CCACHE_VOLUME = modal.Volume.from_name("transcribe-cu12-sccache", create_if_missing=True)

#: CUDA toolkit pinned per-minor; the nvidia-*-cu12 runtime-wheel floors in
#: bindings/python-native-cu12/pyproject.toml must cover this minor.
CUDA_TOOLKIT = "cuda-toolkit-12-9"

#: sccache: static musl binary (runs on any glibc, unlike EPEL8's ccache 3.x
#: which predates CUDA support). Handles nvcc.
SCCACHE_VERSION = "v0.10.0"
SCCACHE_URL = (
    "https://github.com/mozilla/sccache/releases/download/"
    f"{SCCACHE_VERSION}/sccache-{SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz"
)

# manylinux_2_28 with the CUDA toolkit baked in as an image layer: this is
# the expensive part (multi-GB dnf install) and it happens once, not per run.
build_image = (
    modal.Image.from_registry(
        "quay.io/pypa/manylinux_2_28_x86_64", add_python="3.12"
    )
    .run_commands(
        "dnf install -y dnf-plugins-core",
        "dnf config-manager --add-repo "
        "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo",
        f"dnf install -y {CUDA_TOOLKIT}",
        "dnf clean all",
        "/opt/python/cp312-cp312/bin/pip install --no-cache-dir build",
        f"curl -Ls {SCCACHE_URL} | tar xz --strip-components=1 -C /usr/local/bin"
        " --wildcards '*/sccache' && chmod +x /usr/local/bin/sccache",
    )
    # The repo checkout, attached at runtime (content-addressed: cheap when
    # unchanged). Heavy local dirs never leave the machine.
    .add_local_dir(
        ".",
        "/src",
        ignore=[
            ".git", "build*", "dist*", "wheelhouse*", "models", "canary",
            "tmp", "dumps", "reports", "**/__pycache__", "**/.venv",
            "samples/*", "!samples/jfk.wav",
        ],
    )
)

smoke_image = modal.Image.debian_slim(python_version="3.12")

hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})


def run(cmd, **kwargs) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call([str(c) for c in cmd], **kwargs)


@app.function(
    image=build_image,
    cpu=16.0,
    memory=32768,
    timeout=3600,
    volumes={"/wheels": VOLUME, "/sccache": CCACHE_VOLUME},
    secrets=[hf_secret],
)
def build_and_check() -> str:
    import glob
    import pathlib

    env = os.environ
    env["PATH"] = "/usr/local/cuda/bin:" + env["PATH"]
    env["CUDACXX"] = "/usr/local/cuda/bin/nvcc"
    # Compiler cache (volume-backed): warm runs skip unchanged TUs entirely.
    env["CMAKE_C_COMPILER_LAUNCHER"] = "sccache"
    env["CMAKE_CXX_COMPILER_LAUNCHER"] = "sccache"
    env["CMAKE_CUDA_COMPILER_LAUNCHER"] = "sccache"
    env["SCCACHE_DIR"] = "/sccache"
    env["SCCACHE_CACHE_SIZE"] = "5G"

    pybin = "/opt/python/cp312-cp312/bin"

    # 1. Build the raw wheel (scikit-build-core drives CMake; the full lane
    #    posture lives in the package's pyproject cmake.args).
    run([f"{pybin}/python", "-m", "build", "--wheel",
         "/src/bindings/python-native-cu12", "--outdir", "/tmp/raw"])
    [raw] = glob.glob("/tmp/raw/*.whl")

    # 2. Repair. The CUDA runtime is never vendored: libcuda.so.1 is the
    #    system driver and cudart/cublas come from the nvidia-*-cu12 wheels
    #    the package depends on (preloaded by its prepare() hook).
    for old in pathlib.Path("/wheels").glob("*.whl"):
        old.unlink()
    run(["auditwheel", "repair", raw,
         "--exclude", "libcuda.so.1",
         "--exclude", "libcudart.so*",
         "--exclude", "libcublas.so*",
         "--exclude", "libcublasLt.so*",
         "-w", "/wheels"])
    [repaired] = glob.glob("/wheels/*.whl")
    size_mb = os.path.getsize(repaired) / 1e6
    print(f"repaired wheel: {os.path.basename(repaired)} ({size_mb:.0f} MB)")

    # 3. The pure API wheel rides along so the smokes can install the pair
    #    from the volume alone.
    run([f"{pybin}/pip", "wheel", "--no-deps", "-q",
         "/src/bindings/python", "-w", "/wheels"])

    run(["sccache", "--show-stats"])
    VOLUME.commit()
    CCACHE_VOLUME.commit()
    return os.path.basename(repaired)


@app.function(
    image=build_image,
    cpu=4.0,
    timeout=1800,
    volumes={"/wheels": VOLUME},
    secrets=[hf_secret],
)
def packaging_check() -> str:
    """GPU-less smoke on the REPAIRED wheel from the volume: cu12 provider is
    selected, the cuda module quietly fails without a driver (no libcuda —
    the same degradation contract as Vulkan), CPU transcribes, and the full
    pytest suite runs (canary via HF_TOKEN). Separate from the build so smoke
    iteration never repeats the 20-minute compile (`--skip-build`)."""
    pybin = "/opt/python/cp312-cp312/bin"
    run([f"{pybin}/python", "-m", "venv", "/tmp/tv"])
    run(["/tmp/tv/bin/pip", "install", "-q", "--find-links", "/wheels",
         "transcribe-cpp-native-cu12", "pytest>=7", "numpy", "huggingface_hub"])
    smoke_env = dict(os.environ)
    smoke_env["TRANSCRIBE_SMOKE_PROVIDER"] = "transcribe-cpp-native-cu12"
    # The default provider is deliberately absent here (its pin can't
    # resolve pre-release); wheel_smoke installs the API package dep-free.
    smoke_env["TRANSCRIBE_SMOKE_PIP_NO_DEPS"] = "1"
    smoke_env["CI"] = "1"
    print("+ wheel_smoke.py (GPU-less packaging check)", flush=True)
    subprocess.check_call(
        ["/tmp/tv/bin/python", "/src/scripts/ci/wheel_smoke.py", "/src"],
        env=smoke_env,
    )
    return "packaging check ok"


@app.function(
    image=smoke_image,
    gpu="T4",
    timeout=900,
    volumes={"/wheels": VOLUME},
    secrets=[hf_secret],
)
def gpu_smoke() -> str:
    import array
    import wave

    # nvidia runtime wheels come from PyPI; the transcribe pair from /wheels.
    # The API package installs dep-free: its hard pin names the DEFAULT
    # provider, which doesn't exist on an index pre-release.
    run([sys.executable, "-m", "pip", "install", "-q", "--find-links", "/wheels",
         "transcribe-cpp-native-cu12", "huggingface_hub"])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "--find-links", "/wheels", "transcribe-cpp"])

    from huggingface_hub import hf_hub_download

    model = hf_hub_download(
        "handy-computer/whisper-tiny-gguf", "whisper-tiny-Q5_K_M.gguf"
    )

    import transcribe_cpp as t

    devs = [(d.name, d.kind) for d in t.backends()]
    print("provider:", t.native_provider())
    print("devices: ", devs)
    assert t.native_provider() == "transcribe-cpp-native-cu12"
    assert t.backend_available("cuda"), devs

    with wave.open("/wheels/assets/jfk.wav", "rb") as w:
        pcm16 = array.array("h")
        pcm16.frombytes(w.readframes(w.getnframes()))
    pcm = array.array("f", (s / 32768.0 for s in pcm16))
    with t.Model(model, backend="cuda") as m:
        print("model backend:", m.backend)
        with m.session() as s:
            text = s.run(pcm).text
    print("text:", text.strip())
    assert "country" in text.lower(), text
    return f"ok: transcribed on {devs[0][0]} (cuda)"


@app.function(image=build_image, volumes={"/wheels": VOLUME}, timeout=300)
def stage_assets() -> None:
    """Copy the smoke audio onto the volume for the GPU function."""
    import shutil

    os.makedirs("/wheels/assets", exist_ok=True)
    shutil.copy("/src/samples/jfk.wav", "/wheels/assets/jfk.wav")
    VOLUME.commit()


@app.local_entrypoint()
def main(skip_gpu: bool = False, skip_build: bool = False) -> None:
    if skip_build:
        print("(--skip-build: reusing the wheel already on the volume)")
    else:
        wheel = build_and_check.remote()
        print(f"build ok: {wheel}")
    stage_assets.remote()
    print(packaging_check.remote())
    if skip_gpu:
        print("(--skip-gpu: skipping the T4 runtime smoke)")
        return
    print(gpu_smoke.remote())
