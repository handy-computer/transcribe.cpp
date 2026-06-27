#!/usr/bin/env python3
"""Modal host for the VibeVoice-ASR reference dumper.

The 9B model can't run on the local 16 GB M4, so the canonical dumper
(scripts/dump_reference_vibevoice_author.py) executes in a Modal L40S container
and the f32 tensor artifacts are tarred back to the laptop, landing under
build/validate/vibevoice/vibevoice-asr/.

    uv run --project scripts/envs/qwen3_asr modal run scripts/_vibevoice_modal_dump.py
"""
import io
import tarfile
from pathlib import Path

import modal

VIBEVOICE_SHA = "303b2833e01cff4578ec278bbfe536da54bd19fe"
REPO = Path(__file__).resolve().parents[1]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "ffmpeg", "libsndfile1", "ca-certificates")
    .pip_install(
        "torch", "transformers==4.51.3", "accelerate>=1.6.0", "diffusers>=0.30",
        "librosa>=0.10", "soundfile>=0.12", "ml-collections>=0.1.1",
        "absl-py>=2.0", "einops>=0.7", "numpy>=1.26", "scipy>=1.11",
        "huggingface_hub>=0.30",
    )
    .run_commands(
        "git clone https://github.com/microsoft/VibeVoice /opt/VibeVoice",
        f"cd /opt/VibeVoice && git checkout {VIBEVOICE_SHA}",
    )
    .add_local_file("samples/jfk.wav", "/root/jfk.wav")
    .add_local_file("scripts/dump_reference_vibevoice_author.py",
                    "/root/scripts/dump_reference_vibevoice_author.py")
    .add_local_file("scripts/lib/__init__.py", "/root/scripts/lib/__init__.py")
    .add_local_file("scripts/lib/ref_dump.py", "/root/scripts/lib/ref_dump.py")
)

hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App("vibevoice-dump")


@app.function(image=image, gpu="L40S", timeout=3000,
              volumes={"/root/.cache/huggingface": hf_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def dump() -> bytes:
    import os
    import subprocess

    env = {**os.environ, "VIBEVOICE_SRC": "/opt/VibeVoice",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    cmd = [
        "python", "/root/scripts/dump_reference_vibevoice_author.py", "dump",
        "--model", "microsoft/VibeVoice-ASR",
        "--audio", "/root/jfk.wav",
        "--out", "/root/out/jfk/ref",
        "--device", "cuda",
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("/root/out", arcname=".")
    return buf.getvalue()


@app.local_entrypoint()
def main():
    data = dump.remote()
    dest = REPO / "build" / "validate" / "vibevoice" / "vibevoice-asr"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest)
    print(f"extracted {len(data)} bytes -> {dest}")
    for p in sorted(dest.rglob("*.json")):
        print(" ", p.relative_to(REPO))
