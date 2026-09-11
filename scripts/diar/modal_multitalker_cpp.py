#!/usr/bin/env python3
"""
modal_multitalker_cpp.py - run the C++ multitalker bundle over the AMI
acceptance manifest on Modal CUDA and emit SegLST hypotheses.

The long AMI meetings need more attention-buffer memory than the local
16 GB M4 can give (a ~50-min meeting wants a ~5.5 GB F32 mask; a local
attempt hard-crashed the machine), so the C++ side runs on an L40S with
a CUDA build. The image bakes a TRANSCRIBE_CUDA=ON build of the repo
source; bundle GGUF + audio come from the `transcribe-ami-cpp` volume:

    modal volume create transcribe-ami-cpp
    modal volume put transcribe-ami-cpp \
        models/multitalker-parakeet-streaming-0.6b-v1/bundle/multitalker-parakeet-streaming-0.6b-v1-F32.gguf \
        /model/bundle-F32.gguf
    modal volume put transcribe-ami-cpp samples/diar/ami-ihm-test /audio

One container per (meeting, mode) cell, batch size 1 (multitalker is a
transcribe_run path). Mode is selected via TRANSCRIBE_MULTITALKER_MODE,
same contract as run_cpp_multitalker.py. Hypotheses land in
reports/cpwer/cpp-<mode>/<meeting>.seglst.json for score_cpwer.py.

Usage:
    modal run scripts/diar/modal_multitalker_cpp.py --modes both
    modal run scripts/diar/modal_multitalker_cpp.py --modes kernel --meetings IS1009a
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent.parent

app = modal.App("transcribe-multitalker-cpp")
vol = modal.Volume.from_name("transcribe-ami-cpp")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "git", "ninja-build")
    .pip_install("cmake")
    .add_local_file(REPO / "CMakeLists.txt", "/repo/CMakeLists.txt", copy=True)
    .add_local_dir(REPO / "include", "/repo/include", copy=True)
    .add_local_dir(REPO / "cmake", "/repo/cmake", copy=True)
    .add_local_dir(REPO / "ggml", "/repo/ggml", copy=True)
    .add_local_dir(REPO / "src", "/repo/src", copy=True)
    .add_local_dir(REPO / "examples", "/repo/examples", copy=True)
    .run_commands(
        "cmake -S /repo -B /repo/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
        "-DTRANSCRIBE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 "
        "-DTRANSCRIBE_BUILD_TESTS=OFF -DTRANSCRIBE_BUILD_EXAMPLES=ON",
        "cmake --build /repo/build -j 16 --target transcribe-cli",
    )
)


@app.function(image=image, gpu="L40S", volumes={"/data": vol}, timeout=3600)
def run_meeting(meeting_wav: str, mode: str, ref_bg_compat: bool = False) -> dict:
    mid = Path(meeting_wav).name.split(".")[0]
    batch_list = "/tmp/batch.list"
    Path(batch_list).write_text(f"/data/audio/{meeting_wav}\n")

    env = dict(os.environ)
    if ref_bg_compat:
        env["TRANSCRIBE_MULTITALKER_REF_BG_COMPAT"] = "1"
    env["TRANSCRIBE_MULTITALKER_MODE"] = mode

    cmd = [
        "/repo/build/bin/transcribe-cli", "-q", "--diarize", "--backend", "cuda",
        "--batch", batch_list, "--batch-jsonl",
        "-m", "/data/model/bundle-F32.gguf",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return {"id": mid, "mode": mode, "error": f"exit {proc.returncode}: {proc.stderr[-2000:]}"}

    row = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cand.get("segments"):
                row = cand
    if row is None:
        return {"id": mid, "mode": mode, "error": "no JSONL row with segments"}

    seglst = []
    for seg in row["segments"]:
        text = (seg.get("text") or "").strip()
        if not text or seg.get("speaker_id", 0) <= 0:
            continue
        seglst.append(
            {
                "session_id": mid,
                "speaker": f"S{seg['speaker_id']}",
                "start_time": seg["t0_ms"] / 1000.0,
                "end_time": seg["t1_ms"] / 1000.0,
                "words": text,
            }
        )
    return {"id": mid, "mode": mode, "seglst": seglst}


@app.local_entrypoint()
def main(modes: str = "both", meetings: str = "", ref_bg_compat: bool = False):
    manifest = REPO / "samples" / "diar" / "ami-ihm-test.manifest.jsonl"
    entries = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    want = {m.strip() for m in meetings.split(",") if m.strip()}
    mode_list = ["masked", "kernel"] if modes == "both" else [modes]

    cells = []
    for e in entries:
        mid = e["id"].split(".")[0]
        if want and mid not in want:
            continue
        for mode in mode_list:
            cells.append((Path(e["audio"]).name, mode, ref_bg_compat))

    n_ok = 0
    for res in run_meeting.starmap(cells, order_outputs=False):
        if res.get("error"):
            print(f"FAIL {res['id']} [{res['mode']}]: {res['error']}")
            continue
        suffix = "-refbug" if ref_bg_compat else ""
        out_dir = REPO / "reports" / "cpwer" / f"cpp-{res['mode']}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{res['id']}.seglst.json"
        out.write_text(json.dumps(res["seglst"], indent=1) + "\n")
        n_ok += 1
        print(f"ok   {res['id']} [{res['mode']}]: {len(res['seglst'])} segments -> {out.relative_to(REPO)}")

    print(f"{n_ok}/{len(cells)} cells succeeded")
