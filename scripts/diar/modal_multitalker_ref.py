"""
modal_multitalker_ref.py - NeMo multitalker baseline (ASR + streaming
Sortformer) on Modal L40S for the AMI-IHM cpWER acceptance run.

Fans out one container per (meeting, supervision-mode) cell, each running
scripts/dump_reference_multitalker_nemo.py (--device cuda --light) over
one staged AMI meeting, and collects the SegLST hypotheses locally under
reports/cpwer/nemo-<mode>/<meeting>.seglst.json for scoring with
scripts/diar/score_cpwer.py.

Batch size is 1 by design: one meeting per container, the pipeline's
audio_file mode.

Usage (from the repo root):
  # smoke: one meeting, both modes
  modal run scripts/diar/modal_multitalker_ref.py --meetings ES2004a --modes both

  # full acceptance sweep (meetings from the staged manifest)
  modal run scripts/diar/modal_multitalker_ref.py --modes both
"""

from __future__ import annotations

import json
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
NEMO_REV = "6967f48fda2a776a68bbaed6a71d7fea78ccc3f6"

app = modal.App("transcribe-multitalker-cpwer")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ca-certificates", "ffmpeg", "libsndfile1", "build-essential")
    .pip_install("Cython", "packaging", "huggingface_hub")
    .pip_install(f"nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/NeMo.git@{NEMO_REV}")
    .add_local_file(
        REPO / "scripts" / "dump_reference_multitalker_nemo.py",
        "/src/dump_reference_multitalker_nemo.py",
    )
    .add_local_dir(REPO / "samples" / "diar" / "ami-ihm-test", "/ami")
)

hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    timeout=4 * 3600,
    volumes={"/root/.cache/huggingface": hf_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_meeting(meeting: str, masked: bool, strict_fp32: bool = False) -> dict:
    import subprocess

    from huggingface_hub import hf_hub_download

    asr = hf_hub_download(
        "nvidia/multitalker-parakeet-streaming-0.6b-v1",
        "multitalker-parakeet-streaming-0.6b-v1.nemo",
    )
    diar = hf_hub_download(
        "nvidia/diar_streaming_sortformer_4spk-v2.1",
        "diar_streaming_sortformer_4spk-v2.1.nemo",
    )
    hf_vol.commit()

    wav = f"/ami/{meeting}.Mix-Headset.wav"
    out = f"/tmp/out/{meeting}"
    mode = "true" if masked else "false"
    cmd = [
        "python", "/src/dump_reference_multitalker_nemo.py",
        "--audio", wav,
        "--asr-model", asr,
        "--diar-model", diar,
        "--masked-asr", mode,
        "--device", "cuda",
        "--light",
        "--out", out,
    ]
    if strict_fp32:
        cmd.append("--strict-fp32")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-30:])
    if proc.returncode != 0:
        return {"meeting": meeting, "masked": masked, "ok": False, "log": tail}
    seglst = json.loads(pathlib.Path(out, "seglst.json").read_text())
    return {"meeting": meeting, "masked": masked, "ok": True, "seglst": seglst, "log": tail}


@app.local_entrypoint()
def main(meetings: str = "", modes: str = "both", strict_fp32: bool = False):
    if meetings:
        ids = [m.strip() for m in meetings.split(",") if m.strip()]
    else:
        manifest = REPO / "samples" / "diar" / "ami-ihm-test.manifest.jsonl"
        ids = sorted(
            {json.loads(l)["id"].split(".")[0] for l in manifest.read_text().splitlines() if l.strip()}
        )
    mode_list = [True, False] if modes == "both" else [modes == "masked"]

    cells = [(m, masked, strict_fp32) for m in ids for masked in mode_list]
    print(f"dispatching {len(cells)} cells ({len(ids)} meetings x {len(mode_list)} modes) on L40S")

    n_ok = 0
    for res in run_meeting.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"FAIL (exception): {res}")
            continue
        mode_name = "masked" if res["masked"] else "kernel"
        if not res["ok"]:
            print(f"FAIL {res['meeting']} [{mode_name}]:\n{res['log']}")
            continue
        suffix = "-fp32" if strict_fp32 else ""
        out_dir = REPO / "reports" / "cpwer" / f"nemo-{mode_name}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{res['meeting']}.seglst.json"
        out_path.write_text(json.dumps(res["seglst"], indent=1) + "\n")
        n_ok += 1
        print(f"ok   {res['meeting']} [{mode_name}]: {len(res['seglst'])} segments -> {out_path.relative_to(REPO)}")
    print(f"{n_ok}/{len(cells)} cells succeeded")
