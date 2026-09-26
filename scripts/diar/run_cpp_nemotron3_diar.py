#!/usr/bin/env python3
"""
run_cpp_nemotron3_diar.py - C++ Nemotron-3-Diarization over a diarization
manifest (the C++ counterpart of run_reference_nemotron3_diar_nemo.py).

For each meeting this runs transcribe-cli with the streaming operating point
pinned (--diar-preset, the public RUN-slot extension; `small` goes through
the TRANSCRIBE_NEMOTRON3_DIAR_PRESET validation hook) and the tensor dump
enabled. The dumped raw `diar.probs` [T, 8] (10 ms) is post-processed with the
SAME function the reference runner uses (probs_to_rttm: NeMo
ts_vad_post_processing, onset/offset 0.5, no filtering), so only the
probabilities differ between reference and port. Writes per meeting:
  <pred-dir>/<uri>.probs.npy, <pred-dir>/<uri>.rttm
and a summary JSONL. With --ref-pred-dir, each meeting's probs are also
diffed against the reference run's saved <uri>.probs.npy (the one-for-one
debug oracle) and the per-meeting max |d| is recorded.

  uv run --project scripts/envs/nemotron3_diar scripts/diar/run_cpp_nemotron3_diar.py \\
    --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \\
    --gguf models/Nemotron-3-Diarization/Nemotron-3-Diarization-BF16.gguf \\
    --preset very_high_latency \\
    --pred-dir reports/diar/pred/Nemotron-3-Diarization-BF16-ami-ihm-test-fa-very_high_latency \\
    --ref-pred-dir reports/diar/pred/Nemotron-3-Diarization-REF-ami-ihm-test-fa-very_high_latency \\
    --out reports/diar/Nemotron-3-Diarization-BF16.ami-ihm-test-fa.very_high_latency.jsonl
  uv run scripts/diar/score_der.py --manifest ... --pred-dir ... --out ....score.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "diar"))
from run_reference_nemotron3_diar_nemo import load_vad_cfg, probs_to_rttm  # noqa: E402

CARD_PRESETS = ("very_high_latency", "low_latency", "very_low_latency", "ultra_low_latency")


def _load_probs(dump_dir: Path) -> np.ndarray:
    meta = json.loads((dump_dir / "diar.probs.json").read_text())
    shape = tuple(int(x) for x in meta["shape"])
    data = np.fromfile(dump_dir / "diar.probs.f32", dtype="<f4")
    if data.size != int(np.prod(shape)):
        raise SystemExit(f"error: diar.probs size {data.size} != shape {shape} product")
    return data.reshape(shape)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="very_high_latency", choices=CARD_PRESETS + ("small",))
    ap.add_argument("--ref-pred-dir", default=None, help="Reference run's pred dir (<uri>.probs.npy) to diff against.")
    ap.add_argument("--cli", default=str(REPO / "build" / "bin" / "transcribe-cli"))
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="0 -> CLI default")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="> 1: run groups of N meetings through one transcribe_run_batch call "
                         "(per-meeting probs from diar.probs.b<i>); compute_sec is then the group's "
                         "wall time split evenly.")
    ap.add_argument("--postprocessing-yaml", default=None,
                    help="Optional NeMo post-processing YAML; default = onset/offset 0.5, no filtering.")
    ap.add_argument("--limit", type=int, default=0, help="Run only the first N meetings (smoke).")
    args = ap.parse_args()

    entries = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit:
        entries = entries[: args.limit]
    pred_dir, out_path = Path(args.pred_dir), Path(args.out)
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_vad = load_vad_cfg(args.postprocessing_yaml)

    def run_group(group: list[dict]) -> tuple[list[np.ndarray], float]:
        """One CLI call: a single meeting, or one transcribe_run_batch over the group."""
        wavs = [e["audio"] if Path(e["audio"]).is_absolute() else str(REPO / e["audio"]) for e in group]
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="n3d-cpp-", dir=REPO.parent / "tmp") as tmp:
            env = os.environ.copy()
            env["TRANSCRIBE_DUMP_DIR"] = tmp
            cmd = [args.cli, "--backend", args.backend, "-m", args.gguf]
            if args.preset in CARD_PRESETS:
                cmd += ["--diar-preset", args.preset]
            else:
                env["TRANSCRIBE_NEMOTRON3_DIAR_PRESET"] = args.preset
            if args.threads > 0:
                cmd += ["--threads", str(args.threads)]
            if args.batch_size > 1:
                lst = Path(tmp) / "list.txt"
                lst.write_text("\n".join(wavs) + "\n")
                cmd += ["--batch", str(lst), "--batch-jsonl", "-q", "--batch-size", str(len(group))]
            else:
                cmd.append(wavs[0])
            res = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, errors="replace")
            if res.returncode != 0:
                sys.stderr.write(res.stdout or "")
                raise SystemExit(f"error: transcribe-cli failed on {[e['id'] for e in group]} (exit {res.returncode})")
            if args.batch_size > 1:
                probs = []
                for i in range(len(group)):
                    d = Path(tmp) / f"b{i}"
                    d.mkdir()
                    for ext in ("f32", "json"):
                        (Path(tmp) / f"diar.probs.b{i}.{ext}").rename(d / f"diar.probs.{ext}")
                    probs.append(_load_probs(d))
            else:
                probs = [_load_probs(Path(tmp))]
        return probs, time.time() - t0

    rows = []
    step = max(1, args.batch_size)
    for g0 in range(0, len(entries), step):
        group = entries[g0:g0 + step]
        group_probs, group_dt = run_group(group)
        for i, (e, probs) in enumerate(zip(group, group_probs), start=g0):
            dt = group_dt / len(group)
            rows.append(_emit(args, pred_dir, cfg_vad, i, e, probs, dt))

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_path} ({len(rows)} meetings)")
    return 0


def _emit(args, pred_dir: Path, cfg_vad, i: int, e: dict, probs: np.ndarray, dt: float) -> dict:
    """Save one meeting's probs + RTTM and return its summary row."""
    uri = e["id"]
    np.save(pred_dir / f"{uri}.probs.npy", probs)
    lines = probs_to_rttm(uri, probs, cfg_vad, offset=float(e.get("offset", 0.0)))
    (pred_dir / f"{uri}.rttm").write_text("\n".join(lines) + "\n")
    dur = e.get("duration") or 1.0
    row = {"id": uri, "hyp_rttm": str(pred_dir / f"{uri}.rttm"), "probs": str(pred_dir / f"{uri}.probs.npy"),
           "n_segments": len(lines), "n_frames": int(probs.shape[0]),
           "n_active_speakers": int((probs.max(0) > 0.5).sum()), "num_speakers_ref": e.get("num_speakers"),
           "duration": dur, "compute_sec": round(dt, 2), "rtf": round(dt / max(dur, 1e-6), 4),
           "preset": args.preset, "batch_size": args.batch_size}
    msg = ""
    if args.ref_pred_dir:
        ref = np.load(Path(args.ref_pred_dir) / f"{uri}.probs.npy")
        n = min(len(ref), len(probs))
        d = np.abs(ref[:n] - probs[:n])
        row.update({"ref_frames": int(len(ref)), "probs_max_abs": float(d.max()),
                    "probs_mean_abs": float(d.mean()),
                    "frames_flipped": int(((ref[:n] > 0.5) != (probs[:n] > 0.5)).sum())})
        msg = f", vs ref max|d| {row['probs_max_abs']:.2e} flips {row['frames_flipped']}"
        if len(ref) != len(probs):
            msg += f" FRAMES {len(probs)} != ref {len(ref)}"
    print(f"  [{i}] {uri}: {len(lines)} segs, {row['n_active_speakers']} spk, {dt:.1f}s "
          f"(rtf {row['rtf']:.3f}){msg}", flush=True)
    return row


if __name__ == "__main__":
    sys.exit(main())
