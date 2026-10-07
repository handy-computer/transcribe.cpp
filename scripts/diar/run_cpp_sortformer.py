#!/usr/bin/env python3
"""
run_cpp_sortformer.py - C++ Sortformer diarization over an AMI-style manifest
(the C++ counterpart of run_reference_sortformer_nemo.py).

For each meeting this runs the transcribe-cli C++ binary with the streaming
operating point selected (TRANSCRIBE_SORTFORMER_STREAM_PRESET) and the tensor
dump enabled (TRANSCRIBE_DUMP_DIR). The dumped raw `diar.probs` [T, n_spk]
sigmoid matrix is then post-processed with NeMo's *own* `ts_vad_post_processing`
(same dihard3 YAML the reference uses) so the timestamp post-processing is
bit-identical to the reference; only the probabilities differ. The resulting
per-speaker segments are written as one predicted RTTM per meeting.

Score with scripts/diar/score_der.py against the forced-alignment RTTMs, e.g.:

  uv run --project scripts/envs/sortformer \
    scripts/diar/run_cpp_sortformer.py \
    --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
    --gguf models/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1-F32.gguf \
    --preset very_high_latency \
    --postprocessing-yaml scripts/diar/postprocessing/diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml \
    --pred-dir reports/diar/pred/diar_streaming_sortformer_4spk-v2.1-cpp-ami-ihm-test \
    --out reports/diar/diar_streaming_sortformer_4spk-v2.1-CPP.ami-ihm-test.jsonl

  uv run scripts/diar/score_der.py \
    --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
    --pred-dir reports/diar/pred/diar_streaming_sortformer_4spk-v2.1-cpp-ami-ihm-test \
    --out reports/diar/diar_streaming_sortformer_4spk-v2.1-CPP.ami-ihm-test.score.json
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

# Diar frame = 80 ms = 8 * 10 ms (subsampling_factor).
UNIT_10MS_FRAME_COUNT = 8


def _find_cli(repo: Path, override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            raise SystemExit(f"error: --cli not found: {p}")
        return p
    for cand in (repo / "build" / "bin" / "transcribe-cli", repo / "build" / "transcribe-cli"):
        if cand.exists():
            return cand
    raise SystemExit("error: transcribe-cli not found; build it first "
                     "(cmake --build build --target transcribe-cli)")


def _load_probs(dump_dir: Path) -> np.ndarray:
    """Load the dumped diar.probs as [T, n_spk] float32."""
    meta = json.loads((dump_dir / "diar.probs.json").read_text())
    shape = tuple(int(x) for x in meta["shape"])  # slow-to-fast, e.g. [T, n_spk]
    data = np.fromfile(dump_dir / "diar.probs.f32", dtype="<f4")
    if data.size != int(np.prod(shape)):
        raise SystemExit(f"error: diar.probs size {data.size} != shape {shape} product")
    return data.reshape(shape).astype(np.float32)


def _probs_to_rttm(uri: str, probs: np.ndarray, cfg_vad, offset: float = 0.0) -> list[str]:
    """Apply NeMo ts_vad_post_processing per speaker and emit RTTM lines."""
    import torch
    from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing

    lines: list[str] = []
    n_spk = probs.shape[1]
    for spk in range(n_spk):
        ts_vad_vec = torch.from_numpy(np.ascontiguousarray(probs[:, spk]))
        ts_mat = ts_vad_post_processing(
            ts_vad_vec,
            cfg_vad_params=cfg_vad,
            unit_10ms_frame_count=UNIT_10MS_FRAME_COUNT,
            bypass_postprocessing=False,
        )
        for stt, end in ts_mat.tolist():
            stt = round(stt + offset, 2)
            end = round(end + offset, 2)
            dur = end - stt
            if dur <= 0:
                continue
            lines.append(f"SPEAKER {uri} 1 {stt:.3f} {dur:.3f} <NA> <NA> speaker_{spk} <NA> <NA>")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cli", default=None, help="Path to transcribe-cli (default: build/bin/transcribe-cli)")
    ap.add_argument("--preset", default="very_high_latency",
                    help="Streaming operating point (TRANSCRIBE_SORTFORMER_STREAM_PRESET)")
    ap.add_argument("--postprocessing-yaml", default=None,
                    help="dihard3 PP YAML (onset/offset/pad/min_duration); bypass PP if omitted.")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="0 -> CLI default")
    ap.add_argument("--limit", type=int, default=0, help="Run only the first N meetings (smoke).")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    repo = Path(__file__).resolve().parent.parent.parent
    cli = _find_cli(repo, args.cli)
    entries = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit:
        entries = entries[:args.limit]
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.postprocessing_yaml:
        pp = OmegaConf.load(args.postprocessing_yaml)
        cfg_vad = pp.parameters if "parameters" in pp else pp
        bypass = False
    else:
        cfg_vad = OmegaConf.create({"onset": 0.5, "offset": 0.5, "pad_onset": 0.0, "pad_offset": 0.0,
                                    "min_duration_on": 0.0, "min_duration_off": 0.0})
        bypass = True
    if bypass:
        print("warning: no --postprocessing-yaml; using bypass thresholds (not the DER gate config)", flush=True)

    rows = []
    for i, e in enumerate(entries):
        wav = str(repo / e["audio"]) if not Path(e["audio"]).is_absolute() else e["audio"]
        uri = e["id"]
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="sf-cpp-") as tmp:
            env = os.environ.copy()
            env["TRANSCRIBE_DUMP_DIR"] = tmp
            env["TRANSCRIBE_SORTFORMER_STREAM_PRESET"] = args.preset
            cmd = [str(cli), "--backend", args.backend, "-m", str(args.gguf)]
            if args.threads > 0:
                cmd += ["--threads", str(args.threads)]
            cmd.append(wav)
            res = subprocess.run(cmd, cwd=repo, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, errors="replace")
            if res.returncode != 0:
                sys.stderr.write(res.stdout or "")
                raise SystemExit(f"error: transcribe-cli failed on {uri} (exit {res.returncode})")
            probs = _load_probs(Path(tmp))
        lines = _probs_to_rttm(uri, probs, cfg_vad, offset=float(e.get("offset", 0.0)))
        (pred_dir / f"{uri}.rttm").write_text("\n".join(lines) + "\n")
        dt = time.time() - t0
        rtf = dt / max(e.get("duration", 1.0), 1e-6)
        rows.append({"id": uri, "hyp_rttm": str(Path(args.pred_dir) / f"{uri}.rttm"),
                     "n_segments": len(lines), "n_frames": int(probs.shape[0]),
                     "duration": e.get("duration"), "compute_sec": round(dt, 2), "rtf": round(rtf, 4)})
        print(f"  [{i}] {uri}: {len(lines)} segs, {probs.shape[0]} frames, {dt:.1f}s (rtf {rtf:.3f})", flush=True)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_path} ({len(rows)} meetings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
