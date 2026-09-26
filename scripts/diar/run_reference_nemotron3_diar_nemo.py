#!/usr/bin/env python3
"""
run_reference_nemotron3_diar_nemo.py - measured Nemotron-3-Diarization
reference baseline over a diarization manifest (diarization analog of
scripts/wer/run_reference_<family>_*.py).

Runs the NeMo Speech reference (the BF16 .nemo, fp32 compute on CPU) over
every meeting of an AMI/NOTSOFAR-style manifest at one streaming preset
and writes, per meeting:
  <pred-dir>/<uri>.probs.npy   raw [T, 8] sigmoid activity at 10 ms (the
                               one-for-one debug oracle for C++ drift)
  <pred-dir>/<uri>.rttm        segments from those probs
plus a summary JSONL. Score with scripts/diar/score_der.py.

Segments are derived from the saved probs with NeMo's
ts_vad_post_processing (the same call the C++ runner uses on the C++
probs), so reference and port are post-processed identically. Without
--postprocessing-yaml the thresholds are onset=offset=0.5 with no
padding / min-duration filtering, which is what NeMo's
e2e_diarize_speech.py applies when no post-processing YAML is given (the
model-card protocol).

This is the ONE-TIME reference run: Stage 4 and Stage 7 compare the C++
port's DER/JER against this measured baseline, not the publisher's number.

  uv run --project scripts/envs/nemotron3_diar \\
    scripts/diar/run_reference_nemotron3_diar_nemo.py \\
    --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \\
    --model nvidia/Nemotron-3-Diarization --revision f667ed73aee57d40cc39428eb768b4fd87a0a29e \\
    --preset very_high_latency \\
    --pred-dir reports/diar/pred/Nemotron-3-Diarization-REF-ami-ihm-test-fa-very_high_latency \\
    --out reports/diar/Nemotron-3-Diarization-REF.ami-ihm-test-fa.very_high_latency.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import dump_reference_nemotron3_diar_nemo as dumper  # noqa: E402

# Nemotron-3-Diarization emits one prediction per 10 ms feature frame.
UNIT_10MS_FRAME_COUNT = 1


def probs_to_rttm(uri: str, probs: np.ndarray, cfg_vad, offset: float = 0.0) -> list[str]:
    """Apply NeMo ts_vad_post_processing per speaker and emit RTTM lines."""
    import torch
    from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing

    lines: list[str] = []
    for spk in range(probs.shape[1]):
        ts_mat = ts_vad_post_processing(
            torch.from_numpy(np.ascontiguousarray(probs[:, spk])),
            cfg_vad_params=cfg_vad,
            unit_10ms_frame_count=UNIT_10MS_FRAME_COUNT,
            bypass_postprocessing=False,
        )
        for stt, end in ts_mat.tolist():
            stt, end = round(stt + offset, 2), round(end + offset, 2)
            if end - stt <= 0:
                continue
            lines.append(f"SPEAKER {uri} 1 {stt:.3f} {end - stt:.3f} <NA> <NA> speaker_{spk} <NA> <NA>")
    return lines


def load_vad_cfg(postprocessing_yaml: str | None):
    from omegaconf import OmegaConf

    if postprocessing_yaml:
        pp = OmegaConf.load(postprocessing_yaml)
        return pp.parameters if "parameters" in pp else pp
    return OmegaConf.create({"onset": 0.5, "offset": 0.5, "pad_onset": 0.0, "pad_offset": 0.0,
                             "min_duration_on": 0.0, "min_duration_off": 0.0})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True, help="HF repo id or path to the .nemo checkpoint")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--preset", default="very_high_latency", choices=list(dumper.PRESETS))
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=1, help="Accepted for the runner contract; meetings run one at a time.")
    ap.add_argument("--postprocessing-yaml", default=None,
                    help="Optional NeMo post-processing YAML; default = onset/offset 0.5, no filtering.")
    ap.add_argument("--limit", type=int, default=0, help="Run only the first N meetings (smoke).")
    args = ap.parse_args()

    import soundfile as sf
    import torch

    entries = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit:
        entries = entries[: args.limit]
    pred_dir, out_path = Path(args.pred_dir), Path(args.out)
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m = dumper._load_nemo(args.model, args.revision).to(args.device)
    geom = dumper._apply_preset(m, args.preset)
    cfg_vad = load_vad_cfg(args.postprocessing_yaml)
    print(f"preset={args.preset} {geom}", flush=True)

    rows = []
    for i, e in enumerate(entries):
        wav = e["audio"] if Path(e["audio"]).is_absolute() else str(REPO / e["audio"])
        uri = e["id"]
        audio, sr = sf.read(wav, dtype="float32", always_2d=False)
        t0 = time.time()
        with torch.no_grad():
            _, probs = m.diarize(audio=[audio], batch_size=1, sample_rate=sr, include_tensor_outputs=True)
        dt = time.time() - t0
        p = dumper._f32(torch.as_tensor(np.asarray(probs[0])))
        np.save(pred_dir / f"{uri}.probs.npy", p)
        lines = probs_to_rttm(uri, p, cfg_vad, offset=float(e.get("offset", 0.0)))
        (pred_dir / f"{uri}.rttm").write_text("\n".join(lines) + "\n")
        dur = e.get("duration") or len(audio) / sr
        rows.append({"id": uri, "hyp_rttm": str(pred_dir / f"{uri}.rttm"), "probs": str(pred_dir / f"{uri}.probs.npy"),
                     "n_segments": len(lines), "n_frames": int(p.shape[0]),
                     "n_active_speakers": int((p.max(0) > 0.5).sum()), "num_speakers_ref": e.get("num_speakers"),
                     "duration": dur, "compute_sec": round(dt, 2), "rtf": round(dt / max(dur, 1e-6), 4),
                     "preset": args.preset})
        print(f"  [{i}] {uri}: {len(lines)} segs, {rows[-1]['n_active_speakers']}/{e.get('num_speakers')} spk, "
              f"{dt:.1f}s (rtf {rows[-1]['rtf']:.3f})", flush=True)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_path} ({len(rows)} meetings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
