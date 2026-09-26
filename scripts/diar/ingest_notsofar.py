#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface-hub>=0.20",
#     "numpy>=1.26",
#     "soundfile>=0.12",
# ]
# ///
"""
ingest_notsofar.py - build a NOTSOFAR-1 eval "MHM" diarization manifest.

Diarization analog of scripts/wer/ingest.py for the NOTSOFAR-1 meeting
corpus (3-7 speakers). It covers speaker counts above AMI's 3-4, which
8-speaker diarizers (Nemotron-3-Diarization) need.

Sources:
  audio   microsoft/NOTSOFAR (HF dataset, CC-BY-4.0, ungated),
          benchmark-datasets/eval_set/240825.1_eval_full_with_GT/MTG/<MTG>/
          close_talk/CT_*.wav: one headset per participant. Downloaded into
          the HF hub cache.
  labels  forced-alignment RTTMs from popcornell/FastMSS
          (resources/notsofar1-channels_mfa_rttms.tar.gz, `eval/` split),
          the reference labels NVIDIA's Nemotron-3-Diarization card uses.
          The per-recording RTTMs of one meeting are identical (devices share
          the meeting timeline), so one RTTM labels the meeting.

"MHM" (mix of headset microphones) is built here as the sample-wise sum
of the meeting's close-talk channels, zero-padded to the longest channel
(truncating would drop labeled speech if one headset stops early) and
peak-normalized to 0.9, then written as 16-bit PCM mono 16 kHz.
NVIDIA does not publish its exact mixing recipe; our DER gate compares
the reference model and the C++ port on this same audio, so the mix only
needs to be fixed and documented.

Only the eval meetings with a forced-alignment RTTM are ingested (80).

Output:
  samples/diar/notsofar-mhm-eval/<MTG>.wav
  samples/diar/notsofar-mhm-eval/<MTG>.rttm      (FA labels, uri = <MTG>)
  samples/diar/notsofar-mhm-eval-fa.manifest.jsonl
      {"id","audio","rttm","duration","num_speakers"}

Usage:
  uv run scripts/diar/ingest_notsofar.py [--min-speakers 5] [--limit N]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
REPO = Path(__file__).resolve().parent.parent.parent
HF_REPO = "microsoft/NOTSOFAR"
EVAL_PREFIX = "benchmark-datasets/eval_set/240825.1_eval_full_with_GT/MTG"
FA_URL = "https://raw.githubusercontent.com/popcornell/FastMSS/master/resources/notsofar1-channels_mfa_rttms.tar.gz"
DS_ID = "notsofar-mhm-eval"


def _fa_rttms() -> dict[str, list[str]]:
    """meeting ('MTG_32000') -> RTTM lines of its first eval recording."""
    raw = urllib.request.urlopen(FA_URL).read()
    out: dict[str, list[str]] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            parts = member.name.split("/")
            if len(parts) != 2 or parts[0] != "eval" or not parts[1].endswith(".rttm"):
                continue
            meeting = f"MTG_{parts[1][1:6]}"
            if meeting in out:
                continue
            text = tar.extractfile(member).read().decode()
            out[meeting] = [l for l in text.splitlines() if l.strip()]
    return out


def _mix_close_talk(meeting: str, listing: list[str]) -> tuple[np.ndarray, list[float]]:
    from huggingface_hub import hf_hub_download

    prefix = f"{EVAL_PREFIX}/{meeting}/close_talk/"
    files = sorted(f for f in listing if f.startswith(prefix) and f.endswith(".wav"))
    if not files:
        raise SystemExit(f"error: no close-talk channels for {meeting}")
    chans = []
    for f in files:
        audio, sr = sf.read(hf_hub_download(HF_REPO, f, repo_type="dataset"), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            raise SystemExit(f"error: {f} is {sr} Hz, expected {SR}")
        chans.append(audio)
    mix = np.zeros(max(len(c) for c in chans), dtype=np.float64)
    for c in chans:
        mix[: len(c)] += c
    peak = float(np.max(np.abs(mix))) or 1.0
    return (mix / peak * 0.9).astype(np.float32), [len(c) / SR for c in chans]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-speakers", type=int, default=0, help="Keep only meetings with >= N speakers.")
    ap.add_argument("--limit", type=int, default=0, help="Ingest only the first N meetings (smoke).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    out_dir = REPO / "samples" / "diar" / DS_ID
    manifest = REPO / "samples" / "diar" / f"{DS_ID}-fa.manifest.jsonl"
    if manifest.exists() and not args.force:
        n = sum(1 for _ in open(manifest))
        print(f"OK already exists: {manifest} ({n} meetings). Use --force to rebuild.")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    listing = HfApi().list_repo_files(HF_REPO, repo_type="dataset")
    rttms = _fa_rttms()
    meetings = sorted(rttms)
    print(f"{len(meetings)} eval meetings with forced-alignment RTTMs", flush=True)

    entries, total_dur = [], 0.0
    for meeting in meetings:
        lines = rttms[meeting]
        n_spk = len({l.split()[7] for l in lines})
        if n_spk < args.min_speakers:
            continue
        if args.limit and len(entries) >= args.limit:
            break
        audio, ch_secs = _mix_close_talk(meeting, listing)
        meta = json.load(open(hf_hub_download(HF_REPO, f"{EVAL_PREFIX}/{meeting}/gt_meeting_metadata.json",
                                              repo_type="dataset")))
        last_label = max(float(l.split()[3]) + float(l.split()[4]) for l in lines)
        wav, rttm = out_dir / f"{meeting}.wav", out_dir / f"{meeting}.rttm"
        sf.write(str(wav), audio, SR, subtype="PCM_16")
        # Re-key the RTTM uri from the recording id (S32000201) to the meeting.
        rttm.write_text("\n".join(" ".join([p[0], meeting, *p[2:]]) for p in (l.split() for l in lines)) + "\n")
        dur = len(audio) / SR
        total_dur += dur
        entries.append({"id": meeting, "audio": str(wav.relative_to(REPO)), "rttm": str(rttm.relative_to(REPO)),
                        "duration": round(dur, 2), "num_speakers": n_spk,
                        "close_talk_sec": [round(x, 2) for x in ch_secs],
                        "meeting_duration_sec": round(float(meta["MeetingDurationSec"]), 2),
                        "last_label_end_sec": round(last_label, 2)})
        flag = "  WARN labels past audio" if last_label > dur + 0.5 else ""
        print(f"  [{len(entries) - 1}] {meeting}: {dur / 60:.1f}min, {n_spk} spk, {len(lines)} turns, "
              f"ct {min(ch_secs):.1f}-{max(ch_secs):.1f}s, meta {float(meta['MeetingDurationSec']):.1f}s, "
              f"last label {last_label:.1f}s{flag}", flush=True)

    with open(manifest, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"\nmanifest: {manifest.relative_to(REPO)}")
    print(f"{len(entries)} meetings, {total_dur / 60:.1f} min total audio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
