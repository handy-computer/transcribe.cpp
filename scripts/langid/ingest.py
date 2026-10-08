#!/usr/bin/env python3
"""
ingest.py - build the language ID corpus from the FLEURS test parquet at the
pinned revision FLEURS_REVISION (read through the Hugging Face hub cache).

  uv run --project scripts/envs/ecapa_tdnn scripts/langid/ingest.py fleurs --lang all

Output (gitignored):
  samples/langid/fleurs-<code>/<id>.wav        16-bit PCM mono 16 kHz
  samples/langid/fleurs-<code>.manifest.jsonl  {"id","audio","language","duration_s",...}

The languages are the langid publication profile's `pooled_languages`. Per
language: the first N utterances in parquet order that last at least 1 s.
Ids are `fleurs-<code>-<NNNN>` by selection index (FLEURS sentence ids repeat
across speakers); the sentence id is kept as `fleurs_id`. An existing
manifest is left alone unless --force.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "wer"))
from languages import FLEURS_LANGS  # noqa: E402

SAMPLE_RATE = 16000
MIN_DURATION_S = 1.0
SPLIT = "test"
DATASET = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
LICENCE = "CC-BY-4.0"


def profile_languages() -> list[str]:
    data = json.loads((REPO_ROOT / "catalog" / "_benchmark_profiles.json").read_text())
    return data["profiles"][data["roles"]["langid"]]["accuracy"][0]["pooled_languages"]


def ingest_language(code: str, n: int, force: bool) -> None:
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    config = FLEURS_LANGS[code]
    out_dir = REPO_ROOT / "samples" / "langid" / f"fleurs-{code}"
    manifest_path = out_dir.with_name(f"fleurs-{code}.manifest.jsonl")
    if manifest_path.exists() and not force:
        print(f"{code}: {manifest_path.name} exists; skipping (--force to rebuild)")
        return

    parquet = hf_hub_download(
        repo_id=DATASET, repo_type="dataset", revision=FLEURS_REVISION,
        filename=f"parquet-data/{config}/{SPLIT}-00000-of-00001.parquet")
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    rows = (row for batch in pq.ParquetFile(parquet).iter_batches(batch_size=16)
            for row in batch.to_pylist())
    for row_index, row in enumerate(rows):
        if len(entries) >= n:
            break
        audio = row["audio"]
        pcm, sr = sf.read(io.BytesIO(audio["bytes"] if isinstance(audio, dict) else audio),
                          dtype="float32", always_2d=False)
        if sr != SAMPLE_RATE:
            raise SystemExit(f"error: {code} row {row_index} is {sr} Hz; FLEURS is 16 kHz")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        if pcm.size < MIN_DURATION_S * SAMPLE_RATE:
            continue
        utt_id = f"fleurs-{code}-{len(entries):04d}"
        wav_path = out_dir / f"{utt_id}.wav"
        sf.write(str(wav_path), pcm, SAMPLE_RATE, subtype="PCM_16")
        entries.append({
            "id": utt_id,
            "audio": str(wav_path.relative_to(REPO_ROOT)),
            "language": code,
            "duration_s": round(pcm.size / SAMPLE_RATE, 3),
            "fleurs_id": int(row["id"]),
            "config": config,
            "split": SPLIT,
            "row_index": row_index,
            "dataset": DATASET,
            "licence": LICENCE,
        })
    if len(entries) < n:
        print(f"warning: {code}: only {len(entries)} of {n} utterances", file=sys.stderr)
    manifest_path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    print(f"{code}: {len(entries)} utterances -> {manifest_path.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="source", required=True)
    fp = sub.add_parser("fleurs", help="FLEURS parquet at the pinned revision")
    fp.add_argument("--lang", required=True,
                    help="comma-separated VoxLingua107 codes from the profile, or 'all'")
    fp.add_argument("--n", type=int, default=200, help="utterances per language")
    fp.add_argument("--force", action="store_true",
                    help="rebuild even when the manifest already exists")
    args = p.parse_args(argv)

    known = profile_languages()
    codes = known if args.lang == "all" else args.lang.split(",")
    unknown = sorted(set(codes) - set(known))
    if unknown:
        p.error(f"unknown language(s) {unknown}; known: {', '.join(known)} or 'all'")
    for code in codes:
        ingest_language(code, args.n, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
