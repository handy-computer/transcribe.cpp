#!/usr/bin/env python3
"""
ingest.py - build an evaluation corpus from a named dataset source.

Sources:
  fleurs   Reads the FLEURS parquet already present in the local Hugging
           Face hub cache (`datasets--google--fleurs`). Nothing is
           downloaded; the 15 configs the evaluation uses are cached.

Usage:
  uv run --project scripts/envs/ecapa_tdnn scripts/langid/ingest.py fleurs --lang en
  uv run --project scripts/envs/ecapa_tdnn scripts/langid/ingest.py fleurs --lang all

Output (gitignored, see .gitignore `/samples/langid/`):
  samples/langid/fleurs-<code>/<id>.wav        16-bit PCM mono 16 kHz
  samples/langid/fleurs-<code>.manifest.jsonl  {"id","audio","language","duration_s",...}

Only the FLEURS `test` split is read. Selection rule: the first N
utterances in parquet order whose decoded duration is >= 1 s. Parquet order
is the only ordering; no shuffling, no hashing, so the same cache always
produces the same manifest.

Utterance ids are `fleurs-<code>-<NNNN>` where NNNN is the index among the
selected rows, NOT the FLEURS sentence id: FLEURS records the same sentence
with several speakers, so sentence ids repeat within a split and would
collide as filenames. The sentence id is kept in the manifest as
`fleurs_id` for provenance.

Idempotent: an existing manifest is left alone unless --force.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_RATE = 16000
MIN_DURATION_S = 1.0
DEFAULT_N = 200
SPLIT = "test"
DATASET = "google/fleurs"
LICENCE = "CC-BY-4.0"

# FLEURS config -> VoxLingua107 label code, in the dataset gate's order.
# `es_419` is Latin-American Spanish, `cmn_hans_cn` is Mandarin in simplified
# Han, `nb_no` is Bokmal (the label set spells Norwegian `no`); VoxLingua107
# has one label for each.
FLEURS_LANGUAGES: list[tuple[str, str]] = [
    ("en_us", "en"),
    ("cmn_hans_cn", "zh"),
    ("de_de", "de"),
    ("fr_fr", "fr"),
    ("es_419", "es"),
    ("pt_br", "pt"),
    ("ja_jp", "ja"),
    ("ko_kr", "ko"),
    ("ru_ru", "ru"),
    ("cs_cz", "cs"),
    ("sk_sk", "sk"),
    ("nb_no", "no"),
    ("da_dk", "da"),
    ("id_id", "id"),
    ("ms_my", "ms"),
]

CODE_TO_CONFIG = {code: config for config, code in FLEURS_LANGUAGES}


def fleurs_parquet(config: str) -> Path:
    pattern = (
        Path.home()
        / ".cache/huggingface/hub/datasets--google--fleurs/snapshots/*/parquet-data"
        / config
        / f"{SPLIT}-00000-of-00001.parquet"
    )
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        raise SystemExit(
            f"error: no cached FLEURS parquet for config {config!r} "
            f"(looked at {pattern}). This script never downloads; populate "
            f"the Hugging Face cache first."
        )
    return Path(matches[-1])


def iter_rows(path: Path):
    """Yield parquet rows in file order without materialising the table.

    FLEURS test splits carry ~400-900 utterances of ~12 s each; a whole-table
    `to_pylist()` is a few hundred MB per language for the 200 rows we keep.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    cols = set(pf.schema_arrow.names)
    for needed in ("audio", "id"):
        if needed not in cols:
            raise SystemExit(
                f"error: {path} has no {needed!r} column "
                f"(columns: {sorted(cols)})"
            )
    for batch in pf.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            yield row


def decode_audio(row) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio = row["audio"]
    data = audio["bytes"] if isinstance(audio, dict) else audio
    pcm, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    return np.ascontiguousarray(pcm, dtype=np.float32), int(sr)


def resample_to_16k(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz. Only fires if FLEURS ever ships a non-16k config.

    Resampling rather than skipping the row keeps "the first N in parquet
    order" true; skipping would silently change which utterances the
    evaluation covers.
    """
    if sr == SAMPLE_RATE:
        return pcm
    import torch
    import torchaudio

    out = torchaudio.functional.resample(
        torch.from_numpy(pcm).unsqueeze(0), sr, SAMPLE_RATE
    )
    return np.ascontiguousarray(out.squeeze(0).numpy(), dtype=np.float32)


def write_wav(path: Path, pcm: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), pcm, SAMPLE_RATE, subtype="PCM_16")


def ingest_language(config: str, code: str, args) -> dict:
    out_dir = REPO_ROOT / "samples" / "langid" / f"fleurs-{code}"
    manifest_path = REPO_ROOT / "samples" / "langid" / f"fleurs-{code}.manifest.jsonl"

    if manifest_path.exists() and not args.force:
        rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
        print(f"{code}: {manifest_path.name} exists with {len(rows)} rows; "
              f"skipping (--force to rebuild)", file=sys.stderr)
        return summarise(code, config, rows, manifest_path)

    parquet = fleurs_parquet(config)
    entries: list[dict] = []
    n_short = 0
    n_resampled = 0
    for row_index, row in enumerate(iter_rows(parquet)):
        if len(entries) >= args.n:
            break
        pcm, sr = decode_audio(row)
        if sr != SAMPLE_RATE:
            pcm = resample_to_16k(pcm, sr)
            n_resampled += 1
        if pcm.size < MIN_DURATION_S * SAMPLE_RATE:
            n_short += 1
            continue
        utt_id = f"fleurs-{code}-{len(entries):04d}"
        wav_path = out_dir / f"{utt_id}.wav"
        write_wav(wav_path, pcm)
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

    if len(entries) < args.n:
        print(f"warning: {code}: only {len(entries)} utterances of the "
              f"requested {args.n} (split exhausted)", file=sys.stderr)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    info = summarise(code, config, entries, manifest_path)
    print(f"{code}: {len(entries)} utts  dur min/mean/max = "
          f"{info['duration_min_s']:.2f}/{info['duration_mean_s']:.2f}/"
          f"{info['duration_max_s']:.2f} s  (skipped {n_short} under "
          f"{MIN_DURATION_S} s, resampled {n_resampled})  -> {manifest_path.name}",
          file=sys.stderr)
    return info


def summarise(code: str, config: str, rows: list[dict], manifest_path: Path) -> dict:
    durs = [r["duration_s"] for r in rows] or [0.0]
    return {
        "language": code,
        "config": config,
        "n": len(rows),
        "duration_min_s": min(durs),
        "duration_mean_s": sum(durs) / len(durs),
        "duration_max_s": max(durs),
        "total_s": sum(durs),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
    }


def cmd_fleurs(args) -> int:
    if args.lang == "all":
        wanted = list(FLEURS_LANGUAGES)
    else:
        codes = [c.strip() for c in args.lang.split(",") if c.strip()]
        wanted = []
        for c in codes:
            if c in CODE_TO_CONFIG:
                wanted.append((CODE_TO_CONFIG[c], c))
            elif c in dict(FLEURS_LANGUAGES):
                wanted.append((c, dict(FLEURS_LANGUAGES)[c]))
            else:
                raise SystemExit(
                    f"error: unknown language {c!r}; known: "
                    f"{', '.join(sorted(CODE_TO_CONFIG))} or 'all'"
                )

    infos = [ingest_language(config, code, args) for config, code in wanted]

    print("\nlang  n     min_s  mean_s  max_s   total_s  manifest")
    for i in infos:
        print(f"{i['language']:<5} {i['n']:<5} {i['duration_min_s']:6.2f} "
              f"{i['duration_mean_s']:7.2f} {i['duration_max_s']:6.2f} "
              f"{i['total_s']:8.0f}  {i['manifest']}")
    total = sum(i["n"] for i in infos)
    longest = max((i["duration_max_s"] for i in infos), default=0.0)
    print(f"\ntotal utterances: {total}  longest clip: {longest:.2f} s")
    # The C++ context scores at most `max_audio_ms` (default 30 s) and uses
    # the tail of longer input, while SpeechBrain scores the whole clip. If
    # any clip is longer than 30 s the two engines would score different
    # audio on the `full` crop unless run.py raises the limit.
    if longest > 30.0:
        print("note: clips longer than 30 s exist; the cpp engine needs an "
              "explicit --max-audio-ms above the longest clip for the `full` "
              "crop to score the same audio as SpeechBrain.")
    else:
        print("note: every clip is under 30 s, so the C++ default "
              "max_audio_ms=30000 never truncates; both engines score the "
              "same audio on every crop.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="source", required=True)

    fp = sub.add_parser("fleurs", help="FLEURS from the local HF parquet cache")
    fp.add_argument("--lang", required=True,
                    help="VoxLingua107 code, comma-separated list, or 'all'")
    fp.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"utterances per language (default: {DEFAULT_N})")
    fp.add_argument("--force", action="store_true",
                    help="rebuild even when the manifest already exists")
    fp.set_defaults(func=cmd_fleurs)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
