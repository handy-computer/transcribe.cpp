#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets==3.6.0",
#     "librosa>=0.10",
#     "numpy>=1.26",
#     "soundfile>=0.12",
# ]
# ///
"""
ingest.py — build a WER manifest from a named dataset source.

Sources:
  librispeech   Walks an extracted LibriSpeech split (downloads if absent).
  fleurs        Downloads google/fleurs for a single BCP-47 language.

Usage:
  uv run scripts/wer/ingest.py librispeech [--split test-clean]
  uv run scripts/wer/ingest.py fleurs --lang es [--split test]
  uv run scripts/wer/ingest.py fleurs --lang zh

Output paths (consistent across sources):
  samples/wer/<source>-<id>/<utt>.wav        16-bit PCM mono 16 kHz
  samples/wer/<source>-<id>.manifest.jsonl   {"id","audio","ref_text","language"}

Adding a new source: write a `def ingest_<name>(repo, args)` function and
register it in SOURCES + add an argparse subparser for its flags. The
contract is: fetch raw data, decode to 16 kHz mono WAVs, write a manifest
with `language` set to the BCP-47 short code for the utterance.

Score.py picks WER vs CER from the `language` field at score time.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import numpy as np
import soundfile as sf

from languages import FLEURS_LANGS


# -------- Shared helpers --------------------------------------------------

def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("cannot locate repo root")


def write_wav_16k_mono(data: np.ndarray, sr: int, out_path: Path) -> None:
    """Write 16-bit PCM mono wav at 16 kHz. Resamples via linear interp
    if needed (mostly a no-op since LibriSpeech and FLEURS are 16 kHz)."""
    if sr != 16000:
        n_out = int(len(data) * 16000 / sr)
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, n_out)
        data = np.interp(x_new, x_old, data)
        sr = 16000

    if data.dtype in (np.float32, np.float64):
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767).astype(np.int16)

    sf.write(str(out_path), data, sr, subtype="PCM_16", format="WAV")


def write_manifest(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


# -------- LibriSpeech adapter ---------------------------------------------

LIBRISPEECH_URLS = {
    "test-clean":  "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other":  "https://www.openslr.org/resources/12/test-other.tar.gz",
    "dev-clean":   "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other":   "https://www.openslr.org/resources/12/dev-other.tar.gz",
}


def _librispeech_fetch(split: str, raw_dir: Path) -> Path:
    """Download and extract a LibriSpeech split into raw_dir. Idempotent."""
    extract_dir = raw_dir / "LibriSpeech" / split
    if extract_dir.is_dir():
        return extract_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / f"{split}.tar.gz"
    if not archive.exists():
        url = LIBRISPEECH_URLS[split]
        print(f"downloading {url}")
        urlretrieve(url, archive)
    print(f"extracting {archive}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(raw_dir)
    return extract_dir


def ingest_librispeech(repo: Path, args: argparse.Namespace) -> int:
    split = args.split
    raw_dir = repo / "samples/wer/raw"
    out_dir = repo / f"samples/wer/librispeech-{split}"
    manifest = repo / f"samples/wer/librispeech-{split}.manifest.jsonl"

    extract_dir = _librispeech_fetch(split, raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trans_files = sorted(extract_dir.rglob("*.trans.txt"))
    if not trans_files:
        print(f"error: no .trans.txt files found in {extract_dir}",
              file=sys.stderr)
        return 2

    entries: list[dict] = []
    n_converted = 0
    n_skipped = 0
    for tf in trans_files:
        chap_dir = tf.parent
        for line in tf.read_text().strip().splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, ref_text = parts
            flac_path = chap_dir / f"{utt_id}.flac"
            if not flac_path.exists():
                print(f"warning: {flac_path} not found, skipping",
                      file=sys.stderr)
                continue
            wav_path = out_dir / f"{utt_id}.wav"
            if not wav_path.exists():
                data, sr = sf.read(str(flac_path), dtype="float32")
                if data.ndim > 1:
                    data = data[:, 0]
                write_wav_16k_mono(data, sr, wav_path)
                n_converted += 1
            else:
                n_skipped += 1
            entries.append({
                "id": utt_id,
                "audio": str(wav_path),
                "ref_text": ref_text,
                "language": "en",
            })

    entries.sort(key=lambda e: e["id"])
    write_manifest(entries, manifest)

    print(f"manifest: {manifest}")
    print(f"  {len(entries)} utterances")
    print(f"  {n_converted} converted, {n_skipped} skipped (already existed)")
    return 0


# -------- FLEURS adapter --------------------------------------------------

def ingest_fleurs(repo: Path, args: argparse.Namespace) -> int:
    lang = args.lang.lower()
    config = FLEURS_LANGS.get(lang)
    if not config:
        codes = ", ".join(sorted(FLEURS_LANGS))
        print(f"error: no FLEURS mapping for '{lang}'.\n"
              f"available codes: {codes}",
              file=sys.stderr)
        return 2

    out_dir = repo / f"samples/wer/fleurs-{lang}"
    manifest = repo / f"samples/wer/fleurs-{lang}.manifest.jsonl"

    if manifest.exists() and not args.force:
        n_existing = sum(1 for _ in open(manifest))
        print(f"OK already exists: {manifest} ({n_existing} utterances). "
              f"Pass --force to regenerate.")
        return 0

    print(f"loading google/fleurs[{config}] split={args.split}")
    # Defer the heavy import so --help is snappy and a librispeech-only
    # invocation doesn't initialize HF datasets state.
    from datasets import load_dataset

    ds = load_dataset("google/fleurs", config, split=args.split,
                      trust_remote_code=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    n_converted = 0
    n_skipped = 0
    for row in ds:
        # FLEURS row schema: id (int, transcription/sentence id), path
        # (full path to source wav), audio {array, sampling_rate, path},
        # transcription (lowercased normalized), raw_transcription,
        # num_samples, lang_id, language, gender, lang_group_id.
        #
        # FLEURS has multiple speakers per transcription, so `id` is NOT
        # unique. The unique key per recording is the source wav basename
        # (a long numeric Google id).
        audio_stem = Path(row["path"]).stem
        utt_id = f"fleurs-{lang}-{audio_stem}"
        wav_path = out_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            audio = row["audio"]
            data = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            write_wav_16k_mono(data, sr, wav_path)
            n_converted += 1
        else:
            n_skipped += 1
        entries.append({
            "id": utt_id,
            "audio": str(wav_path),
            "ref_text": row["transcription"],
            "language": lang,
        })

    entries.sort(key=lambda e: e["id"])
    write_manifest(entries, manifest)

    print(f"manifest: {manifest}")
    print(f"  {len(entries)} utterances ({args.split} split, "
          f"google/fleurs[{config}])")
    print(f"  {n_converted} converted, {n_skipped} skipped (already existed)")
    return 0


# -------- eka-medical-asr-evaluation-dataset (English + Hindi) -----------
#
# https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset
# Domain-relevant for medasr (medical dictation / doctor-patient
# conversations); MIT license; ungated. ~3,620 EN + 320 HI utterances,
# test split only. Schema: `audio` (HF audio column) + `text` (string).

EKA_MEDICAL_ASR_LANGS: dict[str, str] = {
    "en": "en",
    "hi": "hi",
}


def ingest_eka_medical_asr(repo: Path, args: argparse.Namespace) -> int:
    lang = args.lang.lower()
    config = EKA_MEDICAL_ASR_LANGS.get(lang)
    if not config:
        codes = ", ".join(sorted(EKA_MEDICAL_ASR_LANGS))
        print(f"error: no eka-medical-asr config for '{lang}'.\n"
              f"available codes: {codes}",
              file=sys.stderr)
        return 2

    out_dir = repo / f"samples/wer/eka-medical-asr-{lang}"
    manifest = repo / f"samples/wer/eka-medical-asr-{lang}.manifest.jsonl"

    if manifest.exists() and not args.force:
        n_existing = sum(1 for _ in open(manifest))
        print(f"OK already exists: {manifest} ({n_existing} utterances). "
              f"Pass --force to regenerate.")
        return 0

    print(f"loading ekacare/eka-medical-asr-evaluation-dataset[{config}] "
          f"split={args.split}")
    from datasets import load_dataset

    ds = load_dataset("ekacare/eka-medical-asr-evaluation-dataset",
                      config, split=args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The eka dataset's `file_name` column is NOT unique — 39 stems
    # recur across multiple rows (one stem appears 28 times), each row
    # being a distinct recording with its own `text`. We pre-scan for
    # collisions and disambiguate duplicates with the row index so each
    # row maps to its own wav and ref_text pair.
    from collections import Counter
    file_names = [row.get("file_name") or "" for row in ds]
    stem_counts = Counter(Path(fn).stem if fn else "" for fn in file_names)

    entries: list[dict] = []
    n_converted = 0
    n_skipped = 0
    for i, row in enumerate(ds):
        file_name = file_names[i]
        raw_stem = Path(file_name).stem if file_name else ""
        if raw_stem and stem_counts[raw_stem] == 1:
            audio_stem = raw_stem
        elif raw_stem:
            audio_stem = f"{raw_stem}-row{i:05d}"
        else:
            audio_stem = f"row{i:05d}"
        utt_id = f"eka-medical-asr-{lang}-{audio_stem}"
        wav_path = out_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            audio = row["audio"]
            data = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            write_wav_16k_mono(data, sr, wav_path)
            n_converted += 1
        else:
            n_skipped += 1
        entries.append({
            "id": utt_id,
            "audio": str(wav_path),
            "ref_text": row["text"],
            "language": lang,
        })

    entries.sort(key=lambda e: e["id"])
    write_manifest(entries, manifest)

    print(f"manifest: {manifest}")
    print(f"  {len(entries)} utterances ({args.split} split, "
          f"ekacare/eka-medical-asr-evaluation-dataset[{config}])")
    print(f"  {n_converted} converted, {n_skipped} skipped (already existed)")
    return 0


# -------- TED-LIUM 3 long-form (11 full test talks) -----------------------
#
# https://huggingface.co/datasets/distil-whisper/tedlium-long-form
# Full recordings (10-20 min) with concatenated, lowercased references.
# Schema: `audio`, `text`, `speaker_id`. No clip is under 30 s.

def ingest_tedlium_longform(repo: Path, args: argparse.Namespace) -> int:
    out_dir = repo / "samples/wer/tedlium-longform"
    manifest = repo / "samples/wer/tedlium-longform.manifest.jsonl"

    if manifest.exists() and not args.force:
        n_existing = sum(1 for _ in open(manifest))
        print(f"OK already exists: {manifest} ({n_existing} talks). "
              f"Pass --force to regenerate.")
        return 0

    print("loading distil-whisper/tedlium-long-form split=test")
    from datasets import load_dataset

    ds = load_dataset("distil-whisper/tedlium-long-form", split="test")
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for row in ds:
        utt_id = f"tedlium-longform-{row['speaker_id']}"
        wav_path = out_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            audio = row["audio"]
            write_wav_16k_mono(np.asarray(audio["array"], dtype=np.float32),
                               int(audio["sampling_rate"]), wav_path)
        entries.append({
            "id": utt_id,
            "audio": str(wav_path),
            "ref_text": " ".join(row["text"].split()),
            "language": "en",
        })

    entries.sort(key=lambda e: e["id"])
    write_manifest(entries, manifest)
    print(f"manifest: {manifest}")
    print(f"  {len(entries)} talks")
    return 0


# -------- Multilingual TEDx long-form (OpenSLR 100) -------------------------
#
# https://www.openslr.org/100/  CC BY-NC-ND 4.0 (same class as TED-LIUM).
# Full TEDx talks in es, fr, pt, it, ru, el, ar, de with sentence-level
# transcripts aligned to the talk. The long-form set is built the way
# distil-whisper built tedlium-long-form: one entry per test-split talk, the
# talk's original audio, and the reference is that talk's segment transcripts
# joined in time order. Archive layout: <lang>-<lang>/data/test/wav/<talk>.flac,
# txt/segments ("<talk>_<n> <talk> <start> <end>") and txt/test.<lang> (one
# transcript per segments line).

MTEDX_URL = "https://www.openslr.org/resources/100/mtedx_{lang}.tgz"
MTEDX_LANGS = ("ar", "de", "el", "es", "fr", "it", "pt", "ru")


def _mtedx_fetch_test(lang: str, raw_dir: Path) -> Path:
    """Download mtedx_<lang>.tgz and extract only its data/test tree into
    raw_dir/mtedx/<lang>. Idempotent. The per-language archive is mostly
    training audio (2.6 GB for de, 35 GB for es); the test split is a few
    dozen talks, so the stream is read once and everything else is skipped."""
    extract_dir = raw_dir / "mtedx" / lang
    marker = extract_dir / ".test-extracted"
    if marker.exists():
        return extract_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / f"mtedx_{lang}.tgz"
    url = MTEDX_URL.format(lang=lang)
    # Download to a .part file and only rename once the byte count matches
    # Content-Length, so an interrupted or throttled transfer (openslr.org,
    # tens of GB) never leaves a short archive that later reads as
    # "unexpected end of data". A short archive from an older run is
    # re-fetched the same way.
    for attempt in range(3):
        if archive.exists():
            break
        part = archive.with_suffix(".tgz.part")
        print(f"downloading {url}" + (f" (attempt {attempt + 1})" if attempt else ""))
        with urlopen(url) as resp, open(part, "wb") as out:
            expected = int(resp.headers.get("Content-Length") or 0)
            shutil.copyfileobj(resp, out, length=1 << 20)
        got = part.stat().st_size
        if expected and got != expected:
            print(f"warning: {url}: got {got} of {expected} bytes, retrying",
                  file=sys.stderr)
            part.unlink()
            continue
        part.rename(archive)
    if not archive.exists():
        raise RuntimeError(f"{url}: download failed after 3 attempts")
    print(f"extracting data/test from {archive}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        with tarfile.open(archive, "r|gz") as tf:
            for member in tf:
                parts = Path(member.name).parts
                # <lang>-<lang>/data/test/...
                if len(parts) < 4 or parts[1] != "data" or parts[2] != "test":
                    continue
                member.name = str(Path(*parts[3:]))
                tf.extract(member, extract_dir, filter="data")
                n += 1
    except tarfile.ReadError as e:
        # Truncated archive: drop it so the next run downloads it again.
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"{archive}: {e}; removed, rerun to re-download") from e
    if n == 0:
        raise RuntimeError(f"{archive}: no data/test members found")
    marker.write_text(f"{n} members\n")
    return extract_dir


def ingest_mtedx_longform(repo: Path, args: argparse.Namespace) -> int:
    lang = args.lang.lower()
    if lang not in MTEDX_LANGS:
        print(f"error: mTEDx has no '{lang}' (available: {', '.join(MTEDX_LANGS)})",
              file=sys.stderr)
        return 2

    raw_dir = repo / "samples/wer/raw"
    out_dir = repo / f"samples/wer/mtedx-longform-{lang}"
    manifest = repo / f"samples/wer/mtedx-longform-{lang}.manifest.jsonl"

    if manifest.exists() and not args.force:
        n_existing = sum(1 for _ in open(manifest))
        print(f"OK already exists: {manifest} ({n_existing} talks). "
              f"Pass --force to regenerate.")
        return 0

    test_dir = _mtedx_fetch_test(lang, raw_dir)
    seg_path = test_dir / "txt" / "segments"
    txt_path = test_dir / "txt" / f"test.{lang}"
    if not seg_path.exists() or not txt_path.exists():
        print(f"error: expected {seg_path} and {txt_path}", file=sys.stderr)
        return 2
    seg_lines = seg_path.read_text(encoding="utf-8").splitlines()
    txt_lines = txt_path.read_text(encoding="utf-8").splitlines()
    if len(seg_lines) != len(txt_lines):
        print(f"error: {seg_path} has {len(seg_lines)} lines but {txt_path} "
              f"has {len(txt_lines)}", file=sys.stderr)
        return 2

    # talk_id -> [(start, text)]
    talks: dict[str, list[tuple[float, str]]] = {}
    for seg, text in zip(seg_lines, txt_lines):
        fields = seg.split()
        if len(fields) != 4:
            print(f"error: bad segments line: {seg!r}", file=sys.stderr)
            return 2
        _, talk_id, start, _end = fields
        talks.setdefault(talk_id, []).append((float(start), text.strip()))

    import librosa

    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    n_converted = 0
    for talk_id in sorted(talks):
        src = test_dir / "wav" / f"{talk_id}.flac"
        if not src.exists():
            cands = list((test_dir / "wav").glob(f"{talk_id}.*"))
            if not cands:
                print(f"error: no audio for talk {talk_id} under {test_dir / 'wav'}",
                      file=sys.stderr)
                return 2
            src = cands[0]
        utt_id = f"mtedx-longform-{lang}-{talk_id}"
        wav_path = out_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            # Talks ship at the source rate (44.1/48 kHz): resample properly
            # (soxr via librosa) rather than the linear-interp helper.
            data, _ = librosa.load(str(src), sr=16000, mono=True)
            write_wav_16k_mono(np.asarray(data, dtype=np.float32), 16000, wav_path)
            n_converted += 1
        segs = sorted(talks[talk_id], key=lambda t: t[0])
        ref = " ".join(" ".join(t.split()) for _, t in segs if t)
        entries.append({
            "id": utt_id,
            "audio": str(wav_path),
            "ref_text": ref,
            "language": lang,
        })

    write_manifest(entries, manifest)
    print(f"manifest: {manifest}")
    print(f"  {len(entries)} talks ({n_converted} converted), "
          f"{sum(len(v) for v in talks.values())} reference segments")
    return 0


# -------- Dispatch --------------------------------------------------------

SOURCES = {
    "librispeech": ingest_librispeech,
    "fleurs": ingest_fleurs,
    "eka-medical-asr": ingest_eka_medical_asr,
    "tedlium-longform": ingest_tedlium_longform,
    "mtedx-longform": ingest_mtedx_longform,
}


def main() -> int:
    repo = find_repo_root(Path(__file__).parent)

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="source", required=True,
                           metavar="{librispeech,fleurs,eka-medical-asr,tedlium-longform,mtedx-longform}")

    p_ls = sub.add_parser("librispeech",
                          help="LibriSpeech split (English-only).")
    p_ls.add_argument("--split", default="test-clean",
                      choices=list(LIBRISPEECH_URLS),
                      help="LibriSpeech split (default: test-clean)")

    p_fl = sub.add_parser("fleurs",
                          help="FLEURS per-language split (102 languages).")
    p_fl.add_argument("--lang", required=True,
                      help="BCP-47 code; see FLEURS_LANGS in this file "
                           "for the supported set")
    p_fl.add_argument("--split", default="test",
                      choices=("test", "validation", "train"),
                      help="FLEURS split (default: test)")
    p_fl.add_argument("--force", action="store_true",
                      help="Regenerate even if manifest already exists.")

    p_ek = sub.add_parser("eka-medical-asr",
                          help="ekacare/eka-medical-asr-evaluation-dataset "
                               "(English + Hindi medical conversations).")
    p_ek.add_argument("--lang", required=True,
                      help="'en' or 'hi'")
    p_ek.add_argument("--split", default="test",
                      help="dataset split (default: test — the only split "
                           "ekacare publishes)")
    p_ek.add_argument("--force", action="store_true",
                      help="Regenerate even if manifest already exists.")

    p_tl = sub.add_parser("tedlium-longform",
                          help="distil-whisper/tedlium-long-form: the 11 full "
                               "TED-LIUM 3 test talks (long-form).")
    p_tl.add_argument("--force", action="store_true",
                      help="Regenerate even if manifest already exists.")

    p_mx = sub.add_parser("mtedx-longform",
                          help="OpenSLR 100 Multilingual TEDx: full test-split "
                               "talks for one language (long-form).")
    p_mx.add_argument("--lang", required=True,
                      help="one of " + ", ".join(MTEDX_LANGS))
    p_mx.add_argument("--force", action="store_true",
                      help="Regenerate even if manifest already exists.")

    args = p.parse_args()
    return SOURCES[args.source](repo, args)


if __name__ == "__main__":
    raise SystemExit(main())
