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
  uv run scripts/wer/ingest.py fleurs --lang zh-tw

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
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import soundfile as sf


# -------- BCP-47 → FLEURS config mapping ---------------------------------
#
# FLEURS uses underscore-region codes. Region-ambiguous BCP-47 codes
# (zh, no) pick a sensible default; explicit regional aliases live
# alongside (zh-cn, zh-tw). FLEURS only ships one regional variant for
# most languages, so most entries are unambiguous.
#
# Full FLEURS coverage (102 languages). Add aliases here when a new
# BCP-47 → config shorthand is wanted.
FLEURS_LANGS: dict[str, str] = {
    # African
    "af": "af_za", "am": "am_et", "ff": "ff_sn", "ha": "ha_ng",
    "ig": "ig_ng", "kam": "kam_ke", "kea": "kea_cv", "lg": "lg_ug",
    "ln": "ln_cd", "luo": "luo_ke", "nso": "nso_za", "ny": "ny_mw",
    "om": "om_et", "sn": "sn_zw", "so": "so_so", "sw": "sw_ke",
    "umb": "umb_ao", "wo": "wo_sn", "xh": "xh_za", "yo": "yo_ng",
    "zu": "zu_za",
    # Arabic, Hebrew, Persian, Kurdish
    "ar": "ar_eg", "he": "he_il", "fa": "fa_ir", "ckb": "ckb_iq",
    "ps": "ps_af", "ur": "ur_pk",
    # South Asian
    "as": "as_in", "bn": "bn_in", "gu": "gu_in", "hi": "hi_in",
    "kn": "kn_in", "ml": "ml_in", "mr": "mr_in", "ne": "ne_np",
    "or": "or_in", "pa": "pa_in", "sd": "sd_in", "ta": "ta_in",
    "te": "te_in",
    # East / Southeast Asian
    "my": "my_mm", "fil": "fil_ph", "tl": "fil_ph",  # tl alias
    "id": "id_id", "ja": "ja_jp", "jv": "jv_id", "km": "km_kh",
    "ko": "ko_kr", "lo": "lo_la", "ms": "ms_my", "th": "th_th",
    "vi": "vi_vn", "ceb": "ceb_ph",
    # Chinese / Cantonese
    "zh": "cmn_hans_cn",           # simplified Mandarin (default)
    "zh-cn": "cmn_hans_cn",        # simplified Mandarin (explicit)
    "zh-tw": "cmn_hant_tw",        # traditional Mandarin
    "yue": "yue_hant_hk",          # Cantonese (traditional)
    # Central Asian
    "az": "az_az", "kk": "kk_kz", "ky": "ky_kg", "mn": "mn_mn",
    "tg": "tg_tj", "uz": "uz_uz", "hy": "hy_am", "ka": "ka_ge",
    # European
    "ast": "ast_es", "be": "be_by", "bg": "bg_bg", "bs": "bs_ba",
    "ca": "ca_es", "cs": "cs_cz", "cy": "cy_gb", "da": "da_dk",
    "de": "de_de", "el": "el_gr", "en": "en_us", "es": "es_419",
    "et": "et_ee", "fi": "fi_fi", "fr": "fr_fr", "ga": "ga_ie",
    "gl": "gl_es", "hr": "hr_hr", "hu": "hu_hu", "is": "is_is",
    "it": "it_it", "lb": "lb_lu", "lt": "lt_lt", "lv": "lv_lv",
    "mi": "mi_nz", "mk": "mk_mk", "mt": "mt_mt", "nb": "nb_no",
    "no": "nb_no",                 # Norwegian macro → Bokmål
    "nl": "nl_nl", "oc": "oc_fr", "pl": "pl_pl", "pt": "pt_br",
    "ro": "ro_ro", "ru": "ru_ru", "sk": "sk_sk", "sl": "sl_si",
    "sr": "sr_rs", "sv": "sv_se", "tr": "tr_tr", "uk": "uk_ua",
}


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


# -------- Dispatch --------------------------------------------------------

SOURCES = {
    "librispeech": ingest_librispeech,
    "fleurs": ingest_fleurs,
    "eka-medical-asr": ingest_eka_medical_asr,
}


def main() -> int:
    repo = find_repo_root(Path(__file__).parent)

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="source", required=True,
                           metavar="{librispeech,fleurs,eka-medical-asr}")

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

    args = p.parse_args()
    return SOURCES[args.source](repo, args)


if __name__ == "__main__":
    raise SystemExit(main())
