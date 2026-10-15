#!/usr/bin/env python3
"""
run_reference_whistle_engine.py — Whistle reference transcripts from the
closed Cactus Needle engine (the Oracle), over a WER manifest.

Writes run.py-compatible JSONL for scripts/wer/score.py. The engine is
loaded via ctypes (scripts/lib/whistle_engine.py) at the pinned engine
and weights revisions; decode is the engine's compiled default search
(5 beams per the model card; there is no API knob). The language comes
from each manifest row (like run.py feeding transcribe-cli --language);
--language auto lets the engine detect it instead.

The engine rejects audio over 30 s. Use the le30s acceptance manifest
(docs/porting/families/whistle.md); longer rows are recorded as errors.

Usage (from repo root):
    uv run --project scripts/envs/whistle \\
      scripts/wer/run_reference_whistle_engine.py \\
        --manifest samples/wer/librispeech-test-clean-le30s.manifest.jsonl \\
        --out      reports/wer/whistle-REF.librispeech-test-clean-le30s.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib import whistle_engine as we  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True, help="Input manifest JSONL (id/audio/ref_text).")
    p.add_argument("--out", type=Path, required=True, help="Output JSONL path (run.py-compatible).")
    p.add_argument("--model", default=we.WEIGHTS_REPO, help="HF repo id (pinned) or a whistle.cact path")
    p.add_argument("--device", default="cpu", help="Accepted for the uniform contract; the engine is CPU-only.")
    p.add_argument("--batch-size", type=int, default=1, help="Accepted for the uniform contract; must be 1.")
    p.add_argument("--language", default=None,
                   help="Override every row's language; 'auto' lets the engine detect it.")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N utterances (0 = all).")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    if args.batch_size != 1:
        print("error: the engine transcribes one clip per call; --batch-size must be 1", file=sys.stderr)
        return 2
    cact = Path(args.model) if args.model.endswith(".cact") else None
    if cact is None and args.model != we.WEIGHTS_REPO:
        print(f"error: --model must be {we.WEIGHTS_REPO} or a .cact path", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    engine = we.Engine(cact=cact)
    load_ms = round((time.monotonic() - t0) * 1000, 1)

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[: args.limit]
    total = len(manifest)
    # score.py routes its normalizer by the header language ("en" -> EnglishTextNormalizer),
    # so record the manifest's language, not a placeholder.
    row_langs = {e.get("language") or "en" for e in manifest}
    header_lang = args.language if args.language not in (None, "auto", "detect") else (
        row_langs.pop() if len(row_langs) == 1 else None)
    print(f"engine:   {engine.lib_path}\nweights:  {engine.cact}")
    print(f"manifest: {args.manifest} ({total} utterances)\noutput:   {args.out}")

    n_done = n_errors = 0
    with open(args.out, "w") as fout:
        fout.write(json.dumps({"type": "batch_header", "load_ms": load_ms, **we.provenance(),
                               "device": "cpu", "search": "engine default (5-beam per model card)",
                               "language": header_lang,
                               "language_mode": args.language or "per-row"}) + "\n")
        t_loop = time.monotonic()
        for entry in manifest:
            lang = args.language or entry.get("language") or "en"
            lang = None if lang in ("auto", "detect") else lang
            t_start = time.monotonic()
            err, res = "", {}
            try:
                res = engine.transcribe(we.load_audio(entry["audio"]), language=lang)
            except Exception as e:  # noqa: BLE001 - recorded per row, scored as an error
                err = f"{type(e).__name__}: {e}"
                n_errors += 1
            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
            fout.write(json.dumps({
                "id": entry["id"],
                "ref_text": entry.get("ref_text", ""),
                "hyp_text": res.get("text", ""),
                "raw_text": res.get("text", ""),
                "language": res.get("language", ""),
                "n_tokens": res.get("n_tokens", 0),
                "mel_ms": 0,
                "encode_ms": 0,
                "decode_ms": elapsed_ms,
                "latency_ms": elapsed_ms,
                "error": err,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 100 == 0 or n_done == total:
                wall = time.monotonic() - t_loop
                rate = n_done / wall if wall > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{total}] {rate:.2f} utt/s, ETA {eta / 60:.1f} min, errors={n_errors}",
                      flush=True)

    wall = time.monotonic() - t_loop
    print(f"\ndone. {n_done} utterances in {wall:.1f}s ({n_done / wall:.2f} utt/s), {n_errors} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
