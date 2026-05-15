#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
run.py — transcribe-cli driver for WER evaluation.

Reads a manifest.jsonl, invokes transcribe-cli in --batch mode (model
loads once, processes all utterances in a single process), and writes
hypotheses + per-utterance timings to a report JSONL file.

Usage:
    # Existing flow: explicit manifest
    uv run scripts/wer/run.py \\
        --model models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf \\
        --manifest samples/wer/test-clean.manifest.jsonl

    # New: dataset shorthand. Resolves the manifest path and runs
    # ingest.py to fetch the data on first use. Auto-sets --language.
    uv run scripts/wer/run.py \\
        --model models/Fun-ASR-MLT-Nano-2512/Fun-ASR-MLT-Nano-2512-F16.gguf \\
        --dataset fleurs:es

    # --language CODE applies the language hint to every utterance in the
    # batch. transcribe-cli only supports one language per batch run, so a
    # manifest with mixed languages must be split before scoring.
    uv run scripts/wer/run.py \\
        --model ...gguf --manifest samples/wer/fleurs-zh.manifest.jsonl \\
        --language zh

  Options:
    --out PATH      output report file (default: auto-derived from model,
                    manifest, and timestamp mode)
    --cli PATH      transcribe-cli binary (default: build/bin/transcribe-cli)
    --dataset SPEC  shorthand for ingest + manifest. Supported forms:
                      fleurs:<code>        -> samples/wer/fleurs-<code>.manifest.jsonl
                      librispeech:<split>  -> samples/wer/librispeech-<split>.manifest.jsonl
                    Runs ingest.py if the manifest is missing. Overrides
                    --manifest. Auto-sets --language when applicable.
    --language CODE BCP-47 hint passed through to transcribe-cli. If the
                    manifest's entries declare a language and it disagrees
                    with this flag, run.py fails loud. If --language is
                    omitted and the manifest has a single consistent
                    language, it is inferred automatically.

Output JSONL:
    - First line (batch header):
        {"type": "batch_header", "load_ms": ...}
      Captured once from transcribe-cli's --batch-jsonl header line. score.py
      and compare.py ignore this line.
    - Remaining lines (one per utterance):
        {"id": "...", "ref_text": "...", "hyp_text": "...", "mel_ms": ...,
         "encode_ms": ..., "decode_ms": ...}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("cannot locate repo root")


def derive_out_path(
    repo: Path, model: Path, manifest: Path, timestamps: str
) -> Path:
    model_stem = model.stem
    dataset = manifest.stem.replace(".manifest", "")
    dataset = f"{dataset}-timestamps_{timestamps}"
    return repo / "reports" / "wer" / f"{model_stem}.{dataset}.jsonl"


def resolve_dataset(repo: Path, spec: str) -> tuple[Path, str | None]:
    """Parse --dataset SPEC, run ingest.py if needed, return manifest path
    and the default BCP-47 language for the dataset.

    Supported specs:
      fleurs:<code>        e.g. fleurs:es, fleurs:zh-tw
      librispeech:<split>  e.g. librispeech:test-clean

    Adding a new protocol means adding a branch here and a matching
    adapter in ingest.py.
    """
    if ":" not in spec:
        raise SystemExit(
            f"error: --dataset must be PROTO:ARG (e.g. fleurs:es), got {spec!r}"
        )
    proto, arg = spec.split(":", 1)
    if proto == "fleurs":
        manifest = repo / f"samples/wer/fleurs-{arg}.manifest.jsonl"
        ingest_args = ["fleurs", "--lang", arg]
        default_lang: str | None = arg
    elif proto == "librispeech":
        manifest = repo / f"samples/wer/librispeech-{arg}.manifest.jsonl"
        ingest_args = ["librispeech", "--split", arg]
        default_lang = "en"
    else:
        raise SystemExit(
            f"error: unknown --dataset protocol {proto!r}; "
            f"supported: fleurs, librispeech"
        )

    if not manifest.exists():
        print(f"manifest missing; running scripts/wer/ingest.py {' '.join(ingest_args)}")
        r = subprocess.run(
            ["uv", "run", str(repo / "scripts/wer/ingest.py"), *ingest_args],
            check=False,
        )
        if r.returncode != 0:
            raise SystemExit(
                f"error: ingest.py {' '.join(ingest_args)} exited {r.returncode}"
            )
        if not manifest.exists():
            raise SystemExit(
                f"error: ingest.py succeeded but {manifest} was not created"
            )
    return manifest, default_lang


def main() -> int:
    repo = find_repo_root(Path(__file__).parent)

    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True,
                   help="GGUF model file")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Input manifest (default: "
                        "samples/wer/test-clean.manifest.jsonl). "
                        "Overridden by --dataset.")
    p.add_argument("--dataset", default=None,
                   help="Dataset shorthand: fleurs:<code> or "
                        "librispeech:<split>. Runs ingest.py on first use; "
                        "auto-sets --language.")
    p.add_argument("--language", default=None,
                   help="BCP-47 code passed to transcribe-cli --language. "
                        "Auto-set from --dataset when applicable. "
                        "If a manifest entry's language disagrees, fail loud.")
    p.add_argument("--cli", type=Path,
                   default=repo / "build/bin/transcribe-cli",
                   help="transcribe-cli binary")
    p.add_argument("--out", type=Path, default=None,
                   help="Output report file (default: auto-derived)")
    p.add_argument("--backend",
                   choices=("auto", "cpu", "cpu_accel", "metal", "vulkan"),
                   default=None,
                   help="Compute backend (default: transcribe-cli default)")
    p.add_argument("--kv-type",
                   choices=("auto", "f32", "f16"),
                   default=None,
                   help="Flash-attn KV cache type passthrough")
    p.add_argument("--timestamps",
                   choices=("auto", "none", "segment", "word", "token"),
                   default="none",
                   help="Timestamp granularity passthrough (default: none)")
    p.add_argument("--stream-chunk-ms", type=int, default=0,
                   help="When > 0, drive each utterance through the "
                        "streaming API in N-ms chunks. Requires a model "
                        "that advertises supports_streaming.")
    p.add_argument("--stream-att-right", type=int, default=None,
                   help="Parakeet streaming: pick a right-context "
                        "(lookahead) setting from the model's training "
                        "menu. Ignored unless --stream-chunk-ms > 0.")
    args = p.parse_args()

    # Resolve --dataset first (may run ingest.py). --dataset takes
    # precedence over --manifest; if neither is set fall back to the
    # historical default for backward compat.
    if args.dataset:
        if args.manifest:
            print(f"warning: --dataset overrides --manifest", file=sys.stderr)
        args.manifest, default_lang = resolve_dataset(repo, args.dataset)
        if args.language is None:
            args.language = default_lang
    elif args.manifest is None:
        args.manifest = repo / "samples/wer/test-clean.manifest.jsonl"

    for path in (args.model, args.manifest, args.cli):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    out_path = args.out or derive_out_path(
        repo, args.model, args.manifest, args.timestamps
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load manifest and build id→ref_text lookup.
    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    total = len(manifest)

    # Cross-check / infer --language against the manifest's language
    # field. transcribe-cli batch mode applies one language per run, so a
    # mixed-language manifest is a user error here.
    manifest_langs = {
        e["language"] for e in manifest if e.get("language")
    }
    if args.language is None and len(manifest_langs) == 1:
        args.language = next(iter(manifest_langs))
        print(f"language: inferred {args.language!r} from manifest")
    elif args.language and manifest_langs and manifest_langs != {args.language}:
        bad = sorted(manifest_langs - {args.language})
        print(
            f"error: --language {args.language!r} disagrees with manifest "
            f"entries (also seen: {bad}). transcribe-cli batch mode applies "
            f"one language per run; split the manifest by language first.",
            file=sys.stderr,
        )
        return 2
    elif args.language is None and len(manifest_langs) > 1:
        print(
            f"error: manifest has mixed languages {sorted(manifest_langs)}; "
            f"pass --language to pick one (and split the manifest by "
            f"language if you want every language scored).",
            file=sys.stderr,
        )
        return 2

    # Build audio→entry lookup keyed on audio path.
    audio_to_entry: dict[str, dict] = {}
    for e in manifest:
        audio_to_entry[e["audio"]] = e

    print(f"model:    {args.model}")
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"language: {args.language or '(default)'}")
    print(f"output:   {out_path}")
    print(f"mode:     batch (single process, model loads once)")

    # Write the batch file list (one wav path per line).
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".list", delete=False
    ) as tf:
        batch_path = tf.name
        for e in manifest:
            tf.write(e["audio"] + "\n")

    # Invoke transcribe-cli in batch mode.
    cmd = [
        str(args.cli), "-q",
        "-m", str(args.model),
        "--batch", batch_path,
        "--batch-jsonl",
    ]
    if args.backend:
        cmd += ["--backend", args.backend]
    if args.kv_type:
        cmd += ["--kv-type", args.kv_type]
    if args.language:
        cmd += ["--language", args.language]
    cmd += ["--timestamps", args.timestamps]
    if args.stream_chunk_ms > 0:
        cmd += ["--stream-chunk-ms", str(args.stream_chunk_ms)]
        if args.stream_att_right is not None:
            cmd += ["--stream-att-right", str(args.stream_att_right)]
    print(f"  $ {' '.join(cmd[:6])} ...")

    t_start = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",
        )
    except Exception as e:
        print(f"error: failed to start transcribe-cli: {e}", file=sys.stderr)
        return 1

    n_done = 0
    n_errors = 0

    with open(out_path, "w") as fout:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            # CLI emits a one-shot batch_header before any per-file line.
            # Persist it as the first record in the output JSONL, augmented
            # with the run-level language so score.py can auto-route the
            # metric/normalizer without the caller repeating --language.
            if result.get("type") == "batch_header":
                if args.language:
                    result["language"] = args.language
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                continue

            audio_path = result.get("file", "")
            entry = audio_to_entry.get(audio_path, {})

            out_entry = {
                "id": entry.get("id", Path(audio_path).stem),
                "ref_text": entry.get("ref_text", ""),
                "hyp_text": result.get("text", ""),
                "mel_ms": result.get("mel_ms", 0),
                "encode_ms": result.get("encode_ms", 0),
                "decode_ms": result.get("decode_ms", 0),
                "error": result.get("error", ""),
            }
            # Compute total latency for backwards compat with score.py.
            out_entry["latency_ms"] = round(
                out_entry["mel_ms"] + out_entry["encode_ms"] +
                out_entry["decode_ms"], 1
            )
            if out_entry.get("error"):
                n_errors += 1

            fout.write(json.dumps(out_entry) + "\n")
            fout.flush()
            n_done += 1

            if n_done % 200 == 0 or n_done == total:
                elapsed = time.monotonic() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{total}] "
                      f"{rate:.1f} utt/s, ETA {eta:.0f}s, "
                      f"errors={n_errors}")

    proc.wait()
    wall = time.monotonic() - t_start

    # Clean up temp file.
    Path(batch_path).unlink(missing_ok=True)

    print(f"\ndone. {n_done} utterances in {wall:.1f}s "
          f"({n_done/wall:.1f} utt/s), {n_errors} errors")
    print(f"report: {out_path}")
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
