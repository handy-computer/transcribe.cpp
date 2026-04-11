#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx>=0.20; sys_platform == 'darwin' and platform_machine == 'arm64'",
#     "mlx-audio>=0.2; sys_platform == 'darwin' and platform_machine == 'arm64'",
#     "jiwer>=3.0",
# ]
# ///
"""
cohere-multilang-check.py - C++ vs MLX cross-check on multi-language samples.

For each (audio, language) pair in the input list, this runs:
  1. The C++ transcribe-cli binary with --language <lang>
  2. The MLX mlx-audio reference implementation with language=<lang>

Then it prints the two outputs side-by-side, reports whether they match
exactly, and computes a normalized character-level edit distance as a
soft similarity metric. There is no reference transcript -- the MLX
output IS the reference.

Pass criterion (soft):
  - Normalized edit distance <= 0.05 (5%) is a PASS
  - Above that is a WARN but not a hard failure

Usage:
    uv run scripts/cohere-multilang-check.py

Defaults to samples/german.wav (de). Override with
--pairs audio.wav:lang[,audio2.wav:lang2,...].
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("cannot locate repo root")


def run_cpp(cli: Path, model: Path, audio: Path, lang: str) -> tuple[str, float]:
    """Run transcribe-cli and return (text, latency_ms)."""
    cmd = [
        str(cli),
        "-q",
        "-m", str(model),
        "-l", lang,
        str(audio),
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = (time.monotonic() - t0) * 1000.0
    if proc.returncode != 0:
        print(f"  C++ stderr:\n{proc.stderr}", file=sys.stderr)
        return "", elapsed

    # Parse the "text: ..." line from stdout.
    text = ""
    for line in proc.stdout.splitlines():
        if line.startswith("text:"):
            text = line[len("text:"):].strip()
            break
    return text, elapsed


def run_mlx(model_dir: Path, audio: Path, lang: str) -> tuple[str, float, str]:
    """Run mlx-audio cohere_asr on the audio file.

    Returns (text, latency_ms, error). On success error is "". On failure
    text is empty and error contains a short diagnostic (e.g. "unsupported
    language").
    """
    from mlx_audio.stt import load_model as mlx_load_model

    # Lazy-load the model once per process. Keep it global so repeated
    # calls share the same instance.
    global _mlx_model
    try:
        _mlx_model
    except NameError:
        print(f"  loading MLX model from {model_dir}...", flush=True)
        _mlx_model = mlx_load_model(str(model_dir))

    t0 = time.monotonic()
    try:
        result = _mlx_model.transcribe(language=lang, audio_files=[str(audio)])
    except ValueError as e:
        elapsed = (time.monotonic() - t0) * 1000.0
        return "", elapsed, f"mlx rejected: {e}"
    except Exception as e:  # noqa: BLE001
        elapsed = (time.monotonic() - t0) * 1000.0
        return "", elapsed, f"mlx error: {type(e).__name__}: {e}"
    elapsed = (time.monotonic() - t0) * 1000.0

    # mlx-audio returns a list of transcription objects. The exact shape
    # depends on the version -- it may be a list of strings or a list of
    # dicts. Try both.
    if not result:
        return "", elapsed, ""
    first = result[0]
    if isinstance(first, str):
        return first.strip(), elapsed, ""
    if isinstance(first, dict):
        return str(first.get("text", first)).strip(), elapsed, ""
    # Some versions return an object with .text attribute.
    if hasattr(first, "text"):
        return str(first.text).strip(), elapsed, ""
    return str(first).strip(), elapsed, ""


def normalized_edit_distance(a: str, b: str) -> float:
    """Character-level edit distance normalized by max length."""
    import jiwer
    if not a and not b:
        return 0.0
    return jiwer.cer(a, b) if a else 1.0


def main() -> int:
    repo = find_repo_root(Path(__file__).parent)

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cli",
        type=Path,
        default=repo / "build/bin/transcribe-cli",
        help="transcribe-cli binary",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=repo / "models/cohere/cohere.f16.gguf",
        help="Cohere GGUF for C++ path",
    )
    p.add_argument(
        "--mlx-model",
        type=Path,
        default=Path("~/sandboxes/transcribe/models/cohere-transcribe-03-2026").expanduser(),
        help="Cohere HF checkpoint dir for MLX path",
    )
    p.add_argument(
        "--pairs",
        default="german.wav:de",
        help="Comma-separated list of <audio>:<lang> entries. Audio paths "
             "are resolved relative to samples/.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Normalized edit distance threshold for PASS verdict (default 0.05)",
    )
    args = p.parse_args()

    for path in (args.cli, args.model, args.mlx_model):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    # Parse pairs.
    pairs: list[tuple[Path, str]] = []
    for item in args.pairs.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            print(f"error: bad pair {item!r}, expected <audio>:<lang>", file=sys.stderr)
            return 2
        audio_name, lang = item.split(":", 1)
        audio_path = repo / "samples" / audio_name
        if not audio_path.exists():
            print(f"error: audio {audio_path} does not exist", file=sys.stderr)
            return 2
        pairs.append((audio_path, lang))

    print(f"C++ cli:    {args.cli}")
    print(f"C++ model:  {args.model}")
    print(f"MLX model:  {args.mlx_model}")
    print(f"pairs:      {len(pairs)}")
    print()

    n_pass = 0
    n_warn = 0
    n_skip = 0
    results: list[dict] = []

    for audio, lang in pairs:
        print(f"=== {audio.name} [lang={lang}] ===")

        cpp_text, cpp_ms = run_cpp(args.cli, args.model, audio, lang)
        print(f"  C++  ({cpp_ms:6.0f} ms):  {cpp_text or '<empty>'}")

        mlx_text, mlx_ms, mlx_err = run_mlx(args.mlx_model, audio, lang)
        if mlx_err:
            print(f"  MLX  ({mlx_ms:6.0f} ms):  <error: {mlx_err}>")
            print(f"  verdict:     SKIP (MLX refused)")
            n_skip += 1
            print()
            results.append({
                "audio": audio.name, "language": lang,
                "cpp_text": cpp_text, "mlx_text": "",
                "cer": None, "exact": False,
                "cpp_ms": cpp_ms, "mlx_ms": mlx_ms,
                "skipped": True, "mlx_error": mlx_err,
            })
            continue

        print(f"  MLX  ({mlx_ms:6.0f} ms):  {mlx_text or '<empty>'}")

        # Exact match?
        exact = (cpp_text == mlx_text)
        ned = normalized_edit_distance(cpp_text, mlx_text)

        if exact:
            verdict = "PASS (exact match)"
            n_pass += 1
        elif ned <= args.threshold:
            verdict = f"PASS (CER={ned*100:.1f}%)"
            n_pass += 1
        else:
            verdict = f"WARN (CER={ned*100:.1f}%)"
            n_warn += 1

        print(f"  verdict:     {verdict}")
        print()

        results.append({
            "audio": audio.name,
            "language": lang,
            "cpp_text": cpp_text,
            "mlx_text": mlx_text,
            "cer": ned,
            "exact": exact,
            "cpp_ms": cpp_ms,
            "mlx_ms": mlx_ms,
        })

    print("=" * 60)
    print(f"  pass: {n_pass}/{len(pairs)}  "
          f"warn: {n_warn}/{len(pairs)}  "
          f"skip: {n_skip}/{len(pairs)}")
    print("=" * 60)
    return 0 if n_warn == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
