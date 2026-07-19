#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""
validate.py - convention-driven numerical validation for transcribe.cpp.

Orchestrates the full validation flow: generate reference dumps, generate
C++ dumps, and compare tensors. All paths are derived from convention;
the only required input is the family name.

Usage:
    uv run scripts/validate.py ref     --family cohere
    uv run scripts/validate.py cpp     --family cohere
    uv run scripts/validate.py compare --family cohere
    uv run scripts/validate.py all     --family cohere

    # Override model source (local path instead of HF download)
    uv run scripts/validate.py all --family cohere --model /local/path

    # Override GGUF path
    uv run scripts/validate.py cpp --family cohere --gguf models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-BF16.gguf

Conventions:
    Manifest:    tests/golden/{family}/*.manifest.json
    Dump script: manifest reference.entrypoint
    Python env:  scripts/envs/{family}/
    Tolerances:  tests/tolerances/{family}.json
    Audio:       samples/jfk.wav
    GGUF:        models/{slug}/ where slug starts with {family}
                 (prefers *-BF16.gguf > *-F32.gguf > *-F16.gguf > first match)
    Ref output:  build/validate/{family}/{variant}/{case}/ref/
    C++ output:  build/validate/{family}/{variant}/{case}/cpp/
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def normalize_text_for_compare(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]+", " ", text.lower()).split())


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise SystemExit("error: cannot locate transcribe.cpp repo root")


def load_manifest(
    repo: Path, family: str, variant: str | None = None
) -> dict[str, Any]:
    manifest_dir = repo / "tests" / "golden" / family
    if variant:
        path = manifest_dir / f"{variant}.manifest.json"
        if not path.exists():
            raise SystemExit(f"error: manifest not found: {path}")
        return json.loads(path.read_text())

    pattern = manifest_dir / "*.manifest.json"
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        raise SystemExit(
            f"error: no manifest found for family '{family}' "
            f"(looked in tests/golden/{family}/)"
        )
    if len(matches) > 1:
        names = [Path(m).stem.replace(".manifest", "") for m in matches]
        raise SystemExit(
            f"error: multiple manifests for '{family}': {names}\n"
            f"  Use --variant to pick one"
        )
    return json.loads(Path(matches[0]).read_text())


def manifest_source_model(manifest: dict[str, Any]) -> str:
    source_model = manifest.get("source_model")
    if not isinstance(source_model, dict):
        raise SystemExit("error: manifest missing source_model object")
    hf_repo = source_model.get("hf_repo")
    if not hf_repo:
        raise SystemExit("error: manifest missing source_model.hf_repo")
    return str(hf_repo)


def manifest_reference(manifest: dict[str, Any]) -> dict[str, Any]:
    reference = manifest.get("reference")
    if not isinstance(reference, dict):
        raise SystemExit("error: manifest missing reference object")
    if not reference.get("kind"):
        raise SystemExit("error: manifest missing reference.kind")
    if not reference.get("source"):
        raise SystemExit("error: manifest missing reference.source")
    if not reference.get("entrypoint"):
        raise SystemExit("error: manifest missing reference.entrypoint")
    return reference


def manifest_dump_script(repo: Path, manifest: dict[str, Any]) -> Path:
    reference = manifest_reference(manifest)
    script = repo / str(reference["entrypoint"])
    if not script.exists():
        raise SystemExit(f"error: reference entrypoint not found: {script}")
    return script


def case_audio(case) -> str:
    """A case is either a bare string (legacy) or a dict with at least
    {audio: <stem>}. Returns the audio stem."""
    if isinstance(case, str):
        return case
    if isinstance(case, dict) and "audio" in case:
        return str(case["audio"])
    raise SystemExit(f"error: malformed case in manifest: {case!r}")


def case_language(case) -> str | None:
    """Per-case language code.

    Legacy bare-string cases and dict cases that omit the field keep the
    historical default of "en". Dict cases may set language to "auto",
    "detect", or null to exercise the model's native language-detection path;
    callers should omit the language flag in that case.
    """
    if isinstance(case, dict):
        value = case.get("language", "en")
        if value is None:
            return None
        language = str(value)
        if language == "" or language.lower() in {"auto", "detect"}:
            return None
        return language
    return "en"


def case_transcript_compare(manifest: dict[str, Any], case) -> str:
    value = manifest.get("transcript_compare", "exact")
    if isinstance(case, dict) and "transcript_compare" in case:
        value = case["transcript_compare"]
    if value is None:
        value = "exact"
    mode = str(value).lower()
    if mode not in {"exact", "normalized", "dediarized"}:
        raise SystemExit(
            f"error: unsupported transcript_compare={value!r}; "
            "expected 'exact', 'normalized', or 'dediarized'"
        )
    return mode


def dediarize_text(text: str) -> str:
    """Strip inline [start]/[Sxx]/[end] bracket spans from an emergent-diarization
    transcript, then normalize. The reference `text` field is already de-diarized,
    so this is a no-op there; the C++ runtime returns the raw diarized transcript,
    so bracket spans (whose numeric timestamps carry bf16-vs-f32 digit jitter) are
    removed before comparison. Mirrors the WER harness de-diarization."""
    stripped = re.sub(r"\[[^\]]*\]", " ", text)
    return normalize_text_for_compare(stripped)


def text_for_compare(text: str, mode: str) -> str:
    if mode == "exact":
        return text
    if mode == "dediarized":
        return dediarize_text(text)
    return normalize_text_for_compare(text)


def find_gguf(repo: Path, family: str, slug: str | None = None,
              variant: str | None = None) -> Path:
    """Find a GGUF under models/.

    Discovery order:
      1. If `variant` is provided and `models/<variant>/<variant>-<Q>.gguf`
         exists, prefer it. This disambiguates families like gigaam where
         multiple manifest variants share a single hf_repo (and therefore
         a single hf-derived slug) but ship as separate per-variant GGUF
         directories.
      2. If `slug` is provided (derived from the manifest's
         source_model.hf_repo, e.g. "Qwen3-ASR-0.6B" from
         "Qwen/Qwen3-ASR-0.6B"), look for
         models/<slug>/<slug>-<QUANT>.gguf directly. This is the
         converter's output convention and is case-accurate — which
         matters for families whose HF slug does not case-fold to the
         family key (e.g. family="qwen3_asr", slug="Qwen3-ASR-0.6B").
      3. Legacy fallback: scan models/*/ for any GGUF whose stem
         starts with `family`. Kept so older manifests (or manual
         layouts) still work.

    Preferred quant order: BF16 > F32 > F16 > first match. If multiple
    variants match the fallback, the first sort-order wins; use
    --gguf to pick explicitly.
    """
    model_root = repo / "models"
    if not model_root.is_dir():
        raise SystemExit(f"error: model root not found: {model_root}")

    preferred_quants = ["BF16", "F32", "F16"]

    def lookup_in(name: str) -> Path | None:
        variant_dir = model_root / name
        if not variant_dir.is_dir():
            return None
        for quant in preferred_quants:
            candidate = variant_dir / f"{name}-{quant}.gguf"
            if candidate.exists():
                return candidate
        matches = sorted(variant_dir.glob(f"{name}-*.gguf"))
        return matches[0] if matches else None

    # 1. Manifest-variant-driven lookup (handles multi-variant families
    # sharing a single hf_repo, e.g. gigaam).
    if variant:
        found = lookup_in(variant)
        if found is not None:
            return found

    # 2. Manifest-slug-driven lookup.
    if slug:
        found = lookup_in(slug)
        if found is not None:
            return found

    # 2. Legacy family-prefix fallback.
    def for_family(paths: list[Path]) -> list[Path]:
        return [p for p in paths if p.stem.startswith(family)]

    for quant in preferred_quants:
        matches = for_family(sorted(model_root.glob(f"*/*-{quant}.gguf")))
        if matches:
            return matches[0]
    matches = for_family(sorted(model_root.glob("*/*.gguf")))
    if matches:
        return matches[0]

    hint = f" (manifest slug '{slug}' also matched nothing)" if slug else ""
    raise SystemExit(
        f"error: no GGUF files found under {model_root} for family "
        f"'{family}'{hint}.\n"
        f"  Convert a GGUF first or set --gguf."
    )


def find_cli(repo: Path) -> Path:
    for candidate in [
        repo / "build" / "bin" / "transcribe-cli",
        repo / "build" / "transcribe-cli",
    ]:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "error: transcribe-cli not found in build/bin/\n"
        "  Run: cmake --build build --target transcribe-cli\n"
        "  For --mel-from-ref and per-layer dumps, configure with\n"
        "  -DTRANSCRIBE_ENABLE_VALIDATION_HOOKS=ON (or: cmake --preset validation)."
    )


def validation_hooks_enabled(repo: Path) -> bool:
    """True if build/ was configured with the validation hooks compiled in.

    Reads build/CMakeCache.txt next to the CLI that find_cli() runs. Used to
    hard-fail --mel-from-ref against a build that would otherwise silently
    ignore it (the hook code is compiled out unless the flag is ON).
    """
    cache = repo / "build" / "CMakeCache.txt"
    try:
        for line in cache.read_text().splitlines():
            if line.startswith("TRANSCRIBE_ENABLE_VALIDATION_HOOKS:"):
                return line.rstrip().endswith("=ON")
    except OSError:
        pass
    return False


def run_cmd(cmd: list[str], repo: Path, label: str) -> None:
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  {label}", file=sys.stderr)
    print(f"  {' '.join(cmd)}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    env = os.environ.copy()
    # Prevent uv from inheriting an active venv.
    if len(cmd) >= 2 and cmd[0] == "uv" and cmd[1] == "run":
        env.pop("VIRTUAL_ENV", None)

    result = subprocess.run(cmd, cwd=repo, env=env)
    if result.returncode != 0:
        raise SystemExit(
            f"error: {label} failed with exit code {result.returncode}"
        )


def parse_cli_result(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("type") != "batch_header"
            and isinstance(payload.get("text"), str)
        ):
            return payload

    for line in reversed(output.splitlines()):
        if line.startswith("text: "):
            return {"text": line[len("text: "):], "words": []}
    return None


def parse_cli_transcript(output: str) -> str | None:
    payload = parse_cli_result(output)
    return str(payload["text"]) if payload is not None else None


def write_cpp_transcript(
    out_dir: Path,
    *,
    family: str,
    variant: str,
    case: str,
    gguf: Path,
    backend: str,
    text: str,
    words: list[dict[str, Any]],
    tokens: list[dict[str, Any]] | None,
) -> None:
    path = out_dir / "transcript.json"
    payload = {
        "schema": "transcribe-cpp-transcript-v2",
        "family": family,
        "variant": variant,
        "case": case,
        "text": text,
        "normalized_text": normalize_text(text),
        "words": words,
        "source": {
            "kind": "transcribe.cpp",
            "gguf": str(gguf),
            "backend": backend,
        },
    }
    if tokens is not None:
        payload["tokens"] = tokens
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {path}", file=sys.stderr)


def timestamp_tolerance_ms(path: Path | None) -> float | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    entry = payload.get("timestamps")
    if entry is None:
        return None
    if not isinstance(entry, dict) or "max_abs_ms" not in entry:
        raise SystemExit(
            f"error: {path}: timestamps must contain max_abs_ms"
        )
    value = float(entry["max_abs_ms"])
    if not math.isfinite(value) or value < 0.0:
        raise SystemExit(
            f"error: {path}: timestamps.max_abs_ms must be finite and non-negative"
        )
    return value


def case_timestamp_compare(
    reference: dict[str, Any],
    cpp: dict[str, Any],
    *,
    text_mode: str,
    tolerance_ms: float | None,
) -> dict[str, Any]:
    ref_words = reference.get("words")
    cpp_words = cpp.get("words")
    if not isinstance(ref_words, list):
        return {"match": False, "reason": "reference words field is not a list"}
    if not isinstance(cpp_words, list):
        return {"match": False, "reason": "C++ words field is not a list"}
    if tolerance_ms is None:
        return {
            "match": False,
            "reason": "timestamp tolerance is missing from the tolerance file",
        }

    def word_texts(rows: list[Any], side: str) -> tuple[list[str] | None, str | None]:
        texts: list[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or "text" not in row:
                return None, f"{side} word {index} lacks a text field"
            texts.append(str(row["text"]))
        return texts, None

    ref_texts, error = word_texts(ref_words, "reference")
    if error is not None:
        return {"match": False, "reason": error}
    cpp_texts, error = word_texts(cpp_words, "C++")
    if error is not None:
        return {"match": False, "reason": error}
    assert ref_texts is not None and cpp_texts is not None

    ref_aligned = [text_for_compare(text, text_mode) for text in ref_texts]
    cpp_aligned = [text_for_compare(text, text_mode) for text in cpp_texts]
    if len(ref_words) != len(cpp_words):
        divergent_index = next(
            (
                index
                for index, (ref_text, cpp_text) in enumerate(
                    zip(ref_aligned, cpp_aligned)
                )
                if ref_text != cpp_text
            ),
            min(len(ref_aligned), len(cpp_aligned)),
        )
        window_start = max(0, divergent_index - 2)
        window_end = divergent_index + 3
        return {
            "match": False,
            "reason": (
                f"word count mismatch: reference={len(ref_words)}, "
                f"C++={len(cpp_words)}; first divergent index={divergent_index}; "
                f"reference[{window_start}:{min(window_end, len(ref_aligned))}]="
                f"{ref_aligned[window_start:window_end]!r}; "
                f"C++[{window_start}:{min(window_end, len(cpp_aligned))}]="
                f"{cpp_aligned[window_start:window_end]!r}"
            ),
        }

    for index, (ref_text, cpp_text) in enumerate(zip(ref_aligned, cpp_aligned)):
        if ref_text != cpp_text:
            return {
                "match": False,
                "reason": (
                    f"word text mismatch at index {index}: "
                    f"reference={ref_texts[index]!r} ({ref_text!r}), "
                    f"C++={cpp_texts[index]!r} ({cpp_text!r})"
                ),
            }

    starts: list[float] = []
    ends: list[float] = []
    for index, (ref_word, cpp_word) in enumerate(zip(ref_words, cpp_words)):
        try:
            ref_t0_ms = float(ref_word["start_s"]) * 1000.0
            ref_t1_ms = float(ref_word["end_s"]) * 1000.0
            cpp_t0_ms = float(cpp_word["t0_ms"])
            cpp_t1_ms = float(cpp_word["t1_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "match": False,
                "reason": f"invalid timestamp fields at word {index}: {exc}",
            }
        values = (ref_t0_ms, ref_t1_ms, cpp_t0_ms, cpp_t1_ms)
        if not all(math.isfinite(value) for value in values):
            return {
                "match": False,
                "reason": f"non-finite timestamp at word {index}: {values!r}",
            }
        starts.append(abs(cpp_t0_ms - ref_t0_ms))
        ends.append(abs(cpp_t1_ms - ref_t1_ms))

    deviations = starts + ends
    max_deviation_ms = max(deviations, default=0.0)
    mean_deviation_ms = (
        sum(deviations) / len(deviations) if deviations else 0.0
    )
    within_tolerance = max_deviation_ms <= tolerance_ms
    result = {
        "match": within_tolerance,
        "n_words": len(ref_words),
        "tolerance_ms": tolerance_ms,
        "max_deviation_ms": max_deviation_ms,
        "mean_deviation_ms": mean_deviation_ms,
        "max_start_deviation_ms": max(starts, default=0.0),
        "mean_start_deviation_ms": sum(starts) / len(starts) if starts else 0.0,
        "max_end_deviation_ms": max(ends, default=0.0),
        "mean_end_deviation_ms": sum(ends) / len(ends) if ends else 0.0,
    }
    if not result["match"]:
        result["reason"] = (
            f"max endpoint deviation {max_deviation_ms:.3f} ms exceeds "
            f"{tolerance_ms:.3f} ms"
        )
    return result


def cmd_ref(args: argparse.Namespace) -> int:
    repo = find_repo_root(Path(__file__).parent)
    manifest = load_manifest(repo, args.family, getattr(args, "variant", None))
    variant = manifest["variant"]
    reference = manifest_reference(manifest)
    model = args.model or manifest_source_model(manifest)
    if not model:
        raise SystemExit("error: no model specified and none in manifest")

    dump_script = manifest_dump_script(repo, manifest)
    env_dir = repo / "scripts" / "envs" / args.family

    cases = manifest.get("cases", ["jfk"])
    for case in cases:
        case_name = case_audio(case)
        language = case_language(case)
        audio = repo / "samples" / f"{case_name}.wav"
        if not audio.exists():
            raise SystemExit(f"error: audio not found: {audio}")

        out_dir = repo / "build" / "validate" / args.family / variant / case_name / "ref"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        base_args = [
            "uv", "run", "--project", str(env_dir),
            str(dump_script),
        ]
        common_args = [
            "--model", str(model),
            "--audio", str(audio),
            "--out", str(out_dir),
            "--torch-threads", "1",
        ]
        if language is not None:
            common_args += ["--language", language]

        # Pin the HF revision when the manifest declares one and the
        # family's dumper accepts --revision. The dumper itself ignores
        # --revision when --model resolves to a local directory.
        hf_revision = (manifest.get("source_model") or {}).get("hf_revision")
        if hf_revision and args.family in ("qwen3_asr", "granite_nar"):
            common_args += ["--revision", str(hf_revision)]

        # Forward any manifest-declared dumper args verbatim. Used today
        # for parakeet-unified-en-0.6b, which needs --offline-only to
        # force att_context_style='regular' (its v1 C++ port targets
        # offline / full-context mode). Cache-aware streaming variants
        # like nemotron-speech-streaming-en-0.6b deliberately omit this
        # so their native chunked_limited style is preserved.
        extra_dump_args = reference.get("dump_args") or []
        if not isinstance(extra_dump_args, list) or not all(
            isinstance(a, str) for a in extra_dump_args
        ):
            raise SystemExit(
                f"error: manifest reference.dump_args must be a list of strings; "
                f"got {extra_dump_args!r}"
            )
        common_args += list(extra_dump_args)

        # Run both encoder and decode subcommands. Some families dump
        # everything from decode (cohere); others split encoder and
        # decoder intermediates across subcommands (parakeet). Running
        # both is safe — if a tensor is dumped by both, the decode
        # pass overwrites the encoder pass (same values).
        for stage in ["encoder", "decode"]:
            cmd = base_args + [stage] + common_args
            run_cmd(
                cmd,
                repo,
                f"ref {stage} [{args.family}/{variant}/{case_name}/{reference['kind']}]",
            )

    return 0


def cmd_cpp(args: argparse.Namespace) -> int:
    repo = find_repo_root(Path(__file__).parent)
    manifest = load_manifest(repo, args.family, getattr(args, "variant", None))
    variant = manifest["variant"]
    cli = find_cli(repo)
    slug = manifest_source_model(manifest).split("/", 1)[-1]
    gguf = Path(args.gguf) if args.gguf else find_gguf(
        repo, args.family, slug, variant=variant)

    cases = manifest.get("cases", ["jfk"])
    for case in cases:
        case_name = case_audio(case)
        language = case_language(case)
        audio = repo / "samples" / f"{case_name}.wav"
        if not audio.exists():
            raise SystemExit(f"error: audio not found: {audio}")

        out_dir = repo / "build" / "validate" / args.family / variant / case_name / "cpp"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        env = os.environ.copy()
        env["TRANSCRIBE_DUMP_DIR"] = str(out_dir)

        ref_transcript = (
            repo / "build" / "validate" / args.family / variant
            / case_name / "ref" / "transcript.json"
        )
        reference_data: dict[str, Any] = {}
        if ref_transcript.exists():
            reference_data = json.loads(ref_transcript.read_text())
        else:
            print(
                "  note: reference transcript artifact is absent; the CLI dump "
                "is running without reference-informed timestamp mode, and the "
                "later compare will report a word-count mismatch.",
                file=sys.stderr,
            )
        reference_words = reference_data.get("words")
        reference_tokens = reference_data.get("tokens")
        has_reference_words = isinstance(reference_words, list)
        has_reference_tokens = (
            isinstance(reference_tokens, list)
            and all(
                isinstance(row, dict)
                and {"start_s", "end_s", "text"}.issubset(row)
                for row in reference_tokens
            )
        )
        timestamp_request = None
        if has_reference_words:
            timestamp_request = "token" if has_reference_tokens else "word"

        # Whisper: by default, exercise the production C++ MelFrontend so
        # the per-tensor compare covers the full mel→encoder→decoder
        # pipeline. The env-var ref-mel injection is preserved as an
        # opt-in debug knob (--mel-from-ref): when a regression fires,
        # re-running with --mel-from-ref isolates whether the drift
        # originates in the C++ mel or downstream in the graph. Defaulting
        # to ref-mel hid a base.en regression once (the mel-precision
        # change in 4613129); we don't want that blind spot back.
        if args.family == "whisper" and getattr(args, "mel_from_ref", False):
            if not validation_hooks_enabled(repo):
                raise SystemExit(
                    "error: --mel-from-ref requires validation hooks compiled "
                    "in, but build/ has TRANSCRIBE_ENABLE_VALIDATION_HOOKS=OFF "
                    "(the hook is compiled out and would be silently ignored).\n"
                    "  Reconfigure: cmake --preset validation && "
                    "cmake --build build --target transcribe-cli"
                )
            ref_dir = repo / "build" / "validate" / args.family / variant / case_name / "ref"
            env["TRANSCRIBE_MEL_FROM_REF"] = str(ref_dir)

        cmd = [
            str(cli),
            "--backend", args.backend,
            # Default 1 thread for the documented reproducible gate; ggml CPU
            # parallelizes over independent output rows (full per-element
            # reductions), so VALIDATE_CPP_THREADS>1 only speeds the dump up —
            # the compared tensors are bit-identical.
            "--threads", os.environ.get("VALIDATE_CPP_THREADS", "1"),
            "-m", str(gguf),
        ]
        if language is not None:
            cmd += ["--language", language]
        if args.family == "whisper":
            cmd += ["--timestamps", "none"]
            timestamp_request = None
        elif timestamp_request is not None:
            cmd += ["--timestamps", timestamp_request]
        if args.family in ("sensevoice", "parakeet"):
            # The reference dumper emits the raw token stream including
            # control / language tags (sensevoice: language / event /
            # emotion / itn; multilingual parakeet: the <xx-XX> language
            # tag from auto language detection). The production C++ path
            # strips these by default, so the validate dump must pass
            # --raw-tokens to keep them and match the reference exactly.
            cmd += ["--raw-tokens"]

        batch_tmp = None
        if timestamp_request is not None:
            batch_tmp = tempfile.TemporaryDirectory(prefix="transcribe-validate-")
            batch_list = Path(batch_tmp.name) / "audio.txt"
            batch_list.write_text(f"{audio}\n")
            cmd += ["--batch", str(batch_list), "--batch-jsonl"]
        else:
            cmd.append(str(audio))

        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"  cpp dump [{args.family}/{case_name}]", file=sys.stderr)
        print(f"  TRANSCRIBE_DUMP_DIR={out_dir}", file=sys.stderr)
        print(f"  {' '.join(cmd)}", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

        try:
            result = subprocess.run(
                cmd,
                cwd=repo,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
        finally:
            if batch_tmp is not None:
                batch_tmp.cleanup()
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            raise SystemExit(
                f"error: cpp dump [{args.family}/{case_name}] failed "
                f"with exit code {result.returncode}"
            )
        cli_result = parse_cli_result(result.stdout or "")
        if cli_result is None:
            raise SystemExit(
                f"error: cpp dump [{args.family}/{case_name}] did not emit a transcript line"
            )
        words = cli_result.get("words", [])
        tokens = cli_result.get("tokens")
        if not isinstance(words, list):
            raise SystemExit(
                f"error: cpp dump [{args.family}/{case_name}] emitted a non-list words field"
            )
        if tokens is not None and not isinstance(tokens, list):
            raise SystemExit(
                f"error: cpp dump [{args.family}/{case_name}] emitted a non-list tokens field"
            )
        write_cpp_transcript(
            out_dir,
            family=args.family,
            variant=variant,
            case=case_name,
            gguf=gguf,
            backend=args.backend,
            text=str(cli_result["text"]),
            words=words,
            tokens=tokens,
        )

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    repo = find_repo_root(Path(__file__).parent)
    manifest = load_manifest(repo, args.family, getattr(args, "variant", None))
    variant = manifest["variant"]

    # Prefer per-variant tolerances declared in the manifest; fall
    # back to the family default. This lets larger variants (e.g.
    # Qwen3-ASR-1.7B vs 0.6B) carry scaled limits without stomping on
    # the family file.
    manifest_tol = manifest.get("tolerance_file")
    if manifest_tol:
        tolerances = repo / manifest_tol
    else:
        tolerances = repo / "tests" / "tolerances" / f"{args.family}.json"
    if not tolerances.exists():
        print(
            f"warning: no tolerance file at {tolerances}, using defaults",
            file=sys.stderr,
        )
        tolerances = None

    compare_script = repo / "scripts" / "compare_tensors.py"
    report_mode = getattr(args, "report", False)
    timestamp_tolerance = timestamp_tolerance_ms(tolerances)

    cases = manifest.get("cases", ["jfk"])
    all_passed = True
    compare_outputs: list[dict[str, Any]] = []
    transcript_results: list[dict[str, Any]] = []
    timestamp_results: list[dict[str, Any]] = []
    cmd_log: list[dict[str, Any]] = []

    for case in cases:
        case_name = case_audio(case)
        cpp_dir = repo / "build" / "validate" / args.family / variant / case_name / "cpp"
        ref_dir = repo / "build" / "validate" / args.family / variant / case_name / "ref"

        if not cpp_dir.exists():
            print(f"SKIP {case_name}: no C++ dumps at {cpp_dir}", file=sys.stderr)
            all_passed = False
            continue
        if not ref_dir.exists():
            print(f"SKIP {case_name}: no reference dumps at {ref_dir}", file=sys.stderr)
            all_passed = False
            continue

        cmd = [
            "uv", "run", str(compare_script),
            str(cpp_dir), str(ref_dir),
        ]
        if tolerances:
            cmd += ["--tolerances", str(tolerances)]

        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"  compare [{args.family}/{case_name}]", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

        if report_mode:
            result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            compare_outputs.append({
                "case": case_name,
                "returncode": result.returncode,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            })
        else:
            result = subprocess.run(cmd, cwd=repo)
        cmd_log.append({"case": case_name, "cmd": cmd, "returncode": result.returncode})
        if result.returncode != 0:
            all_passed = False

        # Transcript comparison: if the reference produced a transcript.json,
        # verify the C++ transcript. Manifests can opt into normalized compare
        # for models whose generation differs only in punctuation/casing.
        ref_transcript = ref_dir / "transcript.json"
        if ref_transcript.exists():
            transcript_compare = case_transcript_compare(manifest, case)
            ref_data = json.loads(ref_transcript.read_text())
            ref_text = str(ref_data.get("text", ""))
            cpp_transcript = cpp_dir / "transcript.json"
            if not cpp_transcript.exists():
                print(
                    f"FAIL transcript: missing C++ transcript artifact: {cpp_transcript}",
                    file=sys.stderr,
                )
                all_passed = False
                transcript_results.append({
                    "case": case_name, "match": False,
                    "reason": "missing C++ transcript artifact",
                })
                if "words" in ref_data:
                    timestamp_results.append({
                        "case": case_name,
                        "match": False,
                        "reason": "missing C++ transcript artifact",
                    })
                continue

            cpp_data = json.loads(cpp_transcript.read_text())
            cpp_text = str(cpp_data.get("text", ""))
            ref_compare = text_for_compare(ref_text, transcript_compare)
            cpp_compare = text_for_compare(cpp_text, transcript_compare)
            match = cpp_compare == ref_compare
            transcript_results.append({
                "case": case_name, "match": match,
                "reference": ref_text, "cpp": cpp_text,
                "mode": transcript_compare,
            })
            if not match:
                print(f"\nFAIL transcript mismatch ({transcript_compare})")
                print(f"  reference: {ref_text!r}")
                print(f"  c++:       {cpp_text!r}")
                if transcript_compare != "exact":
                    print(f"  reference normalized: {ref_compare!r}")
                    print(f"  c++ normalized:       {cpp_compare!r}")
                all_passed = False
            else:
                print(f"\n  Transcript: ok ({transcript_compare}) {cpp_text!r}")

            if "words" in ref_data:
                timestamp_result = case_timestamp_compare(
                    ref_data,
                    cpp_data,
                    text_mode=transcript_compare,
                    tolerance_ms=timestamp_tolerance,
                )
                timestamp_result = {
                    "case": case_name,
                    "mode": transcript_compare,
                    **timestamp_result,
                }
                timestamp_results.append(timestamp_result)
                if timestamp_result["match"]:
                    print(
                        "  Word timestamps: ok "
                        f"({timestamp_result['n_words']} words, "
                        f"max={timestamp_result['max_deviation_ms']:.3f} ms, "
                        f"mean={timestamp_result['mean_deviation_ms']:.3f} ms, "
                        f"tolerance={timestamp_result['tolerance_ms']:.3f} ms)"
                    )
                else:
                    print("\nFAIL word timestamp mismatch")
                    print(f"  {timestamp_result['reason']}")
                    if "max_deviation_ms" in timestamp_result:
                        print(
                            f"  max={timestamp_result['max_deviation_ms']:.3f} ms, "
                            f"mean={timestamp_result['mean_deviation_ms']:.3f} ms"
                        )
                    all_passed = False

    if report_mode:
        write_report_bundle(
            repo=repo,
            family=args.family,
            variant=variant,
            compare_outputs=compare_outputs,
            transcript_results=transcript_results,
            timestamp_results=timestamp_results,
            cmd_log=cmd_log,
            overall_passed=all_passed,
        )

    return 0 if all_passed else 1


def cmd_mel(args: argparse.Namespace) -> int:
    repo = find_repo_root(Path(__file__).parent)
    manifest = load_manifest(repo, args.family, getattr(args, "variant", None))
    variant = manifest["variant"]
    if args.family != "whisper":
        print("mel parity is currently only defined for whisper", file=sys.stderr)
        return 0

    cli = find_cli(repo)
    slug = manifest_source_model(manifest).split("/", 1)[-1]
    gguf = Path(args.gguf) if getattr(args, "gguf", None) else find_gguf(
        repo, args.family, slug, variant=variant)
    compare_script = repo / "scripts" / "compare_tensors.py"
    cases = manifest.get("cases", ["jfk"])
    all_passed = True

    for case in cases:
        case_name = case_audio(case)
        language = case_language(case)
        audio = repo / "samples" / f"{case_name}.wav"
        if not audio.exists():
            raise SystemExit(f"error: audio not found: {audio}")

        out_dir = repo / "build" / "validate" / args.family / variant / case_name / "mel_cpp"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        env = os.environ.copy()
        env["TRANSCRIBE_DUMP_DIR"] = str(out_dir)
        env.pop("TRANSCRIBE_MEL_FROM_REF", None)

        cmd = [
            str(cli),
            "--backend", getattr(args, "backend", "cpu"),
            "--threads", "1",
            "-m", str(gguf),
            "--timestamps", "none",
            str(audio),
        ]
        if language is not None:
            cmd[-1:-1] = ["--language", language]

        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"  mel parity cpp dump [{args.family}/{case_name}]", file=sys.stderr)
        print(f"  TRANSCRIBE_DUMP_DIR={out_dir}", file=sys.stderr)
        print(f"  {' '.join(cmd)}", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

        result = subprocess.run(
            cmd, cwd=repo, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            all_passed = False
            continue

        ref_dir = repo / "build" / "validate" / args.family / variant / case_name / "ref"
        with tempfile.TemporaryDirectory() as td:
            tol = Path(td) / "mel-tolerances.json"
            # fp32 STFT path: worst observed drift on librispeech-style
            # speech is 3.4e-4 max_abs (german) at peak signal ~1.4 →
            # ~2.5e-4 relative. 5e-4 leaves headroom for new clips
            # without crossing into territory that would shift WER.
            tol.write_text(json.dumps({
                "enc.mel.in": {"max_abs": 5e-4, "mean_abs": 5e-6},
            }) + "\n")
            cmp_cmd = [
                "uv", "run", str(compare_script),
                str(out_dir), str(ref_dir),
                "--max-abs", "1e9",
                "--mean-abs", "1e9",
                "--tolerances", str(tol),
                "--quiet",
            ]
            print(f"\n{'=' * 60}", file=sys.stderr)
            print(f"  mel parity compare [{args.family}/{case_name}]", file=sys.stderr)
            print(f"{'=' * 60}", file=sys.stderr)
            cmp = subprocess.run(cmp_cmd, cwd=repo)
            if cmp.returncode != 0:
                all_passed = False

    return 0 if all_passed else 1


def git_head_sha(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_report_bundle(
    *,
    repo: Path,
    family: str,
    variant: str,
    compare_outputs: list[dict[str, Any]],
    transcript_results: list[dict[str, Any]],
    timestamp_results: list[dict[str, Any]],
    cmd_log: list[dict[str, Any]],
    overall_passed: bool,
) -> Path:
    """Write an ephemeral validation report bundle.

    The bundle captures what's not reproducible from git alone: the compare
    stdout, exact command invocations, transcript comparison, and timestamp
    comparison at this moment. The manifest, tolerance file, and
    compare_tensors.py live in the repo — pin them via `validated_at_sha` (the
    git HEAD at bundle time) rather than duplicating them here.
    """
    now = dt.datetime.now(dt.UTC)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    sha = git_head_sha(repo)
    bundle_id = f"{ts}-{sha[:7]}" if sha != "unknown" else ts

    bundle_dir = repo / "reports" / "porting" / family / variant / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "commands.json").write_text(
        json.dumps({"validated_at_sha": sha, "commands": cmd_log}, indent=2) + "\n"
    )

    summary = [
        f"# Validation Report — {family}/{variant}",
        "",
        f"- Generated: {now.isoformat().replace('+00:00', 'Z')}",
        f"- Repo SHA: `{sha}`",
        f"- Overall: **{'PASS' if overall_passed else 'FAIL'}**",
        "",
    ]
    for co in compare_outputs:
        summary += [
            f"## compare_tensors — {co['case']}",
            "",
            f"Exit code: `{co['returncode']}`",
            "",
            "```",
            co["stdout"].rstrip() or "(no stdout)",
            "```",
            "",
        ]
    if transcript_results:
        summary += ["## Transcript comparison", ""]
        for tr in transcript_results:
            summary.append(f"### {tr['case']}")
            summary.append("")
            if tr["match"]:
                summary.append(f"- Match: **yes**")
                summary.append(f"- Mode: `{tr.get('mode', 'exact')}`")
                summary.append(f"- text: `{tr.get('cpp', '')!r}`")
            else:
                summary.append(f"- Match: **no**")
                summary.append(f"- Mode: `{tr.get('mode', 'exact')}`")
                if "reason" in tr:
                    summary.append(f"- Reason: {tr['reason']}")
                else:
                    summary.append(f"- reference: `{tr.get('reference', '')!r}`")
                    summary.append(f"- c++:       `{tr.get('cpp', '')!r}`")
            summary.append("")
    if timestamp_results:
        summary += ["## Word timestamp comparison", ""]
        for result in timestamp_results:
            summary.append(f"### {result['case']}")
            summary.append("")
            summary.append(
                f"- Match: **{'yes' if result['match'] else 'no'}**"
            )
            if result["match"]:
                summary.append(f"- Words: `{result['n_words']}`")
                summary.append(
                    f"- Max endpoint deviation: `{result['max_deviation_ms']:.3f} ms`"
                )
                summary.append(
                    f"- Mean endpoint deviation: `{result['mean_deviation_ms']:.3f} ms`"
                )
                summary.append(
                    f"- Tolerance: `{result['tolerance_ms']:.3f} ms`"
                )
            else:
                summary.append(f"- Reason: {result['reason']}")
            summary.append("")
    (bundle_dir / "summary.md").write_text("\n".join(summary))

    repro = f"""# Reproducing this validation run

Check out repo at SHA `{sha}`, then run:

```bash
uv run scripts/validate.py all --family {family} --variant {variant}
```

Or step by step:

```bash
uv run scripts/validate.py ref     --family {family} --variant {variant}
uv run scripts/validate.py cpp     --family {family} --variant {variant}
uv run scripts/validate.py compare --family {family} --variant {variant}
```

Inputs (all version-controlled; pinned by `validated_at_sha` in commands.json):

- Golden manifest: `tests/golden/{family}/{variant}.manifest.json`
- Tolerance file: `tests/tolerances/{family}.json`
- Comparator: `scripts/compare_tensors.py`

Bundle contents are ephemeral evidence — the compare stdout and transcript
check at validation time. The validated commit SHA is authoritative;
snapshots of repo-tracked files are intentionally omitted.
"""
    (bundle_dir / "reproduce.md").write_text(repro)

    print(f"\nReport bundle: {bundle_dir.relative_to(repo)}", file=sys.stderr)
    return bundle_dir


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_ref(args)
    if rc != 0:
        return rc
    if args.family == "whisper":
        rc = cmd_mel(args)
        if rc != 0:
            return rc
    rc = cmd_cpp(args)
    if rc != 0:
        return rc
    return cmd_compare(args)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convention-driven numerical validation for transcribe.cpp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared arguments.
    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--family", required=True,
            help="Model family (e.g. parakeet, cohere)",
        )
        sp.add_argument(
            "--variant",
            help="Model variant (e.g. cohere-transcribe-03-2026). "
                 "Required when the family has multiple manifests.",
        )

    # ref
    sp_ref = sub.add_parser("ref", help="Generate reference dumps")
    add_common(sp_ref)
    sp_ref.add_argument(
        "--model",
        help="HF model ID or local path (overrides manifest default)",
    )
    sp_ref.set_defaults(func=cmd_ref)

    # cpp
    sp_cpp = sub.add_parser("cpp", help="Generate C++ dumps")
    add_common(sp_cpp)
    sp_cpp.add_argument("--gguf", help="GGUF path (overrides auto-detection)")
    sp_cpp.add_argument(
        "--backend", default="cpu",
        choices=["auto", "cpu", "cpu_accel", "metal", "vulkan"],
        help="Compute backend (default: cpu)",
    )
    sp_cpp.add_argument(
        "--mel-from-ref", action="store_true",
        help="Whisper-only validation hook: inject the reference mel via "
             "TRANSCRIBE_MEL_FROM_REF so enc.mel.in is bit-"
             "identical to HF's WhisperFeatureExtractor. Use to isolate "
             "graph drift from frontend drift when a regression fires. "
             "Requires a build with -DTRANSCRIBE_ENABLE_VALIDATION_HOOKS=ON. "
             "Default is the production C++ MelFrontend.",
    )
    sp_cpp.set_defaults(func=cmd_cpp)

    # mel
    sp_mel = sub.add_parser("mel", help="Compare production C++ mel vs reference mel")
    add_common(sp_mel)
    sp_mel.add_argument("--gguf", help="GGUF path (overrides auto-detection)")
    sp_mel.add_argument(
        "--backend", default="cpu",
        choices=["auto", "cpu", "cpu_accel", "metal", "vulkan"],
        help="Compute backend (default: cpu)",
    )
    sp_mel.set_defaults(func=cmd_mel)

    # compare
    sp_cmp = sub.add_parser("compare", help="Compare C++ vs reference dumps")
    add_common(sp_cmp)
    sp_cmp.add_argument(
        "--report", action="store_true",
        help="Emit a report bundle under reports/porting/<family>/<variant>/<ts>-<sha>/",
    )
    sp_cmp.set_defaults(func=cmd_compare)

    # all
    sp_all = sub.add_parser("all", help="Run ref + cpp + compare")
    add_common(sp_all)
    sp_all.add_argument("--model", help="HF model ID or local path")
    sp_all.add_argument("--gguf", help="GGUF path")
    sp_all.add_argument("--backend", default="cpu", choices=["auto", "cpu", "cpu_accel", "metal", "vulkan"])
    sp_all.add_argument(
        "--report", action="store_true",
        help="Emit a report bundle after compare completes.",
    )
    sp_all.add_argument(
        "--mel-from-ref", action="store_true",
        help="Whisper-only validation hook: inject the reference mel via "
             "TRANSCRIBE_MEL_FROM_REF for the cpp dump. Requires a build with "
             "-DTRANSCRIBE_ENABLE_VALIDATION_HOOKS=ON. Default is the "
             "production C++ MelFrontend.",
    )
    sp_all.set_defaults(func=cmd_all)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
