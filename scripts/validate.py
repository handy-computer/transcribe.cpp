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
    uv run scripts/validate.py cpp --family cohere --gguf models/cohere/cohere.bf16.gguf

Conventions:
    Manifest:    tests/golden/{family}/*.manifest.json
    Dump script: scripts/dump_reference_{family}_{reference}.py
    Python env:  scripts/envs/{family}/
    Tolerances:  tests/tolerances/{family}.json
    Audio:       samples/jfk.wav
    GGUF:        models/{family}/ (first *.gguf found, or *.bf16.gguf, or *.f32.gguf)
    Ref output:  build/validate/{family}/{variant}/{case}/ref/
    C++ output:  build/validate/{family}/{variant}/{case}/cpp/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


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


def find_dump_script(repo: Path, family: str, reference: str) -> Path:
    script = repo / "scripts" / f"dump_reference_{family}_{reference}.py"
    if script.exists():
        return script
    # Try without reference suffix as fallback.
    fallback = repo / "scripts" / f"dump_reference_{family}.py"
    if fallback.exists():
        return fallback
    raise SystemExit(
        f"error: dump script not found: {script}\n"
        f"  (also tried {fallback})"
    )


def find_gguf(repo: Path, family: str) -> Path:
    model_dir = repo / "models" / family
    if not model_dir.is_dir():
        raise SystemExit(
            f"error: model directory not found: {model_dir}\n"
            f"  Convert a GGUF first or set --gguf"
        )
    # Prefer bf16, then f32, then f16, then first match.
    for suffix in [".bf16.gguf", ".f32.gguf", ".f16.gguf"]:
        matches = sorted(model_dir.glob(f"*{suffix}"))
        if matches:
            return matches[0]
    matches = sorted(model_dir.glob("*.gguf"))
    if matches:
        return matches[0]
    raise SystemExit(f"error: no GGUF files found in {model_dir}")


def find_cli(repo: Path) -> Path:
    for candidate in [
        repo / "build" / "bin" / "transcribe-cli",
        repo / "build" / "transcribe-cli",
    ]:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "error: transcribe-cli not found in build/bin/\n"
        "  Run: cmake --build build --target transcribe-cli"
    )


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


def parse_cli_transcript(output: str) -> str | None:
    for line in reversed(output.splitlines()):
        if line.startswith("text: "):
            return line[len("text: "):]
    return None


def write_cpp_transcript(
    out_dir: Path,
    *,
    family: str,
    variant: str,
    case: str,
    gguf: Path,
    backend: str,
    text: str,
) -> None:
    path = out_dir / "transcript.json"
    payload = {
        "schema": "transcribe-cpp-transcript-v1",
        "family": family,
        "variant": variant,
        "case": case,
        "text": text,
        "normalized_text": normalize_text(text),
        "source": {
            "kind": "transcribe.cpp",
            "gguf": str(gguf),
            "backend": backend,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {path}", file=sys.stderr)


def cmd_ref(args: argparse.Namespace) -> int:
    repo = find_repo_root(Path(__file__).parent)
    manifest = load_manifest(repo, args.family, getattr(args, "variant", None))
    variant = manifest["variant"]
    reference = manifest.get("reference", args.family)
    model = args.model or manifest.get("model", "")
    if not model:
        raise SystemExit("error: no model specified and none in manifest")

    dump_script = find_dump_script(repo, args.family, reference)
    env_dir = repo / "scripts" / "envs" / args.family

    cases = manifest.get("cases", ["jfk"])
    for case in cases:
        audio = repo / "samples" / f"{case}.wav"
        if not audio.exists():
            raise SystemExit(f"error: audio not found: {audio}")

        out_dir = repo / "build" / "validate" / args.family / variant / case / "ref"
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

        # Run both encoder and decode subcommands. Some families dump
        # everything from decode (cohere); others split encoder and
        # decoder intermediates across subcommands (parakeet). Running
        # both is safe — if a tensor is dumped by both, the decode
        # pass overwrites the encoder pass (same values).
        for stage in ["encoder", "decode"]:
            cmd = base_args + [stage] + common_args
            run_cmd(cmd, repo, f"ref {stage} [{args.family}/{variant}/{case}]")

    return 0


def cmd_cpp(args: argparse.Namespace) -> int:
    repo = find_repo_root(Path(__file__).parent)
    manifest = load_manifest(repo, args.family, getattr(args, "variant", None))
    variant = manifest["variant"]
    cli = find_cli(repo)
    gguf = Path(args.gguf) if args.gguf else find_gguf(repo, args.family)

    cases = manifest.get("cases", ["jfk"])
    for case in cases:
        audio = repo / "samples" / f"{case}.wav"
        if not audio.exists():
            raise SystemExit(f"error: audio not found: {audio}")

        out_dir = repo / "build" / "validate" / args.family / variant / case / "cpp"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        env = os.environ.copy()
        env["TRANSCRIBE_DUMP_DIR"] = str(out_dir)

        cmd = [
            str(cli),
            "--backend", args.backend,
            "--threads", "1",
            "-m", str(gguf),
            str(audio),
        ]

        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"  cpp dump [{args.family}/{case}]", file=sys.stderr)
        print(f"  TRANSCRIBE_DUMP_DIR={out_dir}", file=sys.stderr)
        print(f"  {' '.join(cmd)}", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

        result = subprocess.run(
            cmd,
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            raise SystemExit(
                f"error: cpp dump [{args.family}/{case}] failed "
                f"with exit code {result.returncode}"
            )
        transcript = parse_cli_transcript(result.stdout or "")
        if transcript is None:
            raise SystemExit(
                f"error: cpp dump [{args.family}/{case}] did not emit a transcript line"
            )
        write_cpp_transcript(
            out_dir,
            family=args.family,
            variant=variant,
            case=case,
            gguf=gguf,
            backend=args.backend,
            text=transcript,
        )

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    repo = find_repo_root(Path(__file__).parent)
    manifest = load_manifest(repo, args.family, getattr(args, "variant", None))
    variant = manifest["variant"]

    tolerances = repo / "tests" / "tolerances" / f"{args.family}.json"
    if not tolerances.exists():
        print(
            f"warning: no tolerance file at {tolerances}, using defaults",
            file=sys.stderr,
        )
        tolerances = None

    compare_script = repo / "scripts" / "compare_tensors.py"

    cases = manifest.get("cases", ["jfk"])
    all_passed = True
    for case in cases:
        cpp_dir = repo / "build" / "validate" / args.family / variant / case / "cpp"
        ref_dir = repo / "build" / "validate" / args.family / variant / case / "ref"

        if not cpp_dir.exists():
            print(f"SKIP {case}: no C++ dumps at {cpp_dir}", file=sys.stderr)
            all_passed = False
            continue
        if not ref_dir.exists():
            print(f"SKIP {case}: no reference dumps at {ref_dir}", file=sys.stderr)
            all_passed = False
            continue

        cmd = [
            "uv", "run", str(compare_script),
            str(cpp_dir), str(ref_dir),
        ]
        if tolerances:
            cmd += ["--tolerances", str(tolerances)]

        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"  compare [{args.family}/{case}]", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

        result = subprocess.run(cmd, cwd=repo)
        if result.returncode != 0:
            all_passed = False

        # Transcript comparison: if the reference produced a
        # transcript.json, verify the C++ transcript matches exactly.
        ref_transcript = ref_dir / "transcript.json"
        if ref_transcript.exists():
            ref_data = json.loads(ref_transcript.read_text())
            ref_text = str(ref_data.get("text", ""))
            cpp_transcript = cpp_dir / "transcript.json"
            if not cpp_transcript.exists():
                print(
                    f"FAIL transcript: missing C++ transcript artifact: {cpp_transcript}",
                    file=sys.stderr,
                )
                all_passed = False
                continue

            cpp_data = json.loads(cpp_transcript.read_text())
            cpp_text = str(cpp_data.get("text", ""))
            if cpp_text != ref_text:
                print("\nFAIL transcript mismatch")
                print(f"  reference: {ref_text!r}")
                print(f"  c++:       {cpp_text!r}")
                all_passed = False
            else:
                print(f"\n  Transcript: ok {cpp_text!r}")

    return 0 if all_passed else 1


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_ref(args)
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
        choices=["auto", "cpu", "metal", "vulkan"],
        help="Compute backend (default: cpu)",
    )
    sp_cpp.set_defaults(func=cmd_cpp)

    # compare
    sp_cmp = sub.add_parser("compare", help="Compare C++ vs reference dumps")
    add_common(sp_cmp)
    sp_cmp.set_defaults(func=cmd_compare)

    # all
    sp_all = sub.add_parser("all", help="Run ref + cpp + compare")
    add_common(sp_all)
    sp_all.add_argument("--model", help="HF model ID or local path")
    sp_all.add_argument("--gguf", help="GGUF path")
    sp_all.add_argument("--backend", default="cpu", choices=["auto", "cpu", "metal", "vulkan"])
    sp_all.set_defaults(func=cmd_all)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
