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
import os
import shutil
import subprocess
import sys
import tempfile
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


def find_gguf(repo: Path, family: str, slug: str | None = None) -> Path:
    """Find a GGUF under models/.

    Discovery order:
      1. If `slug` is provided (derived from the manifest's
         source_model.hf_repo, e.g. "Qwen3-ASR-0.6B" from
         "Qwen/Qwen3-ASR-0.6B"), look for
         models/<slug>/<slug>-<QUANT>.gguf directly. This is the
         converter's output convention and is case-accurate — which
         matters for families whose HF slug does not case-fold to the
         family key (e.g. family="qwen3_asr", slug="Qwen3-ASR-0.6B").
      2. Legacy fallback: scan models/*/ for any GGUF whose stem
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

    # 1. Manifest-slug-driven lookup.
    if slug:
        variant_dir = model_root / slug
        if variant_dir.is_dir():
            for quant in preferred_quants:
                candidate = variant_dir / f"{slug}-{quant}.gguf"
                if candidate.exists():
                    return candidate
            matches = sorted(variant_dir.glob(f"{slug}-*.gguf"))
            if matches:
                return matches[0]

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
    reference = manifest_reference(manifest)
    model = args.model or manifest_source_model(manifest)
    if not model:
        raise SystemExit("error: no model specified and none in manifest")

    dump_script = manifest_dump_script(repo, manifest)
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

        # Pin the HF revision when the manifest declares one and the
        # family's dumper accepts --revision. Currently only the
        # qwen3_asr dumper has been wired; Parakeet and Cohere will
        # follow in a separate commit. The dumper itself ignores
        # --revision when --model resolves to a local directory.
        hf_revision = (manifest.get("source_model") or {}).get("hf_revision")
        if hf_revision and args.family == "qwen3_asr":
            common_args += ["--revision", str(hf_revision)]

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
                f"ref {stage} [{args.family}/{variant}/{case}/{reference['kind']}]",
            )

    return 0


def cmd_cpp(args: argparse.Namespace) -> int:
    repo = find_repo_root(Path(__file__).parent)
    manifest = load_manifest(repo, args.family, getattr(args, "variant", None))
    variant = manifest["variant"]
    cli = find_cli(repo)
    slug = manifest_source_model(manifest).split("/", 1)[-1]
    gguf = Path(args.gguf) if args.gguf else find_gguf(repo, args.family, slug)

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

        # Whisper: inject the reference mel via TRANSCRIBE_WHISPER_MEL_FROM_REF
        # so enc.mel.in is bit-identical to HF's WhisperFeatureExtractor
        # output. This isolates encoder/decoder numerical drift from mel-
        # frontend drift so each can be tolerance-managed separately.
        # The C++ mel frontend is validated by `validate.py mel` against
        # its own tolerance (see mel_parity below).
        if args.family == "whisper":
            ref_dir = repo / "build" / "validate" / args.family / variant / case / "ref"
            env["TRANSCRIBE_WHISPER_MEL_FROM_REF"] = str(ref_dir)

        cmd = [
            str(cli),
            "--backend", args.backend,
            "--threads", "1",
            "-m", str(gguf),
        ]
        if args.family == "whisper":
            cmd += ["--timestamps", "none", "--language", "en"]
        cmd.append(str(audio))

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

    cases = manifest.get("cases", ["jfk"])
    all_passed = True
    compare_outputs: list[dict[str, Any]] = []
    transcript_results: list[dict[str, Any]] = []
    cmd_log: list[dict[str, Any]] = []

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

        if report_mode:
            result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            compare_outputs.append({
                "case": case,
                "returncode": result.returncode,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            })
        else:
            result = subprocess.run(cmd, cwd=repo)
        cmd_log.append({"case": case, "cmd": cmd, "returncode": result.returncode})
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
                transcript_results.append({
                    "case": case, "match": False,
                    "reason": "missing C++ transcript artifact",
                })
                continue

            cpp_data = json.loads(cpp_transcript.read_text())
            cpp_text = str(cpp_data.get("text", ""))
            match = cpp_text == ref_text
            transcript_results.append({
                "case": case, "match": match,
                "reference": ref_text, "cpp": cpp_text,
            })
            if not match:
                print("\nFAIL transcript mismatch")
                print(f"  reference: {ref_text!r}")
                print(f"  c++:       {cpp_text!r}")
                all_passed = False
            else:
                print(f"\n  Transcript: ok {cpp_text!r}")

    if report_mode:
        write_report_bundle(
            repo=repo,
            family=args.family,
            variant=variant,
            compare_outputs=compare_outputs,
            transcript_results=transcript_results,
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
    gguf = Path(args.gguf) if getattr(args, "gguf", None) else find_gguf(repo, args.family, slug)
    compare_script = repo / "scripts" / "compare_tensors.py"
    cases = manifest.get("cases", ["jfk"])
    all_passed = True

    for case in cases:
        audio = repo / "samples" / f"{case}.wav"
        if not audio.exists():
            raise SystemExit(f"error: audio not found: {audio}")

        out_dir = repo / "build" / "validate" / args.family / variant / case / "mel_cpp"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        env = os.environ.copy()
        env["TRANSCRIBE_DUMP_DIR"] = str(out_dir)
        env.pop("TRANSCRIBE_WHISPER_MEL_FROM_REF", None)

        cmd = [
            str(cli),
            "--backend", getattr(args, "backend", "cpu"),
            "--threads", "1",
            "-m", str(gguf),
            "--timestamps", "none",
            "--language", "en",
            str(audio),
        ]

        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"  mel parity cpp dump [{args.family}/{case}]", file=sys.stderr)
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

        ref_dir = repo / "build" / "validate" / args.family / variant / case / "ref"
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
            print(f"  mel parity compare [{args.family}/{case}]", file=sys.stderr)
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
    cmd_log: list[dict[str, Any]],
    overall_passed: bool,
) -> Path:
    """Write an ephemeral validation report bundle.

    The bundle captures what's not reproducible from git alone: the compare
    stdout, exact command invocations, and transcript comparison at this
    moment. The manifest, tolerance file, and compare_tensors.py live in the
    repo — pin them via `validated_at_sha` (the git HEAD at bundle time)
    rather than duplicating them here.
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
                summary.append(f"- text: `{tr.get('cpp', '')!r}`")
            else:
                summary.append(f"- Match: **no**")
                if "reason" in tr:
                    summary.append(f"- Reason: {tr['reason']}")
                else:
                    summary.append(f"- reference: `{tr.get('reference', '')!r}`")
                    summary.append(f"- c++:       `{tr.get('cpp', '')!r}`")
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
    sp_all.set_defaults(func=cmd_all)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
