#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
golden.py - manifest-driven golden/reference artifact helper.

This script intentionally starts small. It validates committed golden
manifests, prints the commands needed to generate reference payloads, and
can run those commands to populate build/goldens or build/validate output
directories. Generated payload indexes live next to the payloads; the
committed manifest remains the human-readable contract.

Usage:
    uv run scripts/golden.py verify --manifest tests/golden/cohere/cohere-transcribe-03-2026.manifest.json
    uv run scripts/golden.py plan   --manifest tests/golden/cohere/cohere-transcribe-03-2026.manifest.json
    uv run scripts/golden.py generate --manifest tests/golden/cohere/cohere-transcribe-03-2026.manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "transcribe-golden-manifest-v1"
INDEX_SCHEMA = "transcribe-golden-index-v1"


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("cannot locate transcribe.cpp repo root")


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text())
    except OSError as e:
        raise SystemExit(f"error: cannot read manifest {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: invalid JSON in {path}: {e}") from e
    if obj.get("schema") != SCHEMA:
        raise SystemExit(
            f"error: {path}: schema must be {SCHEMA!r}, got {obj.get('schema')!r}"
        )
    return obj


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(raw: str, repo: Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = repo / p
    return p.resolve()


def resolved_paths(manifest: dict[str, Any], repo: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name, spec in (manifest.get("paths") or {}).items():
        if not isinstance(spec, dict):
            raise SystemExit(f"error: paths.{name} must be an object")
        env = spec.get("env")
        default = spec.get("default")
        val = os.environ.get(env, "") if env else ""
        if not val:
            val = default
        if not val:
            raise SystemExit(
                f"error: path {name!r} has no value; set {env} or add default"
            )
        out[name] = resolve_path(str(val), repo)
    return out


def expand_template(
    value: str,
    *,
    repo: Path,
    paths: dict[str, Path],
    audio: Path | None = None,
    out: Path | None = None,
) -> str:
    s = value.replace("{repo}", str(repo))
    if audio is not None:
        s = s.replace("{audio}", str(audio))
    if out is not None:
        s = s.replace("{out}", str(out))
    for key, path in paths.items():
        s = s.replace("{path:" + key + "}", str(path))
    return s


def check_inputs(manifest: dict[str, Any], repo: Path, paths: dict[str, Path]) -> int:
    failures = 0
    for name, path in paths.items():
        if not path.exists():
            print(f"FAIL path {name}: not found: {path}", file=sys.stderr)
            failures += 1

    for check in manifest.get("checks") or []:
        raw = check.get("path")
        if not raw:
            print("FAIL check without path", file=sys.stderr)
            failures += 1
            continue
        path = resolve_path(expand_template(str(raw), repo=repo, paths=paths), repo)
        if not path.exists():
            print(f"FAIL check path not found: {path}", file=sys.stderr)
            failures += 1
            continue
        expected = check.get("sha256")
        if expected:
            actual = sha256_file(path)
            if actual != expected:
                print(
                    f"FAIL sha256 {path}: got {actual}, expected {expected}",
                    file=sys.stderr,
                )
                failures += 1
    return failures


def stage_commands(
    manifest: dict[str, Any],
    repo: Path,
    paths: dict[str, Path],
) -> list[tuple[str, str, Path, list[str]]]:
    cmds: list[tuple[str, str, Path, list[str]]] = []
    family = str(manifest["family"])
    variant = str(manifest["variant"])
    for case in manifest.get("cases") or []:
        case_name = str(case["name"])
        audio = resolve_path(str(case["audio"]), repo)
        for stage in case.get("stages") or []:
            stage_name = str(stage["name"])
            out_raw = stage.get(
                "out",
                f"build/validate/{family}/{variant}/{case_name}/{stage_name}/ref",
            )
            out = resolve_path(str(out_raw), repo)
            raw_cmd = stage.get("command")
            if not isinstance(raw_cmd, list) or not raw_cmd:
                raise SystemExit(
                    f"error: case {case_name} stage {stage_name} missing command list"
                )
            cmd = [
                expand_template(
                    str(part), repo=repo, paths=paths, audio=audio, out=out
                )
                for part in raw_cmd
            ]
            cmds.append((case_name, stage_name, out, cmd))
    return cmds


def write_index(out_dir: Path, manifest: dict[str, Any], case: str, stage: str) -> None:
    outputs: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for f32 in sorted(out_dir.glob("*.f32")):
        meta_path = f32.with_suffix(".json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                meta = {}
        outputs[f32.name] = {
            "sha256": sha256_file(f32),
            "bytes": f32.stat().st_size,
            "dtype": meta.get("dtype", "f32"),
            "shape": meta.get("shape"),
            "sidecar": meta_path.name if meta_path.exists() else None,
        }
    for path in sorted(out_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == "golden-index.json" or path.suffix == ".f32":
            continue
        if path.suffix == ".json" and path.with_suffix(".f32").exists():
            continue
        artifacts[path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    index = {
        "schema": INDEX_SCHEMA,
        "family": manifest.get("family"),
        "variant": manifest.get("variant"),
        "case": case,
        "stage": stage,
        "outputs": outputs,
        "artifacts": artifacts,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "golden-index.json").write_text(json.dumps(index, indent=2) + "\n")


def cmd_verify(args: argparse.Namespace) -> int:
    repo = find_repo_root(Path(__file__).parent)
    manifest_path = resolve_path(str(args.manifest), repo)
    manifest = load_manifest(manifest_path)
    paths = resolved_paths(manifest, repo)
    failures = check_inputs(manifest, repo, paths)
    if failures:
        return 1
    print(f"ok: {manifest_path}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    repo = find_repo_root(Path(__file__).parent)
    manifest_path = resolve_path(str(args.manifest), repo)
    manifest = load_manifest(manifest_path)
    paths = resolved_paths(manifest, repo)
    failures = check_inputs(manifest, repo, paths)
    if failures:
        return 1
    for case, stage, out, cmd in stage_commands(manifest, repo, paths):
        print(f"[{case}/{stage}]")
        print(f"  out: {out}")
        print("  " + " ".join(cmd))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    repo = find_repo_root(Path(__file__).parent)
    manifest_path = resolve_path(str(args.manifest), repo)
    manifest = load_manifest(manifest_path)
    paths = resolved_paths(manifest, repo)
    failures = check_inputs(manifest, repo, paths)
    if failures:
        return 1
    for case, stage, out, cmd in stage_commands(manifest, repo, paths):
        print(f"[{case}/{stage}] {' '.join(cmd)}", file=sys.stderr)
        if args.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if len(cmd) >= 2 and cmd[0] == "uv" and cmd[1] == "run":
            env.pop("VIRTUAL_ENV", None)
        res = subprocess.run(cmd, cwd=repo, env=env)
        if res.returncode != 0:
            print(
                f"error: {case}/{stage} generation failed with exit {res.returncode}",
                file=sys.stderr,
            )
            return res.returncode
        write_index(out, manifest, case, stage)
        print(f"wrote {out / 'golden-index.json'}", file=sys.stderr)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, func in (
        ("verify", cmd_verify),
        ("plan", cmd_plan),
        ("generate", cmd_generate),
    ):
        sp = sub.add_parser(name)
        sp.add_argument("--manifest", required=True, type=Path)
        if name == "generate":
            sp.add_argument("--dry-run", action="store_true")
        sp.set_defaults(func=func)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
