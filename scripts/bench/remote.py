#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
remote.py — dispatch the bench-fleet workflow and merge results back.

Wraps the full round trip for .github/workflows/bench.yml:

    dispatch (gh workflow run) → locate the run → watch it → download the
    perf-* artifacts → merge reports into local reports/perf/

Usage:
    uv run scripts/bench/remote.py --models parakeet-tdt-0.6b-v3
    uv run scripts/bench/remote.py --models Qwen3-ASR-0.6B \
        --quants q8_0,q4_k_m --samples jfk,dots --iters 3 --warmup 1 \
        --name Qwen3-ASR-0.6B-publication
    uv run scripts/bench/remote.py --models <variant> --machines t14-vulkan
    uv run scripts/bench/remote.py --fetch 1234567890   # download-only

The fleet benches the PUSHED state of --ref (default: the current
branch). Unpushed local commits are a hard error — silently benching
stale code is the failure mode this script exists to prevent. A dirty
working tree is only a warning (local edits aren't benched either way).

Options mirror the bench.yml inputs:
    --models M[,M...]     variant slugs (required unless --fetch)
    --quants Q[,Q...]     default q8_0,q4_k_m
    --samples S[,S...]    default jfk,dots
    --iters N             default 3
    --warmup N            default 1
    --name LABEL          stable report label (default: timestamp names)
    --machines WHICH      all | m4-mini | t14-vulkan (default: all)
    --hf-repo REPO        override the per-variant handy-computer/<v>-gguf
    --ref BRANCH          branch to dispatch on (default: current branch)
    --no-watch            dispatch, print the run URL, and exit
    --fetch RUN_ID        skip dispatch; download + merge RUN_ID's artifacts

Requires the `gh` CLI, authenticated with write access to the repo. The
workflow file must exist on the default branch before the first dispatch
(GitHub only registers workflow_dispatch workflows from there); after
that, --ref can point at any pushed branch.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

WORKFLOW = "bench.yml"


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("cannot locate repo root")


def sh(args: list[str], cwd: Path, check: bool = True,
       capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, check=check,
                          capture_output=capture, text=True)


def git(repo: Path, *args: str) -> str | None:
    proc = sh(["git", *args], cwd=repo, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def preflight_ref(repo: Path, ref: str | None) -> str:
    """Resolve the ref to dispatch on and refuse obviously-stale state."""
    current = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if ref is None:
        if current is None or current == "HEAD":
            print("error: detached HEAD; pass --ref <branch>", file=sys.stderr)
            sys.exit(2)
        ref = current

    if ref == current:
        if git(repo, "rev-parse", "--abbrev-ref", "@{u}") is None:
            print(f"error: branch {ref!r} has no upstream — push it first "
                  f"(the fleet benches the remote state)", file=sys.stderr)
            sys.exit(2)
        ahead = git(repo, "rev-list", "--count", "@{u}..HEAD")
        if ahead and int(ahead) > 0:
            print(f"error: {ahead} local commit(s) not pushed on {ref!r} — "
                  f"the fleet would bench stale code. Push first.",
                  file=sys.stderr)
            sys.exit(2)
        dirty = git(repo, "status", "--porcelain")
        if dirty:
            print(f"warning: working tree is dirty; local edits are NOT "
                  f"benched (fleet runs the pushed {ref})", file=sys.stderr)
    else:
        remote = git(repo, "ls-remote", "--heads", "origin", ref)
        if not remote:
            print(f"error: branch {ref!r} not found on origin", file=sys.stderr)
            sys.exit(2)
    return ref


def dispatch(repo: Path, args: argparse.Namespace, ref: str) -> None:
    cmd = ["gh", "workflow", "run", WORKFLOW, "--ref", ref,
           "-f", f"models={args.models}",
           "-f", f"quants={args.quants}",
           "-f", f"samples={args.samples}",
           "-f", f"iters={args.iters}",
           "-f", f"warmup={args.warmup}",
           "-f", f"machines={args.machines}"]
    if args.name:
        cmd += ["-f", f"name={args.name}"]
    if args.hf_repo:
        cmd += ["-f", f"hf_repo={args.hf_repo}"]
    proc = sh(cmd, cwd=repo, check=False)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        if "could not find any workflows" in proc.stderr.lower():
            print(f"hint: {WORKFLOW} must be on the default branch before "
                  f"gh can dispatch it", file=sys.stderr)
        sys.exit(proc.returncode)


def locate_run(repo: Path, ref: str, not_before: datetime,
               timeout_s: float = 60.0) -> dict:
    """Find the run created by our dispatch (gh workflow run returns no id)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        proc = sh(["gh", "run", "list", "--workflow", WORKFLOW,
                   "--event", "workflow_dispatch", "--branch", ref,
                   "--limit", "5",
                   "--json", "databaseId,createdAt,url,status"],
                  cwd=repo, check=False)
        if proc.returncode == 0:
            runs = json.loads(proc.stdout or "[]")
            fresh = [r for r in runs
                     if datetime.fromisoformat(
                         r["createdAt"].replace("Z", "+00:00")) >= not_before]
            if fresh:
                return max(fresh, key=lambda r: r["createdAt"])
        time.sleep(3)
    print("error: dispatched run did not appear within "
          f"{timeout_s:.0f}s; check `gh run list --workflow {WORKFLOW}`",
          file=sys.stderr)
    sys.exit(1)


def watch(repo: Path, run_id: str) -> int:
    proc = subprocess.run(["gh", "run", "watch", run_id,
                           "--exit-status", "--interval", "15"], cwd=repo)
    return proc.returncode


def download_and_merge(repo: Path, run_id: str) -> list[Path]:
    """Merge every perf-* artifact into reports/perf/. Artifact contents are
    <machine-slug>/<report>.json, so different machines never collide."""
    proc = sh(["gh", "api",
               f"repos/{{owner}}/{{repo}}/actions/runs/{run_id}/artifacts",
               "--jq", ".artifacts[].name"], cwd=repo, check=False)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(1)
    names = [n for n in proc.stdout.split() if n.startswith("perf-")]
    if not names:
        print("no perf-* artifacts on this run (all machines failed?)",
              file=sys.stderr)
        return []

    dest = repo / "reports/perf"
    merged: list[Path] = []
    for name in names:
        with tempfile.TemporaryDirectory() as tmp:
            proc = sh(["gh", "run", "download", run_id, "-n", name,
                       "-D", tmp], cwd=repo, check=False)
            if proc.returncode != 0:
                print(f"warning: failed to download {name}: {proc.stderr}",
                      file=sys.stderr)
                continue
            for src in Path(tmp).rglob("*.json"):
                rel = src.relative_to(tmp)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                merged.append(target)
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=str, default=None)
    p.add_argument("--quants", type=str, default="q8_0,q4_k_m")
    p.add_argument("--samples", type=str, default="jfk,dots")
    p.add_argument("--iters", type=str, default="3")
    p.add_argument("--warmup", type=str, default="1")
    p.add_argument("--name", type=str, default="")
    p.add_argument("--machines", type=str, default="all",
                   choices=["all", "m4-mini", "t14-vulkan"])
    p.add_argument("--hf-repo", type=str, default="")
    p.add_argument("--ref", type=str, default=None)
    p.add_argument("--no-watch", action="store_true")
    p.add_argument("--fetch", type=str, default=None, metavar="RUN_ID")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = find_repo_root(Path(__file__).parent)

    if args.fetch:
        run_id = args.fetch
        rc = 0
    else:
        if not args.models:
            print("error: --models is required (or use --fetch RUN_ID)",
                  file=sys.stderr)
            return 2
        ref = preflight_ref(repo, args.ref)
        not_before = datetime.now(timezone.utc) - timedelta(seconds=10)
        dispatch(repo, args, ref)
        run = locate_run(repo, ref, not_before)
        run_id = str(run["databaseId"])
        print(f"dispatched: {run['url']}")
        if args.no_watch:
            print(f"fetch results later with: "
                  f"uv run scripts/bench/remote.py --fetch {run_id}")
            return 0
        rc = watch(repo, run_id)
        if rc != 0:
            print("run did not fully succeed; downloading whatever "
                  "artifacts exist (fail-fast is off per machine)",
                  file=sys.stderr)

    merged = download_and_merge(repo, run_id)
    if merged:
        print("merged reports:")
        for path in sorted(merged):
            print(f"  {path.relative_to(repo)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
