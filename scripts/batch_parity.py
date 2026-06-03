#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
batch_parity.py - verify transcribe_run_batch produces the same transcripts
as the single-shot path, across a set of varied-length utterances.

This is the correctness gate for offline batching. It drives transcribe-cli
in two modes over the SAME wav list:

  - serial   : --batch <list>                  (one transcribe_run per file)
  - batched  : --batch <list> --batch-size N    (transcribe_run_batch groups)

and asserts the per-file hypothesis text is identical between them, for every
requested batch size. Because the serial path is the established source of
truth, "batched == serial in the same build" catches any regression the
batched encoder/dispatch introduces, with no stale fixture to maintain.

Golden mode additionally freezes a baseline so a FUTURE change to the encoder
itself (which would move BOTH serial and batched output together, hiding the
drift from a same-build diff) is caught against today's known-good text:

  # capture today's serial output as the frozen baseline:
  uv run scripts/batch_parity.py --model M.gguf --list samples.txt \\
      --golden-out tests/golden/batch/parakeet-tdt-0.6b-v2.json

  # later, gate both serial and batched against the frozen baseline:
  uv run scripts/batch_parity.py --model M.gguf --list samples.txt \\
      --golden-in tests/golden/batch/parakeet-tdt-0.6b-v2.json

Exit code is non-zero on any mismatch, so this is CI-friendly.

Usage:
    uv run scripts/batch_parity.py \\
        --model models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F16.gguf \\
        --list tmp/batch/list.txt \\
        --batch-sizes 2,4,8 --backend cpu
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("cannot locate repo root")


def run_cli(cli: Path, model: Path, list_file: Path, backend: str,
            language: str | None, batch_size: int) -> dict[str, str]:
    """Run transcribe-cli in batch JSONL mode; return {file: text}."""
    cmd = [
        str(cli), "-m", str(model),
        "--batch", str(list_file), "--batch-jsonl",
        "--backend", backend, "-q",
    ]
    if batch_size > 1:
        cmd += ["--batch-size", str(batch_size)]
    if language:
        cmd += ["--language", language]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"transcribe-cli failed (rc={proc.returncode})")
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("type") == "batch_header":
            continue
        out[d["file"]] = d.get("text", "")
    return out


def diff_texts(a: dict[str, str], b: dict[str, str]) -> list[str]:
    keys = sorted(set(a) | set(b))
    diffs = []
    for k in keys:
        if a.get(k) != b.get(k):
            diffs.append(k)
    return diffs


def report(label: str, ref: dict[str, str], cur: dict[str, str]) -> bool:
    diffs = diff_texts(ref, cur)
    if not diffs:
        print(f"  [OK]   {label}: {len(cur)} utterances match")
        return True
    print(f"  [FAIL] {label}: {len(diffs)}/{len(cur)} mismatch")
    for k in diffs[:8]:
        print(f"    DIFF {k}")
        print(f"      ref: {ref.get(k, '<missing>')[:160]}")
        print(f"      cur: {cur.get(k, '<missing>')[:160]}")
    return False


def main() -> int:
    repo = find_repo_root(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cli", type=Path,
                    default=repo / "build/bin/transcribe-cli")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--list", type=Path,
                     help="file with one wav path per line")
    src.add_argument("--samples-dir", type=Path,
                     help="directory of *.wav to build the list from")
    ap.add_argument("--batch-sizes", default="2,4,8",
                    help="comma-separated batch sizes to test")
    ap.add_argument("--backend", default="cpu",
                    help="cpu (default, deterministic) / auto / metal / ...")
    ap.add_argument("--language", default=None,
                    help="optional BCP-47 hint passed to every utterance")
    ap.add_argument("--golden-out", type=Path,
                    help="capture the serial baseline to this JSON and exit")
    ap.add_argument("--golden-in", type=Path,
                    help="also gate serial AND batched against this frozen JSON")
    args = ap.parse_args()

    # Resolve the wav list.
    if args.samples_dir:
        wavs = sorted(glob.glob(str(args.samples_dir / "*.wav")))
        if not wavs:
            raise SystemExit(f"no wavs under {args.samples_dir}")
        list_file = repo / "tmp" / "batch_parity_list.txt"
        list_file.parent.mkdir(parents=True, exist_ok=True)
        list_file.write_text("\n".join(wavs) + "\n")
    elif args.list:
        list_file = args.list
    else:
        raise SystemExit("one of --list / --samples-dir is required")

    for p in (args.model, args.cli, list_file):
        if not Path(p).exists():
            raise SystemExit(f"missing: {p}")

    print(f"model:   {args.model}")
    print(f"list:    {list_file}")
    print(f"backend: {args.backend}")

    # Serial baseline (source of truth for same-build parity).
    serial = run_cli(args.cli, args.model, list_file, args.backend,
                     args.language, batch_size=1)
    print(f"serial:  {len(serial)} utterances")

    # Golden capture mode: freeze serial and exit.
    if args.golden_out:
        args.golden_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": str(args.model),
            "backend": args.backend,
            "language": args.language,
            "texts": serial,
        }
        args.golden_out.write_text(json.dumps(payload, indent=2,
                                              ensure_ascii=False) + "\n")
        print(f"captured golden baseline -> {args.golden_out} "
              f"({len(serial)} utterances)")
        return 0

    ok = True
    sizes = [int(s) for s in args.batch_sizes.split(",") if s.strip()]

    # Optional: gate serial itself against a frozen golden (catches encoder
    # drift that would move serial+batched together).
    if args.golden_in:
        golden = json.loads(Path(args.golden_in).read_text())["texts"]
        ok &= report("serial vs golden", golden, serial)

    # Batched vs serial (same build) for each requested batch size.
    for n in sizes:
        batched = run_cli(args.cli, args.model, list_file, args.backend,
                          args.language, batch_size=n)
        ok &= report(f"batch-size {n} vs serial", serial, batched)
        if args.golden_in:
            golden = json.loads(Path(args.golden_in).read_text())["texts"]
            ok &= report(f"batch-size {n} vs golden", golden, batched)

    print("PARITY OK" if ok else "PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
