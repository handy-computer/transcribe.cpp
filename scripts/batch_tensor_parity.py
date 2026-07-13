#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""
batch_tensor_parity.py - numerical gate for the parakeet batched encoder.

Proves that each per-utterance encoder output from transcribe_run_batch
matches its single-shot output. A list of varied-length WAVs exercises padding
masks and causal subsampling lengths; repeating one WAV covers same-length
batching.

Apples-to-apples requires both paths on the same attention kernel. Manual F32
attention remains the default for this tool; --flash validates the production
flash-attention policy separately.

Mechanism (TRANSCRIBE_DUMP_DIR): each single-shot run dumps `dec.enc_out`;
the batched run dumps `dec.enc_out.b{i}`. The valid slices must have identical
sizes and values.

Usage:
    uv run scripts/batch_tensor_parity.py \\
        --model models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F16.gguf \\
        --wav samples/jfk.wav --batch 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "CMakeLists.txt").exists() and (p / "scripts").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("cannot locate repo root")


def run(cli, model, args, dump_dir, backend, extra_env=None):
    env = dict(os.environ)
    env["TRANSCRIBE_DUMP_DIR"] = str(dump_dir)
    if extra_env:
        for key, value in extra_env.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    cmd = [str(cli), "-m", str(model), "--backend", backend, "-q"] + args
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"transcribe-cli failed (rc={r.returncode})")


def load(d: Path, name: str) -> np.ndarray:
    return np.fromfile(d / f"{name}.f32", dtype=np.float32)


def main() -> int:
    repo = find_repo_root(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--wav", type=Path,
                     help="single WAV repeated --batch times (default: samples/jfk.wav)")
    src.add_argument("--list", type=Path,
                     help="varied-length WAV list; its length must equal --batch")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--cli", type=Path, default=repo / "build/bin/transcribe-cli")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max abs diff allowed (0 = bit-exact, the CPU expectation)")
    ap.add_argument("--dump-name", default="dec.enc_out",
                    help="dump basename: single-shot writes <name>, the batched "
                         "run writes <name>.b{i} per utterance. parakeet uses "
                         "dec.enc_out; sensevoice (no host decoder) uses "
                         "ctc.log_probs.")
    attn = ap.add_mutually_exclusive_group()
    attn.add_argument("--no-flash", action="store_true",
                      help="force the manual F32 attention path (the default)")
    attn.add_argument("--flash", action="store_true",
                      help="use the default production flash-attention policy")
    args = ap.parse_args()

    for p in (args.model, args.cli):
        if not Path(p).exists():
            raise SystemExit(f"missing: {p}")
    if args.batch < 1:
        raise SystemExit("--batch must be positive")

    if args.list:
        if not args.list.exists():
            raise SystemExit(f"missing: {args.list}")
        wavs = [Path(line.strip()) for line in args.list.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        if len(wavs) != args.batch:
            raise SystemExit(
                f"--list has {len(wavs)} entries; expected --batch={args.batch}")
    else:
        wav = args.wav or (repo / "samples/jfk.wav")
        wavs = [wav] * args.batch
    for wav in wavs:
        if not wav.exists():
            raise SystemExit(f"missing: {wav}")

    # Manual attention remains the default for backward compatibility. The
    # --flash mode separately validates the production policy.
    env = {"TRANSCRIBE_NO_FLASH": None} if args.flash else {"TRANSCRIBE_NO_FLASH": "1"}

    tmp_root = repo / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        td = Path(td)
        batch_dir = td / "batch"
        batch_dir.mkdir()

        singles = []
        for i, wav in enumerate(wavs):
            single_dir = td / f"single-{i}"
            single_dir.mkdir()
            run(args.cli, args.model, [str(wav)], single_dir, args.backend,
                extra_env=env)
            single = load(single_dir, args.dump_name)
            singles.append(single)
            print(f"single {i}: {wav}  {single.size} elems  "
                  f"rms={np.sqrt((single**2).mean()):.4e}")

        list_file = td / "list.txt"
        list_file.write_text("\n".join(str(wav) for wav in wavs) + "\n")
        run(args.cli, args.model,
            ["--batch", str(list_file), "--batch-size", str(args.batch)],
            batch_dir, args.backend, extra_env=env)

        ok = True
        for i, single in enumerate(singles):
            b = load(batch_dir, f"{args.dump_name}.b{i}")
            if not np.isfinite(b).all():
                print(f"  [FAIL] utt {i}: batched dump contains non-finite values")
                ok = False
                continue
            if b.size != single.size:
                print(f"  [FAIL] utt {i}: size {b.size} != {single.size}")
                ok = False
                continue
            d = float(np.abs(b - single).max())
            status = "OK" if d <= args.tol else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"  [{status}] utt {i}: max_abs={d:.3e}")

        print("TENSOR PARITY OK" if ok else "TENSOR PARITY FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
