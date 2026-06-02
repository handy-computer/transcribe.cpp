#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""
batch_tensor_parity.py - numerical gate for the parakeet batched encoder.

Proves that the per-utterance encoder output of transcribe_run_batch is
identical, tensor-for-tensor, to the single-shot encoder output for the same
wav. This is the gate the variable-length padding mask must keep green: a
wrong mask perturbs real frames and shows up here as drift, where WER alone
would miss it.

Apples-to-apples requires both paths on the SAME attention kernel. The
single-shot path defaults to flash attention (which casts the rel-pos mask to
F16); the batched path always runs the manual F32 attention. We force the
single-shot path to manual with TRANSCRIBE_NO_FLASH=1 so the comparison is
exact on CPU.

Mechanism (TRANSCRIBE_DUMP_DIR): single-shot dumps `dec.enc_out`; the batched
run dumps `dec.enc_out.b{i}` per utterance. We feed B copies of one clip and
assert every batched slice equals the single-shot dump.

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
        env.update(extra_env)
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
    ap.add_argument("--wav", type=Path, default=repo / "samples/jfk.wav")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--cli", type=Path, default=repo / "build/bin/transcribe-cli")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max abs diff allowed (0 = bit-exact, the CPU expectation)")
    args = ap.parse_args()

    for p in (args.model, args.wav, args.cli):
        if not Path(p).exists():
            raise SystemExit(f"missing: {p}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        single_dir = td / "single"
        batch_dir = td / "batch"
        single_dir.mkdir(); batch_dir.mkdir()

        # Single-shot, manual attention (TRANSCRIBE_NO_FLASH).
        run(args.cli, args.model, [str(args.wav)], single_dir, args.backend,
            extra_env={"TRANSCRIBE_NO_FLASH": "1"})

        # Batched run of B copies (same length -> the real fused encoder).
        list_file = td / "list.txt"
        list_file.write_text("\n".join([str(args.wav)] * args.batch) + "\n")
        run(args.cli, args.model,
            ["--batch", str(list_file), "--batch-size", str(args.batch)],
            batch_dir, args.backend, extra_env={"TRANSCRIBE_NO_FLASH": "1"})

        single = load(single_dir, "dec.enc_out")
        print(f"single-shot enc_out: {single.size} elems  "
              f"rms={np.sqrt((single**2).mean()):.4e}")

        ok = True
        for i in range(args.batch):
            b = load(batch_dir, f"dec.enc_out.b{i}")
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
