#!/usr/bin/env python3
"""
check_nemotron3_diar_compress.py - selection-emulation gate for the
Nemotron-3-Diarization speaker-cache compression.

The C++ port (src/arch/nemotron3_diar/stream.cpp) re-implements NeMo's
_compress_spkcache on the host. Its correctness property is: GIVEN THE SAME
INPUT PREDICTIONS, it keeps exactly the frames NeMo keeps. (Comparing picks
against a reference *run* conflates this with fp32 GEMM noise in the inputs,
which can legitimately flip a near-tied pick.)

For every C++ compression dump (TRANSCRIBE_NEMOTRON3_DIAR_COMPRESS_DUMP=1,
unforced run), this replays NeMo's SortformerModules._compress_spkcache on the
C++ `input_preds` and requires `topk_indices` and `is_disabled` to be
identical. Exit 1 on any mismatch.

  TRANSCRIBE_DUMP_DIR=<dir> TRANSCRIBE_NEMOTRON3_DIAR_COMPRESS_DUMP=1 \\
    TRANSCRIBE_NEMOTRON3_DIAR_PRESET=<preset> build/bin/transcribe-cli ... <wav>
  uv run --project scripts/envs/nemotron3_diar scripts/diar/check_nemotron3_diar_compress.py \\
    --run <dir>:<preset> [--run <dir2>:<preset2> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import dump_reference_nemotron3_diar_nemo as dumper  # noqa: E402

MODEL = "nvidia/Nemotron-3-Diarization"
REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"


def _load_f32(d: Path, name: str) -> np.ndarray:
    meta = json.loads((d / f"{name}.json").read_text())
    return np.fromfile(d / f"{name}.f32", dtype="<f4").reshape(meta["shape"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="<cpp dump dir>:<preset>")
    args = ap.parse_args()

    import torch

    m = dumper._load_nemo(MODEL, REVISION)
    sm = m.sortformer_modules
    captured: dict[str, np.ndarray] = {}
    orig = sm._get_topk_indices

    def hook(scores):
        idx, dis = orig(scores)
        captured["idx"] = idx[0].numpy().astype(np.int64)
        captured["dis"] = dis[0].numpy().astype(bool)
        return idx, dis

    sm._get_topk_indices = hook

    total = bad = 0
    for spec in args.run:
        d, preset = spec.rsplit(":", 1)
        d = Path(d)
        dumper._apply_preset(m, preset)
        ks = sorted(int(p.name.split(".")[1]) for p in d.glob("compress.*.input_preds.json"))
        if not ks:
            print(f"FAIL {d}: no compress dumps (run with TRANSCRIBE_NEMOTRON3_DIAR_COMPRESS_DUMP=1)")
            return 1
        n_bad = 0
        for k in ks:
            preds = _load_f32(d, f"compress.{k:03d}.input_preds")
            c_idx = _load_f32(d, f"compress.{k:03d}.topk_indices").astype(np.int64)
            c_dis = _load_f32(d, f"compress.{k:03d}.is_disabled") != 0
            with torch.no_grad():
                sm._compress_spkcache(torch.zeros(1, preds.shape[0], sm.fc_d_model),
                                      torch.from_numpy(np.ascontiguousarray(preds))[None].clone(),
                                      torch.zeros(1, sm.fc_d_model))
            if not (np.array_equal(captured["idx"], c_idx) and np.array_equal(captured["dis"], c_dis)):
                n_bad += 1
                print(f"  MISMATCH {d.name} compression {k}: "
                      f"{int((captured['idx'] != c_idx).sum())} picks, "
                      f"{int((captured['dis'] != c_dis).sum())} disabled flags differ")
        total += len(ks)
        bad += n_bad
        print(f"{'ok  ' if n_bad == 0 else 'FAIL'} {d} [{preset}]: {len(ks) - n_bad}/{len(ks)} compressions "
              f"select NeMo's frames on the port's own inputs")
    print(f"{total - bad}/{total} compressions identical")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
