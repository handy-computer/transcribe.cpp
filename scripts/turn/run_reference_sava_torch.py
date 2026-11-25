#!/usr/bin/env python3
"""
run_reference_sava_torch.py - score pipecat-ai/smart-turn-data-v3.2-test with
upstream's own SAVA package (inworld-sava 1.0.0, PyTorch CPU fp32), one JSONL
row per clip, plus an accuracy summary. This is the per-clip reference the C++
SAVA runs (scripts/turn/eval_cpp.py) are compared against, as
run_reference_smart_turn_onnx.py is for Smart Turn.

  uv run --project scripts/envs/sava --with pyarrow scripts/turn/run_reference_sava_torch.py \\
      --out reports/turn/sava-REF.smart-turn-data-v3.2-test.jsonl

  # agreement of a C++ run with it
  uv run --project scripts/envs/smart_turn scripts/turn/score.py \\
      reports/turn/sava-F32.smart-turn-data-v3.2-test.jsonl \\
      --ref-jsonl reports/turn/sava-REF.smart-turn-data-v3.2-test.jsonl

A clip's score is what transcribe_turn_run computes (docs/porting/families/
sava.md, "For transcribe_turn_run"): inworld.sava.run(model, clip) over the
whole clip, p = turns[n - 1, 0], the released turn score of the last real
frame. The decision is p > 0.5, as Inworld's and Pipecat's benchmarks use.
Clips are 16-bit FLAC at 16 kHz, read as float32 (int16 / 32768).

Rows: {"id","language","dataset","synthetic","label","n_samples","p_ref"}.
Rows are appended to <out>.partial as they finish (progress: its line count);
<out> is written with the header and summary at the end.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "turn"))

from dump_reference_sava_author import (  # noqa: E402
    HF_REPO, HF_REVISION, SAMPLE_RATE, SAVA_VERSION, WEIGHTS_SHA256, resolve_weights)

DATASET = "pipecat-ai/smart-turn-data-v3.2-test"
DATASET_REVISION = "dfc179241fee65b0efe81ca37786f41ce26f38ec"
N_SHARDS = 10
THRESHOLD = 0.5


def shard_path(i: int) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DATASET, f"data/train-{i:05d}-of-{N_SHARDS:05d}.parquet",
                                repo_type="dataset", revision=DATASET_REVISION))


_W: dict = {}


def _init(weights: str) -> None:
    import torch
    from inworld import sava
    from inworld.sava import checkpoint

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sava.__version__ != SAVA_VERSION:
        raise SystemExit(f"error: inworld-sava {sava.__version__} installed, pinned to {SAVA_VERSION}")
    _W["sava"] = sava
    _W["model"] = checkpoint.load(weights)  # CPU, fp32 (bf16 weights upcast)


def clip_p(sava, model, audio: np.ndarray) -> float:
    """transcribe_turn_run's p: the last real frame's released turn score."""
    import torch

    with torch.inference_mode():
        curves = sava.run(model, audio)
    turns = np.asarray(curves.turns)
    n = sava.streaming.num_frames(audio.size)
    if turns.shape[0] != n:
        raise SystemExit(f"error: sava.run gave {turns.shape[0]} frames, expected {n}")
    return float(turns[n - 1, 0])


def _score(item: tuple[dict, bytes]) -> dict:
    import soundfile as sf

    meta, flac = item
    pcm, sr = sf.read(io.BytesIO(flac), dtype="float32", always_2d=False)
    if pcm.ndim != 1 or sr != SAMPLE_RATE:
        raise SystemExit(f"error: {meta['id']}: not {SAMPLE_RATE} Hz mono")
    audio = np.ascontiguousarray(pcm, dtype=np.float32)
    row = dict(meta)
    row["n_samples"] = int(audio.size)
    row["p_ref"] = clip_p(_W["sava"], _W["model"], audio)
    return row


def items(shards: list[int], ids: set[str] | None = None, skip: set[str] | None = None):
    import pyarrow.parquet as pq

    cols = ["id", "language", "dataset", "synthetic", "endpoint_bool"]
    for i in shards:
        pf = pq.ParquetFile(shard_path(i))
        for batch in pf.iter_batches(batch_size=256, columns=cols + ["audio"]):
            for r in batch.to_pylist():
                if ids is not None and r["id"] not in ids:
                    continue
                if skip and r["id"] in skip:
                    continue
                meta = {"id": r["id"], "language": r["language"], "dataset": r["dataset"],
                        "synthetic": r["synthetic"], "label": bool(r["endpoint_bool"])}
                yield meta, r["audio"]["bytes"]


def summary(rows: list[dict]) -> dict:
    def acc(rs):
        if not rs:
            return None
        tp = sum(1 for r in rs if r["label"] and r["p_ref"] > THRESHOLD)
        tn = sum(1 for r in rs if not r["label"] and r["p_ref"] <= THRESHOLD)
        fp = sum(1 for r in rs if not r["label"] and r["p_ref"] > THRESHOLD)
        fn = sum(1 for r in rs if r["label"] and r["p_ref"] <= THRESHOLD)
        return {"n": len(rs), "acc_pct": round(100.0 * (tp + tn) / len(rs), 2),
                "fpr_pct": round(100.0 * fp / len(rs), 2), "fnr_pct": round(100.0 * fn / len(rs), 2)}

    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r)
    return {"ref": acc(rows), "by_language": {k: acc(v) for k, v in sorted(by_lang.items())},
            "note": "FPR / FNR are percentages of all clips, as upstream's benchmark reports them."}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--shards", type=int, nargs="*", default=list(range(N_SHARDS)))
    p.add_argument("--ids", nargs="*", default=None, help="only these clip ids (a spot check)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2))
    p.add_argument("--resume", action="store_true", help="keep the rows already in <out>.partial")
    args = p.parse_args(argv)

    weights = resolve_weights(HF_REPO, HF_REVISION)
    import hashlib

    h = hashlib.sha256()
    with weights.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != WEIGHTS_SHA256:
        raise SystemExit(f"error: {weights}: sha256 {h.hexdigest()}, expected {WEIGHTS_SHA256}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    rows: list[dict] = []
    if args.resume and partial.exists():
        rows = [json.loads(line) for line in partial.read_text().splitlines() if line.strip()]
    done = {r["id"] for r in rows}
    t0 = time.time()
    with partial.open("a" if args.resume else "w") as pf, \
            mp.Pool(args.jobs, initializer=_init, initargs=(str(weights),)) as pool:
        src = items(args.shards, set(args.ids) if args.ids else None, done)
        for i, row in enumerate(pool.imap(_score, src, chunksize=8)):
            rows.append(row)
            pf.write(json.dumps(row) + "\n")
            if (i + 1) % 1000 == 0:
                pf.flush()
                print(f"  {len(rows)} clips, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    s = summary(rows)
    header = {"schema": "transcribe-turn-ref-v1", "dataset": DATASET, "dataset_revision": DATASET_REVISION,
              "shards": args.shards, "model_repo": HF_REPO, "model_revision": HF_REVISION,
              "weights_sha256": WEIGHTS_SHA256, "framework": f"inworld-sava {SAVA_VERSION}",
              "precision": "torch CPU fp32, 1 thread per worker", "entrypoint": "inworld.sava.run",
              "score": "p = curves.turns[n - 1, 0] (transcribe_turn_run's p)", "threshold": THRESHOLD,
              "rule": "p > 0.5", "summary": s}
    with args.out.open("w") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(s["ref"], indent=2))
    print(f"wrote {args.out} ({len(rows)} clips, {time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
