#!/usr/bin/env python3
"""
run_reference_smart_turn_onnx.py - score pipecat-ai/smart-turn-data-v3.2-test
with upstream's Smart Turn v3.2 ONNX files (fp32 and int8) exactly as
smart-turn's inference.py does, one JSONL row per clip, plus an accuracy
summary. This is the measured reference baseline the C++ port is compared
against (decision parity and accuracy), as reports/wer/*-REF.* is for ASR.

  uv run --project scripts/envs/smart_turn scripts/turn/run_reference_smart_turn_onnx.py \\
      --shards 0 --out reports/turn/smart-turn-v3.2-REF.smart-turn-data-v3.2-test.shard0.jsonl

  # all 10 shards (31,527 clips, ~4.8 GB of parquet in the HF cache)
  uv run --project scripts/envs/smart_turn scripts/turn/run_reference_smart_turn_onnx.py \\
      --out reports/turn/smart-turn-v3.2-REF.smart-turn-data-v3.2-test.jsonl

Rows: {"id","language","dataset","synthetic","label","n_samples","p_fp32",
"p_int8"}. label is the dataset's endpoint_bool (true = turn complete);
upstream's decision is p > 0.5. Clips are 16-bit FLAC at 16 kHz, read as
float32 (int16 / 32768) as soundfile does for the WAV fixtures.
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

from lib import smart_turn_onnx as st  # noqa: E402

DATASET = "pipecat-ai/smart-turn-data-v3.2-test"
DATASET_REVISION = "dfc179241fee65b0efe81ca37786f41ce26f38ec"
N_SHARDS = 10


def shard_path(i: int) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DATASET, f"data/train-{i:05d}-of-{N_SHARDS:05d}.parquet",
                                repo_type="dataset", revision=DATASET_REVISION))


_W: dict = {}


def _init(fp32: str, int8: str) -> None:
    import onnxruntime as ort

    def sess(p):
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(p, sess_options=so, providers=["CPUExecutionProvider"])

    _W["fe"] = st.feature_extractor()
    _W["fp32"] = sess(fp32)
    _W["int8"] = sess(int8)


def _score(item: tuple[dict, bytes]) -> dict:
    import soundfile as sf

    meta, flac = item
    pcm, sr = sf.read(io.BytesIO(flac), dtype="float32", always_2d=False)
    if pcm.ndim != 1 or sr != st.SAMPLE_RATE:
        raise SystemExit(f"error: {meta['id']}: not {st.SAMPLE_RATE} Hz mono")
    x = st.features(_W["fe"], st.truncate_audio_to_last_n_seconds(pcm))
    row = dict(meta)
    row["n_samples"] = int(len(pcm))
    row["p_fp32"] = float(_W["fp32"].run(None, {"input_features": x})[0][0].item())
    row["p_int8"] = float(_W["int8"].run(None, {"input_features": x})[0][0].item())
    return row


def items(shards: list[int]):
    import pyarrow.parquet as pq

    cols = ["id", "language", "dataset", "synthetic", "endpoint_bool"]
    for i in shards:
        pf = pq.ParquetFile(shard_path(i))
        for batch in pf.iter_batches(batch_size=256, columns=cols + ["audio"]):
            for r in batch.to_pylist():
                meta = {"id": r["id"], "language": r["language"], "dataset": r["dataset"],
                        "synthetic": r["synthetic"], "label": bool(r["endpoint_bool"])}
                yield meta, r["audio"]["bytes"]


def summary(rows: list[dict]) -> dict:
    def acc(rs, key):
        if not rs:
            return None
        tp = sum(1 for r in rs if r["label"] and r[key] > st.THRESHOLD)
        tn = sum(1 for r in rs if not r["label"] and r[key] <= st.THRESHOLD)
        fp = sum(1 for r in rs if not r["label"] and r[key] > st.THRESHOLD)
        fn = sum(1 for r in rs if r["label"] and r[key] <= st.THRESHOLD)
        return {"n": len(rs), "acc_pct": round(100.0 * (tp + tn) / len(rs), 2),
                "fpr_pct": round(100.0 * fp / len(rs), 2), "fnr_pct": round(100.0 * fn / len(rs), 2)}

    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r)
    agree = sum(1 for r in rows if (r["p_fp32"] > st.THRESHOLD) == (r["p_int8"] > st.THRESHOLD))
    dp = np.abs(np.array([r["p_fp32"] - r["p_int8"] for r in rows]))
    return {
        "fp32": acc(rows, "p_fp32"),
        "int8": acc(rows, "p_int8"),
        "int8_vs_fp32_decision_agreement_pct": round(100.0 * agree / len(rows), 2),
        "int8_vs_fp32_p_abs_diff": {"max": float(dp.max()), "mean": float(dp.mean())},
        "by_language_fp32": {k: acc(v, "p_fp32") for k, v in sorted(by_lang.items())},
        "note": "FPR / FNR are percentages of all clips, as upstream's benchmark reports them.",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--shards", type=int, nargs="*", default=list(range(N_SHARDS)))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2))
    args = p.parse_args(argv)

    fp32 = st.fetch(st.ONNX_FP32)
    int8 = st.fetch(st.ONNX_INT8)
    t0 = time.time()
    rows = []
    with mp.Pool(args.jobs, initializer=_init, initargs=(str(fp32), str(int8))) as pool:
        for i, row in enumerate(pool.imap(_score, items(args.shards), chunksize=8)):
            rows.append(row)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1} clips, {time.time() - t0:.0f} s", file=sys.stderr)
    s = summary(rows)
    header = {"schema": "transcribe-turn-ref-v1", "dataset": DATASET, "dataset_revision": DATASET_REVISION,
              "shards": args.shards, "model_repo": st.HF_REPO, "model_revision": st.HF_REVISION,
              "fp32": st.ONNX_FP32, "int8": st.ONNX_INT8, "threshold": st.THRESHOLD,
              "rule": "p > 0.5 (inference.py)", "summary": s}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in s.items() if k != "by_language_fp32"}, indent=2))
    print(f"wrote {args.out} ({len(rows)} clips, {time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
