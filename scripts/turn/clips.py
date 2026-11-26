"""pipecat-ai/smart-turn-data-v3.2-test as the TURN scripts read it: 16-bit
FLAC clips at 16 kHz (read as float32, int16 / 32768), label = endpoint_bool
(true = turn complete), and a clip called complete when p > 0.5."""

from __future__ import annotations

import io

import numpy as np

DATASET = "pipecat-ai/smart-turn-data-v3.2-test"
DATASET_REVISION = "dfc179241fee65b0efe81ca37786f41ce26f38ec"
N_SHARDS = 10
SAMPLE_RATE = 16000
THRESHOLD = 0.5


def items(shards: list[int], only: set[str] | None = None, skip: frozenset[str] = frozenset()):
    """(meta, flac bytes) per clip, in dataset order."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    cols = ["id", "language", "dataset", "synthetic", "endpoint_bool", "audio"]
    for i in shards:
        path = hf_hub_download(DATASET, f"data/train-{i:05d}-of-{N_SHARDS:05d}.parquet",
                               repo_type="dataset", revision=DATASET_REVISION)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=cols):
            for r in batch.to_pylist():
                if (only is not None and r["id"] not in only) or r["id"] in skip:
                    continue
                meta = {"id": r["id"], "language": r["language"], "dataset": r["dataset"],
                        "synthetic": r["synthetic"], "label": bool(r["endpoint_bool"])}
                yield meta, r["audio"]["bytes"]


def read(flac: bytes, what: str = "clip") -> np.ndarray:
    import soundfile as sf

    pcm, sr = sf.read(io.BytesIO(flac), dtype="float32", always_2d=False)
    if pcm.ndim != 1 or sr != SAMPLE_RATE:
        raise SystemExit(f"error: {what}: not {SAMPLE_RATE} Hz mono")
    return np.ascontiguousarray(pcm, dtype=np.float32)


def accuracy(rows: list[dict], key: str) -> dict | None:
    """Accuracy, false positives and false negatives of rows[key] > THRESHOLD,
    in percent of all clips (upstream's benchmark convention)."""
    if not rows:
        return None
    n = len(rows)
    fp = sum(1 for r in rows if not r["label"] and r[key] > THRESHOLD)
    fn = sum(1 for r in rows if r["label"] and r[key] <= THRESHOLD)
    return {"n": n, "acc_pct": round(100.0 * (n - fp - fn) / n, 2),
            "fpr_pct": round(100.0 * fp / n, 2), "fnr_pct": round(100.0 * fn / n, 2)}
