from __future__ import annotations

from pathlib import Path

from dataset_specs import dataset_id


def write_ref_hyp(
    root: Path,
    hyp_jsonl: str,
    variant: str,
    dataset_spec: str,
    batch_size: int | None = None,
    mode: str = "offline",
) -> Path:
    out_dir = root / "reports" / "wer"
    out_dir.mkdir(parents=True, exist_ok=True)
    bs_tag = "" if not batch_size or batch_size <= 1 else f".b{batch_size}"
    mode_tag = "" if mode == "offline" else f".{mode}"
    out_path = out_dir / f"{variant}-REF.{dataset_id(dataset_spec)}{bs_tag}{mode_tag}.jsonl"
    out_path.write_text(hyp_jsonl)
    return out_path


def write_hyp(
    root: Path,
    hyp_jsonl: str,
    model_file: str,
    dataset_spec: str,
    batch_size: int | None = None,
    timestamps: str = "none",
    stream_chunk_ms: int = 0,
) -> Path:
    out_dir = root / "reports" / "wer"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model_file.replace(".gguf", "")
    ds = dataset_id(dataset_spec)
    bs_tag = "" if batch_size is None else f".b{batch_size}"
    ts_tag = "" if timestamps == "none" else f".ts-{timestamps}"
    stream_tag = "" if stream_chunk_ms <= 0 else f".stream{stream_chunk_ms}ms"
    out_path = out_dir / f"{slug}.{ds}{bs_tag}{ts_tag}{stream_tag}.jsonl"
    out_path.write_text(hyp_jsonl)
    return out_path
