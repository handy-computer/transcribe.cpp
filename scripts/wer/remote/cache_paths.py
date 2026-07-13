from __future__ import annotations

import hashlib

from dataset_specs import dataset_id


def hyp_cache_paths(
    hyp_fp: str,
    model_file: str,
    dataset_spec: str,
    n_utts: int | None,
    batch_size: int = 1,
    sort_by_length: bool = True,
    timestamps: str = "none",
    language: str = "",
    stream_chunk_ms: int = 0,
    stream_att_right: int = -1,
    env_extra: str = "",
) -> tuple[str, str]:
    """Deterministic Volume paths for the (model, dataset, subset, batch,
    sort) tuple.

    Keyed under HYP_FP (source + run.py + ingest.py); changes to any of
    these invalidate the cache. Dispatcher edits do not. batch_size and the
    sort flag are part of the key because, while the hyp text is identical,
    the summary's wall_s / rtf are not (length-sorting changes how much
    padding each batch wastes). The default sorted case carries no extra
    tag so it stays compatible with already-cached entries. Returns
    (hyp_jsonl_path, summary_json_path).
    """
    slug = model_file.replace(".gguf", "")
    subset_tag = "full" if n_utts is None else f"n{n_utts}"
    bs_tag = "" if batch_size <= 1 else f".b{batch_size}"
    sort_tag = "" if sort_by_length else ".nosort"
    # Default "none" carries no tag so it stays compatible with already-cached
    # entries; any other timestamp mode gets its own cache slot.
    ts_tag = "" if timestamps == "none" else f".ts-{timestamps}"
    # Language override (default "" = let run.py infer from dataset). Folded
    # in so e.g. en vs en-US do not share a cache entry.
    lang_tag = "" if not language else f".lang-{language}"
    # Streaming (--stream-chunk-ms) produces different hyps than the offline
    # path, so it gets its own cache slot. 0 = offline (no tag, compatible).
    stream_tag = "" if stream_chunk_ms <= 0 else f".stream{stream_chunk_ms}ms"
    # Cache-aware latency preset (--stream-att-right). -1 = unset (model
    # default R), no tag so it stays compatible with already-cached entries;
    # any explicit R gets its own slot so e.g. R=13 and R=0 never collide.
    r_tag = "" if stream_att_right < 0 else f".r{stream_att_right}"
    # Extra TRANSCRIBE_* env vars injected into the CLI run (A/B attribution
    # sweeps). Default "" carries no tag so it stays compatible with
    # already-cached entries; any override gets its own slot.
    env_tag = ""
    if env_extra:
        env_tag = "." + hashlib.sha256(env_extra.encode()).hexdigest()[:8]
    base = (f"/data/wer/hyps/{hyp_fp}/{slug}."
            f"{dataset_id(dataset_spec)}.{subset_tag}{bs_tag}{sort_tag}{ts_tag}{lang_tag}{stream_tag}{r_tag}{env_tag}")
    return f"{base}.jsonl", f"{base}.summary.json"


def ref_hyp_cache_paths(
    variant: str,
    dataset_spec: str,
    n_utts: int | None,
    batch_size: int,
    env_fp: str,
    mode: str = "offline",
    extra_args: list | None = None,
) -> tuple[str, str]:
    subset_tag = "full" if n_utts is None else f"n{n_utts}"
    bs_tag = "" if batch_size <= 1 else f".b{batch_size}"
    # "offline" carries no tag (compatible with already-cached entries); any
    # other mode (e.g. "streaming") gets its own cache slot.
    mode_tag = "" if mode == "offline" else f".{mode}"
    # Fold extra runner args into the key so e.g. --language en and
    # --language en-US do not share a cache entry.
    if extra_args:
        args_tag = "." + hashlib.sha256(
            " ".join(extra_args).encode()).hexdigest()[:8]
    else:
        args_tag = ""
    base = (f"/data/wer/ref-hyps/{env_fp}/{variant}-REF."
            f"{dataset_id(dataset_spec)}.{subset_tag}{bs_tag}{mode_tag}{args_tag}")
    return f"{base}.jsonl", f"{base}.summary.json"
