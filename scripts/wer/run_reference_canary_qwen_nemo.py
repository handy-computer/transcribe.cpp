#!/usr/bin/env python3
"""
run_reference_canary_qwen_nemo.py — NeMo SALM (canary-qwen) batch transcribe.

Loads `nemo.collections.speechlm2.models.SALM` once and runs it over a WER
manifest, writing run.py-compatible JSONL so scripts/wer/score.py can
score the output. Mirrors the dump-time setup
(scripts/dump_reference_canary_qwen_nemo.py): SALM chat-style prompt with
the audio_locator placeholder, greedy generate(), tokenizer.ids_to_text
for the decode.

Default (local) usage — uses the family's uv env for NeMo / torch:

    uv run --project scripts/envs/canary_qwen \\
      scripts/wer/run_reference_canary_qwen_nemo.py \\
        --model    nvidia/canary-qwen-2.5b \\
        --manifest samples/wer/test-clean.512.manifest.jsonl \\
        --out      reports/wer/canary-qwen-2.5b-REF.test-clean-512.jsonl

Optional Modal (GPU) speed-up — submits the same run to a Modal container
with a GPU and streams the JSONL back. Requires `modal` installed
locally (`uv tool install modal` and `modal token new`); it is NOT a
dependency of the canary_qwen env, so this path is opt-in:

    uv run --with modal scripts/wer/run_reference_canary_qwen_nemo.py \\
        --modal --gpu H100 \\
        --model nvidia/canary-qwen-2.5b \\
        --dataset librispeech:test-clean \\
        [--out reports/wer/<auto>.jsonl] [--limit N]

In --modal mode the dataset is ingested inside the container (cached on
a Modal Volume), so no local manifest or audio upload is needed. The
existing --manifest path is rejected with --modal to avoid silent audio
mismatch between host and container.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path


def _resolve_repo_root() -> Path:
    """Resolve transcribe.cpp/ from either layout.

    Local: scripts/wer/run_reference_canary_qwen_nemo.py -> parents[2].
    Modal container: the script is shipped flat to /root/ — no repo
    layout exists, so fall back to /repo (where the function body
    hardcodes its working tree).
    """
    here = Path(__file__).resolve()
    if len(here.parents) >= 3 and (here.parents[2] / "CMakeLists.txt").exists():
        return here.parents[2]
    return Path("/repo")


REPO_ROOT = _resolve_repo_root()


# ---------- argparse --------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Local-mode args (also forwarded to the container re-exec under --modal).
    p.add_argument("--manifest", type=Path, default=None,
                   help="Input manifest JSONL (id/audio/ref_text). "
                        "Required in local mode; rejected with --modal.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSONL path. Required in local mode; "
                        "auto-derived in --modal mode if omitted.")
    p.add_argument("--model", default="nvidia/canary-qwen-2.5b",
                   help="HF repo id for the SALM checkpoint.")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="Per-utterance generation cap (matches the model card example).")
    p.add_argument("--prompt-template",
                   default="Transcribe the following: {locator}",
                   help="Chat-content template; {locator} is replaced with the model's audio_locator_tag.")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch intra-op threads for deterministic runs.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N utterances (0 = all).")
    p.add_argument("--device", default="cpu",
                   help="torch device (cpu / cuda / mps). Default: cpu for determinism. "
                        "Forced to 'cuda' under --modal.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Number of utterances per SALM.generate() call.")
    p.add_argument("--model-dtype", choices=("auto", "bf16", "f16", "f32"),
                   default="auto",
                   help="Optionally move floating model parameters to this dtype after load.")
    p.add_argument("--autocast", choices=("off", "bf16", "f16"), default="off",
                   help="Run generate() under torch.autocast with this dtype.")
    p.add_argument("--cuda-tf32", action="store_true",
                   help="Allow TF32 matmul/cuDNN kernels for any FP32 CUDA ops.")
    p.add_argument("--cuda-sync-timing", action="store_true",
                   help="Synchronize CUDA before/after each timed generate() call.")
    p.add_argument("--warmup-batches", type=int, default=0,
                   help="Run this many untimed batches before writing the report.")

    # Modal opt-in.
    g = p.add_argument_group("Modal (opt-in GPU acceleration)")
    g.add_argument("--modal", action="store_true",
                   help="Submit this run to Modal instead of executing locally.")
    g.add_argument("--gpu", default="A10G",
                   help="Modal GPU spec (A10G, A100-40GB, A100-80GB, H100, L4...).")
    g.add_argument("--dataset", default=None,
                   help="proto:arg shorthand (e.g. librispeech:test-clean, fleurs:es). "
                        "Required with --modal; the container runs scripts/wer/ingest.py "
                        "to build the manifest, so no local audio upload is needed.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.modal:
        return _submit_to_modal(args)

    # Local-mode required args.
    missing = []
    if args.manifest is None:
        missing.append("--manifest")
    if args.out is None:
        missing.append("--out")
    if missing:
        print(f"error: {' and '.join(missing)} required in local mode "
              f"(omit --modal, or pass them for --modal too)",
              file=sys.stderr)
        return 2
    return _run_local(args)


# ---------- Local mode (existing behaviour) ---------------------------------

def _chunks(rows: list[dict], batch_size: int):
    batch_size = max(1, batch_size)
    for i in range(0, len(rows), batch_size):
        yield i // batch_size, rows[i:i + batch_size]


def _torch_dtype(torch, name: str):
    if name in ("auto", "off"):
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "f16":
        return torch.float16
    if name == "f32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")


def _dtype_name(dtype) -> str:
    text = str(dtype)
    return text.removeprefix("torch.")


def _model_param_summary(model) -> dict:
    total_params = 0
    total_bytes = 0
    by_device: dict[str, int] = {}
    by_dtype: dict[str, int] = {}
    first_device = None
    first_dtype = None

    for p in model.parameters():
        n = int(p.numel())
        total_params += n
        total_bytes += n * int(p.element_size())
        dev = str(p.device)
        dt = _dtype_name(p.dtype)
        by_device[dev] = by_device.get(dev, 0) + n
        by_dtype[dt] = by_dtype.get(dt, 0) + n
        if first_device is None:
            first_device = dev
            first_dtype = dt

    return {
        "param_count": total_params,
        "param_bytes": total_bytes,
        "param_gib": round(total_bytes / (1024 ** 3), 3),
        "param_elements_by_device": by_device,
        "param_elements_by_dtype": by_dtype,
        "first_param_device": first_device,
        "first_param_dtype": first_dtype,
    }


def _cuda_snapshot(torch, device) -> dict:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "cuda_device_index": int(index),
        "cuda_device_name": torch.cuda.get_device_name(index),
        "cuda_capability": f"{props.major}.{props.minor}",
        "cuda_total_memory_bytes": int(props.total_memory),
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
    }


def _run_local(args: argparse.Namespace) -> int:
    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("error: --batch-size must be >= 1", file=sys.stderr)
        return 2
    if args.warmup_batches < 0:
        print("error: --warmup-batches must be >= 0", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            print("error: --device cuda requested but torch.cuda is unavailable",
                  file=sys.stderr)
            return 2
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.cuda_tf32)
        try:
            torch.backends.cudnn.allow_tf32 = bool(args.cuda_tf32)
        except AttributeError:
            pass
        if args.cuda_tf32:
            torch.set_float32_matmul_precision("high")

    print(f"loading: {args.model}")
    t0 = time.monotonic()
    import nemo.collections.speechlm2 as slm
    model = slm.SALM.from_pretrained(
        args.model, map_location=args.device, strict=True,
    )
    target_dtype = _torch_dtype(torch, args.model_dtype)
    if target_dtype is None:
        model.to(device=device)
    else:
        model.to(device=device, dtype=target_dtype)
    model.eval()

    # Match the dump-time determinism overrides: zero out preprocessor
    # dither so repeat runs produce identical mel inputs (the SALM cfg
    # declares 1e-5 by default which adds tiny per-run noise).
    pre = getattr(model, "perception", None)
    pre = getattr(pre, "preprocessor", None) if pre is not None else None
    if pre is not None and hasattr(pre, "featurizer") and hasattr(pre.featurizer, "dither"):
        prev = float(pre.featurizer.dither)
        if prev != 0.0:
            print(f"  overriding preprocessor dither {prev} -> 0.0 for deterministic runs")
            pre.featurizer.dither = 0.0

    locator = getattr(model, "audio_locator_tag", "<|audioplaceholder|>")
    content_template = args.prompt_template
    if "{locator}" not in content_template:
        # Append a locator if the user gave a template without one;
        # SALM's audio_locator scatter requires at least one placeholder.
        content_template = content_template.rstrip() + " " + locator

    load_ms = (time.monotonic() - t0) * 1000
    param_summary = _model_param_summary(model)
    cuda_after_load = _cuda_snapshot(torch, device)

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[: args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")
    print(f"prompt:   '{content_template.format(locator=locator)}'")
    print(
        f"max_new_tokens: {args.max_new_tokens}  device: {args.device}  "
        f"batch_size: {args.batch_size}"
    )
    print(
        f"model params: {param_summary['param_gib']:.3f} GiB, "
        f"devices={param_summary['param_elements_by_device']}, "
        f"dtypes={param_summary['param_elements_by_dtype']}"
    )
    if cuda_after_load:
        print(
            "cuda after load: "
            f"allocated={cuda_after_load['cuda_memory_allocated_bytes'] / (1024 ** 3):.2f} GiB, "
            f"reserved={cuda_after_load['cuda_memory_reserved_bytes'] / (1024 ** 3):.2f} GiB, "
            f"device={cuda_after_load['cuda_device_name']}"
        )

    tokenizer = getattr(model, "tokenizer", None)

    def decode_ids(ids) -> str:
        if hasattr(ids, "cpu"):
            ids = ids.detach().cpu().tolist()
        elif not isinstance(ids, list):
            ids = list(ids)
        if tokenizer is not None and hasattr(tokenizer, "ids_to_text"):
            try:
                return tokenizer.ids_to_text([int(x) for x in ids]).strip()
            except Exception as exc:
                print(f"  warning: ids_to_text failed: {exc}")
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                return tokenizer.decode([int(x) for x in ids],
                                        skip_special_tokens=True).strip()
            except Exception as exc:
                print(f"  warning: tokenizer.decode failed: {exc}")
        return ""

    autocast_dtype = _torch_dtype(torch, args.autocast)

    def autocast_context():
        if autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)

    def cuda_sync():
        if args.cuda_sync_timing and device.type == "cuda":
            torch.cuda.synchronize(device)

    def rows_from_answer_ids(answer_ids, expected: int):
        if hasattr(answer_ids, "shape") and hasattr(answer_ids, "__getitem__"):
            rows = []
            n = int(answer_ids.shape[0]) if len(answer_ids.shape) > 0 else 0
            for i in range(expected):
                rows.append(answer_ids[i] if i < n else [])
            return rows
        if isinstance(answer_ids, (list, tuple)):
            if len(answer_ids) == expected:
                return list(answer_ids)
            if expected == 1:
                return [answer_ids[0] if answer_ids else []]
            rows = list(answer_ids[:expected])
            rows.extend([] for _ in range(expected - len(rows)))
            return rows
        return [[] for _ in range(expected)]

    def generate_batch(batch: list[dict]):
        content = content_template.format(locator=locator)
        prompts = [[{
            "role": "user",
            "content": content,
            "audio": [entry["audio"]],
        }] for entry in batch]
        cuda_sync()
        t_start = time.monotonic()
        try:
            with torch.inference_mode():
                with autocast_context():
                    answer_ids = model.generate(
                        prompts=prompts,
                        max_new_tokens=args.max_new_tokens,
                    )
            cuda_sync()
            err = ""
            rows = rows_from_answer_ids(answer_ids, len(batch))
        except Exception as e:
            cuda_sync()
            err = f"{type(e).__name__}: {e}"
            rows = [[] for _ in batch]
        elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
        return rows, elapsed_ms, err

    cuda_after_warmup = {}
    if args.warmup_batches and manifest:
        warm_batch = manifest[:args.batch_size]
        for i in range(args.warmup_batches):
            _, warm_ms, warm_err = generate_batch(warm_batch)
            status = f" error={warm_err}" if warm_err else ""
            print(f"warmup batch {i + 1}/{args.warmup_batches}: {warm_ms:.1f} ms{status}",
                  flush=True)
        cuda_after_warmup = _cuda_snapshot(torch, device)

    n_done = 0
    n_errors = 0
    t_loop = time.monotonic()

    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header",
            "load_ms": round(load_ms, 1),
            "framework": "nemo-salm",
            "model": args.model,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "model_dtype_arg": args.model_dtype,
            "autocast": args.autocast,
            "cuda_tf32": bool(args.cuda_tf32),
            "cuda_sync_timing": bool(args.cuda_sync_timing),
            "warmup_batches": args.warmup_batches,
            "torch_threads": args.torch_threads,
            "torch_version": getattr(torch, "__version__", None),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "model_param_summary": param_summary,
            "cuda_after_load": cuda_after_load,
            "cuda_after_warmup": cuda_after_warmup,
            "prompt_template": content_template,
            "audio_locator": locator,
        }) + "\n")
        fout.flush()

        for batch_index, batch in _chunks(manifest, args.batch_size):
            rows, batch_ms, batch_err = generate_batch(batch)
            per_item_ms = round(batch_ms / max(1, len(batch)), 1)
            cuda_after_batch = _cuda_snapshot(torch, device)
            if batch_err:
                n_errors += len(batch)

            for entry, row_ids in zip(batch, rows):
                hyp_text = "" if batch_err else decode_ids(row_ids)
                rec = {
                    "id": entry["id"],
                    "ref_text": entry.get("ref_text", ""),
                    "hyp_text": hyp_text,
                    "raw_text": hyp_text,
                    "mel_ms": 0,
                    "encode_ms": 0,
                    "decode_ms": per_item_ms,
                    "latency_ms": per_item_ms,
                    "batch_ms": batch_ms,
                    "batch_size": len(batch),
                    "batch_index": batch_index,
                    "cuda_after_batch": cuda_after_batch,
                    "error": batch_err,
                }
                fout.write(json.dumps(rec) + "\n")
                n_done += 1
            fout.flush()
            if n_done % 25 == 0 or n_done == total:
                wall = time.monotonic() - t_loop
                rate = n_done / wall if wall > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(
                    f"  [{n_done}/{total}] {rate:.2f} utt/s, "
                    f"ETA {eta/60:.1f} min, errors={n_errors}",
                    flush=True,
                )

    wall = time.monotonic() - t_loop
    print(
        f"\ndone. {n_done} utterances in {wall:.1f}s "
        f"({n_done / wall:.2f} utt/s), {n_errors} errors"
    )
    print(f"report: {args.out}")
    return 0 if n_errors == 0 else 1


# ---------- Modal mode (opt-in, lazy imports) ------------------------------

_MODAL_DEPS = [
    "torch>=2.2",
    "nemo_toolkit[asr]>=2.5",
    "transformers>=4.45",
    "peft>=0.13",
    "huggingface-hub>=0.20",
    "soundfile>=0.12",
    "numpy>=1.26",
    "sentencepiece>=0.2",
    "librosa>=0.10",
    "datasets>=3.6",
]
_MODAL_APT = ["ffmpeg", "libsndfile1", "git", "build-essential"]


def _modal_default_out(args: argparse.Namespace) -> Path:
    """reports/wer/<model-slug>-REF-<gpu>.<dataset-slug>.jsonl"""
    model_slug = args.model.split("/")[-1]
    dataset_slug = args.dataset.replace(":", "-")
    gpu_slug = args.gpu.lower().replace("-", "")
    return (
        REPO_ROOT / "reports" / "wer"
        / f"{model_slug}-REF-{gpu_slug}.{dataset_slug}.jsonl"
    )


def _modal_default_out(model: str, dataset: str, gpu: str) -> Path:
    """reports/wer/<model-slug>-REF-<gpu>.<dataset-slug>.jsonl"""
    model_slug = model.split("/")[-1]
    dataset_slug = dataset.replace(":", "-")
    gpu_slug = gpu.lower().replace("-", "")
    return (
        REPO_ROOT / "reports" / "wer"
        / f"{model_slug}-REF-{gpu_slug}.{dataset_slug}.jsonl"
    )


# ---- Module-level Modal app (only defined if `modal` is importable) -------
#
# `modal` is NOT a dep of scripts/envs/canary_qwen — it's an opt-in for
# `--modal` mode. The conditional import keeps local-mode invocations
# working in the family env (which has no `modal`) while still letting
# `uv run --with modal ... --modal` users invoke the GPU path.
#
# Module-level definition (vs the previous in-function `serialized=True`
# pattern) avoids cloudpickle's Python-version-parity requirement: the
# local interpreter can be any version, and only the container's Python
# (pinned in debian_slim below) needs to match nemo_toolkit's support
# matrix.

try:
    import modal as _modal
    _HAS_MODAL = True
except ImportError:
    _HAS_MODAL = False

if _HAS_MODAL:
    _modal_app = _modal.App("transcribe-wer-canary-qwen")
    _modal_hf_vol = _modal.Volume.from_name(
        "transcribe-hf-cache", create_if_missing=True
    )
    _modal_samples_vol = _modal.Volume.from_name(
        "transcribe-wer-samples", create_if_missing=True
    )

    _modal_image = (
        _modal.Image.debian_slim(python_version="3.11")
        .apt_install(*_MODAL_APT)
        .pip_install(*_MODAL_DEPS)
        .env({"HF_HOME": "/cache/hf", "HF_HUB_CACHE": "/cache/hf/hub"})
        # ingest.py walks up looking for CMakeLists.txt + scripts/ to find
        # the repo root. Fake just enough of that here.
        .run_commands("mkdir -p /repo && touch /repo/CMakeLists.txt")
        # Only ship scripts/wer/ — shipping all of scripts/ would upload
        # the 11 GB of per-family scripts/envs/*/.venv directories and
        # silently stall the build.
        .add_local_dir(
            str(REPO_ROOT / "scripts" / "wer"),
            "/repo/scripts/wer",
            copy=True,
        )
    )

    _MODAL_VOLUMES = {
        "/cache/hf": _modal_hf_vol,
        "/repo/samples": _modal_samples_vol,
    }
    _MODAL_TIMEOUT = 60 * 60 * 3

    def _modal_remote_body(
        model: str,
        dataset: str,
        max_new_tokens: int,
        torch_threads: int,
        limit: int,
        prompt_template: str,
        batch_size: int,
        model_dtype: str,
        autocast: str,
        cuda_tf32: bool,
        cuda_sync_timing: bool,
        warmup_batches: int,
    ) -> bytes:
        """Shared body called by each per-GPU function below."""
        import os
        import subprocess
        import tempfile

        os.chdir("/repo")
        proto, arg = dataset.split(":", 1)
        if proto == "librispeech":
            manifest = f"/repo/samples/wer/librispeech-{arg}.manifest.jsonl"
            ingest_cmd = [
                "python", "scripts/wer/ingest.py",
                "librispeech", "--split", arg,
            ]
        elif proto == "fleurs":
            manifest = f"/repo/samples/wer/fleurs-{arg}.manifest.jsonl"
            ingest_cmd = [
                "python", "scripts/wer/ingest.py", "fleurs", "--lang", arg,
            ]
        else:
            raise ValueError(f"unsupported dataset proto {proto!r}")

        if not os.path.exists(manifest):
            print(f"[container] ingesting {dataset}", flush=True)
            subprocess.run(ingest_cmd, check=True)
            _modal_samples_vol.commit()
        else:
            print(f"[container] manifest cache hit: {manifest}", flush=True)

        out_path = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False
        ).name

        # Re-exec this same script in local mode inside the container.
        cmd = [
            "python", "scripts/wer/run_reference_canary_qwen_nemo.py",
            "--model", model,
            "--manifest", manifest,
            "--out", out_path,
            "--device", "cuda",
            "--max-new-tokens", str(max_new_tokens),
            "--torch-threads", str(torch_threads),
            "--prompt-template", prompt_template,
            "--batch-size", str(batch_size),
            "--model-dtype", model_dtype,
            "--autocast", autocast,
            "--warmup-batches", str(warmup_batches),
        ]
        if cuda_tf32:
            cmd += ["--cuda-tf32"]
        if cuda_sync_timing:
            cmd += ["--cuda-sync-timing"]
        if limit > 0:
            cmd += ["--limit", str(limit)]
        print(f"[container] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

        with open(out_path, "rb") as f:
            return f.read()

    # Modal's `@app.function` binds the GPU at decoration time and has no
    # runtime override (`with_options` doesn't exist for resources). So we
    # define one wrapper per GPU and dispatch by name in _submit_to_modal.
    # All share `_modal_image` so the heavy nemo install layer is built
    # exactly once.

    @_modal_app.function(image=_modal_image, gpu="T4",
                         volumes=_MODAL_VOLUMES, timeout=_MODAL_TIMEOUT)
    def _remote_T4(model, dataset, max_new_tokens, torch_threads, limit,
                   prompt_template, batch_size, model_dtype, autocast,
                   cuda_tf32, cuda_sync_timing, warmup_batches):
        return _modal_remote_body(model, dataset, max_new_tokens,
                                  torch_threads, limit, prompt_template,
                                  batch_size, model_dtype, autocast,
                                  cuda_tf32, cuda_sync_timing,
                                  warmup_batches)

    @_modal_app.function(image=_modal_image, gpu="L4",
                         volumes=_MODAL_VOLUMES, timeout=_MODAL_TIMEOUT)
    def _remote_L4(model, dataset, max_new_tokens, torch_threads, limit,
                   prompt_template, batch_size, model_dtype, autocast,
                   cuda_tf32, cuda_sync_timing, warmup_batches):
        return _modal_remote_body(model, dataset, max_new_tokens,
                                  torch_threads, limit, prompt_template,
                                  batch_size, model_dtype, autocast,
                                  cuda_tf32, cuda_sync_timing,
                                  warmup_batches)

    @_modal_app.function(image=_modal_image, gpu="A10G",
                         volumes=_MODAL_VOLUMES, timeout=_MODAL_TIMEOUT)
    def _remote_A10G(model, dataset, max_new_tokens, torch_threads, limit,
                     prompt_template, batch_size, model_dtype, autocast,
                     cuda_tf32, cuda_sync_timing, warmup_batches):
        return _modal_remote_body(model, dataset, max_new_tokens,
                                  torch_threads, limit, prompt_template,
                                  batch_size, model_dtype, autocast,
                                  cuda_tf32, cuda_sync_timing,
                                  warmup_batches)

    @_modal_app.function(image=_modal_image, gpu="L40S",
                         volumes=_MODAL_VOLUMES, timeout=_MODAL_TIMEOUT)
    def _remote_L40S(model, dataset, max_new_tokens, torch_threads, limit,
                     prompt_template, batch_size, model_dtype, autocast,
                     cuda_tf32, cuda_sync_timing, warmup_batches):
        return _modal_remote_body(model, dataset, max_new_tokens,
                                  torch_threads, limit, prompt_template,
                                  batch_size, model_dtype, autocast,
                                  cuda_tf32, cuda_sync_timing,
                                  warmup_batches)

    @_modal_app.function(image=_modal_image, gpu="A100-40GB",
                         volumes=_MODAL_VOLUMES, timeout=_MODAL_TIMEOUT)
    def _remote_A100_40GB(model, dataset, max_new_tokens, torch_threads,
                          limit, prompt_template, batch_size, model_dtype,
                          autocast, cuda_tf32, cuda_sync_timing,
                          warmup_batches):
        return _modal_remote_body(model, dataset, max_new_tokens,
                                  torch_threads, limit, prompt_template,
                                  batch_size, model_dtype, autocast,
                                  cuda_tf32, cuda_sync_timing,
                                  warmup_batches)

    @_modal_app.function(image=_modal_image, gpu="A100-80GB",
                         volumes=_MODAL_VOLUMES, timeout=_MODAL_TIMEOUT)
    def _remote_A100_80GB(model, dataset, max_new_tokens, torch_threads,
                          limit, prompt_template, batch_size, model_dtype,
                          autocast, cuda_tf32, cuda_sync_timing,
                          warmup_batches):
        return _modal_remote_body(model, dataset, max_new_tokens,
                                  torch_threads, limit, prompt_template,
                                  batch_size, model_dtype, autocast,
                                  cuda_tf32, cuda_sync_timing,
                                  warmup_batches)

    @_modal_app.function(image=_modal_image, gpu="H100",
                         volumes=_MODAL_VOLUMES, timeout=_MODAL_TIMEOUT)
    def _remote_H100(model, dataset, max_new_tokens, torch_threads, limit,
                     prompt_template, batch_size, model_dtype, autocast,
                     cuda_tf32, cuda_sync_timing, warmup_batches):
        return _modal_remote_body(model, dataset, max_new_tokens,
                                  torch_threads, limit, prompt_template,
                                  batch_size, model_dtype, autocast,
                                  cuda_tf32, cuda_sync_timing,
                                  warmup_batches)

    _MODAL_GPU_FNS = {
        "T4":         _remote_T4,
        "L4":         _remote_L4,
        "A10G":       _remote_A10G,
        "L40S":       _remote_L40S,
        "A100-40GB":  _remote_A100_40GB,
        "A100-80GB":  _remote_A100_80GB,
        "H100":       _remote_H100,
    }


def _submit_to_modal(args: argparse.Namespace) -> int:
    if not _HAS_MODAL:
        print(
            "error: --modal requires the `modal` library.\n"
            "  Install: `uv tool install modal` (or `pip install --user modal`)\n"
            "  Then re-run WITHOUT --project: \n"
            "    uv run --with modal scripts/wer/run_reference_canary_qwen_nemo.py --modal ...",
            file=sys.stderr,
        )
        return 2
    if args.manifest is not None:
        print("error: --manifest is not supported with --modal; pass --dataset "
              "(e.g. librispeech:test-clean) so the container can ingest the "
              "audio locally on a Modal Volume.", file=sys.stderr)
        return 2
    if args.dataset is None:
        print("error: --modal requires --dataset proto:arg "
              "(e.g. librispeech:test-clean, fleurs:es)", file=sys.stderr)
        return 2
    if ":" not in args.dataset:
        print(f"error: --dataset must be proto:arg, got {args.dataset!r}",
              file=sys.stderr)
        return 2

    if args.out is None:
        args.out = _modal_default_out(args.model, args.dataset, args.gpu)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fn = _MODAL_GPU_FNS.get(args.gpu)
    if fn is None:
        print(f"error: --gpu {args.gpu!r} not in {sorted(_MODAL_GPU_FNS)}",
              file=sys.stderr)
        return 2

    print(f"submitting modal job: gpu={args.gpu} dataset={args.dataset} "
          f"limit={args.limit or 'all'} batch_size={args.batch_size} "
          f"dtype={args.model_dtype} autocast={args.autocast}",
          flush=True)
    print(f"output: {args.out}", flush=True)
    print(f"connecting to modal (image build + container start happens here; "
          f"~2-15 min cold, ~30s warm). For live progress, in another shell run:",
          flush=True)
    print(f"  modal app logs transcribe-wer-canary-qwen", flush=True)
    t0 = time.monotonic()
    with _modal_app.run():
        print(f"[{time.monotonic()-t0:.1f}s] modal app started, calling "
              f"remote_run...", flush=True)
        jsonl_bytes = fn.remote(
            model=args.model,
            dataset=args.dataset,
            max_new_tokens=args.max_new_tokens,
            torch_threads=args.torch_threads,
            limit=args.limit,
            prompt_template=args.prompt_template,
            batch_size=args.batch_size,
            model_dtype=args.model_dtype,
            autocast=args.autocast,
            cuda_tf32=args.cuda_tf32,
            cuda_sync_timing=args.cuda_sync_timing,
            warmup_batches=args.warmup_batches,
        )
        print(f"[{time.monotonic()-t0:.1f}s] remote_run returned "
              f"({len(jsonl_bytes)} bytes)", flush=True)
    wall = time.monotonic() - t0
    print(f"modal job complete in {wall:.1f}s ({wall/60:.1f} min)", flush=True)

    args.out.write_bytes(jsonl_bytes)
    n_lines = jsonl_bytes.count(b"\n")
    print(f"wrote {args.out} ({len(jsonl_bytes)} bytes, {n_lines} lines)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
