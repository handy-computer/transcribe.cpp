"""
modal_sweep.py — dispatch transcribe.cpp WER runs to Modal GPUs.

Fans out one Modal container per (model, quant) cell against LibriSpeech or
FLEURS manifests, builds a CUDA `transcribe-cli` once per source state, and
returns the per-cell hyp JSONLs to the laptop for local scoring with
`scripts/wer/score.py`.

Scoring is intentionally NOT done on Modal. Only hyp generation runs remote.

Usage
-----

  # Sweep one or more models on LibriSpeech test-clean (default), L4 GPU:
  modal run scripts/wer/remote/modal_sweep.py::sweep \\
      --models moonshine-base,moonshine-tiny

  # Models without an hf_card use a HF repo path instead of a slug:
  modal run scripts/wer/remote/modal_sweep.py::sweep \\
      --models handy-computer/granite-4.0-1b-speech-gguf,handy-computer/granite-speech-4.1-2b-gguf

  # Quick check on one quant for one model (the smoke-test pattern):
  modal run scripts/wer/remote/modal_sweep.py::sweep \\
      --models moonshine-base --quants Q8_0 --n-utts 128

  # Restrict to a quant subset or switch dataset / GPU:
  modal run scripts/wer/remote/modal_sweep.py::sweep \\
      --models cohere-transcribe-03-2026 --dataset fleurs:es --quants Q8_0,Q4_K_M

  # Force a clean build (after a branch switch):
  modal run scripts/wer/remote/modal_sweep.py::sweep --models ... --clean

Outputs
-------

  reports/wer/<gguf-stem>.<dataset-id>.jsonl          hyp JSONL (run.py format)
  reports/wer/remote_sweep.<dataset-id>.summary.tsv   rtf / wall per cell

Score locally:

  for f in reports/wer/*.<dataset-id>.jsonl; do
      uv run scripts/wer/score.py "$f"
  done

Requirements
------------

  - Modal CLI installed, authenticated, with billing.
  - Modal Secret `huggingface-secret` (key HF_TOKEN) for HF private repos.
  - CUDA wiring in CMakeLists / source on the current branch.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import modal

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------

# REPO is only used on the local invoker: by add_local_dir at image build time
# and by write_hyp in the local entrypoint. Inside Modal containers the module
# is re-imported from /root/modal_sweep.py, where parents[3] does not exist.
_parents = pathlib.Path(__file__).resolve().parents
REPO = str(_parents[3]) if len(_parents) > 3 else "/src"

_REMOTE_DIR = pathlib.Path(__file__).resolve().parent
_REPO_REMOTE_DIR = pathlib.Path(REPO) / "scripts" / "wer" / "remote"
for _import_dir in (_REMOTE_DIR, _REPO_REMOTE_DIR):
    if _import_dir.is_dir() and str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))

from cache_paths import hyp_cache_paths, ref_hyp_cache_paths
from dataset_specs import (
    dataset_id,
    dataset_prepare_status,
    dataset_summary_fields,
    format_dataset_status,
    ingest_args_for,
    parse_dataset_spec,
)
from fingerprints import hyp_fingerprint, source_fingerprint
from gpu_config import GPU_TO_ARCH, build_dir
from model_specs import resolve_model
from output_paths import write_hyp, write_ref_hyp
from reference_specs import resolve_reference
from subprocess_io import run_subprocess_capturing_stderr
from workdir import prepare_work


# SRC_FP keys the build cache (C++ binary). HYP_FP keys the hyp cache and
# folds SRC_FP in so a binary change invalidates hyps too. Splitting them
# means dispatcher edits (modal_sweep.py) rotate NEITHER cache; only the
# files that actually affect the artifact rotate their respective cache.
SRC_FP = source_fingerprint(pathlib.Path(REPO))
HYP_FP = hyp_fingerprint(pathlib.Path(REPO), SRC_FP)


def _build_dir(gpu_id: str) -> str:
    return build_dir(SRC_FP, gpu_id)

# ---------------------------------------------------------------------------
# Modal image + volumes.
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential", "cmake", "ninja-build", "git", "ca-certificates",
        "zlib1g-dev", "libopenblas-dev", "curl", "tar", "rsync", "ccache",
    )
    .pip_install("huggingface_hub", "pyyaml")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
    )
    .add_local_dir(
        REPO,
        "/src",
        ignore=[
            "build/**", "build-*/**", "models/**", "tmp/**",
            "samples/wer/raw/**", "samples/wer/librispeech-*/**",
            "samples/wer/fleurs-*/**", "samples/wer/*.manifest.jsonl",
            "reports/**",
            "scripts/envs/**", "**/__pycache__/**", "**/.venv/**",
            "**/.uv/**", "**/.git/**", ".git/**",
            ".claude/**", "docs/**", "tests/golden/**",
            "**/*.gguf", "**/*.safetensors", "**/*.onnx", "**/*.onnx.data",
            "**/*.nemo", ".DS_Store",
        ],
    )
)

# Reference image: runs the upstream reference framework (NeMo / Transformers /
# author repo) per family, NOT transcribe-cli. Nothing compiles -> CUDA
# *runtime* base, not devel. Unlike `image` it KEEPS
# scripts/envs/<family>/{pyproject.toml,uv.lock} (needed for `uv sync`) while
# still dropping the heavy vendored local .venv (885 MB - 1.4 GB per family).
ref_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ca-certificates", "ffmpeg", "libsndfile1", "curl",
                 "build-essential")
    .pip_install("huggingface_hub")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
    )
    .add_local_dir(
        REPO, "/src",
        ignore=[
            "build/**", "build-*/**", "models/**", "tmp/**",
            "samples/wer/raw/**", "samples/wer/librispeech-*/**",
            "samples/wer/fleurs-*/**", "samples/wer/*.manifest.jsonl",
            "reports/**",
            "scripts/envs/**/.venv/**", "**/__pycache__/**", "**/.uv/**",
            "**/.git/**", ".git/**",
            ".claude/**", "docs/**", "tests/golden/**",
            "**/*.gguf", "**/*.safetensors", "**/*.onnx", "**/*.onnx.data",
            "**/*.nemo", ".DS_Store",
        ],
    )
)

build_vol = modal.Volume.from_name("transcribe-build", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("transcribe-data", create_if_missing=True)
# Compiler cache: survives build-dir rotation so ggml's TUs aren't recompiled
# on every source edit. Content-addressed -> the binary is identical to a
# clean build; only what actually changed runs through the compiler.
ccache_vol = modal.Volume.from_name("transcribe-ccache", create_if_missing=True)

app = modal.App("transcribe-wer-remote")


# ---------------------------------------------------------------------------
# 1. Build (CPU container; nvcc does not need a GPU).
# ---------------------------------------------------------------------------

@app.function(image=image, timeout=1800,
              volumes={"/build": build_vol, "/root/.cache/ccache": ccache_vol})
def build(arch: str, build_dir: str, clean: bool = False) -> None:
    """Build transcribe-cli with CUDA on a Modal Volume.

    arch       Single SM number (e.g. '89' for L4). The build is compiled for
               exactly this arch; switching --gpu triggers a one-time rebuild.
    build_dir  /build/cuda-<src_fp>-sm<arch>. Computed identically on the
               laptop and in the container from the source fingerprint + GPU.
    clean      Wipe build_dir before reconfiguring. Almost never needed since
               source changes already rotate the dir via the fingerprint.
    """
    cli_path = f"{build_dir}/bin/transcribe-cli"
    if clean and os.path.exists(build_dir):
        print(f"[build] clean=True; removing {build_dir}")
        shutil.rmtree(build_dir)

    # ccache env. The build dir (/build/cuda-<SRC_FP>-sm<arch>) rotates on every
    # source edit and shows up in compile commands (generated-header -I paths,
    # output dirs), which would bust the cache. BASEDIR rewrites absolute paths
    # under /build to cwd-relative and NOHASHDIR drops cwd from the hash, so an
    # unchanged ggml TU hashes the same across rotations -> cache hit. The cache
    # itself lives on the ccache Volume, shared across all builds.
    ccache_dir = "/root/.cache/ccache"
    # Cache storage lives on the Volume, but ccache's temp dir must be on local
    # disk: the Volume filesystem rejects the hardlink() ccache uses while
    # capturing preprocessor output ("Operation not permitted").
    ccache_tmp = "/tmp/ccache-tmp"
    os.makedirs(ccache_dir, exist_ok=True)
    os.makedirs(ccache_tmp, exist_ok=True)
    env = {
        **os.environ,
        "CCACHE_DIR": ccache_dir,
        "CCACHE_TEMPDIR": ccache_tmp,
        "CCACHE_BASEDIR": "/build",
        "CCACHE_NOHASHDIR": "1",
        "CCACHE_COMPILERCHECK": "content",
        "CCACHE_MAXSIZE": "10G",
    }
    subprocess.check_call(["ccache", "--zero-stats"], env=env)
    # cmake reads /src directly (read-only mount) and only writes to build_dir,
    # so this is the one function that does not need a /work mirror.
    subprocess.check_call([
        "cmake", "-S", "/src", "-B", build_dir, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DTRANSCRIBE_CUDA=ON",
        "-DTRANSCRIBE_BUILD_TESTS=OFF",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
    ], env=env)
    subprocess.check_call([
        "cmake", "--build", build_dir, "--target", "transcribe-cli",
        "-j", str(os.cpu_count()),
    ], env=env)
    subprocess.run(["ccache", "--show-stats", "--verbose"], env=env)
    ccache_vol.commit()
    build_vol.commit()
    assert os.path.exists(cli_path)
    print(f"[build] {cli_path} ready")


# ---------------------------------------------------------------------------
# 2. Dataset prefetch (idempotent; cached on /data volume).
# ---------------------------------------------------------------------------

@app.function(
    image=image, timeout=1800, gpu="L4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/build": build_vol,
        "/root/.cache/huggingface": hf_vol,
        "/data": data_vol,
    },
)
def debug_dump(model_repo: str, model_file: str) -> dict:
    """Run transcribe-cli once on jfk.wav with TRANSCRIBE_MEDASR_DEBUG_STATS
    set, capturing stderr (where the encoder writes per-block tensor stats).
    Returns the model_repo, model_file, and full stderr string so the local
    caller can grep for the first divergence point between F16 and Q4_K_M.
    """
    from huggingface_hub import hf_hub_download
    prepare_work()
    build_dir = _build_dir("L4")
    build.remote(arch=GPU_TO_ARCH["L4"], build_dir=build_dir, clean=False)
    # build runs in a different container that commits build_vol; our view
    # of /build is stale until we reload the volume.
    build_vol.reload()
    cli_path = f"{build_dir}/bin/transcribe-cli"
    assert os.path.exists(cli_path), f"missing CLI: {cli_path}"
    model_path = hf_hub_download(
        repo_id=model_repo,
        filename=model_file,
        token=os.environ["HF_TOKEN"],
    )
    audio = "/work/samples/jfk.wav"
    assert os.path.exists(audio), f"missing sample: {audio}"
    # TRANSCRIBE_DUMP_DIR pins intermediate tensors via ggml_set_output so
    # ggml_backend_tensor_get reads each one before the scheduler reuses
    # its buffer slot (otherwise post_* and early blocks alias to the
    # final-compute scratch state and the stats are wrong). The dump
    # files written into /tmp/dumps are not retrieved.
    os.makedirs("/tmp/dumps", exist_ok=True)
    env = {**os.environ,
           "TRANSCRIBE_MEDASR_DEBUG_STATS": "1",
           "TRANSCRIBE_DUMP_DIR": "/tmp/dumps"}
    r = subprocess.run(
        [cli_path, "-m", model_path, "--threads", "1", "-q", audio],
        cwd="/work", env=env, capture_output=True, text=True,
    )
    return {
        "model_repo": model_repo,
        "model_file": model_file,
        "rc": r.returncode,
        "stdout": r.stdout,
        "stderr": r.stderr,
    }


@app.local_entrypoint()
def dump_debug(model_repo: str, model_file: str) -> None:
    """Local entrypoint: `modal run scripts/wer/remote/modal_sweep.py::dump_debug
    --model-repo handy-computer/medasr-gguf --model-file medasr-Q4_K_M.gguf`.
    Prints stderr lines that contain STATS so we can diff F16 vs Q4_K_M."""
    result = debug_dump.remote(model_repo, model_file)
    print(f"=== {result['model_repo']} / {result['model_file']} (rc={result['rc']}) ===")
    for line in result["stderr"].splitlines():
        if "STATS " in line or "FAIL" in line.upper() or "ERR" in line.upper():
            print(line)
    print(f"--- transcript from stdout ---")
    for line in result["stdout"].splitlines():
        if line.startswith("text:"):
            print(line)


@app.function(
    image=image,
    timeout=3600,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def prefetch_dataset(spec: str) -> dict:
    """Ensure the manifest for `spec` is present on /data.

    Returns a structured dataset-preparation record so dispatchers and worker
    cells can report the same artifact state instead of inferring it from logs.
    """
    t0 = time.time()
    manifest = prepare_work(spec)
    if os.path.exists(manifest) and os.path.getsize(manifest) > 0:
        status = dataset_prepare_status(spec, manifest, "cached", time.time() - t0)
        print(format_dataset_status(status))
        return status

    try:
        kind, _ = parse_dataset_spec(spec)
        if kind == "librispeech":
            print(f"[data] {spec}: downloading + extracting LibriSpeech raw...")
            subprocess.check_call(["bash", "scripts/wer/setup.sh"], cwd="/work")
        print(f"[data] {spec}: ingest.py {' '.join(ingest_args_for(spec))}")
        subprocess.check_call(
            ["uv", "run", "scripts/wer/ingest.py", *ingest_args_for(spec)],
            cwd="/work",
        )
        data_vol.commit()
        status = dataset_prepare_status(spec, manifest, "created", time.time() - t0)
        print(format_dataset_status(status))
        return status
    except Exception as e:
        status = dataset_prepare_status(
            spec, manifest, "failed", time.time() - t0, error=repr(e)
        )
        print(format_dataset_status(status))
        raise


def _log_prepared_dataset(prefix: str, dataset_status: dict | None) -> None:
    if dataset_status:
        print(format_dataset_status(dataset_status, prefix=f"[{prefix}:data]"))


def _prepare_manifest_for_cell(
    prefix: str,
    dataset_spec: str,
    dataset_status: dict | None,
) -> str:
    manifest = prepare_work(dataset_spec)
    _log_prepared_dataset(prefix, dataset_status)
    if dataset_status and dataset_status.get("manifest") != manifest:
        print(
            f"[{prefix}:data] WARN prepared manifest path "
            f"{dataset_status.get('manifest')} != cell path {manifest}"
        )
    assert os.path.exists(manifest), (
        f"manifest missing at {manifest}; call prefetch_dataset first"
    )
    return manifest


def _subset_manifest_for_cell(
    prefix: str,
    manifest: str,
    n_utts: int | None,
    dataset_status: dict | None,
) -> tuple[str, int]:
    total = dataset_status.get("utterances") if dataset_status else None
    if n_utts is None:
        subset_path = manifest
        subset_count = sum(1 for _ in open(subset_path))
        scope = "full manifest"
    else:
        subset_path = "/tmp/subset.manifest.jsonl"
        with open(manifest) as fin, open(subset_path, "w") as fout:
            for i, line in enumerate(fin):
                if i >= n_utts:
                    break
                fout.write(line)
        subset_count = sum(1 for _ in open(subset_path))
        total_display = total if total is not None else "?"
        scope = f"first {subset_count}/{total_display} requested={n_utts}"
    print(f"[{prefix}:data] subset={scope} path={subset_path}")
    if n_utts is not None and subset_count < n_utts:
        print(
            f"[{prefix}:data] WARN requested {n_utts} utterances but only "
            f"{subset_count} were available"
        )
    return subset_path, subset_count


# ---------------------------------------------------------------------------
# 3. List .gguf files in a HF repo (used to expand a sweep matrix).
# ---------------------------------------------------------------------------

@app.function(
    image=image, timeout=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def list_ggufs(repos: list[str]) -> list[tuple[str, list[str]]]:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    out: list[tuple[str, list[str]]] = []
    for repo in repos:
        try:
            files = api.list_repo_files(repo)
        except Exception as e:
            print(f"  WARN: failed to list {repo}: {e}")
            out.append((repo, []))
            continue
        ggufs = sorted(f for f in files if f.endswith(".gguf"))
        out.append((repo, ggufs))
        print(f"  {repo}: {len(ggufs)} quants -> {ggufs}")
    return out


# ---------------------------------------------------------------------------
# 4. Per-cell WER run. Modal 1.4.x removed Function.with_options, so we
# pre-register one wrapper per supported GPU class and look up by --gpu at
# sweep time (see _GPU_FNS below). Stdout is NOT captured -> live progress
# streams to Modal logs.
# ---------------------------------------------------------------------------

def _run_wer_impl(
    model_repo: str,
    model_file: str,
    dataset_spec: str,
    n_utts: int | None,
    build_dir: str,
    batch_size: int = 1,
    sort_by_length: bool = True,
    timestamps: str = "none",
    stream_chunk_ms: int = 0,
    dataset_status: dict | None = None,
) -> dict:
    """Run scripts/wer/run.py on `n_utts` (or full manifest) and return the
    hyp JSONL contents + a summary dict back to the dispatcher.

    Volume-cached by (SRC_FP, model_file, dataset, subset). On a cache hit the
    container returns immediately without downloading the model or running
    the CLI; this is what makes interrupted sweeps resumable.

    stream_chunk_ms > 0 drives each utterance through the streaming API in
    N-ms chunks (run.py --stream-chunk-ms); the model's default transcription
    delay applies (voxtral_realtime: 6 tokens = 480 ms).
    """
    from huggingface_hub import hf_hub_download

    # Cache lookup first; skips prepare_work + model download + run.py entirely
    # when a hyp for this (fingerprint, model, dataset, subset) already exists.
    cache_hyp, cache_sum = hyp_cache_paths(
        HYP_FP, model_file, dataset_spec, n_utts, batch_size, sort_by_length,
        timestamps, stream_chunk_ms)
    if os.path.exists(cache_hyp) and os.path.exists(cache_sum) \
       and os.path.getsize(cache_hyp) > 0:
        _log_prepared_dataset("wer", dataset_status)
        print(f"[wer] cache hit: {cache_hyp}")
        with open(cache_sum) as f:
            summary = json.load(f)
        summary.update(dataset_summary_fields(dataset_status))
        with open(cache_hyp) as f:
            hyp_jsonl = f.read()
        return {"hyp_jsonl": hyp_jsonl, "summary": summary, "cached": True}

    manifest = _prepare_manifest_for_cell("wer", dataset_spec, dataset_status)

    cli_path = f"{build_dir}/bin/transcribe-cli"
    assert os.path.exists(cli_path), (
        f"missing CLI: {cli_path}; call build() first for this GPU's arch"
    )

    print(f"[wer] downloading {model_repo}/{model_file}")
    t0 = time.time()
    model_path = hf_hub_download(
        repo_id=model_repo,
        filename=model_file,
        token=os.environ["HF_TOKEN"],
    )
    hf_vol.commit()
    size_gb = os.path.getsize(model_path) / 1e9
    print(f"[wer] model {size_gb:.2f} GB ready in {time.time()-t0:.1f}s")

    subset_path, subset_count = _subset_manifest_for_cell(
        "wer", manifest, n_utts, dataset_status
    )

    hyp_path = "/tmp/hyps.jsonl"
    if os.path.exists(hyp_path):
        os.unlink(hyp_path)

    # Force per-line stdout flushing so progress streams live to Modal logs.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    cmd = [
        "uv", "run", "scripts/wer/run.py",
        "--cli", cli_path,
        "--model", model_path,
        "--manifest", subset_path,
        "--out", hyp_path,
    ]
    if batch_size and batch_size > 1:
        cmd += ["--batch-size", str(batch_size)]
        if sort_by_length:
            cmd += ["--sort-by-length"]
    if timestamps and timestamps != "none":
        cmd += ["--timestamps", timestamps]
    if stream_chunk_ms and stream_chunk_ms > 0:
        cmd += ["--stream-chunk-ms", str(stream_chunk_ms)]
    print(f"[wer] $ {' '.join(cmd)}")
    t0 = time.time()
    rc, stderr_tail = run_subprocess_capturing_stderr(cmd, cwd="/work", env=env)
    wall_s = time.time() - t0
    if rc != 0:
        raise RuntimeError(
            f"run.py exited {rc}; stderr tail:\n{stderr_tail.rstrip()}"
        )

    # Audio duration from per-utt wav headers.
    import wave
    audio_s = 0.0
    with open(subset_path) as f:
        for line in f:
            ent = json.loads(line)
            with wave.open(ent["audio"], "rb") as w:
                audio_s += w.getnframes() / w.getframerate()

    hyp_jsonl = open(hyp_path).read()
    # Stage totals (sum of per-utt ms) so the dispatcher can show where the
    # wall goes: amortized GPU encode vs host decode.
    mel_ms = enc_ms = dec_ms = 0.0
    for line in hyp_jsonl.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "batch_header":
            continue
        mel_ms += d.get("mel_ms", 0) or 0
        enc_ms += d.get("encode_ms", 0) or 0
        dec_ms += d.get("decode_ms", 0) or 0
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip()
    summary = {
        "model_repo": model_repo,
        "model_file": model_file,
        "dataset": dataset_spec,
        "n_utts": subset_count,
        "audio_s": audio_s,
        "wall_s": wall_s,
        "rtf_wall": audio_s / wall_s if wall_s > 0 else 0.0,
        "utt_per_s": subset_count / wall_s if wall_s > 0 else 0.0,
        "batch_size": batch_size,
        "stream_chunk_ms": stream_chunk_ms,
        "mel_s": mel_ms / 1000.0,
        "encode_s": enc_ms / 1000.0,
        "decode_s": dec_ms / 1000.0,
        "gpu": gpu,
    }
    summary.update(dataset_summary_fields(dataset_status))
    print(f"[wer] done: {summary}")

    # Persist to Volume for future resume / re-invocation.
    os.makedirs(os.path.dirname(cache_hyp), exist_ok=True)
    shutil.copyfile(hyp_path, cache_hyp)
    with open(cache_sum, "w") as f:
        json.dump(summary, f)
    data_vol.commit()
    return {"hyp_jsonl": hyp_jsonl, "summary": summary, "cached": False}


_RUN_WER_KWARGS = dict(
    image=image, timeout=43200,
    volumes={
        "/build": build_vol,
        "/root/.cache/huggingface": hf_vol,
        "/data": data_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


def _register_runner(gpu_id: str):
    """Wrap _run_wer_impl in a uniquely-named module-level function and
    decorate it with a GPU-specific @app.function. The closure captures this
    GPU's build_dir so the runner always reads the right transcribe-cli.

    Modal needs both a distinct Python name per registration AND a module-level
    attribute binding so the container can re-import the function by name."""
    name = f"run_wer_{gpu_id.lower().replace('-', '_')}"
    default_build_dir = _build_dir(gpu_id)

    def runner(
        model_repo: str,
        model_file: str,
        dataset_spec: str = "librispeech:test-clean",
        n_utts: int | None = None,
        batch_size: int = 1,
        sort_by_length: bool = True,
        build_dir: str | None = None,
        timestamps: str = "none",
        stream_chunk_ms: int = 0,
        dataset_status: dict | None = None,
    ) -> dict:
        # Prefer the build_dir the local entrypoint computed and built into:
        # SRC_FP can drift between the laptop and the container, so recomputing
        # it here (default_build_dir) risks pointing at a dir build() never
        # wrote. The local value is the single source of truth.
        return _run_wer_impl(
            model_repo, model_file, dataset_spec, n_utts,
            build_dir or default_build_dir,
            batch_size=batch_size, sort_by_length=sort_by_length,
            timestamps=timestamps, stream_chunk_ms=stream_chunk_ms,
            dataset_status=dataset_status,
        )

    runner.__name__ = name
    runner.__qualname__ = name
    decorated = app.function(gpu=gpu_id, **_RUN_WER_KWARGS)(runner)
    globals()[name] = decorated  # required for container-side import to find it
    return decorated


# Single source of truth for "which GPUs we register and what arch they map to":
# iterate GPU_TO_ARCH.
_GPU_FNS = {gpu: _register_runner(gpu) for gpu in GPU_TO_ARCH}


# ===========================================================================
# Reference sweep: run the upstream reference framework (NeMo / Transformers /
# author repo) per variant on a GPU, producing the `<variant>-REF.<dataset>`
# baseline that porting-2-oracle Step 7 consumes. Mirrors the sweep above, but
# the "build" is a per-family `uv sync` (ENV_FP) instead of a C++ compile
# (SRC_FP), and the model download is delegated to the runner's own
# `from_pretrained`.
# ===========================================================================

def _ensure_local_venv(family: str) -> str:
    """`uv sync` the family env onto LOCAL container disk and return the venv
    path. The venv/cache CANNOT live on a Modal Volume: uv takes advisory file
    locks the Volume FUSE filesystem rejects ('Could not acquire lock:
    Operation not permitted') — the same class of issue build() works around
    for ccache. Synced once per container; both batch sizes reuse it. The cost
    is one `uv sync` per (variant, container); a cross-run wheel cache is a
    follow-up optimization (e.g. a tarball'd venv keyed by ENV_FP)."""
    venv = f"/tmp/venv-{family}"
    if os.path.exists(os.path.join(venv, "bin", "python")):
        return venv
    env = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": venv,    # LOCAL disk, not a Volume
        "UV_CACHE_DIR": "/tmp/uv-cache",   # LOCAL disk
        "UV_LINK_MODE": "copy",
    }
    print(f"[ref] uv sync --frozen --project scripts/envs/{family} -> {venv}")
    subprocess.check_call(
        ["uv", "sync", "--frozen", "--project", f"scripts/envs/{family}"],
        cwd="/work", env=env,
    )
    assert os.path.exists(os.path.join(venv, "bin", "python"))
    return venv


def _run_one_reference(family, variant, framework, upstream_repo, runner_rel,
                       subset_path, subset_count, venv, dataset_spec, n_utts,
                       env_fp, batch_size, device, mode="offline",
                       dataset_status=None) -> dict:
    """One reference run at a single batch size. Hyp-cached on /data (plain
    file writes, no flock, so the Volume is fine here — only uv needs local
    disk)."""
    cache_hyp, cache_sum = ref_hyp_cache_paths(
        variant, dataset_spec, n_utts, batch_size, env_fp, mode)
    if os.path.exists(cache_hyp) and os.path.exists(cache_sum) \
       and os.path.getsize(cache_hyp) > 0:
        print(f"[ref] cache hit: {cache_hyp}")
        with open(cache_sum) as f:
            summary = json.load(f)
        summary.update(dataset_summary_fields(dataset_status))
        return {"hyp_jsonl": open(cache_hyp).read(), "summary": summary,
                "cached": True}

    out_path = f"/tmp/ref-hyps.b{batch_size}.jsonl"
    if os.path.exists(out_path):
        os.unlink(out_path)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "UV_PROJECT_ENVIRONMENT": venv,
        "UV_CACHE_DIR": "/tmp/uv-cache",
        "HF_HOME": "/root/.cache/huggingface",
    }
    # Uniform reference CLI contract: --manifest --model --out --device --batch-size
    cmd = [
        "uv", "run", "--project", f"scripts/envs/{family}", "--no-sync",
        runner_rel,
        "--manifest", subset_path,
        "--model", upstream_repo,
        "--out", out_path,
        "--device", device,
        "--batch-size", str(batch_size),
    ]
    # The uniform reference contract is --manifest/--model/--out/--device/
    # --batch-size; only pass --mode for non-default modes so runners that
    # predate the flag (every family but voxtral_realtime) are unaffected.
    if mode and mode != "offline":
        cmd += ["--mode", mode]
    print(f"[ref] bs={batch_size} mode={mode} $ {' '.join(cmd)}")
    t0 = time.time()
    rc, stderr_tail = run_subprocess_capturing_stderr(cmd, cwd="/work", env=env)
    wall_s = time.time() - t0
    hf_vol.commit()
    if rc != 0:
        raise RuntimeError(
            f"reference runner (bs={batch_size}) exited {rc}; "
            f"stderr tail:\n{stderr_tail.rstrip()}")

    hyp_jsonl = open(out_path).read()
    import wave
    audio_s = 0.0
    with open(subset_path) as f:
        for line in f:
            ent = json.loads(line)
            try:
                with wave.open(ent["audio"], "rb") as w:
                    audio_s += w.getnframes() / w.getframerate()
            except Exception:
                pass
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip()
    summary = {
        "variant": variant, "family": family, "framework": framework,
        "model": upstream_repo, "dataset": dataset_spec,
        "n_utts": subset_count, "audio_s": audio_s, "wall_s": wall_s,
        "rtf_wall": audio_s / wall_s if wall_s > 0 else 0.0,
        "utt_per_s": subset_count / wall_s if wall_s > 0 else 0.0,
        "batch_size": batch_size, "mode": mode, "device": device, "gpu": gpu,
    }
    summary.update(dataset_summary_fields(dataset_status))
    print(f"[ref] done bs={batch_size}: {summary}")
    os.makedirs(os.path.dirname(cache_hyp), exist_ok=True)
    shutil.copyfile(out_path, cache_hyp)
    with open(cache_sum, "w") as f:
        json.dump(summary, f)
    data_vol.commit()
    return {"hyp_jsonl": hyp_jsonl, "summary": summary, "cached": False}


def _run_reference_impl(
    family: str,
    variant: str,
    framework: str,
    upstream_repo: str,
    runner_rel: str,
    dataset_spec: str,
    n_utts: int | None,
    env_fp: str,
    batch_sizes: list,
    device: str = "cuda",
    mode: str = "offline",
    dataset_status: dict | None = None,
) -> dict:
    """One container per variant: `uv sync` the env once (local disk), then run
    the reference runner at each requested batch size. Returns
    {str(batch_size): {hyp_jsonl, summary, cached}}."""
    manifest = _prepare_manifest_for_cell("ref", dataset_spec, dataset_status)
    venv = _ensure_local_venv(family)

    subset_path, subset_count = _subset_manifest_for_cell(
        "ref", manifest, n_utts, dataset_status
    )
    print(f"[ref] {family}/{variant} fw={framework} batches={batch_sizes}: "
          f"{subset_count} utts")

    results = {}
    for bs in batch_sizes:
        results[str(bs)] = _run_one_reference(
            family, variant, framework, upstream_repo, runner_rel,
            subset_path, subset_count, venv, dataset_spec, n_utts, env_fp,
            bs, device, mode, dataset_status)
    return results


_REF_RUN_KWARGS = dict(
    image=ref_image, timeout=43200,
    volumes={
        "/root/.cache/huggingface": hf_vol,
        "/data": data_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


def _register_reference_runner(gpu_id: str):
    """Per-GPU wrapper for _run_reference_impl, same trick as _register_runner
    (Modal 1.4.x needs a distinct module-level function per GPU class)."""
    name = f"run_ref_{gpu_id.lower().replace('-', '_')}"

    def runner(
        family: str,
        variant: str,
        framework: str,
        upstream_repo: str,
        runner_rel: str,
        env_fp: str,
        dataset_spec: str = "librispeech:test-clean",
        n_utts: int | None = None,
        batch_sizes: list | None = None,
        device: str = "cuda",
        mode: str = "offline",
        dataset_status: dict | None = None,
    ) -> dict:
        return _run_reference_impl(
            family, variant, framework, upstream_repo, runner_rel,
            dataset_spec, n_utts, env_fp, batch_sizes or [1], device=device,
            mode=mode, dataset_status=dataset_status)

    runner.__name__ = name
    runner.__qualname__ = name
    decorated = app.function(gpu=gpu_id, **_REF_RUN_KWARGS)(runner)
    globals()[name] = decorated
    return decorated


_REF_GPU_FNS = {gpu: _register_reference_runner(gpu) for gpu in GPU_TO_ARCH}


# ---------------------------------------------------------------------------
# Local entrypoints.
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def sweep(
    models: str,
    dataset: str = "librispeech:test-clean",
    quants: str = "",
    gpu: str = "L4",
    n_utts: int = -1,
    clean: bool = False,
    batch_sizes: str = "1",
    sort_by_length: bool = True,
    timestamps: str = "none",
    stream_chunk_ms: int = 0,
) -> None:
    """Fan WER across one or more models on one GPU class.

    --models      Comma-separated. Each entry is either an hf_card slug
                  (e.g. "moonshine-base" → scripts/hf_cards/moonshine-base.yaml)
                  or a HF repo path (e.g. "handy-computer/foo-gguf").
                  Slugs pin the quant set; repo paths discover via HF API.
    --dataset     "librispeech:test-clean" (default), "librispeech:<split>",
                  or "fleurs:<bcp47>".
    --quants      Optional substring filter, e.g. "Q8_0,F16".
                  Default: all quants in each model.
    --gpu         Modal CUDA SKU. Default L4.
                  Supported: T4, L4, A10G, L40S, A100, A100-80GB, H100, H200.
    --n-utts      Cap each manifest to N utterances. Default -1 = full.
    --clean       Force a clean build (use after a branch switch).
    """
    specs = [s.strip() for s in models.split(",") if s.strip()]
    if not specs:
        raise SystemExit("--models is required (comma-separated)")
    quant_filter = [q.strip() for q in quants.split(",") if q.strip()] if quants else None

    print("==================== sweep config ====================")
    print(f"  models   = {specs}")
    print(f"  dataset  = {dataset}")
    print(f"  quants   = {quant_filter if quant_filter else 'all'}")
    print(f"  gpu      = {gpu}")
    print(f"  n_utts   = {'full manifest' if n_utts < 0 else n_utts}")
    print(f"  clean    = {clean}")
    print("======================================================")

    repo_root = pathlib.Path(REPO)
    resolved = [(s, *resolve_model(repo_root, s)) for s in specs]

    needs_listing = sorted({repo for _, repo, fns in resolved if fns is None})
    listings = dict(list_ggufs.remote(needs_listing)) if needs_listing else {}

    sizes = [int(x) for x in batch_sizes.split(",") if x.strip()] or [1]

    cells: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for spec, repo, fns in resolved:
        if fns is None:
            fns = listings.get(repo, [])
            if not fns:
                skipped.append((spec, f"no .gguf files in {repo}"))
                continue
        if quant_filter:
            fns = [f for f in fns if any(q in f for q in quant_filter)]
            if not fns:
                skipped.append((spec, f"no quants matched filter {quant_filter}"))
                continue
        for f in fns:
            for bs in sizes:
                cells.append({"repo": repo, "file": f, "dataset": dataset, "bs": bs})

    if skipped:
        print(">>> skipped (no cells generated):")
        for s, r in skipped:
            print(f"  {s}: {r}")
    print(f">>> {len(cells)} cells to run")
    if not cells:
        raise SystemExit("nothing to do")

    runner = _GPU_FNS.get(gpu)
    if runner is None:
        raise SystemExit(
            f"--gpu {gpu!r} not registered; choose one of: {sorted(_GPU_FNS)}"
        )

    arch = GPU_TO_ARCH[gpu]
    build_dir = _build_dir(gpu)
    print(f">>> build (arch sm_{arch} -> {build_dir})")
    build.remote(arch=arch, build_dir=build_dir, clean=clean)
    print(f">>> prefetch dataset ({dataset})")
    dataset_status = prefetch_dataset.remote(dataset)
    print(f">>> dataset ready: {dataset_status['dataset_id']} "
          f"n={dataset_status['utterances']} "
          f"sha={dataset_status['manifest_sha256'][:12]}")

    print(f">>> launching {len(cells)} {gpu} containers in parallel...")
    n = None if n_utts < 0 else n_utts
    futs = [(c, runner.spawn(c["repo"], c["file"], c["dataset"], n,
                             c["bs"], sort_by_length, build_dir, timestamps,
                             stream_chunk_ms, dataset_status))
            for c in cells]

    rows, failures = [], []
    for c, fut in futs:
        bs = c["bs"]
        slug = c["file"].replace(".gguf", "") + (f" b{bs}" if bs != 1 else "")
        try:
            res = fut.get()
            # b1 stays untagged (matches the published-run filenames); b>1 is
            # tagged .b{bs} so batched hyps score independently for comparison.
            p = write_hyp(repo_root, res["hyp_jsonl"], c["file"], c["dataset"],
                          batch_size=(bs if bs != 1 else None),
                          timestamps=timestamps,
                          stream_chunk_ms=stream_chunk_ms)
            s = res["summary"]
            rows.append((slug, c["dataset"], s["n_utts"], s["audio_s"],
                         s["wall_s"], s["rtf_wall"], str(p)))
            tag = "CACHED" if res.get("cached") else "OK"
            print(f"  [{tag}] {slug}: {s['wall_s']:.1f}s, RTF {s['rtf_wall']:.1f}x -> {p}")
        except Exception as e:
            failures.append((slug, repr(e)))
            print(f"  [FAIL] {slug}: {e} (check Modal dashboard for stderr)")

    if rows:
        ds_id = dataset_id(dataset)
        summary_path = pathlib.Path(REPO) / "reports" / "wer" / \
                       f"remote_sweep.{ds_id}.summary.tsv"
        with open(summary_path, "w") as f:
            f.write("slug\tdataset\tn_utts\taudio_s\twall_s\trtf\tpath\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        print(f"\nsummary: {summary_path}")

    print("\n========== sweep summary ==========")
    print(f"{'slug':<48} {'dataset':<24} {'n':>5} {'audio':>9} {'wall':>8} {'rtf':>6}")
    for slug, ds, n_, audio, wall, rtf, _ in rows:
        print(f"{slug:<48} {ds:<24} {n_:>5} {audio:>9.1f} {wall:>8.1f} {rtf:>6.1f}")
    if failures:
        print("\nfailures:")
        for slug, err in failures:
            print(f"  {slug}: {err}")
    if skipped:
        print(f"\nskipped: {len(skipped)} entries (listed at config time above)")
    print(f"\nscore locally:  for f in reports/wer/*.{dataset_id(dataset)}.jsonl; "
          f"do uv run scripts/wer/score.py \"$f\"; done")


@app.local_entrypoint()
def batch_sweep(
    model: str,
    quant: str = "F16",
    dataset: str = "librispeech:test-clean",
    gpu: str = "L40S",
    n_utts: int = 128,
    batch_sizes: str = "1,2,4,8,16",
    sort_by_length: bool = True,
    clean: bool = False,
) -> None:
    """Throughput sweep of transcribe_run_batch across batch sizes for ONE
    model+quant on one GPU.

    The batched encoder is numerically validated (hyp text is identical
    across batch sizes), so this measures only wall / RTF / utt-per-s per
    batch size — WER is unchanged. Each run length-sorts the manifest so a
    batch groups similar-length clips (the encoder pads to the group max).

    --model        hf_card slug or HF repo path (ONE model).
    --quant        substring picking ONE quant filename (default F16).
    --dataset      librispeech:<split> or fleurs:<bcp47>.
    --gpu          Modal CUDA SKU (default L40S).
    --n-utts        utterances per run (default 128).
    --batch-sizes   comma list (default 1,2,4,8,16).
    --sort-by-length  group similar-length clips per batch (default True).
                    Pass --no-sort-by-length to measure the naive case where
                    the manifest order is arbitrary and each batch pads to its
                    longest clip (shows the cost of NOT bucketing).
    --clean         force a clean build.

      modal run scripts/wer/remote/modal_sweep.py::batch_sweep \\
          --model parakeet-tdt-0.6b-v2 --gpu L40S --n-utts 128 \\
          --batch-sizes 1,2,4,8,16
    """
    repo_root = pathlib.Path(REPO)
    repo, fns = resolve_model(repo_root, model)
    if fns is None:
        fns = dict(list_ggufs.remote([repo])).get(repo, [])
    fns = [f for f in fns if quant in f]
    if not fns:
        raise SystemExit(f"no quant matching {quant!r} for {model} in {repo}")
    model_file = sorted(fns, key=len)[0]  # shortest matching filename
    sizes = [int(s) for s in batch_sizes.split(",") if s.strip()]

    runner = _GPU_FNS.get(gpu)
    if runner is None:
        raise SystemExit(
            f"--gpu {gpu!r} not registered; choose: {sorted(_GPU_FNS)}")
    arch = GPU_TO_ARCH[gpu]
    build_dir = _build_dir(gpu)

    print("==================== batch_sweep config ====================")
    print(f"  model    = {model_file}  (repo {repo})")
    print(f"  dataset  = {dataset}")
    print(f"  gpu      = {gpu}")
    print(f"  n_utts   = {n_utts}")
    print(f"  batches  = {sizes}")
    print(f"  sort     = {'length-sorted' if sort_by_length else 'NAIVE (unsorted)'}")
    print("============================================================")

    print(f">>> build (arch sm_{arch} -> {build_dir})")
    build.remote(arch=arch, build_dir=build_dir, clean=clean)
    print(f">>> prefetch dataset ({dataset})")
    dataset_status = prefetch_dataset.remote(dataset)
    print(f">>> dataset ready: {dataset_status['dataset_id']} "
          f"n={dataset_status['utterances']} "
          f"sha={dataset_status['manifest_sha256'][:12]}")

    n = None if n_utts < 0 else n_utts
    # Launch all batch sizes in parallel (each its own container). Pass the
    # locally-computed build_dir so the runner reads exactly what build() wrote.
    futs = [(bs, runner.spawn(repo, model_file, dataset, n, bs, sort_by_length,
                              build_dir, "none", 0, dataset_status))
            for bs in sizes]

    rows: list[tuple] = []
    hyp_paths: list[tuple[int, pathlib.Path]] = []
    for bs, fut in futs:
        try:
            res = fut.get()
            s = res["summary"]
            rows.append((bs, s["n_utts"], s["audio_s"], s["wall_s"],
                         s["rtf_wall"], s.get("utt_per_s", 0.0),
                         s.get("encode_s", 0.0), s.get("decode_s", 0.0)))
            tag = "CACHED" if res.get("cached") else "OK"
            print(f"  [{tag}] b{bs}: wall {s['wall_s']:.1f}s  "
                  f"RTF {s['rtf_wall']:.1f}x  {s.get('utt_per_s', 0.0):.1f} utt/s  "
                  f"(encode {s.get('encode_s', 0.0):.1f}s / "
                  f"decode {s.get('decode_s', 0.0):.1f}s)")
            # Write a hyp per batch size so WER can be scored and compared
            # independently (verifies the "identical across batch sizes" claim).
            p = write_hyp(repo_root, res["hyp_jsonl"], model_file, dataset, batch_size=bs)
            hyp_paths.append((bs, p))
        except Exception as e:
            print(f"  [FAIL] b{bs}: {e} (check Modal dashboard for stderr)")

    if not rows:
        raise SystemExit("no successful runs")

    rows.sort()
    base_wall = next((w for b, _, _, w, _, _, _, _ in rows if b == 1), None)
    print("\n========== batch throughput sweep ==========")
    print(f"{'n_batch':>7} {'n_utts':>6} {'audio_s':>9} {'wall_s':>8} "
          f"{'rtf':>7} {'utt/s':>7} {'enc_s':>7} {'dec_s':>7} {'speedup':>8}")
    for b, nn, aud, wall, rtf, ups, enc, dec in rows:
        sp = (base_wall / wall) if (base_wall and wall > 0) else 0.0
        print(f"{b:>7} {nn:>6} {aud:>9.1f} {wall:>8.1f} {rtf:>7.1f} "
              f"{ups:>7.1f} {enc:>7.1f} {dec:>7.1f} {sp:>7.2f}x")

    sort_suffix = "" if sort_by_length else "_nosort"
    out = (pathlib.Path(REPO) / "reports" / "perf" / gpu.lower() /
           f"{model_file.replace('.gguf', '')}_batch_wer_{gpu.lower()}{sort_suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        [{"n_batch": b, "n_utts": nn, "audio_s": aud, "wall_s": wall,
          "rtf_wall": rtf, "utt_per_s": ups, "encode_s": enc, "decode_s": dec,
          "gpu": gpu}
         for b, nn, aud, wall, rtf, ups, enc, dec in rows], indent=2) + "\n")
    print(f"\nwrote {out}")
    print("score each batch's hyp to confirm WER is unchanged:")
    for bs, p in sorted(hyp_paths):
        print(f"  uv run scripts/wer/score.py {p}")


@app.local_entrypoint()
def run(
    model: str,
    quant: str,
    dataset: str = "librispeech:test-clean",
    gpu: str = "L4",
    n_utts: int = -1,
    clean: bool = False,
) -> None:
    """Run a single (model, quant) cell. Thin wrapper over sweep for the
    common 'one model, one quant' case.

    --model       hf_card slug (e.g. 'parakeet-tdt-0.6b-v3') or HF repo path.
    --quant       Substring against the gguf filename, e.g. 'Q8_0', 'F16'.
    --dataset, --gpu, --n-utts, --clean: same as sweep.

    Equivalent to:
      modal run ...::sweep --models <model> --quants <quant> ...
    """
    sweep(
        models=model,
        dataset=dataset,
        quants=quant,
        gpu=gpu,
        n_utts=n_utts,
        clean=clean,
    )


@app.local_entrypoint()
def reference_sweep(
    variants: str,
    dataset: str = "librispeech:test-clean",
    gpu: str = "L4",
    n_utts: int = -1,
    batch_sizes: str = "1",
    clean: bool = False,
    runner: str = "",
    mode: str = "offline",
    model_override: str = "",
) -> None:
    """Fan the REFERENCE framework (NeMo / Transformers / author repo) across
    one or more variants on a GPU, producing `<variant>-REF.<dataset>.jsonl`
    baselines (porting-2-oracle Step 7). Scoring stays local (score.py).

    --variants     Comma-separated variant specs. Prefer '<family>:<variant>'
                   (e.g. 'parakeet:parakeet-tdt-0.6b-v2'); a bare '<variant>'
                   is resolved by scanning reports/porting/*/<variant>/.
    --dataset      'librispeech:test-clean' (default) / 'librispeech:<split>' /
                   'fleurs:<bcp47>'.
    --gpu          Modal CUDA SKU (default L4). T4,L4,A10G,L40S,A100,H100,H200.
    --n-utts       Cap each manifest to N utts. Default -1 = full.
    --batch-sizes  Comma list of reference --batch-size values, e.g. '1,4'.
    --clean        Rebuild the family venv (wipe /uvcache/<env_fp>).
    --runner       Explicit runner path override (needed when a family has
                   several run_reference_<family>_*.py, e.g. parakeet).

    Examples:
      # Smoke one Transformers + one NeMo variant at bs 1 and 4 (4 cells):
      modal run scripts/wer/remote/modal_sweep.py::reference_sweep \\
          --variants granite:granite-4.0-1b-speech,parakeet:parakeet-tdt-0.6b-v2 \\
          --n-utts 64 --batch-sizes 1,4 --gpu L4
    """
    specs = [s.strip() for s in variants.split(",") if s.strip()]
    if not specs:
        raise SystemExit("--variants is required (comma-separated)")
    sizes = [int(x) for x in batch_sizes.split(",") if x.strip()] or [1]

    repo_root = pathlib.Path(REPO)
    resolved = [resolve_reference(repo_root, s, runner) for s in specs]
    # Optional: redirect from the intake's upstream repo (e.g. a gated
    # mistralai/... repo) to a private mirror the Modal HF token can reach.
    if model_override:
        for r in resolved:
            r["upstream_repo"] = model_override

    print("================ reference_sweep config ================")
    print(f"  variants = {[r['variant'] for r in resolved]}")
    print(f"  dataset  = {dataset}")
    print(f"  gpu      = {gpu}")
    print(f"  n_utts   = {'full manifest' if n_utts < 0 else n_utts}")
    print(f"  batches  = {sizes}")
    print(f"  mode     = {mode}")
    if model_override:
        print(f"  model_override = {model_override}")
    for r in resolved:
        print(f"  - {r['family']}:{r['variant']}  fw={r['framework']}  "
              f"repo={r['upstream_repo']}  runner={r['runner_rel']}  "
              f"env_fp={r['env_fp']}")
    print("========================================================")

    runner_fn = _REF_GPU_FNS.get(gpu)
    if runner_fn is None:
        raise SystemExit(
            f"--gpu {gpu!r} not registered; choose one of: {sorted(_REF_GPU_FNS)}")

    print(f">>> prefetch dataset ({dataset})")
    dataset_status = prefetch_dataset.remote(dataset)
    print(f">>> dataset ready: {dataset_status['dataset_id']} "
          f"n={dataset_status['utterances']} "
          f"sha={dataset_status['manifest_sha256'][:12]}")

    # One container per variant: it `uv sync`s the env once (on local disk) and
    # runs all requested batch sizes. Distinct variants fan out in parallel.
    n = None if n_utts < 0 else n_utts
    print(f">>> launching {len(resolved)} {gpu} reference cells "
          f"(batch sizes {sizes} each)...")
    futs = [(r, runner_fn.spawn(
                r["family"], r["variant"], r["framework"], r["upstream_repo"],
                r["runner_rel"], r["env_fp"], dataset, n, sizes, "cuda", mode,
                dataset_status))
            for r in resolved]

    rows, failures = [], []
    for r, fut in futs:
        try:
            per_bs = fut.get()  # {str(bs): {hyp_jsonl, summary, cached}}
        except Exception as e:
            failures.append((r["variant"], repr(e)))
            print(f"  [FAIL] {r['variant']}: {e} (check Modal dashboard)")
            continue
        for bs in sizes:
            res = per_bs.get(str(bs))
            slug = r["variant"] + (f" b{bs}" if bs != 1 else "")
            if res is None:
                failures.append((slug, "no result for this batch size"))
                continue
            p = write_ref_hyp(repo_root, res["hyp_jsonl"], r["variant"], dataset,
                              batch_size=(bs if bs != 1 else None),
                              mode=mode)
            s = res["summary"]
            rows.append((slug, s["n_utts"], s["audio_s"], s["wall_s"],
                         s["rtf_wall"], str(p)))
            tag = "CACHED" if res.get("cached") else "OK"
            print(f"  [{tag}] {slug}: {s['wall_s']:.1f}s, "
                  f"RTF {s['rtf_wall']:.1f}x -> {p}")

    print("\n========== reference_sweep summary ==========")
    print(f"{'variant':<44} {'n':>5} {'audio':>9} {'wall':>8} {'rtf':>6}")
    for slug, n_, audio, wall, rtf, _ in rows:
        print(f"{slug:<44} {n_:>5} {audio:>9.1f} {wall:>8.1f} {rtf:>6.1f}")
    if failures:
        print("\nfailures:")
        for slug, err in failures:
            print(f"  {slug}: {err}")
    print(f"\nscore locally:  for f in reports/wer/*-REF.{dataset_id(dataset)}*.jsonl; "
          f"do uv run scripts/wer/score.py \"$f\"; done")
