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

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
from collections import deque
import sys
import time

import modal

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------

# REPO is only used on the local invoker: by add_local_dir at image build time
# and by _write_hyp in the local entrypoint. Inside Modal containers the module
# is re-imported from /root/modal_sweep.py, where parents[3] does not exist.
_parents = pathlib.Path(__file__).resolve().parents
REPO = str(_parents[3]) if len(_parents) > 3 else "/src"


def _source_fingerprint(root: pathlib.Path) -> str:
    """Hash of the C++ source set that affects the build binary.

    Includes CMakeLists.txt content + the path list of C++ source/header
    files. Does NOT include Python scripts (they don't affect the build).
    Both the laptop (REPO = repo root) and the container (REPO = /src) reach
    the same hash because add_local_dir's ignore list mirrors the skip set.
    """
    if not root.is_dir():
        return "unknown"
    # Mirrors the add_local_dir(ignore=...) list. envs is the big one:
    # scripts/envs/<family>/ contains tens of thousands of vendored Python
    # files. Pruning at traversal time keeps this function sub-second.
    #
    # `skip_dir_prefixes` mirrors the glob patterns in add_local_dir's
    # ignore list ("build-*/**", ...) — the bare set above only catches
    # literal name matches, which leaves stale `build-vulkan/`,
    # `build-san/`, etc. visible to the laptop while the container's
    # add_local_dir excludes them. The two enumerators must agree or
    # the build-cache lookup downstream looks at a different SRC_FP
    # than the build wrote.
    skip_dirs = {"build", ".cache", ".git", ".venv", "__pycache__",
                 "reports", "models", "samples", "envs", "node_modules"}
    skip_dir_prefixes = ("build-",)
    h = hashlib.sha256()
    entries: list[tuple[str, pathlib.Path]] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames
                       if d not in skip_dirs
                       and not d.startswith(".")
                       and not any(d.startswith(p) for p in skip_dir_prefixes)]
        for fn in filenames:
            p = pathlib.Path(dirpath, fn)
            entries.append((str(p.relative_to(root)), p))
    entries.sort()
    for rel, p in entries:
        rel_b = rel.encode()
        if p.name == "CMakeLists.txt":
            h.update(b"cm\0"); h.update(rel_b); h.update(b"\0")
            h.update(p.read_bytes()); h.update(b"\0")
        elif p.suffix in (".cpp", ".cu", ".h", ".hpp", ".cuh"):
            # Content is hashed: source edits rotate BUILD_DIR, sidestepping
            # ninja mtime quirks when /src is uploaded with reset timestamps.
            h.update(b"src\0"); h.update(rel_b); h.update(b"\0")
            h.update(p.read_bytes()); h.update(b"\0")
    return h.hexdigest()[:12]


def _hyp_extra_hash(root: pathlib.Path) -> str:
    """Hash of the Python pieces that affect hyp output: run.py + ingest.py.

    Folded with SRC_FP into the hyp cache key. Edits to the dispatcher
    (modal_sweep.py) or to local-only scripts (score.py) do NOT invalidate
    the hyp cache because they cannot change what the cell produces.
    """
    h = hashlib.sha256()
    for name in ("run.py", "ingest.py"):
        p = root / "scripts" / "wer" / name
        if p.is_file():
            h.update(name.encode()); h.update(b"\0")
            h.update(p.read_bytes()); h.update(b"\0")
    return h.hexdigest()[:12]


# SRC_FP keys the build cache (C++ binary). HYP_FP keys the hyp cache and
# folds SRC_FP in so a binary change invalidates hyps too. Splitting them
# means dispatcher edits (modal_sweep.py) rotate NEITHER cache; only the
# files that actually affect the artifact rotate their respective cache.
SRC_FP = _source_fingerprint(pathlib.Path(REPO))
HYP_FP = hashlib.sha256(
    (SRC_FP + _hyp_extra_hash(pathlib.Path(REPO))).encode()
).hexdigest()[:12]

# Per-GPU SM arch. Single-arch builds finish nvcc in ~2 min vs ~6 min for a
# fat 75;86;89 build. Switching --gpu rebuilds for the new arch the first time
# (then sits cached in the build Volume alongside any other arch you've used).
GPU_TO_ARCH = {
    "T4": "75",
    "L4": "89",
    "A10G": "86",
    "L40S": "89",
    "A100": "80",
    "A100-80GB": "80",
    "H100": "90",
    "H200": "90",
}


def _build_dir(gpu_id: str) -> str:
    """Build dir for (current source fingerprint, this GPU's SM arch).
    Co-keyed so two GPUs never collide and source changes always rebuild."""
    return f"/build/cuda-{SRC_FP}-sm{GPU_TO_ARCH[gpu_id]}"

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
        "zlib1g-dev", "libopenblas-dev", "curl", "tar", "rsync",
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
            "build/**", "build-*/**", "models/**",
            "samples/wer/raw/**", "samples/wer/librispeech-*/**",
            "samples/wer/fleurs-*/**", "samples/wer/*.manifest.jsonl",
            "reports/**",
            "scripts/envs/**", "**/__pycache__/**", "**/.venv/**",
            "**/.uv/**", "**/.git/**", ".git/**",
            "tests/golden/**",
            "**/*.gguf", "**/*.safetensors", "**/*.onnx", "**/*.onnx.data",
            "**/*.nemo", ".DS_Store",
        ],
    )
)

build_vol = modal.Volume.from_name("transcribe-build", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("transcribe-data", create_if_missing=True)

app = modal.App("transcribe-wer-remote")


# ---------------------------------------------------------------------------
# Dataset spec helpers.
# ---------------------------------------------------------------------------

def parse_dataset_spec(spec: str) -> tuple[str, str]:
    """'librispeech:test-clean' -> ('librispeech', 'test-clean')
       'fleurs:zh'              -> ('fleurs', 'zh')
       'librispeech'            -> ('librispeech', 'test-clean')  (default)
    """
    kind, _, val = spec.partition(":")
    kind = kind.strip().lower()
    val = val.strip()
    if kind == "librispeech":
        return "librispeech", val or "test-clean"
    if kind == "fleurs":
        if not val:
            raise ValueError("fleurs: requires a language, e.g. fleurs:zh")
        return "fleurs", val
    raise ValueError(f"unknown dataset kind: {spec!r}")


def dataset_id(spec: str) -> str:
    """Slug used in filenames + volume paths. 'librispeech-test-clean' / 'fleurs-zh'."""
    kind, val = parse_dataset_spec(spec)
    return f"{kind}-{val}"


def manifest_path_for(spec: str) -> str:
    return f"/data/wer/{dataset_id(spec)}.manifest.jsonl"


def ingest_args_for(spec: str) -> list[str]:
    kind, val = parse_dataset_spec(spec)
    if kind == "librispeech":
        return ["librispeech", "--split", val]
    if kind == "fleurs":
        return ["fleurs", "--lang", val]
    raise ValueError(spec)


# ---------------------------------------------------------------------------
# Workdir setup (called inside any function that uses ingest/run.py).
# /src is read-only; copy to /work and symlink samples/wer/* into /data.
# ---------------------------------------------------------------------------

def _prepare_work(spec: str | None = None) -> str:
    """Build a writable /work mirror of /src with samples/wer/* pointed at the
    /data volume. Returns the manifest path for the given spec (if any)."""
    if os.path.exists("/work"):
        shutil.rmtree("/work")
    shutil.copytree("/src", "/work")
    os.makedirs("/data/wer", exist_ok=True)
    samples_wer = "/work/samples/wer"
    os.makedirs(samples_wer, exist_ok=True)

    # Per-known-subdir symlinks. Keeps unrelated /src/samples/wer/* untouched.
    for sub in ("raw",):
        link, target = os.path.join(samples_wer, sub), f"/data/wer/{sub}"
        os.makedirs(target, exist_ok=True)
        _replace_with_symlink(link, target)

    if spec is not None:
        ds_id = dataset_id(spec)
        for sub in (ds_id,):
            link, target = os.path.join(samples_wer, sub), f"/data/wer/{sub}"
            os.makedirs(target, exist_ok=True)
            _replace_with_symlink(link, target)
        manifest_link = os.path.join(samples_wer, f"{ds_id}.manifest.jsonl")
        manifest_target = f"/data/wer/{ds_id}.manifest.jsonl"
        _replace_with_symlink(manifest_link, manifest_target)
        return manifest_target
    return ""


def _replace_with_symlink(link: str, target: str) -> None:
    # The rmtree branch only fires on real dirs at /work/samples/wer/<sub>; the
    # add_local_dir ignore list keeps those paths empty in /src, so we are only
    # deleting the placeholder dir created moments earlier by _prepare_work.
    if os.path.lexists(link):
        if os.path.islink(link) or os.path.isfile(link):
            os.unlink(link)
        elif os.path.isdir(link):
            shutil.rmtree(link)
    os.symlink(target, link)


# ---------------------------------------------------------------------------
# 1. Build (CPU container; nvcc does not need a GPU).
# ---------------------------------------------------------------------------

@app.function(image=image, timeout=1800, volumes={"/build": build_vol})
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
    # cmake reads /src directly (read-only mount) and only writes to build_dir,
    # so this is the one function that does not need a /work mirror.
    subprocess.check_call([
        "cmake", "-S", "/src", "-B", build_dir, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DTRANSCRIBE_CUDA=ON",
        "-DTRANSCRIBE_BUILD_TESTS=OFF",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
    ])
    subprocess.check_call([
        "cmake", "--build", build_dir, "--target", "transcribe-cli",
        "-j", str(os.cpu_count()),
    ])
    build_vol.commit()
    assert os.path.exists(cli_path)
    print(f"[build] {cli_path} ready")


# ---------------------------------------------------------------------------
# 2. Dataset prefetch (idempotent; cached on /data volume).
# ---------------------------------------------------------------------------

@app.function(image=image, timeout=3600, volumes={"/data": data_vol})
def prefetch_dataset(spec: str) -> str:
    """Ensure the manifest for `spec` is present on /data. Returns its path."""
    manifest = _prepare_work(spec)
    if os.path.exists(manifest) and os.path.getsize(manifest) > 0:
        n = sum(1 for _ in open(manifest))
        print(f"[data] {spec}: cached at {manifest} ({n} utts)")
        return manifest

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
    n = sum(1 for _ in open(manifest))
    print(f"[data] {spec}: manifest ready ({n} utts)")
    return manifest


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

def _hyp_cache_paths(model_file: str, dataset_spec: str,
                     n_utts: int | None,
                     batch_size: int = 1,
                     sort_by_length: bool = True) -> tuple[str, str]:
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
    base = (f"/data/wer/hyps/{HYP_FP}/{slug}."
            f"{dataset_id(dataset_spec)}.{subset_tag}{bs_tag}{sort_tag}")
    return f"{base}.jsonl", f"{base}.summary.json"


def _run_subprocess_capturing_stderr(cmd: list[str], cwd: str,
                                     env: dict) -> tuple[int, str]:
    """Run cmd with stdout inheriting (live progress to Modal logs) and stderr
    tee'd through a background thread so we both stream and capture it.

    Returns (returncode, stderr_tail) where stderr_tail is up to the last 80
    lines, suitable for inclusion in a failure exception.
    """
    tail: deque[str] = deque(maxlen=80)

    def _tee(stream):
        for line in stream:
            sys.stderr.write(line)
            sys.stderr.flush()
            tail.append(line)

    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    thr = threading.Thread(target=_tee, args=(proc.stderr,), daemon=True)
    thr.start()
    rc = proc.wait()
    thr.join(timeout=5)
    return rc, "".join(tail)


def _run_wer_impl(
    model_repo: str,
    model_file: str,
    dataset_spec: str,
    n_utts: int | None,
    build_dir: str,
    batch_size: int = 1,
    sort_by_length: bool = True,
) -> dict:
    """Run scripts/wer/run.py on `n_utts` (or full manifest) and return the
    hyp JSONL contents + a summary dict back to the dispatcher.

    Volume-cached by (SRC_FP, model_file, dataset, subset). On a cache hit the
    container returns immediately without downloading the model or running
    the CLI; this is what makes interrupted sweeps resumable.
    """
    from huggingface_hub import hf_hub_download

    # Cache lookup first; skips _prepare_work + model download + run.py entirely
    # when a hyp for this (fingerprint, model, dataset, subset) already exists.
    cache_hyp, cache_sum = _hyp_cache_paths(
        model_file, dataset_spec, n_utts, batch_size, sort_by_length)
    if os.path.exists(cache_hyp) and os.path.exists(cache_sum) \
       and os.path.getsize(cache_hyp) > 0:
        print(f"[wer] cache hit: {cache_hyp}")
        with open(cache_sum) as f:
            summary = json.load(f)
        with open(cache_hyp) as f:
            hyp_jsonl = f.read()
        return {"hyp_jsonl": hyp_jsonl, "summary": summary, "cached": True}

    manifest = _prepare_work(dataset_spec)
    assert os.path.exists(manifest), (
        f"manifest missing at {manifest}; call prefetch_dataset first"
    )

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

    if n_utts is None:
        subset_path = manifest
        subset_count = sum(1 for _ in open(subset_path))
    else:
        subset_path = "/tmp/subset.manifest.jsonl"
        with open(manifest) as fin, open(subset_path, "w") as fout:
            for i, line in enumerate(fin):
                if i >= n_utts:
                    break
                fout.write(line)
        subset_count = sum(1 for _ in open(subset_path))
    print(f"[wer] manifest: {subset_count} utts")

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
    print(f"[wer] $ {' '.join(cmd)}")
    t0 = time.time()
    rc, stderr_tail = _run_subprocess_capturing_stderr(cmd, cwd="/work", env=env)
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
        "mel_s": mel_ms / 1000.0,
        "encode_s": enc_ms / 1000.0,
        "decode_s": dec_ms / 1000.0,
        "gpu": gpu,
    }
    print(f"[wer] done: {summary}")

    # Persist to Volume for future resume / re-invocation.
    os.makedirs(os.path.dirname(cache_hyp), exist_ok=True)
    shutil.copyfile(hyp_path, cache_hyp)
    with open(cache_sum, "w") as f:
        json.dump(summary, f)
    data_vol.commit()
    return {"hyp_jsonl": hyp_jsonl, "summary": summary, "cached": False}


_RUN_WER_KWARGS = dict(
    image=image, timeout=7200,
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
    build_dir = _build_dir(gpu_id)

    def runner(
        model_repo: str,
        model_file: str,
        dataset_spec: str = "librispeech:test-clean",
        n_utts: int | None = None,
        batch_size: int = 1,
        sort_by_length: bool = True,
    ) -> dict:
        return _run_wer_impl(
            model_repo, model_file, dataset_spec, n_utts, build_dir,
            batch_size=batch_size, sort_by_length=sort_by_length,
        )

    runner.__name__ = name
    runner.__qualname__ = name
    decorated = app.function(gpu=gpu_id, **_RUN_WER_KWARGS)(runner)
    globals()[name] = decorated  # required for container-side import to find it
    return decorated


# Single source of truth for "which GPUs we register and what arch they map to":
# iterate GPU_TO_ARCH.
_GPU_FNS = {gpu: _register_runner(gpu) for gpu in GPU_TO_ARCH}



# ---------------------------------------------------------------------------
# Model resolution: a spec is either an hf_card slug or a bare HF repo.
# ---------------------------------------------------------------------------

def _resolve_model(spec: str) -> tuple[str, list[str] | None]:
    """Map a model spec to (HF repo, filenames or None).

    Rules:
    - Spec contains '/': treat as a HF repo path. Filenames are None
      (caller will discover via the HF API at dispatch time).
    - Otherwise: treat as an hf_card slug. Reads
      scripts/hf_cards/<slug>.yaml, returns its target_repo and the
      pinned `quants[].filename` list.
    """
    if "/" in spec:
        return spec, None
    card_path = pathlib.Path(__file__).resolve().parents[2] / "hf_cards" / f"{spec}.yaml"
    if not card_path.exists():
        raise SystemExit(
            f"no hf_card at {card_path}; pass a HF repo path "
            f"(e.g. handy-computer/{spec}-gguf) if the card doesn't exist yet"
        )
    # Tiny manual parser so the local entrypoint has no non-stdlib deps. The card
    # schema has target_repo at top level and filename: only under quants[].
    target_repo: str | None = None
    filenames: list[str] = []
    for raw in open(card_path):
        line = raw.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if line.startswith("target_repo:"):
            target_repo = line.split(":", 1)[1].strip()
        elif stripped.startswith("filename:"):
            filenames.append(stripped.split(":", 1)[1].strip())
    if not target_repo:
        raise SystemExit(f"hf_card {spec!r}: missing target_repo")
    if not filenames:
        raise SystemExit(f"hf_card {spec!r} has no quants[].filename entries")
    return target_repo, filenames


def _write_hyp(hyp_jsonl: str, model_file: str, dataset_spec: str) -> pathlib.Path:
    out_dir = pathlib.Path(REPO) / "reports" / "wer"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model_file.replace(".gguf", "")
    ds = dataset_id(dataset_spec)
    out_path = out_dir / f"{slug}.{ds}.jsonl"
    out_path.write_text(hyp_jsonl)
    return out_path


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

    resolved = [(s, *_resolve_model(s)) for s in specs]

    needs_listing = sorted({repo for _, repo, fns in resolved if fns is None})
    listings = dict(list_ggufs.remote(needs_listing)) if needs_listing else {}

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
            cells.append({"repo": repo, "file": f, "dataset": dataset})

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
    prefetch_dataset.remote(dataset)

    print(f">>> launching {len(cells)} {gpu} containers in parallel...")
    n = None if n_utts < 0 else n_utts
    futs = [(c, runner.spawn(c["repo"], c["file"], c["dataset"], n))
            for c in cells]

    rows, failures = [], []
    for c, fut in futs:
        slug = c["file"].replace(".gguf", "")
        try:
            res = fut.get()
            p = _write_hyp(res["hyp_jsonl"], c["file"], c["dataset"])
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
    repo, fns = _resolve_model(model)
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
    prefetch_dataset.remote(dataset)

    n = None if n_utts < 0 else n_utts
    # Launch all batch sizes in parallel (each its own container).
    futs = [(bs, runner.spawn(repo, model_file, dataset, n, bs, sort_by_length))
            for bs in sizes]

    rows: list[tuple] = []
    hyp_written = False
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
            if not hyp_written:
                _write_hyp(res["hyp_jsonl"], model_file, dataset)
                hyp_written = True
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
    print("WER is identical across batch sizes; score the hyp once:")
    print(f"  uv run scripts/wer/score.py "
          f"reports/wer/{model_file.replace('.gguf', '')}."
          f"{dataset_id(dataset)}.jsonl")


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

