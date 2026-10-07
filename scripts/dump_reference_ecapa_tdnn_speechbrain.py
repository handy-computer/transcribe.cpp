#!/usr/bin/env python3
"""
dump_reference_ecapa_tdnn_speechbrain.py - generate SpeechBrain
VoxLingua107 ECAPA-TDNN reference tensors.

SpeechBrain is the only published implementation of this checkpoint: the
`hyperparams.yaml` in the HF repo instantiates SpeechBrain classes
(`lobes.features.Fbank`, `lobes.models.ECAPA_TDNN.ECAPA_TDNN`,
`lobes.models.Xvector.Classifier`) and the model card's inference example
is `EncoderClassifier.from_hparams`. There is no Transformers shim.

Usage:

    uv run --project scripts/envs/ecapa_tdnn \
      scripts/dump_reference_ecapa_tdnn_speechbrain.py encoder \
      --model speechbrain/lang-id-voxlingua107-ecapa \
      --audio samples/fleurs-en.wav \
      --out build/validate/ecapa_tdnn/lang-id-voxlingua107-ecapa/fleurs-en/ref

Writes:
    <name>.f32        raw little-endian float32, row-major
    <name>.json       per-tensor sidecar via shared scripts.lib.ref_dump
    prediction.json   behavioural artifact (top-1 label + top-5 log-probs)

There is a single stage, `encoder`: language ID is one forward pass, so
the front end, the ECAPA-TDNN embedding network and the classifier head
are all dumped from one `classify_batch` call. Running the model twice
would fire every hook twice; the script asserts each hook fired exactly
once.

Architecture summary:

    audio (16 kHz mono)
      -> Fbank: STFT(n_fft=400, win=400, hop=160, hamming periodic,
                     center, constant pad) -> power spectrum [T, 201]
                60 triangular mel filters -> 10*log10 -> 80 dB floor
      -> InputNormalization(norm_type="sentence", std_norm=False)
                     => fe.mel [T, 60]
      -> ECAPA_TDNN: blocks.0 TDNN(60->1024, k=5)
                     blocks.1..3 SERes2Net(1024, k=3, d=2/3/4)
                     mfa TDNN(3072->3072, k=1) over concat(blocks.1..3)
                     asp AttentiveStatisticsPooling(3072, attn=128)
                     asp_bn -> fc(6144->256)   => enc.emb [256]
      -> Xvector.Classifier: LeakyReLU -> BN -> Linear(256->512)
                     -> LeakyReLU -> BN -> Linear(512->107)
                     -> log_softmax

Dump points (the contract with src/arch/ecapa_tdnn/graph.h, in that exact
order and with those exact names). Every activation is written TIME-MAJOR `[T, C]`:
PyTorch carries ECAPA activations as `[B, C, T]`, while the C++ debug
dumper writes ggml `ne=[C, T]` as row-major `[T, C]`. Squeezing the batch
dim and transposing here is what makes the two directories comparable.

    fe.mel                  [T, 60]     mean_var_norm output
    enc.blk.0.out           [T, 1024]
    enc.blk.1.tdnn1.out     [T, 1024]   SERes2Net sub-steps, for bisecting
    enc.blk.1.res2.out      [T, 1024]
    enc.blk.1.se.out        [T, 1024]   before the residual add
    enc.blk.{1,2,3}.out     [T, 1024]
    enc.mfa.out             [T, 3072]
    enc.asp.attn_logits     [T, 3072]   asp.conv (asp.attn.weight) output, pre-softmax
    enc.asp.out             [6144]
    enc.emb                 [256]       fc output
    cls.hidden              [512]       DNN.block_0.act output (pre-BN1)
    cls.logits_raw          [107]       the gate

Plus two run-invariant front-end tables, dumped so the converter and the
C++ mel unit test can be checked against SpeechBrain directly:

    fe.window               [400]       STFT.window (periodic Hamming)
    fe.filters              [201, 60]   the fbank matrix used this forward

These two are NOT part of the tolerance contract: the C++ runtime does
not dump them (the window is recomputed and the filterbank is read from
the GGUF), so compare_tensors reports them MISSING-left without failing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.lib.ref_dump import write_tensor  # noqa: E402
from lib.gguf_common import slug_from_repo_id  # noqa: E402
from lib.hf_source import looks_like_repo_id, resolve_model_dir  # noqa: E402

DEFAULT_REPO = "speechbrain/lang-id-voxlingua107-ecapa"
DEFAULT_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"

SAMPLE_RATE = 16000
HOP = 160
N_FFT = 400
WIN = 400
N_STFT = N_FFT // 2 + 1  # 201
N_MELS = 60
EMB_DIM = 256
N_LABELS = 107
HIDDEN = 512


# ---------------------------------------------------------------------------
# Torch setup
# ---------------------------------------------------------------------------


def configure_torch(args: argparse.Namespace) -> None:
    import torch

    torch.manual_seed(0)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Already set by an earlier call in this process; harmless.
            pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# Shape conversion
# ---------------------------------------------------------------------------


def act_time_major(t, channels: int, *, name: str) -> np.ndarray:
    """`[1, C, T]` torch activation -> `[T, C]` float32 numpy.

    The channel count is asserted rather than inferred. A bare transpose
    would silently succeed on a tensor that is already `[1, T, C]`, and the
    resulting dump would look plausible while being the transpose of what
    the C++ side writes.
    """
    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor, got {type(t)!r}")
    if t.dim() != 3 or t.shape[0] != 1:
        raise ValueError(f"{name}: expected [1, C, T], got {tuple(t.shape)}")
    if t.shape[1] != channels:
        raise ValueError(
            f"{name}: expected {channels} channels on dim 1, "
            f"got {tuple(t.shape)}"
        )
    a = t.detach().to(dtype=torch.float32, device="cpu")[0].transpose(0, 1).numpy()
    return np.ascontiguousarray(a, dtype=np.float32)


def feat_time_major(t, channels: int, *, name: str) -> np.ndarray:
    """`[1, T, C]` torch feature tensor -> `[T, C]` float32 numpy.

    SpeechBrain's front end is already time-major (`[B, T, n_mels]`), so
    this only drops the batch dim — but it asserts the channel count for
    the same reason `act_time_major` does.
    """
    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor, got {type(t)!r}")
    if t.dim() != 3 or t.shape[0] != 1:
        raise ValueError(f"{name}: expected [1, T, C], got {tuple(t.shape)}")
    if t.shape[2] != channels:
        raise ValueError(
            f"{name}: expected {channels} channels on dim 2, "
            f"got {tuple(t.shape)}"
        )
    a = t.detach().to(dtype=torch.float32, device="cpu")[0].numpy()
    return np.ascontiguousarray(a, dtype=np.float32)


def vec(t, n: int, *, name: str) -> np.ndarray:
    """Any torch tensor with exactly `n` elements -> 1-D float32 numpy."""
    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor, got {type(t)!r}")
    if t.numel() != n:
        raise ValueError(f"{name}: expected {n} elements, got {t.numel()} "
                         f"(shape {tuple(t.shape)})")
    a = t.detach().to(dtype=torch.float32, device="cpu").reshape(-1).numpy()
    return np.ascontiguousarray(a, dtype=np.float32)


def mat(t, shape: tuple[int, int], *, name: str) -> np.ndarray:
    """Torch matrix -> 2-D float32 numpy, with an exact shape assertion."""
    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor, got {type(t)!r}")
    if tuple(t.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(t.shape)}")
    a = t.detach().to(dtype=torch.float32, device="cpu").numpy()
    return np.ascontiguousarray(a, dtype=np.float32)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


class CaptureHook:
    """Record the output a module emits during forward, and count firings.

    The count is checked after the pass: a hook that fired zero times means
    the module was not on the path we think it was, and a hook that fired
    twice means the model ran more than once (which would make the dump
    reflect the second call). Both are silent-wrong-number bugs otherwise.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.value = None
        self.calls = 0

    def __call__(self, module, inputs, output) -> None:
        self.calls += 1
        if isinstance(output, tuple):
            output = output[0]
        self.value = output


def hook(module, name: str, registry: dict[str, CaptureHook], handles: list):
    h = CaptureHook(name)
    handles.append(module.register_forward_hook(h))
    registry[name] = h
    return h


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def resolve_checkpoint_dir(model_arg: str, revision: str | None) -> Path:
    """Return a local SpeechBrain checkpoint directory (HF cache or local)."""
    p = Path(model_arg).expanduser()
    if not p.is_dir() and not looks_like_repo_id(model_arg):
        raise SystemExit(
            f"error: --model {model_arg!r} is neither an existing directory "
            f"nor an org/name Hugging Face repo id"
        )
    return resolve_model_dir(model_arg, revision)


def load_reference(args: argparse.Namespace):
    """Build the SpeechBrain `EncoderClassifier` for this checkpoint.

    Two things here are load-bearing:

    1. `overrides={"pretrained_path": <local dir>}`. The shipped
       `hyperparams.yaml` sets `pretrained_path:
       speechbrain/lang-id-voxlingua107-ecapa`, and `Pretrainer.collect_files`
       splits each `paths[...]` entry into (source, filename) — so without
       the override SpeechBrain re-fetches the `.ckpt` files from the HF
       `main` branch, ignoring our pinned revision entirely.
    2. `savedir != source`. SpeechBrain symlinks the collected loadables
       into `savedir`; pointing it at the snapshot directory makes it
       symlink files onto themselves.
    """
    import torch
    from speechbrain.inference.classifiers import EncoderClassifier

    src = resolve_checkpoint_dir(args.model, args.revision)
    savedir = REPO_ROOT / "build" / "speechbrain" / src.parent.name
    savedir.mkdir(parents=True, exist_ok=True)

    print(f"Loading EncoderClassifier: source={src} savedir={savedir} "
          f"device={args.device}")
    clf = EncoderClassifier.from_hparams(
        source=str(src),
        savedir=str(savedir),
        overrides={"pretrained_path": str(src)},
        run_opts={"device": args.device},
    )

    # Force eval + fp32 + no grad ourselves rather than trusting run_opts.
    # A BatchNorm left in train mode would use this batch's statistics
    # (batch of one!) instead of the running stats, and every tensor from
    # that layer onward would be plausible-looking garbage.
    clf.mods.eval()
    for p in clf.mods.parameters():
        p.requires_grad_(False)

    training = [n for n, m in clf.mods.named_modules() if m.training]
    if training:
        raise SystemExit(f"error: modules still in training mode: {training[:5]}")

    n_bn = 0
    for name, m in clf.mods.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            n_bn += 1
            if m.running_mean is None or m.running_var is None:
                raise SystemExit(
                    f"error: BatchNorm {name} has no running statistics; "
                    "the checkpoint did not load as expected"
                )
    print(f"eval mode ok; {n_bn} BatchNorm layers carry running statistics")

    dtypes = {str(p.dtype) for p in clf.mods.parameters()}
    if dtypes != {"torch.float32"}:
        raise SystemExit(f"error: expected an fp32 model, got dtypes {dtypes}")

    return clf, src


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def load_audio(audio_path: Path) -> np.ndarray:
    import soundfile as sf

    pcm, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if int(sr) != SAMPLE_RATE:
        raise SystemExit(
            f"error: {audio_path} is {sr} Hz; the model front end is "
            f"{SAMPLE_RATE} Hz only. Resample before dumping."
        )
    return np.ascontiguousarray(pcm, dtype=np.float32)


def split_label(label: str) -> tuple[str, str]:
    """`'zh: Chinese'` -> `('zh', 'Chinese')`.

    The label encoder stores the code and the English name in one string.
    A label without the separator is passed through as its own code with an
    empty name rather than crashing, so a future label-set change surfaces
    as odd metadata instead of a traceback mid-dump.
    """
    if ": " in label:
        code, name = label.split(": ", 1)
        return code.strip(), name.strip()
    return label.strip(), ""


# ---------------------------------------------------------------------------
# encoder stage
# ---------------------------------------------------------------------------


def cmd_encoder(args: argparse.Namespace) -> int:
    import torch
    import speechbrain

    configure_torch(args)
    clf, src_dir = load_reference(args)

    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pcm = load_audio(audio_path)
    n_samples = int(pcm.size)
    expected_T = n_samples // HOP + 1
    print(f"audio: {audio_path.name} samples={n_samples} "
          f"({n_samples / SAMPLE_RATE:.3f} s) expected_frames={expected_T}")

    device = torch.device(args.device)
    wav = torch.from_numpy(pcm).to(device).unsqueeze(0)          # [1, N]
    # `wav_lens` is RELATIVE (fraction of the padded batch max), not a
    # sample count. Batch of one, unpadded -> 1.0. Passing n_samples here
    # would make InputNormalization and the ASP/SE masks compute over
    # round(n_samples * T) frames, i.e. the wrong span, silently.
    wav_lens = torch.ones(1, device=device)

    fe = clf.mods.compute_features
    em = clf.mods.embedding_model
    cl = clf.mods.classifier

    handles: list = []
    hooks: dict[str, CaptureHook] = {}

    hook(clf.mods.mean_var_norm, "fe.mel", hooks, handles)
    hook(em.blocks[0], "enc.blk.0.out", hooks, handles)
    hook(em.blocks[1].tdnn1, "enc.blk.1.tdnn1.out", hooks, handles)
    hook(em.blocks[1].res2net_block, "enc.blk.1.res2.out", hooks, handles)
    hook(em.blocks[1].se_block, "enc.blk.1.se.out", hooks, handles)
    hook(em.blocks[1], "enc.blk.1.out", hooks, handles)
    hook(em.blocks[2], "enc.blk.2.out", hooks, handles)
    hook(em.blocks[3], "enc.blk.3.out", hooks, handles)
    hook(em.mfa, "enc.mfa.out", hooks, handles)
    hook(em.asp.conv, "enc.asp.attn_logits", hooks, handles)
    hook(em.asp, "enc.asp.out", hooks, handles)
    hook(em.fc, "enc.emb", hooks, handles)
    hook(cl.DNN.block_0.act, "cls.hidden", hooks, handles)
    hook(cl.out, "cls.logits_raw", hooks, handles)
    hook(cl.softmax, "cls.log_probs", hooks, handles)

    # SpeechBrain 1.1.1's Filterbank builds its [201, 60] matrix inside
    # forward() from `f_central` / `band`; there is no `fbank_matrix`
    # attribute or buffer to read. Wrapping the real method captures exactly
    # the matrix this forward pass used, which is stronger than recomputing it.
    fbanks = fe.compute_fbanks
    captured_filters: dict[str, Any] = {}
    original_create = fbanks._create_fbank_matrix

    def capture_create(f_central_mat, band_mat):
        m = original_create(f_central_mat, band_mat)
        captured_filters.setdefault("value", m)
        captured_filters["calls"] = captured_filters.get("calls", 0) + 1
        return m

    fbanks._create_fbank_matrix = capture_create

    try:
        with torch.inference_mode():
            out_prob, score, index, text_lab = clf.classify_batch(wav, wav_lens)
    finally:
        fbanks._create_fbank_matrix = original_create
        for h in handles:
            h.remove()

    # Every hook must have fired exactly once. Zero means the module was
    # not on the forward path; more than one means the model ran twice and
    # the dump would reflect the last call.
    bad = {n: h.calls for n, h in hooks.items() if h.calls != 1}
    if bad:
        raise SystemExit(f"error: unexpected hook call counts: {bad}")
    if captured_filters.get("calls") != 1:
        raise SystemExit(
            f"error: Filterbank._create_fbank_matrix fired "
            f"{captured_filters.get('calls')} times, expected 1"
        )

    T = int(hooks["fe.mel"].value.shape[1])
    if T != expected_T:
        raise SystemExit(
            f"error: frame count mismatch: fe.mel has T={T}, but "
            f"floor({n_samples}/{HOP}) + 1 = {expected_T}. The C++ front end "
            f"derives T from this formula; a mismatch means its STFT "
            f"framing assumption (center=True, hop {HOP}) is wrong."
        )
    print(f"frames: T={T} == floor({n_samples}/{HOP}) + 1  [ok]")

    source: dict[str, Any] = {
        "kind": "ecapa-tdnn-speechbrain",
        "framework": "speechbrain",
        "framework_version": speechbrain.__version__,
        "torch_version": torch.__version__,
        "model": args.model,
        "revision": args.revision,
        "checkpoint_dir": str(src_dir),
        "device": args.device,
        "torch_threads": args.torch_threads,
        "model_dtype": "f32",
        "audio": audio_path.name,
        "n_samples": n_samples,
        "sample_rate": SAMPLE_RATE,
        "n_frames": T,
        "n_mels": N_MELS,
    }

    def dump(name: str, array: np.ndarray, stage: str) -> None:
        print(f"  {name}: shape={array.shape} min={array.min():.4e} "
              f"max={array.max():.4e} mean={array.mean():.6e}")
        write_tensor(name, array, stage=stage, source=source, out_dir=out_dir)

    # ---- front end -------------------------------------------------------
    dump("fe.mel", feat_time_major(hooks["fe.mel"].value, N_MELS, name="fe.mel"),
         "frontend")

    # ---- encoder ---------------------------------------------------------
    for name, channels in (
        ("enc.blk.0.out", 1024),
        ("enc.blk.1.tdnn1.out", 1024),
        ("enc.blk.1.res2.out", 1024),
        ("enc.blk.1.se.out", 1024),
        ("enc.blk.1.out", 1024),
        ("enc.blk.2.out", 1024),
        ("enc.blk.3.out", 1024),
        ("enc.mfa.out", 3072),
        ("enc.asp.attn_logits", 3072),
    ):
        dump(name, act_time_major(hooks[name].value, channels, name=name), "encoder")

    dump("enc.asp.out", vec(hooks["enc.asp.out"].value, 6144, name="enc.asp.out"),
         "encoder")
    dump("enc.emb", vec(hooks["enc.emb"].value, EMB_DIM, name="enc.emb"), "encoder")

    # ---- classifier ------------------------------------------------------
    dump("cls.hidden", vec(hooks["cls.hidden"].value, HIDDEN, name="cls.hidden"),
         "classifier")
    dump("cls.logits_raw",
         vec(hooks["cls.logits_raw"].value, N_LABELS, name="cls.logits_raw"),
         "classifier")
    log_probs = vec(hooks["cls.log_probs"].value, N_LABELS, name="cls.log_probs")

    # The hooked softmax output must be the tensor classify_batch returned;
    # if it is not, the hook is on the wrong module.
    returned = out_prob.detach().to(dtype=torch.float32, device="cpu").reshape(-1).numpy()
    if not np.array_equal(returned, log_probs):
        raise SystemExit(
            "error: cls.log_probs hook does not match classify_batch's out_prob"
        )

    # ---- run-invariant front-end tables ---------------------------------
    window = fe.compute_STFT.window
    expected_window = torch.hamming_window(WIN)
    if not torch.equal(window.to(torch.float32).cpu(), expected_window):
        raise SystemExit(
            "error: STFT window is not torch.hamming_window(400); the periodic "
            "Hamming window the C++ front end builds does not describe this "
            "checkpoint"
        )
    dump("fe.window", vec(window, WIN, name="fe.window"), "frontend")

    filters = captured_filters["value"]
    dump("fe.filters", mat(filters, (N_STFT, N_MELS), name="fe.filters"), "frontend")

    # ---- prediction ------------------------------------------------------
    label_index = int(index.item())
    label_encoder = clf.hparams.label_encoder
    ind2lab = label_encoder.ind2lab
    if len(ind2lab) != N_LABELS:
        raise SystemExit(
            f"error: label encoder has {len(ind2lab)} labels, expected {N_LABELS}"
        )
    code, name = split_label(str(ind2lab[label_index]))
    decoded = text_lab[0] if isinstance(text_lab, list) else str(text_lab)
    if str(decoded) != str(ind2lab[label_index]):
        raise SystemExit(
            f"error: label_encoder.decode_torch gave {decoded!r} but "
            f"ind2lab[{label_index}] is {ind2lab[label_index]!r}"
        )

    order = np.argsort(-log_probs)[:5]
    top5 = []
    for i in order:
        c, n = split_label(str(ind2lab[int(i)]))
        top5.append({
            "index": int(i),
            "code": c,
            "name": n,
            "log_prob": float(log_probs[int(i)]),
        })

    prediction = {
        "label_index": label_index,
        "code": code,
        "name": name,
        "log_prob": float(log_probs[label_index]),
        "top5": top5,
        "source": {**source, "score": float(score.item())},
    }
    (out_dir / "prediction.json").write_text(json.dumps(prediction, indent=2) + "\n")
    print(f"prediction: {code} ({name}) log_prob={prediction['log_prob']:.6f} "
          f"prob={math.exp(prediction['log_prob']):.4f}")
    print("top5: " + ", ".join(f"{e['code']}={e['log_prob']:.3f}" for e in top5))

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_REPO,
                   help=f"HF repo id or local checkpoint dir (default: {DEFAULT_REPO})")
    p.add_argument("--revision", default=DEFAULT_REVISION,
                   help="HF revision (commit SHA) to pin the download to. "
                        "Ignored when --model is a local directory.")
    p.add_argument("--audio", required=True, help="16 kHz mono WAV path")
    p.add_argument("--out", required=True, help="Output directory for dumps")
    p.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch.set_num_threads (0 = unchanged)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="SpeechBrain VoxLingua107 ECAPA-TDNN reference dumper.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser(
        "encoder",
        help="Run the full forward pass; dump front end, encoder, classifier "
             "and prediction.json")
    add_common_args(ep)
    ep.set_defaults(func=cmd_encoder)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
