#!/usr/bin/env python3
"""
convert-ecapa_tdnn.py - convert SpeechBrain's VoxLingua107 ECAPA-TDNN
language-ID checkpoint into the F32 reference GGUF that transcribe.cpp's
ecapa_tdnn family (LANGID role) loads. F16 / Q8_0 come from
tools/transcribe-quantize, never from here.

Source format:
    SpeechBrain checkpoint directory (HF repo or local), with:

      hyperparams.yaml       instantiates lobes.features.Fbank,
                             lobes.models.ECAPA_TDNN.ECAPA_TDNN and
                             lobes.models.Xvector.Classifier by name
      embedding_model.ckpt   ECAPA-TDNN encoder (21,058,432 params)
      classifier.ckpt        Xvector.Classifier head (188,011 params)
      label_encoder.txt      107 `'<code>: <name>' => <index>` lines

There is no Transformers shim for this checkpoint: SpeechBrain is the only
implementation. So this converter LOADS THE MODEL THROUGH SPEECHBRAIN
(exactly the way `scripts/dump_reference_ecapa_tdnn_speechbrain.py` does,
including `overrides={"pretrained_path": ...}` — without it SpeechBrain
silently re-fetches the .ckpt files from the HF `main` branch and the
pinned revision is a lie) and reads weights off the live module tree. The
checkpoint resolves through scripts/lib/hf_source.py (HF hub cache, or the
$TRANSCRIBE_MODELS_DIR mirror when set).

Loading through SpeechBrain also lets the converter assert the
architecture against the *instantiated modules and tensor shapes* rather
than against `hyperparams.yaml`: the yaml carries only `input_size`,
`channels`, `kernel_sizes`, `dilations`, `attention_channels`,
`lin_neurons` and four front-end fields. Everything else (res2net_scale, se_channels, global_context, padding mode, STFT
geometry, top_db, amin, ...) is a SpeechBrain *default* of the pinned
version, so the yaml cannot confirm it.

Four exact rewrites happen here:

  1. Every TDNN BatchNorm becomes a `scale`/`shift` vector pair. It cannot
     be folded into its conv because SpeechBrain's TDNNBlock is
     conv -> ReLU -> BN.
  2. The three BNs that feed a linear map with no activation between
     (`asp_bn -> fc`, `cls.bn0 -> cls.l1`, `cls.bn1 -> cls.out`) are folded
     FORWARD into that linear map and disappear.
  3. `asp.tdnn`'s 9216-wide input weight is split into `x`/`mean`/`std`
     weights, one per member of SpeechBrain's `cat([x, mean, std])`.
  4. `mfa`'s 3072-wide input weight is split into three weights, one per
     SERes2Net block output.

Layout and naming conventions (the loader contract). Names follow the
shared tools/transcribe-quantize rules, so the quant policy needs no
ECAPA-specific entries:
  * k>1 kernels are `*.conv.weight`, stored TAP-MAJOR: numpy `[K, OC, IC]`
    (= `torch_w.permute(2, 0, 1)`), i.e. ggml `ne = [IC, OC, K]`, so each
    tap is a contiguous `[IC, OC]` matmul operand. `.conv.` keeps them in
    the quantizer's Conv bucket (F32 / F16 only).
  * 1x1 kernels are 2-D `*.weight`: numpy `[OC, IC]`, ggml `ne = [IC, OC]`
    (ordinary matmul operands; the quantizer's Linear bucket).
  * `frontend.mel_filterbank` is mel-major: numpy `[60, 201]` (=
    SpeechBrain's `[201, 60]` fbank matrix transposed), ggml
    `ne = [201, 60]`, so each filter's 201 weights are contiguous.
  * Biases (`*.bias`) and BN `*.bn.scale` / `*.bn.shift` are 1-D, F32.

CLI (writes models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-F32.gguf):

    uv run --project scripts/envs/ecapa_tdnn \
      scripts/convert-ecapa_tdnn.py speechbrain/lang-id-voxlingua107-ecapa

Quantize with tools/transcribe-quantize afterwards.

Single-file, top-to-bottom — no hidden helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# XET-backed transfers occasionally stall on this checkpoint's small .ckpt
# files; the plain HTTP path is fast enough for 21 M parameters. Set before
# huggingface_hub is imported (indirectly, via speechbrain) so it takes.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from gguf import GGMLQuantizationType, LlamaFileType

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.gguf_common import (  # noqa: E402
    add_general_identity,
    canonicalize_normalize,
    encode_for_gguf,
    gguf_name,
    gguf_writer,
    slug_from_repo_id,
)
from lib.hf_source import looks_like_repo_id, resolve_model_dir  # noqa: E402

DEFAULT_REPO = "speechbrain/lang-id-voxlingua107-ecapa"
DEFAULT_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
DEFAULT_VARIANT = "lang-id-voxlingua107-ecapa"

ARCH = "ecapa_tdnn"

# Architecture constants. Every one of these is ASSERTED against the live
# module tree below, never assumed — see `assert_architecture`.
SAMPLE_RATE = 16000
N_FFT = 400
WIN = 400
HOP = 160
N_STFT = N_FFT // 2 + 1  # 201
N_MELS = 60
LOG_FLOOR = 1e-10
TOP_DB = 80.0

CHANNELS = [1024, 1024, 1024, 1024, 3072]
KERNEL_SIZES = [5, 3, 3, 3, 1]
DILATIONS = [1, 2, 3, 4, 1]
RES2NET_SCALE = 8
SE_CHANNELS = 128
ATTENTION_CHANNELS = 128
ASP_EPS = 1e-12
EMB_DIM = 256
HIDDEN = 512
N_LABELS = 107
LEAKY_SLOPE = 0.01
BN_EPS = 1e-5

# Measured on the pinned checkpoint.
N_PARAMS_EMBEDDING = 21_058_432
N_PARAMS_CLASSIFIER = 188_011

# `he`/`jv`/`fil`/`nb` are the modern ISO codes users type; the VoxLingua107
# label set predates them and uses the legacy forms. Stored as metadata so
# the LANGID role resolves them without a per-family table.
LABEL_ALIASES = ["he=iw", "jv=jw", "fil=tl", "nb=no"]


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def resolve_checkpoint_dir(model_arg: str, revision: str | None) -> Path:
    """Return a local SpeechBrain checkpoint directory (HF cache or local)."""
    p = Path(model_arg).expanduser()
    if not p.is_dir() and not looks_like_repo_id(model_arg):
        raise SystemExit(
            f"error: {model_arg!r} is neither an existing directory nor an "
            f"org/name Hugging Face repo id"
        )
    src = resolve_model_dir(model_arg, revision)
    needed = ["hyperparams.yaml", "embedding_model.ckpt",
              "classifier.ckpt", "label_encoder.txt"]
    missing = [f for f in needed if not (src / f).exists()]
    if missing:
        raise SystemExit(f"error: {src} is missing {missing}")
    return src


def load_reference(src: Path, device: str = "cpu"):
    """Build the SpeechBrain `EncoderClassifier`, in eval mode, fp32.

    `overrides={"pretrained_path": <local dir>}` is the load-bearing line:
    `hyperparams.yaml` sets `pretrained_path` to the HF repo id and
    `Pretrainer.collect_files` splits every `paths[...]` entry into
    (source, filename), so without the override the .ckpt files are
    re-fetched from the HF `main` branch and `--revision` means nothing.

    `savedir != source` because SpeechBrain symlinks the collected
    loadables into savedir; aiming it at the snapshot would symlink files
    onto themselves.
    """
    from speechbrain.inference.classifiers import EncoderClassifier

    torch.manual_seed(0)
    torch.set_num_threads(1)

    savedir = REPO_ROOT / "build" / "speechbrain" / ARCH
    savedir.mkdir(parents=True, exist_ok=True)

    print(f"Loading EncoderClassifier: source={src} savedir={savedir}")
    clf = EncoderClassifier.from_hparams(
        source=str(src),
        savedir=str(savedir),
        overrides={"pretrained_path": str(src)},
        run_opts={"device": device},
    )
    clf.mods.eval()
    for p in clf.mods.parameters():
        p.requires_grad_(False)

    training = [n for n, m in clf.mods.named_modules() if m.training]
    if training:
        raise SystemExit(f"error: modules still in training mode: {training[:5]}")

    dtypes = {str(p.dtype) for p in clf.mods.parameters()}
    if dtypes != {"torch.float32"}:
        raise SystemExit(f"error: expected an fp32 model, got dtypes {dtypes}")

    return clf


# ---------------------------------------------------------------------------
# Architecture assertions
# ---------------------------------------------------------------------------


class ArchError(SystemExit):
    """Raised (as exit 2) when the checkpoint is not the model we expect."""

    def __init__(self, msg: str) -> None:
        super().__init__(2)
        self.msg = msg


_checks: list[str] = []


def check(cond: bool, what: str, got: Any = None) -> None:
    """Assert one architectural fact, recording it for the summary."""
    if not cond:
        detail = f" (got {got!r})" if got is not None else ""
        print(f"\nERROR: architecture mismatch: {what}{detail}", file=sys.stderr)
        print("This converter only handles SpeechBrain's VoxLingua107 "
              "ECAPA-TDNN as pinned in scripts/envs/ecapa_tdnn.", file=sys.stderr)
        raise ArchError(what)
    _checks.append(what)


def bn_of(module) -> torch.nn.BatchNorm1d:
    """Resolve a SpeechBrain BatchNorm1d wrapper to the torch module.

    The wrapper depth is NOT uniform in this checkpoint: TDNN blocks and the
    classifier reach the torch layer at `<block>.norm.norm`, while
    `embedding_model.asp_bn` is itself the wrapper and its torch layer is
    `asp_bn.norm`. Walking to the first real `torch.nn.BatchNorm1d` instead
    of hardcoding a depth is what stops a one-level slip from silently
    producing plausible-looking garbage.
    """
    m = module
    for _ in range(4):
        if isinstance(m, torch.nn.BatchNorm1d):
            return m
        if hasattr(m, "norm"):
            m = m.norm
            continue
        break
    raise ArchError(f"no torch BatchNorm1d under {type(module).__name__}")


def assert_conv(conv, oc: int, ic: int, k: int, d: int, label: str) -> None:
    """Assert one SpeechBrain Conv1d: shape, dilation, padding, bias."""
    w = conv.conv.weight
    check(tuple(w.shape) == (oc, ic, k), f"{label}.weight is [{oc}, {ic}, {k}]",
          tuple(w.shape))
    check(conv.conv.bias is not None, f"{label} has a bias")
    check(int(conv.dilation) == d, f"{label}.dilation == {d}", conv.dilation)
    check(conv.padding == "same", f"{label}.padding == 'same'", conv.padding)
    check(conv.padding_mode == "reflect", f"{label}.padding_mode == 'reflect'",
          conv.padding_mode)
    check(int(conv.conv.groups) == 1, f"{label}.groups == 1", conv.conv.groups)


def assert_tdnn_block(blk, oc: int, ic: int, k: int, d: int, label: str) -> None:
    """conv -> ReLU -> BN(eps 1e-5) -> Dropout1d(p=0) (identity in eval)."""
    assert_conv(blk.conv, oc, ic, k, d, f"{label}.conv")
    check(isinstance(blk.activation, torch.nn.ReLU),
          f"{label}.activation is ReLU", type(blk.activation).__name__)
    bn = bn_of(blk.norm)
    check(int(bn.num_features) == oc, f"{label}.norm has {oc} features",
          bn.num_features)
    check(abs(float(bn.eps) - BN_EPS) < 1e-12, f"{label}.norm.eps == 1e-5", bn.eps)
    check(bn.running_mean is not None and bn.running_var is not None,
          f"{label}.norm carries running statistics")
    # SpeechBrain 1.1.1 added a Dropout1d child (oracle report D3). At p=0 in
    # eval mode it is an exact identity, so the C++ graph ignores it — but a
    # nonzero p would mean this checkpoint is not the one we modelled.
    check(float(blk.dropout.p) == 0.0, f"{label}.dropout.p == 0", blk.dropout.p)


def assert_architecture(clf) -> dict[str, Any]:
    """Verify every hyperparameter the C++ graph assumes against live modules.

    Returns the label table (codes/names) as a side effect of validating it.
    """
    em = clf.mods.embedding_model
    cl = clf.mods.classifier
    fe = clf.mods.compute_features
    mvn = clf.mods.mean_var_norm

    # ---- front end -------------------------------------------------------
    st = fe.compute_STFT
    check(int(st.n_fft) == N_FFT, "Fbank STFT n_fft == 400", st.n_fft)
    check(int(st.win_length) == WIN, "Fbank STFT win_length == 400 samples "
          "(25 ms)", st.win_length)
    check(int(st.hop_length) == HOP, "Fbank STFT hop_length == 160 samples "
          "(10 ms)", st.hop_length)
    check(int(st.sample_rate) == SAMPLE_RATE, "Fbank sample_rate == 16000",
          st.sample_rate)
    check(bool(st.center) is True, "Fbank STFT center == True", st.center)
    check(st.pad_mode == "constant", "Fbank STFT pad_mode == 'constant' "
          "(zero pad 200 each side)", st.pad_mode)
    check(bool(st.onesided) is True, "Fbank STFT onesided == True", st.onesided)
    check(bool(st.normalized_stft) is False, "Fbank STFT normalized == False",
          st.normalized_stft)
    check(torch.equal(st.window.to(torch.float32).cpu(),
                      torch.hamming_window(WIN)),
          "Fbank STFT window == torch.hamming_window(400) (periodic)")

    fb = fe.compute_fbanks
    check(int(fb.n_mels) == N_MELS, "Filterbank n_mels == 60", fb.n_mels)
    check(fb.filter_shape == "triangular", "Filterbank filter_shape == "
          "'triangular'", fb.filter_shape)
    check(float(fb.f_min) == 0.0, "Filterbank f_min == 0", fb.f_min)
    check(float(fb.f_max) == 8000.0, "Filterbank f_max == 8000", fb.f_max)
    check(int(fb.n_stft) == N_STFT, "Filterbank n_stft == 201", fb.n_stft)
    check(int(fb.power_spectrogram) == 2, "Filterbank power_spectrogram == 2 "
          "(multiplier 10)", fb.power_spectrogram)
    check(int(fb.multiplier) == 10, "Filterbank dB multiplier == 10",
          fb.multiplier)
    check(float(fb.amin) == LOG_FLOOR, "Filterbank amin == 1e-10", fb.amin)
    check(float(fb.ref_value) == 1.0, "Filterbank ref_value == 1.0 "
          "(db_multiplier 0, no reference term)", fb.ref_value)
    check(float(fb.db_multiplier) == 0.0, "Filterbank db_multiplier == 0",
          fb.db_multiplier)
    check(float(fb.top_db) == TOP_DB, "Filterbank top_db == 80", fb.top_db)
    check(bool(fb.log_mel) is True, "Filterbank log_mel == True", fb.log_mel)
    check(bool(fb.freeze) is True, "Filterbank freeze == True (filters are "
          "constants, not parameters)", fb.freeze)
    check(float(fb.param_rand_factor) == 0.0,
          "Filterbank param_rand_factor == 0", fb.param_rand_factor)
    check(bool(fe.deltas) is False, "Fbank deltas == False", fe.deltas)
    check(bool(fe.context) is False, "Fbank context window == False", fe.context)

    check(mvn.norm_type == "sentence", "InputNormalization norm_type == "
          "'sentence'", mvn.norm_type)
    check(bool(mvn.std_norm) is False, "InputNormalization std_norm == False "
          "(mean subtraction only)", mvn.std_norm)
    check(int(mvn.length_dim) == 1, "InputNormalization length_dim == 1 "
          "(per-bin mean over time)", mvn.length_dim)
    check(bool(mvn.avoid_padding_norm) is False,
          "InputNormalization avoid_padding_norm == False", mvn.avoid_padding_norm)

    # ---- embedding network ----------------------------------------------
    check(len(em.blocks) == 4, "ECAPA_TDNN has 4 blocks (1 TDNN + 3 SERes2Net)",
          len(em.blocks))

    assert_tdnn_block(em.blocks[0], CHANNELS[0], N_MELS, KERNEL_SIZES[0],
                      DILATIONS[0], "blocks.0")

    for i in (1, 2, 3):
        b = em.blocks[i]
        pfx = f"blocks.{i}"
        check(b.shortcut is None, f"{pfx} has no shortcut projection "
              "(in_channels == out_channels)")
        assert_tdnn_block(b.tdnn1, CHANNELS[i], CHANNELS[i - 1], 1, 1,
                          f"{pfx}.tdnn1")
        r2 = b.res2net_block
        check(int(r2.scale) == RES2NET_SCALE, f"{pfx}.res2net scale == 8",
              r2.scale)
        check(len(r2.blocks) == RES2NET_SCALE - 1,
              f"{pfx}.res2net has 7 sub-blocks", len(r2.blocks))
        width = CHANNELS[i] // RES2NET_SCALE
        check(width == 128, f"{pfx}.res2net chunk width == 128", width)
        for j in range(RES2NET_SCALE - 1):
            assert_tdnn_block(r2.blocks[j], width, width, KERNEL_SIZES[i],
                              DILATIONS[i], f"{pfx}.res2net.{j}")
        assert_tdnn_block(b.tdnn2, CHANNELS[i], CHANNELS[i], 1, 1,
                          f"{pfx}.tdnn2")
        se = b.se_block
        assert_conv(se.conv1, SE_CHANNELS, CHANNELS[i], 1, 1, f"{pfx}.se.conv1")
        assert_conv(se.conv2, CHANNELS[i], SE_CHANNELS, 1, 1, f"{pfx}.se.conv2")
        check(isinstance(se.relu, torch.nn.ReLU), f"{pfx}.se activation is ReLU",
              type(se.relu).__name__)
        check(isinstance(se.sigmoid, torch.nn.Sigmoid),
              f"{pfx}.se gate is Sigmoid", type(se.sigmoid).__name__)

    mfa_in = CHANNELS[3] * 3
    assert_tdnn_block(em.mfa, CHANNELS[4], mfa_in, KERNEL_SIZES[4],
                      DILATIONS[4], "mfa")

    asp = em.asp
    check(bool(asp.global_context) is True, "asp global_context == True",
          asp.global_context)
    check(float(asp.eps) == ASP_EPS, "asp eps == 1e-12", asp.eps)
    assert_tdnn_block(asp.tdnn, ATTENTION_CHANNELS, CHANNELS[4] * 3, 1, 1,
                      "asp.tdnn")
    check(isinstance(asp.tanh, torch.nn.Tanh), "asp gate activation is Tanh",
          type(asp.tanh).__name__)
    assert_conv(asp.conv, CHANNELS[4], ATTENTION_CHANNELS, 1, 1, "asp.conv")

    asp_bn = bn_of(em.asp_bn)
    check(int(asp_bn.num_features) == CHANNELS[4] * 2,
          "asp_bn is BatchNorm1d(6144)", asp_bn.num_features)
    check(abs(float(asp_bn.eps) - BN_EPS) < 1e-12, "asp_bn.eps == 1e-5",
          asp_bn.eps)
    assert_conv(em.fc, EMB_DIM, CHANNELS[4] * 2, 1, 1, "fc")

    # ---- classifier ------------------------------------------------------
    check(isinstance(cl.act, torch.nn.LeakyReLU),
          "classifier.act is LeakyReLU", type(cl.act).__name__)
    check(float(cl.act.negative_slope) == LEAKY_SLOPE,
          "classifier.act slope == 0.01", cl.act.negative_slope)
    bn0 = bn_of(cl.norm)
    check(int(bn0.num_features) == EMB_DIM, "classifier bn0 is BatchNorm1d(256)",
          bn0.num_features)
    check(abs(float(bn0.eps) - BN_EPS) < 1e-12, "classifier bn0.eps == 1e-5",
          bn0.eps)
    check(len(cl.DNN) == 1, "classifier has 1 linear block", len(cl.DNN))
    blk = cl.DNN.block_0
    check(tuple(blk.linear.w.weight.shape) == (HIDDEN, EMB_DIM),
          "cls.l1 weight is [512, 256]", tuple(blk.linear.w.weight.shape))
    check(isinstance(blk.act, torch.nn.LeakyReLU),
          "classifier hidden activation is LeakyReLU", type(blk.act).__name__)
    check(float(blk.act.negative_slope) == LEAKY_SLOPE,
          "classifier hidden slope == 0.01", blk.act.negative_slope)
    bn1 = bn_of(blk.norm)
    check(int(bn1.num_features) == HIDDEN, "classifier bn1 is BatchNorm1d(512)",
          bn1.num_features)
    check(abs(float(bn1.eps) - BN_EPS) < 1e-12, "classifier bn1.eps == 1e-5",
          bn1.eps)
    check(tuple(cl.out.w.weight.shape) == (N_LABELS, HIDDEN),
          "cls.out weight is [107, 512]", tuple(cl.out.w.weight.shape))

    # ---- parameter counts ------------------------------------------------
    n_emb = sum(int(p.numel()) for p in em.parameters())
    n_cls = sum(int(p.numel()) for p in cl.parameters())
    check(n_emb == N_PARAMS_EMBEDDING,
          f"embedding network has {N_PARAMS_EMBEDDING:,} parameters", n_emb)
    check(n_cls == N_PARAMS_CLASSIFIER,
          f"classifier has {N_PARAMS_CLASSIFIER:,} parameters", n_cls)

    # ---- labels ----------------------------------------------------------
    ind2lab = clf.hparams.label_encoder.ind2lab
    check(len(ind2lab) == N_LABELS, f"label encoder has {N_LABELS} labels",
          len(ind2lab))
    codes: list[str] = []
    names: list[str] = []
    for i in range(N_LABELS):
        raw = str(ind2lab[i])
        if ": " not in raw:
            raise ArchError(f"label {i} ({raw!r}) has no '<code>: <name>' form")
        code, name = raw.split(": ", 1)
        codes.append(code.strip())
        names.append(name.strip())
    check(len(set(codes)) == N_LABELS, "all 107 label codes are distinct",
          len(set(codes)))
    for alias in LABEL_ALIASES:
        a, c = alias.split("=", 1)
        check(c in codes, f"alias target {c!r} is in the label set")
        check(a not in codes, f"alias name {a!r} is NOT already a label code")

    print(f"Architecture verified: {len(_checks)} checks passed.")
    print(f"  front end  : STFT {N_FFT}/{WIN}/{HOP} periodic-hamming, center, "
          f"zero pad; {N_MELS} triangular mels 0-8000 Hz; "
          f"10*log10(max(x,{LOG_FLOOR:g})); top_db {TOP_DB:g}; sentence-mean norm")
    print(f"  encoder    : channels {CHANNELS} kernels {KERNEL_SIZES} "
          f"dilations {DILATIONS}; res2net scale {RES2NET_SCALE} "
          f"({RES2NET_SCALE - 1} blocks of {CHANNELS[1] // RES2NET_SCALE}->"
          f"{CHANNELS[1] // RES2NET_SCALE} k={KERNEL_SIZES[1]}); "
          f"SE {SE_CHANNELS}; attention {ATTENTION_CHANNELS}; "
          f"global_context True; asp eps {ASP_EPS:g}; embedding {EMB_DIM}")
    print(f"  classifier : LeakyReLU({LEAKY_SLOPE}) -> BN(256) -> "
          f"Linear(256->{HIDDEN}) -> LeakyReLU -> BN({HIDDEN}) -> "
          f"Linear({HIDDEN}->{N_LABELS}) -> log_softmax")
    print(f"  every conv : padding='same', padding_mode='reflect', groups=1, "
          f"bias=True; every BN eps={BN_EPS:g}")
    print(f"  parameters : embedding {n_emb:,} + classifier {n_cls:,} = "
          f"{n_emb + n_cls:,}")
    print(f"  labels     : {N_LABELS} codes, aliases {LABEL_ALIASES}")

    return {"codes": codes, "names": names}


# ---------------------------------------------------------------------------
# Filterbank capture
# ---------------------------------------------------------------------------


def capture_fbank_matrix(clf) -> np.ndarray:
    """Return the exact [201, 60] fbank matrix the forward pass builds.

    SpeechBrain 1.1.1's `Filterbank` has no `fbank_matrix` attribute or
    buffer. The matrix is constructed
    inside `forward()` by `_create_fbank_matrix(f_central_mat, band_mat)`.

    So we wrap that method, push 0.1 s of silence through the real Fbank,
    and keep what it returned. This is stronger than recomputing the
    triangle geometry in float64: a rebuild matches to only ~1.2e-05
    because `f_central` / `band` are fp32 buffers, whereas the capture is
    bit-identical to what the reference used.
    """
    fbanks = clf.mods.compute_features.compute_fbanks
    original = fbanks._create_fbank_matrix
    captured: dict[str, Any] = {"calls": 0}

    def capture(f_central_mat, band_mat):
        m = original(f_central_mat, band_mat)
        captured.setdefault("value", m)
        captured["calls"] += 1
        return m

    fbanks._create_fbank_matrix = capture
    try:
        with torch.inference_mode():
            clf.mods.compute_features(torch.zeros(1, SAMPLE_RATE // 10))
    finally:
        fbanks._create_fbank_matrix = original

    if captured["calls"] != 1:
        raise ArchError(f"_create_fbank_matrix fired {captured['calls']} times, "
                        "expected 1")
    m = captured["value"]
    check(tuple(m.shape) == (N_STFT, N_MELS),
          f"fbank matrix is [{N_STFT}, {N_MELS}]", tuple(m.shape))
    return np.ascontiguousarray(
        m.detach().to(dtype=torch.float32, device="cpu").numpy())


# ---------------------------------------------------------------------------
# Weight extraction / conversion-time algebra
# ---------------------------------------------------------------------------


def w2d(conv) -> np.ndarray:
    """1x1 SpeechBrain Conv1d weight `[OC, IC, 1]` -> numpy `[OC, IC]` (f64)."""
    w = conv.conv.weight.detach().to(torch.float64).cpu().numpy()
    if w.shape[2] != 1:
        raise ArchError(f"expected a 1x1 kernel, got K={w.shape[2]}")
    return np.ascontiguousarray(w[:, :, 0])


def w3d(conv) -> np.ndarray:
    """k>1 Conv1d weight `[OC, IC, K]` -> TAP-MAJOR numpy `[K, OC, IC]` (f64).

    ggml sees `ne = [IC, OC, K]`, so tap `t` is the contiguous `[IC, OC]`
    matrix the graph multiplies against a `t*dilation`-shifted view of the
    reflect-padded activation.
    """
    w = conv.conv.weight.detach().to(torch.float64).cpu().numpy()
    return np.ascontiguousarray(np.transpose(w, (2, 0, 1)))


def bias(conv) -> np.ndarray:
    return np.ascontiguousarray(
        conv.conv.bias.detach().to(torch.float64).cpu().numpy())


def bn_affine(module) -> tuple[np.ndarray, np.ndarray]:
    """BatchNorm -> (scale, shift) with `y = x * scale + shift`.

    `scale = gamma / sqrt(var + eps)`, `shift = beta - mean * scale`,
    computed in float64. This is NOT folded into the preceding conv: a
    TDNNBlock is conv -> ReLU -> BN, and the ReLU makes a backward fold
    impossible.
    """
    bn = bn_of(module)
    g = bn.weight.detach().to(torch.float64).cpu().numpy()
    b = bn.bias.detach().to(torch.float64).cpu().numpy()
    m = bn.running_mean.detach().to(torch.float64).cpu().numpy()
    v = bn.running_var.detach().to(torch.float64).cpu().numpy()
    scale = g / np.sqrt(v + float(bn.eps))
    shift = b - m * scale
    return np.ascontiguousarray(scale), np.ascontiguousarray(shift)


def fold_bn_into_linear(module, w: np.ndarray, b: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Fold a BatchNorm FORWARD into the linear map that consumes it.

    `y = W(x * s + t) + b = (W * diag(s)) x + (W t + b)`. Valid only when
    nothing sits between the BN and the linear map — true for
    `asp_bn -> fc`, `cls.bn0 -> cls.l1` and `cls.bn1 -> cls.out`. Note the
    classifier's BNs come AFTER their LeakyReLU, so the activation is still
    applied before each folded linear at run time; only the BN disappears.
    """
    s, t = bn_affine(module)
    if w.shape[1] != s.size:
        raise ArchError(f"fold shape mismatch: W is {w.shape}, BN has {s.size} "
                        "features")
    return (np.ascontiguousarray(w * s[None, :]),
            np.ascontiguousarray(w @ t + b))


# ---------------------------------------------------------------------------
# GGUF writing
# ---------------------------------------------------------------------------


def ggml_ne(shape: tuple[int, ...]) -> list[int]:
    """numpy shape -> ggml ne (fastest axis first)."""
    return list(reversed(shape))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(clf, out_path: Path, *, variant: str, repo_id: str,
            revision: str, filters_201x60: np.ndarray,
            labels: dict[str, list[str]]) -> dict[str, Any]:
    em = clf.mods.embedding_model
    cl = clf.mods.classifier

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf_writer(str(out_path), ARCH)

    # ----- general.* ------------------------------------------------------
    add_general_identity(
        writer,
        name=repo_id,
        basename=slug_from_repo_id(repo_id),
        size_label="21M",
        file_type=int(LlamaFileType.ALL_F32),
        license="apache-2.0",
        license_name="Apache License 2.0",
        license_link="https://www.apache.org/licenses/LICENSE-2.0",
        author="SpeechBrain",
        organization="speechbrain",
        repo_url=f"https://huggingface.co/{repo_id}",
        source_url=f"https://huggingface.co/{repo_id}",
        languages=labels["codes"],
        tags=["language-identification", "ecapa-tdnn", "voxlingua107",
              "speechbrain"],
        description=(
            "SpeechBrain ECAPA-TDNN spoken-language identification trained "
            "on VoxLingua107 (107 languages). Converted from "
            f"{repo_id} for transcribe.cpp."
        ),
    )
    # `general.source.commit` has no helper in gguf-py; write it plainly so
    # every artifact records the exact upstream snapshot it came from.
    writer.add_string("general.source.commit", revision)

    # ----- stt.* ----------------------------------------------------------
    writer.add_string("stt.variant", variant)

    # SpeechBrain Fbank + InputNormalization: zero-padded centered STFT,
    # periodic Hamming, power spectrum, mel, 10*log10(max(p, log_clamp_min)),
    # top_db floor over the clip, per-bin mean subtraction.
    writer.add_string("stt.frontend.type", "speechbrain_fbank")
    writer.add_uint32("stt.frontend.sample_rate", SAMPLE_RATE)
    writer.add_uint32("stt.frontend.n_fft", N_FFT)
    writer.add_uint32("stt.frontend.hop_length", HOP)
    writer.add_uint32("stt.frontend.win_length", WIN)
    writer.add_uint32("stt.frontend.num_mels", N_MELS)
    writer.add_string("stt.frontend.window", "hamming_periodic")
    writer.add_string("stt.frontend.pad_mode", "constant")
    writer.add_float32("stt.frontend.log_clamp_min", LOG_FLOOR)
    writer.add_float32("stt.frontend.top_db", TOP_DB)
    writer.add_string("stt.frontend.normalize", canonicalize_normalize("sentence_mean"))

    writer.add_array("stt.ecapa_tdnn.channels", CHANNELS)
    writer.add_array("stt.ecapa_tdnn.kernel_sizes", KERNEL_SIZES)
    writer.add_array("stt.ecapa_tdnn.dilations", DILATIONS)
    writer.add_uint32("stt.ecapa_tdnn.res2net_scale", RES2NET_SCALE)
    writer.add_uint32("stt.ecapa_tdnn.se_channels", SE_CHANNELS)
    writer.add_uint32("stt.ecapa_tdnn.attention_channels", ATTENTION_CHANNELS)
    writer.add_float32("stt.ecapa_tdnn.asp_eps", ASP_EPS)
    writer.add_uint32("stt.ecapa_tdnn.embedding_dim", EMB_DIM)
    writer.add_uint32("stt.ecapa_tdnn.classifier_hidden", HIDDEN)
    writer.add_float32("stt.ecapa_tdnn.classifier_leaky_slope", LEAKY_SLOPE)

    writer.add_array("stt.langid.labels.codes", labels["codes"])
    writer.add_array("stt.langid.labels.names", labels["names"])
    writer.add_array("stt.langid.labels.aliases", LABEL_ALIASES)

    # ----- tensors --------------------------------------------------------
    rows: list[tuple[str, str, str, str, int]] = []
    total_bytes = 0

    def add(name: str, arr: np.ndarray) -> None:
        nonlocal total_bytes
        a = np.ascontiguousarray(arr.astype(np.float32))
        t = GGMLQuantizationType.F32
        encoded, raw_dtype = encode_for_gguf(a, t)
        writer.add_tensor(name, encoded, raw_dtype=raw_dtype)
        nbytes = int(encoded.nbytes)
        rows.append((name, str(list(a.shape)), str(ggml_ne(a.shape)),
                     t.name, nbytes))
        total_bytes += nbytes

    # Front-end filterbank, mel-major so each filter's 201 taps are
    # contiguous: numpy [60, 201] -> ggml ne = [201, 60].
    add("frontend.mel_filterbank", filters_201x60.T)

    # blocks.0: the only k>1 conv outside the res2net stacks.
    b0 = em.blocks[0]
    add("blk.0.conv.weight", w3d(b0.conv))
    add("blk.0.conv.bias", bias(b0.conv))
    s, t = bn_affine(b0.norm)
    add("blk.0.bn.scale", s)
    add("blk.0.bn.shift", t)

    for i in (1, 2, 3):
        b = em.blocks[i]
        p = f"blk.{i}"

        add(f"{p}.tdnn1.weight", w2d(b.tdnn1.conv))
        add(f"{p}.tdnn1.bias", bias(b.tdnn1.conv))
        s, t = bn_affine(b.tdnn1.norm)
        add(f"{p}.tdnn1.bn.scale", s)
        add(f"{p}.tdnn1.bn.shift", t)

        # res2.{j} operates on chunk j+1 of the 8 channel chunks: chunk 0 is
        # passed through untouched and each later chunk is summed with the
        # PREVIOUS sub-block's output before being convolved.
        for j in range(RES2NET_SCALE - 1):
            sub = b.res2net_block.blocks[j]
            add(f"{p}.res2.{j}.conv.weight", w3d(sub.conv))
            add(f"{p}.res2.{j}.conv.bias", bias(sub.conv))
            s, t = bn_affine(sub.norm)
            add(f"{p}.res2.{j}.bn.scale", s)
            add(f"{p}.res2.{j}.bn.shift", t)

        add(f"{p}.tdnn2.weight", w2d(b.tdnn2.conv))
        add(f"{p}.tdnn2.bias", bias(b.tdnn2.conv))
        s, t = bn_affine(b.tdnn2.norm)
        add(f"{p}.tdnn2.bn.scale", s)
        add(f"{p}.tdnn2.bn.shift", t)

        add(f"{p}.se.c1.weight", w2d(b.se_block.conv1))
        add(f"{p}.se.c1.bias", bias(b.se_block.conv1))
        add(f"{p}.se.c2.weight", w2d(b.se_block.conv2))
        add(f"{p}.se.c2.bias", bias(b.se_block.conv2))

    # MFA split: SpeechBrain concatenates blocks.1/2/3 along channels before a
    # 1x1 conv, so the [3072, 3072] weight is three [3072, 1024] blocks in
    # block order. Summing three matmuls costs the same FLOPs and skips the
    # concat copy.
    mfa_w = w2d(em.mfa.conv)
    for k, name in enumerate(("mfa.w1.weight", "mfa.w2.weight", "mfa.w3.weight")):
        add(name, mfa_w[:, k * CHANNELS[3]:(k + 1) * CHANNELS[3]])
    add("mfa.bias", bias(em.mfa.conv))
    s, t = bn_affine(em.mfa.norm)
    add("mfa.bn.scale", s)
    add("mfa.bn.shift", t)

    # ASP split: the attention TDNN consumes cat([x, mean, std]) (that exact
    # order — see AttentiveStatisticsPooling.forward), so its [128, 9216]
    # weight is three [128, 3072] blocks. mean/std are constant over time, so
    # the graph folds wm@mean + ws@std + b into one [128] vector.
    asp_w = w2d(em.asp.tdnn.conv)
    for k, name in enumerate(("asp.tdnn.x.weight", "asp.tdnn.mean.weight", "asp.tdnn.std.weight")):
        add(name, asp_w[:, k * CHANNELS[4]:(k + 1) * CHANNELS[4]])
    add("asp.tdnn.bias", bias(em.asp.tdnn.conv))
    s, t = bn_affine(em.asp.tdnn.norm)
    add("asp.tdnn.bn.scale", s)
    add("asp.tdnn.bn.shift", t)

    add("asp.attn.weight", w2d(em.asp.conv))
    add("asp.attn.bias", bias(em.asp.conv))

    # asp_bn folds forward into fc (nothing between them).
    fc_w, fc_b = fold_bn_into_linear(em.asp_bn, w2d(em.fc), bias(em.fc))
    add("fc.weight", fc_w)
    add("fc.bias", fc_b)

    # cls.bn0 folds into cls.l1, cls.bn1 into cls.out. The LeakyReLUs sit
    # BEFORE each BN, so they survive in the graph: the run-time chain is
    # LeakyReLU -> cls.l1 -> LeakyReLU -> cls.out.
    l1 = cl.DNN.block_0.linear.w
    l1_w = l1.weight.detach().to(torch.float64).cpu().numpy()
    l1_b = l1.bias.detach().to(torch.float64).cpu().numpy()
    l1_w, l1_b = fold_bn_into_linear(cl.norm, l1_w, l1_b)
    add("cls.l1.weight", l1_w)
    add("cls.l1.bias", l1_b)

    out = cl.out.w
    out_w = out.weight.detach().to(torch.float64).cpu().numpy()
    out_b = out.bias.detach().to(torch.float64).cpu().numpy()
    out_w, out_b = fold_bn_into_linear(cl.DNN.block_0.norm, out_w, out_b)
    add("cls.out.weight", out_w)
    add("cls.out.bias", out_b)

    # 1 filterbank + 4 blk.0 + 3*40 SERes2Net + 6 mfa + 6 asp.tdnn
    # + 2 asp.attn + 2 fc + 2 cls.l1 + 2 cls.out
    expected = 1 + 4 + 3 * 40 + 6 + 6 + 2 + 2 + 2 + 2
    if len(rows) != expected:
        raise ArchError(f"tensor count mismatch: emitted {len(rows)}, "
                        f"expected {expected}")

    # ----- table ----------------------------------------------------------
    w_name = max(len(r[0]) for r in rows)
    w_np = max(len(r[1]) for r in rows)
    w_ne = max(len(r[2]) for r in rows)
    print(f"\n{'tensor'.ljust(w_name)}  {'numpy'.ljust(w_np)}  "
          f"{'ggml ne'.ljust(w_ne)}  {'dtype':6}  {'bytes':>12}")
    print("-" * (w_name + w_np + w_ne + 6 + 12 + 8))
    for name, npshape, ne, dt, nb in rows:
        print(f"{name.ljust(w_name)}  {npshape.ljust(w_np)}  {ne.ljust(w_ne)}  "
              f"{dt:6}  {nb:>12,}")
    print("-" * (w_name + w_np + w_ne + 6 + 12 + 8))
    print(f"{len(rows)} tensors, {total_bytes:,} bytes of tensor data "
          f"({total_bytes / (1024 * 1024):.2f} MiB)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = out_path.stat().st_size
    digest = sha256_file(out_path)
    print(f"Wrote {out_path} ({size:,} bytes, {size / (1024 * 1024):.2f} MiB)")
    print(f"sha256: {digest}")

    return {
        "family": ARCH,
        "variant": variant,
        "refdtype": "F32",
        "gguf_path": str(out_path.relative_to(REPO_ROOT))
        if out_path.is_relative_to(REPO_ROOT) else str(out_path),
        "gguf_bytes": int(size),
        "gguf_sha256": digest,
        "source_hf_repo": repo_id,
        "source_hf_revision": revision,
        "converter_script": "scripts/convert-ecapa_tdnn.py",
        "n_tensors": len(rows),
        "tensor_data_bytes": int(total_bytes),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert SpeechBrain's VoxLingua107 ECAPA-TDNN checkpoint "
                    "to the F32 reference GGUF.")
    p.add_argument("model", nargs="?", default=DEFAULT_REPO,
                   help=f"HF repo id or local checkpoint dir "
                        f"(default: {DEFAULT_REPO})")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (derived from the repo id when "
                        "omitted)")
    p.add_argument("--revision", default=DEFAULT_REVISION,
                   help="HF revision (commit SHA) to pin the download to")
    p.add_argument("--repo-id", default=None,
                   help="HF repo id used for the output slug and metadata "
                        "when converting from a local path")
    args = p.parse_args(argv)

    repo_id = args.repo_id or (args.model if looks_like_repo_id(args.model)
                               else None)
    if repo_id is None:
        print("error: cannot infer the source repo id from a local path; "
              "pass --repo-id", file=sys.stderr)
        return 2

    src = resolve_checkpoint_dir(args.model, args.revision)
    slug = slug_from_repo_id(repo_id)
    out_path = args.out_path or (REPO_ROOT / "models" / slug /
                                 gguf_name(slug, "F32"))

    clf = load_reference(src)
    labels = assert_architecture(clf)
    filters = capture_fbank_matrix(clf)

    import speechbrain

    report = convert(clf, out_path, variant=DEFAULT_VARIANT,
                     repo_id=repo_id, revision=args.revision,
                     filters_201x60=filters, labels=labels)
    report["speechbrain_version"] = speechbrain.__version__
    report["torch_version"] = torch.__version__

    rep_dir = REPO_ROOT / "reports" / "convert"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_path = rep_dir / f"{slug}-F32.json"
    rep_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {rep_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArchError as e:
        print(f"error: {e.msg}", file=sys.stderr)
        raise
