#!/usr/bin/env python3
"""
convert-ecapa_tdnn.py - convert SpeechBrain's VoxLingua107 ECAPA-TDNN
language-ID checkpoint into the F32 reference GGUF of the ecapa_tdnn family.
Loads the model through SpeechBrain (the only implementation), asserts the
hyperparameters hard-coded below against the live modules, and writes the
tensor layout of src/arch/ecapa_tdnn/weights.h. Quantize afterwards with
tools/transcribe-quantize. See docs/porting/families/ecapa_tdnn.md.

    uv run --project scripts/envs/ecapa_tdnn \
      scripts/convert-ecapa_tdnn.py speechbrain/lang-id-voxlingua107-ecapa
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# XET-backed transfers occasionally stall on this checkpoint's small .ckpt
# files. Set before huggingface_hub is imported (via speechbrain).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from gguf import GGMLQuantizationType, LlamaFileType

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

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

# Written into the GGUF and asserted against the live modules below.
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
N_PARAMS_EMBEDDING = 21_058_432
N_PARAMS_CLASSIFIER = 188_011

# Modern ISO codes users type -> the legacy codes VoxLingua107 uses.
LABEL_ALIASES = ["he=iw", "jv=jw", "fil=tl", "nb=no"]


def fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_reference(src: Path):
    """SpeechBrain EncoderClassifier in eval mode.

    `pretrained_path` must be overridden to the local snapshot, otherwise
    SpeechBrain re-fetches the .ckpt files from the HF `main` branch and the
    pinned revision is ignored. `savedir` must differ from `src` (SpeechBrain
    symlinks the loadables into it).
    """
    from speechbrain.inference.classifiers import EncoderClassifier

    savedir = REPO_ROOT / "build" / "speechbrain" / ARCH
    savedir.mkdir(parents=True, exist_ok=True)
    clf = EncoderClassifier.from_hparams(
        source=str(src), savedir=str(savedir),
        overrides={"pretrained_path": str(src)}, run_opts={"device": "cpu"})
    clf.mods.eval()
    return clf


def bn_of(module) -> torch.nn.BatchNorm1d:
    """Walk SpeechBrain's BatchNorm1d wrapper(s) to the torch layer (the
    wrapper depth differs between the TDNN blocks and `asp_bn`)."""
    m = module
    for _ in range(4):
        if isinstance(m, torch.nn.BatchNorm1d):
            return m
        m = getattr(m, "norm", None)
    fail(f"no torch BatchNorm1d under {type(module).__name__}")


def assert_architecture(clf) -> dict[str, list[str]]:
    """Check every hyperparameter the GGUF hard-codes that tensor shapes (and
    hence the C++ loader) cannot see. Returns the parsed label table."""
    em, cl = clf.mods.embedding_model, clf.mods.classifier
    fe, mvn = clf.mods.compute_features, clf.mods.mean_var_norm
    st, fb = fe.compute_STFT, fe.compute_fbanks
    asp = em.asp

    def name(m) -> str:
        return type(m).__name__

    checks = [  # (label, actual, expected)
        ("param dtypes", {p.dtype for p in clf.mods.parameters()}, {torch.float32}),
        ("stft.n_fft", st.n_fft, N_FFT),
        ("stft.win_length", st.win_length, WIN),
        ("stft.hop_length", st.hop_length, HOP),
        ("stft.sample_rate", st.sample_rate, SAMPLE_RATE),
        ("stft.center", st.center, True),
        ("stft.pad_mode", st.pad_mode, "constant"),
        ("stft.onesided", st.onesided, True),
        ("stft.normalized_stft", st.normalized_stft, False),
        ("stft.window == hamming_window(400)",
         torch.equal(st.window.to(torch.float32).cpu(), torch.hamming_window(WIN)), True),
        ("fbank.n_mels", fb.n_mels, N_MELS),
        ("fbank.filter_shape", fb.filter_shape, "triangular"),
        ("fbank.f_min", fb.f_min, 0.0),
        ("fbank.f_max", fb.f_max, 8000.0),
        ("fbank.n_stft", fb.n_stft, N_STFT),
        ("fbank.power_spectrogram", fb.power_spectrogram, 2),
        ("fbank.multiplier", fb.multiplier, 10),
        ("fbank.amin", fb.amin, LOG_FLOOR),
        ("fbank.ref_value", fb.ref_value, 1.0),
        ("fbank.db_multiplier", fb.db_multiplier, 0.0),
        ("fbank.top_db", fb.top_db, TOP_DB),
        ("fbank.log_mel", fb.log_mel, True),
        ("fbank.freeze", fb.freeze, True),
        ("fbank.param_rand_factor", fb.param_rand_factor, 0.0),
        ("fbank.deltas", fe.deltas, False),
        ("fbank.context", fe.context, False),
        ("mvn.norm_type", mvn.norm_type, "sentence"),
        ("mvn.std_norm", mvn.std_norm, False),
        ("mvn.length_dim", mvn.length_dim, 1),
        ("mvn.avoid_padding_norm", mvn.avoid_padding_norm, False),
        ("len(blocks)", len(em.blocks), 4),
        ("asp.global_context", asp.global_context, True),
        ("asp.eps", asp.eps, ASP_EPS),
        ("asp.tanh", name(asp.tanh), "Tanh"),
        ("asp_bn.eps", bn_of(em.asp_bn).eps, BN_EPS),
        ("cls.act", name(cl.act), "LeakyReLU"),
        ("cls.act slope", cl.act.negative_slope, LEAKY_SLOPE),
        ("cls.bn0.eps", bn_of(cl.norm).eps, BN_EPS),
        ("len(cls.DNN)", len(cl.DNN), 1),
        ("cls.hidden act", name(cl.DNN.block_0.act), "LeakyReLU"),
        ("cls.hidden slope", cl.DNN.block_0.act.negative_slope, LEAKY_SLOPE),
        ("cls.bn1.eps", bn_of(cl.DNN.block_0.norm).eps, BN_EPS),
        ("embedding params", sum(p.numel() for p in em.parameters()), N_PARAMS_EMBEDDING),
        ("classifier params", sum(p.numel() for p in cl.parameters()), N_PARAMS_CLASSIFIER),
    ]

    def conv(label, c, k, d=1):
        # Every conv: kernel k, dilation d, reflect 'same' pad, bias, groups 1.
        return [
            (f"{label}.kernel", c.conv.weight.shape[2], k),
            (f"{label}.dilation", c.dilation, d),
            (f"{label}.padding", (c.padding, c.padding_mode), ("same", "reflect")),
            (f"{label}.groups", c.conv.groups, 1),
            (f"{label}.has_bias", c.conv.bias is not None, True),
        ]

    def tdnn(label, blk, k, d):
        # conv -> ReLU -> BN -> Dropout1d(p=0) (identity in eval).
        return conv(f"{label}.conv", blk.conv, k, d) + [
            (f"{label}.activation", name(blk.activation), "ReLU"),
            (f"{label}.bn.eps", bn_of(blk.norm).eps, BN_EPS),
            (f"{label}.dropout.p", blk.dropout.p, 0.0),
        ]

    checks += tdnn("blocks.0", em.blocks[0], KERNEL_SIZES[0], DILATIONS[0])
    for i in (1, 2, 3):
        b, p = em.blocks[i], f"blocks.{i}"
        r2 = b.res2net_block
        checks += [(f"{p}.shortcut", b.shortcut, None),
                   (f"{p}.res2net.scale", r2.scale, RES2NET_SCALE),
                   (f"{p}.res2net blocks", len(r2.blocks), RES2NET_SCALE - 1),
                   (f"{p}.se.relu", name(b.se_block.relu), "ReLU"),
                   (f"{p}.se.sigmoid", name(b.se_block.sigmoid), "Sigmoid")]
        checks += tdnn(f"{p}.tdnn1", b.tdnn1, 1, 1) + tdnn(f"{p}.tdnn2", b.tdnn2, 1, 1)
        for j, sub in enumerate(r2.blocks):
            checks += tdnn(f"{p}.res2net.{j}", sub, KERNEL_SIZES[i], DILATIONS[i])
        checks += conv(f"{p}.se.conv1", b.se_block.conv1, 1)
        checks += conv(f"{p}.se.conv2", b.se_block.conv2, 1)
    checks += tdnn("mfa", em.mfa, KERNEL_SIZES[4], DILATIONS[4])
    checks += tdnn("asp.tdnn", asp.tdnn, 1, 1)
    checks += conv("asp.conv", asp.conv, 1) + conv("fc", em.fc, 1)

    ind2lab = clf.hparams.label_encoder.ind2lab
    raw = [str(ind2lab[i]) for i in range(len(ind2lab))]
    pairs = [r.split(": ", 1) for r in raw]
    codes = [c.strip() for c, *_ in pairs]
    checks += [("label count", len(raw), N_LABELS),
               ("labels in '<code>: <name>' form", all(len(x) == 2 for x in pairs), True),
               ("distinct label codes", len(set(codes)), N_LABELS)]
    for alias in LABEL_ALIASES:
        a, c = alias.split("=", 1)
        checks += [(f"alias target {c!r} is a label", c in codes, True),
                   (f"alias {a!r} is not a label", a in codes, False)]

    for label, got, want in checks:
        if got != want:
            fail(f"architecture mismatch: {label}: got {got!r}, expected {want!r} "
                 "(only the pinned SpeechBrain VoxLingua107 ECAPA-TDNN is supported)")
    print(f"Architecture verified: {len(checks)} checks passed.")
    return {"codes": codes, "names": [x[1].strip() for x in pairs]}


def capture_fbank_matrix(clf) -> np.ndarray:
    """The exact [201, 60] fbank matrix SpeechBrain's forward builds.

    Filterbank has no stored matrix; it is built inside forward() by
    `_create_fbank_matrix`. Wrap that, run 0.1 s of silence, keep the result
    (bit-identical to the reference; a float64 rebuild is off by ~1e-5).
    """
    fbanks = clf.mods.compute_features.compute_fbanks
    original = fbanks._create_fbank_matrix
    captured = []

    def capture(*a):
        captured.append(original(*a))
        return captured[-1]

    fbanks._create_fbank_matrix = capture
    try:
        with torch.inference_mode():
            clf.mods.compute_features(torch.zeros(1, SAMPLE_RATE // 10))
    finally:
        fbanks._create_fbank_matrix = original
    if len(captured) != 1 or tuple(captured[0].shape) != (N_STFT, N_MELS):
        fail(f"fbank capture: {len(captured)} calls, expected one [{N_STFT}, {N_MELS}] matrix")
    return captured[0].detach().to(dtype=torch.float32, device="cpu").numpy()


def f64(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float64).cpu().numpy()


def bn_affine(module) -> tuple[np.ndarray, np.ndarray]:
    """BatchNorm -> (scale, shift), `y = x * scale + shift`, in float64."""
    bn = bn_of(module)
    scale = f64(bn.weight) / np.sqrt(f64(bn.running_var) + float(bn.eps))
    return scale, f64(bn.bias) - f64(bn.running_mean) * scale


def fold_bn_into_linear(module, w: np.ndarray, b: np.ndarray):
    """`W(x * s + t) + b = (W diag(s)) x + (W t + b)`: fold a BN forward into
    the linear map that directly consumes it."""
    s, t = bn_affine(module)
    return w * s[None, :], w @ t + b


def convert(clf, out_path: Path, *, variant: str, repo_id: str, revision: str,
            filters_201x60: np.ndarray, labels: dict[str, list[str]]) -> None:
    em, cl = clf.mods.embedding_model, clf.mods.classifier

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf_writer(str(out_path), ARCH)

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
    writer.add_string("general.source.commit", revision)

    writer.add_string("stt.variant", variant)
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

    def add(name: str, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr.astype(np.float32))
        encoded, raw_dtype = encode_for_gguf(a, GGMLQuantizationType.F32)
        writer.add_tensor(name, encoded, raw_dtype=raw_dtype)

    def add_tdnn(p: str, blk, split: tuple[str, ...] = ()) -> None:
        """conv weight + bias, then the BN as a scale/shift pair (TDNNBlock is
        conv -> ReLU -> BN, so the BN cannot fold into the conv).

        k>1 kernels are `<p>.conv.weight`, tap-major numpy [K, OC, IC] (ggml
        ne [IC, OC, K]); 1x1 kernels are `<p>.weight` [OC, IC]. `split` cuts a
        1x1 weight's input columns into one `<p>.<part>.weight` per member of
        the concatenation SpeechBrain feeds it.
        """
        w = f64(blk.conv.conv.weight)
        c = f"{p}.conv" if w.shape[2] > 1 else p
        if w.shape[2] > 1:
            add(f"{c}.weight", np.transpose(w, (2, 0, 1)))
        elif split:
            n = w.shape[1] // len(split)
            for k, part in enumerate(split):
                add(f"{p}.{part}.weight", w[:, k * n:(k + 1) * n, 0])
        else:
            add(f"{c}.weight", w[:, :, 0])
        add(f"{c}.bias", f64(blk.conv.conv.bias))
        s, t = bn_affine(blk.norm)
        add(f"{p}.bn.scale", s)
        add(f"{p}.bn.shift", t)

    # Mel-major: numpy [60, 201] -> ggml ne [201, 60].
    add("frontend.mel_filterbank", filters_201x60.T)
    add_tdnn("blk.0", em.blocks[0])
    for i in (1, 2, 3):
        b, p = em.blocks[i], f"blk.{i}"
        add_tdnn(f"{p}.tdnn1", b.tdnn1)
        # res2.{j} convolves chunk j+1 (plus the previous sub-block's output).
        for j, sub in enumerate(b.res2net_block.blocks):
            add_tdnn(f"{p}.res2.{j}", sub)
        add_tdnn(f"{p}.tdnn2", b.tdnn2)
        for c, m in (("c1", b.se_block.conv1), ("c2", b.se_block.conv2)):
            add(f"{p}.se.{c}.weight", f64(m.conv.weight)[:, :, 0])
            add(f"{p}.se.{c}.bias", f64(m.conv.bias))
    # mfa consumes cat(blocks.1..3); asp.tdnn consumes cat([x, mean, std]).
    add_tdnn("mfa", em.mfa, ("w1", "w2", "w3"))
    add_tdnn("asp.tdnn", em.asp.tdnn, ("x", "mean", "std"))
    add("asp.attn.weight", f64(em.asp.conv.conv.weight)[:, :, 0])
    add("asp.attn.bias", f64(em.asp.conv.conv.bias))

    # BNs with no activation before their linear map fold forward into it:
    # asp_bn -> fc, cls.bn0 -> cls.l1, cls.bn1 -> cls.out. The classifier's
    # LeakyReLUs sit before each BN, so they survive in the graph.
    for name, bn, lin in (
        ("fc", em.asp_bn, em.fc.conv),
        ("cls.l1", cl.norm, cl.DNN.block_0.linear.w),
        ("cls.out", cl.DNN.block_0.norm, cl.out.w),
    ):
        w = f64(lin.weight)
        w, b = fold_bn_into_linear(bn, w.reshape(w.shape[0], w.shape[1]), f64(lin.bias))
        add(f"{name}.weight", w)
        add(f"{name}.bias", b)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


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
        fail("cannot infer the source repo id from a local path; pass --repo-id")

    src = resolve_model_dir(args.model, args.revision)
    needed = ["hyperparams.yaml", "embedding_model.ckpt", "classifier.ckpt",
              "label_encoder.txt"]
    missing = [f for f in needed if not (src / f).exists()]
    if missing:
        fail(f"{src} is missing {missing}")
    slug = slug_from_repo_id(repo_id)
    out_path = args.out_path or (REPO_ROOT / "models" / slug /
                                 gguf_name(slug, "F32"))

    clf = load_reference(src)
    labels = assert_architecture(clf)
    filters = capture_fbank_matrix(clf)
    convert(clf, out_path, variant=DEFAULT_VARIANT, repo_id=repo_id,
            revision=args.revision, filters_201x60=filters, labels=labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
