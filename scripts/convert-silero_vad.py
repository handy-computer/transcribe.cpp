#!/usr/bin/env python3
"""
convert-silero_vad.py - convert the Silero VAD 16 kHz model into the F32
reference GGUF of the silero_vad family (VAD role).

Silero VAD is not distributed on Hugging Face: the canonical checkpoint is the
TorchScript file inside the silero-vad wheel (silero_vad/data/silero_vad.jit),
pinned by scripts/envs/silero_vad. This script loads it the way upstream does
(silero_vad.load_silero_vad), asserts the geometry hard-coded below against
the live modules, and writes the tensor layout of src/arch/silero_vad/. Only
the 16 kHz sub-model (_model) is converted; _model_8k is out of scope. See
docs/porting/families/silero_vad.md.

    uv run --project scripts/envs/silero_vad scripts/convert-silero_vad.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, LlamaFileType

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib.gguf_common import add_general_identity, encode_for_gguf, gguf_name, gguf_writer  # noqa: E402

ARCH = "silero_vad"
DEFAULT_VARIANT = "silero-vad-v6.2"
SILERO_VERSION = "6.2.3"
JIT_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"

# Written into the GGUF and asserted against the live modules below.
SAMPLE_RATE = 16000
FRAME = 512          # samples scored per probability
CONTEXT = 64         # trailing samples of the previous frame prepended
N_FFT = 256          # STFT filter_length (conv kernel)
HOP = 128            # STFT hop (conv stride)
PAD_RIGHT = 64       # ReflectionPad1d((0, 64)) before the STFT
N_BINS = N_FFT // 2 + 1
ENC_CHANNELS = [N_BINS, 128, 64, 64, 128]
ENC_STRIDES = [1, 2, 2, 1]
ENC_KERNEL = 3
HIDDEN = 128
N_PARAMS = 309_633


def fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def jit_path() -> Path:
    return Path(str(importlib.resources.files("silero_vad.data").joinpath("silero_vad.jit")))


def assert_architecture(model) -> None:
    """Check every value the C++ graph hard-codes that tensor shapes cannot show."""
    m = model._model
    st = m.stft
    checks = [
        ("_model.sample_rate", m.sample_rate, SAMPLE_RATE),
        ("_model.context_size_samples", m.context_size_samples, CONTEXT),
        ("stft.filter_length", st.filter_length, N_FFT),
        ("stft.win_length", st.win_length, N_FFT),
        ("stft.hop_length", st.hop_length, HOP),
        ("stft.window", st.window, "hann"),
        ("stft.padding", tuple(st.padding.padding), (0, PAD_RIGHT)),
        ("lstm.hidden_size", m.decoder.rnn.hidden_size, HIDDEN),
        ("lstm.input_size", m.decoder.rnn.input_size, ENC_CHANNELS[-1]),
        ("lstm.bias", m.decoder.rnn.bias, True),
        ("n_params", sum(int(v.numel()) for k, v in model.state_dict().items()
                         if k.startswith("_model.")), N_PARAMS),
    ]
    for i in range(4):
        blk = getattr(m.encoder, str(i))
        c = blk.reparam_conv
        checks += [
            (f"encoder.{i}.in", c.in_channels, ENC_CHANNELS[i]),
            (f"encoder.{i}.out", c.out_channels, ENC_CHANNELS[i + 1]),
            (f"encoder.{i}.kernel", tuple(c.kernel_size), (ENC_KERNEL,)),
            (f"encoder.{i}.stride", tuple(c.stride), (ENC_STRIDES[i],)),
            (f"encoder.{i}.padding", tuple(c.padding), (1,)),
            (f"encoder.{i}.dilation", tuple(c.dilation), (1,)),
            (f"encoder.{i}.groups", c.groups, 1),
            (f"encoder.{i}.padding_mode", c.padding_mode, "zeros"),
        ]
    head = getattr(m.decoder.decoder, "2")
    checks += [
        ("head.kernel", tuple(head.kernel_size), (1,)),
        ("head.out", head.out_channels, 1),
    ]
    bad = [(n, got, want) for n, got, want in checks if got != want]
    if bad:
        fail("architecture mismatch: " + "; ".join(f"{n}={g!r} (want {w!r})" for n, g, w in bad))

    # The STFT basis must be the periodic-Hann real DFT the C++ comments and
    # tests assume: rows [0, 129) cos, rows [129, 258) -sin.
    basis = model.state_dict()["_model.stft.forward_basis_buffer"][:, 0, :].double().numpy()
    n = np.arange(N_FFT)
    k = np.arange(N_BINS)[:, None]
    win = 0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)
    want = np.concatenate([win * np.cos(2 * np.pi * k * n / N_FFT),
                           -win * np.sin(2 * np.pi * k * n / N_FFT)])
    err = float(np.abs(basis - want).max())
    if err > 1e-6:
        fail(f"stft basis is not a periodic-Hann DFT (max err {err:.3e})")


def convert(model, out_path: Path, variant: str, jit_sha: str) -> None:
    sd = {k[len("_model."):]: v.detach().to(torch.float64).numpy()
          for k, v in model.state_dict().items() if k.startswith("_model.")}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf_writer(str(out_path), ARCH)
    add_general_identity(
        writer,
        name="Silero VAD v6.2",
        basename="silero-vad",
        size_label="309K",
        file_type=int(LlamaFileType.ALL_F32),
        author="Silero Team",
        organization="snakers4",
        version="v6.2",
        license="mit",
        license_name="MIT",
        license_link="https://github.com/snakers4/silero-vad/blob/master/LICENSE",
        repo_url="https://github.com/snakers4/silero-vad",
        source_url=f"https://pypi.org/project/silero-vad/{SILERO_VERSION}/",
        description="Silero VAD 16 kHz voice activity detector (STFT, 4-layer conv "
                    "encoder, LSTM cell, sigmoid head): one speech probability per "
                    "32 ms frame.",
        tags=["voice-activity-detection"],
    )
    writer.add_string("stt.variant", variant)
    writer.add_string("stt.silero_vad.source_sha256", jit_sha)
    writer.add_uint32("stt.frontend.sample_rate", SAMPLE_RATE)
    writer.add_uint32("stt.frontend.n_fft", N_FFT)
    writer.add_uint32("stt.frontend.hop_length", HOP)
    writer.add_uint32("stt.vad.frame_samples", FRAME)
    writer.add_uint32("stt.silero_vad.context_samples", CONTEXT)
    writer.add_uint32("stt.silero_vad.reflect_pad", PAD_RIGHT)
    writer.add_array("stt.silero_vad.encoder_channels", ENC_CHANNELS)
    writer.add_array("stt.silero_vad.encoder_strides", ENC_STRIDES)
    writer.add_uint32("stt.silero_vad.lstm_hidden", HIDDEN)
    writer.add_bool("stt.capability.streaming", True)

    def add(name: str, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr.astype(np.float32))
        encoded, raw_dtype = encode_for_gguf(a, GGMLQuantizationType.F32)
        writer.add_tensor(name, encoded, raw_dtype=raw_dtype)

    # [258, 1, 256] -> numpy [258, 256], ggml ne [256, 258]: one basis
    # function per row, so basis x frames is a single mul_mat.
    add("frontend.stft_basis", sd["stft.forward_basis_buffer"][:, 0, :])
    for i in range(4):
        # Conv1d weight as torch stores it, [OC, IC, K] (ggml ne [K, IC, OC]).
        # The loader expands it into a dense per-frame matrix.
        add(f"enc.{i}.conv.weight", sd[f"encoder.{i}.reparam_conv.weight"])
        add(f"enc.{i}.conv.bias", sd[f"encoder.{i}.reparam_conv.bias"])
    # torch LSTMCell gate order: input, forget, cell (g), output.
    add("lstm.weight_ih", sd["decoder.rnn.weight_ih"])
    add("lstm.weight_hh", sd["decoder.rnn.weight_hh"])
    add("lstm.bias_ih", sd["decoder.rnn.bias_ih"])
    add("lstm.bias_hh", sd["decoder.rnn.bias_hh"])
    add("head.weight", sd["decoder.decoder.2.weight"].reshape(HIDDEN))
    add("head.bias", sd["decoder.decoder.2.bias"].reshape(1))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert Silero VAD (16 kHz) to the F32 reference GGUF.")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (default: models/<variant>/<variant>-F32.gguf)")
    args = p.parse_args(argv)

    import silero_vad

    if silero_vad.__version__ != SILERO_VERSION:
        fail(f"silero-vad {silero_vad.__version__} installed, converter pinned to {SILERO_VERSION}")
    path = jit_path()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != JIT_SHA256:
        fail(f"{path} sha256 {sha} != pinned {JIT_SHA256}")

    model = silero_vad.load_silero_vad()
    assert_architecture(model)
    out = args.out_path or (REPO_ROOT / "models" / DEFAULT_VARIANT / gguf_name(DEFAULT_VARIANT, "F32"))
    convert(model, out, DEFAULT_VARIANT, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
