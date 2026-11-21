#!/usr/bin/env python3
"""
convert-smart_turn.py - convert Smart Turn v3.2 (pipecat-ai) into the F32
reference GGUF of the smart_turn family (TURN role).

Source: upstream publishes the model only as ONNX. The reference dtype is the
fp32 export, smart-turn-v3.2-gpu.onnx, from pipecat-ai/smart-turn-v3 at the
pinned revision (sha256-checked); smart-turn-v3.2-cpu.onnx is upstream's own
int8 quantization and is not converted. Weights are read through the graph
walk in scripts/lib/smart_turn_onnx.py, which names the anonymous, transposed
MatMul initializers and is validated by rebuilding train.py's
SmartTurnV3Model from them (see docs/porting/families/smart_turn.md).

Architecture (train.py SmartTurnV3Model): the whisper-tiny encoder with
max_source_positions = 400 (8 s, 800 mel frames), then
    attention pooling  w = softmax_t(Linear(384,256) - Tanh - Linear(256,1))
                       pooled = sum_t w[t] * h[t]
    classifier         Linear(384,256) - LayerNorm(256) - GELU - Dropout -
                       Linear(256,64) - GELU - Linear(64,1)
    sigmoid            P(turn complete)

Layout: the encoder and front end use convert-whisper.py's tensor names,
layouts and KVs (stt.frontend.*, encoder hparams under
stt.smart_turn.encoder.* instead of stt.whisper.encoder.*), so the whisper
encoder loader reads them unchanged. Linear weights are [out, in] (ggml ne
[in, out]), Conv1d [out, in, k]; whisper's k projection has no bias, and
neither has this export's.

    frontend.mel_filterbank   [80, 201]   WhisperFeatureExtractor(chunk_length=8).mel_filters.T
    frontend.window           [400]       periodic Hann
    enc.conv.{0,1}.weight/bias, enc.pos_emb.weight [400, 384],
    enc.blocks.{i}.{norm_attn, attn.q, attn.k (no bias), attn.v, attn.out,
                    norm_ffn, ffn.fc1, ffn.fc2}.*, enc.final_norm.*
    head.pool.0.weight/bias   [256, 384] / [256]
    head.pool.2.weight/bias   [1, 256] / [1]
    head.cls.0.weight/bias    [256, 384] / [256]
    head.cls.1.weight/bias    [256]           LayerNorm (eps 1e-5)
    head.cls.4.weight/bias    [64, 256] / [64]
    head.cls.6.weight/bias    [1, 64] / [1]

    uv run --project scripts/envs/smart_turn scripts/convert-smart_turn.py
    uv run --project scripts/envs/smart_turn scripts/convert-smart_turn.py \\
        --onnx /path/smart-turn-v3.2-gpu.onnx out.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, LlamaFileType

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib import smart_turn_onnx as st  # noqa: E402
from lib.gguf_common import (  # noqa: E402
    add_general_identity,
    encode_for_gguf,
    gguf_name,
    gguf_writer,
    reference_dtype_for,
)

ARCH = "smart_turn"
VARIANT = "smart-turn-v3.2"
REF_TYPE = GGMLQuantizationType.F32

# The 23 languages of the v3.2 model card / benchmark (BCP-47).
LANGUAGES = ["ar", "bn", "zh", "da", "nl", "de", "en", "fi", "fr", "hi", "id", "it",
             "ja", "ko", "mr", "no", "pl", "pt", "ru", "es", "tr", "uk", "vi"]

ENCODER_TOP = [
    ("encoder.conv1.weight",            "enc.conv.0.weight"),
    ("encoder.conv1.bias",              "enc.conv.0.bias"),
    ("encoder.conv2.weight",            "enc.conv.1.weight"),
    ("encoder.conv2.bias",              "enc.conv.1.bias"),
    ("encoder.embed_positions.weight",  "enc.pos_emb.weight"),
    ("encoder.layer_norm.weight",       "enc.final_norm.weight"),
    ("encoder.layer_norm.bias",         "enc.final_norm.bias"),
]

# Whisper attention: q / v / out have bias, k does not.
ENCODER_BLOCK = [
    ("self_attn_layer_norm.weight", "norm_attn.weight"),
    ("self_attn_layer_norm.bias",   "norm_attn.bias"),
    ("self_attn.q_proj.weight",     "attn.q.weight"),
    ("self_attn.q_proj.bias",       "attn.q.bias"),
    ("self_attn.k_proj.weight",     "attn.k.weight"),
    ("self_attn.v_proj.weight",     "attn.v.weight"),
    ("self_attn.v_proj.bias",       "attn.v.bias"),
    ("self_attn.out_proj.weight",   "attn.out.weight"),
    ("self_attn.out_proj.bias",     "attn.out.bias"),
    ("final_layer_norm.weight",     "norm_ffn.weight"),
    ("final_layer_norm.bias",       "norm_ffn.bias"),
    ("fc1.weight",                  "ffn.fc1.weight"),
    ("fc1.bias",                    "ffn.fc1.bias"),
    ("fc2.weight",                  "ffn.fc2.weight"),
    ("fc2.bias",                    "ffn.fc2.bias"),
]

HEAD = [
    ("pool_attention.0.weight", "head.pool.0.weight"),
    ("pool_attention.0.bias",   "head.pool.0.bias"),
    ("pool_attention.2.weight", "head.pool.2.weight"),
    ("pool_attention.2.bias",   "head.pool.2.bias"),
    ("classifier.0.weight",     "head.cls.0.weight"),
    ("classifier.0.bias",       "head.cls.0.bias"),
    ("classifier.1.weight",     "head.cls.1.weight"),
    ("classifier.1.bias",       "head.cls.1.bias"),
    ("classifier.4.weight",     "head.cls.4.weight"),
    ("classifier.4.bias",       "head.cls.4.bias"),
    ("classifier.6.weight",     "head.cls.6.weight"),
    ("classifier.6.bias",       "head.cls.6.bias"),
]


def fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def check_graph(model) -> None:
    """Assert the graph-level facts the C++ hard-codes and the weights cannot
    show: LayerNorm eps 1e-5 everywhere, exact (erf) GELU, conv strides /
    padding, 6 heads of 64, and the single output being a sigmoid."""
    import onnx

    g = model.graph
    attrs = lambda n: {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}  # noqa: E731
    lns = [n for n in g.node if n.op_type == "LayerNormalization"]
    if len(lns) != 2 * st.N_LAYERS + 2:
        fail(f"{len(lns)} LayerNormalization nodes, want {2 * st.N_LAYERS + 2}")
    for n in lns:
        a = attrs(n)
        if abs(a.get("epsilon", 0) - 1e-5) > 1e-9 or a.get("axis", -1) != -1:
            fail(f"LayerNorm {n.name} eps {a.get('epsilon')} axis {a.get('axis')}")
    ops = {n.op_type for n in g.node}
    if "Erf" not in ops or "Tanh" not in ops or {"Gelu", "FastGelu"} & ops:
        fail("expected exact erf GELU and a Tanh in the pooling head")
    convs = [attrs(n) for n in g.node if n.op_type == "Conv"]
    want = [{"strides": [1], "pads": [1, 1]}, {"strides": [2], "pads": [1, 1]}]
    if [{k: c[k] for k in ("strides", "pads")} for c in convs] != want:
        fail(f"conv attributes {convs}")
    from onnx import numpy_helper
    consts = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    reshape_dims = {int(v[0]) for v in consts.values() if v.dtype == np.int64 and v.shape == (1,)}
    if not {st.N_HEADS, st.D_MODEL // st.N_HEADS, st.N_POSITIONS} <= reshape_dims:
        fail("head split constants (6 heads x 64, 400 positions) not found")
    outs = [o.name for o in g.output]
    sig = [n for n in g.node if n.op_type == "Sigmoid"]
    if outs != ["logits"] or len(sig) != 1 or sig[0].output[0] != "logits":
        fail("graph output is not a single sigmoid 'logits'")
    shape = [d.dim_value or d.dim_param for d in g.input[0].type.tensor_type.shape.dim]
    if g.input[0].name != "input_features" or shape[1:] != [st.N_MELS, st.N_FRAMES]:
        fail(f"input {g.input[0].name} {shape}")


def convert(onnx_path: Path, out_path: Path) -> None:
    import onnx

    sha = st.sha256_file(onnx_path)
    if sha != st.SHA256[st.ONNX_FP32]:
        fail(f"{onnx_path} sha256 {sha} != pinned {st.SHA256[st.ONNX_FP32]} ({st.ONNX_FP32})")
    model = onnx.load(str(onnx_path))
    check_graph(model)
    sd, notes = st.map_onnx_weights(model)
    st.check_geometry(sd)

    fe = st.feature_extractor()
    mel_fb = np.ascontiguousarray(np.asarray(fe.mel_filters, dtype=np.float64).T.astype(np.float32))
    if mel_fb.shape != (st.N_MELS, fe.n_fft // 2 + 1):
        fail(f"mel filterbank shape {mel_fb.shape}")
    if (fe.n_fft, fe.hop_length, fe.sampling_rate, fe.n_samples, fe.nb_max_frames) != \
            (400, 160, st.SAMPLE_RATE, st.MAX_AUDIO_SAMPLES, st.N_FRAMES):
        fail("WhisperFeatureExtractor(chunk_length=8) geometry changed")
    n = fe.n_fft
    hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)

    total = sum(int(v.size) for v in sd.values())
    size_label = f"{total / 1e6:.0f}M"
    print(f"{onnx_path.name}: {len(sd)} weights, {total:,} params -> {size_label}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf_writer(str(out_path), ARCH)
    add_general_identity(
        writer,
        name="Smart Turn v3.2",
        basename="smart-turn",
        size_label=size_label,
        file_type=int(LlamaFileType.ALL_F32),
        languages=LANGUAGES,
        author="Daily (Pipecat)",
        organization="pipecat-ai",
        version="v3.2",
        license="bsd-2-clause",
        license_name="BSD 2-Clause",
        license_link="https://github.com/pipecat-ai/smart-turn/blob/main/LICENSE",
        repo_url=f"https://huggingface.co/{st.HF_REPO}",
        source_url=f"https://huggingface.co/{st.HF_REPO}/blob/{st.HF_REVISION}/{st.ONNX_FP32}",
        description="Smart Turn v3.2 end-of-turn detector: whisper-tiny encoder over the "
                    "last 8 s of audio, attention pooling and an MLP head giving "
                    "P(the speaker's turn is complete).",
        tags=["turn-detection", "end-of-turn"],
    )
    writer.add_string("stt.variant", VARIANT)
    writer.add_string("stt.smart_turn.source_sha256", sha)

    writer.add_bool("stt.capability.lang_detect", False)
    writer.add_bool("stt.capability.translate", False)
    writer.add_bool("stt.capability.timestamps", False)
    writer.add_bool("stt.capability.streaming", False)

    writer.add_uint32("stt.smart_turn.encoder.n_layers", st.N_LAYERS)
    writer.add_uint32("stt.smart_turn.encoder.d_model", st.D_MODEL)
    writer.add_uint32("stt.smart_turn.encoder.n_heads", st.N_HEADS)
    writer.add_uint32("stt.smart_turn.encoder.ffn_dim", st.FFN_DIM)
    writer.add_uint32("stt.smart_turn.encoder.num_mel_bins", st.N_MELS)
    writer.add_uint32("stt.smart_turn.encoder.max_source_positions", st.N_POSITIONS)
    writer.add_string("stt.smart_turn.encoder.activation", "gelu")
    writer.add_uint32("stt.smart_turn.max_audio_samples", st.MAX_AUDIO_SAMPLES)
    writer.add_float32("stt.smart_turn.threshold", st.THRESHOLD)
    writer.add_bool("stt.smart_turn.waveform_normalize", True)
    writer.add_float32("stt.smart_turn.waveform_normalize_eps", st.NORMALIZE_EPS)

    # WhisperFeatureExtractor(chunk_length=8): convert-whisper.py's keys.
    writer.add_string("stt.frontend.type", "mel")
    writer.add_uint32("stt.frontend.num_mels", st.N_MELS)
    writer.add_uint32("stt.frontend.sample_rate", st.SAMPLE_RATE)
    writer.add_uint32("stt.frontend.n_fft", fe.n_fft)
    writer.add_uint32("stt.frontend.win_length", fe.n_fft)
    writer.add_uint32("stt.frontend.hop_length", fe.hop_length)
    writer.add_string("stt.frontend.window", "hann_periodic")
    writer.add_string("stt.frontend.normalize", "whisper_logmel")
    writer.add_float32("stt.frontend.dither", 0.0)
    writer.add_float32("stt.frontend.pre_emphasis", 0.0)
    writer.add_float32("stt.frontend.f_min", 0.0)
    writer.add_float32("stt.frontend.f_max", st.SAMPLE_RATE / 2.0)
    writer.add_string("stt.frontend.pad_mode", "reflect")
    writer.add_bool("stt.frontend.center", True)
    writer.add_string("stt.frontend.mel_norm", "slaney")
    writer.add_uint32("stt.frontend.chunk_length", st.MAX_AUDIO_SECONDS)
    writer.add_uint32("stt.frontend.n_samples", st.MAX_AUDIO_SAMPLES)
    writer.add_uint32("stt.frontend.nb_max_frames", st.N_FRAMES)

    written: list[tuple[str, tuple, str]] = []

    def add(dst: str, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr.astype(np.float32))
        encoded, raw_dtype = encode_for_gguf(a, reference_dtype_for(dst, REF_TYPE))
        writer.add_tensor(dst, encoded, raw_dtype=raw_dtype)
        written.append((dst, a.shape, raw_dtype.name))

    add("frontend.mel_filterbank", mel_fb)
    add("frontend.window", hann)
    consumed: set[str] = set()
    table = list(ENCODER_TOP)
    for i in range(st.N_LAYERS):
        table += [(f"encoder.layers.{i}.{s}", f"enc.blocks.{i}.{d}") for s, d in ENCODER_BLOCK]
    table += HEAD
    for src, dst in table:
        if src not in sd:
            fail(f"mapped weights lack {src}")
        add(dst, sd[src])
        consumed.add(src)
    left = sorted(set(sd) - consumed)
    if left:
        fail(f"weights not written: {left}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    for name, shape, dtype in written:
        print(f"  {name:<34} {str(list(shape)):<14} {dtype}")
    print(f"Wrote {out_path} ({len(written)} tensors, {out_path.stat().st_size:,} bytes)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert Smart Turn v3.2 (fp32 ONNX) to the F32 reference GGUF.")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (default: models/smart-turn-v3.2/smart-turn-v3.2-F32.gguf)")
    p.add_argument("--onnx", type=Path, default=None,
                   help=f"Local {st.ONNX_FP32} (default: download {st.HF_REPO}@{st.HF_REVISION[:8]})")
    args = p.parse_args(argv)

    onnx_path = args.onnx.expanduser().resolve() if args.onnx else st.fetch(st.ONNX_FP32)
    out = args.out_path or (REPO_ROOT / "models" / VARIANT / gguf_name(VARIANT, "F32"))
    convert(onnx_path, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
