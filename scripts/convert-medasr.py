#!/usr/bin/env python3
"""
convert-medasr.py — convert google/medasr (LASR-CTC) to a transcribe.cpp GGUF.

Source: HuggingFace `google/medasr` (HF Transformers 5.0.0.dev0 at
commit 65dc261512cbdb1ee72b88ae5b222f2605aad8e5). The model classes
`lasr_ctc` / `lasr_encoder` / `LasrFeatureExtractor` / `LasrTokenizer`
live in upstream transformers at that pin.

GGUF architecture key: `medasr`. The loader dispatches to
src/arch/medasr/ (Stage 4 brings up the arch).

Tensor layout (verified against the safetensors at HF revision
ae1e4845b4b07479735d93e1e591e566435b7104):

  ctc_head.{weight,bias}                       # Conv1d(512, 512, k=1)
  encoder.subsampler.dense_0.{weight,bias}     # Linear(128 -> 512)
  encoder.subsampler.conv_0.{weight,bias}      # Conv1d(512, 512, k=5, s=2)
  encoder.subsampler.conv_1.{weight,bias}      # Conv1d(512, 256, k=5, s=2)
  encoder.subsampler.dense_1.{weight,bias}     # Linear(256 -> 512)
  encoder.out_norm.weight                      # LayerNorm 512 (no bias)
  encoder.rotary_emb.inv_freq                  # buffer; rebuilt in C++
  encoder.layers.{i}.norm_feed_forward1.weight       # LN no-bias
  encoder.layers.{i}.feed_forward1.linear1.weight    # (2048, 512)
  encoder.layers.{i}.feed_forward1.linear2.weight    # (512, 2048)
  encoder.layers.{i}.norm_self_att.weight            # LN no-bias
  encoder.layers.{i}.self_attn.{q,k,v,o}_proj.weight # (512, 512) — no bias
  encoder.layers.{i}.norm_conv.weight                # LN no-bias
  encoder.layers.{i}.conv.pointwise_conv1.weight     # (1024, 512, 1)
  encoder.layers.{i}.conv.depthwise_conv.weight      # (512, 1, 32)
  encoder.layers.{i}.conv.norm.{weight,bias,
                                running_mean,running_var,
                                num_batches_tracked}  # BatchNorm1d(512)
  encoder.layers.{i}.conv.pointwise_conv2.weight     # (512, 512, 1)
  encoder.layers.{i}.norm_feed_forward2.weight       # LN no-bias
  encoder.layers.{i}.feed_forward2.linear1.weight    # (2048, 512)
  encoder.layers.{i}.feed_forward2.linear2.weight    # (512, 2048)
  encoder.layers.{i}.norm_out.weight                 # LN no-bias

All LayerNorms in the encoder block use bias=False (configuration_lasr.py
LasrEncoderConfig.attention_bias defaults False; ditto convolution_bias;
see LasrEncoderBlock.__init__). The Conv1d kernels and BatchNorm1d have
biases. The CTC head Conv1d(k=1) has a bias.

Reference dtype = F32 (intake.dtype.expected; safetensors header is F32).
Per gguf_common.reference_dtype_for: with reference F32 every storage
tensor is F32 (BF16 → F16 conv downgrade is not triggered here).

Frontend: emitted as KVs + two tensors so the C++ MelFrontend can mirror
LasrFeatureExtractor exactly:
  frontend.mel_filterbank   [n_fft//2+1 = 257, n_mels = 128]   F32
                            Kaldi-mel formula with HTK-style DC bin
                            zeroed (row 0 is all zeros), 125–7500 Hz,
                            unnormalized triangular filters.
  frontend.window           [win_length = 400]                 F32
                            Symmetric Hann (torch.hann_window(periodic=False)).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType
from safetensors.torch import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    TOKEN_TYPE_BYTE,
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    TOKEN_TYPE_UNKNOWN,
    TOKEN_TYPE_UNUSED,
    add_general_identity,
    encode_for_gguf,
    reference_dtype_for,
    slug_from_repo_id,
)


REF_DTYPE = "F32"
ARCH_KEY = "medasr"
REFERENCE_FILE_TYPE = LlamaFileType.ALL_F32


# ---- Source resolution ----------------------------------------------------


def hf_resolve(model_arg: str, revision: str | None) -> Path:
    p = Path(model_arg).expanduser().resolve()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_arg, revision=revision))


# ---- Hparam extraction ----------------------------------------------------


def read_hparams(config: dict, preproc: dict) -> dict:
    enc = config["encoder_config"]
    hp: dict = {}
    hp["enc_n_layers"] = enc["num_hidden_layers"]
    hp["enc_hidden"] = enc["hidden_size"]
    hp["enc_n_heads"] = enc["num_attention_heads"]
    hp["enc_n_kv_heads"] = enc.get("num_key_value_heads", enc["num_attention_heads"])
    hp["enc_head_dim"] = enc.get("head_dim", enc["hidden_size"] // enc["num_attention_heads"])
    hp["enc_intermediate"] = enc["intermediate_size"]
    hp["enc_hidden_act"] = enc["hidden_act"]
    hp["enc_conv_kernel"] = enc["conv_kernel_size"]
    hp["enc_conv_residual_w0"] = float(enc["conv_residual_weights"][0])
    hp["enc_conv_residual_w1"] = float(enc["conv_residual_weights"][1])
    hp["enc_ff_residual_w0"] = float(enc["feed_forward_residual_weights"][0])
    hp["enc_ff_residual_w1"] = float(enc["feed_forward_residual_weights"][1])
    hp["enc_layer_norm_eps"] = float(enc["layer_norm_eps"])
    hp["enc_max_pos_emb"] = enc["max_position_embeddings"]
    hp["enc_batch_norm_momentum"] = float(enc.get("batch_norm_momentum", 0.01))
    rope = enc.get("rope_parameters", {})
    hp["enc_rope_theta"] = float(rope.get("rope_theta", 10000.0))
    hp["enc_rope_type"] = str(rope.get("rope_type", "default"))

    # Subsampling: config exposes ONE (kernel, stride, channels) triple
    # but LasrEncoderSubsampling stacks TWO stride-2 convs reusing them.
    hp["enc_num_mel_bins"] = enc["num_mel_bins"]
    hp["enc_sub_kernel"] = enc["subsampling_conv_kernel_size"]
    hp["enc_sub_stride"] = enc["subsampling_conv_stride"]
    hp["enc_sub_channels"] = enc["subsampling_conv_channels"]
    hp["enc_sub_n_layers"] = 2

    # CTC head
    hp["ctc_vocab_size"] = config["vocab_size"]
    hp["ctc_blank_id"] = int(config["pad_token_id"])  # `<epsilon>` = 0

    # Frontend
    hp["fe_type"] = "mel"
    hp["fe_sample_rate"] = preproc.get("sampling_rate", 16000)
    hp["fe_num_mels"] = preproc.get("feature_size", 128)
    hp["fe_n_fft"] = preproc.get("n_fft", 512)
    hp["fe_win_length"] = preproc.get("win_length", 400)
    hp["fe_hop_length"] = preproc.get("hop_length", 160)
    # Pinned in the golden manifest (read from LasrFeatureExtractor source).
    hp["fe_window"] = "hann_symmetric"
    hp["fe_normalize"] = "none"
    hp["fe_pad_mode"] = "zero"
    hp["fe_mel_norm"] = "htk"
    hp["fe_log_clamp_min"] = 1e-5
    hp["fe_mel_lower_hz"] = 125.0
    hp["fe_mel_upper_hz"] = 7500.0

    return hp


# ---- Frontend constants (mel filterbank + window) -------------------------


def _hertz_to_mel_kaldi(f_hz: np.ndarray) -> np.ndarray:
    """Kaldi mel scale (also used by transformers.audio_utils.hertz_to_mel
    with mel_scale='kaldi'). Matches LasrFeatureExtractor exactly when
    invoked at float64 precision."""
    return 1127.0 * np.log(1.0 + f_hz / 700.0)


def lasr_mel_filterbank(
    *,
    num_mel_bins: int,
    num_spectrogram_bins: int,
    sample_rate: int,
    lower_hz: float,
    upper_hz: float,
) -> np.ndarray:
    """Reproduces LasrFeatureExtractor.linear_to_mel_weight_matrix
    bit-for-bit (numpy float64 internals, HTK-style DC-bin exclusion,
    Kaldi mel scale). Returns shape (num_spectrogram_bins, num_mel_bins)
    in float32."""
    internal_dtype = np.float64
    bands_to_zero = 1
    nyquist = sample_rate / 2.0
    linear_freqs = np.linspace(0.0, nyquist, num_spectrogram_bins, dtype=internal_dtype)[bands_to_zero:]
    bins_mel = _hertz_to_mel_kaldi(linear_freqs)[:, np.newaxis]
    edges = np.linspace(
        _hertz_to_mel_kaldi(np.asarray(lower_hz)),
        _hertz_to_mel_kaldi(np.asarray(upper_hz)),
        num_mel_bins + 2,
        dtype=internal_dtype,
    )
    lower = edges[:-2][np.newaxis, :]
    center = edges[1:-1][np.newaxis, :]
    upper = edges[2:][np.newaxis, :]
    lower_slopes = (bins_mel - lower) / (center - lower)
    upper_slopes = (upper - bins_mel) / (upper - center)
    weights = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.pad(weights, [[bands_to_zero, 0], [0, 0]]).astype(np.float32)


def lasr_hann_window(win_length: int) -> np.ndarray:
    """Symmetric Hann (torch.hann_window(periodic=False))."""
    n = win_length
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))).astype(np.float32)


# ---- Tokenizer ------------------------------------------------------------


def extract_sp_tokenizer(sp_model_path: Path, vocab_limit: int) -> dict:
    """Read spiece.model and return the first `vocab_limit` tokens.

    The LASR tokenizer.json exposes 613 entries (512 SP + <pad> + 100
    <extra_id_N>), but the CTC head only emits over the first 512.
    Trim to vocab_limit so the GGUF vocab matches the model's output dim.

    SP ids 0–3 are already the four control tokens:
      0 <epsilon>  (CTC blank)
      1 <s>
      2 </s>
      3 <unk>
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))
    if sp.vocab_size() < vocab_limit:
        raise SystemExit(
            f"spiece.model has {sp.vocab_size()} pieces but vocab_limit="
            f"{vocab_limit}; SP model is too small for the configured "
            "CTC head."
        )

    tokens: list[str] = []
    scores: list[float] = []
    types: list[int] = []
    for i in range(vocab_limit):
        tokens.append(sp.id_to_piece(i))
        scores.append(sp.get_score(i))
        if sp.is_unknown(i):
            types.append(TOKEN_TYPE_UNKNOWN)
        elif sp.is_control(i):
            types.append(TOKEN_TYPE_CONTROL)
        elif sp.is_unused(i):
            types.append(TOKEN_TYPE_UNUSED)
        elif sp.is_byte(i):
            types.append(TOKEN_TYPE_BYTE)
        else:
            types.append(TOKEN_TYPE_NORMAL)
    return {
        "tokens": tokens,
        "scores": scores,
        "types": types,
        "blank_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 3,
    }


# ---- Tensor name maps -----------------------------------------------------


ENC_TOP_MAP = [
    ("encoder.subsampler.dense_0.weight", "enc.subsampling.dense_0.weight"),
    ("encoder.subsampler.dense_0.bias",   "enc.subsampling.dense_0.bias"),
    ("encoder.subsampler.conv_0.weight",  "enc.subsampling.conv_0.weight"),
    ("encoder.subsampler.conv_0.bias",    "enc.subsampling.conv_0.bias"),
    ("encoder.subsampler.conv_1.weight",  "enc.subsampling.conv_1.weight"),
    ("encoder.subsampler.conv_1.bias",    "enc.subsampling.conv_1.bias"),
    ("encoder.subsampler.dense_1.weight", "enc.subsampling.dense_1.weight"),
    ("encoder.subsampler.dense_1.bias",   "enc.subsampling.dense_1.bias"),
    ("encoder.out_norm.weight",           "enc.out_norm.weight"),
    ("ctc_head.weight",                   "ctc.proj.weight"),
    ("ctc_head.bias",                     "ctc.proj.bias"),
]

ENC_BLOCK_MAP = [
    # FF1 macaron (weights only; LN has no bias)
    ("norm_feed_forward1.weight",         "norm_ff1.weight"),
    ("feed_forward1.linear1.weight",      "ff1_up.weight"),
    ("feed_forward1.linear2.weight",      "ff1_down.weight"),
    # Self-attention (Q/K/V/O linears, no bias; LN no bias)
    ("norm_self_att.weight",              "norm_attn.weight"),
    ("self_attn.q_proj.weight",           "attn_q.weight"),
    ("self_attn.k_proj.weight",           "attn_k.weight"),
    ("self_attn.v_proj.weight",           "attn_v.weight"),
    ("self_attn.o_proj.weight",           "attn_o.weight"),
    # Conformer convolution module
    ("norm_conv.weight",                  "norm_conv.weight"),
    ("conv.pointwise_conv1.weight",       "conv_pointwise1.weight"),
    ("conv.depthwise_conv.weight",        "conv_depthwise.weight"),
    ("conv.norm.weight",                  "conv_bn.weight"),
    ("conv.norm.bias",                    "conv_bn.bias"),
    ("conv.norm.running_mean",            "conv_bn.running_mean"),
    ("conv.norm.running_var",             "conv_bn.running_var"),
    ("conv.pointwise_conv2.weight",       "conv_pointwise2.weight"),
    # FF2 macaron
    ("norm_feed_forward2.weight",         "norm_ff2.weight"),
    ("feed_forward2.linear1.weight",      "ff2_up.weight"),
    ("feed_forward2.linear2.weight",      "ff2_down.weight"),
    # Post-block out norm
    ("norm_out.weight",                   "norm_post.weight"),
]

# Tensors we read but don't emit (bookkeeping for BN, rebuilt buffers).
ENC_BLOCK_SKIP = {"conv.norm.num_batches_tracked"}
TOP_LEVEL_SKIP = {"encoder.rotary_emb.inv_freq"}


# ---- Tensor write helpers -------------------------------------------------


def to_f32_numpy(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
    return arr


def add_tensor(writer: GGUFWriter, name: str, t: torch.Tensor) -> None:
    """Encode a float32 array using gguf_common's reference-dtype rules
    and append it to the writer. Reference dtype is F32 for medasr so
    everything ends up F32; helper kept for cross-family consistency.
    """
    arr = to_f32_numpy(t)
    ggml_type = reference_dtype_for(name, GGMLQuantizationType.F32)
    packed, raw_type = encode_for_gguf(arr, ggml_type)
    writer.add_tensor(name, packed, raw_dtype=raw_type)


def add_f32_array(writer: GGUFWriter, name: str, arr: np.ndarray) -> None:
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    writer.add_tensor(name, arr)


# ---- Main -----------------------------------------------------------------


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model", help="HF repo id or local model directory")
    p.add_argument("--repo-id", default=None,
                   help="HF repo id used to derive output slug (default: same as model)")
    p.add_argument("--revision", default=None,
                   help="HF revision (ignored for local paths)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: models/<slug>/)")
    args = p.parse_args(argv)

    repo_id = args.repo_id or args.model
    slug = slug_from_repo_id(repo_id)

    model_dir = hf_resolve(args.model, args.revision)
    print(f"Source: {model_dir}")

    config = json.loads((model_dir / "config.json").read_text())
    preproc = json.loads((model_dir / "preprocessor_config.json").read_text())
    hp = read_hparams(config, preproc)

    # Tokenizer (trim spiece.model to model output dim = ctc vocab).
    tok = extract_sp_tokenizer(model_dir / "spiece.model", hp["ctc_vocab_size"])

    print(f"variant: {slug}")
    print(f"  encoder: {hp['enc_n_layers']} layers, hidden={hp['enc_hidden']}, "
          f"heads={hp['enc_n_heads']} (kv={hp['enc_n_kv_heads']}), "
          f"intermediate={hp['enc_intermediate']}, conv_k={hp['enc_conv_kernel']}")
    print(f"  ff_residual_weights={[hp['enc_ff_residual_w0'], hp['enc_ff_residual_w1']]}  "
          f"conv_residual_weights={[hp['enc_conv_residual_w0'], hp['enc_conv_residual_w1']]}")
    print(f"  subsampling: {hp['enc_sub_n_layers']}x stride={hp['enc_sub_stride']} "
          f"kernel={hp['enc_sub_kernel']} channels={hp['enc_sub_channels']}")
    print(f"  rope_theta={hp['enc_rope_theta']}  max_pos_emb={hp['enc_max_pos_emb']}")
    print(f"  CTC head: vocab={hp['ctc_vocab_size']} blank_id={hp['ctc_blank_id']}")
    print(f"  tokenizer: {len(tok['tokens'])} pieces, specials "
          f"blank={tok['blank_id']} bos={tok['bos_id']} eos={tok['eos_id']} unk={tok['unk_id']}")

    outdir = Path(args.outdir or f"models/{slug}").expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{slug}-{REF_DTYPE}.gguf"
    print(f"Writing GGUF: {out_path}")

    writer = GGUFWriter(str(out_path), ARCH_KEY)

    # ---- general.* ----
    add_general_identity(
        writer,
        name="MedASR",
        basename="medasr",
        file_type=REFERENCE_FILE_TYPE,
        languages=["en"],
        author="Google",
        organization="google",
        license="other",
        license_name="health-ai-developer-foundations",
        license_link="https://developers.google.com/health-ai-developer-foundations/terms",
        repo_url=(f"https://huggingface.co/{repo_id}" if repo_id else None),
    )

    # ---- stt.variant + capabilities ----
    writer.add_string("stt.variant", slug)
    writer.add_bool("stt.capability.translation", False)
    writer.add_bool("stt.capability.lang_detect", False)
    writer.add_bool("stt.capability.word_timestamps", False)
    writer.add_bool("stt.capability.speaker_diarization", False)
    writer.add_bool("stt.capability.streaming", False)

    # ---- tokenizer.ggml.* (SentencePiece BPE, trimmed to model output dim) ----
    # tokenizer.ggml.model = "bpe" matches the existing parakeet/gigaam
    # CTC ports; the C++ loader treats SP-backed BPE the same way.
    writer.add_string("tokenizer.ggml.model", "bpe")
    writer.add_string("tokenizer.ggml.pre", "medasr")
    writer.add_array("tokenizer.ggml.tokens", tok["tokens"])
    writer.add_array("tokenizer.ggml.scores", tok["scores"])
    writer.add_array("tokenizer.ggml.token_type", tok["types"])
    writer.add_uint32("tokenizer.ggml.bos_token_id", tok["bos_id"])
    writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
    writer.add_uint32("tokenizer.ggml.unknown_token_id", tok["unk_id"])
    writer.add_uint32("tokenizer.ggml.blank_token_id", tok["blank_id"])
    # `<epsilon>` doubles as the padding token (config.pad_token_id = 0).
    writer.add_uint32("tokenizer.ggml.padding_token_id", tok["blank_id"])

    # ---- stt.medasr.encoder.* ----
    writer.add_uint32("stt.medasr.encoder.n_layers", hp["enc_n_layers"])
    writer.add_uint32("stt.medasr.encoder.hidden", hp["enc_hidden"])
    writer.add_uint32("stt.medasr.encoder.n_heads", hp["enc_n_heads"])
    writer.add_uint32("stt.medasr.encoder.n_kv_heads", hp["enc_n_kv_heads"])
    writer.add_uint32("stt.medasr.encoder.head_dim", hp["enc_head_dim"])
    writer.add_uint32("stt.medasr.encoder.intermediate", hp["enc_intermediate"])
    writer.add_string("stt.medasr.encoder.hidden_act", hp["enc_hidden_act"])
    writer.add_uint32("stt.medasr.encoder.conv_kernel", hp["enc_conv_kernel"])
    writer.add_float32("stt.medasr.encoder.conv_residual_w0", hp["enc_conv_residual_w0"])
    writer.add_float32("stt.medasr.encoder.conv_residual_w1", hp["enc_conv_residual_w1"])
    writer.add_float32("stt.medasr.encoder.ff_residual_w0", hp["enc_ff_residual_w0"])
    writer.add_float32("stt.medasr.encoder.ff_residual_w1", hp["enc_ff_residual_w1"])
    writer.add_float32("stt.medasr.encoder.layer_norm_eps", hp["enc_layer_norm_eps"])
    writer.add_uint32("stt.medasr.encoder.max_pos_emb", hp["enc_max_pos_emb"])
    writer.add_float32("stt.medasr.encoder.batch_norm_momentum", hp["enc_batch_norm_momentum"])
    writer.add_float32("stt.medasr.encoder.rope_theta", hp["enc_rope_theta"])
    writer.add_string("stt.medasr.encoder.rope_type", hp["enc_rope_type"])
    writer.add_uint32("stt.medasr.encoder.num_mel_bins", hp["enc_num_mel_bins"])
    writer.add_uint32("stt.medasr.encoder.sub_kernel", hp["enc_sub_kernel"])
    writer.add_uint32("stt.medasr.encoder.sub_stride", hp["enc_sub_stride"])
    writer.add_uint32("stt.medasr.encoder.sub_channels", hp["enc_sub_channels"])
    writer.add_uint32("stt.medasr.encoder.sub_n_layers", hp["enc_sub_n_layers"])

    # ---- stt.medasr.ctc.* ----
    writer.add_uint32("stt.medasr.ctc.vocab_size", hp["ctc_vocab_size"])
    writer.add_uint32("stt.medasr.ctc.blank_id", hp["ctc_blank_id"])

    # ---- stt.frontend.* ----
    writer.add_string("stt.frontend.type", hp["fe_type"])
    writer.add_uint32("stt.frontend.sample_rate", hp["fe_sample_rate"])
    writer.add_uint32("stt.frontend.num_mels", hp["fe_num_mels"])
    writer.add_uint32("stt.frontend.n_fft", hp["fe_n_fft"])
    writer.add_uint32("stt.frontend.win_length", hp["fe_win_length"])
    writer.add_uint32("stt.frontend.hop_length", hp["fe_hop_length"])
    writer.add_string("stt.frontend.window", hp["fe_window"])
    writer.add_string("stt.frontend.normalize", hp["fe_normalize"])
    writer.add_string("stt.frontend.pad_mode", hp["fe_pad_mode"])
    writer.add_string("stt.frontend.mel_norm", hp["fe_mel_norm"])
    writer.add_float32("stt.frontend.log_clamp_min", hp["fe_log_clamp_min"])
    writer.add_float32("stt.frontend.mel_lower_hz", hp["fe_mel_lower_hz"])
    writer.add_float32("stt.frontend.mel_upper_hz", hp["fe_mel_upper_hz"])

    # ---- Frontend tensors ----
    n_spec_bins = hp["fe_n_fft"] // 2 + 1
    mel_fb = lasr_mel_filterbank(
        num_mel_bins=hp["fe_num_mels"],
        num_spectrogram_bins=n_spec_bins,
        sample_rate=hp["fe_sample_rate"],
        lower_hz=hp["fe_mel_lower_hz"],
        upper_hz=hp["fe_mel_upper_hz"],
    )
    add_f32_array(writer, "frontend.mel_filterbank", mel_fb)
    window = lasr_hann_window(hp["fe_win_length"])
    add_f32_array(writer, "frontend.window", window)

    # ---- Read weights from safetensors ----
    sf_files = sorted(model_dir.glob("*.safetensors"))
    if not sf_files:
        raise SystemExit(f"no safetensors files in {model_dir}")

    handles = [safe_open(sf, framework="pt") for sf in sf_files]
    key_sets = [set(sh.keys()) for sh in handles]
    consumed: set[str] = set()
    n_emitted = 0

    def get_tensor(name: str) -> torch.Tensor:
        for sh, ks in zip(handles, key_sets):
            if name in ks:
                consumed.add(name)
                return sh.get_tensor(name)
        raise KeyError(f"missing tensor: {name}")

    try:
        all_keys: set = set()
        for ks in key_sets:
            all_keys |= ks

        # Encoder top-level (subsampling + out_norm) + CTC head.
        for src, dst in ENC_TOP_MAP:
            add_tensor(writer, dst, get_tensor(src))
            n_emitted += 1

        # Per-layer Conformer blocks.
        for i in range(hp["enc_n_layers"]):
            for src_suf, dst_suf in ENC_BLOCK_MAP:
                add_tensor(writer, f"enc.blocks.{i}.{dst_suf}",
                           get_tensor(f"encoder.layers.{i}.{src_suf}"))
                n_emitted += 1
            for skip_suf in ENC_BLOCK_SKIP:
                consumed.add(f"encoder.layers.{i}.{skip_suf}")

        # Top-level buffers we deliberately drop (rotary inv_freq is rebuilt in C++).
        for k in TOP_LEVEL_SKIP:
            if k in all_keys:
                consumed.add(k)

        leftover = sorted(all_keys - consumed)
        if leftover:
            print(f"WARNING: {len(leftover)} HF tensors not consumed:")
            for k in leftover[:20]:
                print(f"  {k}")
            if len(leftover) > 20:
                print(f"  ... and {len(leftover) - 20} more")
    finally:
        for sh in handles:
            sh.__exit__(None, None, None)

    print(f"Emitted {n_emitted} weight tensors (plus 2 frontend tensors)")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Done: {out_path}")
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"sha256: {sha}")
    print(f"bytes:  {out_path.stat().st_size:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
