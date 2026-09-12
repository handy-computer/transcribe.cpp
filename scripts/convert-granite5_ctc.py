#!/usr/bin/env python3
"""
convert-granite5_ctc.py - convert the IBM Granite Speech 5.0 TurboCTC HF
checkpoint to a transcribe.cpp GGUF.

TurboCTC is a pure encoder-CTC model: a 16-block Granite Conformer encoder
with block-local Shaw attention, in-block time subsampling, self-conditioned
CTC, and a tied 16384-way BPE CTC head. No projector, no LLM, no decoder loop.
We emit the architecture key `granite_speech5_ctc` (HF `model_type`) so the
C++ loader dispatches to src/arch/granite5_ctc/.

HF state dict (550 tensors, BF16 except the BatchNorm running stats):

  encoder.input_linear.{weight,bias}          # 320 -> 1024
  encoder.out.{weight,bias}                   # CTC head: 1024 -> 16384
  encoder.out_mid.{weight,bias}               # self-cond bypass: 16384 -> 1024
  encoder.layers.{i}.norm_feed_forward1.*
  encoder.layers.{i}.feed_forward1.linear{1,2}.*
  encoder.layers.{i}.norm_self_att.*
  encoder.layers.{i}.self_attn.{q,k,v}_proj.weight     # no bias
  encoder.layers.{i}.self_attn.o_proj.{weight,bias}
  encoder.layers.{i}.self_attn.rel_pos_emb.weight      # [1025, 128]
  encoder.layers.{i}.norm_conv.*
  encoder.layers.{i}.conv.pointwise_lin1.*             # 1024 -> 4096 (GLU)
  encoder.layers.{i}.conv.depthwise_conv.weight        # [2048, 1, 7]
  encoder.layers.{i}.conv.norm.*                       # BatchNorm1d(2048)
  encoder.layers.{i}.conv.pointwise_lin2.*             # 2048 -> 1024
  encoder.layers.{i}.norm_feed_forward2.*
  encoder.layers.{i}.feed_forward2.linear{1,2}.*
  encoder.layers.{i}.norm_out.*

There is NO `ctc_head.*` in the checkpoint: config.tie_word_embeddings=true
ties ctc_head.{weight,bias} to encoder.out.{weight,bias} (verified in Stage 2
by identical data_ptr()). `enc.ctc_proj` therefore serves both the mid-layer
self-conditioning logits and the final logits.

Three mapping decisions are made here rather than deferred to the loader; see
docs/porting/families/granite5_ctc.md "Notes" for the full rationale:

  1. k_proj and v_proj are CONCATENATED into a single `attn.kv` tensor
     ([hidden, 2*inner], K rows first) so src/granite_conformer/shaw_attn.h —
     which takes a fused `attn_kv_w` — is reusable verbatim. Reversible:
     rows [0, inner) are K, rows [inner, 2*inner) are V.
  2. GGUF block tensor names follow the granite 4.x layout emitted by
     convert-granite.py (`attn.q` / `attn.kv` / `attn.out` /
     `attn.rel_pos_emb`, `conv.pointwise{1,2}` / `conv.depthwise` /
     `conv.bn.*`), because Stage 4 forks that encoder. The two names where
     granite 4.x diverges from this checkpoint use the checkpoint's own
     vocabulary instead: `ff{1,2}.linear{1,2}` (HF `feed_forward*.linear*`,
     granite 4.x `ff*.up`/`.down`) and `norm_out` (HF `norm_out`,
     granite 4.x `norm_post`). Both also match src/conformer/conformer.h's
     BlockView field names.
  3. `conv.norm.num_batches_tracked` (I64, 16 of them) is dropped. It is a
     training counter with no inference role.

The mel filterbank and STFT window are baked from torchaudio, NOT librosa:
the reference frontend is torchaudio.transforms.MelSpectrogram at library
defaults, so torchaudio's own htk/norm=None fbank is the bit-exact source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType
from safetensors.torch import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.hf_source import resolve_model_dir  # noqa: E402
from lib.gguf_common import (  # noqa: E402
    add_general_identity,
    canonicalize_normalize,
    encode_for_gguf,
    gguf_writer,
    reference_dtype_for,
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
)

ARCH = "granite_speech5_ctc"

# The shipped reference tier is BF16 (the checkpoint's own storage dtype).
# F32 is a diagnostic tier, not a release artifact: it makes the C++ graph
# comparable to the F32 oracle with zero weight-rounding, which is how
# Stage 4 separates real graph drift from BF16 storage + BF16-activation
# matmul drift. F16 is here for completeness; Stage 5 owns the real matrix.
DTYPE_TIERS = {
    "bf16": (GGMLQuantizationType.BF16, LlamaFileType.MOSTLY_BF16, "BF16"),
    "f16":  (GGMLQuantizationType.F16,  LlamaFileType.MOSTLY_F16,  "F16"),
    "f32":  (GGMLQuantizationType.F32,  LlamaFileType.ALL_F32,     "F32"),
}


# ----- Tokenizer ------------------------------------------------------------


def read_tokenizer(model_dir: Path) -> dict:
    """tokenizer.json (GPT-2-style byte-level BPE, 16384 entries) -> GGUF arrays.

    The vocabulary is contiguous over [0, 16384) and the single added token
    `<|blank|>` occupies id 0, where it doubles as the CTC blank and the
    pad_token_id. No bos/eos/unk: a CTC model never encodes text.
    """
    tok = json.loads((model_dir / "tokenizer.json").read_text())
    model = tok["model"]
    vocab = model["vocab"]
    merges = model.get("merges") or []
    added = tok.get("added_tokens") or []

    vocab_size = max(vocab.values()) + 1
    for at in added:
        vocab_size = max(vocab_size, at["id"] + 1)

    tokens = [""] * vocab_size
    types = [TOKEN_TYPE_NORMAL] * vocab_size
    for piece, idx in vocab.items():
        tokens[idx] = piece
    for at in added:
        tokens[at["id"]] = at["content"]
        types[at["id"]] = TOKEN_TYPE_CONTROL

    missing = [i for i, t in enumerate(tokens) if t == ""]
    if missing:
        raise ValueError(
            f"tokenizer vocabulary has {len(missing)} gaps "
            f"(first at id {missing[0]}); expected a contiguous table"
        )

    # tokenizers >= 0.20 stores merges as [left, right] pairs.
    norm_merges = []
    for m in merges:
        if isinstance(m, list):
            if len(m) != 2:
                raise ValueError(f"malformed merge entry: {m!r}")
            norm_merges.append(f"{m[0]} {m[1]}")
        else:
            norm_merges.append(m)

    blank_id = None
    for at in added:
        if at["content"] == "<|blank|>":
            blank_id = at["id"]
    if blank_id is None:
        raise ValueError("tokenizer.json has no <|blank|> added token")

    return {
        "tokens": tokens,
        "types": types,
        "merges": norm_merges,
        "blank_id": blank_id,
    }


# ----- Hyperparameters ------------------------------------------------------


def read_hparams(config: dict, preproc: dict, processor: dict) -> dict:
    """Merge config.json + the two front-end config files into GGUF hparams.

    The repo ships two front-end descriptions that must agree:
      processor_config.json   mainline GraniteSpeech5FeatureExtractor (the one
                              transformers>=5.16 actually instantiates), which
                              names the stacked width as `feature_size`.
      preprocessor_config.json  the stale trust_remote_code CtcConformerProcessor,
                              the only file that spells `stack_factor`/`deltas`.
    Mainline wins on every shared field; the stale file is read only for the two
    keys mainline omits, and every overlapping value is cross-checked so a
    future snapshot cannot let them drift apart silently.
    """
    enc = config["encoder_config"]
    fe = (processor or {}).get("feature_extractor") or {}
    hp: dict = {}

    n_layers = int(enc["num_hidden_layers"])
    hp["enc_n_layers"] = n_layers
    hp["enc_hidden"] = int(enc["hidden_size"])
    hp["enc_n_heads"] = int(enc["num_attention_heads"])
    hp["enc_head_dim"] = int(enc["head_dim"])
    hp["enc_intermediate"] = int(enc["intermediate_size"])
    hp["enc_conv_kernel_size"] = int(enc["conv_kernel_size"])
    hp["enc_conv_expansion"] = int(enc["conv_expansion_factor"])
    hp["enc_context_size"] = int(enc["context_size"])
    hp["enc_max_pos_emb"] = int(enc["max_position_embeddings"])
    hp["enc_hidden_act"] = str(enc["hidden_act"])
    hp["enc_num_mel_bins"] = int(enc["num_mel_bins"])
    hp["enc_vocab_size"] = int(enc["vocab_size"])
    # Blocks that halve the time axis (stride-2 depthwise + mean-pooled
    # residual). [0, 1] here: 50 Hz -> 25 Hz -> 12.5 Hz.
    hp["enc_subsample_layers"] = [int(x) for x in enc["subsample_layers"]]
    # GraniteSpeech5Encoder injects out_mid(softmax(out(h))) after block
    # index num_hidden_layers // 2 - 1, i.e. before block num_hidden_layers//2.
    # We store the index of the block whose OUTPUT is conditioned.
    hp["enc_self_cond_layer"] = n_layers // 2
    # config.attention_bias=true does NOT put a bias on q/k/v (the encoder
    # attention hard-codes bias=False there); it reaches the inherited
    # feed-forward, giving both FFN linears a bias. o_proj carries its own.
    hp["enc_attention_bias"] = bool(enc.get("attention_bias", False))

    hp["blank_id"] = int(config.get("pad_token_id", 0))
    hp["vocab_size"] = int(config["vocab_size"])
    hp["tie_word_embeddings"] = bool(config.get("tie_word_embeddings", True))

    # Frontend: torchaudio MelSpectrogram at library defaults (power=2.0,
    # periodic Hann, center=True, reflect pad, mel_scale=htk, norm=None),
    # then log10 -> per-utterance floor at (global max - 8 dB) -> x/4 + 1,
    # concat compute_deltas (80 -> 160), stack 2 frames (160 -> 320).
    def pick(mainline_key, legacy_key, default):
        a = fe.get(mainline_key)
        b = preproc.get(legacy_key)
        if a is not None and b is not None and a != b:
            raise SystemExit(
                f"front-end config disagreement: processor_config.json"
                f"[{mainline_key}]={a!r} != preprocessor_config.json"
                f"[{legacy_key}]={b!r}"
            )
        return a if a is not None else (b if b is not None else default)

    hp["fe_type"] = "mel"
    hp["fe_sample_rate"] = int(pick("sampling_rate", "sample_rate", 16000))
    hp["fe_num_mels"] = int(pick("num_mel_bins", "n_mels", 80))
    hp["fe_n_fft"] = int(pick("n_fft", "n_fft", 512))
    hp["fe_win_length"] = int(pick("win_length", "win_length", 400))
    hp["fe_hop_length"] = int(pick("hop_length", "hop_length", 160))
    hp["fe_window"] = "hann_periodic"
    hp["fe_normalize"] = canonicalize_normalize("per_utterance")
    hp["fe_pad_mode"] = "reflect"
    hp["fe_mel_norm"] = "htk"
    hp["fe_dither"] = 0.0
    hp["fe_delta_win_length"] = int(pick("delta_win_length", "delta_win_length", 3))
    hp["fe_logmel_floor_db"] = float(pick("logmel_floor_db", "logmel_floor_db", 8.0))
    # Mainline omits both; only the legacy processor spells them out.
    hp["fe_deltas"] = bool(preproc.get("deltas", True))
    hp["fe_stack_factor"] = int(preproc.get("stack_factor", 2))

    mel_ch = hp["fe_num_mels"] * (2 if hp["fe_deltas"] else 1)
    hp["enc_input_dim"] = mel_ch * hp["fe_stack_factor"]

    # Three independent statements of the stacked encoder width must agree:
    # the mainline feature extractor's feature_size, the derived
    # (mel x delta x stack) product, and the encoder's input_linear.
    feature_size = fe.get("feature_size")
    if feature_size is not None and int(feature_size) != hp["enc_input_dim"]:
        raise SystemExit(
            f"front-end width disagreement: processor_config.json"
            f"[feature_size]={feature_size} but "
            f"{hp['fe_num_mels']} mel x {2 if hp['fe_deltas'] else 1} "
            f"x stack {hp['fe_stack_factor']} = {hp['enc_input_dim']}"
        )
    return hp


def emit_mel_filterbank_and_window(writer: GGUFWriter, hp: dict) -> None:
    """Bake the torchaudio mel filterbank + periodic Hann window as tensors.

    Taken from torchaudio rather than librosa because the reference front-end
    IS torchaudio.transforms.MelSpectrogram: `melscale_fbanks` with
    norm=None / mel_scale="htk" is the same call MelSpectrogram makes
    internally, so the baked buffer is bit-exact against the oracle instead of
    merely close.
    """
    n_fft = hp["fe_n_fft"]
    n_mels = hp["fe_num_mels"]
    sr = hp["fe_sample_rate"]
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=float(sr) / 2.0,
        n_mels=n_mels,
        sample_rate=sr,
        norm=None,
        mel_scale="htk",
    )  # [n_freqs, n_mels]
    fb = fb.transpose(0, 1).contiguous().numpy().astype(np.float32)  # [n_mels, n_freqs]
    writer.add_tensor("frontend.mel_filterbank", fb)

    window = torch.hann_window(hp["fe_win_length"], periodic=True)
    writer.add_tensor("frontend.window", window.numpy().astype(np.float32))


# ----- Tensor name maps -----------------------------------------------------

ENC_TOP_MAP = [
    ("encoder.input_linear.weight", "enc.input_linear.weight"),
    ("encoder.input_linear.bias",   "enc.input_linear.bias"),
    # Tied to ctc_head.{weight,bias}; used for BOTH the mid-layer
    # self-conditioning logits and the final CTC logits.
    ("encoder.out.weight",          "enc.ctc_proj.weight"),
    ("encoder.out.bias",            "enc.ctc_proj.bias"),
    ("encoder.out_mid.weight",      "enc.ctc_bypass.weight"),
    ("encoder.out_mid.bias",        "enc.ctc_bypass.bias"),
]

# Per-block 1:1 renames. k_proj/v_proj are absent here — they are fused into
# `attn.kv` separately (see fuse_kv below).
ENC_BLOCK_MAP = [
    # Macaron FF1.
    ("norm_feed_forward1.weight",    "norm_ff1.weight"),
    ("norm_feed_forward1.bias",      "norm_ff1.bias"),
    ("feed_forward1.linear1.weight", "ff1.linear1.weight"),
    ("feed_forward1.linear1.bias",   "ff1.linear1.bias"),
    ("feed_forward1.linear2.weight", "ff1.linear2.weight"),
    ("feed_forward1.linear2.bias",   "ff1.linear2.bias"),

    # Block-local Shaw attention.
    ("norm_self_att.weight",         "norm_attn.weight"),
    ("norm_self_att.bias",           "norm_attn.bias"),
    ("self_attn.q_proj.weight",      "attn.q.weight"),
    ("self_attn.o_proj.weight",      "attn.out.weight"),
    ("self_attn.o_proj.bias",        "attn.out.bias"),
    ("self_attn.rel_pos_emb.weight", "attn.rel_pos_emb.weight"),

    # Conv module: pointwise -> GLU -> depthwise(stride 1 or 2) -> BN -> SiLU
    # -> pointwise. pointwise_lin{1,2} are nn.Linear here (2-D [out, in]),
    # not Conv1d, but they are the same 1x1 operator the conformer helper
    # expects. The 2-D layout intentionally routes them to the Linear quant
    # bucket, and the loader accepts them with GET_LIN.
    ("norm_conv.weight",             "norm_conv.weight"),
    ("norm_conv.bias",               "norm_conv.bias"),
    ("conv.pointwise_lin1.weight",   "conv.pointwise1.weight"),
    ("conv.pointwise_lin1.bias",     "conv.pointwise1.bias"),
    ("conv.depthwise_conv.weight",   "conv.depthwise.weight"),
    ("conv.norm.weight",             "conv.bn.weight"),
    ("conv.norm.bias",               "conv.bn.bias"),
    ("conv.norm.running_mean",       "conv.bn.running_mean"),
    ("conv.norm.running_var",        "conv.bn.running_var"),
    ("conv.pointwise_lin2.weight",   "conv.pointwise2.weight"),
    ("conv.pointwise_lin2.bias",     "conv.pointwise2.bias"),

    # Macaron FF2.
    ("norm_feed_forward2.weight",    "norm_ff2.weight"),
    ("norm_feed_forward2.bias",      "norm_ff2.bias"),
    ("feed_forward2.linear1.weight", "ff2.linear1.weight"),
    ("feed_forward2.linear1.bias",   "ff2.linear1.bias"),
    ("feed_forward2.linear2.weight", "ff2.linear2.weight"),
    ("feed_forward2.linear2.bias",   "ff2.linear2.bias"),

    # Per-block final LayerNorm.
    ("norm_out.weight",              "norm_out.weight"),
    ("norm_out.bias",                "norm_out.bias"),
]

# Read but never emitted: a training-only counter.
ENC_BLOCK_DROP = ("conv.norm.num_batches_tracked",)


def add_tensor(writer: GGUFWriter, name: str, arr: np.ndarray,
               reference_type: GGMLQuantizationType) -> int:
    """Emit one tensor at the dtype gguf_common.reference_dtype_for picks.

    That helper mirrors tools/transcribe-quantize/policy.cpp::classify_tensor,
    so the reference GGUF and the Stage 5 quantizer agree on which tensors are
    pinned F32 (norms, biases, BN stats, frontend buffers) and which downcast
    to F16 (conv kernels, which the loader has no BF16 path for).
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    ggml_type = reference_dtype_for(name, reference_type)
    packed, raw_type = encode_for_gguf(arr, ggml_type)
    writer.add_tensor(name, packed, raw_dtype=raw_type)
    return int(packed.nbytes)


def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).contiguous().cpu().numpy()


def compute_size_label(total_params: int) -> str:
    if total_params >= 1_000_000_000:
        return f"{total_params / 1_000_000_000:.1f}B"
    if total_params >= 1_000_000:
        return f"{total_params / 1_000_000:.0f}M"
    return f"{total_params / 1_000:.0f}K"


def total_safetensors_params(model_dir: Path) -> int:
    """Element count across all shards, header-only (no tensor data read)."""
    total = 0
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(sf, framework="pt") as h:
            for k in h.keys():
                total += int(np.prod(h.get_slice(k).get_shape()))
    return total


# ----- Main -----------------------------------------------------------------


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("model", help="HF repo id or local model directory")
    p.add_argument("--repo-id", default=None,
                   help="HF repo id used to derive the output filename "
                        "(defaults to the last segment of `model`)")
    p.add_argument("--revision", default=None,
                   help="HF revision (ignored for local paths)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: models/<repo-last-segment>/)")
    p.add_argument("--reference-dtype", default="bf16", choices=sorted(DTYPE_TIERS),
                   help="Storage tier for the weight bucket (default: bf16, the "
                        "checkpoint's own dtype and the shipped reference GGUF). "
                        "f32 is the Stage-4 diagnostic tier: it isolates C++ graph "
                        "drift from BF16 weight/activation rounding.")
    args = p.parse_args(argv)

    reference_ggml_type, reference_file_type, ref_dtype = DTYPE_TIERS[args.reference_dtype]

    repo_id = args.repo_id or args.model
    variant = repo_id.split("/")[-1]

    model_dir = resolve_model_dir(args.model, args.revision)
    print(f"Source: {model_dir}")

    config = json.loads((model_dir / "config.json").read_text())
    preproc = json.loads((model_dir / "preprocessor_config.json").read_text())
    processor_path = model_dir / "processor_config.json"
    processor = json.loads(processor_path.read_text()) if processor_path.exists() else {}

    arch_declared = (config.get("architectures") or [None])[0]
    if arch_declared != "GraniteSpeech5ForCTC":
        raise SystemExit(
            f"unexpected architecture {arch_declared!r}; "
            f"convert-granite5_ctc.py handles GraniteSpeech5ForCTC only"
        )

    hp = read_hparams(config, preproc, processor)
    tok = read_tokenizer(model_dir)

    if tok["blank_id"] != hp["blank_id"]:
        raise SystemExit(
            f"blank id disagreement: tokenizer <|blank|>={tok['blank_id']} "
            f"but config.pad_token_id={hp['blank_id']}"
        )
    if len(tok["tokens"]) != hp["vocab_size"]:
        raise SystemExit(
            f"vocab size disagreement: tokenizer={len(tok['tokens'])} "
            f"but config.vocab_size={hp['vocab_size']}"
        )

    print(f"variant: {variant}")
    print(f"  encoder: {hp['enc_n_layers']} blocks, hidden={hp['enc_hidden']}, "
          f"{hp['enc_n_heads']}x{hp['enc_head_dim']} heads, "
          f"ffn={hp['enc_intermediate']}, conv_k={hp['enc_conv_kernel_size']}")
    print(f"  subsample blocks: {hp['enc_subsample_layers']}, "
          f"self-cond after block {hp['enc_self_cond_layer'] - 1}, "
          f"block-local context={hp['enc_context_size']}")
    print(f"  input_dim: {hp['enc_input_dim']} "
          f"(({hp['fe_num_mels']} mel + delta) x {hp['fe_stack_factor']})")
    print(f"  CTC vocab: {hp['vocab_size']} (blank={hp['blank_id']}), "
          f"{len(tok['merges'])} merges, tied head={hp['tie_word_embeddings']}")

    outdir = Path(args.outdir or f"models/{variant}").expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{variant}-{ref_dtype}.gguf"
    print(f"Writing GGUF: {out_path}")

    size_label = compute_size_label(total_safetensors_params(model_dir))
    print(f"  params: ~{size_label}")

    writer = gguf_writer(str(out_path), ARCH)

    # ---- general.* ----
    add_general_identity(
        writer,
        name="Granite Speech 5.0 470M TurboCTC",
        basename="granite-speech-turboctc",
        version="5.0",
        size_label=size_label,
        file_type=int(reference_file_type),
        languages=["en"],
        author="IBM",
        organization="ibm-granite",
        license="apache-2.0",
        license_name="Apache License 2.0",
        license_link="https://www.apache.org/licenses/LICENSE-2.0",
        repo_url=f"https://huggingface.co/{repo_id}",
        description=(
            "English encoder-CTC ASR. 16-block Granite Conformer encoder with "
            "block-local Shaw attention, 8x temporal subsampling and "
            "self-conditioned CTC over a 16384-entry BPE vocabulary; "
            "non-autoregressive greedy decode."
        ),
        tags=["automatic-speech-recognition", "ctc", "conformer"],
    )

    # ---- stt.variant + capabilities ----
    writer.add_string("stt.variant", variant)
    writer.add_bool("stt.capability.translate", False)
    writer.add_bool("stt.capability.lang_detect", False)
    writer.add_bool("stt.capability.streaming", False)
    writer.add_bool("stt.capability.speaker_diarization", False)
    # Upstream exposes no timing alignment. CTC emission peaks are not word
    # durations, so the runtime intentionally rejects every timestamp request.
    writer.add_bool("stt.capability.timestamps", False)
    writer.add_bool("stt.capability.word_timestamps", False)

    # ---- stt.granite5_ctc.* ----
    writer.add_uint32("stt.granite5_ctc.encoder.n_layers",        hp["enc_n_layers"])
    writer.add_uint32("stt.granite5_ctc.encoder.hidden",          hp["enc_hidden"])
    writer.add_uint32("stt.granite5_ctc.encoder.n_heads",         hp["enc_n_heads"])
    writer.add_uint32("stt.granite5_ctc.encoder.head_dim",        hp["enc_head_dim"])
    writer.add_uint32("stt.granite5_ctc.encoder.intermediate",    hp["enc_intermediate"])
    writer.add_uint32("stt.granite5_ctc.encoder.input_dim",       hp["enc_input_dim"])
    writer.add_uint32("stt.granite5_ctc.encoder.output_dim",      hp["enc_vocab_size"])
    writer.add_uint32("stt.granite5_ctc.encoder.conv_kernel_size", hp["enc_conv_kernel_size"])
    writer.add_uint32("stt.granite5_ctc.encoder.conv_expansion",  hp["enc_conv_expansion"])
    writer.add_uint32("stt.granite5_ctc.encoder.context_size",    hp["enc_context_size"])
    writer.add_uint32("stt.granite5_ctc.encoder.max_pos_emb",     hp["enc_max_pos_emb"])
    writer.add_uint32("stt.granite5_ctc.encoder.self_cond_layer", hp["enc_self_cond_layer"])
    writer.add_string("stt.granite5_ctc.encoder.hidden_act",      hp["enc_hidden_act"])
    writer.add_bool("stt.granite5_ctc.encoder.attention_bias",    hp["enc_attention_bias"])
    writer.add_array("stt.granite5_ctc.encoder.subsample_layers",
                     [int(x) for x in hp["enc_subsample_layers"]])
    writer.add_uint32("stt.granite5_ctc.blank_id", hp["blank_id"])
    writer.add_bool("stt.granite5_ctc.tie_word_embeddings", hp["tie_word_embeddings"])

    # ---- stt.frontend.* ----
    writer.add_string("stt.frontend.type",         hp["fe_type"])
    writer.add_uint32("stt.frontend.sample_rate",  hp["fe_sample_rate"])
    writer.add_uint32("stt.frontend.num_mels",     hp["fe_num_mels"])
    writer.add_uint32("stt.frontend.n_fft",        hp["fe_n_fft"])
    writer.add_uint32("stt.frontend.win_length",   hp["fe_win_length"])
    writer.add_uint32("stt.frontend.hop_length",   hp["fe_hop_length"])
    writer.add_string("stt.frontend.window",       hp["fe_window"])
    writer.add_string("stt.frontend.normalize",    hp["fe_normalize"])
    writer.add_string("stt.frontend.pad_mode",     hp["fe_pad_mode"])
    writer.add_string("stt.frontend.mel_norm",     hp["fe_mel_norm"])
    writer.add_float32("stt.frontend.dither",      hp["fe_dither"])
    # granite5-specific frontend stages that follow the log-mel: first-order
    # deltas concatenated onto the mel channels, then 2-frame stacking.
    writer.add_bool("stt.frontend.deltas",             hp["fe_deltas"])
    writer.add_uint32("stt.frontend.delta_win_length", hp["fe_delta_win_length"])
    writer.add_float32("stt.frontend.logmel_floor_db", hp["fe_logmel_floor_db"])
    writer.add_uint32("stt.frontend.stack_factor",     hp["fe_stack_factor"])

    # ---- tokenizer.ggml.* ----
    writer.add_string("tokenizer.ggml.model", "gpt2")
    # Byte-level BPE with case-insensitive contractions and \p{N}{1,3} digit
    # runs — the same regex granite 4.x ships, hence the same flavor key.
    writer.add_string("tokenizer.ggml.pre",   "granite")
    writer.add_array("tokenizer.ggml.tokens",     tok["tokens"])
    writer.add_array("tokenizer.ggml.token_type", tok["types"])
    writer.add_array("tokenizer.ggml.merges",     tok["merges"])
    # <|blank|> is both the CTC blank and pad_token_id. No bos/eos/unk: the
    # model never encodes text.
    writer.add_uint32("tokenizer.ggml.blank_token_id",   tok["blank_id"])
    writer.add_uint32("tokenizer.ggml.padding_token_id", tok["blank_id"])

    # ---- Frontend tensors ----
    emit_mel_filterbank_and_window(writer, hp)

    # ---- Weights ----
    sf_files = sorted(model_dir.glob("*.safetensors"))
    if not sf_files:
        raise SystemExit(f"no safetensors files in {model_dir}")

    n_emitted = 0
    bytes_out = 0
    consumed: set[str] = set()
    handles = [safe_open(sf, framework="pt") for sf in sf_files]
    key_sets = [set(h.keys()) for h in handles]

    def get(src_key: str) -> torch.Tensor:
        for h, keys in zip(handles, key_sets):
            if src_key in keys:
                consumed.add(src_key)
                return h.get_tensor(src_key)
        raise KeyError(f"missing tensor: {src_key}")

    def emit(src_key: str, dst_key: str) -> None:
        nonlocal n_emitted, bytes_out
        bytes_out += add_tensor(writer, dst_key, to_f32(get(src_key)), reference_ggml_type)
        n_emitted += 1

    def fuse_kv(i: int) -> None:
        """Concatenate k_proj and v_proj into the fused `attn.kv` tensor.

        shaw_block_attn() splits it with the equivalent of
        `kv.chunk(2, dim=-1)`, so K must occupy the FIRST `inner` output rows
        and V the second — matching torch's [out_features, in_features]
        row-major layout.
        """
        nonlocal n_emitted, bytes_out
        k = to_f32(get(f"encoder.layers.{i}.self_attn.k_proj.weight"))
        v = to_f32(get(f"encoder.layers.{i}.self_attn.v_proj.weight"))
        if k.shape != v.shape:
            raise ValueError(f"layer {i}: k/v shape mismatch {k.shape} vs {v.shape}")
        kv = np.concatenate([k, v], axis=0)
        bytes_out += add_tensor(writer, f"enc.blocks.{i}.attn.kv.weight", kv, reference_ggml_type)
        n_emitted += 1

    try:
        all_keys: set[str] = set()
        for ks in key_sets:
            all_keys |= ks

        # Third witness for the stacked front-end width (see read_hparams).
        in_w = get("encoder.input_linear.weight").shape
        if tuple(in_w) != (hp["enc_hidden"], hp["enc_input_dim"]):
            raise SystemExit(
                f"encoder.input_linear.weight is {tuple(in_w)}, expected "
                f"({hp['enc_hidden']}, {hp['enc_input_dim']}) from the "
                f"front-end config"
            )

        for src, dst in ENC_TOP_MAP:
            emit(src, dst)

        for i in range(hp["enc_n_layers"]):
            for src_suf, dst_suf in ENC_BLOCK_MAP:
                emit(f"encoder.layers.{i}.{src_suf}", f"enc.blocks.{i}.{dst_suf}")
            fuse_kv(i)
            for drop in ENC_BLOCK_DROP:
                consumed.add(f"encoder.layers.{i}.{drop}")

        leftover = sorted(all_keys - consumed)
        if leftover:
            raise SystemExit(
                f"{len(leftover)} HF tensors were not consumed, first 10: "
                f"{leftover[:10]}"
            )
    finally:
        for h in handles:
            h.__exit__(None, None, None)

    print(f"Emitted {n_emitted} tensors ({bytes_out:,} bytes of tensor data)")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Done: {out_path}")
    print(f"sha256: {hashlib.sha256(out_path.read_bytes()).hexdigest()}")
    print(f"bytes:  {out_path.stat().st_size:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
