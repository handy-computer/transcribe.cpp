#!/usr/bin/env python3
"""
convert-sortformer.py - convert NVIDIA streaming Sortformer diarizer
(.nemo, NeMo SortformerEncLabelModel) into a reference-dtype GGUF.

Sortformer is an `encoder-diarizer`: no tokenizer, no decoder, no text.
Pipeline and tensor sources (990 tensors in the .nemo state_dict):

    preprocessor.*            -> skipped (C++ recomputes the mel frontend)
    encoder.pre_encode.*      -> enc.pre_encode.*        (dw_striding subsample x8)
    encoder.layers.{i}.*      -> enc.blocks.{i}.*        (17 Conformer/NEST blocks, d=512)
    sortformer_modules.encoder_proj.*  -> diar.encoder_proj.*   (Linear 512 -> 192)
    transformer_encoder.layers.{i}.*   -> tf.blocks.{i}.*       (18 post-LN Transformer blocks, d=192)
    sortformer_modules.first_hidden_to_hidden.* -> diar.fc1.*
    sortformer_modules.hidden_to_spks.*         -> diar.spk_head.*        (4 sigmoid outputs)
    sortformer_modules.single_hidden_to_spks.*  -> diar.single_spk_head.* (single-speaker path)

The Conformer encoder is the same NeMo ConformerEncoder that Parakeet uses,
so the pre-encode + block tensor tables mirror scripts/convert-parakeet.py
(use_bias=true, conv_norm_type=batch_norm with running stats).

Reference dtype is F32 (NeMo diarizer state_dict is fp32).

Usage (via the Sortformer reference env, which has NeMo):
    uv run --project scripts/envs/sortformer \
      scripts/convert-sortformer.py nvidia/diar_streaming_sortformer_4spk-v2.1 \
      --repo-id nvidia/diar_streaming_sortformer_4spk-v2.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    add_general_identity,
    canonicalize_normalize,
    encode_for_gguf,
    gguf_name,
    gguf_writer,
    reference_dtype_for,
    slug_from_repo_id,
)

REFERENCE_TYPE = GGMLQuantizationType.F32
REFERENCE_DTYPE_LABEL = "F32"
REFERENCE_FILE_TYPE = 0  # LlamaFileType ALL_F32

# --- Conformer encoder tables (same NeMo ConformerEncoder as Parakeet) -------

PRE_ENCODE_TABLE = [
    ("encoder.pre_encode.conv.0.weight", "enc.pre_encode.conv.0.weight"),
    ("encoder.pre_encode.conv.0.bias",   "enc.pre_encode.conv.0.bias"),
    ("encoder.pre_encode.conv.2.weight", "enc.pre_encode.conv.2.weight"),
    ("encoder.pre_encode.conv.2.bias",   "enc.pre_encode.conv.2.bias"),
    ("encoder.pre_encode.conv.3.weight", "enc.pre_encode.conv.3.weight"),
    ("encoder.pre_encode.conv.3.bias",   "enc.pre_encode.conv.3.bias"),
    ("encoder.pre_encode.conv.5.weight", "enc.pre_encode.conv.5.weight"),
    ("encoder.pre_encode.conv.5.bias",   "enc.pre_encode.conv.5.bias"),
    ("encoder.pre_encode.conv.6.weight", "enc.pre_encode.conv.6.weight"),
    ("encoder.pre_encode.conv.6.bias",   "enc.pre_encode.conv.6.bias"),
    ("encoder.pre_encode.out.weight",    "enc.pre_encode.out.weight"),
    ("encoder.pre_encode.out.bias",      "enc.pre_encode.out.bias"),
]

# Per-conformer-block suffix map (source suffix under encoder.layers.{i}. ->
# GGUF suffix under enc.blocks.{i}.). Norms keep the "norm_" prefix and BN
# params keep ".bn." so reference_dtype_for / policy.cpp route them to F32.
CONFORMER_BLOCK_TABLE = [
    ("norm_feed_forward1.weight",   "norm_ff1.weight"),
    ("norm_feed_forward1.bias",     "norm_ff1.bias"),
    ("feed_forward1.linear1.weight", "ff1.linear1.weight"),
    ("feed_forward1.linear1.bias",   "ff1.linear1.bias"),
    ("feed_forward1.linear2.weight", "ff1.linear2.weight"),
    ("feed_forward1.linear2.bias",   "ff1.linear2.bias"),
    ("norm_self_att.weight",        "norm_attn.weight"),
    ("norm_self_att.bias",          "norm_attn.bias"),
    ("self_attn.linear_q.weight",   "attn.linear_q.weight"),
    ("self_attn.linear_q.bias",     "attn.linear_q.bias"),
    ("self_attn.linear_k.weight",   "attn.linear_k.weight"),
    ("self_attn.linear_k.bias",     "attn.linear_k.bias"),
    ("self_attn.linear_v.weight",   "attn.linear_v.weight"),
    ("self_attn.linear_v.bias",     "attn.linear_v.bias"),
    ("self_attn.linear_out.weight", "attn.linear_out.weight"),
    ("self_attn.linear_out.bias",   "attn.linear_out.bias"),
    ("self_attn.linear_pos.weight", "attn.linear_pos.weight"),
    ("self_attn.pos_bias_u",        "attn.pos_bias_u"),
    ("self_attn.pos_bias_v",        "attn.pos_bias_v"),
    ("norm_conv.weight",            "norm_conv.weight"),
    ("norm_conv.bias",              "norm_conv.bias"),
    ("conv.pointwise_conv1.weight", "conv.pointwise1.weight"),
    ("conv.pointwise_conv1.bias",   "conv.pointwise1.bias"),
    ("conv.depthwise_conv.weight",  "conv.depthwise.weight"),
    ("conv.depthwise_conv.bias",    "conv.depthwise.bias"),
    ("conv.batch_norm.weight",      "conv.bn.weight"),
    ("conv.batch_norm.bias",        "conv.bn.bias"),
    ("conv.batch_norm.running_mean", "conv.bn.running_mean"),
    ("conv.batch_norm.running_var",  "conv.bn.running_var"),
    ("conv.pointwise_conv2.weight", "conv.pointwise2.weight"),
    ("conv.pointwise_conv2.bias",   "conv.pointwise2.bias"),
    ("norm_feed_forward2.weight",   "norm_ff2.weight"),
    ("norm_feed_forward2.bias",     "norm_ff2.bias"),
    ("feed_forward2.linear1.weight", "ff2.linear1.weight"),
    ("feed_forward2.linear1.bias",   "ff2.linear1.bias"),
    ("feed_forward2.linear2.weight", "ff2.linear2.weight"),
    ("feed_forward2.linear2.bias",   "ff2.linear2.bias"),
    ("norm_out.weight",             "norm_out.weight"),
    ("norm_out.bias",               "norm_out.bias"),
]

# Post-LN Transformer encoder block (NeMo TransformerEncoder, pre_ln=false).
# Norms named norm_1/norm_2 so the "norm_" F32 rule catches them.
TRANSFORMER_BLOCK_TABLE = [
    ("layer_norm_1.weight",              "norm_1.weight"),
    ("layer_norm_1.bias",                "norm_1.bias"),
    ("first_sub_layer.query_net.weight", "attn.q.weight"),
    ("first_sub_layer.query_net.bias",   "attn.q.bias"),
    ("first_sub_layer.key_net.weight",   "attn.k.weight"),
    ("first_sub_layer.key_net.bias",     "attn.k.bias"),
    ("first_sub_layer.value_net.weight", "attn.v.weight"),
    ("first_sub_layer.value_net.bias",   "attn.v.bias"),
    ("first_sub_layer.out_projection.weight", "attn.out.weight"),
    ("first_sub_layer.out_projection.bias",   "attn.out.bias"),
    ("layer_norm_2.weight",              "norm_2.weight"),
    ("layer_norm_2.bias",                "norm_2.bias"),
    ("second_sub_layer.dense_in.weight", "ff.in.weight"),
    ("second_sub_layer.dense_in.bias",   "ff.in.bias"),
    ("second_sub_layer.dense_out.weight", "ff.out.weight"),
    ("second_sub_layer.dense_out.bias",   "ff.out.bias"),
]

# Diarization projection + head (sortformer_modules).
HEAD_TABLE = [
    ("sortformer_modules.encoder_proj.weight",          "diar.encoder_proj.weight"),
    ("sortformer_modules.encoder_proj.bias",            "diar.encoder_proj.bias"),
    ("sortformer_modules.first_hidden_to_hidden.weight", "diar.fc1.weight"),
    ("sortformer_modules.first_hidden_to_hidden.bias",   "diar.fc1.bias"),
    ("sortformer_modules.hidden_to_spks.weight",        "diar.spk_head.weight"),
    ("sortformer_modules.hidden_to_spks.bias",          "diar.spk_head.bias"),
    ("sortformer_modules.single_hidden_to_spks.weight", "diar.single_spk_head.weight"),
    ("sortformer_modules.single_hidden_to_spks.bias",   "diar.single_spk_head.bias"),
]

# state_dict keys the loader recomputes / does not need.
EXPECTED_UNUSED_PREFIXES = ("preprocessor.",)
EXPECTED_UNUSED_SUFFIXES = (".num_batches_tracked",)


def _to_fp32(t) -> np.ndarray:
    import torch
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(t).__name__}")
    if t.dtype != torch.float32:
        raise ValueError(f"expected fp32 tensor, got {t.dtype} — cast at the source")
    return np.ascontiguousarray(t.detach().cpu().numpy())


def _add(writer, name: str, arr: np.ndarray) -> None:
    ggml_type = reference_dtype_for(name, REFERENCE_TYPE)
    data, out_type = encode_for_gguf(arr, ggml_type)
    writer.add_tensor(name, data, raw_dtype=out_type)


def convert(model_spec: str, out_path: Path, repo_id: str | None = None) -> None:
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import SortformerEncLabelModel

    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")
    if model_spec.endswith(".nemo") or Path(model_spec).exists():
        model = SortformerEncLabelModel.restore_from(restore_path=model_spec, map_location="cpu", strict=False)
    else:
        model = SortformerEncLabelModel.from_pretrained(model_spec, map_location="cpu")
    model.eval()
    cfg = OmegaConf.to_container(model.cfg, resolve=True)
    sd = model.state_dict()
    sd_keys = set(sd)

    enc = cfg["encoder"]
    tfc = cfg["transformer_encoder"]
    pre = cfg["preprocessor"]
    n_conf = int(enc["n_layers"])
    n_tf = int(tfc["num_layers"])
    max_spk = int(cfg.get("max_num_of_spks", 4))
    print(f"conformer_layers={n_conf} transformer_layers={n_tf} max_speakers={max_spk}")

    writer = gguf_writer(str(out_path), "sortformer")

    add_general_identity(
        writer,
        name="Streaming Sortformer Diarizer 4spk v2.1",
        basename="sortformer",
        size_label="117M",
        version="v2.1",
        file_type=REFERENCE_FILE_TYPE,
        languages=["en"],
        author="NVIDIA",
        organization="nvidia",
        license="other",
        license_name="nvidia-open-model-license",
        license_link="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
        repo_url=f"https://huggingface.co/{repo_id}" if repo_id else None,
        description="End-to-end streaming speaker diarizer (encoder-diarizer): NEST/FastConformer + Transformer, 4 sigmoid speaker-activity outputs per 80ms frame.",
    )

    writer.add_string("stt.variant", out_path.parent.name)

    # ----- frontend (C++ recomputes; these are the reference params) -----
    sr = int(pre["sample_rate"])
    writer.add_uint32("stt.frontend.sample_rate", sr)
    writer.add_uint32("stt.frontend.num_mels", int(pre["features"]))
    writer.add_uint32("stt.frontend.n_fft", int(pre["n_fft"]))
    writer.add_uint32("stt.frontend.hop_length", int(round(float(pre["window_stride"]) * sr)))
    writer.add_uint32("stt.frontend.win_length", int(round(float(pre["window_size"]) * sr)))
    writer.add_string("stt.frontend.window", str(pre.get("window", "hann")))
    writer.add_string("stt.frontend.normalize", canonicalize_normalize(pre.get("normalize")))
    writer.add_float32("stt.frontend.pre_emphasis", float(pre.get("preemph") if pre.get("preemph") is not None else 0.97))
    writer.add_float32("stt.frontend.dither", float(pre.get("dither", 0.0)))

    # ----- capabilities -----
    writer.add_bool("stt.capability.streaming", True)
    writer.add_bool("stt.capability.speaker_diarization", True)
    writer.add_bool("stt.capability.lang_detect", False)
    writer.add_bool("stt.capability.translate", False)
    writer.add_bool("stt.capability.timestamps", False)

    # ----- architecture dims -----
    writer.add_uint32("stt.sortformer.max_speakers", max_spk)
    writer.add_uint32("stt.sortformer.frame_hop", int(round(float(pre["window_stride"]) * sr)) * int(enc["subsampling_factor"]))
    writer.add_uint32("stt.sortformer.encoder.n_layers", n_conf)
    writer.add_uint32("stt.sortformer.encoder.d_model", int(enc["d_model"]))
    writer.add_uint32("stt.sortformer.encoder.n_heads", int(enc["n_heads"]))
    writer.add_uint32("stt.sortformer.encoder.d_ff", int(enc["d_model"]) * int(enc["ff_expansion_factor"]))
    writer.add_uint32("stt.sortformer.encoder.conv_kernel", int(enc["conv_kernel_size"]))
    writer.add_uint32("stt.sortformer.encoder.subsampling_factor", int(enc["subsampling_factor"]))
    writer.add_uint32("stt.sortformer.encoder.subsampling_channels", int(enc["subsampling_conv_channels"]))
    writer.add_uint32("stt.sortformer.encoder.feat_in", int(enc["feat_in"]))
    writer.add_uint32("stt.sortformer.encoder.pos_emb_max_len", int(enc.get("pos_emb_max_len", 5000)))
    writer.add_string("stt.sortformer.encoder.conv_norm_type", str(enc.get("conv_norm_type", "batch_norm")))
    writer.add_uint32("stt.sortformer.transformer.n_layers", n_tf)
    writer.add_uint32("stt.sortformer.transformer.d_model", int(tfc["hidden_size"]))
    writer.add_uint32("stt.sortformer.transformer.n_heads", int(tfc["num_attention_heads"]))
    writer.add_uint32("stt.sortformer.transformer.d_ff", int(tfc["inner_size"]))
    writer.add_string("stt.sortformer.transformer.activation", str(tfc.get("hidden_act", "relu")))
    writer.add_bool("stt.sortformer.transformer.pre_ln", bool(tfc.get("pre_ln", False)))

    # ----- shipped streaming defaults (runtime-tunable presets live in the runner) -----
    sm = cfg["sortformer_modules"]
    writer.add_uint32("stt.sortformer.stream.chunk_len", int(sm["chunk_len"]))
    writer.add_uint32("stt.sortformer.stream.spkcache_len", int(sm["spkcache_len"]))
    writer.add_uint32("stt.sortformer.stream.fifo_len", int(sm["fifo_len"]))
    writer.add_uint32("stt.sortformer.stream.spkcache_update_period", int(sm["spkcache_update_period"]))

    # ----- tensors -----
    used: set[str] = set()

    def emit(src: str, dst: str):
        if src not in sd:
            raise KeyError(f"missing expected tensor: {src}")
        _add(writer, dst, _to_fp32(sd[src]))
        used.add(src)

    for src, dst in PRE_ENCODE_TABLE:
        emit(src, dst)
    for i in range(n_conf):
        for s_suf, d_suf in CONFORMER_BLOCK_TABLE:
            emit(f"encoder.layers.{i}.{s_suf}", f"enc.blocks.{i}.{d_suf}")
    for i in range(n_tf):
        for s_suf, d_suf in TRANSFORMER_BLOCK_TABLE:
            emit(f"transformer_encoder.layers.{i}.{s_suf}", f"tf.blocks.{i}.{d_suf}")
    for src, dst in HEAD_TABLE:
        emit(src, dst)

    # ----- unused-key audit -----
    unexpected = []
    for k in sd_keys - used:
        if k.startswith(EXPECTED_UNUSED_PREFIXES) or k.endswith(EXPECTED_UNUSED_SUFFIXES):
            continue
        unexpected.append(k)
    if unexpected:
        raise ValueError(f"{len(unexpected)} unmapped state_dict tensors, e.g. {sorted(unexpected)[:8]}")
    print(f"Emitted {len(used)} tensors ({len(sd_keys) - len(used)} skipped: preprocessor + BN counters)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote GGUF: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="HF repo id or path to a .nemo checkpoint")
    ap.add_argument("--repo-id", default=None, help="HF repo id (for slug + provenance)")
    ap.add_argument("--out", default=None, help="Override output GGUF path")
    args = ap.parse_args()

    repo_id = args.repo_id or (args.model if "/" in args.model and not Path(args.model).exists() else None)
    if args.out:
        out_path = Path(args.out)
    else:
        if not repo_id:
            raise SystemExit("error: pass --repo-id (or a HF repo id) so the output slug can be derived")
        slug = slug_from_repo_id(repo_id)
        out_path = Path("models") / slug / gguf_name(slug, REFERENCE_DTYPE_LABEL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    convert(args.model, out_path, repo_id=repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
