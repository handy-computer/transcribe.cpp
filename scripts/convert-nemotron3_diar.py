#!/usr/bin/env python3
"""
convert-nemotron3_diar.py - convert NVIDIA Nemotron-3-Diarization (.nemo,
NeMo SortformerEncLabelModel with a RoPE TransformerEncoder) into a
reference-dtype GGUF.

Nemotron-3-Diarization is an `encoder-diarizer` (Streaming Sortformer
lineage, 8 speakers, 10 ms output): no tokenizer, no decoder, no text.
Pipeline and tensor sources (363 tensors in the .nemo state_dict):

    preprocessor.featurizer.window  -> frontend.window          (BF16-stored symmetric Hann, 400)
    preprocessor.featurizer.fb      -> frontend.mel_filterbank  (BF16-stored slaney mel fb, [128, 257])
    encoder.pre_encode.proj.*       -> enc.pre_encode.proj.*    (feature stacking x8, Linear 1024->512, no bias)
    encoder.embed_norm.*            -> enc.embed.norm.*         (pre-block LayerNorm)
    encoder.layers.{i}.*            -> enc.blocks.{i}.*         (31 pre-LN Transformer blocks, d=512, RoPE)
    encoder.final_norm.*            -> enc.final_norm.*
    sortformer_modules.encoder_proj.*          -> diar.encoder_proj.*     (Linear 512 -> 192)
    sortformer_modules.subpixel_upsample.*     -> diar.upsample.conv.*    (Conv1d 192 -> 192*8, k=3, pad=1)
    sortformer_modules.first_hidden_to_hidden.* -> diar.fc1.*
    sortformer_modules.single_hidden_to_spks.*  -> diar.single_spk_head.* (8 sigmoid outputs)
    sortformer_modules.learnable_sil_emb       -> diar.sil_emb            (AOSC silence embedding, 512)

Dropped (never used by the inference forward):
    sortformer_modules.hidden_to_spks.*  legacy 384-in head, never called
    sortformer_modules.activity_head.*   training-only auxiliary head

Reference dtype is BF16: the .nemo state_dict is all bfloat16. NeMo's
restore_from hands back fp32 tensors, so every tensor is checked to be
BF16-exact before it is written. Storage follows reference_dtype_for:
  - Linear weights                         -> BF16 (lossless)
  - biases, LayerNorms, sil_emb, frontend  -> F32  (lossless upcast)
  - diar.upsample.conv.weight              -> F16  (the loader has no BF16
    conv kernel; BF16 -> F16 rounds only sub-6e-5 weights, max |diff| 3e-8)

The frontend window + filterbank are embedded because NeMo's reference mel
uses the BF16-stored buffers; an fp32 recompute diverges by up to 1.9e-3
in the window.

The fused attention projection is kept fused: enc.blocks.{i}.attn.qkv.weight
is [3*d_model, d_model] in torch order, rows = [q | k | v], each block
laid out head-major (NeMo views it as (T, 3, H, D)).

Usage (via the Nemotron-3-Diarization reference env, which has NeMo Speech):
    uv run --project scripts/envs/nemotron3_diar \
      scripts/convert-nemotron3_diar.py nvidia/Nemotron-3-Diarization \
      --repo-id nvidia/Nemotron-3-Diarization \
      --revision f667ed73aee57d40cc39428eb768b4fd87a0a29e
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, LlamaFileType

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

REFERENCE_TYPE = GGMLQuantizationType.BF16
REFERENCE_DTYPE_LABEL = "BF16"
REFERENCE_FILE_TYPE = LlamaFileType.MOSTLY_BF16

ARCH = "nemotron3_diar"

FRONTEND_TABLE = [
    ("preprocessor.featurizer.window", "frontend.window"),
    ("preprocessor.featurizer.fb",     "frontend.mel_filterbank"),
]

ENCODER_TABLE = [
    ("encoder.pre_encode.proj.weight", "enc.pre_encode.proj.weight"),
    ("encoder.embed_norm.weight",      "enc.embed.norm.weight"),
    ("encoder.embed_norm.bias",        "enc.embed.norm.bias"),
]

# Per-block suffix map (source suffix under encoder.layers.{i}. -> GGUF suffix
# under enc.blocks.{i}.). Norms named norm_1/norm_2 so the "norm_" F32 rule in
# reference_dtype_for / policy.cpp catches them.
BLOCK_TABLE = [
    ("norm1.weight",          "norm_1.weight"),
    ("norm1.bias",            "norm_1.bias"),
    ("attn.w_qkv.weight",     "attn.qkv.weight"),
    ("attn.out_proj.weight",  "attn.out.weight"),
    ("attn.out_proj.bias",    "attn.out.bias"),
    ("norm2.weight",          "norm_2.weight"),
    ("norm2.bias",            "norm_2.bias"),
    ("ffn.net.0.weight",      "ff.in.weight"),
    ("ffn.net.0.bias",        "ff.in.bias"),
    ("ffn.net.3.weight",      "ff.out.weight"),
    ("ffn.net.3.bias",        "ff.out.bias"),
]

FINAL_TABLE = [
    ("encoder.final_norm.weight", "enc.final_norm.weight"),
    ("encoder.final_norm.bias",   "enc.final_norm.bias"),
]

# Diarization projection, subpixel upsampler, head, AOSC silence embedding.
HEAD_TABLE = [
    ("sortformer_modules.encoder_proj.weight",           "diar.encoder_proj.weight"),
    ("sortformer_modules.encoder_proj.bias",             "diar.encoder_proj.bias"),
    ("sortformer_modules.subpixel_upsample.weight",      "diar.upsample.conv.weight"),
    ("sortformer_modules.subpixel_upsample.bias",        "diar.upsample.conv.bias"),
    ("sortformer_modules.first_hidden_to_hidden.weight", "diar.fc1.weight"),
    ("sortformer_modules.first_hidden_to_hidden.bias",   "diar.fc1.bias"),
    ("sortformer_modules.single_hidden_to_spks.weight",  "diar.single_spk_head.weight"),
    ("sortformer_modules.single_hidden_to_spks.bias",    "diar.single_spk_head.bias"),
    ("sortformer_modules.learnable_sil_emb",             "diar.sil_emb"),
]

# state_dict tensors the inference forward never reads.
EXPECTED_UNUSED_PREFIXES = (
    "sortformer_modules.hidden_to_spks.",
    "sortformer_modules.activity_head.",
)


def _to_fp32_bf16_exact(name: str, t) -> np.ndarray:
    import torch
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(t).__name__}")
    t = t.detach().cpu()
    if not torch.equal(t.float(), t.to(torch.bfloat16).float()):
        raise ValueError(f"{name}: not BF16-exact; the .nemo is expected to be all bfloat16")
    return np.ascontiguousarray(t.float().numpy())


def _add(writer, name: str, arr: np.ndarray, counts: dict[str, int]) -> None:
    ggml_type = reference_dtype_for(name, REFERENCE_TYPE)
    data, out_type = encode_for_gguf(arr, ggml_type)
    writer.add_tensor(name, data, raw_dtype=out_type)
    counts[out_type.name] = counts.get(out_type.name, 0) + 1


def _resolve_nemo(model_spec: str, revision: str | None) -> str:
    if model_spec.endswith(".nemo") or Path(model_spec).exists():
        return model_spec
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model_spec, "Nemotron-3-Diarization.nemo", revision=revision)


def convert(model_spec: str, out_path: Path, repo_id: str | None = None, revision: str | None = None) -> None:
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import SortformerEncLabelModel

    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")
    nemo_path = _resolve_nemo(model_spec, revision)
    model = SortformerEncLabelModel.restore_from(restore_path=nemo_path, map_location="cpu", strict=False)
    model.eval()
    cfg = OmegaConf.to_container(model.cfg, resolve=True)
    sd = model.state_dict()
    sd_keys = set(sd)

    enc = cfg["encoder"]
    pre = cfg["preprocessor"]
    sm = cfg["sortformer_modules"]
    if enc.get("self_attention_model") != "rope" or enc.get("subsampling") != "feature_stacking":
        raise ValueError(f"unexpected encoder config: {enc.get('self_attention_model')=} {enc.get('subsampling')=}")
    if enc.get("qkv_bias") or enc.get("qk_norm") or enc.get("xscaling") or not enc.get("pre_block_norm", True):
        raise ValueError("encoder flags differ from the ported graph (qkv_bias/qk_norm/xscaling off, pre_block_norm on)")
    n_layers = int(enc["n_layers"])
    d_model = int(enc["d_model"])
    n_heads = int(enc["n_heads"])
    subsampling = int(enc["subsampling_factor"])
    max_spk = int(cfg.get("max_num_of_spks", sm["num_spks"]))
    out_sub = int(cfg.get("output_subsampling_factor", 1))
    upsample = subsampling // out_sub if cfg.get("high_resolution", False) else 1
    print(f"layers={n_layers} d_model={d_model} heads={n_heads} max_speakers={max_spk} upsample={upsample}")

    writer = gguf_writer(str(out_path), ARCH)

    add_general_identity(
        writer,
        name="Nemotron 3 Diarization",
        basename="nemotron3-diar",
        size_label="100M",
        file_type=REFERENCE_FILE_TYPE,
        languages=["en"],
        author="NVIDIA",
        organization="nvidia",
        license="other",
        license_name="openmdw-1.1",
        repo_url=f"https://huggingface.co/{repo_id}" if repo_id else None,
        description="End-to-end streaming speaker diarizer (encoder-diarizer): 31-layer RoPE Transformer + subpixel upsampler, 8 sigmoid speaker-activity outputs per 10ms frame.",
    )

    writer.add_string("stt.variant", out_path.parent.name)

    # ----- frontend (buffers embedded as frontend.window / frontend.mel_filterbank) -----
    sr = int(pre["sample_rate"])
    hop = int(round(float(pre["window_stride"]) * sr))
    writer.add_uint32("stt.frontend.sample_rate", sr)
    writer.add_uint32("stt.frontend.num_mels", int(pre["features"]))
    writer.add_uint32("stt.frontend.n_fft", int(pre["n_fft"]))
    writer.add_uint32("stt.frontend.hop_length", hop)
    writer.add_uint32("stt.frontend.win_length", int(round(float(pre["window_size"]) * sr)))
    writer.add_string("stt.frontend.window", "hann_symmetric")
    writer.add_string("stt.frontend.normalize", canonicalize_normalize(pre.get("normalize")))
    writer.add_float32("stt.frontend.pre_emphasis", float(pre.get("preemph") if pre.get("preemph") is not None else 0.97))
    # The cfg dither (1e-5) is training-only in NeMo; inference applies none.
    writer.add_float32("stt.frontend.dither", 0.0)

    # ----- capabilities -----
    writer.add_bool("stt.capability.streaming", True)
    writer.add_bool("stt.capability.speaker_diarization", True)
    writer.add_bool("stt.capability.lang_detect", False)
    writer.add_bool("stt.capability.translate", False)
    writer.add_bool("stt.capability.timestamps", False)

    # ----- architecture dims -----
    p = f"stt.{ARCH}"
    writer.add_uint32(f"{p}.max_speakers", max_spk)
    writer.add_uint32(f"{p}.frame_hop", hop * subsampling)        # encoder / AOSC frame, samples
    writer.add_uint32(f"{p}.output_hop", hop * out_sub)           # output frame, samples
    writer.add_uint32(f"{p}.upsample_factor", upsample)
    writer.add_uint32(f"{p}.encoder.n_layers", n_layers)
    writer.add_uint32(f"{p}.encoder.d_model", d_model)
    writer.add_uint32(f"{p}.encoder.n_heads", n_heads)
    writer.add_uint32(f"{p}.encoder.d_ff", int(round(d_model * float(enc["ff_expansion"]))))
    writer.add_uint32(f"{p}.encoder.feat_in", int(enc["feat_in"]))
    writer.add_uint32(f"{p}.encoder.subsampling_factor", subsampling)
    writer.add_string(f"{p}.encoder.subsampling", str(enc["subsampling"]))
    writer.add_float32(f"{p}.encoder.rope_base", float(enc.get("rope_base", 10000.0)))
    writer.add_float32(f"{p}.encoder.rotary_fraction", float(enc.get("rotary_fraction", 1.0)))
    writer.add_float32(f"{p}.encoder.layer_norm_eps", 1e-5)
    writer.add_string(f"{p}.encoder.activation", "gelu")
    writer.add_uint32(f"{p}.head.d_model", int(sm["tf_d_model"]))

    # ----- AOSC compression constants (sortformer_modules) -----
    writer.add_uint32(f"{p}.aosc.spkcache_sil_frames_per_spk", int(sm["spkcache_sil_frames_per_spk"]))
    writer.add_float32(f"{p}.aosc.pred_score_threshold", float(sm["pred_score_threshold"]))
    writer.add_float32(f"{p}.aosc.scores_boost_latest", float(sm["scores_boost_latest"]))
    writer.add_float32(f"{p}.aosc.sil_threshold", float(sm["sil_threshold"]))
    writer.add_float32(f"{p}.aosc.strong_boost_rate", float(sm["strong_boost_rate"]))
    writer.add_float32(f"{p}.aosc.weak_boost_rate", float(sm["weak_boost_rate"]))
    writer.add_float32(f"{p}.aosc.min_pos_scores_rate", float(sm["min_pos_scores_rate"]))
    writer.add_uint32(f"{p}.aosc.max_index", int(sm["max_index"]))

    # ----- default streaming preset (model card very_high_latency, 30.4 s) -----
    # The .nemo sortformer_modules values (fifo 0, chunk 264, rc 0) are
    # training-time settings; the runtime presets live in the runner.
    writer.add_uint32(f"{p}.stream.spkcache_len", 264)
    writer.add_uint32(f"{p}.stream.fifo_len", 40)
    writer.add_uint32(f"{p}.stream.chunk_len", 340)
    writer.add_uint32(f"{p}.stream.chunk_right_context", 40)
    writer.add_uint32(f"{p}.stream.spkcache_update_period", 300)

    # ----- tensors -----
    used: set[str] = set()
    counts: dict[str, int] = {}

    def emit(src: str, dst: str):
        if src not in sd:
            raise KeyError(f"missing expected tensor: {src}")
        arr = _to_fp32_bf16_exact(src, sd[src])
        if dst == "frontend.mel_filterbank" and arr.ndim == 3:
            arr = np.ascontiguousarray(arr[0])  # [1, 128, 257] -> [128, 257]
        _add(writer, dst, arr, counts)
        used.add(src)

    for src, dst in FRONTEND_TABLE + ENCODER_TABLE:
        emit(src, dst)
    for i in range(n_layers):
        for s_suf, d_suf in BLOCK_TABLE:
            emit(f"encoder.layers.{i}.{s_suf}", f"enc.blocks.{i}.{d_suf}")
    for src, dst in FINAL_TABLE + HEAD_TABLE:
        emit(src, dst)

    # ----- unused-key audit -----
    skipped = sorted(sd_keys - used)
    unexpected = [k for k in skipped if not k.startswith(EXPECTED_UNUSED_PREFIXES)]
    if unexpected:
        raise ValueError(f"{len(unexpected)} unmapped state_dict tensors, e.g. {unexpected[:8]}")
    print(f"Emitted {len(used)} tensors {counts}; skipped {len(skipped)}: {skipped}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote GGUF: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="HF repo id or path to a .nemo checkpoint")
    ap.add_argument("--repo-id", default=None, help="HF repo id (for slug + provenance)")
    ap.add_argument("--revision", default=None, help="HF revision to download the .nemo from")
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
    convert(args.model, out_path, repo_id=repo_id, revision=args.revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
