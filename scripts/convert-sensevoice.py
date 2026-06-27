#!/usr/bin/env python3
"""
convert-sensevoice.py - convert a FunASR SenseVoiceSmall checkpoint to a
GGUF that transcribe.cpp's loader can ingest. Preserves the source F32
dtype; block quantization (Q8_0, Q5_K_M, ...) is a Stage 5 concern.

Source format:
    FunASR-native checkpoint directory (HF or local), with:

      config.yaml                         FunASR YAML config
      configuration.json                  FunASR loader manifest
      model.pt                            torch.save() OrderedDict (F32)
      am.mvn                              Kaldi <AddShift> + <Rescale>
                                          stats over 560 LFR features
      chn_jpn_yue_eng_ko_spectok.bpe.model
                                          SentencePiece BPE (25,055 pieces)

Architecture: encoder-CTC. SenseVoiceEncoderSmall is a SAN-M (Self-
Attention with depthwise FSMN memory) transformer with two depth tiers:

    embed (16, 560) — input prefix-token Embedding (lid / event / emo
                       / textnorm direct embeddings, NOT vocabulary
                       outputs)
    encoders0 [1]  — 560 -> 512 projection SAN-M block (norm_attn is
                       560-dim; out projection is 512-dim)
    encoders  [49] — 49 SAN-M blocks at 512-dim (50 main blocks total)
    after_norm     — final LayerNorm of the main tier
    tp_encoders [20]  — 20 SAN-M blocks at 512-dim ("tp" tier)
    tp_norm        — final LayerNorm of the tp tier
    ctc.ctc_lo     — Linear projecting 512 -> 25,055 vocab logits

Per-block layout (13 tensors):
    self_attn.linear_q_k_v.{weight,bias}   fused QKV projection
    self_attn.linear_out.{weight,bias}     attention output projection
    self_attn.fsmn_block.weight            depthwise 1D conv (kernel=11)
                                            for the FSMN memory branch
    feed_forward.w_1.{weight,bias}         FFN up
    feed_forward.w_2.{weight,bias}         FFN down
    norm1.{weight,bias}                    pre-attn LayerNorm (560 in
                                            encoders0[0]; 512 elsewhere)
    norm2.{weight,bias}                    pre-FFN LayerNorm (always 512)

KV emitted:

    general.architecture   = "sensevoice"
    general.basename       = "sensevoice-small"
    general.size_label     = "234M" (or computed)
    general.languages      = [zh, yue, en, ja, ko]

    stt.variant                       = "sensevoice-small"
    stt.capability.lang_detect        = true

    tokenizer.ggml.model              = "spm"
    tokenizer.ggml.tokens             = full SentencePiece vocabulary
    tokenizer.ggml.scores
    tokenizer.ggml.token_type
    tokenizer.ggml.unknown_token_id   (= 0 — also CTC blank id)
    tokenizer.ggml.bos_token_id       (= 1)
    tokenizer.ggml.eos_token_id       (= 2)
    tokenizer.ggml.blank_token_id     (= 0)

    stt.sensevoice.encoder.{n_blocks,tp_blocks,d_model,d_input,n_heads,
                            d_ff,kernel_size,sanm_shift,attention_type,
                            normalize_before}
    stt.sensevoice.special.{lang_zh,lang_en,lang_yue,lang_ja,lang_ko,
                            lang_nospeech, event_speech, event_bgm,
                            event_unk,
                            emotion_happy,emotion_sad,emotion_angry,
                            emotion_neutral, withitn,woitn}

    stt.frontend.{type,num_mels,sample_rate,n_fft,win_length,hop_length,
                  window,normalize,dither,upscale_samples,snip_edges,
                  lfr_m,lfr_n,fbank_style}

CLI:

    # From an HF repo id (downloads via huggingface_hub)
    uv run --project scripts/envs/sensevoice \
      scripts/convert-sensevoice.py FunAudioLLM/SenseVoiceSmall

    # From a local directory
    uv run --project scripts/envs/sensevoice \
      scripts/convert-sensevoice.py \
        ~/.cache/huggingface/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/<sha> \
        --repo-id FunAudioLLM/SenseVoiceSmall

Single-file, top-to-bottom — no hidden helpers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.hf_source import download_snapshot, looks_like_repo_id  # noqa: E402
from lib.gguf_common import (  # noqa: E402
    TOKEN_TYPE_BYTE,
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    TOKEN_TYPE_UNKNOWN,
    TOKEN_TYPE_UNUSED,
    add_general_identity,
    encode_for_gguf,
    gguf_name,
    reference_dtype_for,
    safe_id,
    slug_from_repo_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_DTYPE_LABEL = "F32"
REFERENCE_FILE_TYPE = LlamaFileType.ALL_F32
REFERENCE_GGML_TYPE = GGMLQuantizationType.F32


# Prefix-token row indices into the 16-row enc.embed.weight table.
# These are NOT SentencePiece output IDs — they map a language /
# textnorm choice into the small input-embedding table that gets
# prepended to the encoder input. Values come from
# funasr.models.sense_voice.model.SenseVoiceSmall: lid_dict / textnorm_dict.
# event_emo always uses literal indices [1, 2] (hard-coded in the
# upstream forward), so they do not need to be exposed via KV.
#
# (The C++ output vocabulary's <|en|> = 24885 etc. are unrelated —
# those are how the CTC head LABELS its output, not how the encoder
# is INPUT-prefixed.)
SPECIAL_TOKENS = {
    "lang_auto":        0,
    "lang_zh":          3,
    "lang_en":          4,
    "lang_yue":         7,
    "lang_ja":          11,
    "lang_ko":          12,
    "lang_nospeech":    13,
    # event/emotion are NOT prefix slots — they ride along on the CTC
    # output. The upstream `event_emo_query = embed[[1, 2]]` uses
    # hard-coded indices 1 and 2; expose them here for the loader's
    # documentation only.
    "event_speech":     1,
    "emotion_neutral":  2,
    "withitn":          14,
    "woitn":            15,
}


# Per-block tensor mapping. Same shape applies to encoders0, encoders,
# and tp_encoders — only the dimensions differ for encoders0[0]
# (norm1 + linear_q_k_v see 560-dim input, everything else is 512).
BLOCK_TABLE: list[tuple[str, str]] = [
    ("self_attn.linear_q_k_v.weight",  "attn.qkv.weight"),
    ("self_attn.linear_q_k_v.bias",    "attn.qkv.bias"),
    ("self_attn.linear_out.weight",    "attn.out.weight"),
    ("self_attn.linear_out.bias",      "attn.out.bias"),
    ("self_attn.fsmn_block.weight",    "attn.fsmn.weight"),
    ("feed_forward.w_1.weight",        "ffn.fc1.weight"),
    ("feed_forward.w_1.bias",          "ffn.fc1.bias"),
    ("feed_forward.w_2.weight",        "ffn.fc2.weight"),
    ("feed_forward.w_2.bias",          "ffn.fc2.bias"),
    ("norm1.weight",                   "norm_attn.weight"),
    ("norm1.bias",                     "norm_attn.bias"),
    ("norm2.weight",                   "norm_ffn.weight"),
    ("norm2.bias",                     "norm_ffn.bias"),
]


# ---------------------------------------------------------------------------
# am.mvn parsing
# ---------------------------------------------------------------------------


def parse_am_mvn(am_mvn_path: Path, expected_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Kaldi-style am.mvn file and return (shift, scale) as
    float32 arrays of length `expected_dim`.

    Mirrors funasr.frontends.wav_frontend.load_cmvn. The file contains
    `<AddShift>` (additive bias) and `<Rescale>` (multiplicative gain),
    each followed by a `<LearnRateCoef> 0 [ ... ]` block of floats. The
    forward op is `(x + shift) * scale`.
    """
    lines = am_mvn_path.read_text(encoding="utf-8").splitlines()
    shift: list[float] | None = None
    scale: list[float] | None = None
    for i, line in enumerate(lines):
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "<AddShift>" and shift is None:
            nxt = lines[i + 1].split()
            assert nxt[0] == "<LearnRateCoef>", f"bad am.mvn at line {i+1}: {nxt[0]}"
            shift = [float(x) for x in nxt[3:-1]]
        elif toks[0] == "<Rescale>" and scale is None:
            nxt = lines[i + 1].split()
            assert nxt[0] == "<LearnRateCoef>", f"bad am.mvn at line {i+1}: {nxt[0]}"
            scale = [float(x) for x in nxt[3:-1]]
    if shift is None or scale is None:
        raise ValueError(f"am.mvn missing <AddShift> or <Rescale>: {am_mvn_path}")
    if len(shift) != expected_dim or len(scale) != expected_dim:
        raise ValueError(
            f"am.mvn dim mismatch: shift={len(shift)} scale={len(scale)} "
            f"expected={expected_dim}"
        )
    return (
        np.asarray(shift, dtype=np.float32),
        np.asarray(scale, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Tokenizer extraction (SentencePiece BPE)
# ---------------------------------------------------------------------------


def extract_tokenizer(sp_model_path: Path) -> dict:
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))
    vocab_size = sp.vocab_size()

    tokens: list[str] = []
    scores: list[float] = []
    types:  list[int]   = []

    # SentencePiece marks `<|en|>` / `<|Speech|>` / `<|HAPPY|>` /
    # `<|woitn|>` etc. as `is_control()=False` (i.e. NORMAL) inside the
    # .bpe.model. Semantically they ARE control / metadata tokens — they
    # ride along the CTC output as language / event / emotion / ITN
    # labels, never as transcribed speech. To match the llama.cpp
    # token_type contract (CONTROL = 3) and to let downstream code strip
    # them via Tokenizer::is_control(), we re-classify any piece matching
    # the `<|...|>` pattern as CONTROL. The two genuine SP control tokens
    # `<s>` (id 1) and `</s>` (id 2) are ALSO CONTROL.
    import re as _re
    angle_re = _re.compile(r"^<\|[^|]+\|>$")

    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        if sp.is_unknown(i):
            ttype = TOKEN_TYPE_UNKNOWN
        elif sp.is_control(i) or angle_re.match(piece):
            ttype = TOKEN_TYPE_CONTROL
        elif sp.is_unused(i):
            ttype = TOKEN_TYPE_UNUSED
        elif sp.is_byte(i):
            ttype = TOKEN_TYPE_BYTE
        else:
            ttype = TOKEN_TYPE_NORMAL
        tokens.append(piece)
        scores.append(score)
        types.append(ttype)

    # SentencePiece's vocabulary covers regular pieces plus the
    # SenseVoice control tokens (<|en|>, <|HAPPY|>, ...) which already
    # have is_control()=True semantics inside the .bpe.model. CTC blank
    # is the SP <unk> at id 0; we expose it as both unknown_token_id
    # and blank_token_id so the loader can pick whichever it prefers.

    return {
        "tokens":   tokens,
        "scores":   scores,
        "types":    types,
        "unk_id":   safe_id(sp.unk_id),
        "bos_id":   safe_id(sp.bos_id),
        "eos_id":   safe_id(sp.eos_id),
        "vocab_size": vocab_size,
    }


# ---------------------------------------------------------------------------
# Hparams (config.yaml -> flat dict)
# ---------------------------------------------------------------------------


def read_hparams(config: dict) -> dict:
    enc = config["encoder_conf"]
    fe  = config["frontend_conf"]
    mc  = config["model_conf"]

    n_mels = int(fe["n_mels"])
    lfr_m  = int(fe["lfr_m"])
    d_input = n_mels * lfr_m   # 80 * 7 = 560

    sample_rate  = int(fe["fs"])
    frame_length_ms = int(fe["frame_length"])
    frame_shift_ms  = int(fe["frame_shift"])
    win_length      = int(round(frame_length_ms * sample_rate / 1000))  # 400
    hop_length      = int(round(frame_shift_ms  * sample_rate / 1000))  # 160

    return {
        # Encoder
        "enc_n_blocks":      int(enc["num_blocks"]),
        "enc_tp_blocks":     int(enc["tp_blocks"]),
        "enc_d_model":       int(enc["output_size"]),
        "enc_d_input":       d_input,
        "enc_n_heads":       int(enc["attention_heads"]),
        "enc_d_ff":          int(enc["linear_units"]),
        "enc_kernel_size":   int(enc["kernel_size"]),
        "enc_sanm_shift":    int(enc.get("sanm_shfit", 0)),
        "enc_attn_type":     str(enc["selfattention_layer_type"]),
        "enc_normalize_before": bool(enc.get("normalize_before", True)),

        # Model-level
        "sos":      int(mc["sos"]),
        "eos":      int(mc["eos"]),

        # Frontend (kaldi-style fbank + LFR + per-feature CMVN)
        "fe_type":           "kaldi_fbank_lfr",
        "fe_sample_rate":    sample_rate,
        "fe_num_mels":       n_mels,
        # Kaldi fbank pads the per-frame window to next pow2 (=512 here)
        # for the FFT, but the win_length the window itself sees is the
        # raw 400-sample frame. The C++ frontend will need to mirror this.
        "fe_win_length":     win_length,
        "fe_n_fft":          win_length,
        "fe_hop_length":     hop_length,
        "fe_window":         str(fe["window"]),
        # Per-feature CMVN (the per-feature shift+scale baked under
        # frontend.cmvn.{shift,scale}). The label matches the manifest's
        # generic "per_feature"; kaldi-vs-NeMo disambiguation is via
        # `fbank_style` below.
        "fe_normalize":      "per_feature",
        "fe_dither":         0.0,
        "fe_upscale_samples": True,
        "fe_snip_edges":     True,
        "fe_lfr_m":          lfr_m,
        "fe_lfr_n":          int(fe["lfr_n"]),
        # Marker for the C++ frontend dispatch — distinguishes SenseVoice
        # from Whisper/NeMo mel paths even though many fields overlap.
        "fe_fbank_style":    "kaldi_htk",

        # Languages declared by the SenseVoice vocabulary (LangID prefix
        # tokens 24884-24896). The HF repo card YAML omits "yue"; the
        # vocabulary is authoritative.
        "languages":         ["zh", "yue", "en", "ja", "ko"],
    }


# ---------------------------------------------------------------------------
# Size label
# ---------------------------------------------------------------------------


def compute_size_label(total_params: int) -> str:
    if total_params >= 1_000_000_000:
        return f"{total_params / 1_000_000_000:.1f}B"
    if total_params >= 1_000_000:
        return f"{total_params / 1_000_000:.0f}M"
    return f"{total_params / 1_000:.0f}K"


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def convert(model_dir: Path, out_path: Path, variant: str, repo_id: str | None = None) -> None:
    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")

    config_yaml = model_dir / "config.yaml"
    model_pt    = model_dir / "model.pt"
    am_mvn      = model_dir / "am.mvn"
    bpe_model   = model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"

    for p in (config_yaml, model_pt, am_mvn, bpe_model):
        if not p.is_file():
            raise FileNotFoundError(f"missing required file: {p}")

    print(f"Reading config from {config_yaml}")
    with config_yaml.open() as f:
        config = yaml.safe_load(f)

    hp = read_hparams(config)
    print(
        f"Encoder: {hp['enc_n_blocks']} main + {hp['enc_tp_blocks']} tp blocks; "
        f"d_model={hp['enc_d_model']} d_input={hp['enc_d_input']} "
        f"heads={hp['enc_n_heads']} d_ff={hp['enc_d_ff']} "
        f"kernel={hp['enc_kernel_size']}"
    )

    print(f"Reading tokenizer from {bpe_model}")
    tok = extract_tokenizer(bpe_model)
    print(f"SentencePiece vocab: {tok['vocab_size']}")

    print(f"Reading CMVN stats from {am_mvn}")
    cmvn_shift, cmvn_scale = parse_am_mvn(am_mvn, expected_dim=hp["enc_d_input"])
    print(
        f"CMVN: shift mean={cmvn_shift.mean():.4f} range=[{cmvn_shift.min():.3f}, "
        f"{cmvn_shift.max():.3f}]; scale mean={cmvn_scale.mean():.5f}"
    )

    # SenseVoiceSmall.from_pretrained() under the hood uses
    # torch.load(weights_only=False) because the FunASR registry
    # serialized the OrderedDict with full pickle. We mirror that.
    print(f"Loading state_dict from {model_pt}")
    sd = torch.load(str(model_pt), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and not any(
        k.startswith("encoder.") for k in sd
    ):
        sd = sd["state_dict"]
    sd_keys = set(sd.keys())

    # Confirm dtype from the loaded weights themselves rather than
    # trusting the intake's inferred-from-file-size value.
    dtypes = {v.dtype for v in sd.values()}
    if dtypes != {torch.float32}:
        raise RuntimeError(
            f"unexpected weight dtypes in model.pt: {dtypes!r}; "
            "this converter only handles the published F32 SenseVoiceSmall "
            "checkpoint (no quantized release exists upstream)"
        )

    total_params = sum(int(v.numel()) for v in sd.values())
    size_label = compute_size_label(total_params)
    print(f"Total params: {total_params:,} -> size_label={size_label}")

    print(f"Writing GGUF to {out_path}")
    writer = GGUFWriter(str(out_path), "sensevoice")

    # ----- general.* -----
    # FunASR Model Open Source License Agreement v1.1 attribution
    # requirement (`MODEL_LICENSE` 2.2): "you must attribute the source
    # and author information and retain relevant model names". Bake the
    # canonical attribution into the GGUF KV so downstream consumers
    # see source + author + model names without reading external docs.
    add_general_identity(
        writer,
        name="SenseVoice Small",
        basename=variant,
        size_label=size_label,
        file_type=int(REFERENCE_FILE_TYPE),
        languages=hp["languages"],
        author="Alibaba Group / FunAudioLLM",
        organization="FunAudioLLM",
        license="other",
        license_name="model-license",
        license_link="https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE",
        repo_url=(f"https://huggingface.co/{repo_id}" if repo_id else None),
        url="https://huggingface.co/FunAudioLLM/SenseVoiceSmall",
        source_url="https://github.com/modelscope/FunASR",
        tags=[
            "asr",
            "speech-recognition",
            "encoder-ctc",
            "SenseVoiceSmall",
            "SenseVoiceEncoderSmall",
        ],
        description=(
            "SenseVoiceSmall (Alibaba FunAudioLLM): non-AR "
            "CTC ASR with multilingual + emotion/event/ITN "
            "label heads. Converted from FunAudioLLM/"
            "SenseVoiceSmall; see "
            "https://github.com/modelscope/FunASR/blob/main/"
            "MODEL_LICENSE for FunASR redistribution terms."
        ),
    )

    # ----- stt.variant + capabilities -----
    writer.add_string("stt.variant", variant)
    writer.add_bool("stt.capability.lang_detect", True)

    # ----- tokenizer.ggml.* (SentencePiece BPE) -----
    # llama.cpp tags SentencePiece BPE/unigram vocabularies as "bpe" in
    # tokenizer.ggml.model. The C++ loader's SentencePiece decode path
    # accepts both "unigram" and "bpe"; SenseVoice's
    # chn_jpn_yue_eng_ko_spectok.bpe.model is BPE.
    writer.add_string("tokenizer.ggml.model", "bpe")
    writer.add_array ("tokenizer.ggml.tokens",     tok["tokens"])
    writer.add_array ("tokenizer.ggml.scores",     tok["scores"])
    writer.add_array ("tokenizer.ggml.token_type", tok["types"])
    if tok["unk_id"] is not None:
        writer.add_uint32("tokenizer.ggml.unknown_token_id", tok["unk_id"])
    if tok["bos_id"] is not None:
        writer.add_uint32("tokenizer.ggml.bos_token_id", tok["bos_id"])
    if tok["eos_id"] is not None:
        writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
    # CTC blank id == SP <unk> id (0). Recorded explicitly so the loader
    # does not have to assume the convention.
    writer.add_uint32("tokenizer.ggml.blank_token_id", 0)

    # ----- stt.sensevoice.encoder.* -----
    writer.add_uint32("stt.sensevoice.encoder.n_blocks",     hp["enc_n_blocks"])
    writer.add_uint32("stt.sensevoice.encoder.tp_blocks",    hp["enc_tp_blocks"])
    writer.add_uint32("stt.sensevoice.encoder.d_model",      hp["enc_d_model"])
    writer.add_uint32("stt.sensevoice.encoder.d_input",      hp["enc_d_input"])
    writer.add_uint32("stt.sensevoice.encoder.n_heads",      hp["enc_n_heads"])
    writer.add_uint32("stt.sensevoice.encoder.d_ff",         hp["enc_d_ff"])
    writer.add_uint32("stt.sensevoice.encoder.kernel_size",  hp["enc_kernel_size"])
    writer.add_uint32("stt.sensevoice.encoder.sanm_shift",   hp["enc_sanm_shift"])
    writer.add_string("stt.sensevoice.encoder.attention_type", hp["enc_attn_type"])
    writer.add_bool  ("stt.sensevoice.encoder.normalize_before",
                      hp["enc_normalize_before"])

    # ----- stt.sensevoice.special.* -----
    # The C++ runtime needs these to construct prompts. Emit each as a
    # plain uint32 KV so it survives across schema versions without
    # array-ordering ambiguity.
    for name, tid in SPECIAL_TOKENS.items():
        writer.add_uint32(f"stt.sensevoice.special.{name}", tid)

    # ----- stt.frontend.* -----
    writer.add_string ("stt.frontend.type",          hp["fe_type"])
    writer.add_uint32 ("stt.frontend.num_mels",      hp["fe_num_mels"])
    writer.add_uint32 ("stt.frontend.sample_rate",   hp["fe_sample_rate"])
    writer.add_uint32 ("stt.frontend.n_fft",         hp["fe_n_fft"])
    writer.add_uint32 ("stt.frontend.win_length",    hp["fe_win_length"])
    writer.add_uint32 ("stt.frontend.hop_length",    hp["fe_hop_length"])
    writer.add_string ("stt.frontend.window",        hp["fe_window"])
    writer.add_string ("stt.frontend.normalize",     hp["fe_normalize"])
    writer.add_float32("stt.frontend.dither",        hp["fe_dither"])
    writer.add_bool   ("stt.frontend.upscale_samples", hp["fe_upscale_samples"])
    writer.add_bool   ("stt.frontend.snip_edges",    hp["fe_snip_edges"])
    writer.add_uint32 ("stt.frontend.lfr_m",         hp["fe_lfr_m"])
    writer.add_uint32 ("stt.frontend.lfr_n",         hp["fe_lfr_n"])
    writer.add_string ("stt.frontend.fbank_style",   hp["fe_fbank_style"])

    # ----- tensors -----
    n_added  = 0
    bytes_in = 0
    bytes_out = 0
    consumed: set[str] = set()

    def add_array(dst_name: str, arr: np.ndarray, *, force_f32: bool = False) -> None:
        nonlocal n_added, bytes_in, bytes_out
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if force_f32:
            target_type = GGMLQuantizationType.F32
        else:
            target_type = reference_dtype_for(dst_name, REFERENCE_GGML_TYPE)
        encoded, raw_dtype = encode_for_gguf(np.ascontiguousarray(arr), target_type)
        writer.add_tensor(dst_name, encoded, raw_dtype=raw_dtype)
        bytes_in  += int(arr.nbytes)
        bytes_out += int(encoded.nbytes)
        n_added += 1

    def add(src_name: str, dst_name: str) -> None:
        if src_name not in sd_keys:
            raise KeyError(f"state_dict missing tensor: {src_name!r}")
        consumed.add(src_name)
        t = sd[src_name]
        arr = t.detach().to(dtype=torch.float32).numpy()
        add_array(dst_name, arr)

    # CMVN (per-checkpoint normalization stats — must be baked).
    add_array("frontend.cmvn.shift", cmvn_shift, force_f32=True)
    add_array("frontend.cmvn.scale", cmvn_scale, force_f32=True)

    # Prefix-token embedding.
    add("embed.weight", "enc.embed.weight")

    # encoders0 (1 block: 560-dim input projection block).
    for src_suf, dst_suf in BLOCK_TABLE:
        add(f"encoder.encoders0.0.{src_suf}", f"enc.encoders0.0.{dst_suf}")

    # encoders (49 blocks, 512-dim throughout).
    for i in range(hp["enc_n_blocks"] - 1):  # num_blocks=50 includes encoders0[0]
        for src_suf, dst_suf in BLOCK_TABLE:
            add(f"encoder.encoders.{i}.{src_suf}", f"enc.encoders.{i}.{dst_suf}")

    # after_norm (intermediate LayerNorm between main tier and tp tier).
    add("encoder.after_norm.weight", "enc.after_norm.weight")
    add("encoder.after_norm.bias",   "enc.after_norm.bias")

    # tp_encoders (20 blocks).
    for i in range(hp["enc_tp_blocks"]):
        for src_suf, dst_suf in BLOCK_TABLE:
            add(f"encoder.tp_encoders.{i}.{src_suf}", f"enc.tp_encoders.{i}.{dst_suf}")

    # tp_norm (final LayerNorm).
    add("encoder.tp_norm.weight", "enc.tp_norm.weight")
    add("encoder.tp_norm.bias",   "enc.tp_norm.bias")

    # CTC head.
    add("ctc.ctc_lo.weight", "ctc.head.weight")
    add("ctc.ctc_lo.bias",   "ctc.head.bias")

    expected = (
        2  # cmvn shift + scale
        + 1  # embed
        + len(BLOCK_TABLE)  # encoders0[0]
        + (hp["enc_n_blocks"] - 1) * len(BLOCK_TABLE)
        + 2  # after_norm w+b
        + hp["enc_tp_blocks"] * len(BLOCK_TABLE)
        + 2  # tp_norm w+b
        + 2  # ctc head w+b
    )
    if n_added != expected:
        raise RuntimeError(
            f"tensor count mismatch: added {n_added}, expected {expected}"
        )
    print(f"Added {n_added} tensors "
          f"({bytes_in / (1024 * 1024):.1f} MB fp32 -> "
          f"{bytes_out / (1024 * 1024):.1f} MB on disk)")

    unused = sorted(sd_keys - consumed)
    if unused:
        print(f"WARNING: {len(unused)} state_dict keys not consumed:",
              file=sys.stderr)
        for k in unused[:20]:
            print(f"  {k}", file=sys.stderr)
        if len(unused) > 20:
            print(f"  ... and {len(unused) - 20} more", file=sys.stderr)

    print("Writing header + KV + tensor info...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    print("Writing tensor data...")
    writer.write_tensors_to_file()
    writer.close()

    print(f"Done. Wrote {out_path} "
          f"({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


# Map the upstream HF slug to the canonical kebab-case variant the rest
# of the porting framework uses (manifest, build/validate dir, family
# doc). Keep this explicit rather than camelCase-splitting — there is
# only ever going to be a handful of SenseVoice variants.
SLUG_TO_VARIANT = {
    "SenseVoiceSmall": "sensevoice-small",
}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert a SenseVoice (FunASR) checkpoint to an F32 GGUF.",
    )
    p.add_argument("model", type=str,
                   help="HF repo id (e.g. FunAudioLLM/SenseVoiceSmall) or local dir")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (derived from --repo-id when omitted)")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF repo id used to derive the output slug "
                        "when converting from a local path")
    p.add_argument("--revision", type=str, default=None,
                   help="HF revision to pin the download to (recommended)")
    p.add_argument("--variant", type=str, default=None,
                   help="stt.variant string (default: derived from slug)")
    args = p.parse_args(argv[1:])

    if looks_like_repo_id(args.model):
        repo_id = args.repo_id or args.model
        model_dir = download_snapshot(args.model, args.revision)
    else:
        model_dir = Path(args.model)
        if not model_dir.is_dir():
            print(f"error: {model_dir} is not a directory and not an HF repo id",
                  file=sys.stderr)
            return 2
        repo_id = args.repo_id

    raw_slug = slug_from_repo_id(repo_id) if repo_id else None
    variant = args.variant or (SLUG_TO_VARIANT.get(raw_slug) if raw_slug else None)
    if variant is None:
        if raw_slug:
            variant = raw_slug.lower()
        else:
            print("error: cannot infer variant; pass --variant or --repo-id",
                  file=sys.stderr)
            return 2

    out_path = args.out_path
    if out_path is None:
        # GGUF dir + filename use the upstream HF casing (`raw_slug`); the
        # `stt.variant` KV in the GGUF body stays kebab-case (`variant`)
        # so internal tooling (manifest, build/validate, family doc paths)
        # remains lowercase. Matches the qwen3_asr / parakeet / whisper
        # pattern.
        output_slug = raw_slug or variant
        out_path = REPO_ROOT / "models" / output_slug / gguf_name(output_slug, REFERENCE_DTYPE_LABEL)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    convert(model_dir, out_path, variant, repo_id=repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
