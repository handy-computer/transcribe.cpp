#!/usr/bin/env python3
"""
convert-sava.py - convert Inworld's SAVA release checkpoint (sava.safetensors,
bf16) into a GGUF of the sava family (TURN and VAD roles).

The checkpoint carries its configuration as safetensors metadata; this script
reads it with upstream's own loader (inworld.sava.checkpoint.read), asserts
the values the C++ graph cannot infer from tensor shapes, and writes the
tensor layout documented in docs/porting/families/sava.md. Tensors that are
pure functions of the configuration (the windowed DFT basis, the mel
filterbank, the RoPE frequencies, A = -exp(A_log)) are computed with
upstream's own fp32 code so the C++ never re-derives them.

    uv run --project scripts/envs/sava scripts/convert-sava.py            # F32
    uv run --project scripts/envs/sava scripts/convert-sava.py --outtype bf16

F32 is the reference regime (upstream upcasts the bf16 weights to fp32 on
CPU); BF16 is the lossless source dtype. Either way the per-tensor dtype comes
from lib.gguf_common.reference_dtype_for (norms, biases, frontend tables and
the Mamba-2 scalars / conv kernel stay F32).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, LlamaFileType

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib.gguf_common import (  # noqa: E402
    add_general_identity, encode_for_gguf, gguf_name, gguf_writer, reference_dtype_for)

ARCH = "sava"
DEFAULT_VARIANT = "sava"
HF_REPO = "inworld/sava"
HF_REVISION = "1beae7c1dd0df6b0f92030644fc905eb087dc6a0"
WEIGHTS_SHA256 = "aa610ee9b5d3d1be96f70581a8a3affe7cf20d06816b4ada1efdd8ab6274be34"
SAVA_VERSION = "1.0.0"

FRAME_SAMPLES = 640
NO_SPEECH_LOGIT = -30.0
# The reference turn rule: inworld.sava.segmenter.Policy defaults, asserted
# against the installed package below.
POLICY = {
    "speech_threshold": 0.5,
    "turn_threshold": 0.9,
    "pre_roll_ms": 800,
    "min_silence_before_check_ms": 0,
    "max_silence_ms": 1600,
    "min_speech_ms": 400,
    "min_confident_speech_ms": 200,
    "confident_speech_threshold": 0.9,
}


def fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def resolve_weights(model: str, revision: str) -> Path:
    p = Path(model).expanduser()
    if p.is_file():
        return p.resolve()
    if p.is_dir() and (p / "sava.safetensors").is_file():
        return (p / "sava.safetensors").resolve()
    import huggingface_hub

    return Path(huggingface_hub.hf_hub_download(repo_id=model, filename="sava.safetensors", revision=revision))


def assert_config(cfg, sava) -> None:
    """Values the C++ graph hard-codes or derives, checked against the package."""
    from inworld.sava import config, mel, model as model_lib

    checks = [
        ("SAMPLE_RATE", config.SAMPLE_RATE, 16000),
        ("FRAME_SAMPLES", config.FRAME_SAMPLES, FRAME_SAMPLES),
        ("MEL_BINS", config.MEL_BINS, 128),
        ("MEL_FFT", config.MEL_FFT, 400),
        ("MEL_HOP", config.MEL_HOP, 160),
        ("mel.CONTEXT", mel.CONTEXT, 240),
        ("mel._LOG_FLOOR", mel._LOG_FLOOR, 1e-10),
        ("NO_SPEECH_LOGIT", model_lib.NO_SPEECH_LOGIT, NO_SPEECH_LOGIT),
        ("mamba.num_groups", cfg.mamba.num_groups, 1),
        ("turn_horizons[0]", cfg.turn_horizons[0], 0),
        ("lookahead_frames > 0", cfg.lookahead_frames > 0, True),
        ("mixer_context_frames", cfg.mixer_context_frames, 2),
    ]
    attns = [cfg.attention, cfg.vad_branch.attention, cfg.turn_branch.attention]
    for a in attns:
        checks.append(("rope head_dim / theta shared", (a.head_dim, a.rope_theta),
                       (cfg.attention.head_dim, cfg.attention.rope_theta)))
    for k, v in POLICY.items():
        checks.append((f"Policy.{k}", getattr(sava.Policy(), k), v))
    bad = [(n, got, want) for n, got, want in checks if got != want]
    if bad:
        fail("config mismatch: " + "; ".join(f"{n}={g!r} (want {w!r})" for n, g, w in bad))


def convert(cfg, state: dict[str, torch.Tensor], out_path: Path, outtype: str, sha: str) -> None:
    from inworld.sava import mel, modules

    ref_type = {"f32": GGMLQuantizationType.F32, "bf16": GGMLQuantizationType.BF16}[outtype]
    file_type = {"f32": LlamaFileType.ALL_F32, "bf16": LlamaFileType.MOSTLY_BF16}[outtype]
    sd = {k: v.detach().float() for k, v in state.items()}  # bf16 -> fp32 is exact
    used: set[str] = set()

    def src(key: str) -> torch.Tensor:
        if key not in sd:
            fail(f"checkpoint lacks {key}")
        used.add(key)
        return sd[key]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf_writer(str(out_path), ARCH)
    add_general_identity(
        writer,
        name="SAVA",
        basename="sava",
        size_label="25.5M",
        file_type=int(file_type),
        author="Inworld AI",
        organization="inworld",
        version=SAVA_VERSION,
        license="apache-2.0",
        license_name="Apache License 2.0",
        license_link="https://github.com/inworld-ai/sava/blob/main/LICENSE",
        repo_url=f"https://huggingface.co/{HF_REPO}",
        source_url="https://github.com/inworld-ai/sava",
        description="SAVA (semantic acoustic voice activity): streaming speech and end-of-turn "
                    "probabilities every 40 ms from a causal log-mel front end, a Mamba-2 / "
                    "sliding-window attention trunk and two heads.",
        tags=["voice-activity-detection", "turn-detection", "end-of-turn", "streaming"],
    )
    w = writer
    w.add_string("stt.variant", DEFAULT_VARIANT)
    w.add_string("stt.sava.source_sha256", sha)
    w.add_string("stt.sava.source_revision", HF_REVISION)
    # Front end.
    w.add_uint32("stt.frontend.sample_rate", 16000)
    w.add_uint32("stt.frontend.n_fft", 400)
    w.add_uint32("stt.frontend.win_length", 400)
    w.add_uint32("stt.frontend.hop_length", 160)
    w.add_uint32("stt.frontend.num_mels", 128)
    w.add_string("stt.frontend.window", "hann")
    w.add_uint32("stt.vad.frame_samples", FRAME_SAMPLES)
    w.add_uint32("stt.sava.frame_samples", FRAME_SAMPLES)
    w.add_uint32("stt.sava.context_samples", mel.CONTEXT)
    w.add_float32("stt.sava.mel_floor", mel._LOG_FLOOR)
    w.add_float32("stt.sava.log_energy_floor", cfg.log_energy_floor)
    # Topology.
    w.add_uint32("stt.sava.hidden_size", cfg.hidden_size)
    w.add_uint32("stt.sava.ffn_size", cfg.ffn_size)
    w.add_float32("stt.sava.rms_eps", cfg.rms_eps)
    w.add_array("stt.sava.layers", list(cfg.layers))
    w.add_uint32("stt.sava.mixer_context_frames", cfg.mixer_context_frames)
    m = cfg.mamba
    w.add_uint32("stt.sava.mamba.expand", m.expand)
    w.add_uint32("stt.sava.mamba.head_dim", m.head_dim)
    w.add_uint32("stt.sava.mamba.state_size", m.state_size)
    w.add_uint32("stt.sava.mamba.num_groups", m.num_groups)
    w.add_uint32("stt.sava.mamba.conv_kernel", m.conv_kernel)
    w.add_uint32("stt.sava.mamba.chunk_size", m.chunk_size)

    def add_attention_kv(prefix: str, a) -> None:
        w.add_uint32(f"{prefix}.num_heads", a.num_heads)
        w.add_uint32(f"{prefix}.num_kv_heads", a.num_kv_heads)
        w.add_uint32(f"{prefix}.head_dim", a.head_dim)
        w.add_uint32(f"{prefix}.sliding_window", a.sliding_window)
        w.add_float32(f"{prefix}.rope_theta", a.rope_theta)
        w.add_bool(f"{prefix}.xsa", a.xsa)
        w.add_float32(f"{prefix}.xsa_alpha", a.xsa_alpha)

    add_attention_kv("stt.sava.attention", cfg.attention)
    for branch_name in ("vad_branch", "turn_branch"):
        b = getattr(cfg, branch_name)
        w.add_uint32(f"stt.sava.{branch_name}.hidden_size", b.hidden_size)
        w.add_uint32(f"stt.sava.{branch_name}.ffn_size", b.ffn_size)
        add_attention_kv(f"stt.sava.{branch_name}.attention", b.attention)
    w.add_array("stt.sava.turn_horizons", [int(h) for h in cfg.turn_horizons])
    w.add_uint32("stt.sava.vad_embed_size", cfg.vad_embed_size)
    w.add_uint32("stt.sava.lookahead_frames", cfg.lookahead_frames)
    w.add_float32("stt.sava.no_speech_logit", NO_SPEECH_LOGIT)
    # Decision defaults: turn cut-off (Policy.turn_threshold), speech threshold
    # and the rest of the reference turn policy. Thresholds are float64: the
    # reference compares fp32 probabilities against Python floats, and 0.9
    # rounded to fp32 (0.89999998) would accept p == fl32(0.9).
    w.add_float64("stt.sava.threshold", POLICY["turn_threshold"])
    w.add_float64("stt.sava.speech_threshold", POLICY["speech_threshold"])
    for k in ("pre_roll_ms", "min_silence_before_check_ms", "max_silence_ms", "min_speech_ms",
              "min_confident_speech_ms"):
        w.add_uint32(f"stt.sava.policy.{k}", POLICY[k])
    w.add_float64("stt.sava.policy.confident_speech_threshold", POLICY["confident_speech_threshold"])
    w.add_bool("stt.capability.streaming", True)

    def add(name: str, t: torch.Tensor | np.ndarray) -> None:
        a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
        a = np.ascontiguousarray(a.astype(np.float32))
        target = reference_dtype_for(name, ref_type)
        encoded, raw_dtype = encode_for_gguf(a, target)
        writer.add_tensor(name, encoded, raw_dtype=raw_dtype)

    # Front end: [n_fft, 2 * 201] in torch (windows @ basis) -> numpy [402, 400]
    # = ggml ne [400, 402]: cos rows 0..200, sin rows 201..401, Hann applied.
    add("frontend.dft_basis", mel.dft_basis().T.contiguous())
    add("frontend.mel_filterbank", mel.mel_filterbank())  # [128, 201]
    add("rope.freqs", modules.rope_frequencies(cfg.attention.head_dim, cfg.attention.rope_theta, "cpu"))

    add("mixer.proj.weight", src("mixer.projection.weight"))  # [512, 2 * 513]
    add("mixer.norm_out.weight", src("mixer.norm.weight"))

    def add_mlp(dst: str, sp: str, ffn: int) -> None:
        w13 = src(f"{sp}.mlp.w13.weight")  # rows [gate; up]
        if w13.shape[0] != 2 * ffn:
            fail(f"{sp}.mlp.w13.weight rows {w13.shape[0]} != 2 * {ffn}")
        add(f"{dst}.norm_mlp.weight", src(f"{sp}.mlp_norm.weight"))
        add(f"{dst}.mlp.gate.weight", w13[:ffn])
        add(f"{dst}.mlp.up.weight", w13[ffn:])
        add(f"{dst}.mlp.down.weight", src(f"{sp}.mlp.w2.weight"))

    def add_attention_block(dst: str, sp: str, ffn: int) -> None:
        add(f"{dst}.norm_attn.weight", src(f"{sp}.attn_norm.weight"))
        add(f"{dst}.attn.qkv.weight", src(f"{sp}.attention.qkvr.weight"))  # rows [q; k; v]
        add(f"{dst}.attn.q_norm.weight", src(f"{sp}.attention.q_norm"))
        add(f"{dst}.attn.k_norm.weight", src(f"{sp}.attention.k_norm"))
        add(f"{dst}.attn.out.weight", src(f"{sp}.attention.wo.weight"))
        add_mlp(dst, sp, ffn)

    for i, kind in enumerate(cfg.layers):
        sp, dst = f"trunk.{i}", f"trunk.{i}"
        if kind == "attention":
            add_attention_block(dst, sp, cfg.ffn_size)
            continue
        add(f"{dst}.norm_mix.weight", src(f"{sp}.mixer_norm.weight"))
        add(f"{dst}.ssm.in.weight", src(f"{sp}.mixer.in_proj.weight"))  # rows [z; x; B; C; dt]
        add(f"{dst}.ssm.conv1d.weight", src(f"{sp}.mixer.conv_weight"))  # [conv_dim, K]
        add(f"{dst}.ssm.conv1d.bias", src(f"{sp}.mixer.conv_bias"))
        add(f"{dst}.ssm.dt.bias", src(f"{sp}.mixer.dt_bias"))
        # A = -exp(A_log), as Mamba2Mixer._split computes it in fp32; stored
        # [n_head, 1] = ggml ne [1, n_head], the Mamba-2 layout of ggml_ssm_scan.
        add(f"{dst}.ssm.a", (-torch.exp(src(f"{sp}.mixer.a_log").float())).reshape(-1, 1))
        add(f"{dst}.ssm.d", src(f"{sp}.mixer.d_skip"))
        add(f"{dst}.ssm.norm_gated.weight", src(f"{sp}.mixer.norm"))
        add(f"{dst}.ssm.out.weight", src(f"{sp}.mixer.out_proj.weight"))
        add_mlp(dst, sp, cfg.ffn_size)
    add("trunk.norm_out.weight", src("final_norm.weight"))

    add("head.vad.norm_in.weight", src("vad_norm.weight"))
    add("head.vad.down.weight", src("vad_branch.down.weight"))
    add_attention_block("head.vad.blk", "vad_branch.block", cfg.vad_branch.ffn_size)
    add("head.vad.norm_out.weight", src("vad_branch.out_norm.weight"))
    add("head.vad.readout.weight", src("vad_branch.readout.weight"))  # [1, 512]
    add("head.vad.readout.bias", src("vad_branch.readout.bias"))
    add("head.vad.readout_la.weight", src("vad_la_readout.weight"))  # [1, 512]
    add("head.vad.readout_la.bias", src("vad_la_readout.bias"))

    add("head.turn.norm_in.weight", src("turn_norm.weight"))
    add("head.turn.speech_embed.weight", src("vad_embed.weight").reshape(-1))  # Linear(1, 64)
    add("head.turn.speech_embed.bias", src("vad_embed.bias"))
    add("head.turn.down.weight", src("turn_branch.down.weight"))  # [512, 512 + 64]
    add_attention_block("head.turn.blk", "turn_branch.block", cfg.turn_branch.ffn_size)
    add("head.turn.norm_out.weight", src("turn_branch.out_norm.weight"))
    add("head.turn.readout.weight", src("turn_branch.readout.weight"))  # [4, 512]
    add("head.turn.readout.bias", src("turn_branch.readout.bias"))

    unused = sorted(set(sd) - used)
    if unused:
        fail(f"checkpoint tensors not converted: {unused}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert SAVA (Inworld) to GGUF.")
    p.add_argument("model", nargs="?", default=HF_REPO,
                   help="HF repo id, a directory holding sava.safetensors, or the file (default inworld/sava)")
    p.add_argument("--revision", default=HF_REVISION, help="HF revision when downloading")
    p.add_argument("--outtype", choices=("f32", "bf16"), default="f32")
    p.add_argument("--out", type=Path, default=None,
                   help="Output .gguf path (default: models/sava/sava-<OUTTYPE>.gguf)")
    args = p.parse_args(argv)

    from inworld import sava
    from inworld.sava import checkpoint, registry

    if sava.__version__ != SAVA_VERSION:
        fail(f"inworld-sava {sava.__version__} installed, converter pinned to {SAVA_VERSION}")
    path = resolve_weights(args.model, args.revision)
    sha = registry.sha256_file(str(path))
    if sha != WEIGHTS_SHA256:
        fail(f"{path} sha256 {sha} != pinned {WEIGHTS_SHA256}")
    cfg, state = checkpoint.read(str(path))
    assert_config(cfg, sava)
    # The same state must load strictly into upstream's model (no stray keys).
    checkpoint.build(cfg, state)
    out = args.out or (REPO_ROOT / "models" / DEFAULT_VARIANT / gguf_name(DEFAULT_VARIANT, args.outtype))
    convert(cfg, state, out, args.outtype, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
