#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "librosa>=0.10",
#     "numpy>=1.26",
#     "onnx>=1.16",
#     "onnxruntime>=1.17",
#     "soundfile>=0.12",
#     "parakeet-mlx>=0.3.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
#     "mlx>=0.20; sys_platform == 'darwin' and platform_machine == 'arm64'",
# ]
# ///
"""
dump_reference.py - generate per-stage reference tensors for the
transcribe.cpp numerical accuracy harness.

Each subcommand dumps one or more named tensors as a pair of files:

    <name>.f32   raw little-endian float32, row-major
    <name>.json  sidecar metadata (shape, dtype, source provenance)

Compared against the equivalent C++ dump (from src/transcribe-debug.{h,cpp})
by scripts/compare_tensors.py. The two-file format is symmetric: the
C++ TensorDumper writes the same shape/json pair, so a comparator can
diff a C++ dump dir against a Python dump dir without translation.

Subcommands:
    mel-onnx    Run a NeMo .onnx preprocessor (e.g. nemo128.onnx) on a
                wav and dump the resulting log-mel as the byte-level
                reference. PLAN.md mandates this as the source for the
                Parakeet frontend golden test (open question #1).
    encoder     Run parakeet-mlx's encoder forward on a wav and dump
                named intermediates per-block. Apple Silicon only —
                parakeet-mlx and mlx are only installable there.
    decode      Run parakeet-mlx's predictor + joint forward at the
                first decode step (start state, encoder frame 0) and
                dump the intermediates the C++ decoder also dumps
                under TRANSCRIBE_DUMP_DIR. The first-step set is the
                bring-up bare minimum: dec.embed.0, dec.lstm.{0..N-1}.{h,c}.0,
                dec.joint.0, dec.enc_out. Apple Silicon only.

Usage:
    uv run scripts/dump_reference.py mel-onnx \\
        --onnx ~/code/cjpais/transcribe-rs/models/parakeet-v3-fp32/nemo128.onnx \\
        --audio samples/jfk.wav \\
        --out tests/golden/parakeet \\
        --name jfk-mel

    uv run scripts/dump_reference.py encoder \\
        --model ~/code/cjpais/transcribe-rs/models/parakeet-tdt-0.6b-v2-mlx \\
        --audio samples/jfk.wav \\
        --out /tmp/transcribe-ref-dump

    uv run scripts/dump_reference.py decode \\
        --model ~/code/cjpais/transcribe-rs/models/parakeet-tdt-0.6b-v2-mlx \\
        --audio samples/jfk.wav \\
        --out /tmp/transcribe-ref-dump
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def write_dump(
    out_dir: Path,
    name: str,
    data: np.ndarray,
    *,
    source: dict,
    stage: str,
) -> None:
    """Write a (<name>.f32, <name>.json) pair into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if data.dtype != np.float32:
        raise ValueError(f"only float32 tensors are supported, got {data.dtype}")
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    f32_path = out_dir / f"{name}.f32"
    json_path = out_dir / f"{name}.json"
    data.tofile(f32_path)
    meta = {
        "name": name,
        "stage": stage,
        "shape": list(data.shape),
        "dtype": "f32",
        "layout": "row-major",
        "source": source,
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {f32_path} ({data.size * 4} bytes)")
    print(f"  wrote {json_path}")


def cmd_mel_onnx(args: argparse.Namespace) -> int:
    import onnxruntime as ort
    import soundfile as sf

    onnx_path = Path(args.onnx).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser()

    if not onnx_path.exists():
        print(f"error: onnx file not found: {onnx_path}", file=sys.stderr)
        return 1
    if not audio_path.exists():
        print(f"error: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    pcm, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # nemo128.onnx contract (and the same shape every NeMo
    # AudioToMelSpectrogramPreprocessor export uses):
    #   inputs : waveforms[B, N] f32, waveforms_lens[B] i64
    #   outputs: features[B, n_mels, T] f32, features_lens[B] i64
    waveforms = pcm[None, :]
    waveforms_lens = np.array([pcm.size], dtype=np.int64)
    features, features_lens = sess.run(
        None, {"waveforms": waveforms, "waveforms_lens": waveforms_lens}
    )
    mel = features[0].astype(np.float32)  # [n_mels, T]

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr}")
    print(f"mel  : shape={mel.shape} dtype={mel.dtype}")
    print(
        f"       min={mel.min():.4f} max={mel.max():.4f} "
        f"mean={mel.mean():.6f} std={mel.std():.6f}"
    )
    print(f"features_lens={features_lens.tolist()}")

    # The provenance fields capture only what's reproducible across
    # machines: file basenames, sample counts, sample rate. Absolute
    # paths would make the json dirty across checkouts.
    write_dump(
        out_dir,
        args.name,
        mel,
        source={
            "kind": "onnx",
            "onnx": onnx_path.name,
            "audio": audio_path.name,
            "n_samples": int(pcm.size),
            "sample_rate": int(sr),
        },
        stage="frontend.mel.norm",
    )
    return 0


def cmd_encoder(args: argparse.Namespace) -> int:
    """Dump per-stage encoder activations from parakeet-mlx.

    Phase 4 step 3a only dumps `enc.pre_encode.out`. Subsequent
    sub-stages (3b-3f) will hook the conformer blocks the same way
    and dump per-block intermediates.

    Mel source: defaults to parakeet-mlx's own `audio.get_logmel`,
    but if `--mel-from-cpp <prefix>` is supplied, the script reads
    a `<prefix>.f32` + `<prefix>.json` dump (from the C++ side)
    and uses that instead. The mel-from-cpp path is the right
    choice for encoder validation — it isolates encoder
    correctness from any frontend disagreement between
    parakeet-mlx's audio pipeline and NeMo's reference (the phase 3
    preflight already noted parakeet-mlx uses periodic Hann
    where NeMo uses symmetric).
    """
    import mlx.core as mx
    from parakeet_mlx import audio as mlx_audio
    from parakeet_mlx.utils import from_pretrained

    model_path = Path(args.model).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve() if args.audio else None
    out_dir = Path(args.out).expanduser()

    if not model_path.exists():
        print(f"error: model path not found: {model_path}", file=sys.stderr)
        return 1

    # Load model in fp32 — the C++ encoder runs fp32 throughout, so
    # the reference must too. parakeet-mlx defaults to bfloat16 which
    # would introduce ~3-4 decimal digits of jitter, swamping any
    # real encoder bug.
    print(f"Loading parakeet-mlx model from {model_path} (fp32)...")
    model = from_pretrained(str(model_path), dtype=mx.float32)

    if args.mel_from_cpp is not None:
        # Read a C++ dump pair: <prefix>.f32 + <prefix>.json. The
        # JSON gives us the slow-to-fast shape; for the C++ mel
        # dump it's [n_mels, T_mel] (the natural slow-to-fast view
        # of the row-major [n_mels, T_mel] buffer). parakeet-mlx
        # expects [B, T, n_mels], so we transpose to [T, n_mels]
        # then add a batch dim.
        prefix = Path(args.mel_from_cpp).expanduser().resolve()
        f32_path = Path(str(prefix) + ".f32")
        json_path = Path(str(prefix) + ".json")
        if not f32_path.exists() or not json_path.exists():
            print(f"error: mel dump pair not found at {prefix}.{{f32,json}}",
                  file=sys.stderr)
            return 1
        meta = json.loads(json_path.read_text())
        cpp_shape = tuple(meta["shape"])
        if len(cpp_shape) != 2:
            print(f"error: expected 2-D mel, got shape {cpp_shape}", file=sys.stderr)
            return 1
        n_mels, T_mel = cpp_shape
        cpp_mel = np.fromfile(f32_path, dtype=np.float32).reshape(cpp_shape)
        # Transpose to [T, n_mels] + add batch -> [1, T, n_mels].
        mel_np = cpp_mel.T[None, :, :].astype(np.float32, copy=True)
        mel = mx.array(mel_np)
        print(f"mel (from C++ dump): cpp_shape={cpp_shape} -> mlx_shape={list(mel.shape)}")
    else:
        if audio_path is None or not audio_path.exists():
            print(f"error: audio file required when --mel-from-cpp is not given",
                  file=sys.stderr)
            return 1
        print(f"Loading audio {audio_path}...")
        audio_data = mlx_audio.load_audio(
            audio_path, model.preprocessor_config.sample_rate, mx.float32
        )
        mel = mlx_audio.get_logmel(audio_data, model.preprocessor_config)
        # mel shape: [B=1, T, n_mels] (channels-last)
        print(f"mel (parakeet-mlx): shape={list(mel.shape)} dtype={mel.dtype}")

    source = {
        "kind": "parakeet-mlx",
        "model": model_path.name,
    }
    if audio_path is not None:
        source["audio"] = audio_path.name
    if args.mel_from_cpp is not None:
        source["mel_from_cpp"] = Path(args.mel_from_cpp).name

    def to_np(t: "mx.array") -> np.ndarray:
        """Eval an MLX tensor and return a contiguous fp32 numpy
        array with leading size-1 dims squeezed (so on-disk shapes
        match the C++ side, where ggml's slow-to-fast layout
        converter drops trailing-1s = leading-1s in numpy)."""
        mx.eval(t)
        a = np.array(t, copy=False).astype(np.float32)
        while a.ndim > 1 and a.shape[0] == 1:
            a = a[0]
        return np.ascontiguousarray(a)

    def dump(name: str, t: "mx.array", stage: str) -> None:
        a = to_np(t)
        print(f"  {name}: shape={a.shape} "
              f"min={a.min():.4f} max={a.max():.4f} mean={a.mean():.6f}")
        write_dump(out_dir, name, a, source=source, stage=stage)

    # ----- 1. pre_encode --------------------------------------------
    lengths = mx.array([mel.shape[1]], dtype=mx.int32)
    pre_out, out_lengths = model.encoder.pre_encode(mel, lengths)
    dump("enc.pre_encode.out", pre_out, "encoder.pre_encode")

    # ----- 2. positional encoding -----------------------------------
    #
    # Sinusoidal RelPositionalEncoding (NOT learned). The C++ side
    # computes the same buffer at graph build time; this is the
    # reference. xscaling=False on Parakeet so the input `x` is
    # passed through unchanged.
    pos_enc = model.encoder.pos_enc
    x_after_pos, pos_emb = pos_enc(pre_out, offset=0)
    # x_after_pos should equal pre_out (xscaling False); not dumping it.
    dump("enc.pos_emb", pos_emb, "encoder.pos_emb")

    # ----- 3. per-block intermediates -------------------------------
    #
    # Walk every conformer block, but only dump intermediates from
    # the small set the C++ side asks for. The block list is
    # configurable via --blocks (default: 0, 12, 23). For block 0
    # we dump every sub-step (ff1, attn, conv, ff2, out) so the
    # 3b-3e bring-up loop has full visibility.
    block_set = set(args.blocks)
    full_blocks = {0}  # always dump every sub-step on block 0
    x = pre_out
    for i, block in enumerate(model.encoder.layers):
        # Macaron FF1.
        ff1_in = block.norm_feed_forward1(x)
        ff1_out = block.feed_forward1(ff1_in)
        x = x + 0.5 * ff1_out
        if i in full_blocks:
            dump(f"enc.block.{i}.ff1", x, f"encoder.block{i}.ff1")

        # Self-attention with relative position. Full residual.
        x_attn_in = block.norm_self_att(x)
        attn_out = block.self_attn(
            x_attn_in, x_attn_in, x_attn_in,
            mask=None, pos_emb=pos_emb, cache=None,
        )
        x = x + attn_out
        if i in full_blocks:
            dump(f"enc.block.{i}.attn", x, f"encoder.block{i}.attn")

        # Conv module. Full residual.
        x_conv_in = block.norm_conv(x)
        conv_out = block.conv(x_conv_in, cache=None)
        x = x + conv_out
        if i in full_blocks:
            dump(f"enc.block.{i}.conv", x, f"encoder.block{i}.conv")

        # Macaron FF2.
        ff2_in = block.norm_feed_forward2(x)
        ff2_out = block.feed_forward2(ff2_in)
        x = x + 0.5 * ff2_out
        if i in full_blocks:
            dump(f"enc.block.{i}.ff2", x, f"encoder.block{i}.ff2")

        # Final per-block layer norm.
        x = block.norm_out(x)
        if i in block_set:
            dump(f"enc.block.{i}.out", x, f"encoder.block{i}.out")

    # ----- 4. final encoder output ----------------------------------
    dump("enc.final", x, "encoder.final")

    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    """Dump the first-step decoder intermediates from parakeet-mlx.

    The C++ decoder (src/arch/parakeet/decoder.cpp) dumps a small set
    of host-side fp32 buffers under TRANSCRIBE_DUMP_DIR, gated to the
    first iteration of the greedy decode loop. This subcommand
    produces the matching reference dumps from parakeet-mlx so
    `scripts/compare_tensors.py` can diff the two side by side.

    First-step intermediates dumped (matching decoder.cpp's gating):

      dec.enc_out          - encoder output [T_enc, d_model] (the
                              shared input to every decode step)
      dec.embed.0          - predictor input at step 0; the start
                              state branch of PredictNetwork uses
                              zeros, so this is a vector of pred_h
                              zeros and serves as a sanity check
                              that both sides agree on what the
                              start state is.
      dec.lstm.<l>.h.0     - per-layer hidden state after one LSTM
      dec.lstm.<l>.c.0       step (input = zeros, prev state = zeros)
      dec.joint.0          - joint network output for (encoder
                              frame 0, predictor start state)

    The decoder is run in fp32 throughout (parakeet-mlx defaults to
    bfloat16, which would introduce a few decimal digits of jitter
    relative to the C++ fp32 path).

    Strict-comparison protocol (when chasing a real divergence in
    the joint or LSTM forward):

      1. Run the C++ side with TRANSCRIBE_FORCE_CPU=1 so the encoder
         lands on the strict fp32 reference path (~3.8e-6 vs MLX),
         not Metal's simdgroup-f16 matmul (~3.5e-3 vs MLX).
      2. Pass `--mel-from-cpp <prefix>` to this subcommand so
         parakeet-mlx consumes the C++ mel instead of its own
         periodic-Hann variant (which differs by ~6e-3 from NeMo's
         symmetric-Hann reference at the spectrum peak).

    With both knobs flipped, dec.enc_out and dec.joint.0 should
    compare to within fp32 round-off (max ~1e-5). Without them, the
    comparison is dominated by mel + backend drift and only the
    LSTM intermediates (which take zero-input zero-state and are
    therefore independent of the encoder output) are expected to
    match strictly.
    """
    import mlx.core as mx
    from parakeet_mlx import audio as mlx_audio
    from parakeet_mlx.utils import from_pretrained

    model_path = Path(args.model).expanduser().resolve()
    audio_path = (
        Path(args.audio).expanduser().resolve() if args.audio else None
    )
    out_dir = Path(args.out).expanduser()

    if not model_path.exists():
        print(f"error: model path not found: {model_path}", file=sys.stderr)
        return 1

    print(f"Loading parakeet-mlx model from {model_path} (fp32)...")
    model = from_pretrained(str(model_path), dtype=mx.float32)

    if args.mel_from_cpp is not None:
        # Reuse the same C++-mel ingest pattern as cmd_encoder so the
        # bring-up loop can isolate decoder accuracy from any
        # frontend variance between parakeet-mlx and NeMo.
        prefix = Path(args.mel_from_cpp).expanduser().resolve()
        f32_path = Path(str(prefix) + ".f32")
        json_path = Path(str(prefix) + ".json")
        if not f32_path.exists() or not json_path.exists():
            print(f"error: mel dump pair not found at {prefix}.{{f32,json}}",
                  file=sys.stderr)
            return 1
        meta = json.loads(json_path.read_text())
        cpp_shape = tuple(meta["shape"])
        if len(cpp_shape) != 2:
            print(f"error: expected 2-D mel, got shape {cpp_shape}",
                  file=sys.stderr)
            return 1
        n_mels, T_mel = cpp_shape
        cpp_mel = np.fromfile(f32_path, dtype=np.float32).reshape(cpp_shape)
        mel_np = cpp_mel.T[None, :, :].astype(np.float32, copy=True)
        mel = mx.array(mel_np)
        print(f"mel (from C++ dump): cpp_shape={cpp_shape} -> mlx_shape={list(mel.shape)}")
    else:
        if audio_path is None or not audio_path.exists():
            print(f"error: audio file required when --mel-from-cpp is not given",
                  file=sys.stderr)
            return 1
        print(f"Loading audio {audio_path}...")
        audio_data = mlx_audio.load_audio(
            audio_path, model.preprocessor_config.sample_rate, mx.float32
        )
        mel = mlx_audio.get_logmel(audio_data, model.preprocessor_config)
        # mel: [B=1, T_mel, n_mels]

    # Encoder forward.
    print("Running encoder forward...")
    features, lengths = model.encoder(mel)
    mx.eval(features)
    # features: [B=1, T_enc, d_model]

    source = {
        "kind": "parakeet-mlx",
        "model": model_path.name,
    }
    if audio_path is not None and audio_path.exists():
        source["audio"] = audio_path.name
    if args.mel_from_cpp is not None:
        source["mel_from_cpp"] = Path(args.mel_from_cpp).name

    def to_np(t: "mx.array") -> np.ndarray:
        mx.eval(t)
        a = np.array(t, copy=False).astype(np.float32)
        while a.ndim > 1 and a.shape[0] == 1:
            a = a[0]
        return np.ascontiguousarray(a)

    def dump(name: str, arr: np.ndarray, stage: str) -> None:
        print(f"  {name}: shape={arr.shape} "
              f"min={arr.min():.4e} max={arr.max():.4e} mean={arr.mean():.6e}")
        write_dump(out_dir, name, arr, source=source, stage=stage)

    # ----- Encoder output as the decoder sees it ---------------------
    enc_np = to_np(features)  # [T_enc, d_model] after squeezing batch
    dump("dec.enc_out", enc_np, "decoder.enc_out")

    # ----- First-step predictor pass ---------------------------------
    #
    # PredictNetwork.__call__ with `y=None` builds an embed input of
    # zeros (start of sequence). We mirror that and dump the embed
    # vector explicitly so the C++ side's `dec.embed.0` (also a
    # zero vector) has a comparable counterpart.
    pred_h = model.decoder.pred_hidden
    embed_zeros = np.zeros((pred_h,), dtype=np.float32)
    dump("dec.embed.0", embed_zeros, "decoder.embed")

    # Run the predictor on the start state. PredictNetwork returns
    # (decoder_out, (h, c)) where h, c are stacked across layers
    # via mx.stack(..., axis=0), so they have shape [n_layers, ...].
    decoder_out, (h_all, c_all) = model.decoder(None, None)
    mx.eval(decoder_out, h_all, c_all)

    # Per-layer dumps. h_all / c_all shapes are [n_layers, B=1, pred_h]
    # — squeeze the batch.
    h_np_full = np.array(h_all, copy=False).astype(np.float32)
    c_np_full = np.array(c_all, copy=False).astype(np.float32)
    if h_np_full.ndim == 3 and h_np_full.shape[1] == 1:
        h_np_full = h_np_full[:, 0, :]
        c_np_full = c_np_full[:, 0, :]
    n_layers = h_np_full.shape[0]
    for layer in range(n_layers):
        dump(f"dec.lstm.{layer}.h.0",
             np.ascontiguousarray(h_np_full[layer]),
             "decoder.lstm")
        dump(f"dec.lstm.{layer}.c.0",
             np.ascontiguousarray(c_np_full[layer]),
             "decoder.lstm")

    # ----- First-step joint pass -------------------------------------
    #
    # parakeet-mlx's JointNetwork takes (enc, pred) with broadcast
    # dims; for one frame and one decoder step the result has shape
    # [B=1, T=1, U=1, n_classes]. Pass features[:, 0:1] (the first
    # encoder frame) and the decoder_out we just computed.
    joint_out = model.joint(features[:, 0:1], decoder_out)
    mx.eval(joint_out)
    joint_np = to_np(joint_out)  # squeezed to [n_classes] after dropping leading 1s
    dump("dec.joint.0", joint_np, "decoder.joint")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Dump per-stage reference tensors for the transcribe.cpp "
            "numerical accuracy harness."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    mp = sub.add_parser(
        "mel-onnx",
        help=(
            "Dump log-mel from a NeMo .onnx preprocessor on a wav "
            "(byte-level reference for the Parakeet frontend golden test)"
        ),
    )
    mp.add_argument("--onnx", required=True, help="Path to nemo128.onnx (or another NeMo preprocessor ONNX)")
    mp.add_argument("--audio", required=True, help="16 kHz mono wav file")
    mp.add_argument("--out", required=True, help="Output directory")
    mp.add_argument("--name", default="jfk-mel", help="Base name for output files (default: jfk-mel)")
    mp.set_defaults(func=cmd_mel_onnx)

    ep = sub.add_parser(
        "encoder",
        help=(
            "Dump per-stage encoder activations from parakeet-mlx "
            "(Apple Silicon only)"
        ),
    )
    ep.add_argument("--model", required=True,
                    help="Path to a parakeet-tdt-*-mlx model directory")
    ep.add_argument("--audio",
                    help="16 kHz mono wav file (required unless --mel-from-cpp)")
    ep.add_argument("--mel-from-cpp",
                    help="Optional path prefix (without extension) to a "
                         "C++ mel dump pair (<prefix>.f32 + <prefix>.json). "
                         "When set, parakeet-mlx's own mel is bypassed and "
                         "the C++ mel is fed into the encoder instead — "
                         "isolates encoder accuracy from frontend variance.")
    ep.add_argument("--out", required=True, help="Output directory")
    ep.add_argument("--blocks", type=int, nargs="*", default=[0, 12, 23],
                    help="Block indices to dump `enc.block.N.out` for "
                         "(default: 0 12 23). Block 0 always dumps every "
                         "sub-step (ff1/attn/conv/ff2/out).")
    ep.set_defaults(func=cmd_encoder)

    dp = sub.add_parser(
        "decode",
        help=(
            "Dump first-step predictor + joint intermediates from "
            "parakeet-mlx (Apple Silicon only). Use --mel-from-cpp "
            "if you need the encoder run to consume a C++ mel."
        ),
    )
    dp.add_argument("--model", required=True,
                    help="Path to a parakeet-tdt-*-mlx model directory")
    dp.add_argument("--audio",
                    help="16 kHz mono wav file (required unless --mel-from-cpp)")
    dp.add_argument("--mel-from-cpp",
                    help="Optional path prefix (without extension) to a "
                         "C++ mel dump pair (<prefix>.f32 + <prefix>.json). "
                         "When set, parakeet-mlx's own mel is bypassed and "
                         "the C++ mel is fed into the encoder instead — "
                         "needed for strict joint / encoder comparisons.")
    dp.add_argument("--out", required=True, help="Output directory")
    dp.set_defaults(func=cmd_decode)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
