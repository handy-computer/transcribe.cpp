#!/usr/bin/env python3
"""
dump_reference_sava_author.py - SAVA (Inworld) reference tensors, curves and
turn segments from upstream's own package (inworld-sava 1.0.0, PyTorch).

    uv run --project scripts/envs/sava \
      scripts/dump_reference_sava_author.py encoder \
      --audio samples/sava-speech.wav \
      --out build/validate/sava/sava/sava-speech/ref

The reference is `sava.run(model, audio)`: one causal forward over the clip
zero padded to whole 640-sample frames plus `lookahead_frames` (3) frames of
silence, which is exactly a Session pushed the whole clip and flushed.
Forward hooks capture the intermediates of that same call, so they are the
reference's own values. With n = ceil(samples / 640) frames, the forward runs
S = n + 3 steps; every per-step tensor is stacked over all S steps, time-major:

    fe.mel                [S, 512]   log-mel, 4 hops x 128 bins (hop-major)
    fe.log_energy         [S]        ln(max(rms(frame), 1e-5))
    mixer.out             [S, 512]   frame mixer after its RMSNorm
    trunk.{i}.mix         [S, 512]   block i's Mamba-2 / attention sublayer
                                     output (after out_proj / wo, before the
                                     residual add)
    trunk.{i}.ssm_y       [S, 1024]  Mamba blocks: SSM scan output y = C.h
                                     before the D skip, head-major (h * 64 + p)
    trunk.{i}.ssm_state   [16, 64, 128]  Mamba blocks: SSM state after step S-1
    trunk.{i}.out         [S, 512]   block i output (residual stream)
    trunk.norm            [S, 512]   final_norm(trunk output)
    head.vad.hidden       [S, 512]   VAD branch features (after its out_norm)
    head.vad.logit        [S]        on-time speech logit
    head.vad.logit_la     [S]        lookahead speech logit, as the model
                                     emits it: steps 0..2 are NO_SPEECH_LOGIT
                                     (-30); step t judges frame t - 3
    head.turn.speech_in   [S]        the turn head's speech input: sigmoid of
                                     the previous step's on-time logit, 0 at
                                     step 0
    head.turn.input       [S, 576]   [turn_norm(trunk.norm) | speech embed]
    head.turn.embed       [S, 64]    speech embedding (= head.turn.input[:, 512:])
    head.turn.hidden      [S, 512]   turn branch features (after its out_norm)
    head.turn.logits      [S, 4]     turn logits, horizons 0, 5, 10, 20 frames

The public products (sava.run's Curves, probabilities):

    sava.speech           [n]     released speech probability of frame t
                                  (lookahead output of step t + 3)
    sava.speech_on_time   [n]     on-time speech probability of frame t
    sava.turn             [n, 4]  turn probabilities of frame t (col 0 released)
    sava.flush_turns      [3, 4]  turn probabilities of the 3 silent flush steps

The streamed reference (a sava.Session pushed in chunks of 13, 1280, 777, ...
samples, then flushed), as emitted:

    stream.speech         [S]     speech output of step t (judges frame t - 3)
    stream.turn           [S, 4]  turn output of step t

stream.json records how each chunking compares with sava.run (bit-exact or
max abs difference); segments.json holds sava.segment(curves, Policy(), 3)
and the live TurnSegmenter run over the stream; turn.json is the clip's p,
decision and kept turns in samples, the TURN API's view of the same run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib.ref_dump import write_tensor  # noqa: E402

SAMPLE_RATE = 16000
FRAME = 640
HF_REPO = "inworld/sava"
HF_REVISION = "1beae7c1dd0df6b0f92030644fc905eb087dc6a0"
WEIGHTS_SHA256 = "aa610ee9b5d3d1be96f70581a8a3affe7cf20d06816b4ada1efdd8ab6274be34"
SAVA_VERSION = "1.0.0"
# Session chunkings compared against sava.run; the first is dumped.
CHUNKINGS = {"odd": (13, 1280, 777), "frame": (FRAME,), "whole": None}
# A whole-clip push builds an [S, 512 + S] attention score matrix per head.
WHOLE_PUSH_MAX_FRAMES = 2048


def resolve_weights(model: str, revision: str | None) -> Path:
    p = Path(model).expanduser()
    if p.is_file():
        return p.resolve()
    if p.is_dir() and (p / "sava.safetensors").is_file():
        return (p / "sava.safetensors").resolve()
    import huggingface_hub

    return Path(huggingface_hub.hf_hub_download(repo_id=model, filename="sava.safetensors",
                                                revision=revision or HF_REVISION))


class Taps:
    """Records module outputs during one call; restores everything on close."""

    def __init__(self) -> None:
        self.buf: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._handles = []
        self._patched: list[tuple[object, str, object]] = []

    def _put(self, name: str, t: torch.Tensor) -> None:
        self.buf[name].append(t.detach().float()[0].clone())

    def hook(self, module: torch.nn.Module, name: str) -> None:
        self._handles.append(module.register_forward_hook(lambda _m, _i, out: self._put(name, out)))

    def wrap(self, owner: object, attr: str, fn) -> None:
        orig = getattr(owner, attr)
        self._patched.append((owner, attr, orig))

        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            fn(args, out)
            return out

        setattr(owner, attr, wrapped)

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        for owner, attr, orig in reversed(self._patched):
            setattr(owner, attr, orig)

    def cat(self, name: str) -> np.ndarray:
        parts = self.buf[name]
        t = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return t.numpy().astype(np.float32)


def capture_run(sava, model, audio: np.ndarray):
    """sava.run(model, audio) with the intermediates of its forward captured."""
    from inworld.sava import modules, ssd

    taps = Taps()
    taps.hook(model.embedder, "fe.mel")
    taps.wrap(model.mixer, "rows", lambda _a, out: taps._put("fe.log_energy", out[..., -1]))
    taps.hook(model.mixer, "mixer.out")
    mamba_ids = []
    for i, block in enumerate(model.trunk):
        taps.hook(block, f"trunk.{i}.out")
        if isinstance(block, modules.MambaBlock):
            taps.hook(block.mixer, f"trunk.{i}.mix")
            mamba_ids.append(i)
        else:
            taps.wrap(block.attention, "forward_chunk",
                      lambda _a, out, i=i: taps._put(f"trunk.{i}.mix", out[0]))
    scans: list[tuple[torch.Tensor, torch.Tensor]] = []
    taps.wrap(ssd, "chunked_scan", lambda _a, out: scans.append((out[0].detach(), out[1].detach())))
    taps.hook(model.final_norm, "trunk.norm")
    taps.hook(model.vad_branch.out_norm, "head.vad.hidden")
    taps.hook(model.turn_branch.out_norm, "head.turn.hidden")

    def on_turn_input(args, out):
        taps._put("head.turn.speech_in", args[1])
        taps._put("head.turn.input", out)

    taps.wrap(model, "_turn_input", on_turn_input)
    outputs = []
    taps._handles.append(model.register_forward_hook(lambda _m, _i, out: outputs.append(out)))
    try:
        curves = sava.run(model, audio)
    finally:
        taps.close()

    if len(outputs) != 1 or len(scans) != len(mamba_ids):
        raise SystemExit(f"error: expected one forward and {len(mamba_ids)} scans, "
                         f"got {len(outputs)} / {len(scans)}")
    out = outputs[0]
    t = {name: taps.cat(name) for name in taps.buf}
    for i, (y, state) in zip(mamba_ids, scans):
        t[f"trunk.{i}.ssm_y"] = y[0].reshape(y.shape[1], -1).float().numpy()
        t[f"trunk.{i}.ssm_state"] = state[0].float().numpy()
    t["head.turn.embed"] = t["head.turn.input"][:, -model.cfg.vad_embed_size:]
    t["head.vad.logit"] = out.speech[0].float().numpy()
    t["head.vad.logit_la"] = out.lookahead[0].float().numpy()
    t["head.turn.logits"] = out.turn[0].float().numpy()
    return curves, t


def stream(sava, model, audio: np.ndarray, sizes):
    session = sava.Session(model)
    outs = []
    if sizes is None:
        outs.append(session.push(audio))
    else:
        off, k = 0, 0
        while off < audio.size:
            n = sizes[k % len(sizes)]
            outs.append(session.push(audio[off:off + n]))
            off += n
            k += 1
    outs.append(session.flush())
    return outs


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    return {"bit_exact": bool(np.array_equal(a, b)),
            "max_abs": float(np.abs(a.astype(np.float64) - b).max()) if a.size else 0.0}


def turn_dict(turn, n_samples: int) -> dict:
    return {"start": turn.start, "end": turn.end,
            "start_sample": turn.start * FRAME, "end_sample": min(turn.end * FRAME, n_samples),
            "speech_ms": turn.speech_ms, "reason": turn.reason, "kept": turn.kept}


# sava.segment's reasons under the TURN API's names (transcribe_turn_end).
REASONS = {"turn": "model", "silence": "silence", "end": "audio"}


def write_turn_json(out_dir: Path, seg: dict) -> None:
    """turn.json, what validate.py compares with the CLI's TURN output: the
    clip's p and decision, and the kept turns in samples (dropped turns are
    not reported by the API)."""
    turn = {
        "p": seg["run"]["p"],
        "ended": seg["run"]["ended"],
        "threshold": seg["policy"]["turn_threshold"],
        "segments": [{"start": t["start_sample"], "end": t["end_sample"], "reason": REASONS[t["reason"]]}
                     for t in seg["turns"] if t["kept"]],
    }
    (out_dir / "turn.json").write_text(json.dumps(turn, indent=2) + "\n")


def cmd_encoder(args: argparse.Namespace) -> int:
    import soundfile as sf
    from inworld import sava
    from inworld.sava import checkpoint, config, registry

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if sava.__version__ != SAVA_VERSION:
        raise SystemExit(f"error: inworld-sava {sava.__version__} installed, dumper pinned to {SAVA_VERSION}")
    weights = resolve_weights(args.model, args.revision)
    sha = registry.sha256_file(str(weights))
    if sha != WEIGHTS_SHA256:
        raise SystemExit(f"error: {weights} sha256 {sha} != pinned {WEIGHTS_SHA256}")
    model = checkpoint.load(str(weights))  # CPU, fp32 (bf16 weights upcast)
    lag = model.lookahead_frames

    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pcm, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1 or int(sr) != SAMPLE_RATE:
        raise SystemExit(f"error: {audio_path} is not {SAMPLE_RATE} Hz mono")
    audio = np.ascontiguousarray(pcm, dtype=np.float32)
    n = sava.streaming.num_frames(audio.size)
    steps = n + lag

    source = {
        "kind": "sava-author",
        "framework": "inworld-sava",
        "framework_version": sava.__version__,
        "torch_version": torch.__version__,
        "model": f"{HF_REPO}@{HF_REVISION} sava.safetensors (bf16, loaded as fp32)",
        "weights_sha256": sha,
        "entrypoint": "inworld.sava.run",
        "device": "cpu",
        "torch_threads": args.torch_threads,
        "model_dtype": "f32",
        "audio": audio_path.name,
        "n_samples": int(audio.size),
        "n_frames": n,
        "n_steps": steps,
    }

    curves, t = capture_run(sava, model, audio)
    for name, arr in t.items():
        if name != "trunk.3.ssm_state" and not name.endswith("ssm_state") and arr.shape[0] != steps:
            raise SystemExit(f"error: {name} has {arr.shape[0]} steps, expected {steps}")
    # The captured logits are sava.run's own: re-derive its curves from them.
    sig = lambda x: torch.sigmoid(torch.from_numpy(x)).numpy()  # noqa: E731
    if not (np.array_equal(sig(t["head.vad.logit_la"])[lag:lag + n], curves.speech)
            and np.array_equal(sig(t["head.turn.logits"])[:n], curves.turns)
            and np.array_equal(sig(t["head.vad.logit"])[:n], curves.speech_on_time)):
        raise SystemExit("error: captured logits do not reproduce sava.run's curves")

    stages = {"fe.": "frontend", "mixer.": "frontend", "trunk.": "trunk", "head.": "heads"}
    for name, arr in sorted(t.items()):
        stage = next(v for k, v in stages.items() if name.startswith(k))
        write_tensor(name, np.ascontiguousarray(arr, dtype=np.float32), stage=stage, source=source,
                     out_dir=out_dir)
    for name, arr in (("sava.speech", curves.speech), ("sava.speech_on_time", curves.speech_on_time),
                      ("sava.turn", curves.turns), ("sava.flush_turns", curves.flush_turns)):
        write_tensor(name, np.ascontiguousarray(arr, dtype=np.float32), stage="output", source=source,
                     out_dir=out_dir)

    # Streams versus run.
    run_speech = np.concatenate([np.full(lag, np.float32(torch.sigmoid(torch.tensor(-30.0)).item())),
                                 curves.speech])
    run_turn = np.concatenate([curves.turns, curves.flush_turns])
    stream_report = {"reference": "sava.run (speech prefixed with the lag floor, turns + flush_turns)"}
    dumped = None
    for label, sizes in CHUNKINGS.items():
        if sizes is None and steps > WHOLE_PUSH_MAX_FRAMES:
            stream_report[label] = {"skipped": f"more than {WHOLE_PUSH_MAX_FRAMES} frames"}
            continue
        outs = stream(sava, model, audio, sizes)
        sp = np.concatenate([o.speech for o in outs]).astype(np.float32)
        tu = np.concatenate([o.turns for o in outs]).astype(np.float32)
        if sp.size != steps:
            raise SystemExit(f"error: stream {label} stepped {sp.size} frames, expected {steps}")
        stream_report[label] = {
            "chunk_sizes": list(sizes) if sizes else [int(audio.size)],
            "speech": compare(sp, run_speech),
            "turn": compare(tu, run_turn),
            "turn_col0_real_frames": compare(tu[:n, 0], curves.turn),
            "flush_frames": [len(outs[-1]), int(outs[-1].first)],
        }
        if dumped is None:
            dumped = (label, outs, sp, tu)
    label, outs, sp, tu = dumped
    write_tensor("stream.speech", sp, stage="stream", source={**source, "chunking": label}, out_dir=out_dir)
    write_tensor("stream.turn", tu, stage="stream", source={**source, "chunking": label}, out_dir=out_dir)
    (out_dir / "stream.json").write_text(json.dumps({**stream_report, "source": source}, indent=2) + "\n")

    # Turn segments: offline (sava.segment on the curves) and live
    # (TurnSegmenter fed the dumped stream's pushes).
    policy = sava.Policy()
    turns = sava.segment(curves, policy, lag)
    live = sava.TurnSegmenter(policy, lag)
    live_turns = []
    for o in outs:
        live_turns += live.feed(o)
    live_turns += live.finish()
    payload = {
        "n_samples": int(audio.size),
        "n_frames": n,
        "frame_samples": FRAME,
        "policy": {k: getattr(policy, k) for k in policy.__dataclass_fields__},
        "lag": lag,
        "turns": [turn_dict(x, audio.size) for x in turns],
        "live_turns_equal": [turn_dict(x, audio.size) for x in live_turns] == [turn_dict(x, audio.size) for x in turns],
        "run": {"p": float(curves.turn[-1]), "ended": bool(curves.turn[-1] >= policy.turn_threshold)},
        "source": source,
    }
    (out_dir / "segments.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_turn_json(out_dir, payload)

    if args.expected_npz:
        # Upstream's golden fixture: logits of model(clip) (no lag padding).
        with np.load(args.expected_npz) as d:
            exp = {k: d[k] for k in d.files}
        m = len(exp["speech"])
        golden = {
            "file": str(args.expected_npz),
            "speech_logit_max_abs": float(np.abs(t["head.vad.logit"][:m] - exp["speech"]).max()),
            "lookahead_logit_max_abs": float(np.abs(t["head.vad.logit_la"][lag:m] - exp["lookahead"][lag:]).max()),
            "turn_logit_max_abs": float(np.abs(t["head.turn.logits"][:m] - exp["turn"]).max()),
            "upstream_atol": 2e-4,
        }
        (out_dir / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")
        print("golden:", json.dumps(golden))

    print(f"frames: {n}  steps: {steps}  turns: {len(turns)}  "
          + "  ".join(f"{k}: {v['speech']['bit_exact']}/{v['turn']['bit_exact']} "
                      f"({max(v['speech']['max_abs'], v['turn']['max_abs']):.2e})"
                      for k, v in stream_report.items() if isinstance(v, dict) and "speech" in v))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SAVA reference dumper (inworld-sava package).")
    sub = p.add_subparsers(dest="cmd", required=True)
    ep = sub.add_parser("encoder", help="Run the clip; dump per-step tensors, curves, streams and segments")
    ep.add_argument("--model", default=HF_REPO, help="HF repo id, a directory holding sava.safetensors, or the file")
    ep.add_argument("--revision", default=None, help=f"HF revision (default {HF_REVISION})")
    ep.add_argument("--audio", required=True, help="16 kHz mono WAV path")
    ep.add_argument("--out", required=True, help="Output directory for dumps")
    ep.add_argument("--torch-threads", type=int, default=1, help="torch.set_num_threads (0 = unchanged)")
    ep.add_argument("--expected-npz", default=None,
                    help="Upstream golden logits (tests/fixtures/speech_expected.npz) to compare against")
    ep.set_defaults(func=cmd_encoder)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
