#!/usr/bin/env python3
"""
dump_reference_ecapa_tdnn_speechbrain.py - generate SpeechBrain VoxLingua107
ECAPA-TDNN reference tensors (SpeechBrain is the only implementation).

    uv run --project scripts/envs/ecapa_tdnn \
      scripts/dump_reference_ecapa_tdnn_speechbrain.py encoder \
      --audio samples/fleurs-en.wav \
      --out build/validate/ecapa_tdnn/lang-id-voxlingua107-ecapa/fleurs-en/ref

The single `encoder` stage hooks one `classify_batch` call and writes
`<name>.f32` / `<name>.json` pairs (scripts/lib/ref_dump.py) for the dump
points listed in cmd_encoder, plus prediction.json (top-1 label). Every hook must
fire exactly once. Activations are written time-major `[T, C]` (PyTorch's
`[1, C, T]` transposed) to match the C++ dumper (src/arch/ecapa_tdnn/graph.h).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib.hf_source import resolve_model_dir  # noqa: E402
from lib.ref_dump import write_tensor  # noqa: E402

DEFAULT_REPO = "speechbrain/lang-id-voxlingua107-ecapa"
DEFAULT_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"

SAMPLE_RATE = 16000
HOP = 160
N_MELS = 60
N_LABELS = 107


def configure_torch(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already set earlier in this process
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_reference(args: argparse.Namespace):
    """Return `(EncoderClassifier, checkpoint_dir)`, in eval mode.

    `pretrained_path` must be overridden to the local snapshot, otherwise
    SpeechBrain re-fetches the .ckpt files from the HF `main` branch and the
    pinned revision is ignored. `savedir` must differ from the snapshot
    (SpeechBrain symlinks the loadables into it).
    """
    from speechbrain.inference.classifiers import EncoderClassifier

    src = resolve_model_dir(args.model, args.revision)
    savedir = REPO_ROOT / "build" / "speechbrain" / src.parent.name
    savedir.mkdir(parents=True, exist_ok=True)
    clf = EncoderClassifier.from_hparams(
        source=str(src), savedir=str(savedir),
        overrides={"pretrained_path": str(src)}, run_opts={"device": args.device})
    clf.mods.eval()
    return clf, src


def split_label(label: str) -> tuple[str, str]:
    """`'zh: Chinese'` -> `('zh', 'Chinese')`."""
    code, _, name = label.partition(": ")
    return code.strip(), name.strip()


def to_np(t: torch.Tensor, shape: tuple[int, ...], *, ct: bool = False) -> np.ndarray:
    """Hooked tensor -> contiguous f32 numpy of exactly `shape`.

    1-D shapes flatten; otherwise the batch dim is dropped and `ct=True`
    transposes `[C, T]` to time-major `[T, C]`. The shape is asserted so a
    tensor in the wrong layout cannot pass as its transpose.
    """
    a = t.detach().to(dtype=torch.float32, device="cpu")
    if len(shape) == 1:
        a = a.reshape(-1)
    else:
        a = a[0].transpose(0, 1) if ct else a[0]
    if tuple(a.shape) != shape:
        raise SystemExit(f"error: hooked tensor is {tuple(a.shape)}, expected {shape}")
    return np.ascontiguousarray(a.numpy(), dtype=np.float32)


def cmd_encoder(args: argparse.Namespace) -> int:
    import soundfile as sf
    import speechbrain

    configure_torch(args)
    clf, src_dir = load_reference(args)

    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pcm, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if int(sr) != SAMPLE_RATE:
        raise SystemExit(f"error: {audio_path} is {sr} Hz; resample to {SAMPLE_RATE} Hz")
    pcm = np.ascontiguousarray(pcm, dtype=np.float32)
    n_samples = int(pcm.size)
    expected_T = n_samples // HOP + 1

    device = torch.device(args.device)
    wav = torch.from_numpy(pcm).to(device).unsqueeze(0)  # [1, N]
    # wav_lens is RELATIVE to the padded batch max (1.0 for one unpadded
    # clip); a sample count would mask the wrong span, silently.
    wav_lens = torch.ones(1, device=device)

    em, cl = clf.mods.embedding_model, clf.mods.classifier
    # (dump name, module, stage, channels, layout). Layouts: "tc" is already
    # [1, T, C]; "ct" is [1, C, T] and gets transposed; "vec" is flattened.
    points = [
        ("fe.mel", clf.mods.mean_var_norm, "frontend", N_MELS, "tc"),
        ("enc.blk.0.out", em.blocks[0], "encoder", 1024, "ct"),
        ("enc.blk.1.tdnn1.out", em.blocks[1].tdnn1, "encoder", 1024, "ct"),
        ("enc.blk.1.res2.out", em.blocks[1].res2net_block, "encoder", 1024, "ct"),
        ("enc.blk.1.se.out", em.blocks[1].se_block, "encoder", 1024, "ct"),  # pre-residual
        ("enc.blk.1.out", em.blocks[1], "encoder", 1024, "ct"),
        ("enc.blk.2.out", em.blocks[2], "encoder", 1024, "ct"),
        ("enc.blk.3.out", em.blocks[3], "encoder", 1024, "ct"),
        ("enc.mfa.out", em.mfa, "encoder", 3072, "ct"),
        ("enc.asp.attn_logits", em.asp.conv, "encoder", 3072, "ct"),  # pre-softmax
        ("enc.asp.out", em.asp, "encoder", 6144, "vec"),
        ("enc.emb", em.fc, "encoder", 256, "vec"),
        ("cls.hidden", cl.DNN.block_0.act, "classifier", 512, "vec"),  # pre-BN1
        ("cls.logits_raw", cl.out, "classifier", N_LABELS, "vec"),
    ]

    captured: dict[str, list] = {name: [] for name, *_ in points}

    def grab(name):
        def hook(_mod, _inp, out):
            captured[name].append(out[0] if isinstance(out, tuple) else out)
        return hook

    handles = [m.register_forward_hook(grab(name)) for name, m, *_ in points]
    try:
        with torch.inference_mode():
            out_prob, score, index, text_lab = clf.classify_batch(wav, wav_lens)
    finally:
        for h in handles:
            h.remove()

    # Zero firings = module not on the path; two = the model ran twice.
    bad = {n: len(v) for n, v in captured.items() if len(v) != 1}
    if bad:
        raise SystemExit(f"error: unexpected hook call counts: {bad}")
    out = {n: v[0] for n, v in captured.items()}
    if not torch.equal(torch.log_softmax(out["cls.logits_raw"].reshape(-1), -1),
                       out_prob.reshape(-1)):
        raise SystemExit("error: log_softmax(cls.logits_raw) != classify_batch out_prob")

    T = int(out["fe.mel"].shape[1])
    if T != expected_T:
        raise SystemExit(f"error: fe.mel has T={T}, but the C++ front end assumes "
                         f"floor({n_samples}/{HOP}) + 1 = {expected_T}")

    source = {
        "kind": "ecapa-tdnn-speechbrain",
        "framework": "speechbrain",
        "framework_version": speechbrain.__version__,
        "torch_version": torch.__version__,
        "model": args.model,
        "revision": args.revision,
        "checkpoint_dir": str(src_dir),
        "device": args.device,
        "torch_threads": args.torch_threads,
        "model_dtype": "f32",
        "audio": audio_path.name,
        "n_samples": n_samples,
        "sample_rate": SAMPLE_RATE,
        "n_frames": T,
        "n_mels": N_MELS,
    }
    for name, _, stage, c, layout in points:
        a = to_np(out[name], (c,) if layout == "vec" else (T, c), ct=layout == "ct")
        print(f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} "
              f"mean={a.mean():.6e}")
        write_tensor(name, a, stage=stage, source=source, out_dir=out_dir)

    label_index = int(index.item())
    label = str(clf.hparams.label_encoder.ind2lab[label_index])
    if str(text_lab[0]) != label:
        raise SystemExit(f"error: decoded label {text_lab[0]!r} != ind2lab[{label_index}] {label!r}")
    code, name = split_label(label)
    prediction = {"label_index": label_index, "code": code}
    (out_dir / "prediction.json").write_text(json.dumps(prediction, indent=2) + "\n")
    print(f"prediction: {code} ({name}) prob={math.exp(float(score.item())):.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="SpeechBrain VoxLingua107 ECAPA-TDNN reference dumper.")
    sub = p.add_subparsers(dest="cmd", required=True)
    ep = sub.add_parser("encoder", help="Run the full forward pass; dump front "
                        "end, encoder, classifier and prediction.json")
    ep.add_argument("--model", default=DEFAULT_REPO,
                    help=f"HF repo id or local checkpoint dir (default: {DEFAULT_REPO})")
    ep.add_argument("--revision", default=DEFAULT_REVISION,
                    help="HF revision (commit SHA); ignored for a local --model")
    ep.add_argument("--audio", required=True, help="16 kHz mono WAV path")
    ep.add_argument("--out", required=True, help="Output directory for dumps")
    ep.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    ep.add_argument("--torch-threads", type=int, default=1,
                    help="torch.set_num_threads (0 = unchanged)")
    ep.set_defaults(func=cmd_encoder)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
