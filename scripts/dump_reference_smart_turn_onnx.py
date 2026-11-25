#!/usr/bin/env python3
"""
dump_reference_smart_turn_onnx.py - Smart Turn v3.2 reference tensors and the
end-of-turn probability, from upstream's fp32 ONNX (onnxruntime, canonical)
and a rebuilt PyTorch twin (instrumented cross-check).

    uv run --project scripts/envs/smart_turn \
      scripts/dump_reference_smart_turn_onnx.py encoder \
      --audio samples/jfk.wav \
      --out build/validate/smart_turn/smart-turn-v3.2/jfk/ref

One clip, one forward pass, as smart-turn's inference.py runs it: keep the
last 8 s (or left-pad with zeros to 8 s), WhisperFeatureExtractor(chunk_length=8,
numpy front end, do_normalize=True), then the ONNX session with inference.py's
options. Writes (batch dimension dropped):

    fe.window         [128000]    8 s window after zero-mean / unit-variance
    enc.mel.in        [800, 80]   log-mel input_features, time-major (the
                                  whisper family's dump layout; ONNX input is
                                  its transpose [80, 800])
    enc.conv1.out     [800, 384]  GELU(conv1), time-major
    enc.conv2.out     [400, 384]  GELU(conv2), time-major
    enc.pos_emb       [400, 384]  positional table (a weight)
    enc.embed.out     [400, 384]  conv2 + pos_emb
    enc.block.{0..3}.out [400, 384]
    enc.final         [400, 384]  encoder final LayerNorm
    head.pool_scores  [400]       Linear-Tanh-Linear score per position
    head.pool_weights [400]       softmax over positions
    head.pooled       [384]       sum_t weights[t] * enc.final[t]
    head.logit        [1]         classifier output before the sigmoid
    head.prob         [1]         sigmoid: P(turn complete), canonical session

fe.window and enc.mel.in come from the front end (transformers), enc.pos_emb
is the ONNX initializer, everything else from onnxruntime. head.prob is the
unmodified session
(the canonical number); the intermediates come from a second session whose
graph also exposes them as outputs, and its probability must match the
canonical one to 1e-6 (fusions differ). The same tensors from the PyTorch twin
(scripts/lib/smart_turn_onnx.py: train.py's SmartTurnV3Model with the ONNX
weights mapped in) go to <out>/torch/, and the per-tensor twin-vs-ONNX
differences to <out>/bridge.json. turn.json holds p, the upstream decision
(p > 0.5) and, for later comparison, the p of upstream's int8 ONNX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib import smart_turn_onnx as st  # noqa: E402
from lib.ref_dump import write_tensor  # noqa: E402

# Twin vs ONNX bound on head.prob; observed ~5e-7 (fp32 reduction order).
BRIDGE_P_MAX = 1e-5
# Exposing intermediates as graph outputs stops some onnxruntime fusions, so
# the instrumented session's p may differ from the canonical one in the last
# bits (observed ~6e-8).
INSTRUMENTED_P_MAX = 1e-6


def session(path: Path, threads: int, onnx_model=None):
    """inference.py's build_session (plus intra-op threads, which Pipecat's
    analyzer sets to cpu_count), CPU provider."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    if threads > 0:
        so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    src = onnx_model.SerializeToString() if onnx_model is not None else str(path)
    return ort.InferenceSession(src, sess_options=so, providers=["CPUExecutionProvider"])


def onnx_taps(model) -> dict[str, str]:
    """Map dump names to ONNX value names by graph structure."""
    g = model.graph
    by_op: dict[str, list] = {}
    for n in g.node:
        by_op.setdefault(n.op_type, []).append(n)
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    def only(nodes, what):
        if len(nodes) != 1:
            raise SystemExit(f"error: onnx taps: {len(nodes)} candidates for {what}")
        return nodes[0]

    def gelu_after(value: str) -> str:
        # GELU(x) = x * 0.5 * (1 + erf(x / sqrt 2)): the Mul that takes x and
        # the 0.5 * (...) factor.
        muls = [c for c in consumers[value] if c.op_type == "Mul"]
        return only(muls, f"GELU({value})").output[0]

    convs = by_op["Conv"]
    if len(convs) != 2:
        raise SystemExit("error: onnx taps: expected 2 Conv nodes")
    taps = {
        "enc.conv1.out": gelu_after(convs[0].output[0]),
        "enc.conv2.out": gelu_after(convs[1].output[0]),
    }
    pos = only([n for n in by_op["Add"] if "inner.encoder.embed_positions.weight" in n.input], "pos add")
    taps["enc.embed.out"] = pos.output[0]
    ln = {n.input[1]: n for n in by_op["LayerNormalization"]}
    for i in range(st.N_LAYERS):
        nxt = (f"inner.encoder.layers.{i + 1}.self_attn_layer_norm.weight" if i + 1 < st.N_LAYERS
               else "inner.encoder.layer_norm.weight")
        taps[f"enc.block.{i}.out"] = ln[nxt].input[0]
    taps["enc.final"] = ln["inner.encoder.layer_norm.weight"].output[0]
    score = only([n for n in by_op["Add"] if "inner.pool_attention.2.bias" in n.input], "pool score")
    taps["head.pool_scores"] = score.output[0]
    taps["head.pool_weights"] = only([c for c in consumers[score.output[0]] if c.op_type == "Softmax"],
                                     "pool softmax").output[0]
    taps["head.pooled"] = only(by_op["ReduceSum"], "pool sum").output[0]
    sig = only(by_op["Sigmoid"], "sigmoid")
    taps["head.logit"] = sig.input[0]
    taps["head.prob"] = sig.output[0]
    if sig.output[0] != g.output[0].name:
        raise SystemExit("error: onnx taps: sigmoid is not the graph output")
    return taps


def with_outputs(model, names: list[str]):
    import copy

    import onnx

    m = copy.deepcopy(model)
    have = {o.name for o in m.graph.output}
    for n in names:
        if n not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, None))
    return m


def tm(a: np.ndarray) -> np.ndarray:
    """[1, C, T] -> [T, C] float32 contiguous."""
    return np.ascontiguousarray(a[0].T, dtype=np.float32)


def squeeze(name: str, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if name in ("enc.conv1.out", "enc.conv2.out"):
        return tm(a)
    if name in ("head.pool_scores", "head.pool_weights"):
        return np.ascontiguousarray(a.reshape(-1))   # [1, 400, 1] -> [400]
    if name in ("head.logit", "head.prob"):
        return np.ascontiguousarray(a.reshape(1))
    return np.ascontiguousarray(a[0])                  # drop batch


def twin_forward(model, x):
    """Step through SmartTurnV3Model.forward and WhisperEncoder.forward
    (transformers 4.48.2) capturing the dump points; returns (taps, p)."""
    import torch
    from torch.nn import functional as F

    enc = model.encoder
    taps: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        h = F.gelu(enc.conv1(x))
        taps["enc.conv1.out"] = tm(h.numpy())
        h = F.gelu(enc.conv2(h))
        taps["enc.conv2.out"] = tm(h.numpy())
        h = h.permute(0, 2, 1)
        h = h + enc.embed_positions.weight
        taps["enc.embed.out"] = h[0].numpy().copy()
        for i, layer in enumerate(enc.layers):
            h = layer(h, None, layer_head_mask=None)[0]
            taps[f"enc.block.{i}.out"] = h[0].numpy().copy()
        h = enc.layer_norm(h)
        taps["enc.final"] = h[0].numpy().copy()
        s = model.pool_attention(h)
        taps["head.pool_scores"] = s.reshape(-1).numpy().copy()
        w = torch.softmax(s, dim=1)
        taps["head.pool_weights"] = w.reshape(-1).numpy().copy()
        pooled = torch.sum(h * w, dim=1)
        taps["head.pooled"] = pooled[0].numpy().copy()
        logit = model.classifier(pooled)
        taps["head.logit"] = logit.reshape(1).numpy().copy()
        p = torch.sigmoid(logit)
        taps["head.prob"] = p.reshape(1).numpy().copy()
        p_e2e = model(x)["logits"]
    if not torch.equal(p, p_e2e):
        raise SystemExit(f"error: twin stepwise p {p.item()!r} != forward() {p_e2e.item()!r}")
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in taps.items()}, float(p.item())


def cmd_encoder(args: argparse.Namespace) -> int:
    import onnx
    import onnxruntime as ort
    import soundfile as sf
    import torch
    import transformers

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    onnx_path = st.resolve_onnx(args.model, args.revision)
    onnx_sha = st.sha256_file(onnx_path)
    audio_path = Path(args.audio).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pcm, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1 or int(sr) != st.SAMPLE_RATE:
        raise SystemExit(f"error: {audio_path} is not {st.SAMPLE_RATE} Hz mono")
    n = len(pcm)

    fe = st.feature_extractor()
    window = st.truncate_audio_to_last_n_seconds(pcm, n_seconds=st.MAX_AUDIO_SECONDS)
    x = st.features(fe, window)                       # [1, 80, 800]
    if x.shape != (1, st.N_MELS, st.N_FRAMES):
        raise SystemExit(f"error: features shape {x.shape}")
    wave = st.normalized_window(fe, window)

    # Canonical: the unmodified graph, inference.py's session.
    model = onnx.load(str(onnx_path))
    p = float(session(onnx_path, args.torch_threads).run(None, {"input_features": x})[0][0].item())

    # Instrumented: same graph, intermediates exposed as outputs.
    taps = onnx_taps(model)
    names = list(taps)
    inst = session(onnx_path, args.torch_threads, with_outputs(model, [taps[k] for k in names]))
    outs = dict(zip([o.name for o in inst.get_outputs()], inst.run(None, {"input_features": x})))
    ref = {k: squeeze(k, outs[taps[k]]) for k in names}
    p_inst = float(ref["head.prob"][0])
    if abs(p_inst - p) > INSTRUMENTED_P_MAX:
        raise SystemExit(f"error: instrumented session p {p_inst!r} vs canonical {p!r}")
    from onnx import numpy_helper
    pos = next(numpy_helper.to_array(t) for t in model.graph.initializer
               if t.name == "inner.encoder.embed_positions.weight")

    # PyTorch twin.
    sd, _ = st.map_onnx_weights(model)
    st.check_geometry(sd)
    twin = st.build_twin(sd)
    tw, p_twin = twin_forward(twin, torch.from_numpy(x))
    tw["enc.pos_emb"] = np.ascontiguousarray(twin.encoder.embed_positions.weight.detach().numpy())

    p_int8 = None
    if not args.no_int8:
        int8_path = st.resolve_onnx(None if Path(args.model).expanduser().is_file() else args.model,
                                    args.revision, st.ONNX_INT8)
        p_int8 = float(session(int8_path, args.torch_threads).run(None, {"input_features": x})[0][0].item())

    source = {
        "kind": "smart-turn-onnx",
        "framework": "onnxruntime",
        "framework_version": ort.__version__,
        "onnx_version": onnx.__version__,
        "transformers_version": transformers.__version__,
        "model": f"{st.HF_REPO}@{args.revision}/{onnx_path.name}",
        "model_sha256": onnx_sha,
        "front_end": "WhisperFeatureExtractor(chunk_length=8) numpy path, do_normalize=True",
        "device": "cpu",
        "threads": args.torch_threads,
        "model_dtype": "f32",
        "audio": audio_path.name,
        "n_samples": int(n),
    }
    stages = {"fe.window": "frontend", "enc.mel.in": "frontend"}
    write_tensor("fe.window", wave, stage="frontend", source=source, out_dir=out_dir)
    mel = np.ascontiguousarray(x[0].T)                # [800, 80]
    write_tensor("enc.mel.in", mel, stage="frontend", source=source, out_dir=out_dir)
    write_tensor("enc.pos_emb", np.ascontiguousarray(pos, dtype=np.float32), stage="encoder",
                 source=source, out_dir=out_dir)
    for k in names:
        stage = "encoder" if k.startswith("enc.") else "head"
        write_tensor(k, ref[k], stage=stage, source=source, out_dir=out_dir)
        stages[k] = stage
    # head.prob is the canonical session's number; head.logit is the instrumented one's.
    write_tensor("head.prob", np.asarray([p], dtype=np.float32), stage="head", source=source, out_dir=out_dir)

    tsrc = dict(source, kind="smart-turn-torch-twin", framework="torch", framework_version=torch.__version__,
                model=f"SmartTurnV3Model (train.py) with weights mapped from {onnx_path.name}")
    tdir = out_dir / "torch"
    bridge = {}
    for k, v in tw.items():
        write_tensor(k, v, stage=stages.get(k, "encoder"), source=tsrc, out_dir=tdir)
        r = pos if k == "enc.pos_emb" else ref[k]
        if r.shape != v.shape:
            raise SystemExit(f"error: twin {k} shape {v.shape} != onnx {r.shape}")
        d = np.abs(v.astype(np.float64) - r.astype(np.float64))
        bridge[k] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                     "ref_max_abs": float(np.abs(r).max())}
    if abs(p_twin - p) > BRIDGE_P_MAX:
        raise SystemExit(f"error: bridge: twin p {p_twin!r} vs onnx p {p!r} differ by more than {BRIDGE_P_MAX}")
    (out_dir / "bridge.json").write_text(json.dumps(
        {"p_onnx": p, "p_twin": p_twin, "p_abs_diff": abs(p_twin - p), "tensors": bridge}, indent=2) + "\n")

    crop = max(0, n - st.MAX_AUDIO_SAMPLES)
    pad = max(0, st.MAX_AUDIO_SAMPLES - n)
    turn = {
        "p": p,
        "ended": p > st.THRESHOLD,
        "threshold": st.THRESHOLD,
        "rule": "inference.py: prediction = 1 if probability > 0.5",
        "n_samples": int(n),
        "window": "crop" if crop else ("pad" if pad else "exact"),
        "crop_samples": crop,
        "pad_samples": pad,
        "p_instrumented": p_inst,
        "p_twin": p_twin,
        "p_int8": p_int8,
        "int8_model": None if p_int8 is None else f"{st.HF_REPO}@{args.revision}/{st.ONNX_INT8}",
        "source": source,
    }
    (out_dir / "turn.json").write_text(json.dumps(turn, indent=2) + "\n")
    worst = max(bridge.items(), key=lambda kv: kv[1]["max_abs"])
    print(f"{audio_path.name}: n={n} {turn['window']} p={p:.6f} ended={turn['ended']} "
          f"twin|dp|={abs(p_twin - p):.2e} worst {worst[0]} {worst[1]['max_abs']:.2e}"
          + ("" if p_int8 is None else f" int8 p={p_int8:.6f}"))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Smart Turn v3.2 reference dumper (onnxruntime + PyTorch twin).")
    sub = p.add_subparsers(dest="cmd", required=True)
    ep = sub.add_parser("encoder", help="Score one clip; dump stage tensors, turn.json and bridge.json")
    ep.add_argument("--model", default=st.HF_REPO,
                    help=f"{st.HF_REPO} (pinned download) or a local fp32 .onnx path")
    ep.add_argument("--revision", default=st.HF_REVISION, help="HF revision (default: the pinned commit)")
    ep.add_argument("--audio", required=True, help="16 kHz mono WAV path")
    ep.add_argument("--out", required=True, help="Output directory for dumps")
    ep.add_argument("--torch-threads", type=int, default=1,
                    help="torch.set_num_threads and onnxruntime intra-op threads (0 = library default)")
    ep.add_argument("--no-int8", action="store_true", help="Skip upstream's int8 ONNX (turn.json p_int8)")
    ep.set_defaults(func=cmd_encoder)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
