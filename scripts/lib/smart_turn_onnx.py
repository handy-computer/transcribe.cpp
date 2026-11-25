"""Smart Turn v3.2 (pipecat-ai) source handling shared by the converter and the
reference scripts. Upstream publishes only ONNX, so this is the single copy of
the pinned files, upstream's preprocessing (inference.py / audio_utils.py),
the ONNX -> PyTorch weight map and train.py's SmartTurnV3Model. See
docs/porting/families/smart_turn.md.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

HF_REPO = "pipecat-ai/smart-turn-v3"
HF_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
ONNX_FP32 = "smart-turn-v3.2-gpu.onnx"   # fp32: the reference dtype
ONNX_INT8 = "smart-turn-v3.2-cpu.onnx"   # upstream's static int8 quant
SHA256 = {
    ONNX_FP32: "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be",
    ONNX_INT8: "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
}

SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 8
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE  # 128000
N_MELS = 80
N_FRAMES = 800          # mel frames fed to the encoder
N_POSITIONS = 400       # encoder positions (conv2 has stride 2)
THRESHOLD = 0.5         # inference.py: prediction = 1 if probability > 0.5
NORMALIZE_EPS = 1e-7    # zero_mean_unit_var_norm: / sqrt(var + 1e-7)

# whisper-tiny encoder (train.py: from_pretrained("openai/whisper-tiny"),
# max_source_positions = 400) and the head; asserted against the ONNX shapes.
D_MODEL = 384
N_LAYERS = 4
N_HEADS = 6
FFN_DIM = 1536
POOL_DIM = 256
CLS_DIMS = (256, 64)


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(filename: str = ONNX_FP32, revision: str = HF_REVISION) -> Path:
    """One file of HF_REPO, sha256-checked at the pinned revision; into
    $TRANSCRIBE_MODELS_DIR/smart-turn-v3@<revision>/ when that is set."""
    from huggingface_hub import hf_hub_download

    local_dir = None
    root = os.environ.get("TRANSCRIBE_MODELS_DIR")
    if root:
        local_dir = Path(root) / f"smart-turn-v3@{revision}"
        local_dir.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(HF_REPO, filename, revision=revision,
                                local_dir=str(local_dir) if local_dir else None))
    if revision == HF_REVISION and filename in SHA256:
        sha = sha256_file(path)
        if sha != SHA256[filename]:
            raise SystemExit(f"error: {path} sha256 {sha} != pinned {SHA256[filename]}")
    return path


def resolve_onnx(model: str | None, revision: str = HF_REVISION, filename: str = ONNX_FP32) -> Path:
    """--model is the HF repo id (pinned download) or a local .onnx path."""
    if model and Path(model).expanduser().is_file():
        return Path(model).expanduser().resolve()
    if model and model != HF_REPO:
        raise SystemExit(f"error: --model must be {HF_REPO} or a local .onnx file, got {model!r}")
    return fetch(filename, revision)


# Preprocessing (upstream inference.py)


def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    """Truncate audio to last n seconds or pad with zeros to meet n seconds.

    Verbatim from smart-turn audio_utils.py."""
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        # Pad with zeros at the beginning
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
    return audio_array


def feature_extractor():
    """WhisperFeatureExtractor(chunk_length=8) pinned to its numpy front end:
    transformers picks the torch STFT when torch is importable, but upstream's
    inference requirements and Pipecat's analyzer run the numpy path."""
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(chunk_length=MAX_AUDIO_SECONDS)
    fe._torch_extract_fbank_features = fe._np_extract_fbank_features
    return fe


def features(fe, window: np.ndarray) -> np.ndarray:
    """inference.py's feature call on an 8 s window -> [1, 80, 800] float32."""
    inputs = fe(window, sampling_rate=SAMPLE_RATE, return_tensors="np", padding="max_length",
                max_length=MAX_AUDIO_SAMPLES, truncation=True, do_normalize=True)
    return inputs.input_features.astype(np.float32)


def normalized_window(fe, window: np.ndarray) -> np.ndarray:
    """The waveform the STFT sees: zero_mean_unit_var_norm over the whole
    window, padding included (the attention mask is all ones)."""
    col = np.asarray([np.asarray(window, dtype=np.float32)]).T  # [n, 1], as __call__ builds it
    out = fe.zero_mean_unit_var_norm([col], attention_mask=[np.ones(len(window), np.int32)],
                                     padding_value=fe.padding_value)
    return np.asarray(out[0][:, 0], dtype=np.float32)


# ONNX -> PyTorch weight map


def map_onnx_weights(model_proto) -> dict[str, np.ndarray]:
    """{torch state_dict name: array in torch layout}.

    Named initializers (inner.<name>) map 1:1. Each anonymous MatMul weight
    W (x @ W, stored [in, out]) is named by the bias Add consuming its
    output or, for the bias-free k_proj, by the self_attn_layer_norm feeding
    it; the torch weight is W.T. Fails on a leftover initializer or a slot
    filled twice.
    """
    from onnx import numpy_helper

    g = model_proto.graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    producers = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    out = {k.removeprefix("inner."): np.ascontiguousarray(v.astype(np.float32))
           for k, v in inits.items() if k.startswith("inner.")}
    used = {k for k in inits if k.startswith("inner.")}
    for node in g.node:
        if node.op_type != "MatMul" or node.input[1] not in inits:
            continue
        wname = node.input[1]
        w = inits[wname]
        if w.ndim != 2:
            raise SystemExit(f"error: weight map: MatMul weight {wname} is {w.shape}")
        bias_adds = [c for c in consumers.get(node.output[0], [])
                     if c.op_type == "Add" and any(i.startswith("inner.") and i.endswith(".bias")
                                                   for i in c.input)]
        if len(bias_adds) == 1:
            bias = next(i for i in bias_adds[0].input if i.startswith("inner."))
            target = bias[len("inner."):-len(".bias")] + ".weight"
        elif not bias_adds:
            ln = producers.get(node.input[0])
            m = None
            if ln is not None and ln.op_type == "LayerNormalization":
                m = re.fullmatch(r"inner\.encoder\.layers\.(\d+)\.self_attn_layer_norm\.weight", ln.input[1])
            if m is None:
                raise SystemExit(f"error: weight map: bias-free MatMul {wname} is not fed by a "
                                 f"self_attn_layer_norm")
            target = f"encoder.layers.{m[1]}.self_attn.k_proj.weight"
        else:
            raise SystemExit(f"error: weight map: MatMul {wname} has {len(bias_adds)} bias consumers")
        if target in out:
            raise SystemExit(f"error: weight map: {target} filled twice (second from {wname})")
        out[target] = np.ascontiguousarray(w.T.astype(np.float32))
        used.add(wname)

    # Every remaining initializer must be a known scalar / shape constant.
    left = {k: v for k, v in inits.items() if k not in used}
    for k, v in left.items():
        if v.size > 4:
            raise SystemExit(f"error: weight map: initializer {k} {v.shape} not consumed")
    # The attention scale is split over q and k: (q * s) @ (k * s)^T with
    # s = head_dim ** -0.25 (torch SDPA decomposition).
    scales = [v for v in left.values() if v.dtype == np.float32 and v.shape == (1,)
              and abs(float(v[0]) - (D_MODEL // N_HEADS) ** -0.25) < 1e-7]
    if len(scales) != 1:
        raise SystemExit("error: weight map: attention scale head_dim**-0.25 not found")
    return out


def check_geometry(sd: dict[str, np.ndarray]) -> None:
    """Assert the shapes the converter writes as hyperparameters."""
    want = {
        "encoder.conv1.weight": (D_MODEL, N_MELS, 3),
        "encoder.conv2.weight": (D_MODEL, D_MODEL, 3),
        "encoder.embed_positions.weight": (N_POSITIONS, D_MODEL),
        "pool_attention.0.weight": (POOL_DIM, D_MODEL),
        "pool_attention.2.weight": (1, POOL_DIM),
        "classifier.0.weight": (CLS_DIMS[0], D_MODEL),
        "classifier.1.weight": (CLS_DIMS[0],),
        "classifier.4.weight": (CLS_DIMS[1], CLS_DIMS[0]),
        "classifier.6.weight": (1, CLS_DIMS[1]),
    }
    for i in range(N_LAYERS):
        p = f"encoder.layers.{i}."
        for k in ("q", "k", "v", "out"):
            want[p + f"self_attn.{k}_proj.weight"] = (D_MODEL, D_MODEL)
        want[p + "fc1.weight"] = (FFN_DIM, D_MODEL)
        want[p + "fc2.weight"] = (D_MODEL, FFN_DIM)
    bad = [(k, sd[k].shape if k in sd else None, v) for k, v in want.items()
           if k not in sd or sd[k].shape != v]
    if bad:
        raise SystemExit("error: geometry: " + "; ".join(f"{k}={g} (want {w})" for k, g, w in bad))
    if f"encoder.layers.{N_LAYERS}.fc1.weight" in sd:
        raise SystemExit(f"error: geometry: more than {N_LAYERS} encoder layers")
    if any(k.endswith("k_proj.bias") for k in sd):
        raise SystemExit("error: geometry: unexpected k_proj bias")


# PyTorch twin (upstream train.py SmartTurnV3Model)


def build_twin(sd: dict[str, np.ndarray]):
    """SmartTurnV3Model (train.py, verbatim) on openai/whisper-tiny's encoder
    config, with the mapped weights loaded strictly, in eval mode."""
    import torch
    from torch import nn
    from torch.nn.functional import softmax
    from transformers import WhisperConfig, WhisperPreTrainedModel
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    # ---- verbatim from smart-turn train.py (labels branch dropped) ----
    class SmartTurnV3Model(WhisperPreTrainedModel):
        def __init__(self, config: WhisperConfig):
            super().__init__(config)

            config.max_source_positions = 400
            self.encoder = WhisperEncoder(config)

            # Use the encoder's hidden size
            hidden_size = config.d_model

            self.pool_attention = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

        def forward(self, input_features, labels=None):
            encoder_outputs = self.encoder(input_features=input_features)

            hidden_states = encoder_outputs.last_hidden_state

            attention_weights = self.pool_attention(hidden_states)
            attention_weights = softmax(attention_weights, dim=1)
            pooled = torch.sum(hidden_states * attention_weights, dim=1)

            logits = self.classifier(pooled)

            probs = torch.sigmoid(logits)
            return {"logits": probs}
    # ---- end verbatim ----

    config = WhisperConfig(
        d_model=D_MODEL, encoder_layers=N_LAYERS, encoder_attention_heads=N_HEADS,
        encoder_ffn_dim=FFN_DIM, num_mel_bins=N_MELS, max_source_positions=1500,
        activation_function="gelu", dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
        scale_embedding=False)
    config._attn_implementation = "sdpa"
    model = SmartTurnV3Model(config)
    state = {k: torch.from_numpy(v.copy()) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"error: twin load: missing {missing} unexpected {unexpected}")
    model.eval()
    return model
