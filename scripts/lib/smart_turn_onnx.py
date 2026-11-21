"""Smart Turn v3.2 (pipecat-ai) source handling shared by the converter and the
reference dumper.

Upstream publishes Smart Turn only as ONNX files (no PyTorch checkpoint), so
this module is the single copy of:

  - the pinned source: repo, revision and sha256 of every file we read;
  - upstream's preprocessing, verbatim from smart-turn's inference.py /
    audio_utils.py (keep the last 8 s or left-pad with zeros, then
    WhisperFeatureExtractor(chunk_length=8) with do_normalize=True);
  - the ONNX -> PyTorch weight map. The exporter stored most Linear weights
    as anonymous, transposed MatMul initializers (val_17, ...); they are
    named here by walking the graph (the bias Add that consumes each MatMul,
    or for the bias-free k_proj, the layer's LayerNorm feeding it);
  - SmartTurnV3Model, copied from smart-turn's train.py, so the mapped
    weights can be loaded into the upstream module tree and run as an
    instrumented twin of the ONNX graph.

See docs/porting/families/smart_turn.md.
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
# github.com/pipecat-ai/smart-turn: inference.py, audio_utils.py, train.py.
UPSTREAM_CODE = "https://github.com/pipecat-ai/smart-turn"
UPSTREAM_CODE_COMMIT = "4e12f2b"

SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 8
MAX_AUDIO_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE  # 128000
N_MELS = 80
N_FRAMES = 800          # mel frames fed to the encoder
N_POSITIONS = 400       # encoder positions (conv2 has stride 2)
THRESHOLD = 0.5         # inference.py: prediction = 1 if probability > 0.5
NORMALIZE_EPS = 1e-7    # zero_mean_unit_var_norm: / sqrt(var + 1e-7)

# whisper-tiny encoder geometry (train.py: from_pretrained("openai/whisper-tiny")
# with config.max_source_positions = 400). Asserted against the ONNX shapes.
D_MODEL = 384
N_LAYERS = 4
N_HEADS = 6
FFN_DIM = 1536
POOL_DIM = 256
CLS_DIMS = (256, 64)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(filename: str = ONNX_FP32, revision: str = HF_REVISION) -> Path:
    """Download one pinned file of HF_REPO and verify its sha256.

    Lands in $TRANSCRIBE_MODELS_DIR/smart-turn-v3@<revision>/ when that is
    set (the hf_source.download_snapshot convention), else the HF cache.
    """
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


# ---------------------------------------------------------------------------
# Preprocessing (upstream inference.py)
# ---------------------------------------------------------------------------


def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    """Truncate audio to last n seconds or pad with zeros to meet n seconds.

    Verbatim from smart-turn audio_utils.py (and Pipecat's
    LocalSmartTurnAnalyzerV3._predict_endpoint)."""
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        # Pad with zeros at the beginning
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
    return audio_array


def feature_extractor():
    """WhisperFeatureExtractor(chunk_length=8) on its numpy front end.

    transformers picks the torch STFT when torch is importable. Upstream's
    inference requirements (requirements_inference.txt) and Pipecat's
    production analyzer (a vendored numpy copy, _whisper_features.py) both
    run the numpy path, so it is the canonical one; pin it here."""
    import transformers.models.whisper.feature_extraction_whisper as few

    fe = few.WhisperFeatureExtractor(chunk_length=MAX_AUDIO_SECONDS)
    fe_call_numpy = fe._np_extract_fbank_features

    def numpy_only(waveform, device="cpu"):
        return fe_call_numpy(waveform, device)

    fe._torch_extract_fbank_features = numpy_only
    return fe


def features(fe, window: np.ndarray) -> np.ndarray:
    """inference.py's feature call on an 8 s window -> [1, 80, 800] float32."""
    inputs = fe(
        window,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=MAX_AUDIO_SAMPLES,
        truncation=True,
        do_normalize=True,
    )
    x = inputs.input_features.squeeze(0).astype(np.float32)
    return np.expand_dims(x, axis=0)


def normalized_window(fe, window: np.ndarray) -> np.ndarray:
    """The waveform the STFT sees: WhisperFeatureExtractor's own
    zero_mean_unit_var_norm over the whole 128000-sample window (padding
    included; the attention mask is all ones)."""
    col = np.asarray([np.asarray(window, dtype=np.float32)]).T  # [n, 1], as __call__ builds it
    out = fe.zero_mean_unit_var_norm([col], attention_mask=[np.ones(len(window), np.int32)],
                                     padding_value=fe.padding_value)
    return np.asarray(out[0][:, 0], dtype=np.float32)


# ---------------------------------------------------------------------------
# ONNX -> PyTorch weight map
# ---------------------------------------------------------------------------

_LAYER_BIAS = re.compile(r"^inner\.(encoder\.layers\.(\d+)\.(self_attn\.(q|v|out)_proj|fc1|fc2))\.bias$")


def map_onnx_weights(model_proto) -> tuple[dict[str, np.ndarray], list[str]]:
    """Return ({torch state_dict name: array in torch layout}, notes).

    Named initializers (inner.<name>) map 1:1. Each anonymous MatMul weight
    W (x @ W, stored [in, out]) is named by the bias Add consuming its
    output, or, for k_proj (WhisperAttention has no k bias), by the layer's
    self_attn_layer_norm feeding it; the torch weight is W.T ([out, in]).
    Raises when any initializer is left over or a slot is filled twice.
    """
    from onnx import numpy_helper

    g = model_proto.graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    producers = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    out: dict[str, np.ndarray] = {}
    notes: list[str] = []

    def put(name: str, arr: np.ndarray, src: str) -> None:
        if name in out:
            raise SystemExit(f"error: weight map: {name} filled twice (second from {src})")
        out[name] = np.ascontiguousarray(arr.astype(np.float32))
        notes.append(f"{src} -> {name} {list(arr.shape)}")

    for name, arr in inits.items():
        if name.startswith("inner."):
            put(name[len("inner."):], arr, name)

    used = {n for n in inits if n.startswith("inner.")}
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
        put(target, w.T, wname)
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
    notes.append(f"attention scale constant {float(scales[0][0])!r} = head_dim**-0.25 on q and k")
    return out, notes


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


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> np.ndarray:
    """transformers' Whisper sinusoids() (float32 torch math), for the
    positional-table check."""
    import torch

    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1).numpy()


# ---------------------------------------------------------------------------
# PyTorch twin (upstream train.py SmartTurnV3Model)
# ---------------------------------------------------------------------------


def whisper_tiny_config():
    """openai/whisper-tiny's encoder config (config.json), as train.py loads it."""
    from transformers import WhisperConfig

    return WhisperConfig(
        d_model=D_MODEL,
        encoder_layers=N_LAYERS,
        encoder_attention_heads=N_HEADS,
        encoder_ffn_dim=FFN_DIM,
        num_mel_bins=N_MELS,
        max_source_positions=1500,  # SmartTurnV3Model.__init__ overrides to 400
        activation_function="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        scale_embedding=False,
    )


def build_twin(sd: dict[str, np.ndarray], attn_implementation: str = "sdpa"):
    """SmartTurnV3Model (train.py, verbatim forward) with the mapped weights,
    loaded strictly, in eval mode."""
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

    config = whisper_tiny_config()
    config._attn_implementation = attn_implementation
    model = SmartTurnV3Model(config)
    state = {k: torch.from_numpy(v.copy()) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"error: twin load: missing {missing} unexpected {unexpected}")
    model.eval()
    return model
