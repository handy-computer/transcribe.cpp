"""ecapa_numpy.py - the readable specification of the ecapa_tdnn forward pass.

A complete ECAPA-TDNN language-ID forward pass in NumPy, driven entirely by
an ecapa_tdnn GGUF: weights, layout, and every hyperparameter come out of the
file, nothing is hardcoded from the checkpoint. It exists for three reasons:

  1. It proves the converter's four conversion-time rewrites (the C++ graph — BN as affine, forward BN folds, ASP split, MFA split) are
     exact: its dumps are compared against the SpeechBrain reference dumps
     with the same tolerance file the C++ has to meet.
  2. It is the executable version of the C++ graph, in ~300 lines
     a reviewer can read top to bottom, for when the ggml graph disagrees
     with SpeechBrain and someone has to work out which one is wrong.
  3. It bisects failures: every stage tensor in the C++ graph is
     returned, so a C++ mismatch can be localised to a layer without
     rebuilding anything.

Deliberate numerical choices:

  * Everything after the FFT is float32, accumulated with NumPy's float32
    `@`. float64 would hide exactly the accumulation-order differences the
    C++ is being measured for. `np.fft.rfft` always computes in float64
    (NumPy has no float32 FFT), so the power spectrum is cast straight back
    down to float32 — that is the only place this file is wider than the
    reference, and it is the reason `fe.mel` is the loosest tensor in the
    tolerance file.
  * The conv is written the way the C++ graph does it:
    reflect-pad the time axis once, then sum K matmuls against shifted
    slices, one per kernel tap. Not `np.convolve`, not im2col — the point is
    to mirror the graph, not to be fast.
  * Activations are carried TIME-MAJOR `[T, C]`. That is what both dumpers
    write to disk (C++ ggml `ne = [C, T]` lands row-major as `[T, C]`), so
    no dump-time transpose can silently paper over a layout error.

Usage:

    from lib.ecapa_numpy import EcapaNumpy
    model = EcapaNumpy("models/.../lang-id-voxlingua107-ecapa-F32.gguf")
    out = model.forward(pcm_float32_16k)
    out["cls.logits_raw"]      # [107]
    out["prediction"]["code"]  # e.g. "en"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import gguf
from gguf import GGMLQuantizationType


class EcapaNumpy:
    """A GGUF-driven ECAPA-TDNN language identifier, in NumPy."""

    def __init__(self, gguf_path: str | Path) -> None:
        self.path = Path(gguf_path)
        reader = gguf.GGUFReader(str(self.path), "r")

        # ---- metadata ----------------------------------------------------
        self.meta: dict[str, Any] = {}
        for name, field in reader.fields.items():
            if not field.types:
                continue
            self.meta[name] = field.contents()

        arch = self.meta.get("general.architecture")
        if arch != "ecapa_tdnn":
            raise ValueError(
                f"{self.path}: general.architecture is {arch!r}, expected "
                "'ecapa_tdnn'")
        fmt = int(self.meta.get("stt.ecapa_tdnn.format_version", -1))
        if fmt != 1:
            raise ValueError(
                f"{self.path}: stt.ecapa_tdnn.format_version is {fmt}, expected 1")

        # ---- tensors -----------------------------------------------------
        # Everything is dequantised up front to float32 in its numpy shape
        # (= reversed ggml ne). F16 widens by cast; Q8_0 goes through
        # gguf.quants.dequantize, which is the same block-scale arithmetic
        # ggml's vec_dot does, so an F16/Q8_0 run here is a fair proxy for
        # the C++ run on the same file.
        self.tensors: dict[str, np.ndarray] = {}
        self.tensor_types: dict[str, str] = {}
        for t in reader.tensors:
            data = t.data
            if t.tensor_type == GGMLQuantizationType.F32:
                arr = np.asarray(data, dtype=np.float32)
            elif t.tensor_type == GGMLQuantizationType.F16:
                arr = np.asarray(data).astype(np.float32)
            else:
                arr = gguf.quants.dequantize(
                    np.asarray(data), t.tensor_type).astype(np.float32)
            self.tensors[t.name] = np.ascontiguousarray(arr)
            self.tensor_types[t.name] = t.tensor_type.name

        # ---- shape / hyperparameter cache --------------------------------
        self.sample_rate = int(self.meta["stt.frontend.sample_rate"])
        self.n_fft = int(self.meta["stt.frontend.n_fft"])
        self.hop = int(self.meta["stt.frontend.hop_length"])
        self.win = int(self.meta["stt.frontend.win_length"])
        self.n_mels = int(self.meta["stt.frontend.num_mels"])
        self.log_floor = float(self.meta["stt.frontend.log_clamp_min"])
        self.top_db = float(self.meta["stt.frontend.top_db"])
        self.window_kind = str(self.meta["stt.frontend.window"])
        self.normalize = str(self.meta["stt.frontend.normalize"])

        self.channels = [int(c) for c in self.meta["stt.ecapa_tdnn.channels"]]
        self.kernel_sizes = [int(k) for k in
                             self.meta["stt.ecapa_tdnn.kernel_sizes"]]
        self.dilations = [int(d) for d in self.meta["stt.ecapa_tdnn.dilations"]]
        self.res2net_scale = int(self.meta["stt.ecapa_tdnn.res2net_scale"])
        self.asp_eps = float(self.meta["stt.ecapa_tdnn.asp_eps"])
        self.embedding_dim = int(self.meta["stt.ecapa_tdnn.embedding_dim"])
        self.leaky_slope = float(self.meta["stt.ecapa_tdnn.classifier_leaky_slope"])

        self.codes = list(self.meta["stt.langid.labels.codes"])
        self.names = list(self.meta["stt.langid.labels.names"])
        self.aliases = {a: c for a, c in
                        (s.split("=", 1)
                         for s in self.meta["stt.langid.labels.aliases"])}

        if self.window_kind != "hamming_periodic":
            raise ValueError(f"unsupported window {self.window_kind!r}")
        if self.normalize != "sentence_mean":
            raise ValueError(f"unsupported normalize {self.normalize!r}")

        # Periodic Hamming: w[n] = 0.54 - 0.46*cos(2*pi*n/N), n = 0..N-1.
        # NOT the symmetric variant (which divides by N-1) — the two differ
        # by 5.5e-03 at the edges, far above any tolerance here.
        n = np.arange(self.win, dtype=np.float64)
        self.window = (0.54 - 0.46 * np.cos(2.0 * np.pi * n / self.win)
                       ).astype(np.float32)

        # [60, 201] mel-major, straight from SpeechBrain's Filterbank.
        self.filters = self.tensors["frontend.mel_filterbank"]
        if self.filters.shape != (self.n_mels, self.n_fft // 2 + 1):
            raise ValueError(
                f"frontend.mel_filterbank is {self.filters.shape}, expected "
                f"{(self.n_mels, self.n_fft // 2 + 1)}")

    # -- small helpers -----------------------------------------------------

    def t(self, name: str) -> np.ndarray:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"{self.path.name}: missing tensor {name!r}") from None

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, np.float32(0.0))

    def _leaky(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, x, x * np.float32(self.leaky_slope)
                        ).astype(np.float32)

    @staticmethod
    def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """`[T, IC] @ [OC, IC]^T + [OC]` -> `[T, OC]`, all float32.

        This is the 1x1 convolution: with channel-innermost activations a
        pointwise conv is exactly a matmul, which is why the GGUF stores 1x1
        kernels as plain 2-D `[OC, IC]` matrices.
        """
        return (x @ w.T + b).astype(np.float32)

    @staticmethod
    def _bn(x: np.ndarray, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
        """`y = x * scale + shift`, the converter's affine form of BatchNorm.

        Broadcasts over time. The scale/shift pair already absorbed
        gamma/beta/running_mean/running_var/eps at conversion time. It cannot
        fold into the conv: a TDNNBlock is conv -> ReLU -> BN, and the ReLU
        sits between them.
        """
        return (x * scale + shift).astype(np.float32)

    def _conv_taps(self, x: np.ndarray, w: np.ndarray, b: np.ndarray,
                   dilation: int) -> np.ndarray:
        """Dilated conv as a sum of per-tap matmuls, mirroring the ggml graph.

        `w` is tap-major `[K, OC, IC]`. SpeechBrain uses `padding="same"` with
        `padding_mode="reflect"`, i.e. `p = dilation*(K-1)/2` reflected frames
        on each side, so the output keeps T frames. One `np.pad(mode=
        "reflect")` here stands in for the C++'s single `ggml_get_rows`
        gather with a precomputed mirror index; the arithmetic is identical.
        """
        k = w.shape[0]
        if k == 1:
            return self._linear(x, w[0], b)
        p = dilation * (k - 1) // 2
        if x.shape[0] <= p:
            raise ValueError(
                f"need more than {p} frames for a k={k} d={dilation} conv, "
                f"got {x.shape[0]}")
        xp = np.pad(x, ((p, p), (0, 0)), mode="reflect")
        T = x.shape[0]
        y = np.zeros((T, w.shape[1]), dtype=np.float32)
        for tap in range(k):
            off = tap * dilation
            y += xp[off:off + T] @ w[tap].T
        return (y + b).astype(np.float32)

    def _tdnn(self, x: np.ndarray, prefix: str, dilation: int,
              tap_major: bool) -> np.ndarray:
        """SpeechBrain TDNNBlock: conv -> ReLU -> BN.

        (The Dropout1d(p=0) fourth child is an exact identity in eval mode
        and carries no weights, so it is absent here and from the GGUF.)
        """
        stem = f"{prefix}.conv" if tap_major else prefix
        w = self.t(f"{stem}.weight")
        b = self.t(f"{stem}.bias")
        y = (self._conv_taps(x, w, b, dilation) if tap_major
             else self._linear(x, w, b))
        y = self._relu(y)
        return self._bn(y, self.t(f"{prefix}.bn.scale"),
                        self.t(f"{prefix}.bn.shift"))

    # -- front end ---------------------------------------------------------

    def mel(self, pcm: np.ndarray) -> np.ndarray:
        """16 kHz mono float32 -> `[T, 60]` sentence-mean-normalised log-mel.

        Matches SpeechBrain's `Fbank` + `InputNormalization(norm_type=
        "sentence", std_norm=False)` exactly: centred STFT with 200 samples
        of ZERO padding each side (`pad_mode="constant"`, not reflect — the
        reflect padding in this model is inside the convolutions), periodic
        Hamming window, power spectrum, 60 triangular mel filters,
        `10*log10(max(x, 1e-10))`, an 80 dB floor taken over time AND
        frequency of the whole utterance, then per-bin mean subtraction.
        """
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        n = pcm.size
        T = n // self.hop + 1
        pad = self.n_fft // 2
        xp = np.pad(pcm, (pad, pad), mode="constant")

        idx = np.arange(self.win)[None, :] + (np.arange(T) * self.hop)[:, None]
        frames = xp[idx] * self.window                       # [T, 400] f32

        spec = np.fft.rfft(frames, n=self.n_fft, axis=-1)    # f64, always
        # Power spectrum re^2 + im^2 (SpeechBrain's spectral_magnitude with
        # power=1 over a squared-and-summed real/imag pair). Back to f32
        # immediately so the rest of the front end matches the reference's
        # precision.
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)

        mel = (power @ self.filters.T).astype(np.float32)    # [T, 60]
        db = (10.0 * np.log10(np.maximum(mel, np.float32(self.log_floor)))
              ).astype(np.float32)
        # ref_value = 1.0 so db_multiplier = log10(1) = 0: no reference term.
        db = np.maximum(db, db.max() - np.float32(self.top_db))
        return (db - db.mean(axis=0, keepdims=True)).astype(np.float32)

    # -- forward -----------------------------------------------------------

    def forward(self, pcm: np.ndarray) -> dict[str, Any]:
        """Run the whole model; return every the C++ graph stage tensor."""
        out: dict[str, Any] = {}

        x = self.mel(pcm)                                    # [T, 60]
        out["fe.mel"] = x

        # blocks.0: TDNN 60 -> 1024, k=5, d=1 (reflect pad 2).
        x = self._tdnn(x, "blk.0", self.dilations[0], tap_major=True)
        out["enc.blk.0.out"] = x

        # blocks.1..3: SERes2Net, in == out channels so the residual is
        # the identity (no shortcut projection in this checkpoint).
        block_outs: list[np.ndarray] = []
        for i in (1, 2, 3):
            p = f"blk.{i}"
            residual = x

            x = self._tdnn(x, f"{p}.tdnn1", 1, tap_major=False)
            if i == 1:
                out["enc.blk.1.tdnn1.out"] = x

            # Res2Net, scale 8: split into 8 chunks of 128 channels.
            #   y0 = x0                       (passed through untouched)
            #   y1 = B0(x1)
            #   yi = B(i-1)(xi + y(i-1))      i = 2..7
            # The addend is the PREVIOUS sub-block's output, which is what
            # makes this a cascade rather than 7 independent branches.
            chunks = np.split(x, self.res2net_scale, axis=1)
            ys: list[np.ndarray] = [chunks[0]]
            y_prev = chunks[0]
            for j in range(1, self.res2net_scale):
                inp = chunks[j] if j == 1 else (chunks[j] + y_prev
                                                ).astype(np.float32)
                y_prev = self._tdnn(inp, f"{p}.res2.{j - 1}",
                                    self.dilations[i], tap_major=True)
                ys.append(y_prev)
            x = np.ascontiguousarray(np.concatenate(ys, axis=1))
            if i == 1:
                out["enc.blk.1.res2.out"] = x

            x = self._tdnn(x, f"{p}.tdnn2", 1, tap_major=False)

            # Squeeze-and-excitation: mean over time, two 1x1 convs, sigmoid
            # gate, per-channel rescale.
            s = x.mean(axis=0).astype(np.float32)
            s = self._relu(self.t(f"{p}.se.c1.weight") @ s + self.t(f"{p}.se.c1.bias"))
            s = 1.0 / (1.0 + np.exp(-(self.t(f"{p}.se.c2.weight") @ s
                                      + self.t(f"{p}.se.c2.bias"))))
            x = (x * s.astype(np.float32)).astype(np.float32)
            if i == 1:
                out["enc.blk.1.se.out"] = x

            x = (x + residual).astype(np.float32)
            out[f"enc.blk.{i}.out"] = x
            block_outs.append(x)

        # MFA. SpeechBrain concatenates blocks.1/2/3 and runs one 1x1 conv;
        # the converter split that weight into three blocks, so the concat
        # copy disappears and this is three matmuls summed. Same FLOPs, same
        # result.
        y = (block_outs[0] @ self.t("mfa.w1.weight").T
             + block_outs[1] @ self.t("mfa.w2.weight").T
             + block_outs[2] @ self.t("mfa.w3.weight").T
             + self.t("mfa.bias")).astype(np.float32)
        y = self._bn(self._relu(y), self.t("mfa.bn.scale"),
                     self.t("mfa.bn.shift"))
        out["enc.mfa.out"] = y

        # Attentive statistics pooling with global context.
        eps = np.float32(self.asp_eps)
        mean = y.mean(axis=0).astype(np.float32)
        var = ((y - mean) ** 2).mean(axis=0).astype(np.float32)
        std = np.sqrt(np.maximum(var, eps)).astype(np.float32)

        # cat([x, mean, std]) -> 1x1 conv, split at conversion time into
        # wx (time-varying) and wm/ws (constant over T), so the mean/std
        # contribution collapses into the bias vector.
        const = (self.t("asp.tdnn.mean.weight") @ mean + self.t("asp.tdnn.std.weight") @ std
                 + self.t("asp.tdnn.bias")).astype(np.float32)
        a = (y @ self.t("asp.tdnn.x.weight").T + const).astype(np.float32)
        a = self._bn(self._relu(a), self.t("asp.tdnn.bn.scale"),
                     self.t("asp.tdnn.bn.shift"))
        a = np.tanh(a).astype(np.float32)
        logits = self._linear(a, self.t("asp.attn.weight"), self.t("asp.attn.bias"))
        out["enc.asp.attn_logits"] = logits                  # [T, 3072]

        # Softmax over TIME, per channel. (SpeechBrain masks padded frames
        # to -inf first; with a single unpadded utterance the mask is all
        # ones and the masked_fill is a no-op.)
        z = logits - logits.max(axis=0, keepdims=True)
        e = np.exp(z).astype(np.float32)
        attn = (e / e.sum(axis=0, keepdims=True)).astype(np.float32)

        mu = (attn * y).sum(axis=0).astype(np.float32)
        sigma = np.sqrt(np.maximum((attn * (y - mu) ** 2).sum(axis=0), eps)
                        ).astype(np.float32)
        pooled = np.concatenate([mu, sigma]).astype(np.float32)  # [6144]
        out["enc.asp.out"] = pooled

        # asp_bn is folded into fc, so this single matmul is BN + linear.
        emb = (self.t("fc.weight") @ pooled + self.t("fc.bias")).astype(np.float32)
        out["enc.emb"] = emb                                 # [256]

        # Classifier. cls.bn0 is folded into cls.l1 and cls.bn1 into
        # cls.out, but both LeakyReLUs come BEFORE their BN in SpeechBrain,
        # so they survive: leaky -> l1 -> leaky -> out.
        h = self._leaky(emb)
        h = (self.t("cls.l1.weight") @ h + self.t("cls.l1.bias")).astype(np.float32)
        hidden = self._leaky(h)
        out["cls.hidden"] = hidden                           # [512]

        z = (self.t("cls.out.weight") @ hidden + self.t("cls.out.bias")
             ).astype(np.float32)
        out["cls.logits_raw"] = z                            # [107]

        m = z.max()
        log_probs = (z - m - np.log(np.exp(z - m).sum())).astype(np.float32)
        out["cls.log_probs"] = log_probs

        order = np.argsort(-log_probs)
        best = int(order[0])
        out["prediction"] = {
            "label_index": best,
            "code": self.codes[best],
            "name": self.names[best],
            "log_prob": float(log_probs[best]),
            "top5": [
                {
                    "index": int(i),
                    "code": self.codes[int(i)],
                    "name": self.names[int(i)],
                    "log_prob": float(log_probs[int(i)]),
                }
                for i in order[:5]
            ],
        }
        return out
