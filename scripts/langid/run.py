#!/usr/bin/env python3
"""
run.py - score a language ID evaluation corpus with the SpeechBrain
reference or transcribe.cpp (through the Python binding, one model load per
run), and write one self-describing JSONL report.

Usage:
  uv run --project scripts/envs/ecapa_tdnn scripts/langid/run.py \
      --engine speechbrain --model speechbrain/lang-id-voxlingua107-ecapa \
      --manifest samples/langid/fleurs-*.manifest.jsonl \
      --crops 3,5,10,full --trim none \
      --out reports/langid/ref-speechbrain-untrimmed.jsonl

  uv run --project scripts/envs/ecapa_tdnn scripts/langid/run.py \
      --engine cpp --model models/lang-id-voxlingua107-ecapa/lang-id-voxlingua107-ecapa-F32.gguf \
      --manifest samples/langid/fleurs-*.manifest.jsonl \
      --crops 3,5,10,full --trim none --backend cpu --threads 4 \
      --library build-shared/src/libtranscribe.dylib \
      --out reports/langid/cpp-f32-untrimmed.jsonl

Output JSONL:
  line 1   {"type":"header","engine","model","recipe":{...},"labels":[107],"created"}
  line n   {"id","language","crop_s","trim","audio_s","logits":[107],"top1","top1_prob"}

All 107 raw logits are stored per (utterance, crop) so every restricted
selection can be scored offline by score.py from a single sweep
. `top1`/`top1_prob` are the open-set decision, kept as a
convenience; score.py recomputes everything from `logits`.

Crop semantics
--------------
`--trim none`  : the first N seconds of the raw clip. This is what the
                 langid.cpp original evaluation measured, so "3 s" is roughly
                 2 s of speech on FLEURS (the clips open with silence).
`--trim energy`: drop everything before the first 20 ms frame whose RMS
                 exceeds 5% of the clip's maximum frame RMS, then take the
                 first N seconds of what is left.
`full`         : the whole clip (after trimming, for --trim energy).
A clip shorter than the crop is used whole; the row's `audio_s` records
what was actually scored.

Both engines read the SAME 16-bit wav files
-------------------------------------------
Crops are materialised once to `build/langid/crops/<trim>/<crop>/<id>.wav`
and both engines read those files back. Scoring in-memory floats on one
side and 16-bit wavs on the other would make compare.py's 99.9% agreement
gate measure PCM rounding instead of the port. The crop directory is
keyed by the recipe, not by the run name, so a cpp run and the reference
run with the same recipe share bytes on disk.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SAMPLE_RATE = 16000
N_LABELS = 107
TRIM_FRAME_MS = 20
TRIM_REL_THRESHOLD = 0.05
MIN_KEEP_S = 0.5  # the LANGID role's TRANSCRIBE_ERR_INPUT_TOO_SHORT floor


# ---------------------------------------------------------------------------
# Manifests, crops
# ---------------------------------------------------------------------------


def read_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    pcm, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if int(sr) != SAMPLE_RATE:
        raise SystemExit(f"error: {path} is {sr} Hz, expected {SAMPLE_RATE}")
    return np.ascontiguousarray(pcm, dtype=np.float32)


def write_wav(path: Path, pcm: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), pcm, SAMPLE_RATE, subtype="PCM_16")


def energy_trim_start(pcm: np.ndarray) -> int:
    """First sample of speech: the start of the first 20 ms frame whose RMS
    exceeds 5% of the loudest frame's RMS.

    Relative to the clip's own maximum, so it is level-independent. The
    result is clamped so at least MIN_KEEP_S remains: a clip whose only
    loud frame is at the very end would otherwise be trimmed below the
    library's 500 ms floor.
    """
    frame = int(TRIM_FRAME_MS * SAMPLE_RATE / 1000)  # 320
    n_frames = pcm.size // frame
    if n_frames == 0:
        return 0
    frames = pcm[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = float(rms.max())
    if peak <= 0.0:
        return 0
    above = np.flatnonzero(rms > TRIM_REL_THRESHOLD * peak)
    if above.size == 0:
        return 0
    start = int(above[0]) * frame
    return min(start, max(0, pcm.size - int(MIN_KEEP_S * SAMPLE_RATE)))


def make_crop(pcm: np.ndarray, crop: str | int, trim: str) -> np.ndarray:
    start = energy_trim_start(pcm) if trim == "energy" else 0
    cut = pcm[start:]
    if crop == "full":
        return cut
    n = int(float(crop) * SAMPLE_RATE)
    return cut[:n] if cut.size > n else cut


def crop_dir(trim: str, crop: str | int) -> Path:
    return REPO_ROOT / "build" / "langid" / "crops" / trim / str(crop)


def materialise_crops(rows: list[dict], crop: str | int, trim: str) -> list[tuple[dict, Path]]:
    """Write every (utterance, crop) wav and return (row, path) pairs.

    Always rewrites: the write is cheap next to a forward pass, and an
    unconditional write means a stale crop from an earlier recipe can never
    be scored by accident.
    """
    out = crop_dir(trim, crop)
    out.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[dict, Path]] = []
    for row in rows:
        src = REPO_ROOT / row["audio"]
        dst = out / f"{row['id']}.wav"
        write_wav(dst, make_crop(read_wav(src), crop, trim))
        pairs.append((row, dst))
    return pairs


# ---------------------------------------------------------------------------
# SpeechBrain engine
# ---------------------------------------------------------------------------


class _DumperArgs:
    """Argument shim for the dumper's loader (same pattern as
    make_samples_fleurs.py)."""

    def __init__(self, model: str, revision: str, threads: int) -> None:
        self.model = model
        self.revision = revision
        self.device = "cpu"
        self.torch_threads = threads


class SpeechBrainEngine:
    """The pinned SpeechBrain checkpoint, loaded once, scored one clip at a
    time.

    Batch of one on purpose. Batching pads the shorter clips and drives
    `wav_lens` off 1.0, which changes InputNormalization's span and the ASP
    masks - i.e. it changes the numbers this file is supposed to be the
    reference for.

    The logits are taken from a forward hook on `mods.classifier.out`, the
    Linear(512->107) whose output is pre-softmax, exactly as the reference
    dumper's `cls.logits_raw`. `classify_batch` only returns log-softmax.
    """

    logit_kind = "pre_softmax"

    def __init__(self, model: str, revision: str, threads: int) -> None:
        import torch

        import dump_reference_ecapa_tdnn_speechbrain as dumper

        self.dumper = dumper
        self.torch = torch
        args = _DumperArgs(model, revision, threads)
        dumper.configure_torch(args)
        self.clf, self.checkpoint_dir = dumper.load_reference(args)

        ind2lab = self.clf.hparams.label_encoder.ind2lab
        if len(ind2lab) != N_LABELS:
            raise SystemExit(f"error: label encoder has {len(ind2lab)} labels, "
                             f"expected {N_LABELS}")
        self.labels = [dumper.split_label(str(ind2lab[i]))[0] for i in range(N_LABELS)]

        self._captured: list = []

        def capture(module, inputs, output):
            self._captured.append(output)

        self._handle = self.clf.mods.classifier.out.register_forward_hook(capture)

        import speechbrain

        self.versions = {
            "speechbrain": speechbrain.__version__,
            "torch": torch.__version__,
        }

    def logits(self, pcm: np.ndarray) -> np.ndarray:
        torch = self.torch
        self._captured.clear()
        wav = torch.from_numpy(pcm).unsqueeze(0)
        # wav_lens is RELATIVE to the padded batch max; batch of one and
        # unpadded means 1.0. A sample count here would silently mask the
        # wrong span.
        with torch.inference_mode():
            self.clf.classify_batch(wav, torch.ones(1))
        if len(self._captured) != 1:
            raise SystemExit(f"error: classifier.out fired {len(self._captured)} "
                             f"times for one clip, expected 1")
        t = self._captured[0]
        if t.numel() != N_LABELS:
            raise SystemExit(f"error: classifier.out produced {t.numel()} values, "
                             f"expected {N_LABELS}")
        return np.ascontiguousarray(
            t.detach().to(dtype=torch.float32, device="cpu").reshape(-1).numpy(),
            dtype=np.float32)

    def close(self) -> None:
        self._handle.remove()


# ---------------------------------------------------------------------------
# C++ engine
# ---------------------------------------------------------------------------


class CppEngine:
    """transcribe.cpp through the Python binding (bindings/python): the GGUF
    loads once and every clip is one ``LangIdSession.run`` with ``top_k`` 0,
    so all 107 logits come back and are reassembled by label index."""

    logit_kind = "pre_softmax"

    def __init__(self, model: Path, backend: str, threads: int, max_audio_ms: int,
                 library: Path | None) -> None:
        import hashlib
        import os

        if library is not None:
            os.environ["TRANSCRIBE_LIBRARY"] = str(library.resolve())
        binding = REPO_ROOT / "bindings" / "python" / "src"
        if str(binding) not in sys.path:
            sys.path.insert(0, str(binding))
        import transcribe_cpp

        if not model.exists():
            raise SystemExit(f"error: {model} does not exist")
        self.model = transcribe_cpp.Model(str(model), backend=backend)
        self.session = self.model.langid_session(n_threads=threads, max_audio_ms=max_audio_ms)
        self.labels = [code for code, _name in self.model.langid_labels]
        if len(self.labels) != N_LABELS:
            raise SystemExit(f"error: {model} has {len(self.labels)} labels, expected {N_LABELS}")
        h = hashlib.sha256()
        with open(model, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self.provenance = {
            "gguf_sha256": h.hexdigest(),
            "gguf_source_commit": _gguf_string(model, "general.source.commit"),
            "gguf_source_repo": _gguf_string(model, "general.name"),
            "native_version": transcribe_cpp.native_version(),
            "native_commit": transcribe_cpp.native_commit(),
            "backend_bound": self.model.backend,
        }

    def logits(self, pcm: np.ndarray) -> np.ndarray:
        result = self.session.run(pcm, top_k=0)
        logits = np.zeros(N_LABELS, dtype=np.float32)
        for c in result.candidates:
            logits[c.index] = c.logit
        return logits

    def close(self) -> None:
        self.session.close()
        self.model.close()


def _gguf_string(path: Path, key: str) -> str | None:
    """A string KV from the GGUF header, for provenance."""
    from gguf import GGUFReader

    field = GGUFReader(str(path)).fields.get(key)
    return None if field is None else str(field.contents())


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def softmax_top(logits: np.ndarray) -> tuple[int, float]:
    z = logits.astype(np.float64)
    z = z - z.max()
    e = np.exp(z)
    p = e / e.sum()
    i = int(np.argmax(p))
    return i, float(p[i])


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_crops(spec: str) -> list[str | int]:
    crops: list[str | int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "full":
            crops.append("full")
        else:
            crops.append(int(tok) if float(tok).is_integer() else float(tok))
    if not crops:
        raise SystemExit("error: --crops is empty")
    return crops


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True, choices=("speechbrain", "cpp"))
    p.add_argument("--model", required=True,
                   help="HF repo id / checkpoint dir (speechbrain) or GGUF path (cpp)")
    p.add_argument("--manifest", required=True, nargs="+", type=Path)
    p.add_argument("--crops", default="3,5,10,full")
    p.add_argument("--trim", default="none", choices=("none", "energy"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--library", type=Path, default=None,
                   help="cpp engine only; a shared libtranscribe (TRANSCRIBE_LIBRARY)")
    p.add_argument("--backend", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-utts", type=int, default=0,
                   help="cap utterances per manifest (0 = all); for smoke tests")
    p.add_argument("--max-audio-ms", type=int, default=0,
                   help="cpp engine only; 0 = derived from the longest clip")
    p.add_argument("--revision",
                   default="0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9",
                   help="speechbrain engine only; pinned HF revision")
    args = p.parse_args(argv)

    crops = parse_crops(args.crops)

    # Deterministic ordering: manifests in the order given, utterances in
    # manifest order, crops in the order given.
    manifests = [Path(m) for m in args.manifest]
    rows: list[dict] = []
    for m in manifests:
        if not m.exists():
            raise SystemExit(f"error: manifest {m} does not exist")
        mr = read_manifest(m)
        if args.max_utts > 0:
            mr = mr[: args.max_utts]
        rows.extend(mr)
    if not rows:
        raise SystemExit("error: manifests are empty")
    longest_s = max(r["duration_s"] for r in rows)
    print(f"{len(rows)} utterances from {len(manifests)} manifests, "
          f"{len(crops)} crops, trim={args.trim}", file=sys.stderr)

    max_audio_ms = args.max_audio_ms or int(math.ceil(longest_s / 10.0) * 10.0 * 1000)

    engine: SpeechBrainEngine | CppEngine
    if args.engine == "speechbrain":
        engine = SpeechBrainEngine(args.model, args.revision, args.threads)
        labels = engine.labels
        extra_recipe = {
            "revision": args.revision,
            "checkpoint_dir": str(engine.checkpoint_dir),
            "device": "cpu",
            "model_dtype": "f32",
            "batch_size": 1,
            "torch_threads": args.threads,
            **engine.versions,
        }
    else:
        engine = CppEngine(Path(args.model), args.backend, args.threads, max_audio_ms,
                           args.library)
        labels = engine.labels
        extra_recipe = {
            "backend": args.backend,
            "threads": args.threads,
            "max_audio_ms": max_audio_ms,
            **engine.provenance,
        }

    recipe = {
        "crops": [str(c) for c in crops],
        "trim": args.trim,
        "trim_rule": (f"first {TRIM_FRAME_MS} ms frame with RMS > "
                      f"{TRIM_REL_THRESHOLD:g} * max frame RMS"
                      if args.trim == "energy" else
                      "none (crops start at sample 0 of the raw clip)"),
        "crop_dir": str(crop_dir(args.trim, "<crop>").relative_to(REPO_ROOT)),
        "manifests": [str(m) for m in manifests],
        "manifest_sha256": {str(m): _sha256_file(m) for m in manifests},
        "n_utterances": len(rows),
        "sample_rate": SAMPLE_RATE,
        "crop_dtype": "pcm_s16le wav (both engines read the same files)",
        "logit_kind": engine.logit_kind,
        "longest_clip_s": longest_s,
        **extra_recipe,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    n_written = 0
    with open(args.out, "w") as f:
        pending_header = True

        for crop in crops:
            t_crop = time.time()
            print(f"[{args.trim}/{crop}] materialising {len(rows)} crops ...",
                  file=sys.stderr)
            pairs = materialise_crops(rows, crop, args.trim)

            batch_rows = []
            for i, (row, wav_path) in enumerate(pairs):
                pcm = read_wav(wav_path)
                logits = engine.logits(pcm)
                top_i, top_p = softmax_top(logits)
                batch_rows.append({
                    "id": row["id"],
                    "language": row["language"],
                    "crop_s": crop,
                    "trim": args.trim,
                    "audio_s": round(pcm.size / SAMPLE_RATE, 4),
                    "logits": [round(float(x), 6) for x in logits],
                    "top1": labels[top_i],
                    "top1_prob": round(top_p, 6),
                })
                if (i + 1) % 100 == 0 or i + 1 == len(pairs):
                    el = time.time() - t_crop
                    print(f"  [{args.trim}/{crop}] {i + 1}/{len(pairs)}  "
                          f"{el:.0f}s  ({el / (i + 1) * 1000:.0f} ms/utt)",
                          file=sys.stderr)

            if pending_header:
                header = {
                    "type": "header",
                    "engine": args.engine,
                    "model": str(args.model),
                    "recipe": recipe,
                    "labels": labels,
                    "created": _dt.datetime.now(_dt.timezone.utc)
                                  .replace(microsecond=0).isoformat(),
                }
                f.write(json.dumps(header) + "\n")
                pending_header = False
            for r in batch_rows:
                f.write(json.dumps(r) + "\n")
                n_written += 1
            f.flush()
            print(f"[{args.trim}/{crop}] done in {time.time() - t_crop:.0f}s",
                  file=sys.stderr)

    engine.close()
    dt = time.time() - t_start
    print(f"wrote {args.out} ({n_written} rows) in {dt:.0f}s "
          f"({dt / max(n_written, 1) * 1000:.0f} ms/row)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
