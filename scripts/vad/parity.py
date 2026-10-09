#!/usr/bin/env python3
"""Compare VAD segments and probabilities with silero-vad==6.2.3 defaults.

Legacy CLI segment comparison remains supported:
    uv run --project scripts/envs/silero_vad scripts/vad/parity.py \
        --gguf models/silero-vad-v6.2/silero-vad-v6.2-F32.gguf \
        samples/diar/ami-ihm-test/*.wav

For corpus runs, load the native model once and write a publication report:
    uv run --project scripts/envs/silero_vad scripts/vad/parity.py \
        --library build-vad-review-shared/src/libtranscribe.dylib \
        --gguf models/silero-vad-v6.2/silero-vad-v6.2-F32.gguf \
        --dataset golden --split validation --language mul \
        --out reports/vad/golden.json \
        tests/golden/silero_vad/silero-vad-v6.2.manifest.json

Inputs are WAVs, WER JSONL manifests (rows with an audio path), or a golden
manifest. JSON reports require --library: the native build SHA and maximum
probability delta cannot be inferred from CLI segment output. Segmentation
parity is reference agreement, not ground-truth speech accuracy. Exit 1 on
any differing segment list (a failing report is still written).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEGMENT = re.compile(r"segment: \d+ start=(\d+) end=(\d+)")


def expand(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                if line.strip():
                    audio = Path(json.loads(line)["audio"])
                    out.append(audio if audio.is_absolute() else REPO / audio)
        elif path.name.endswith(".manifest.json"):
            manifest = json.loads(path.read_text())
            if manifest.get("schema") != "transcribe-golden-manifest-v1":
                raise ValueError(f"{path}: not a golden manifest")
            for case in manifest["cases"]:
                audio = Path(case["audio"])
                out.append(audio if audio.is_absolute() else (
                    REPO / audio if audio.suffix else REPO / "samples" / f"{audio}.wav"))
        else:
            out.append(path)
    return out


def cpp_segments(cli: Path, gguf: Path, wav: Path, threads: int) -> list[dict[str, int]]:
    result = subprocess.run([str(cli), "--backend", "cpu", "--threads", str(threads),
                             "-m", str(gguf), str(wav)],
                            capture_output=True, text=True, check=True)
    return [{"start": int(match[1]), "end": int(match[2])}
            for match in (SEGMENT.match(line.strip()) for line in result.stdout.splitlines()) if match]


class ProbabilityCapture:
    """Observe the pinned get_speech_timestamps calls without duplicating it."""

    def __init__(self, model):
        self.model = model
        self.probs: list[float] = []

    def reset_states(self):
        self.probs.clear()
        self.model.reset_states()

    def __call__(self, chunk, sample_rate):
        result = self.model(chunk, sample_rate)
        self.probs.append(result.item())
        return result


def probability_delta(reference, native) -> float:
    if len(reference) != len(native):
        raise ValueError(f"probability count differs: reference {len(reference)}, native {len(native)}")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (*reference, *native)):
        raise ValueError("invalid probability (expected finite values in [0,1])")
    return max((abs(a - b) for a, b in zip(reference, native)), default=0.0)


def path_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def measurement_metadata(variant: str, *, require_record: bool = False) -> tuple[dict, dict]:
    """The catalog owns publication source identity; golden pins legacy runs."""
    record_path = REPO / "catalog" / f"{variant}.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record.get("role") != "vad":
            raise ValueError(f"{variant}: not a VAD record")
        return record["source_artifact"], record["vad_info"]
    if require_record:
        raise ValueError(f"{variant}: publication reports require a catalog record")
    manifest = json.loads((REPO / "tests/golden/silero_vad" / f"{variant}.manifest.json").read_text())
    source = manifest["source_model"]
    package, version = source["package"].split("==")
    return {"package": package, "version": version, "filename": source["file"],
            "sha256": source["sha256"], "url": manifest["reference"]["source"]}, {
                field: manifest["frontend"][field] for field in ("sample_rate", "frame_samples")}


def validate_corpus_files(files: list[Path], dataset: str, split: str, language: str, variant: str) -> None:
    """A publication stamp covers the whole locally staged canonical suite."""
    if (dataset, split, language) == ("golden", "validation", "mul"):
        expected = expand([str(REPO / "tests/golden/silero_vad" / f"{variant}.manifest.json")])
    elif (dataset, split, language) == ("ami", "ihm-test", "en"):
        expected = expand([str(REPO / "samples/diar/ami-ihm-test.manifest.jsonl")])
    else:
        if dataset == "librispeech" and split == "test-clean" and language == "en":
            name = "librispeech-test-clean"
        elif dataset == "fleurs" and split == "test":
            name = f"fleurs-{language}"
        else:
            raise ValueError(f"unsupported parity suite: {dataset}/{split}/{language}")
        expected = expand([str(REPO / "samples/wer" / f"{name}.manifest.jsonl")])
    if not expected:
        raise ValueError(f"canonical {dataset}/{split}/{language} corpus is not staged")
    actual_set = {path.resolve() for path in files}
    expected_set = {path.resolve() for path in expected}
    if len(actual_set) != len(files) or actual_set != expected_set:
        raise ValueError(f"incomplete {dataset}/{split}/{language} corpus: expected {len(expected_set)} files, "
                         f"got {len(files)} ({len(expected_set - actual_set)} missing, "
                         f"{len(actual_set - expected_set)} extra)")


def verify_reference(silero_vad, source: dict) -> dict:
    reference = f"{source['package']}=={source['version']}"
    if source["package"] != "silero-vad" or silero_vad.__version__ != source["version"]:
        raise ValueError(f"reference must be {reference}, got silero-vad=={silero_vad.__version__}")
    relative = Path(source["filename"])
    if relative.parts[0] != "silero_vad" or ".." in relative.parts:
        raise ValueError("invalid reference artifact filename")
    model_file = Path(silero_vad.__file__).parent.joinpath(*relative.parts[1:])
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise ValueError(f"reference TorchScript SHA256 differs: {digest}")
    return dict(source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=REPO / "build/bin/transcribe-cli")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", type=Path, help="native binding; load the model once for all files")
    parser.add_argument("--out", type=Path, help="JSON reference-parity report (requires --library)")
    parser.add_argument("--dataset", default="golden")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--language", default="mul")
    parser.add_argument("--variant", default="silero-vad-v6.2")
    parser.add_argument("--profile", "--publication-profile", default="vad-publication-v1")
    args = parser.parse_args(argv)
    if args.out and not args.library:
        parser.error("--out requires --library for native build provenance and probability comparison")
    if args.threads <= 0:
        parser.error("--threads must be positive")

    import soundfile as sf
    import torch
    import silero_vad

    source, expected_info = measurement_metadata(args.variant, require_record=bool(args.out))
    source = verify_reference(silero_vad, source)
    reference = f"{source['package']}=={source['version']}"
    torch.set_num_threads(1)
    model = silero_vad.load_silero_vad()
    capture = ProbabilityCapture(model)
    files = expand(args.inputs)
    if not files:
        parser.error("no audio files in inputs")
    if len({path.resolve() for path in files}) != len(files):
        parser.error("duplicate audio files in inputs")
    if args.out:
        validate_corpus_files(files, args.dataset, args.split, args.language, args.variant)
    quant_match = re.search(r"-([A-Za-z0-9_]+)\.gguf$", args.gguf.name)
    if not quant_match:
        parser.error("--gguf filename must end in -<QUANT>.gguf")
    n_same = n_paired = n_moved = n_segments = 0
    seconds = max_delta = 0.0
    per_file = []
    engine_sha = None
    native_info = None
    with ExitStack() as stack:
        session = None
        if args.library:
            os.environ["TRANSCRIBE_LIBRARY"] = str(args.library.resolve())
            sys.path.insert(0, str(REPO / "bindings/python/src"))
            import transcribe_cpp as t

            engine_sha = t.native_commit()
            if not re.fullmatch(r"[0-9a-f]{7,40}", engine_sha):
                parser.error(f"native library does not report a build SHA: {engine_sha!r}")
            native_model = stack.enter_context(t.Model(str(args.gguf), backend="cpu"))
            info = native_model.vad_info
            native_info = {"sample_rate": info.sample_rate, "frame_samples": info.frame_samples}
            if native_info != expected_info:
                parser.error("native VAD info does not match the canonical frontend")
            session = stack.enter_context(native_model.vad_session(n_threads=args.threads))
        with torch.inference_mode():
            for wav in files:
                pcm, sr = sf.read(str(wav), dtype="float32", always_2d=False)
                if sr != 16000 or pcm.ndim != 1 or not len(pcm):
                    raise ValueError(f"{wav} must be nonempty 16 kHz mono")
                duration = len(pcm) / 16000.0
                seconds += duration
                ref = silero_vad.get_speech_timestamps(torch.from_numpy(pcm), capture)
                if session is not None:
                    result = session.run(pcm)
                    cpp = [{"start": segment.start_sample, "end": segment.end_sample}
                           for segment in result.segments]
                    delta = probability_delta(capture.probs, result.probs)
                    max_delta = max(max_delta, delta)
                else:
                    cpp = cpp_segments(args.cli, args.gguf, wav, args.threads)
                    delta = None
                identical = ref == cpp
                n_same += identical
                n_segments += len(ref)
                if not identical:
                    first = next((a, b) for a, b in zip(ref + [None], cpp + [None]) if a != b)
                    print(f"DIFF {wav.name}: reference {len(ref)} segments, c++ {len(cpp)}; "
                          f"first difference {first}", flush=True)
                if len(ref) == len(cpp):
                    n_paired += len(ref)
                    n_moved += sum(a != b for a, b in zip(ref, cpp))
                per_file.append({"audio": path_label(wav), "audio_duration_s": duration,
                                 "identical": identical, "n_segments": len(ref),
                                 "n_native_segments": len(cpp), "n_probs": len(capture.probs),
                                 "max_abs_prob_delta": delta,
                                 **({"reference_segments": ref, "native_segments": cpp}
                                    if not identical else {})})
    print(f"{reference} vs {args.gguf.name}: {len(files)} files, {seconds / 3600:.2f} h")
    print(f"  identical segment lists: {n_same}/{len(files)}")
    print(f"  paired segments: {n_paired}, with a boundary difference: {n_moved}")
    if args.library:
        print(f"  max absolute probability delta: {max_delta:.9g}")
    if args.out:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "schema": "transcribe-vad-parity-v1", "timestamp": timestamp,
            "tool": "scripts/vad/parity.py", "variant": args.variant,
            "model_path": path_label(args.gguf), "library": str(args.library.resolve()),
            "source_artifact": source, "vad_info": native_info, "inputs": args.inputs,
            "dataset": args.dataset, "split": args.split, "language": args.language,
            "quant": quant_match[1], "backend": "cpu", "reference": reference,
            "n_files": len(files), "n_identical": n_same, "n_segments": n_segments,
            "audio_duration_s": seconds, "segmentation_params": "defaults",
            "engine_sha": engine_sha, "measured_on": timestamp[:10],
            "publication_profile": args.profile, "max_abs_prob_delta": max_delta,
            "threads": args.threads, "reference_threads": 1, "files": per_file,
        }, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0 if n_same == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
