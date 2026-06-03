#!/usr/bin/env python3
"""
run_reference_parakeet_streaming_nemo.py — NeMo Parakeet streaming
batch transcribe for WER on cache-aware streaming variants.

Mirrors run_reference_parakeet_nemo.py (which uses offline
model.transcribe()) but drives NeMo's conformer_stream_step per chunk
via CacheAwareStreamingAudioBuffer — the same flow as
examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py.

Usage (from repo root):

    uv run --project scripts/envs/parakeet \\
      scripts/wer/run_reference_parakeet_streaming_nemo.py \\
        --model nvidia/nemotron-speech-streaming-en-0.6b \\
        --manifest samples/wer/test-clean.512.manifest.jsonl \\
        --out reports/wer/nemotron-streaming-REF-R13.test-clean-512.jsonl \\
        --att-context-right 13

Selects one (left, right) entry from the model's training menu via
set_default_att_context_size; nemotron accepts right ∈ {0, 1, 6, 13}
(lookahead 0 / 80 / 480 / 1040 ms). The model's default (first entry,
typically right=13) is used when --att-context-right is omitted.

Output JSONL is run.py-compatible so scripts/wer/score.py scores it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="Input manifest JSONL (id/audio/ref_text).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (run.py-compatible).")
    p.add_argument("--model", required=True,
                   help="HF repo id or local .nemo path for the ASRModel checkpoint.")
    p.add_argument("--att-context-right", type=int, default=None,
                   help="Right-context (lookahead) value. nemotron accepts "
                        "{0, 1, 6, 13}. Default: model's first training entry.")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch intra-op threads for deterministic runs.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N utterances (0 = all).")
    p.add_argument("--pad-and-drop-preencoded", action="store_true",
                   help="Treat the first chunk like subsequent (ONNX-export "
                        "first-chunk semantics).")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)

    # Reuse the offline _patch_conformer hack to drop None-typed kwargs
    # that NeMo 2.7.x's ConformerEncoder.__init__ rejects. Do NOT force
    # regular att style — streaming models must run with chunked_limited.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dump_reference_parakeet_nemo import _patch_conformer_for_offline
    _patch_conformer_for_offline(force_regular_att_style=False)

    print(f"loading: {args.model}")
    t0 = time.monotonic()
    from nemo.collections.asr.models import ASRModel
    try:
        model = ASRModel.from_pretrained(args.model, map_location="cpu")
    except TypeError as e:
        if "abstract class" not in str(e):
            raise
        from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
        from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
        last = e
        for cls in (EncDecRNNTBPEModel, EncDecCTCModelBPE):
            try:
                model = cls.from_pretrained(args.model, map_location="cpu")
                break
            except Exception as e2:
                last = e2
        else:
            raise last
    model.eval()
    load_ms = (time.monotonic() - t0) * 1000

    if model.encoder.att_context_style != "chunked_limited":
        print(f"error: model att_context_style={model.encoder.att_context_style!r}; "
              f"streaming-WER runner only supports chunked_limited.", file=sys.stderr)
        return 2

    # Select R from the training menu via set_default_att_context_size
    # (also re-runs setup_streaming_params internally to update
    # streaming_cfg.{chunk_size, shift_size, pre_encode_cache_size,
    # drop_extra_pre_encoded, valid_out_len} for the new lookahead).
    if args.att_context_right is not None:
        left = model.encoder.att_context_size[0]
        target = [left, int(args.att_context_right)]
        if not hasattr(model.encoder, "set_default_att_context_size"):
            print("error: model does not support multiple lookaheads", file=sys.stderr)
            return 2
        try:
            model.encoder.set_default_att_context_size(att_context_size=target)
        except Exception as e:
            print(f"error: set_default_att_context_size({target}) failed: {e}",
                  file=sys.stderr)
            return 2

    cfg = model.encoder.streaming_cfg
    chunk_size = cfg.chunk_size[1] if isinstance(cfg.chunk_size, list) else cfg.chunk_size
    print(f"streaming: att_context_size={list(model.encoder.att_context_size)} "
          f"chunk_size={chunk_size} "
          f"drop_extra_pre_encoded={cfg.drop_extra_pre_encoded}")

    # Stub validation_ds (some checkpoints lack it).
    if getattr(model.cfg, "validation_ds", None) is None:
        from omegaconf import OmegaConf
        model.cfg.validation_ds = OmegaConf.create({"use_start_end_token": False})

    # Lazy imports for the streaming buffer + helper.
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    pad_and_drop = bool(args.pad_and_drop_preencoded)

    def calc_drop_extra(step_num: int) -> int:
        if step_num == 0 and not pad_and_drop:
            return 0
        return cfg.drop_extra_pre_encoded

    def transcribe_one_streaming(audio_path: str) -> str:
        """Run one utterance through NeMo's streaming inference."""
        buf = CacheAwareStreamingAudioBuffer(
            model=model,
            online_normalization=False,
            pad_and_drop_preencoded=pad_and_drop,
        )
        buf.append_audio_file(audio_path, stream_id=-1)

        cache_lc, cache_lt, channel_len = model.encoder.get_initial_cache_state(
            batch_size=1
        )
        pred_out_stream = None
        previous_hypotheses = None
        transcribed_texts = None
        with torch.inference_mode():
            for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(buf)):
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_lc,
                    cache_lt,
                    channel_len,
                    previous_hypotheses,
                ) = model.conformer_stream_step(
                    processed_signal=chunk_audio.to(torch.float32),
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_lc,
                    cache_last_time=cache_lt,
                    cache_last_channel_len=channel_len,
                    keep_all_outputs=buf.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=calc_drop_extra(step_num),
                    return_transcription=True,
                )
        # extract_transcriptions mirror — last call's output is the
        # full cumulative transcript.
        if isinstance(transcribed_texts, (list, tuple)) and transcribed_texts:
            first = transcribed_texts[0]
            if isinstance(first, (list, tuple)):
                first = first[0]
            if hasattr(first, "text"):
                return first.text
            if isinstance(first, str):
                return first
            return str(first)
        return ""

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[: args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")

    n_done = 0
    n_errors = 0
    t_loop = time.monotonic()

    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header",
            "load_ms": round(load_ms, 1),
            "framework": "nemo",
            "model": args.model,
            "streaming": {
                "att_context_size": [
                    int(model.encoder.att_context_size[0]),
                    int(model.encoder.att_context_size[1]),
                ],
                "chunk_size": int(chunk_size),
                "drop_extra_pre_encoded": int(cfg.drop_extra_pre_encoded),
                "pad_and_drop_preencoded": pad_and_drop,
            },
        }) + "\n")
        fout.flush()

        for entry in manifest:
            uid = entry["id"]
            audio_path = entry["audio"]
            ref_text = entry.get("ref_text", "")

            t_start = time.monotonic()
            err = ""
            hyp_text = ""
            try:
                hyp_text = transcribe_one_streaming(audio_path)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_errors += 1

            elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
            rec = {
                "id": uid,
                "ref_text": ref_text,
                "hyp_text": hyp_text.strip(),
                "raw_text": hyp_text,
                "mel_ms": 0,
                "encode_ms": 0,
                "decode_ms": elapsed_ms,
                "latency_ms": elapsed_ms,
                "error": err,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1

            if n_done % 25 == 0 or n_done == total:
                wall = time.monotonic() - t_loop
                rate = n_done / wall if wall > 0 else 0
                eta = (total - n_done) / rate if rate > 0 else 0
                print(
                    f"  [{n_done}/{total}] {rate:.2f} utt/s, "
                    f"ETA {eta/60:.1f} min, errors={n_errors}",
                    flush=True,
                )

    wall = time.monotonic() - t_loop
    print(
        f"\ndone. {n_done} utterances in {wall:.1f}s "
        f"({n_done / wall:.2f} utt/s), {n_errors} errors"
    )
    print(f"report: {args.out}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
