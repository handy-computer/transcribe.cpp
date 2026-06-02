#!/usr/bin/env python3
"""
run_reference_parakeet_buffered_streaming_nemo.py — NeMo buffered RNN-T
streaming batch transcribe for WER on parakeet-unified-en-0.6b.

Mirrors `scripts/wer/run_reference_parakeet_streaming_nemo.py` (which
drives cache-aware streaming) but runs NeMo's BUFFERED streaming
algorithm — the one from
examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py
(lines 397-447). For each utterance the encoder is re-run over a
sliding [left | chunk | right] PCM window with chunked_limited_with_rc
attention; the RNN-T decoder state is carried across chunks.

Usage:
    uv run --project scripts/envs/parakeet \\
      scripts/wer/run_reference_parakeet_buffered_streaming_nemo.py \\
        --model /path/to/parakeet-unified-en-0.6b.nemo \\
        --manifest samples/wer/test-clean.512.manifest.jsonl \\
        --out reports/wer/parakeet-unified-buffered-REF.test-clean-512.jsonl \\
        [--left-secs 5.6 --chunk-secs 1.04 --right-secs 1.04]

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
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True,
                   help="HF repo id or local .nemo path.")
    p.add_argument("--left-secs",  type=float, default=5.6)
    p.add_argument("--chunk-secs", type=float, default=1.04)
    p.add_argument("--right-secs", type=float, default=1.04)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"),
                   help="Compute device. mps = Apple Silicon GPU; gives "
                        "~5x speedup over cpu on parakeet-unified buffered "
                        "streaming. Defaults to cpu for reproducibility.")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)

    # Reuse the conformer-kwarg drop hack from the offline dumper.
    # force_regular_att_style=False — unified-en must keep its native
    # chunked_limited_with_rc style.
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
        last = e
        try:
            model = EncDecRNNTBPEModel.from_pretrained(args.model, map_location="cpu")
        except Exception as e2:
            last = e2
            local = Path(args.model)
            if local.exists():
                model = EncDecRNNTBPEModel.restore_from(str(local), map_location="cpu")
            else:
                raise last
    except Exception:
        local = Path(args.model)
        if local.exists():
            from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
            model = EncDecRNNTBPEModel.restore_from(str(local), map_location="cpu")
        else:
            raise
    model.eval()
    if args.device != "cpu":
        model = model.to(args.device)
    load_ms = (time.monotonic() - t0) * 1000

    if model.encoder.att_context_style != "chunked_limited_with_rc":
        print(f"error: model att_context_style={model.encoder.att_context_style!r}; "
              f"buffered-WER runner requires chunked_limited_with_rc.", file=sys.stderr)
        return 2

    # Mirror the reference inference script (deterministic preprocessor).
    try:
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
    except AttributeError:
        pass

    # Resolve (L, C, R) in encoder frames.
    feature_stride_sec = float(model.cfg.preprocessor["window_stride"])
    features_per_sec = 1.0 / feature_stride_sec
    encoder_subsampling_factor = int(model.encoder.subsampling_factor)
    sample_rate = int(model.cfg.preprocessor["sample_rate"])
    features_frame2audio_samples = (
        int(sample_rate * feature_stride_sec) // encoder_subsampling_factor
    ) * encoder_subsampling_factor
    encoder_frame2audio_samples = features_frame2audio_samples * encoder_subsampling_factor

    L_frames = int(args.left_secs  * features_per_sec / encoder_subsampling_factor)
    C_frames = int(args.chunk_secs * features_per_sec / encoder_subsampling_factor)
    R_frames = int(args.right_secs * features_per_sec / encoder_subsampling_factor)
    samples_left  = L_frames * encoder_frame2audio_samples
    samples_chunk = C_frames * encoder_frame2audio_samples
    samples_right = R_frames * encoder_frame2audio_samples

    model.encoder.set_default_att_context_size(
        att_context_size=[L_frames, C_frames, R_frames]
    )

    # Configure the decoding pipeline (greedy-batch label-looping).
    from omegaconf import OmegaConf, open_dict
    import copy
    decoding_cfg = copy.deepcopy(model.cfg.decoding) if hasattr(model.cfg, "decoding") else OmegaConf.create({})
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy_batch"
        if "greedy" not in decoding_cfg:
            decoding_cfg.greedy = OmegaConf.create({})
        decoding_cfg.greedy.loop_labels = True
        decoding_cfg.greedy.use_cuda_graph_decoder = False
        decoding_cfg.greedy.preserve_alignments = False
        decoding_cfg.tdt_include_token_duration = False
        decoding_cfg.fused_batch_size = -1
        if "beam" in decoding_cfg:
            decoding_cfg.beam.return_best_hypothesis = True
    model.change_decoding_strategy(decoding_cfg)
    decoding_computer = model.decoding.decoding.decoding_computer

    if getattr(model.cfg, "validation_ds", None) is None:
        model.cfg.validation_ds = OmegaConf.create({"use_start_end_token": False})

    print(f"buffered streaming: (L,C,R)=[{L_frames}, {C_frames}, {R_frames}] frames "
          f"= [{args.left_secs:.2f}, {args.chunk_secs:.2f}, {args.right_secs:.2f}]s")

    from nemo.collections.asr.parts.utils.streaming_utils import (
        ContextSize,
        StreamingBatchedAudioBuffer,
    )
    from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
    import soundfile as sf
    import numpy as np

    context_samples = ContextSize(
        left=samples_left, chunk=samples_chunk, right=samples_right,
    )

    def transcribe_one(audio_path: str) -> str:
        pcm, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        if sr != sample_rate:
            raise RuntimeError(f"sample rate mismatch: {sr} vs {sample_rate}")
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        audio_tensor = torch.tensor(pcm, dtype=torch.float32, device=args.device).unsqueeze(0)
        audio_lengths = torch.tensor([pcm.size], dtype=torch.long, device=args.device)

        buffer = StreamingBatchedAudioBuffer(
            batch_size=1,
            context_samples=context_samples,
            dtype=audio_tensor.dtype,
            device=audio_tensor.device,
        )
        rest_audio_lengths = audio_lengths.clone()
        current_batched_hyps = None
        state = None
        left_sample = 0
        right_sample = min(
            context_samples.chunk + context_samples.right,
            audio_tensor.shape[1],
        )

        with torch.inference_mode():
            while left_sample < audio_tensor.shape[1]:
                chunk_length = min(right_sample, audio_tensor.shape[1]) - left_sample
                is_last_chunk_batch = chunk_length >= rest_audio_lengths
                is_last_chunk = right_sample >= audio_tensor.shape[1]
                chunk_lengths_batch = torch.where(
                    is_last_chunk_batch,
                    rest_audio_lengths,
                    torch.full_like(rest_audio_lengths, fill_value=chunk_length),
                )
                buffer.add_audio_batch_(
                    audio_tensor[:, left_sample:right_sample],
                    audio_lengths=chunk_lengths_batch,
                    is_last_chunk=is_last_chunk,
                    is_last_chunk_batch=is_last_chunk_batch,
                )

                encoder_output, encoder_output_len = model(
                    input_signal=buffer.samples,
                    input_signal_length=buffer.context_size_batch.total(),
                )
                encoder_output = encoder_output.transpose(1, 2)
                encoder_context = buffer.context_size.subsample(
                    factor=encoder_frame2audio_samples
                )
                encoder_context_batch = buffer.context_size_batch.subsample(
                    factor=encoder_frame2audio_samples
                )
                encoder_output_chunk = encoder_output[:, encoder_context.left:]

                chunk_batched_hyps, _, state = decoding_computer(
                    x=encoder_output_chunk,
                    out_len=torch.where(
                        is_last_chunk_batch,
                        encoder_output_len - encoder_context_batch.left,
                        encoder_context_batch.chunk,
                    ),
                    prev_batched_state=state,
                    multi_biasing_ids=None,
                )
                if current_batched_hyps is None:
                    current_batched_hyps = chunk_batched_hyps
                else:
                    current_batched_hyps.merge_(chunk_batched_hyps)

                rest_audio_lengths = rest_audio_lengths - chunk_lengths_batch
                left_sample = right_sample
                right_sample = min(right_sample + context_samples.chunk,
                                   audio_tensor.shape[1])

        hyps_view = batched_hyps_to_hypotheses(
            current_batched_hyps, None, batch_size=1
        )
        if not hyps_view or hyps_view[0].y_sequence is None:
            return ""
        ids = [int(x) for x in hyps_view[0].y_sequence.tolist()]
        return model.tokenizer.ids_to_text(ids) if ids else ""

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        manifest = manifest[: args.limit]
    total = len(manifest)
    print(f"manifest: {args.manifest} ({total} utterances)")
    print(f"output:   {args.out}")

    t_loop = time.monotonic()
    n_done, n_errors = 0, 0
    with open(args.out, "w") as fout:
        fout.write(json.dumps({
            "type": "batch_header",
            "load_ms": round(load_ms, 1),
            "framework": "nemo",
            "model": args.model,
            "buffered_streaming": {
                "att_context_size": [L_frames, C_frames, R_frames],
                "left_secs":  float(args.left_secs),
                "chunk_secs": float(args.chunk_secs),
                "right_secs": float(args.right_secs),
            },
        }) + "\n")
        fout.flush()

        for entry in manifest:
            uid = str(entry.get("id") or entry.get("audio_filepath"))
            audio = entry.get("audio") or entry.get("audio_filepath")
            t0 = time.monotonic()
            try:
                text = transcribe_one(audio)
                err = None
            except Exception as e:
                text = ""
                err = repr(e)
                n_errors += 1
            t_inf = (time.monotonic() - t0) * 1000
            row = {
                "id": uid,
                "audio": audio,
                "ref_text": entry.get("ref_text") or entry.get("text") or "",
                "hyp_text": text,
                "infer_ms": round(t_inf, 1),
                "audio_ms": entry.get("duration_ms"),
            }
            if err is not None:
                row["error"] = err
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 25 == 0 or n_done == total:
                elapsed = time.monotonic() - t_loop
                print(f"  [{n_done}/{total}] {uid}  elapsed={elapsed:.1f}s  errors={n_errors}")

    elapsed = time.monotonic() - t_loop
    print(f"done. {n_done} utterances, {n_errors} errors, {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
