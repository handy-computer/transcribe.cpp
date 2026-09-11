#!/usr/bin/env python3
"""
dump_reference_multitalker_nemo.py - NeMo reference dump for the multitalker
Parakeet + streaming Sortformer pipeline.

Replicates the pinned NeMo example
(examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py
at rev 6967f48fda2a, single-audio-file mode, parallel speaker strategy) and
instruments it so the C++ port has an oracle for every hand-off:

  - per-step diar predictions (cumulative stream and per-chunk slice)
  - per-step active-speaker selection (cache gating)
  - per-step ASR supervision: feature masks (masked_asr=true) or
    spk/bg speaker-kernel targets (masked_asr=false)
  - final SegLST and per-speaker word streams
  - the effective streaming configs actually in force (sortformer module
    attrs, ASR streaming_cfg) so the C++ side mirrors reality, not defaults

All defaults follow MultitalkerTranscriptionConfig at the pin; the only
deliberate overrides are CPU placement and use_amp=False (parity runs are
fp32 CPU, per the porting convention).

Usage:
    uv run --project scripts/envs/parakeet scripts/dump_reference_multitalker_nemo.py \
      --audio samples/multitalker-2spk-mix.wav \
      --asr-model  ~/.cache/huggingface/.../multitalker-parakeet-streaming-0.6b-v1.nemo \
      --diar-model ~/.cache/huggingface/.../diar_streaming_sortformer_4spk-v2.1.nemo \
      --masked-asr true \
      --out build/validate/parakeet/multitalker-parakeet-streaming-0.6b-v1/multitalker/masked
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def build_cfg(args) -> OmegaConf:
    # Field-for-field mirror of MultitalkerTranscriptionConfig defaults at
    # the pinned rev; overrides limited to input/output paths, CPU, no AMP,
    # and the masked_asr toggle under test.
    cfg = {
        "diar_model": str(args.diar_model),
        "diar_pretrained_name": None,
        "max_num_of_spks": 4,
        "parallel_speaker_strategy": True,
        "masked_asr": args.masked_asr,
        "mask_preencode": False,
        "cache_gating": True,
        "cache_gating_buffer_size": 2,
        "single_speaker_mode": False,
        "feat_len_sec": 0.01,
        "session_len_sec": -1,
        "num_workers": 0,
        "random_seed": None,
        "log": False,
        "streaming_mode": True,
        "spkcache_len": 188,
        "spkcache_refresh_rate": 0,
        "fifo_len": 188,
        "chunk_len": 0,
        "chunk_left_context": 0,
        "chunk_right_context": 0,
        "cuda": None,
        "allow_mps": False,
        "matmul_precision": "highest",
        "asr_model": str(args.asr_model),
        "device": "cpu",
        "audio_file": str(args.audio),
        "manifest_file": None,
        "att_context_size": [70, 13],
        "use_amp": False,
        "debug_mode": False,
        "deploy_mode": False,
        "batch_size": 1,
        "chunk_size": -1,
        "shift_size": -1,
        "left_chunks": 2,
        "online_normalization": False,
        "output_path": None,
        "pad_and_drop_preencoded": False,
        "generate_realtime_scripts": False,
        "spk_supervision": "diar",
        "binary_diar_preds": False,
        "verbose": False,
        "word_window": 50,
        "sent_break_sec": 30.0,
        "fix_prev_words_count": 5,
        "update_prev_words_sentence": 5,
        "left_frame_shift": -1,
        "right_frame_shift": 0,
        "min_sigmoid_val": 1e-2,
        "discarded_frames": 8,
        "print_time": False,
        "print_sample_indices": [0],
        "colored_text": False,
        "real_time_mode": False,
        "print_path": None,
        "ignored_initial_frame_steps": 5,
        "finetune_realtime_ratio": 0.01,
    }
    return OmegaConf.create(cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--asr-model", required=True, help="path to the multitalker .nemo")
    ap.add_argument("--diar-model", required=True, help="path to the sortformer .nemo")
    ap.add_argument("--masked-asr", choices=["true", "false"], required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="cpu (parity dumps) or cuda (long-form baseline runs)")
    ap.add_argument("--dump-compress", action="store_true",
                    help="record per-call _compress_spkcache selection (compress-ref/ dumps)")
    ap.add_argument("--strict-fp32", action="store_true",
                    help="disable TF32 everywhere (cudnn convs + cuda matmul); "
                         "isolates precision when comparing against true-F32 backends")
    ap.add_argument("--light", action="store_true",
                    help="skip per-step instrumentation dumps (seglst + config echo only); "
                         "required for long meetings where per-chunk npz would be huge")
    args = ap.parse_args()
    args.masked_asr = args.masked_asr == "true"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
    from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    cfg = build_cfg(args)
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if args.strict_fp32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)

    diar_model = SortformerEncLabelModel.restore_from(restore_path=cfg.diar_model, map_location=device)
    diar_model = diar_model.eval().to(device)
    compress_calls = None
    if args.dump_compress:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parent))
        from dump_reference_sortformer_nemo import _install_compress_hooks, _write_compress_dump
        compress_calls = _install_compress_hooks(diar_model)
    diar_model.streaming_mode = cfg.streaming_mode
    diar_model.sortformer_modules.chunk_len = cfg.chunk_len
    diar_model.sortformer_modules.spkcache_len = cfg.spkcache_len
    diar_model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
    diar_model.sortformer_modules.chunk_right_context = cfg.chunk_right_context
    diar_model.sortformer_modules.fifo_len = cfg.fifo_len
    diar_model.sortformer_modules.log = cfg.log
    diar_model.sortformer_modules.spkcache_refresh_rate = cfg.spkcache_refresh_rate

    asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=cfg.asr_model, map_location=device)
    asr_model.encoder.set_default_att_context_size(att_context_size=list(cfg.att_context_size))
    asr_model = asr_model.to(device).eval()

    # ------------------------------------------------------------------ #
    # Instrumentation: record every hand-off the C++ port must reproduce #
    # ------------------------------------------------------------------ #
    steps: list[dict] = []
    instrument = not args.light

    orig_fss = diar_model.forward_streaming_step

    def rec_fss(*a, **kw):
        state, preds = orig_fss(*a, **kw)
        steps.append({"kind": "diar_preds", "step": len([s for s in steps if s["kind"] == "diar_preds"]),
                      "total_preds": preds.detach().cpu().numpy().copy()})
        return state, preds

    if instrument:
        diar_model.forward_streaming_step = rec_fss

    orig_sst = asr_model.set_speaker_targets

    def rec_sst(spk_targets=None, bg_spk_targets=None):
        steps.append({"kind": "spk_targets",
                      "spk": None if spk_targets is None else spk_targets.detach().cpu().numpy().copy(),
                      "bg": None if bg_spk_targets is None else bg_spk_targets.detach().cpu().numpy().copy()})
        return orig_sst(spk_targets, bg_spk_targets)

    if instrument:
        asr_model.set_speaker_targets = rec_sst

    streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)

    orig_find = streamer._find_active_speakers

    def rec_find(diar_preds, n_active_speakers_per_stream):
        res = orig_find(diar_preds, n_active_speakers_per_stream)
        steps.append({"kind": "active_speakers", "active": [list(map(int, r)) for r in res]})
        return res

    if instrument:
        streamer._find_active_speakers = rec_find

    orig_mask_feat = streamer.mask_features

    def rec_mask_feat(chunk_audio, mask, threshold=0.5, mask_value=-16.6355):
        steps.append({"kind": "feature_mask", "mask": mask.detach().cpu().numpy().copy()})
        return orig_mask_feat(chunk_audio, mask, threshold, mask_value)

    if instrument:
        streamer.mask_features = rec_mask_feat

    # ------------------------------------------------------------------ #
    # Streaming loop (mirrors launch_parallel_streaming at the pin)      #
    # ------------------------------------------------------------------ #
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=cfg.online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    streaming_buffer.append_audio_file(audio_filepath=cfg.audio_file, stream_id=-1)

    autocast = torch.amp.autocast("cpu", enabled=cfg.use_amp)
    for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(streaming_buffer)):
        drop_extra_pre_encoded = (
            0 if step_num == 0 and not cfg.pad_and_drop_preencoded
            else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
        )
        with torch.inference_mode(), autocast, torch.no_grad():
            streamer.perform_parallel_streaming_stt_spk(
                step_num=step_num,
                chunk_audio=chunk_audio,
                chunk_lengths=chunk_lengths,
                is_buffer_empty=streaming_buffer.is_buffer_empty(),
                drop_extra_pre_encoded=drop_extra_pre_encoded,
            )

    samples = [{"audio_filepath": cfg.audio_file}]
    seglst = streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)

    # ------------------------------------------------------------------ #
    # Dump                                                               #
    # ------------------------------------------------------------------ #
    # Accumulated diar preds are tiny and always worth keeping (light
    # mode included): they are the supervision ground truth for parity.
    diar_states = streamer.instance_manager.diar_states
    final_preds = diar_states.diar_pred_out_stream
    np.save(out_dir / "diar_total_preds.npy", final_preds.detach().cpu().numpy())
    if compress_calls is not None:
        _write_compress_dump(compress_calls, out_dir / "compress-ref")

    npz: dict[str, np.ndarray] = {}
    counters: dict[str, int] = {}
    step_index = []
    for s in steps:
        i = counters.get(s["kind"], 0)
        counters[s["kind"]] = i + 1
        if s["kind"] == "diar_preds":
            npz[f"diar_preds_{i:03d}"] = s["total_preds"]
        elif s["kind"] == "spk_targets":
            if s["spk"] is not None:
                npz[f"spk_targets_{i:03d}"] = s["spk"]
            if s["bg"] is not None:
                npz[f"bg_targets_{i:03d}"] = s["bg"]
        elif s["kind"] == "feature_mask":
            npz[f"feature_mask_{i:03d}"] = s["mask"]
        elif s["kind"] == "active_speakers":
            step_index.append(s["active"])
    np.savez(out_dir / "per_step.npz", **npz)

    scfg = asr_model.encoder.streaming_cfg
    echo = {
        "masked_asr": cfg.masked_asr,
        "binary_diar_preds": cfg.binary_diar_preds,
        "cache_gating": cfg.cache_gating,
        "cache_gating_buffer_size": cfg.cache_gating_buffer_size,
        "single_speaker_mode": cfg.single_speaker_mode,
        "att_context_size": list(cfg.att_context_size),
        "nframes_per_chunk": streamer._nframes_per_chunk,
        "sent_break_sec": streamer._sent_break_sec,
        "active_speakers_per_step": step_index,
        "sortformer_modules": {
            k: getattr(diar_model.sortformer_modules, k)
            for k in ("chunk_len", "spkcache_len", "fifo_len", "chunk_left_context",
                      "chunk_right_context", "spkcache_refresh_rate")
        },
        "asr_streaming_cfg": {
            "chunk_size": list(scfg.chunk_size) if hasattr(scfg.chunk_size, "__len__") else scfg.chunk_size,
            "shift_size": list(scfg.shift_size) if hasattr(scfg.shift_size, "__len__") else scfg.shift_size,
            "pre_encode_cache_size": list(scfg.pre_encode_cache_size)
            if hasattr(scfg.pre_encode_cache_size, "__len__") else scfg.pre_encode_cache_size,
            "drop_extra_pre_encoded": scfg.drop_extra_pre_encoded,
            "valid_out_len": scfg.valid_out_len,
        },
    }
    (out_dir / "config_echo.json").write_text(json.dumps(echo, indent=2, default=str) + "\n")
    (out_dir / "seglst.json").write_text(json.dumps(seglst, indent=2) + "\n")

    per_speaker: dict[str, str] = {}
    for seg in seglst:
        per_speaker.setdefault(seg["speaker"], [])
    for seg in sorted(seglst, key=lambda s: s["start_time"]):
        per_speaker[seg["speaker"]] = per_speaker.get(seg["speaker"], [])
        per_speaker[seg["speaker"]].append(seg["words"])
    per_speaker_text = {k: " ".join(v) for k, v in per_speaker.items()}
    (out_dir / "per_speaker_text.json").write_text(json.dumps(per_speaker_text, indent=2) + "\n")

    print(f"steps recorded: {counters}")
    print(f"seglst segments: {len(seglst)}")
    for spk, text in per_speaker_text.items():
        print(f"  {spk}: {text[:120]}")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
