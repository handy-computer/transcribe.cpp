#!/usr/bin/env python3
"""
run_reference_parakeet_nemo.py — NeMo Parakeet batch transcribe for WER.

Loads an ASRModel once via NeMo and runs it over a WER manifest, writing
run.py-compatible JSONL so scripts/wer/score.py can score the output.
Mirrors the dump-time setup (scripts/dump_reference_parakeet_nemo.py):
greedy decoding via the model's default decoding strategy.

Usage (from repo root):

    uv run --project scripts/envs/parakeet \\
      scripts/wer/run_reference_parakeet_nemo.py \\
        --model nvidia/parakeet-tdt_ctc-110m \\
        --manifest samples/wer/test-clean.512.manifest.jsonl \\
        --out      reports/wer/parakeet-tdt_ctc-110m-REF.test-clean-512.jsonl

Works for any of the parakeet TDT / RNNT / CTC variants. Uses
ASRModel.from_pretrained, which handles all the head kinds uniformly.
For variants whose .nemo cfg blocks NeMo instantiation (e.g.
parakeet-unified-en-0.6b's streaming-context kwargs), the load fails
loudly — same gap that blocks tensor validation for those variants.
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
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch intra-op threads for deterministic runs.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N utterances (0 = all).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="NeMo transcribe() batch size (groups N utterances).")
    p.add_argument("--device", default="cpu",
                   help="torch device: 'cpu' or 'cuda' (default cpu). "
                        "Uniform across reference runners so the Modal "
                        "reference_sweep can drive any family the same way.")
    p.add_argument("--offline-only", action="store_true",
                   help=(
                       "Force att_context_style='regular'. Required only for "
                       "parakeet-unified-en-0.6b (whose v1 port targets offline "
                       "mode). Cache-aware streaming variants like "
                       "nemotron-speech-streaming-en-0.6b must NOT pass this — "
                       "their published WER assumes chunked_limited attention."
                   ))
    p.add_argument("--language", default="en",
                   help=(
                       "Target language tag. Ignored for English-only parakeet "
                       "variants. For prompt-conditioned multilingual models "
                       "(nemotron-3.5-asr-streaming-0.6b) this must be a key in "
                       "the model's prompt_dictionary (e.g. 'en-US', 'es-ES')."
                   ))
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    # Some parakeet variants (e.g. parakeet-unified-en-0.6b) carry
    # streaming-only kwargs in their .nemo cfg that NeMo 2.7.x's
    # ConformerEncoder.__init__ rejects, and other fields set to None
    # (use_bias=None, att_chunk_size=None) that explicitly-pass-None
    # breaks. Dropping those unrecognised / None-typed kwargs is always
    # safe. The att_context_style override (--offline-only) is OPT-IN:
    # only parakeet-unified-en-0.6b needs it. Cache-aware streaming
    # models (nemotron-speech-streaming-en-0.6b and similar) must
    # preserve their native chunked_limited style to score correct WER.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dump_reference_parakeet_nemo import (
        _patch_conformer_for_offline,
        is_prompt_aware,
        resolve_prompt_id,
        apply_prompt,
    )
    _patch_conformer_for_offline(force_regular_att_style=args.offline_only)

    print(f"loading: {args.model}")
    t0 = time.monotonic()
    # Some checkpoints (parakeet-unified-en-0.6b) have a cfg where the
    # top-level _target_ is None, so `ASRModel.from_pretrained` can't
    # auto-dispatch through the abstract class. The concrete class is
    # always one of EncDecRNNTBPEModel / EncDecCTCModelBPE /
    # EncDecHybridRNNTCTCBPEModel for parakeet; iterate them as
    # fallbacks when the abstract dispatcher fails.
    from nemo.collections.asr.models import ASRModel
    local = Path(args.model).expanduser()
    is_local = local.exists()
    loader = "restore_from" if is_local else "from_pretrained"
    try:
        if is_local:
            model = ASRModel.restore_from(str(local), map_location="cpu")
        else:
            model = ASRModel.from_pretrained(args.model, map_location="cpu")
    except TypeError as e:
        if "abstract class" not in str(e):
            raise
        from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
        from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
        last = e
        for cls in (EncDecRNNTBPEModel, EncDecCTCModelBPE):
            try:
                if is_local:
                    model = cls.restore_from(str(local), map_location="cpu")
                else:
                    model = cls.from_pretrained(args.model, map_location="cpu")
                break
            except Exception as e2:
                last = e2
        else:
            raise last
    model.eval()
    if args.device != "cpu":
        model = model.to(args.device)
    load_ms = (time.monotonic() - t0) * 1000

    # Parakeet variants without a validation_ds in cfg need this stub
    # so model.transcribe() doesn't crash reading
    # validation_ds.use_start_end_token. Mirrors the dump script's
    # workaround.
    if getattr(model.cfg, "validation_ds", None) is None:
        from omegaconf import OmegaConf
        model.cfg.validation_ds = OmegaConf.create({"use_start_end_token": False})

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
        }) + "\n")
        fout.flush()

        # Batch utterances in groups of --batch-size. NeMo's transcribe()
        # takes a list of audio paths and batches them natively, returning
        # results aligned to input order. A batch that raises falls back to
        # per-utterance so one bad file can't drop the whole group.
        def _hyp_of(item) -> str:
            if hasattr(item, "text"):
                return item.text
            if isinstance(item, str):
                return item
            return str(item)

        def _texts(out, k: int) -> list[str]:
            # RNNT/CTC return a flat list of k hypotheses; hybrid models can
            # return a 1-tuple / (rnnt_hyps, ctc_hyps) tuple of lists. Unwrap
            # to the first decoder's list, then map each to text.
            if isinstance(out, tuple):
                out = out[0] if out else []
            if not isinstance(out, (list, tuple)):
                out = [out]
            return [_hyp_of(x) for x in out][:k]

        # Prompt-aware multilingual models (nemotron-3.5-asr-streaming-0.6b)
        # cannot use model.transcribe(): NeMo's temp manifest lacks per-cut
        # language tags, and the Lhotse prompt_index dataset throws on
        # prompt_key='None'. We bypass it by running preprocess + encoder +
        # prompt_kernel + decoding directly. Mirrors the dump-side bypass.
        prompt_mode = is_prompt_aware(model)
        prompt_id = resolve_prompt_id(model, args.language) if prompt_mode else None
        if prompt_mode:
            print(f"prompt mode: target_lang={args.language!r} prompt_id={prompt_id}")
        import soundfile as sf
        import numpy as np

        def _prompt_transcribe(paths):
            audios = []
            lens = []
            for p_ in paths:
                pcm, sr = sf.read(p_, dtype="float32", always_2d=False)
                if pcm.ndim > 1:
                    pcm = pcm.mean(axis=1).astype("float32")
                if sr != 16000:
                    raise RuntimeError(f"audio {p_} sample_rate={sr}, expected 16000")
                audios.append(pcm)
                lens.append(pcm.size)
            T = max(lens)
            batch = np.zeros((len(audios), T), dtype="float32")
            for i, a in enumerate(audios):
                batch[i, : a.size] = a
            audio_t = torch.tensor(batch, device=args.device)
            length_t = torch.tensor(lens, dtype=torch.long, device=args.device)
            with torch.inference_mode():
                processed, proc_len = model.preprocessor(
                    input_signal=audio_t, length=length_t
                )
                encoded, encoded_len = model.encoder(
                    audio_signal=processed, length=proc_len
                )
                encoded = apply_prompt(model, encoded, prompt_id)
                hyps = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=encoded,
                    encoded_lengths=encoded_len,
                    return_hypotheses=False,
                )
            return _texts(hyps, len(paths))

        bs = max(1, args.batch_size)
        for start in range(0, total, bs):
            group = manifest[start:start + bs]
            paths = [e["audio"] for e in group]
            k = len(group)
            t_start = time.monotonic()
            errs = [""] * k
            hyps = [""] * k
            try:
                if prompt_mode:
                    texts = _prompt_transcribe(paths)
                else:
                    texts = _texts(model.transcribe(audio=paths, batch_size=k), k)
                for i in range(k):
                    hyps[i] = texts[i] if i < len(texts) else ""
            except Exception:
                for i, e_ in enumerate(group):
                    try:
                        if prompt_mode:
                            t1 = _prompt_transcribe([e_["audio"]])
                        else:
                            t1 = _texts(model.transcribe(audio=[e_["audio"]],
                                                         batch_size=1), 1)
                        hyps[i] = t1[0] if t1 else ""
                    except Exception as e2:
                        errs[i] = f"{type(e2).__name__}: {e2}"
                        n_errors += 1
            per_ms = round((time.monotonic() - t_start) * 1000 / k, 1)

            for i, entry in enumerate(group):
                rec = {
                    "id": entry["id"],
                    "ref_text": entry.get("ref_text", ""),
                    "hyp_text": (hyps[i] or "").strip(),
                    "raw_text": hyps[i] or "",
                    "mel_ms": 0,
                    "encode_ms": 0,
                    "decode_ms": per_ms,
                    "latency_ms": per_ms,
                    "error": errs[i],
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n_done += 1

            if start // bs % 10 == 0 or n_done == total:
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
