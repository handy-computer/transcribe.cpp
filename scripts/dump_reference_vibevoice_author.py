#!/usr/bin/env python3
"""
dump_reference_vibevoice_author.py - VibeVoice-ASR reference tensors from the
microsoft/VibeVoice author package (the canonical implementation; the HF repo
ships weights + config only).

VibeVoice-ASR is an audio-LLM: raw 24 kHz waveform -> two parallel causal-conv
VAE tokenizers (acoustic vae_dim=64, semantic vae_dim=128) -> SpeechConnectors
(-> 3584) -> element-wise SUM -> scattered into a Qwen2.5-7B LM at the
acoustic-token positions -> lm_head -> structured JSON (Speaker/Start/End/Content).

DETERMINISM. The stock acoustic path samples a Gaussian VAE latent
(`sample(dist_type='gaussian')`, fix_std=0.5), which is non-deterministic and
unreproducible across a C++ port. The acoustic *mean* (`mode()`) is
deterministic, and we verified the transcription content is identical to the
sampled path (the noise only perturbs post-JSON junk). So the numerical contract
is the MEAN path: this dumper patches `sample` -> mean for all contract tensors,
and additionally records a sampled-envelope (per-tensor std/range over N noisy
passes) so Stage 4 has the "what should it look like" band the published model
operates in.

Runs in-container on a >=40 GB GPU (the 9B BF16 model + unused TTS decoder /
diffusion-head weights occupy ~21.6 GB; 24 GB cards OOM). The vibevoice source
path comes from $VIBEVOICE_SRC (default models/_vendor/VibeVoice).

Usage:
    VIBEVOICE_SRC=/opt/VibeVoice \
    python scripts/dump_reference_vibevoice_author.py dump \
      --model microsoft/VibeVoice-ASR \
      --audio samples/jfk.wav \
      --out build/validate/vibevoice/vibevoice-asr/jfk

Dump points (match tests/tolerances/vibevoice.json):
    enc.input_waveform      normalized 24kHz waveform the model encodes
    enc.acoustic.mean       acoustic VAE latent mean      [T, 64]
    enc.acoustic.feat       acoustic_connector(mean)      [T, 3584]
    enc.semantic.mean       semantic VAE latent mean      [T, 128]
    enc.semantic.feat       semantic_connector(mean)      [T, 3584]
    enc.combined            acoustic.feat + semantic.feat [T, 3584]  (injected)
    dec.token_emb           LM input embeddings pre-injection   [L, 3584]
    dec.inputs_embeds       embeddings after speech scatter     [L, 3584]
    dec.block.<i>.out       selected Qwen2 layer outputs (pre-norm)
    dec.final_norm          final RMSNorm output                [L, 3584]
    dec.logits_lastpos      lm_head logits at the last prompt position [vocab]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.ref_dump import write_tensor, write_transcript  # noqa: E402

MODEL_DEFAULT = "microsoft/VibeVoice-ASR"
LM_TOKENIZER = "Qwen/Qwen2.5-7B"
HIDDEN = 3584
DUMP_LAYERS = (0, 13, 27)   # first / middle / last Qwen2 decoder layer (28 total)
ENVELOPE_N = 8              # sampled passes for the stochastic-acoustic band


def _vibevoice_src() -> str:
    src = os.environ.get("VIBEVOICE_SRC")
    if not src:
        src = str(Path(__file__).resolve().parents[1] / "models" / "_vendor" / "VibeVoice")
    if not Path(src, "vibevoice").is_dir():
        raise SystemExit(f"vibevoice source not found at {src} (set $VIBEVOICE_SRC)")
    return src


def _np(t) -> np.ndarray:
    import torch
    return t.detach().to(dtype=torch.float32, device="cpu").numpy()


def _as(mod, x):
    """Cast x to the module's parameter dtype. The reference loads the VAE
    tokenizers / connectors in fp32 while the Qwen2.5 LM is bf16; matching the
    module dtype mirrors what the model's own forward does internally."""
    return x.to(next(mod.parameters()).dtype)


def load_reference(model_id: str, device: str):
    import torch
    import transformers
    from transformers import PretrainedConfig

    sys.path.insert(0, _vibevoice_src())
    transformers.logging.set_verbosity_error()
    # transformers 4.51.3 eagerly builds `logger.info(f"Model config {config}")`;
    # this model's nested sub-configs carry torch.dtype under the new `dtype`
    # field, which to_json_string can't serialize. Make repr crash-safe.
    def _safe_repr(self):
        try:
            return f"{self.__class__.__name__} {self.to_json_string()}"
        except Exception:
            return f"{self.__class__.__name__}(<unserializable>)"
    PretrainedConfig.__repr__ = _safe_repr

    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    processor = VibeVoiceASRProcessor.from_pretrained(
        model_id, language_model_pretrained_name=LM_TOKENIZER)
    # eager attention: deterministic + hookable (no fused SDPA kernels).
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager",
        device_map=device).eval()
    return processor, model


def _patch_mean(restore=False):
    """Patch the VAE EncoderOutput.sample to return the deterministic mean.
    Returns the original callable so the caller can restore it."""
    import vibevoice.modular.modular_vibevoice_tokenizer as tok
    orig = tok.VibeVoiceTokenizerEncoderOutput.sample
    if restore:
        return orig
    def mean_sample(self, dist_type="fix"):
        return self.mean, self.std
    tok.VibeVoiceTokenizerEncoderOutput.sample = mean_sample
    return orig


def cmd_dump(args: argparse.Namespace) -> None:
    import torch
    import librosa
    sys.path.insert(0, _vibevoice_src())
    import vibevoice.modular.modular_vibevoice_tokenizer as tok

    device = args.device
    # validate.py invokes the dumper with --out .../<case>/ref and compares that
    # dir against .../<case>/cpp; write flat into whatever --out we're given.
    out_dir = Path(args.out)
    src = {"framework": "author_repo_vibevoice", "model": args.model}

    processor, model = load_reference(args.model, device)
    print(f"loaded {args.model} on {device}")

    wav, sr = librosa.load(args.audio, sr=24000, mono=True)
    print(f"audio: {len(wav)} samples @ {sr} ({len(wav)/sr:.1f}s)")
    inputs = processor(audio=[wav.astype(np.float32)], sampling_rate=24000,
                       return_tensors="pt", add_generation_prompt=True)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    speech = inputs["speech_tensors"].to(device)  # [B, samples]; dtype matched per-module below
    mask = inputs["acoustic_input_mask"].to(device)
    input_ids = inputs["input_ids"].to(device)

    orig_sample = _patch_mean()           # deterministic mean path for all contract tensors
    try:
        with torch.no_grad():
            write_tensor("enc.input_waveform", _np(speech[0]), "encoder", src, out_dir=out_dir)

            # --- encoder: acoustic + semantic VAE means -> connectors -> sum ---
            at, st = model.model.acoustic_tokenizer, model.model.semantic_tokenizer
            ac, sc = model.model.acoustic_connector, model.model.semantic_connector
            a_mean = at.encode(_as(at, speech.unsqueeze(1))).mean   # [B, T, 64]
            a_feat = ac(_as(ac, a_mean))                            # [B, T, 3584]
            s_mean = st.encode(_as(st, speech.unsqueeze(1))).mean   # [B, T, 128]
            s_feat = sc(_as(sc, s_mean))                            # [B, T, 3584]
            combined = a_feat + s_feat                              # [B, T, 3584]
            for name, t in [("enc.acoustic.mean", a_mean), ("enc.acoustic.feat", a_feat),
                            ("enc.semantic.mean", s_mean), ("enc.semantic.feat", s_feat),
                            ("enc.combined", combined)]:
                write_tensor(name, _np(t[0]), "encoder", src, out_dir=out_dir)

            # --- decoder: embed, scatter speech, run Qwen2 with hidden states ---
            token_emb = model.get_input_embeddings()(input_ids)        # [B, L, 3584]
            write_tensor("dec.token_emb", _np(token_emb[0]), "decoder", src, out_dir=out_dir)

            captured = {}
            def pre_hook(mod, a, kw):
                captured["inputs_embeds"] = kw.get("inputs_embeds")
            h = model.model.language_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
            out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            h.remove()

            if captured.get("inputs_embeds") is not None:
                write_tensor("dec.inputs_embeds", _np(captured["inputs_embeds"][0]),
                             "decoder", src, out_dir=out_dir)
            hs = out.hidden_states  # (emb, layer1, ..., layerN); index i+1 == output of layer i
            for i in DUMP_LAYERS:
                write_tensor(f"dec.block.{i}.out", _np(hs[i + 1][0]), "decoder", src, out_dir=out_dir)
            final_norm = model.model.language_model.norm(hs[-1])
            write_tensor("dec.final_norm", _np(final_norm[0]), "decoder", src, out_dir=out_dir)
            write_tensor("dec.logits_lastpos", _np(out.logits[0, -1, :]), "decoder", src, out_dir=out_dir)

        # --- sampled envelope: per-tensor std/range over N noisy acoustic passes ---
        tok.VibeVoiceTokenizerEncoderOutput.sample = orig_sample
        env = _sampled_envelope(model, inputs, speech, mask, input_ids, device)
        (out_dir / "envelope.json").write_text(json.dumps(env, indent=2) + "\n")
        print(f"  wrote {out_dir/'envelope.json'}")
    finally:
        tok.VibeVoiceTokenizerEncoderOutput.sample = orig_sample

    # --- transcript (clean stop at JSON end; greedy) ---
    transcript = _transcribe(processor, model, inputs, device)
    write_transcript(out_dir, transcript["text"], source=src, tokens=transcript["tokens"])
    (out_dir / "transcript.parsed.json").write_text(
        json.dumps(transcript["parsed"], indent=2) + "\n")
    print(f"  transcript: {transcript['parsed']}")


def _sampled_envelope(model, inputs, speech, mask, input_ids, device):
    import torch
    a_samps, logit_samps = [], []
    with torch.no_grad():
        for n in range(ENVELOPE_N):
            torch.manual_seed(1000 + n)
            if device != "cpu":
                torch.cuda.manual_seed_all(1000 + n)
            at = model.model.acoustic_tokenizer
            ao = at.encode(_as(at, speech.unsqueeze(1)))
            a_tok = ao.sample(dist_type=at.std_dist_type)[0]
            a_samps.append(_np(a_tok[0]))
            out = model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
            logit_samps.append(_np(out.logits[0, -1, :]))
    a = np.stack(a_samps); lg = np.stack(logit_samps)

    def stats(arr_std, arr_mean_axis):
        s = arr_std.std(axis=0)
        return {"std_mean": float(s.mean()), "std_p99": float(np.quantile(s, 0.99)),
                "std_max": float(s.max())}
    # how much the sampled acoustic latent deviates run-to-run, and the logit spread
    argmax_stable = len(set(int(x.argmax()) for x in logit_samps)) == 1
    return {
        "n_passes": ENVELOPE_N,
        "acoustic_latent_sample_std": stats(a, 0),
        "logits_lastpos_sample_std": stats(lg, 0),
        "logits_lastpos_argmax_stable_across_samples": argmax_stable,
        "note": "Acoustic path is sampled (gaussian, fix_std=0.5). Contract tensors use the mean; "
                "this band shows the run-to-run spread of the published stochastic path.",
    }


def _transcribe(processor, model, inputs, device):
    import torch
    orig = _patch_mean()  # transcript on the deterministic (mean) path too
    try:
        eos = processor.tokenizer.eos_token_id
        with torch.no_grad():
            torch.manual_seed(0)
            out = model.generate(**inputs, max_new_tokens=448, do_sample=False, num_beams=1,
                                  eos_token_id=eos, pad_token_id=eos,
                                  stop_strings=["}]"], tokenizer=processor.tokenizer)
    finally:
        import vibevoice.modular.modular_vibevoice_tokenizer as tok
        tok.VibeVoiceTokenizerEncoderOutput.sample = orig
    gen_ids = out[0].tolist()
    text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    # pull the JSON array the model emits after "assistant"
    parsed = None
    import re
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            parsed = None
    return {"text": text, "tokens": gen_ids, "parsed": parsed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="dump reference tensors + transcript for one audio case")
    d.add_argument("--model", default=MODEL_DEFAULT)
    d.add_argument("--audio", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--device", default="cuda")
    d.set_defaults(func=cmd_dump)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
