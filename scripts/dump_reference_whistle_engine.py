#!/usr/bin/env python3
"""Whistle reference dumper over the closed Cactus Needle engine.

The engine is a black-box oracle (intake reference_framework
author_repo_cactus_needle_engine): it exposes only the encoder output and the
final decode result, so this dumper writes exactly those.

Subcommands (validate.py `ref` runs both per case):
  encoder   enc.final   [frames, 512] needle_embed output, one row per 80 ms
            plus intermediates from the numpy reference model
            (scripts/lib/whistle_ref.py): enc.mel.in, enc.stem.out,
            enc.blk.<i>.out (lane mean after each encoder layer)
  decode    transcript.json (text) + engine_result.json (text, language,
            words with start/end/probability, n_tokens) from
            needle_transcribe with word timestamps on, plus reference-model
            decoder logits: dec.logits_raw (after <s><|lang|>) and
            dec.logits_raw.gen8 (teacher-forced after 8 text tokens of the
            reference model's greedy decode)

The engine exposes nothing between its input and enc.final / the text, so the
intermediate gate tensors come from the numpy reference model, which was
matched to the engine by probing (user decision 2026-10-09; see
reports/porting/whistle/forward-map.md).

The engine decodes with its compiled default search (5 beams per the model
card; the C API has no width knob). Output layout per scripts/lib/ref_dump.py.

Usage:
    uv run --project scripts/envs/whistle \\
      scripts/dump_reference_whistle_engine.py encoder \\
        --model Cactus-Compute/whistle --audio samples/jfk.wav \\
        --out build/validate/whistle/whistle/jfk/ref
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib import whistle_engine as we  # noqa: E402
from scripts.lib.ref_dump import write_tensor, write_transcript  # noqa: E402

import numpy as np  # noqa: E402

LANG_IDS = {"en": 8192, "de": 8193, "fr": 8194, "es": 8195, "it": 8196, "nl": 8197, "pl": 8198}


def _ref_model(engine: we.Engine):
    from scripts.lib import whistle_ref as wr

    return wr, wr.Weights(engine.cact)


def _ref_source(engine: we.Engine, what: str) -> dict:
    src = we.provenance()
    src.update({"framework": "whistle_ref_numpy", "hook": what, "cact": str(engine.cact),
                "notes": "numpy reference model matched to the engine by probing; float64"})
    return src


def _engine(args) -> we.Engine:
    cact = None
    model = Path(args.model)
    if model.suffix == ".cact" and model.exists():
        cact = model
    elif args.model != we.WEIGHTS_REPO:
        raise SystemExit(f"error: --model must be {we.WEIGHTS_REPO} or a .cact path, got {args.model}")
    if args.revision and args.revision != we.WEIGHTS_REVISION:
        raise SystemExit(f"error: weights are pinned to {we.WEIGHTS_REVISION}; got --revision {args.revision}")
    return we.Engine(cact=cact)


def _source(engine: we.Engine, hook: str, **extra) -> dict:
    src = we.provenance()
    src.update({"hook": hook, "cact": str(engine.cact), "engine_lib": str(engine.lib_path)})
    src.update(extra)
    return src


def cmd_encoder(args) -> int:
    engine = _engine(args)
    pcm = we.load_audio(args.audio)
    flat = engine.embed(pcm)
    width = 512
    if flat.size % width:
        raise SystemExit(f"error: needle_embed returned {flat.size} floats, not a multiple of {width}")
    enc = flat.reshape(-1, width)
    write_tensor(
        "enc.final", enc, "encoder",
        _source(engine, "needle_embed",
                notes="encoder output per 80 ms frame as exposed by the engine; whether this is "
                      "post encoder/final_norm (the cross-attention input) is not documented",
                n_samples=int(pcm.size)),
        out_dir=args.out,
    )
    print(f"enc.final {list(enc.shape)} -> {args.out}")

    wr, W = _ref_model(engine)
    m = wr.mel(pcm, W("frontend.mel_filterbank"))
    write_tensor("enc.mel.in", m.astype(np.float32), "encoder", _ref_source(engine, "mel"), out_dir=args.out)
    x0 = wr.stem(m, W)
    write_tensor("enc.stem.out", x0.astype(np.float32), "encoder", _ref_source(engine, "stem"), out_dir=args.out)
    blocks: list = []
    wr.encoder(x0, W, blocks)
    for i, b in enumerate(blocks):
        write_tensor(f"enc.blk.{i}.out", b.astype(np.float32), "encoder",
                     _ref_source(engine, f"encoder layer {i} lane mean"), out_dir=args.out)
    return 0


def cmd_decode(args) -> int:
    engine = _engine(args)
    pcm = we.load_audio(args.audio)
    res = engine.transcribe(pcm, language=args.language, word_timestamps=True)
    src = _source(engine, "needle_transcribe", language_arg=args.language,
                  detected_language=res.get("language", ""), search="engine default (5-beam per model card)")
    write_transcript(args.out, res.get("text", ""), source=src)
    payload = {k: v for k, v in res.items() if k not in ("ttft_ms", "decode_tps")}
    payload["language_arg"] = args.language
    (Path(args.out) / "engine_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[{res.get('language', '')}] {res.get('text', '')}")

    if not res.get("text"):
        return 0  # no-speech: the C++ path never reaches the decoder either
    wr, W = _ref_model(engine)
    enc = wr.encoder(wr.stem(wr.mel(pcm, W("frontend.mel_filterbank")), W), W)
    mem = wr.memory(enc, W)
    if args.language:
        lang = LANG_IDS[args.language]
    else:
        lg = wr.decoder_logits([2], mem, W)[-1]
        lang = 8192 + int(np.argmax(lg[8192:8199]))
    prompt = [2, lang]
    write_tensor("dec.logits_raw", wr.decoder_logits(prompt, mem, W)[-1].astype(np.float32), "decoder",
                 _ref_source(engine, "logits after <s><|lang|>"), out_dir=args.out)
    toks = wr.greedy(mem, W, prompt, max_new=8)
    if len(toks) == 8:
        write_tensor("dec.logits_raw.gen8", wr.decoder_logits(prompt + toks, mem, W)[-1].astype(np.float32),
                     "decoder", _ref_source(engine, f"teacher-forced logits after 8 greedy tokens {toks}"),
                     out_dir=args.out)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("encoder", cmd_encoder), ("decode", cmd_decode)):
        s = sub.add_parser(name)
        s.add_argument("--model", default=we.WEIGHTS_REPO, help="HF repo id (pinned) or a whistle.cact path")
        s.add_argument("--revision", default=None, help="must equal the pinned weights revision if given")
        s.add_argument("--audio", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--language", default=None, help="en de fr es it nl pl; omit to detect")
        s.add_argument("--torch-threads", type=int, default=0, help="accepted for validate.py; unused")
        s.set_defaults(fn=fn)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
