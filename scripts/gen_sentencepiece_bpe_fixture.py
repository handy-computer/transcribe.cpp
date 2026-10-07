"""Generate SentencePiece BPE id fixtures for tests/sentencepiece_bpe_parity.cpp.

Encodes a text list with the reference `sentencepiece` model and writes one
JSON line per text: {"text": ..., "ids": [...]}, ids offset into the GGUF
vocabulary (aggregate tokenizers put each language's sub-vocab at an offset).

Usage:
  uv run --project scripts/envs/canary scripts/gen_sentencepiece_bpe_fixture.py \
      --model <tokenizer.model> [--offset N] --texts <file, one per line> --out <jsonl>
"""
import argparse
import json

import sentencepiece as spm

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--texts", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

sp = spm.SentencePieceProcessor(model_file=args.model)
with open(args.texts, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
    for line in f:
        text = line.rstrip("\n")
        ids = [i + args.offset for i in sp.encode(text)]
        out.write(json.dumps({"text": text, "ids": ids}, ensure_ascii=False) + "\n")
