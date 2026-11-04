#!/usr/bin/env python3
"""
convert-indic-canary.py - convert Bodhan AI's Indic-Transcribe (Flex / Core)
NeMo checkpoints into transcribe.cpp canary GGUFs, without a NeMo install.

    python scripts/convert-indic-canary.py \
        path/to/indic_transcribe_flex.nemo out.gguf \
        --variant indic-transcribe-flex --default-mode mixed

The .nemo archive is a tar of model_config.yaml, model_weights.ckpt and the
tokenizer's SentencePiece models. This script rebuilds the three things
convert-canary.py reads from a NeMo model -- cfg, state_dict() and the
aggregate CanaryTokenizer -- from those files directly, then hands off to
convert-canary.py's convert(). The tokenizer is Bodhan's
CanaryMultilingualTokenizer: an aggregate of `spl_tokens` (prompt tokens) and
one shared `multilingual` sub-tokenizer, routed exactly like NeMo's
CanaryTokenizer, so the aggregate extraction path applies unchanged.

On top of stock canary2 it writes:
    stt.canary.special.romanized_id / noromanized_id   10-token prompt slot
    stt.canary.default_mode    native | mixed | romanized (mode used when
                               the caller leaves itn at DEFAULT)
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import re
import sys
import tarfile
import tempfile
from pathlib import Path

import sentencepiece as spm
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("convert_canary", HERE / "convert-canary.py")
cc = importlib.util.module_from_spec(_spec)
sys.modules["convert_canary"] = cc
_spec.loader.exec_module(cc)

LANGUAGES = [
    "as", "bgc", "bhb", "bho", "bn", "brx", "doi", "en", "gu", "hi", "hne", "kn", "kok",
    "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
]
LICENSE = {
    "license": "other",
    "license_name": "Indic Open Model License v1.0",
    "license_link": "https://github.com/Bodhan-AI/bodhan-model-info/blob/main/licenses/"
                    "indic-open-model-license/v1/Indic_Open_Model_License.md",
}
PROFILES = {
    "indic-transcribe-flex": {"display_name": "Indic-Transcribe-Flex", "version": "flex",
                              "repo": "bodhan-ai/indic-transcribe-flex"},
    "indic-transcribe-core": {"display_name": "Indic-Transcribe-Core", "version": "core",
                              "repo": "bodhan-ai/indic-transcribe-core"},
}
SPECIAL_RE = re.compile(r"^<\|.+\|>$")


class _SubTokenizer:
    def __init__(self, path: Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.vocab_size = self.sp.GetPieceSize()
        self.unk_id = self.sp.unk_id()


class CanaryTokenizer:
    """Just the surface convert-canary.py's _extract_aggregate() reads.
    The class name matters: extract_tokenizer() dispatches on it."""

    def __init__(self, subs: dict[str, _SubTokenizer]):
        self.tokenizers_dict = subs
        self.token_id_offset, off = {}, 0
        for name, sub in subs.items():
            self.token_id_offset[name] = off
            off += sub.vocab_size
        self.vocab_size = off
        spl = subs["spl_tokens"]
        self.special_tokens = {
            p: i for i in range(spl.vocab_size)
            if SPECIAL_RE.match(p := spl.sp.IdToPiece(i)) or p == "<pad>"
        }
        self.bos_id = self.special_tokens["<|startoftranscript|>"]
        self.eos_id = self.special_tokens["<|endoftext|>"]
        self.pad_id = self.special_tokens["<pad>"]

    def ids_to_tokens(self, ids):
        out = []
        for i in ids:
            for name, off in reversed(list(self.token_id_offset.items())):
                if i >= off:
                    out.append(self.tokenizers_dict[name].sp.IdToPiece(i - off))
                    break
        return out


class NemoArchive:
    """cfg / state_dict() / tokenizer from an extracted .nemo, NeMo-free."""

    def __init__(self, nemo_path: Path, workdir: Path):
        print(f"Extracting {nemo_path} ...")
        with tarfile.open(nemo_path) as tar:
            tar.extractall(workdir, filter="data")
        files = {p.name: p for p in workdir.rglob("*") if p.is_file()}
        self.cfg = OmegaConf.load(files["model_config.yaml"])
        print("Loading model_weights.ckpt ...")
        self._sd = torch.load(files["model_weights.ckpt"], map_location="cpu", weights_only=True)
        langs = self.cfg.tokenizer.langs
        subs = {}
        for name in langs:  # config order == NeMo's id-offset order
            ref = str(langs[name].get("model_path") or langs[name].dir)
            ref = ref.removeprefix("nemo:")
            cand = [p for n, p in files.items() if n.endswith(Path(ref).name) and n.endswith(".model")]
            if not cand:
                cand = [p for n, p in files.items() if n.endswith(".model") and name in n]
            if len(cand) != 1:
                raise RuntimeError(f"cannot locate SentencePiece model for {name!r} ({ref}); have {sorted(files)}")
            subs[name] = _SubTokenizer(cand[0])
        if list(subs)[0] != "spl_tokens":
            raise RuntimeError(f"expected spl_tokens first, got {list(subs)}")
        self.tokenizer = CanaryTokenizer(subs)

    def state_dict(self):
        return self._sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nemo", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--variant", default="indic-transcribe-flex", choices=sorted(PROFILES))
    ap.add_argument("--default-mode", default="native", choices=["native", "mixed", "romanized"])
    args = ap.parse_args()

    prof = PROFILES[args.variant]
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        archive = NemoArchive(args.nemo, Path(tmp))
        cc.load_canary_model = lambda _spec: archive

        tok = archive.tokenizer
        extra = {name: tok.special_tokens.get(name) for name in ("<|romanized|>", "<|noromanized|>")}
        if None in extra.values():
            raise RuntimeError("checkpoint has no <|romanized|>/<|noromanized|> tokens; not an Indic-Transcribe model")

        real_writer = cc.gguf_writer

        def writer_with_indic_kvs(path, arch):
            w = real_writer(path, arch)
            # Flex/Core are ASR-only and their timestamp slot is untested in
            # this port: advertise neither, whatever convert() writes later.
            real_add_bool = w.add_bool

            def add_bool(key, val):
                if key in ("stt.capability.translate", "stt.capability.timestamps"):
                    val = False
                return real_add_bool(key, val)

            w.add_bool = add_bool
            w.add_uint32("stt.canary.special.romanized_id", int(extra["<|romanized|>"]))
            w.add_uint32("stt.canary.special.noromanized_id", int(extra["<|noromanized|>"]))
            w.add_string("stt.canary.default_mode", args.default_mode)
            w.add_string("general.author", "Bodhan AI / AI4Bharat")
            w.add_string("general.organization", "bodhan-ai")
            w.add_string("general.base_model.0.name", "Canary 1B v2")
            w.add_string("general.base_model.0.repo_url", "https://huggingface.co/nvidia/canary-1b-v2")
            w.add_string("general.description",
                         f"Built with {prof['display_name']} from Bodhan AI / AI4Bharat. "
                         "Licensed under the Indic Open Model License v1.0.")
            return w

        cc.gguf_writer = writer_with_indic_kvs
        # add_general_identity hardcodes author/organization from the profile
        # call site ("NVIDIA"); drop those two so ours above stand.
        real_identity = cc.add_general_identity

        def identity(writer, **kw):
            kw.pop("author", None)
            kw.pop("organization", None)
            return real_identity(writer, **kw)

        cc.add_general_identity = identity
        profile = {"size_label": "1B", "prompt_format": "canary2", **LICENSE, **prof}
        # "<code>-Latn" (BCP-47 script subtag) requests romanized output; listing
        # them in general.languages lets transcribe.cpp's language gate accept
        # them. They get no token-id KV, so the runtime maps them to the base code.
        languages = LANGUAGES + [f"{code}-Latn" for code in LANGUAGES]
        cc.convert(str(args.nemo), args.out, args.variant, profile, languages, repo_id=prof["repo"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
