#!/usr/bin/env python3
"""
compose-multitalker-bundle.py - merge a multitalker Parakeet ASR GGUF and a
Sortformer diarizer GGUF into one self-contained bundle GGUF.

The multitalker checkpoint CONSUMES external speaker-activity supervision at
inference time (layer-0 speaker-kernel injection); NVIDIA validated it paired
with streaming Sortformer. Shipping the pair as one file removes the
mismatched-pair support surface: one artifact, one tested behavior, and the
model-level DIARIZATION capability stays a property of the file.

Layout of the bundle:

  KV       every ASR KV verbatim (arch stays 'parakeet'), plus:
             stt.parakeet.diarizer.embedded        = true
             stt.parakeet.diarizer.variant         = <diarizer stt.variant>
             stt.parakeet.diarizer.tensor_prefix   = 'sortformer.'
             stt.sortformer.*                        copied verbatim
           The diarizer's frontend KVs are NOT copied: this pairing shares
           one mel frontend (both are 128-mel, 16 kHz, hann/512/400/160,
           normalize=none, same dither/pre-emphasis) and the compose step
           VERIFIES that agreement instead of duplicating the keys.
  tensors  every ASR tensor verbatim, then every diarizer tensor renamed
           with the 'sortformer.' prefix (the diarizer reuses Parakeet's
           NEST encoder, so its enc.* names collide with the ASR's own
           enc.* catalog without the prefix).

The C++ parakeet loader materializes and uploads every tensor in the file
and resolves its catalog by name, so a bundle loads and runs the shipped
single-speaker path unchanged even before the multitalker composition code
exists; the sortformer.* tensors ride along for it to claim.

Dtype policy mirrors the shipped matrices: the ASR half may be any shipped
quant; the diarizer half should be F32/F16/Q8_0 only (sortformer k-quants
are withdrawn). Pick the diarizer file explicitly per bundle tier.

Usage (any env with gguf, e.g. the moonshine or sortformer env):
    uv run --project scripts/envs/moonshine scripts/compose-multitalker-bundle.py \
      models/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1-F32.gguf \
      models/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1-F32.gguf \
      -o models/multitalker-parakeet-streaming-0.6b-v1/bundle/multitalker-parakeet-streaming-0.6b-v1-F32.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gguf import GGUFReader, GGUFValueType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import gguf_writer  # noqa: E402

DIAR_TENSOR_PREFIX = "sortformer."

# KVs the two halves must agree on for the bundle to share one mel frontend.
FRONTEND_AGREEMENT_KEYS = [
    "stt.frontend.sample_rate",
    "stt.frontend.num_mels",
    "stt.frontend.n_fft",
    "stt.frontend.hop_length",
    "stt.frontend.win_length",
    "stt.frontend.window",
    "stt.frontend.normalize",
    "stt.frontend.pre_emphasis",
    "stt.frontend.dither",
]

# Keys the GGUFReader synthesizes (virtual) or the writer emits itself.
SKIP_COPY_KEYS = {"general.architecture"}

# Diarizer checkpoints the runtime's multitalker path is validated against.
# run_multitalker pins the reference operating point (14-frame chunk
# cadence, spkcache/FIFO/update 188, gating threshold 0.5, 2-chunk gating
# buffer) measured on streaming Sortformer v2.1; composing an unvalidated
# variant would silently run a foreign checkpoint at v2.1 settings.
VALIDATED_DIARIZER_VARIANTS = {"diar_streaming_sortformer_4spk-v2.1"}


def copy_kv(writer, field) -> None:
    vtype = field.types[0]
    sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub_type)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asr_gguf", help="multitalker Parakeet GGUF (any shipped quant)")
    ap.add_argument("diar_gguf", help="Sortformer diarizer GGUF (F32/F16/Q8_0)")
    ap.add_argument("-o", "--out", required=True, help="output bundle path")
    args = ap.parse_args()

    asr = GGUFReader(args.asr_gguf)
    diar = GGUFReader(args.diar_gguf)

    asr_arch = asr.fields["general.architecture"].contents()
    diar_arch = diar.fields["general.architecture"].contents()
    if asr_arch != "parakeet":
        print(f"error: ASR input arch is '{asr_arch}', expected 'parakeet'", file=sys.stderr)
        return 1
    if diar_arch != "sortformer":
        print(f"error: diarizer input arch is '{diar_arch}', expected 'sortformer'", file=sys.stderr)
        return 1
    if "stt.parakeet.encoder.spk_kernel_layers" not in asr.fields:
        print("error: ASR input has no speaker-kernel tensors; not a multitalker checkpoint", file=sys.stderr)
        return 1

    diar_variant = diar.fields["stt.variant"].contents() if "stt.variant" in diar.fields else None
    if diar_variant not in VALIDATED_DIARIZER_VARIANTS:
        print(
            f"error: diarizer variant {diar_variant!r} is not validated for the multitalker bundle "
            f"(validated: {sorted(VALIDATED_DIARIZER_VARIANTS)}); the runtime pins the v2.1 operating point",
            file=sys.stderr,
        )
        return 1

    # The supervision hand-off maps diar frames 1:1 onto ASR encoder frames
    # (same mel hop x same subsampling -> same 80 ms cadence). Frontend
    # agreement below covers the hop; enforce the subsampling factor too.
    a_sub = asr.fields["stt.parakeet.encoder.subsampling_factor"].contents()
    d_sub = diar.fields["stt.sortformer.encoder.subsampling_factor"].contents()
    if a_sub != d_sub:
        print(f"error: subsampling mismatch: asr={a_sub} diar={d_sub} (frame cadences must agree)", file=sys.stderr)
        return 1

    # The bundle carries a single stt.frontend.* block (the ASR's). Refuse to
    # compose a pairing whose diarizer disagrees with it rather than silently
    # feeding the diarizer the wrong mel geometry.
    for key in FRONTEND_AGREEMENT_KEYS:
        a = asr.fields[key].contents() if key in asr.fields else None
        d = diar.fields[key].contents() if key in diar.fields else None
        if a != d:
            print(f"error: frontend mismatch on {key}: asr={a!r} diar={d!r}", file=sys.stderr)
            return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf_writer(str(out_path), "parakeet")

    for name, field in asr.fields.items():
        if name.startswith("GGUF.") or name in SKIP_COPY_KEYS:
            continue
        copy_kv(writer, field)

    writer.add_bool("stt.parakeet.diarizer.embedded", True)
    writer.add_string("stt.parakeet.diarizer.variant", diar.fields["stt.variant"].contents())
    writer.add_string("stt.parakeet.diarizer.tensor_prefix", DIAR_TENSOR_PREFIX)
    for name, field in diar.fields.items():
        if name.startswith("stt.sortformer."):
            copy_kv(writer, field)

    for t in asr.tensors:
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    for t in diar.tensors:
        writer.add_tensor(DIAR_TENSOR_PREFIX + t.name, t.data, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    n_asr, n_diar = len(asr.tensors), len(diar.tensors)
    print(f"wrote {out_path}: {n_asr} ASR + {n_diar} diarizer tensors ({n_asr + n_diar} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
