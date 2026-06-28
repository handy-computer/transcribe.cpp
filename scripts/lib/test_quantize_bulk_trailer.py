#!/usr/bin/env python3
"""Regression test: tools/transcribe-quantize keeps general.file_type (and all
small scalar KVs) ahead of the multi-MB tokenizer trailer.

The C++ quantizer copies every input KV (gguf_set_kv) then overrides
general.file_type. Because every ggml gguf setter is remove-then-append, that
override lands file_type at the very end — behind the tokenizer tables — unless
the quantizer re-asserts the bulk-KV trailer (move_bulk_kv_last). This builds a
reference GGUF with the converter's real trailer layout (via gguf_writer()),
runs the built binary, and asserts the fix holds and the bulk values survive.

Exit-code driven:  uv run --project scripts/envs/moonshine \
                     scripts/lib/test_quantize_bulk_trailer.py [path-to-binary]
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType, LlamaFileType
from gguf.quants import dequantize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.gguf_common import (  # noqa: E402
    BULK_KV_KEYS,
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    add_general_identity,
    gguf_writer,
)

# Small fake vocab — exercises every bulk key type: tokens/merges (string
# arrays), token_type (int32 array), scores (float32 array).
TOKENS = ["<unk>", "<s>", "</s>"] + [f"tok{i}" for i in range(29)]
TYPES = [TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL] + \
        [TOKEN_TYPE_NORMAL] * 29
SCORES = [float(-i) for i in range(len(TOKENS))]
MERGES = ["t o", "to k", "tok 0"]

# One quantizable linear (n_per_row=64, a clean multiple of the Q8_0 block) plus
# a norm + bias that stay F32 — mirrors the real per-tensor policy buckets.
FC1 = np.arange(8 * 64, dtype=np.float32).reshape(8, 64) * 0.01 - 2.0
NORM = np.linspace(0.5, 1.5, 64, dtype=np.float32)
BIAS = np.linspace(-0.1, 0.1, 64, dtype=np.float32)


def build_reference(path: Path) -> None:
    w = gguf_writer(str(path), "moonshine")
    add_general_identity(
        w,
        name="Synthetic Moonshine",
        basename="moonshine",
        size_label="0M",
        file_type=int(LlamaFileType.ALL_F32),
        languages=["en"],
    )
    w.add_string("stt.variant", "synthetic")
    w.add_bool("stt.capability.timestamps", False)

    w.add_string("tokenizer.ggml.model", "bpe")
    w.add_string("tokenizer.ggml.pre", "default")
    w.add_array("tokenizer.ggml.tokens", TOKENS)
    w.add_array("tokenizer.ggml.token_type", TYPES)
    w.add_array("tokenizer.ggml.scores", SCORES)
    w.add_array("tokenizer.ggml.merges", MERGES)
    w.add_uint32("tokenizer.ggml.bos_token_id", 1)

    w.add_tensor("dec.blocks.0.ffn.fc1.weight", FC1, raw_dtype=GGMLQuantizationType.F32)
    w.add_tensor("dec.blocks.0.norm_self.weight", NORM, raw_dtype=GGMLQuantizationType.F32)
    w.add_tensor("dec.blocks.0.ffn.fc1.bias", BIAS, raw_dtype=GGMLQuantizationType.F32)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def kv_keys(path: Path) -> list[str]:
    return list(GGUFReader(str(path)).fields.keys())


def check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL  {msg}")
        raise SystemExit(1)
    print(f"ok    {msg}")


def main(argv: list[str]) -> int:
    binary = Path(argv[1]) if len(argv) > 1 else \
        Path(__file__).resolve().parents[2] / "bin" / "transcribe-quantize"
    if not binary.is_file():
        print(f"FAIL  quantizer binary not found: {binary}")
        return 1

    # Honour TRANSCRIBE_TMPDIR; otherwise reuse the sandbox tmp/ next to the
    # repo when present, else the system default. Keeps local runs inside the
    # sandbox without pinning a machine-specific path into the test.
    tmp_root = os.environ.get("TRANSCRIBE_TMPDIR")
    if not tmp_root:
        sandbox_tmp = Path(__file__).resolve().parents[3] / "tmp"
        tmp_root = str(sandbox_tmp) if sandbox_tmp.is_dir() else None

    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        ref = Path(td) / "ref-F32.gguf"
        out = Path(td) / "out-Q8_0.gguf"
        build_reference(ref)

        # Precondition: the reference really does carry the trailer layout.
        rk = kv_keys(ref)
        bulk_in_ref = [k for k in BULK_KV_KEYS if k in rk]
        first_bulk = min(rk.index(k) for k in bulk_in_ref)
        check(rk.index("general.file_type") < first_bulk,
              "reference: file_type precedes the tokenizer trailer")
        check(rk.index("tokenizer.ggml.tokens") >= len(rk) - len(bulk_in_ref),
              "reference: bulk keys are the trailer")

        r = subprocess.run(
            [str(binary), str(ref), str(out), "--quant", "Q8_0"],
            capture_output=True, text=True,
        )
        print(r.stdout.strip())
        check(r.returncode == 0, f"quantizer exit 0 (got {r.returncode}); stderr={r.stderr.strip()}")
        check(out.is_file(), "quantizer produced an output file")

        # --- The fix: file_type stays ahead of the tokenizer trailer ---
        ok = kv_keys(out)
        check("general.file_type" in ok, "output carries general.file_type")
        present_bulk = [k for k in BULK_KV_KEYS if k in ok]
        first_bulk_out = min(ok.index(k) for k in present_bulk)
        check(ok.index("general.file_type") < first_bulk_out,
              "output: file_type precedes the tokenizer trailer (THE FIX)")

        # Bulk keys are the final KVs, in canonical BULK_KV_KEYS order.
        tail = ok[-len(present_bulk):]
        check(tail == present_bulk,
              f"output: bulk keys are last, in canonical order (got {tail})")

        # --- Values survived the move-then-reappend round trip ---
        rd = GGUFReader(str(out))

        def arr_str(key):
            f = rd.fields[key]
            return [bytes(f.parts[i]).decode("utf-8") for i in f.data]

        def arr_num(key):
            f = rd.fields[key]
            return [f.parts[i][0] for i in f.data]

        check(arr_str("tokenizer.ggml.tokens") == TOKENS, "tokens intact")
        check(arr_str("tokenizer.ggml.merges") == MERGES, "merges intact")
        check([int(x) for x in arr_num("tokenizer.ggml.token_type")] == TYPES,
              "token_type intact")
        got_scores = [float(x) for x in arr_num("tokenizer.ggml.scores")]
        check(np.allclose(got_scores, SCORES), "scores intact")

        ft = int(rd.fields["general.file_type"].parts[rd.fields["general.file_type"].data[0]][0])
        check(ft == int(LlamaFileType.MOSTLY_Q8_0),
              f"file_type updated to MOSTLY_Q8_0 ({int(LlamaFileType.MOSTLY_Q8_0)}), got {ft}")

        # --- Tensor actually requantized and round-trips within Q8_0 error ---
        fc1 = next(t for t in rd.tensors if t.name == "dec.blocks.0.ffn.fc1.weight")
        check(fc1.tensor_type == GGMLQuantizationType.Q8_0,
              f"fc1 is Q8_0 (got {fc1.tensor_type})")
        # GGUFReader hands back the raw Q8_0 block bytes; dequantize to compare.
        deq = dequantize(fc1.data, GGMLQuantizationType.Q8_0).reshape(FC1.shape)
        check(np.max(np.abs(deq - FC1)) < 0.05,
              "fc1 dequantizes back within Q8_0 tolerance")

    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
