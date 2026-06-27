#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gguf>=0.10",
#   "numpy",
# ]
# ///
"""Unit tests for the streaming-friendly GGUF KV layout.

Locks the invariant that range-read consumers depend on: the bulk tokenizer KVs
(tokens / scores / token_type / merges, chat_template) are written *after* every
scalar KV, so a remote reader can fetch the small metadata prefix without
pulling the multi-MB tokenizer tables.

Run standalone (exit-code driven):   uv run scripts/lib/test_gguf_writer.py
Or under pytest:                      pytest scripts/lib/test_gguf_writer.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from gguf import GGUFWriter
from gguf.gguf_reader import GGUFReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo scripts/
from lib.gguf_common import (  # noqa: E402
    BULK_KV_KEYS,
    gguf_writer,
    move_bulk_metadata_last,
)


def _emit_mixed(writer) -> None:
    """Emit KVs in the natural anti-pattern order: a bulk array lands in the
    middle, with small scalar metadata both before and after it."""
    writer.add_string("general.name", "unit-test")
    writer.add_bool("stt.capability.timestamps", True)
    writer.add_string("tokenizer.ggml.model", "gpt2")        # identity scalar
    writer.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])  # BULK, mid-stream
    writer.add_array("tokenizer.ggml.merges", ["a b"])          # BULK, mid-stream
    writer.add_uint32("stt.demo.encoder.n_layers", 4)        # scalar AFTER the blob
    writer.add_array("stt.demo.suppress_tokens", [1, 2, 3])  # small array, NOT bulk


def test_move_reorders_bulk_to_end():
    w = GGUFWriter(tempfile.mktemp(suffix=".gguf"), "demo")
    _emit_mixed(w)
    moved = move_bulk_metadata_last(w)
    keys = list(w.kv_data[0].keys())
    bulk = {"tokenizer.ggml.tokens", "tokenizer.ggml.merges"}
    assert set(moved) == bulk, moved
    first_bulk = min(keys.index(k) for k in bulk)
    last_scalar = max(keys.index(k) for k in keys if k not in bulk)
    assert first_bulk > last_scalar, keys
    # canonical trailer order (tokens before merges) and small array untouched
    assert keys.index("tokenizer.ggml.tokens") < keys.index("tokenizer.ggml.merges")
    assert keys.index("stt.demo.suppress_tokens") < first_bulk, keys


def test_move_is_noop_without_bulk_keys():
    w = GGUFWriter(tempfile.mktemp(suffix=".gguf"), "demo")
    w.add_string("general.name", "x")
    w.add_uint32("stt.demo.n", 1)
    before = list(w.kv_data[0].keys())
    moved = move_bulk_metadata_last(w)
    assert moved == []
    assert list(w.kv_data[0].keys()) == before


def test_split_scalars_precede_trailer():
    """Mirrors the real write order: GGUFWriter.write_header_to_file() appends
    split.* scalars (via add_shard_kv_data) AFTER the converter's bulk arrays;
    the factory's write_kv_data_to_file hook then runs, and must push the bulk
    arrays past those split.* scalars too. This is why the hook is on
    write_kv_data_to_file, not write_header_to_file."""
    w = GGUFWriter(tempfile.mktemp(suffix=".gguf"), "demo")
    w.add_string("general.name", "x")
    w.add_array("tokenizer.ggml.tokens", ["a", "b"])   # bulk, emitted first
    w.add_uint16("split.no", 0)                         # appended after bulk
    w.add_uint16("split.count", 2)                      # (simulates add_shard_kv_data)
    move_bulk_metadata_last(w)
    keys = list(w.kv_data[0].keys())
    assert keys.index("split.no") < keys.index("tokenizer.ggml.tokens"), keys
    assert keys.index("split.count") < keys.index("tokenizer.ggml.tokens"), keys


def test_factory_writes_trailer_on_disk():
    """End-to-end: build via gguf_writer(), write a real file, read it back, and
    confirm every scalar KV precedes the bulk trailer in the actual bytes."""
    out = Path(tempfile.mkdtemp()) / "t.gguf"
    w = gguf_writer(str(out), "demo")
    _emit_mixed(w)
    w.add_tensor("enc.weight", np.zeros((2, 2), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    fields = [k for k in GGUFReader(str(out)).fields if not k.startswith("GGUF.")]
    bulk = {"tokenizer.ggml.tokens", "tokenizer.ggml.merges"}
    present = [k for k in fields if k in bulk]
    assert present, fields
    first_bulk = min(fields.index(k) for k in present)
    last_scalar = max(fields.index(k) for k in fields if k not in bulk)
    assert first_bulk > last_scalar, fields
    assert fields.index("stt.demo.encoder.n_layers") < first_bulk, fields


def test_bulk_keys_are_tokenizer_only():
    """Guard: the bulk set must not accidentally include scalar identity KVs."""
    assert "tokenizer.ggml.model" not in BULK_KV_KEYS
    assert "tokenizer.ggml.tokens" in BULK_KV_KEYS


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
