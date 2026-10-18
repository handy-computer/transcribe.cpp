"""Container format tests. Run in scripts/envs/parakeet-phonon."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.phonon import _decode_record, read_container, verify_sha256
from lib import phonon


class PhononExtractionTests(unittest.TestCase):
    def setUp(self):
        import zstandard

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.archive = Path(directory.name) / "bundle.tar.zst"
        self.files = {"model.fermion": b"test container", "config.json": b"{}"}
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for name, content in self.files.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
        self.archive.write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue()))
        for mock in (
            patch("huggingface_hub.hf_hub_download", return_value=str(self.archive)),
            patch.object(phonon, "ARCHIVE_SHA256", hashlib.sha256(self.archive.read_bytes()).hexdigest()),
            patch.object(phonon, "CONTAINER_SHA256", hashlib.sha256(self.files["model.fermion"]).hexdigest()),
        ):
            mock.start()
            self.addCleanup(mock.stop)

    def test_concurrent_first_extraction(self):
        barrier = threading.Barrier(2)
        state = threading.local()
        copy = shutil.copyfileobj

        def overlapping_copy(src, dst):
            copy(src, dst)
            if not getattr(state, "copied", False):
                state.copied = True
                barrier.wait(timeout=10)

        with patch("shutil.copyfileobj", side_effect=overlapping_copy):
            with ThreadPoolExecutor(max_workers=2) as pool:
                roots = list(pool.map(phonon.source_directory, [phonon.REPO] * 2))
        self.assertEqual(roots[0], roots[1])
        self.assertEqual({p.name: p.read_bytes() for p in roots[0].iterdir()}, self.files)

    def test_failed_copy_removes_temporary_file(self):
        def failed_copy(src, dst):
            dst.write(b"partial")
            raise OSError("injected copy failure")

        with patch("shutil.copyfileobj", side_effect=failed_copy):
            with self.assertRaisesRegex(OSError, "injected copy failure"):
                phonon.source_directory(phonon.REPO)
        root = self.archive.parent / "phonon-2-extracted"
        self.assertEqual(list(root.iterdir()), [])


class PhononContainerTests(unittest.TestCase):
    def write(self, directory, records, blobs):
        header = json.dumps({"format": "fermion-five-value-parakeet-v1", "index": records}).encode()
        path = Path(directory) / "model.fermion"
        path.write_bytes(len(header).to_bytes(8, "little") + header + b"".join(blobs))
        return path

    def test_five_values_and_partial_rows(self):
        # Independent encoder: signs padded per row; high bits are packed
        # over nonzero values across rows, least significant bit first.
        codes = np.array([[0, 1, 2, 0, 2, 1, 2], [2, 0, 1, 2, 0, 2, 1]], dtype=np.uint8)
        high = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.uint8)
        signs = bytearray()
        for row in codes:
            padded = np.pad(row, (0, 3), constant_values=1)
            for block in padded.reshape(-1, 5):
                signs.append(sum(int(value) * 3**i for i, value in enumerate(block)))
        lo = np.array([0.25, 0.5], dtype="<f2")
        hi = np.array([1.0, 2.0], dtype="<f2")
        blob = bytes(signs) + np.packbits(high, bitorder="little").tobytes() + lo.tobytes() + hi.tobytes()
        actual = _decode_record("five_value", (2, 7), blob)
        expected = np.array([[-.25, 0, 1, -1, .25, 0, 1], [.5, -2, 0, .5, -2, 2, 0]], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_int6_signed_extremes_and_tail(self):
        q = [-32, -1, 0, 31, 1, -2]
        padded = q + [0, 0]
        payload = bytearray()
        for start in (0, 4):
            packed = sum((value + 32) << (6 * i) for i, value in enumerate(padded[start:start + 4]))
            payload.extend(packed.to_bytes(3, "little"))
        blob = bytes(payload) + np.array([.25, .5], dtype="<f2").tobytes()
        np.testing.assert_array_equal(_decode_record("int6", (2, 3), blob),
                                      np.array([[-8, -.25, 0], [15.5, .5, -1]], np.float32))

    def test_int8(self):
        blob = np.array([-128, 127, 1, -2], np.int8).tobytes() + np.array([.5, 2], "<f2").tobytes()
        np.testing.assert_array_equal(_decode_record("int8", (2, 2), blob), [[-64, 63.5], [2, -4]])

    def test_pointwise_and_scalar_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            blobs = [np.array([1, 2], "<f2").tobytes(), np.array(np.inf, "<f2").tobytes()]
            records = [
                {"n": "encoder.layers.0.conv.pointwise_conv1.weight", "k": "fp16", "shape": [1, 2], "b": 4},
                {"n": "encoder.layers.0.conv.norm.num_batches_tracked", "k": "fp16", "shape": [], "b": 2},
            ]
            tensors = read_container(self.write(directory, records, blobs))
            self.assertEqual(tensors[records[0]["n"]].shape, (1, 2, 1))
            self.assertEqual(tensors[records[1]["n"]].item(), 0)
            self.assertEqual(tensors[records[1]["n"]].dtype, np.int64)

    def test_malformed_records(self):
        cases = [
            ("fp16", (2,), b"\0\0"),
            ("int6", (2, 3), b"\0" * 9),
            ("int8", (2,), b"\0"),
            ("five_value", (1, 1), b"\xff" + b"\0" * 5),
            ("five_value", (1, 1), b"\0" * 5),
            ("other", (1,), b""),
        ]
        for kind, shape, blob in cases:
            with self.subTest(kind=kind, blob=blob), self.assertRaises(ValueError):
                _decode_record(kind, shape, blob)

    def test_duplicate_trailing_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            record = {"n": "weight", "k": "fp16", "shape": [1], "b": 2}
            valid = np.array([1], "<f2").tobytes()
            path = self.write(directory, [record, record], [valid, valid])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_container(path)
            path = self.write(directory, [record], [valid, b"x"])
            with self.assertRaisesRegex(ValueError, "trailing"):
                read_container(path)
            path = self.write(directory, [record], [np.array([np.nan], "<f2").tobytes()])
            with self.assertRaisesRegex(ValueError, "non-finite"):
                read_container(path)

    def test_truncated_header_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad"
            for payload in (b"", (10).to_bytes(8, "little"), (2**40).to_bytes(8, "little")):
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    read_container(path)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_sha256(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
