"""PCM input-validation battery for the FFI boundary.

The contract: 16 kHz mono float32, as a buffer-protocol object, raw bytes,
or a sequence of floats. Everything else must raise InvalidArgument BEFORE
any native call — a wrong dtype silently reinterpreted as float32 would
produce garbage transcription, not an error, so the gate lives in Python.

These drive the module-level ``_pcm_to_carray`` helper directly: it is the
single choke point ``run`` / ``run_batch`` / ``Stream.feed`` all share, and
testing it needs no model on disk.
"""

from __future__ import annotations

import array
import ctypes

import pytest

import transcribe_cpp as t
from transcribe_cpp import InvalidArgument, _pcm_to_carray


# --- accepted forms ---------------------------------------------------------


def test_float32_array_module():
    arr, n = _pcm_to_carray(array.array("f", [0.0, 0.5, -0.5]))
    assert n == 3
    assert abs(arr[1] - 0.5) < 1e-6


def test_raw_bytes_interpreted_as_float32_le():
    raw = array.array("f", [1.0, -1.0]).tobytes()
    arr, n = _pcm_to_carray(raw)
    assert n == 2
    assert arr[0] == 1.0 and arr[1] == -1.0


def test_bytearray_and_memoryview():
    raw = bytearray(array.array("f", [0.25]).tobytes())
    assert _pcm_to_carray(raw)[1] == 1
    assert _pcm_to_carray(memoryview(raw))[1] == 1


def test_sequence_of_floats():
    arr, n = _pcm_to_carray([0.0, 0.1, 0.2, 0.3])
    assert n == 4


def test_ctypes_float_array_fast_path_no_copy():
    src = (ctypes.c_float * 3)(0.0, 1.0, 2.0)
    arr, n = _pcm_to_carray(src)
    assert arr is src  # documented zero-copy fast path
    assert n == 3


# --- rejected forms ---------------------------------------------------------


@pytest.mark.parametrize("empty", [[], b"", bytearray(), array.array("f")])
def test_empty_buffers_rejected(empty):
    with pytest.raises(InvalidArgument, match="empty"):
        _pcm_to_carray(empty)


def test_odd_length_bytes_rejected():
    with pytest.raises(InvalidArgument, match="multiple of 4"):
        _pcm_to_carray(b"\x00\x00\x00\x00\x00")


def test_float64_buffer_rejected():
    with pytest.raises(InvalidArgument, match="float32"):
        _pcm_to_carray(array.array("d", [0.0, 1.0]))


def test_int16_buffer_rejected():
    with pytest.raises(InvalidArgument, match="float32"):
        _pcm_to_carray(array.array("h", [0, 1, 2]))


def test_non_contiguous_buffer_rejected():
    mv = memoryview(array.array("f", [0.0, 1.0, 2.0, 3.0]))[::2]
    with pytest.raises(InvalidArgument, match="contiguous"):
        _pcm_to_carray(mv)


def test_non_iterable_rejected():
    with pytest.raises((InvalidArgument, TypeError)):
        _pcm_to_carray(object())


# --- numpy interop (optional dependency; skipped when absent) ---------------


def test_numpy_float32_accepted():
    np = pytest.importorskip("numpy")
    arr, n = _pcm_to_carray(np.zeros(160, dtype=np.float32))
    assert n == 160


def test_numpy_float64_rejected():
    np = pytest.importorskip("numpy")
    with pytest.raises(InvalidArgument, match="float32"):
        _pcm_to_carray(np.zeros(160, dtype=np.float64))


def test_numpy_non_contiguous_rejected():
    np = pytest.importorskip("numpy")
    with pytest.raises(InvalidArgument, match="contiguous"):
        _pcm_to_carray(np.zeros((4, 160), dtype=np.float32)[:, 0])


# --- dimensionality (the silent-stereo-truncation regression) ---------------
#
# A C-contiguous (frames, channels) float32 array used to pass validation and
# transcribe shape[0] SAMPLES — two samples of garbage for a (2, N) stereo
# buffer. Multi-dimensional input is now rejected outright; the caller owns
# the downmix/flatten decision.


def test_2d_contiguous_float32_rejected():
    flat = array.array("f", range(6))
    mv = memoryview(flat.tobytes()).cast("f", (2, 3))
    assert mv.contiguous and mv.ndim == 2  # exactly the trap shape
    with pytest.raises(InvalidArgument, match="1-D"):
        _pcm_to_carray(mv)


def test_3d_contiguous_float32_rejected():
    flat = array.array("f", range(8))
    mv = memoryview(flat.tobytes()).cast("f", (2, 2, 2))
    with pytest.raises(InvalidArgument, match="1-D"):
        _pcm_to_carray(mv)


def test_2d_byte_buffer_rejected():
    # The raw-bytes path must apply the same rule: a 2-D uint8 view is not
    # "a byte stream of float32" no matter how it is shaped.
    mv = memoryview(bytes(8)).cast("B", (2, 4))
    with pytest.raises(InvalidArgument, match="1-D"):
        _pcm_to_carray(mv)


def test_numpy_stereo_rejected():
    np = pytest.importorskip("numpy")
    stereo = np.zeros((160, 2), dtype=np.float32)  # (frames, channels)
    assert stereo.flags.c_contiguous
    with pytest.raises(InvalidArgument, match="downmix"):
        _pcm_to_carray(stereo)
