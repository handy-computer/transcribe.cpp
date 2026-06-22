"""Model-gated device-selection tests.

These take the ``model_path`` / ``transcribe_cpp`` fixtures, which ``skip``
when the default whisper-tiny.en asset is absent (override with
``TRANSCRIBE_SMOKE_MODEL``). They pin the device-selection surface added
alongside the per-device ``index`` field: ``Model.device`` reports where the
model landed (its ``.index`` is ``None`` because it did not come from
enumeration), and an out-of-range / negative ``gpu_device`` is rejected with
``InvalidArgument``.
"""

from __future__ import annotations

import pytest

import transcribe_cpp as t


def test_model_device_matches_enumeration(transcribe_cpp, model_path):
    # The model lands on some registered device. Model.device does not come
    # from enumeration, so its .index is None; correlate it back to backends()
    # by name (and by device_id when that is reported).
    with transcribe_cpp.Model(model_path) as model:
        dev = model.device
        assert isinstance(dev, transcribe_cpp.BackendDevice)
        assert dev.index is None, "Model.device should not carry a registry index"

        devices = transcribe_cpp.backends()
        by_name = [d for d in devices if d.name == dev.name]
        assert by_name, (
            f"model device {dev.name!r} not found among backends() "
            f"{[d.name for d in devices]}"
        )
        if dev.device_id is not None:
            assert any(d.device_id == dev.device_id for d in by_name), (
                f"model device_id {dev.device_id!r} matched no enumerated device"
            )


def test_negative_gpu_device_rejected(transcribe_cpp, model_path):
    with pytest.raises(transcribe_cpp.InvalidArgument):
        transcribe_cpp.Model(model_path, gpu_device=-1)


def test_out_of_range_gpu_device_rejected(transcribe_cpp, model_path):
    bad = len(transcribe_cpp.backends()) + 1000
    with pytest.raises(transcribe_cpp.InvalidArgument):
        transcribe_cpp.Model(model_path, gpu_device=bad)


def test_cpu_backend_with_gpu_index_rejected(transcribe_cpp, model_path):
    # Hardware-independent: a CPU backend has no GPU to select, so a non-zero
    # gpu_device is invalid regardless of what hardware is present.
    with pytest.raises(transcribe_cpp.InvalidArgument):
        transcribe_cpp.Model(model_path, backend="cpu", gpu_device=1)
