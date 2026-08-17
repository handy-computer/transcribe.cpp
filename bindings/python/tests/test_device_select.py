"""Exact opaque-device selection tests."""

import pytest


def _primary_devices(transcribe_cpp):
    return [d for d in transcribe_cpp.backends() if d.device_type != "accel"]


def test_enumerated_device_can_be_passed_to_model(transcribe_cpp, model_path):
    devices = _primary_devices(transcribe_cpp)
    if not devices:
        pytest.skip("no selectable devices")

    device = devices[0]
    try:
        with transcribe_cpp.Model(model_path, device=device) as model:
            assert model.device == device
    except transcribe_cpp.BackendError:
        # Registered devices may still fail driver initialization. Exact
        # selection must report that failure rather than choosing another.
        pass


def test_device_argument_rejects_non_device(transcribe_cpp, model_path):
    with pytest.raises(TypeError):
        transcribe_cpp.Model(model_path, device=0)


def test_backend_must_match_explicit_device(transcribe_cpp, model_path):
    gpu = next(
        (d for d in transcribe_cpp.backends() if d.device_type in ("gpu", "igpu")),
        None,
    )
    if gpu is None:
        pytest.skip("no GPU device")
    with pytest.raises(transcribe_cpp.InvalidArgument):
        transcribe_cpp.Model(model_path, backend="cpu", device=gpu)
