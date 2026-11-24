"""A target's size cap is policy, and that policy decides what is required.

    uv run --with pytest pytest scripts/catalog/test_profiles.py

The cap exists so weak reference machines (a base M4, a Ryzen 4750U host) are
not required to run models far outside their memory class, and so that policy
is declared once per target instead of as one hand-written exception per large
model.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import profiles  # noqa: E402


def record(sizes: dict[str, int]) -> dict:
    return {"downloads": [{"quant": quant, "size_bytes": size}
                          for quant, size in sizes.items()]}


CAPPED = {"machine": "m4", "backends": ["cpu"], "max_model_gib": 5}
UNCAPPED = {"machine": "m4-max", "backends": ["cpu"]}
GIB = 1 << 30

PROFILE = {
    "speed": {
        "quants": ["Q8_0"],
        "samples": ["jfk"],
        "iterations": 1,
        "warmup": 0,
        "targets": [CAPPED, UNCAPPED],
    }
}


def test_published_gib_prefers_q8():
    assert profiles.published_gib(record({"Q8_0": 5 * GIB, "Q4_K_M": 3 * GIB})) == 5.0


def test_published_gib_falls_back_to_the_largest_published_quant():
    assert profiles.published_gib(record({"F16": 4 * GIB, "Q4_K_M": 2 * GIB})) == 4.0


def test_published_gib_is_none_without_sizes():
    assert profiles.published_gib({"downloads": []}) is None


def test_a_target_without_a_cap_admits_everything():
    assert profiles.target_admits(record({"Q8_0": 100 * GIB}), UNCAPPED)


def test_the_cap_is_inclusive_at_the_boundary():
    assert profiles.target_admits(record({"Q8_0": 5 * GIB}), CAPPED)
    assert not profiles.target_admits(record({"Q8_0": 5 * GIB + 1}), CAPPED)


def test_an_unknown_size_never_waives_a_cell():
    assert profiles.target_admits({"downloads": []}, CAPPED)


def test_expected_speed_drops_an_out_of_scope_target():
    cells = profiles.expected_speed(record({"Q8_0": 24 * GIB}), PROFILE)
    assert {cell["machine"] for cell in cells} == {"m4-max"}


def test_expected_speed_keeps_an_in_scope_target():
    cells = profiles.expected_speed(record({"Q8_0": 4 * GIB}), PROFILE)
    assert {cell["machine"] for cell in cells} == {"m4", "m4-max"}


def test_admission_reason_names_the_policy():
    reason = profiles.admission_reason(record({"Q8_0": 24 * GIB}), CAPPED)
    assert reason and "24.00 GiB" in reason and "max_model_gib=5" in reason


def test_the_shipped_profile_keeps_the_voxtral_minis_and_drops_the_24b():
    _, profile = profiles.load_profile()
    target = profiles.target_for_machine(profile, "apple-m4")
    records = profiles.common.load_records()
    assert profiles.target_admits(records["voxtral-mini-3b-2507"], target)
    assert profiles.target_admits(records["voxtral-mini-4b-realtime-2602"], target)
    assert not profiles.target_admits(records["voxtral-small-24b-2507"], target)


STREAMING_PROFILE = {
    "speed": {
        "quants": "all-downloads",
        "samples": ["jfk-8s"],
        "streaming_samples": ["jfk-40ms", "jfk-40ms-x8"],
        "iterations": 1,
        "warmup": 0,
        "targets": [UNCAPPED],
    }
}


def test_streaming_samples_apply_only_to_streaming_models():
    still = record({"F32": GIB})
    live = {**record({"F32": GIB}), "capabilities": {"streaming": {"supported": True, "verified": True}}}
    assert profiles.speed_samples(still, STREAMING_PROFILE) == ["jfk-8s"]
    assert profiles.speed_samples(live, STREAMING_PROFILE) == ["jfk-8s", "jfk-40ms", "jfk-40ms-x8"]
    assert "jfk-40ms-x8" in profiles.all_speed_samples(STREAMING_PROFILE, {})


def test_turn_records_follow_the_turn_profile_on_the_m1_proxy():
    records = profiles.common.load_records()
    for variant in ("smart-turn-v3.2", "sava"):
        profile_id, profile = profiles.profile_for(records[variant])
        assert profile_id == "turn-publication-v1"
        assert profiles.target_for_machine(profile, "apple-m1")["machine"] == "m1"
        cells = profiles.expected_accuracy(records[variant], profile)
        assert {cell["quant"] for cell in cells} == {"F32", "F16", "Q8_0"}
        assert all(cell["threshold"] == 0.5 and cell["language"] == "mul" for cell in cells)
    smart, sava = (profiles.expected_speed(records[v], profiles.profile_for(records[v])[1])
                   for v in ("smart-turn-v3.2", "sava"))
    assert not any("40ms" in cell["sample"] for cell in smart)
    assert any(cell["sample"] == "jfk-40ms-x8" for cell in sava)
