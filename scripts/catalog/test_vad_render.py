"""VAD rendering and backward-compatible HF cards.

    uv run --no-project --with pytest --with pyyaml --with jinja2 \\
        --with huggingface-hub pytest scripts/catalog/test_vad_render.py
"""
import hashlib
import importlib.util
import pathlib
import sys

import pytest
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import render as catalog_render  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "hf_card_generate", common.CARDS_DIR / "generate.py")
hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf)


@pytest.fixture
def record():
    return {
        "variant": "fixture-vad", "family": "silero_vad", "role": "vad",
        "display_name": "Fixture VAD", "params": 309633,
        "license": {"spdx": "mit", "display": "MIT"},
        "upstream_repo": "snakers4/silero-vad", "upstream_commit": "abcdef0",
        "upstream_url": "https://github.com/snakers4/silero-vad",
        "published_repo": None, "languages": [],
        "vad_info": {"sample_rate": 16000, "frame_samples": 512},
        "source_artifact": {
            "package": "fixture-vad", "version": "1.0", "filename": "weights.jit",
            "sha256": "a" * 64, "url": "https://example.org/weights.jit"},
        "capabilities": {"streaming": {"supported": True}},
        "downloads": [{"quant": "F32", "filename": "fixture-vad-F32.gguf",
                       "size_bytes": 1241056}],
        "headline_benchmark": None, "accuracy_benchmarks": [],
        "reference_parity": [{
            "dataset": "fleurs", "split": "test", "language": "en",
            "quant": "F32", "backend": "cpu", "reference": "TorchScript",
            "n_files": 3, "n_identical": 3, "n_segments": 5,
            "audio_duration_s": 12.125, "segmentation_params": "defaults",
            "engine_sha": "abcdef0", "measured_on": "2026-10-09",
            "publication_profile": "fixture-vad-v1", "max_abs_prob_delta": 1.23e-7}],
        "speed_benchmarks": [{
            "machine": "m4", "backend": "cpu", "quant": "F32",
            "sample": f"love-loss-{chunk}ms", "sample_duration_s": chunk / 1000,
            "feed_samples": frames * 512, "frame_samples": 512, "threads": 1,
            "n_calls": 100, "warmup_calls": 10, "total_ms": 0.012345,
            "median_ms": 0.009876, "p95_ms": 0.022222,
            "xrt_compute": chunk / 0.012345, "engine_sha": "abcdef0",
            "measured_on": "2026-10-09", "publication_profile": "fixture-vad-v1"}
            for chunk, frames in ((32, 1), (128, 4), (512, 16))],
    }


@pytest.fixture
def spec():
    return {
        "pin_date": "2026-10-09",
        "validation": {"reference": "TorchScript", "commit": "abcdef0",
                       "date": "2026-10-09"},
        "default_quant": "F32", "pipeline_tag": "audio-classification",
        "tags": ["gguf", "voice-activity-detection"],
        "summary": "Voice activity detection, not transcription.",
        "compatibility": "Other formats are compatibility inputs, not canonical downloads.",
        "wer": {"notes": "Parity is not human-annotated detection accuracy."},
    }


def frontmatter(card):
    return yaml.safe_load(card.split("---", 2)[1])


def test_non_hf_links_and_artifact_provenance(record, spec, monkeypatch):
    monkeypatch.setitem(catalog_render._SPECS, record["variant"], spec)
    intro = "\n".join(catalog_render.block_intro(record, {}))
    pin = "\n".join(catalog_render.block_pin(record, {}))
    for text in (intro, pin):
        assert "https://github.com/snakers4/silero-vad/commit/abcdef0" in text
        assert "https://huggingface.co/snakers4" not in text
    assert record["source_artifact"]["sha256"] in pin
    assert "Source artifact:" not in intro
    assert "309,633 parameters" in intro
    assert "16,000 Hz mono; 512 samples/frame (32 ms)" in intro
    assert "Model:" not in pin
    assert common.fmt_params(record["params"]) == "309K"
    assert "voice activity detection" in common.capabilities_summary(record)


def test_pending_downloads_have_filenames_not_links(record):
    table = "\n".join(catalog_render.block_downloads(record, {}))
    assert "`fixture-vad-F32.gguf`" in table
    assert "Canonical publication pending" in table
    assert "huggingface.co" not in table
    assert "WER" not in table
    assert "None" not in table
    record["published_repo"] = "handy-computer/fixture-vad-gguf"
    published = "\n".join(catalog_render.block_downloads(record, {}))
    assert common.download_url(record, "fixture-vad-F32.gguf") in published
    assert "pending" not in published


def test_card_uses_shared_tables_and_role_aware_metadata(record, spec):
    card = hf.render(hf.build_context(record, spec), "")
    metadata = frontmatter(card)
    assert "base_model" not in metadata
    assert "base_model_relation" not in metadata
    assert "language" not in metadata
    assert "https://huggingface.co/" not in card
    assert "| WER" not in card
    assert "Canonical publication pending" in card
    assert "## Original Model Card" not in card
    assert spec["compatibility"] in card
    assert spec["wer"]["notes"] in card
    assert "\n".join(catalog_render.block_reference_parity(record, {})) in card
    assert "\n".join(catalog_render.block_stream_perf(record, {"machine": "m4"})) in card
    block = metadata["transcribe_cpp"]
    assert block["schema_version"] == 3
    assert block["role"] == "vad"
    assert block["params"] == record["params"]
    assert block["vad_info"] == record["vad_info"]
    assert block["reference_parity"] == record["reference_parity"]
    assert block["latency_benchmarks"] == record["speed_benchmarks"]
    assert not any(key.startswith(("wer_", "accuracy_", "rtf_")) for key in block)
    assert "not labeled VAD accuracy" in card
    assert "1.23e-07" in card


def test_stream_precision_and_latency_scope(record):
    text = "\n".join(catalog_render.block_stream_perf(record, {"machine": "m4"}))
    assert "0.009876 ms" in text
    assert "0.002469 ms" in text  # 128 ms feed: median / 4 frames
    assert "0.00061725 ms" in text  # 512 ms feed: median / 16 frames
    assert "0 ms" not in text
    for phrase in ("32 ms chunk is the headline", "Python ctypes/native API wall call",
                   "excludes model load and audio capture", "preserved stream state",
                   "Warmup calls", "Amortized median/frame", "p95/feed", "Frames/feed"):
        assert phrase in text


def test_narrow_tables_keep_recipe_and_all_metadata(record, spec):
    parity = catalog_render.block_reference_parity(record, {})
    perf = catalog_render.block_stream_perf(record, {"machine": "m4"})
    parity_header = next(line for line in parity if line.startswith("|"))
    perf_header = next(line for line in perf if line.startswith("|"))
    assert len(parity_header.split("|")) - 2 == 7
    assert len(perf_header.split("|")) - 2 == 8
    assert "Backend: `cpu`. Reference: `TorchScript`." in "\n".join(parity)
    assert "Segmentation parameters: `defaults`." in "\n".join(parity)
    assert "Samples/frame: 512. Threads: 1. Warmup calls: 10." in "\n".join(perf)
    assert "Samples/frame" not in perf_header
    assert "Mean/feed" not in perf_header
    block = frontmatter(hf.render(hf.build_context(record, spec), ""))["transcribe_cpp"]
    assert block["latency_benchmarks"] == record["speed_benchmarks"]
    assert block["reference_parity"] == record["reference_parity"]


def test_parity_varying_recipe_retains_row_identity(record):
    record["reference_parity"].append({**record["reference_parity"][0],
                                      "backend": "metal", "reference": "Other reference"})
    lines = catalog_render.block_reference_parity(record, {})
    header = next(line for line in lines if line.startswith("|"))
    assert "Backend" in header
    assert "Reference" in header
    assert "Segmentation parameters" not in header
    assert any("metal" in line and "Other reference" in line for line in lines if line.startswith("|"))


def test_stream_distinct_recipe_values_are_reported(record):
    record["speed_benchmarks"][1].update(threads=2, warmup_calls=20)
    text = "\n".join(catalog_render.block_stream_perf(record, {"machine": "m4"}))
    assert "Threads: 1, 2." in text
    assert "Warmup calls: 10, 20." in text


def test_markers_and_prose_are_reusable(record, spec, tmp_path, monkeypatch):
    monkeypatch.setitem(catalog_render._SPECS, record["variant"], spec)
    path = tmp_path / "fixture-vad.md"
    path.write_text("# Fixture\n\n<!-- catalog:reference-parity -->\n<!-- /catalog -->\n"
                    "<!-- catalog:stream-perf machine=m4 -->\n<!-- /catalog -->\n"
                    "<!-- catalog:prose field=compatibility -->\n<!-- /catalog -->\n")
    output, errors = catalog_render.rewrite(path, {record["variant"]: record})
    assert not errors
    assert spec["compatibility"] in output
    path.write_text(output)
    assert catalog_render.rewrite(path, {record["variant"]: record}) == (output, [])


def test_non_hf_main_never_fetches_and_defaults_to_variant(record, spec, tmp_path, monkeypatch):
    def unexpected_fetch(*args):
        pytest.fail("non-HF source was fetched through the HF API")
    monkeypatch.setattr(hf, "fetch_upstream_card", unexpected_fetch)
    monkeypatch.setattr(hf, "load_spec", lambda path: spec)
    monkeypatch.setattr(hf.common, "load_record", lambda variant: record)
    monkeypatch.setattr(hf, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate.py", "fixture-vad.yaml"])
    assert hf.main() == 0
    assert (tmp_path / "models" / "fixture-vad" / "README.md").exists()
    assert hf.default_output_path(record).parent.name == record["variant"]


def test_pending_measurements_and_optional_geometry(record, spec):
    record["reference_parity"] = []
    record["speed_benchmarks"] = []
    del record["vad_info"]
    card = hf.render(hf.build_context(record, spec), "")
    assert "Reference parity measurements pending." in card
    assert "Streaming latency measurements pending." in card
    assert "vad_info" not in frontmatter(card)["transcribe_cpp"]


def test_editorial_sections_reject_numbers(tmp_path):
    path = tmp_path / "fixture.yaml"
    path.write_text("stream_perf:\n  median_ms: 0.1\n")
    with pytest.raises(SystemExit, match="editorial notes"):
        hf.load_spec(path)


@pytest.mark.parametrize("variant,expected", [
    ("whisper-tiny.en", "563febea819ecb2afefef777d74ba74ba1039211f15c1805d57bd56b2d5c497e"),
    ("lang-id-voxlingua107-ecapa", "dd1f8ad90061ee9470c60430dd0acc1354aeb89f7e117a7062c029afb74b8d1e"),
    ("diar_streaming_sortformer_4spk-v2.1", "f4a241d16b8abc95e5760803b964c27eb624e8b31cc98046724ef186b2f5ef35"),
])
def test_existing_cards_byte_identical(variant, expected):
    # Hashes of the pre-VAD generator with deterministic upstream text.
    ctx = hf.build_context(common.load_record(variant), hf.load_spec(common.CARDS_DIR / f"{variant}.yaml"))
    card = hf.render(ctx, "BASELINE UPSTREAM")
    assert hashlib.sha256(card.encode()).hexdigest() == expected
    assert frontmatter(card)["transcribe_cpp"]["schema_version"] == 2
