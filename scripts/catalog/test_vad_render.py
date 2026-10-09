"""Keep the shared HF generator's role-specific metadata compatible."""
import importlib.util
import pathlib
import sys

import pytest
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

_spec = importlib.util.spec_from_file_location("hf_card_generate", common.CARDS_DIR / "generate.py")
hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf)


@pytest.mark.parametrize("variant,version", [
    ("whisper-tiny.en", 2),
    ("lang-id-voxlingua107-ecapa", 2),
    ("diar_streaming_sortformer_4spk-v2.1", 2),
    ("silero-vad-v6.2", 3),
])
def test_card_role_metadata(variant, version):
    record = common.load_record(variant)
    spec = hf.load_spec(common.CARDS_DIR / f"{variant}.yaml")
    card = hf.render(hf.build_context(record, spec), "")
    metadata = yaml.safe_load(card.split("---", 2)[1])
    block = metadata["transcribe_cpp"]
    assert block["schema_version"] == version
    if record.get("role") == "vad":
        assert block["role"] == "vad"
        assert "base_model" not in metadata
        assert "| WER" not in card
        assert not any(key.startswith(("wer_", "rtf_")) for key in block)
