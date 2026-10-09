"""Record seeding follows source provenance and the public model role."""
import json
import pathlib
import sys
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import new_record
import sync_capabilities


class Field:
    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


def setup_source(tmp_path, monkeypatch, *, revision=None, checksum="a" * 64):
    variant = "test-vad"
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    docs = tmp_path / "docs" / "models"
    docs.mkdir(parents=True)
    (docs / f"{variant}.md").write_text("# Test VAD\n")
    intake = tmp_path / "reports" / "porting" / "test_vad" / variant / "intake.json"
    intake.parent.mkdir(parents=True)
    intake.write_text(json.dumps({"family": "test_vad", "hf_repo": "example/vad",
                                  "hf_revision": revision, "capabilities": {"languages": []}}))
    manifest = tmp_path / "tests" / "golden" / "test_vad" / f"{variant}.manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"role": "vad", "source_model": {
        "package": "example-vad==1.0", "file": "vad.jit", "sha256": checksum}}))
    gguf = tmp_path / "models" / variant / f"{variant}-F32.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"fixture")
    fields = {"general.license": "mit", "general.repo_url": "https://github.com/example/vad",
              "general.source.url": "https://pypi.org/project/example-vad/1.0/",
              "stt.frontend.sample_rate": 16000, "stt.vad.frame_samples": 512,
              "stt.test_vad.source_sha256": "a" * 64}
    reader = SimpleNamespace(fields={key: Field(value) for key, value in fields.items()},
                             tensors=[SimpleNamespace(shape=(2, 3))])
    monkeypatch.setattr(new_record, "GGUFReader", lambda _: reader)
    monkeypatch.setattr(new_record.common, "REPO", tmp_path)
    monkeypatch.setattr(new_record.common, "CATALOG_DIR", catalog)
    monkeypatch.setattr(new_record.common, "DOCS_DIR", docs)
    return variant, catalog / f"{variant}.json"


def seed_args(monkeypatch, variant, *extra):
    monkeypatch.setattr(sys, "argv", ["new_record.py", variant,
        "--long-form", "chunked-unbounded", "--docs-page", f"{variant}.md", *extra])


def test_package_source_and_role_are_seeded(tmp_path, monkeypatch):
    variant, out = setup_source(tmp_path, monkeypatch)
    seed_args(monkeypatch, variant, "--upstream-commit", "123abcd", "--unpublished")
    assert new_record.main() == 0
    record = json.loads(out.read_text())
    assert record["role"] == "vad"
    assert record["published_repo"] is None
    assert record["upstream_commit"] == "123abcd"
    assert record["upstream_url"] == "https://github.com/example/vad"
    assert record["source_artifact"] == {
        "package": "example-vad", "version": "1.0", "filename": "vad.jit",
        "sha256": "a" * 64, "url": "https://pypi.org/project/example-vad/1.0/"}
    assert record["vad_info"] == {"sample_rate": 16000, "frame_samples": 512}
    assert record["params"] == 6
    assert record["capabilities"]["transcribe"] == {"supported": False}
    assert record["reference_parity"] == []
    assert record["headline_benchmark"] is None


def test_missing_revision_does_not_become_none_string(tmp_path, monkeypatch):
    variant, out = setup_source(tmp_path, monkeypatch)
    seed_args(monkeypatch, variant)
    assert new_record.main() == 2
    assert not out.exists()


def test_source_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    variant, out = setup_source(tmp_path, monkeypatch, checksum="b" * 64)
    seed_args(monkeypatch, variant, "--upstream-commit", "123abcd")
    assert new_record.main() == 2
    assert not out.exists()


def test_standalone_roles_never_seed_as_transcription():
    for role in ("vad", "langid", "diarize"):
        caps = sync_capabilities.build({"family": "test", "role": role}, {})
        assert caps["transcribe"] == {"supported": False}
    caps = sync_capabilities.build({"family": "test"}, {})
    assert caps["transcribe"] == {"supported": True, "verified": False}
