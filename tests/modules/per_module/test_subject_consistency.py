"""Tests for subject_consistency module."""

import sys
from types import ModuleType, SimpleNamespace

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_subject_consistency_basics():
    from ayase.modules.subject_consistency import SubjectConsistencyModule

    _test_module_basics(SubjectConsistencyModule, "subject_consistency")


def test_subject_consistency_video(video_sample):
    from ayase.modules.subject_consistency import SubjectConsistencyModule

    video_sample.quality_metrics = QualityMetrics()
    m = SubjectConsistencyModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def _install_loader_fakes(monkeypatch):
    calls = {"processor": [], "model": [], "resource_keys": []}

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["processor"].append((model_name, kwargs))
            return object()

    transformers = ModuleType("transformers")
    transformers.AutoImageProcessor = FakeProcessorLoader
    transformers.AutoModel = object
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    import ayase.runtime as runtime

    monkeypatch.setattr(runtime, "resolve_torch_device", lambda device: "cpu")

    def fake_shared_resource(owner, key, factory):
        calls["resource_keys"].append(key)
        return factory()

    monkeypatch.setattr(runtime, "shared_runtime_resource", fake_shared_resource)

    def fake_model_loader(model_cls, model_name, config, **kwargs):
        calls["model"].append((model_cls, model_name, config, kwargs))
        return FakeModel()

    monkeypatch.setattr(runtime, "from_pretrained_with_attention", fake_model_loader)
    return SimpleNamespace(calls=calls)


def test_subject_consistency_forwards_explicit_revision(monkeypatch):
    from ayase.modules.subject_consistency import SubjectConsistencyModule

    fakes = _install_loader_fakes(monkeypatch)
    revision = "0123456789abcdef0123456789abcdef01234567"
    module = SubjectConsistencyModule({"revision": revision})
    module.setup()

    assert module._ml_available is True
    assert fakes.calls["processor"][0][1]["revision"] == revision
    assert fakes.calls["model"][0][3]["revision"] == revision


def test_subject_consistency_default_omits_revision(monkeypatch):
    from ayase.modules.subject_consistency import SubjectConsistencyModule

    fakes = _install_loader_fakes(monkeypatch)
    module = SubjectConsistencyModule()
    module.setup()

    assert module._ml_available is True
    assert "revision" not in fakes.calls["processor"][0][1]
    assert "revision" not in fakes.calls["model"][0][3]
    assert fakes.calls["resource_keys"][0] == (
        "hf_vision",
        "facebook/dino-vitb16",
        "cpu",
        "auto",
        "safetensors",
    )


def test_subject_consistency_explicit_revisions_have_distinct_resource_keys(monkeypatch):
    from ayase.modules.subject_consistency import SubjectConsistencyModule

    fakes = _install_loader_fakes(monkeypatch)
    revision_a = "0123456789abcdef0123456789abcdef01234567"
    revision_b = "89abcdef0123456789abcdef0123456789abcdef"

    SubjectConsistencyModule({"revision": revision_a}).setup()
    SubjectConsistencyModule({"revision": revision_b}).setup()

    key_a, key_b = fakes.calls["resource_keys"]
    assert key_a != key_b
    assert key_a[-2:] == ("revision", revision_a)
    assert key_b[-2:] == ("revision", revision_b)


def test_subject_consistency_invalid_revision_disables_setup(monkeypatch):
    from ayase.modules.subject_consistency import SubjectConsistencyModule

    fakes = _install_loader_fakes(monkeypatch)
    module = SubjectConsistencyModule({"revision": "   "})
    module.setup()

    assert module._ml_available is False
    assert not fakes.calls["processor"]
    assert not fakes.calls["model"]


def test_subject_consistency_missing_dependency_is_graceful(monkeypatch):
    from ayase.modules.subject_consistency import SubjectConsistencyModule

    monkeypatch.setitem(sys.modules, "transformers", None)
    module = SubjectConsistencyModule({"revision": "0123456789abcdef"})
    module.setup()
    assert module._ml_available is False
