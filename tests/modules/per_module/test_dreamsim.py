"""Tests for dreamsim module."""

import builtins
import sys
from types import ModuleType

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


class _FakeHub:
    def __init__(self):
        self.directory = "/original/torch-hub"

    def get_dir(self):
        return self.directory

    def set_dir(self, directory):
        self.directory = directory


def _fake_torch(hub):
    module = ModuleType("torch")
    module.hub = hub
    return module


def test_dreamsim_basics():
    from ayase.modules.dreamsim_metric import DreamSimModule

    _test_module_basics(DreamSimModule, "dreamsim")


def test_dreamsim_image(image_sample):
    from ayase.modules.dreamsim_metric import DreamSimModule

    image_sample.quality_metrics = QualityMetrics()
    image_sample.reference_path = image_sample.path  # exercise the image-vs-reference path
    m = DreamSimModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    # When the backend loads, the metric must actually be computed. A bare
    # `result is sample` assertion silently passed even while the forward raised
    # (preprocess batching / device mismatch) and the metric stayed None.
    if m._ml_available:
        assert image_sample.quality_metrics.dreamsim is not None


def test_dreamsim_video(video_sample):
    from ayase.modules.dreamsim_metric import DreamSimModule

    video_sample.quality_metrics = QualityMetrics()
    video_sample.reference_path = video_sample.path  # video sample + video reference
    m = DreamSimModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    if m._ml_available:
        assert video_sample.quality_metrics.dreamsim is not None


def test_dreamsim_setup_restores_torch_hub_after_success(monkeypatch):
    """DreamSim's package-local hub override cannot affect a later loader."""
    from ayase.modules.dreamsim_metric import DreamSimModule

    hub = _FakeHub()
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(hub))
    dreamsim_module = ModuleType("dreamsim")

    def load_dreamsim(pretrained=True):
        assert pretrained is True
        hub.set_dir("./models")
        return object(), object()

    dreamsim_module.dreamsim = load_dreamsim
    monkeypatch.setitem(sys.modules, "dreamsim", dreamsim_module)
    module = DreamSimModule()
    monkeypatch.setattr(module, "_ensure_dino_cached", lambda: None)

    module.setup()

    assert module._ml_available is True
    assert hub.get_dir() == "/original/torch-hub"

    def unrelated_loader():
        return hub.get_dir()

    assert unrelated_loader() == "/original/torch-hub"


def test_dreamsim_setup_restores_torch_hub_after_loader_failure(monkeypatch):
    """A DreamSim construction error also restores the process-global hub root."""
    from ayase.modules.dreamsim_metric import DreamSimModule

    hub = _FakeHub()
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(hub))
    dreamsim_module = ModuleType("dreamsim")

    def fail_dreamsim(pretrained=True):
        hub.set_dir("./models")
        raise RuntimeError("synthetic load failure")

    dreamsim_module.dreamsim = fail_dreamsim
    monkeypatch.setitem(sys.modules, "dreamsim", dreamsim_module)
    module = DreamSimModule()
    monkeypatch.setattr(module, "_ensure_dino_cached", lambda: None)

    module.setup()

    assert module._ml_available is False
    assert hub.get_dir() == "/original/torch-hub"


def test_dreamsim_setup_restores_torch_hub_after_import_failure(monkeypatch):
    """The finally path restores the hub root when importing DreamSim fails."""
    from ayase.modules.dreamsim_metric import DreamSimModule

    hub = _FakeHub()
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(hub))
    monkeypatch.delitem(sys.modules, "dreamsim", raising=False)
    original_import = builtins.__import__

    def fail_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dreamsim":
            hub.set_dir("./models")
            raise ImportError("synthetic import failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    module = DreamSimModule()
    monkeypatch.setattr(module, "_ensure_dino_cached", lambda: None)

    module.setup()

    assert module._ml_available is False
    assert hub.get_dir() == "/original/torch-hub"
