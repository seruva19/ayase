"""Tests for commonsense module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_commonsense_basics():
    from ayase.modules.commonsense import CommonsenseModule
    _test_module_basics(CommonsenseModule, "commonsense")

def test_commonsense_image(image_sample):
    from ayase.modules.commonsense import CommonsenseModule
    image_sample.quality_metrics = QualityMetrics()
    m = CommonsenseModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_commonsense_video(video_sample):
    from ayase.modules.commonsense import CommonsenseModule
    video_sample.quality_metrics = QualityMetrics()
    m = CommonsenseModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample



def test_commonsense_uses_config_driven_model_and_processor(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from ayase.modules.commonsense import CommonsenseModule

    default_id = "llava-hf/llava-1.5-7b-hf"
    model_types = {
        default_id: "llava",
        "custom/another-llava-family": "llava_next",
        "custom/another-vlm-family": "qwen2_5_vl",
    }
    calls = []

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("model", model_id))
            model = SimpleNamespace(config=SimpleNamespace(model_type=model_types[model_id]))
            model.to = lambda device: model
            model.eval = lambda: model
            return model

    class AutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("processor", model_id))
            return SimpleNamespace(model_type=model_types[model_id])

    transformers = ModuleType("transformers")
    transformers.AutoModelForImageTextToText = AutoModel
    transformers.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(CommonsenseModule, "test_mode", False)
    for model_id in model_types:
        config = {"device": "cpu", "attention_backend": "default"}
        if model_id != default_id:
            config["vlm_model"] = model_id
        module = CommonsenseModule(config)
        module.setup()
        assert module._ml_available
        assert module._vlm_model.config.model_type == model_types[model_id]
        assert module._vlm_processor.model_type == model_types[model_id]
    assert sorted(calls) == sorted(
        (kind, model_id) for model_id in model_types for kind in ("model", "processor")
    )
