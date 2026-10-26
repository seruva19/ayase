"""Tests for vlm_judge module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics
import pytest


def test_vlm_judge_basics():
    from ayase.modules.vlm_judge import VLMJudgeModule
    _test_module_basics(VLMJudgeModule, "vlm_judge")

def test_vlm_judge_image(image_sample):
    from ayase.modules.vlm_judge import VLMJudgeModule
    image_sample.quality_metrics = QualityMetrics()
    m = VLMJudgeModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_vlm_judge_video(video_sample):
    from ayase.modules.vlm_judge import VLMJudgeModule
    video_sample.quality_metrics = QualityMetrics()
    m = VLMJudgeModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample



@pytest.mark.parametrize("legacy_transformers", [False, True])
def test_vlm_judge_uses_config_driven_model_and_processor(monkeypatch, legacy_transformers):
    import sys
    from types import ModuleType, SimpleNamespace
    from ayase.modules.vlm_judge import VLMJudgeModule

    default_id = "llava-hf/llava-1.5-7b-hf"
    model_types = {
        default_id: "llava",
        "custom/another-llava-family": "llava_next",
        "custom/another-vlm-family": "qwen2_vl",
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
    if legacy_transformers:
        transformers.AutoModelForVision2Seq = AutoModel
    else:
        transformers.AutoModelForImageTextToText = AutoModel
    transformers.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(VLMJudgeModule, "test_mode", False)
    for model_id in model_types:
        config = {"device": "cpu", "attention_backend": "default"}
        if model_id != default_id:
            config["model_name"] = model_id
        module = VLMJudgeModule(config)
        module.setup()
        assert module._ml_available
        assert module._model.config.model_type == model_types[model_id]
        assert module._processor.model_type == model_types[model_id]
    assert sorted(calls) == sorted(
        (kind, model_id) for model_id in model_types for kind in ("model", "processor")
    )
