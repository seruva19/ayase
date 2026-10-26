"""Tests for llm_descriptive_qa module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_llm_descriptive_qa_basics():
    from ayase.modules.llm_descriptive_qa import LLMDescriptiveQAModule
    _test_module_basics(LLMDescriptiveQAModule, "llm_descriptive_qa")

def test_llm_descriptive_qa_image(image_sample):
    from ayase.modules.llm_descriptive_qa import LLMDescriptiveQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = LLMDescriptiveQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_llm_descriptive_qa_video(video_sample):
    from ayase.modules.llm_descriptive_qa import LLMDescriptiveQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = LLMDescriptiveQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample



def test_llm_descriptive_qa_uses_config_driven_model_and_processor(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from ayase.modules.llm_descriptive_qa import LLMDescriptiveQAModule

    default_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    model_types = {
        default_id: "llava_next",
        "custom/another-llava-family": "llava",
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
    monkeypatch.setattr(LLMDescriptiveQAModule, "test_mode", False)
    for model_id in model_types:
        config = {"device": "cpu", "attention_backend": "default"}
        if model_id != default_id:
            config["model_name"] = model_id
        module = LLMDescriptiveQAModule(config)
        module.setup()
        assert module._ml_available
        assert module._model.config.model_type == model_types[model_id]
        assert module._processor.model_type == model_types[model_id]
    assert sorted(calls) == sorted(
        (kind, model_id) for model_id in model_types for kind in ("model", "processor")
    )
