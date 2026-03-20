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
