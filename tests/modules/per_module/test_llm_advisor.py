"""Tests for llm_advisor module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_llm_advisor_basics():
    from ayase.modules.llm_advisor import LLMAdvisorModule
    _test_module_basics(LLMAdvisorModule, "llm_advisor")

def test_llm_advisor_image(image_sample):
    from ayase.modules.llm_advisor import LLMAdvisorModule
    image_sample.quality_metrics = QualityMetrics()
    m = LLMAdvisorModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_llm_advisor_video(video_sample):
    from ayase.modules.llm_advisor import LLMAdvisorModule
    video_sample.quality_metrics = QualityMetrics()
    m = LLMAdvisorModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
