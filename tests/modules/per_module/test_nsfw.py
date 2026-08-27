"""Tests for nsfw module."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_nsfw_basics():
    from ayase.modules.nsfw import NSFWModule
    _test_module_basics(NSFWModule, "nsfw")

def test_nsfw_image(image_sample):
    from ayase.modules.nsfw import NSFWModule
    image_sample.quality_metrics = QualityMetrics()
    m = NSFWModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_nsfw_video(video_sample):
    from ayase.modules.nsfw import NSFWModule
    video_sample.quality_metrics = QualityMetrics()
    m = NSFWModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


@pytest.mark.parametrize(
    ("probabilities", "threshold", "expected"),
    [
        ([0.1, 0.5, 0.7, 0.2], 0.5, 0.5),
        ([0.49, 0.49], 0.5, 0.0),
        ([0.5, 0.9], 0.5, 1.0),
        ([], 0.5, 0.0),
    ],
)
def test_temporal_risk_rate(probabilities, threshold, expected):
    from ayase.modules.nsfw import NSFWModule

    assert NSFWModule._temporal_risk_rate(probabilities, threshold) == expected


def test_custom_model_does_not_inherit_default_revision():
    from ayase.modules.nsfw import NSFWModule

    module = NSFWModule({"model_name": "example/custom-safety-model"})
    assert module.model_revision is None
