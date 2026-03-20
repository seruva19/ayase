"""Tests for object_permanence module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_object_permanence_basics():
    from ayase.modules.object_permanence import ObjectPermanenceModule
    _test_module_basics(ObjectPermanenceModule, "object_permanence")

def test_object_permanence_video(video_sample):
    from ayase.modules.object_permanence import ObjectPermanenceModule
    video_sample.quality_metrics = QualityMetrics()
    m = ObjectPermanenceModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
