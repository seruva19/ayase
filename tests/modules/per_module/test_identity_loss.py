"""Tests for identity_loss module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_identity_loss_basics():
    from ayase.modules.identity_loss import IdentityLossModule
    _test_module_basics(IdentityLossModule, "identity_loss")

def test_identity_loss_image(image_sample):
    from ayase.modules.identity_loss import IdentityLossModule
    image_sample.quality_metrics = QualityMetrics()
    m = IdentityLossModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_identity_loss_video(video_sample):
    from ayase.modules.identity_loss import IdentityLossModule
    video_sample.quality_metrics = QualityMetrics()
    m = IdentityLossModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
