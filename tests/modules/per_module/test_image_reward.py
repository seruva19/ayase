"""Tests for ImageReward module."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_image_reward_basics():
    from ayase.modules.image_reward import ImageRewardModule

    _test_module_basics(ImageRewardModule, "image_reward")


def test_image_reward_skip_no_caption(image_sample):
    """Module should skip gracefully when no caption is available."""
    from ayase.modules.image_reward import ImageRewardModule

    m = ImageRewardModule()
    # Don't call setup so _ml_available stays False
    result = m.process(image_sample)
    assert result is image_sample
    # No score should be set
    if result.quality_metrics:
        assert result.quality_metrics.image_reward_score is None


def test_image_reward_skip_no_ml(image_sample):
    """Module should skip gracefully when ML dependencies are missing."""
    from ayase.modules.image_reward import ImageRewardModule

    image_sample.quality_metrics = QualityMetrics()
    m = ImageRewardModule()
    # Don't call setup — _ml_available stays False
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.image_reward_score is None


def test_image_reward_field_exists():
    """Verify image_reward_score field exists in QualityMetrics."""
    qm = QualityMetrics()
    assert hasattr(qm, "image_reward_score")
    assert qm.image_reward_score is None
    qm.image_reward_score = 1.5
    assert qm.image_reward_score == 1.5


def test_image_reward_field_group():
    """Verify image_reward_score is in the alignment group."""
    qm = QualityMetrics()
    groups = qm._FIELD_GROUPS
    assert groups.get("image_reward_score") == "alignment"
