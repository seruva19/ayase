"""Tests for the vision_reward module (VisionReward, AAAI 2026).

These tests must pass without downloading any model: the root conftest enables
test_mode globally, so setup() is skipped and the module stays in its graceful
"unavailable" state (no checkpoint, metric left None).
"""

import numpy as np

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_vision_reward_basics():
    from ayase.modules.vision_reward import VisionRewardModule

    _test_module_basics(VisionRewardModule, "vision_reward")


def test_vision_reward_default_config():
    from ayase.modules.vision_reward import VisionRewardModule

    m = VisionRewardModule()
    cfg = m.default_config
    # Required config surface the task calls for.
    assert "device" in cfg
    assert "max_frames" in cfg
    assert "checkpoint" in cfg
    assert "judgment_questions" in cfg and "judgment_weights" in cfg
    # Checkpoint must point at a real VisionReward HF repo id.
    assert "VisionReward" in cfg["checkpoint"]
    # Grouped with the alignment/reward metrics.
    assert VisionRewardModule.metric_groups["vision_reward_score"] == "alignment"


def test_vision_reward_embedded_qa_matches_weights():
    """The embedded question set and weight vector must be paired 1:1."""
    from ayase.modules.vision_reward import _VIDEO_QUESTIONS, _VIDEO_WEIGHTS

    assert len(_VIDEO_QUESTIONS) == len(_VIDEO_WEIGHTS) == 29
    assert all('[[prompt]]' in q or q.endswith('?') for q in _VIDEO_QUESTIONS)


def test_vision_reward_setup_unavailable_without_model():
    """Without the checkpoint installed the backend stays 'unavailable'."""
    from ayase.modules.vision_reward import VisionRewardModule

    m = VisionRewardModule()
    m.on_mount()  # test_mode active -> setup skipped, stays unavailable
    assert m._backend == "unavailable"
    assert m._ml_available is False

    # Calling setup() directly must also leave it unavailable (no download).
    m.setup()
    assert m._backend == "unavailable"


def test_vision_reward_image_leaves_metric_none(image_sample):
    from ayase.modules.vision_reward import VisionRewardModule

    image_sample.quality_metrics = QualityMetrics()
    m = VisionRewardModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.vision_reward_score is None


def test_vision_reward_video_leaves_metric_none(video_sample):
    from ayase.modules.vision_reward import VisionRewardModule

    video_sample.quality_metrics = QualityMetrics()
    m = VisionRewardModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.vision_reward_score is None


def test_vision_reward_process_creates_quality_metrics(video_sample):
    """process() must populate quality_metrics even when unavailable."""
    from ayase.modules.vision_reward import VisionRewardModule

    video_sample.quality_metrics = None
    m = VisionRewardModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.vision_reward_score is None


def test_vision_reward_config_override_validation():
    """Mismatched judgment_questions/weights lengths must raise in resolution."""
    from ayase.modules.vision_reward import VisionRewardModule

    m = VisionRewardModule(
        {"judgment_questions": ["q1?", "q2?"], "judgment_weights": [0.5]}
    )
    try:
        m._resolve_questions_and_weights()
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected ValueError on mismatched override lengths"

    # Matching override lengths resolve to the provided set.
    m2 = VisionRewardModule(
        {"judgment_questions": ["q1?", "q2?"], "judgment_weights": [0.5, 0.25]}
    )
    questions, weights = m2._resolve_questions_and_weights()
    assert len(questions) == 2
    assert np.allclose(weights, [0.5, 0.25])
