"""Tests for UnifiedReward 2.0 module."""

from ..conftest import _test_module_basics


def test_unified_reward_2_basics():
    from ayase.modules.unified_reward_2 import UnifiedReward2Module

    _test_module_basics(UnifiedReward2Module, "unified_reward_2")


def test_unified_reward_2_no_backend_is_noop(image_sample):
    from ayase.models import CaptionMetadata
    from ayase.modules.unified_reward_2 import UnifiedReward2Module

    image_sample.caption = CaptionMetadata(text="a red apple", length=11)
    result = UnifiedReward2Module().process(image_sample)

    assert result is image_sample
    if result.quality_metrics is not None:
        assert result.quality_metrics.unified_reward_2_score is None


def test_unified_reward_2_parse_labeled_output():
    from ayase.modules.unified_reward_2 import parse_unified_reward_2_output

    parsed = parse_unified_reward_2_output(
        "Alignment Score (1-5): 4\nCoherence Score (1-5): 3\nStyle Score (1-5): 5"
    )

    assert parsed["alignment"] == 4.0
    assert parsed["coherence"] == 3.0
    assert parsed["style"] == 5.0
    assert parsed["score"] == 4.0


def test_unified_reward_2_parse_json_output():
    from ayase.modules.unified_reward_2 import parse_unified_reward_2_output

    parsed = parse_unified_reward_2_output(
        '{"alignment": {"score": 5}, "coherence": 4, "style_score": "3"}'
    )

    assert parsed["alignment"] == 5.0
    assert parsed["coherence"] == 4.0
    assert parsed["style"] == 3.0
    assert parsed["score"] == 4.0


def test_unified_reward_2_store_scores(image_sample):
    from ayase.modules.unified_reward_2 import UnifiedReward2Module

    UnifiedReward2Module._store_scores(
        image_sample,
        {"alignment": 5.0, "coherence": 4.0, "style": 3.0, "score": 4.0},
    )

    qm = image_sample.quality_metrics
    assert qm.unified_reward_2_alignment_score == 5.0
    assert qm.unified_reward_2_coherence_score == 4.0
    assert qm.unified_reward_2_style_score == 3.0
    assert qm.unified_reward_2_score == 4.0

