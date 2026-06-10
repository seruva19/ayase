"""Tests for UnifiedReward Edit module."""

from ..conftest import _test_module_basics


def test_unified_reward_edit_basics():
    from ayase.modules.unified_reward_edit import UnifiedRewardEditModule

    _test_module_basics(UnifiedRewardEditModule, "unified_reward_edit")


def test_unified_reward_edit_no_backend_is_noop(image_sample):
    from ayase.models import CaptionMetadata
    from ayase.modules.unified_reward_edit import UnifiedRewardEditModule

    image_sample.caption = CaptionMetadata(text="make the apple red", length=18)
    image_sample.reference_path = image_sample.path
    result = UnifiedRewardEditModule().process(image_sample)

    assert result is image_sample
    if result.quality_metrics is not None:
        assert result.quality_metrics.unified_reward_edit_score is None


def test_unified_reward_edit_no_reference_is_noop(image_sample):
    from ayase.modules.unified_reward_edit import UnifiedRewardEditModule

    result = UnifiedRewardEditModule({"backend": "openai", "endpoint_url": "http://localhost"}).process(
        image_sample
    )

    assert result is image_sample


def test_unified_reward_edit_parse_pointwise_json():
    from ayase.modules.unified_reward_edit import parse_unified_reward_edit_output

    parsed = parse_unified_reward_edit_output(
        '{"reasoning": "ok", "score": [20, 22]}',
        "edit_pointwise_score",
    )

    assert parsed["editing_success"] == 20.0
    assert parsed["overediting"] == 22.0
    assert parsed["score"] == 21.0


def test_unified_reward_edit_parse_pairwise_score():
    from ayase.modules.unified_reward_edit import parse_unified_reward_edit_output

    parsed = parse_unified_reward_edit_output(
        "Edited Image 1: 3.5\nEdited Image 2: 4.5",
        "edit_pairwise_score",
    )

    assert parsed["image_1_score"] == 3.5
    assert parsed["image_2_score"] == 4.5
    assert parsed["score"] == 4.0


def test_unified_reward_edit_parse_pairwise_rank():
    from ayase.modules.unified_reward_edit import parse_unified_reward_edit_output

    parsed = parse_unified_reward_edit_output(
        "The second image is better and more faithful.",
        "edit_pairwise_rank",
    )

    assert parsed["winner"] == "Edited image 2"
    assert parsed["winner_score"] == 2.0
    assert parsed["score"] == 2.0


def test_unified_reward_edit_store_scores(image_sample):
    from ayase.modules.unified_reward_edit import UnifiedRewardEditModule

    UnifiedRewardEditModule._store_scores(
        image_sample,
        {"editing_success": 20.0, "overediting": 22.0, "score": 21.0},
    )

    qm = image_sample.quality_metrics
    assert qm.unified_reward_edit_success_score == 20.0
    assert qm.unified_reward_edit_overediting_score == 22.0
    assert qm.unified_reward_edit_score == 21.0
