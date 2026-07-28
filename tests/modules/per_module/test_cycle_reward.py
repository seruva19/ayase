"""Focused tests for the upstream CycleReward module."""

from types import SimpleNamespace

import pytest
import torch

from ayase.models import CaptionMetadata, Sample
from tests.modules.conftest import _test_module_basics


def test_cycle_reward_basics():
    from ayase.modules.cycle_reward import CycleRewardModule

    _test_module_basics(CycleRewardModule, "cycle_reward")


def test_cycle_reward_without_backend_leaves_sample_unchanged(image_sample):
    from ayase.modules.cycle_reward import CycleRewardModule

    module = CycleRewardModule()
    sample = Sample(
        path=image_sample.path,
        is_video=False,
        caption=CaptionMetadata(text="a green image", length=13),
    )
    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_cycle_reward_scores_reference_api(monkeypatch, image_sample):
    from ayase.modules.cycle_reward import CycleRewardModule

    class Preprocess:
        def __call__(self, image):
            assert image.mode == "RGB"
            return torch.zeros(3, 224, 224)

    model = SimpleNamespace(
        score=lambda image, caption: torch.tensor([[2.5]], device=image.device)
    )
    module = CycleRewardModule()
    module._backend = "cyclereward"
    module._device = "cpu"
    module._model = model
    module._preprocess = Preprocess()

    sample = Sample(
        path=image_sample.path,
        is_video=False,
        caption=CaptionMetadata(text="a green image", length=13),
    )
    result = module.process(sample)
    assert result is sample
    assert result.quality_metrics.cycle_reward_score == pytest.approx(2.5)


def test_cycle_reward_only_allows_combo():
    from ayase.modules.cycle_reward import CycleRewardModule

    module = CycleRewardModule({"model_type": "CycleReward-T2I"})
    module.setup()
    assert module._backend == "unavailable"
