"""Tests for rqvqa module."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_rqvqa_basics():
    from ayase.modules.rqvqa import RQVQAModule
    _test_module_basics(RQVQAModule, "rqvqa")


def test_rqvqa_image(image_sample):
    from ayase.modules.rqvqa import RQVQAModule
    image_sample.quality_metrics = QualityMetrics()
    m = RQVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample


def test_rqvqa_video(video_sample):
    from ayase.modules.rqvqa import RQVQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = RQVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_rqvqa_preserves_unbounded_published_score(video_sample):
    from ayase.modules.rqvqa import RQVQAModule

    module = RQVQAModule({"test_mode": True})
    module._backend = "rqvqa"
    module._ml_available = True
    module._score_video = MagicMock(return_value=8.75)

    result = module.process(video_sample)

    assert result is video_sample
    assert result.quality_metrics.rqvqa_score == 8.75
    module._score_video.assert_called_once_with(video_sample)


def test_rqvqa_ensemble_size_is_bounded():
    from ayase.modules.rqvqa import RQVQAModule

    assert RQVQAModule({"ensemble_size": 0}).ensemble_size == 1
    assert RQVQAModule({"ensemble_size": 50}).ensemble_size == 10


def test_rqvqa_decodes_eight_published_anchors(synthetic_video):
    from ayase.modules.rqvqa import RQVQAModule

    frames, clips = RQVQAModule._decode_inputs(synthetic_video)

    assert len(frames) == 8
    assert len(clips) == 8
    assert all(len(clip) == 32 for clip in clips)
    # The 64-frame fixture is shorter than eight seconds, so the published
    # repeat-last-anchor rule must be used.
    assert (frames[-1] == frames[-2]).all()


def test_rqvqa_remaps_old_timm_swin_layout():
    from ayase.modules.rqvqa import _remap_swin_state_dict

    marker = object()
    result = _remap_swin_state_dict(
        {
            "layers.0.downsample.reduction.weight": marker,
            "layers.0.blocks.0.attn.relative_position_index": marker,
            "layers.0.blocks.0.norm1.weight": marker,
        }
    )

    assert result["layers.1.downsample.reduction.weight"] is marker
    assert result["layers.0.blocks.0.norm1.weight"] is marker
    assert not any(key.endswith("relative_position_index") for key in result)


def test_rqvqa_fastvqa_sampling_is_reproducible_without_global_rng_side_effects():
    import torch

    from ayase.modules.rqvqa import RQVQAModule

    module = RQVQAModule({"fastvqa_seed": 17, "test_mode": True})
    module._fastvqa_module = SimpleNamespace(
        device="cpu",
        _prepare_input=lambda _path: {
            "samples": {"fragments": torch.rand(1, 768, 1, 1, 1)}
        },
        _model=SimpleNamespace(
            backbone={"fragments": lambda fragments, **_kwargs: fragments}
        ),
    )
    state_before = torch.random.get_rng_state().clone()

    first = module._fastvqa_features(Path("test.mp4"))
    second = module._fastvqa_features(Path("test.mp4"))

    assert torch.equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), state_before)
