"""Tests for song_eval module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_song_eval_basics():
    from ayase.modules.song_eval import SongEvalModule
    _test_module_basics(SongEvalModule, "song_eval")

def test_song_eval_image(image_sample):
    from ayase.modules.song_eval import SongEvalModule
    image_sample.quality_metrics = QualityMetrics()
    m = SongEvalModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_song_eval_video(video_sample):
    from ayase.modules.song_eval import SongEvalModule
    video_sample.quality_metrics = QualityMetrics()
    m = SongEvalModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_song_eval_generator_is_registered_torch_module():
    import torch.nn as nn

    from ayase.modules.song_eval import _build_song_eval_generator

    model = _build_song_eval_generator(
        in_features=16,
        ffd_hidden_size=32,
        num_classes=5,
        attn_layer_num=1,
    )

    assert isinstance(model, nn.Module)
    assert model.state_dict()
