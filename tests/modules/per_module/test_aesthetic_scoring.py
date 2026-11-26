"""Tests for aesthetic_scoring module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_aesthetic_scoring_basics():
    from ayase.modules.aesthetic_scoring import AestheticScoringModule
    _test_module_basics(AestheticScoringModule, "aesthetic_scoring")

def test_aesthetic_scoring_image(image_sample):
    from ayase.modules.aesthetic_scoring import AestheticScoringModule
    image_sample.quality_metrics = QualityMetrics()
    m = AestheticScoringModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_aesthetic_scoring_video(video_sample):
    from ayase.modules.aesthetic_scoring import AestheticScoringModule
    video_sample.quality_metrics = QualityMetrics()
    m = AestheticScoringModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample

def test_aesthetic_batch_preserves_per_sample_warning_details(monkeypatch, tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    from ayase.models import Sample
    from ayase.modules import aesthetic_scoring as aesthetic

    module = aesthetic.AestheticScoringModule({"batch_frames": True})
    module._ml_available = True
    module._linear = torch.nn.Identity()
    samples = [Sample(path=tmp_path / name, is_video=True) for name in ("a.mp4", "b.mp4")]
    monkeypatch.setattr(module, "_load_images", lambda sample: [sample.path.name])

    def group_features(*args, **kwargs):
        assert kwargs["full_precision"]
        return [torch.tensor([[5.0], [7.0]]), torch.tensor([[4.5002]])]

    def single_features(*args, **kwargs):
        assert args[3] == ["b.mp4"]
        return torch.tensor([[4.4998]])

    monkeypatch.setattr(aesthetic, "cached_clip_image_feature_groups", group_features)
    monkeypatch.setattr(aesthetic, "cached_clip_image_features", single_features)
    assert module.process_batch(samples) == samples
    assert samples[0].quality_metrics.aesthetic_mlp_score == 6.0
    assert not samples[0].validation_issues
    assert samples[1].quality_metrics.aesthetic_mlp_score == pytest.approx(4.4998)
    assert samples[1].validation_issues[0].details["frame_scores"] == pytest.approx([4.4998])
