"""Tests for the Google UVQ 1.5 module."""

from pathlib import Path

from ayase.models import QualityMetrics
from tests.modules.conftest import _test_module_basics


def test_uvq_basics():
    from ayase.modules.uvq import UVQModule

    _test_module_basics(UVQModule, "uvq")


def test_uvq_without_setup_is_graceful(video_sample):
    from ayase.modules.uvq import UVQModule

    result = UVQModule().process(video_sample)
    assert result is video_sample


def test_uvq_skips_images(image_sample):
    from ayase.modules.uvq import UVQModule

    module = UVQModule()
    module._backend = "uvq1p5"
    module._model = object()
    result = module.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics is None


def test_uvq_processes_reference_score(video_sample, monkeypatch):
    from ayase.modules.uvq import UVQModule

    class FakeModel:
        def infer(self, *args, **kwargs):
            return {"uvq1p5_score": 4.25}

    module = UVQModule()
    module._backend = "uvq1p5"
    module._model = FakeModel()
    monkeypatch.setattr(module, "_video_geometry", lambda path: (2, False, 24.0))

    result = module.process(video_sample)

    assert result is video_sample
    assert result.quality_metrics is not None
    assert result.quality_metrics.uvq1p5_score == 4.25


def test_uvq_rejects_out_of_range_score(video_sample, monkeypatch):
    from ayase.modules.uvq import UVQModule

    class FakeModel:
        def infer(self, *args, **kwargs):
            return {"uvq1p5_score": 5.5}

    module = UVQModule()
    module._backend = "uvq1p5"
    module._model = FakeModel()
    monkeypatch.setattr(module, "_video_geometry", lambda path: (2, False, 24.0))

    result = module.process(video_sample)

    assert result is video_sample
    assert result.quality_metrics is None


def test_uvq_field_group():
    from ayase.modules.uvq import UVQModule

    QualityMetrics.register_field_groups(UVQModule.metric_groups)
    metrics = QualityMetrics(uvq1p5_score=4.0)
    assert metrics.to_grouped_dict()["nr_quality"]["uvq1p5_score"] == 4.0


def test_uvq_decoder_does_not_invoke_shell(monkeypatch):
    from ayase.third_party.uvq.utils import video_reader

    observed = {}

    def fake_check_output(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        Path(command[-1]).write_bytes(bytes(range(12)))

    monkeypatch.setattr(video_reader.subprocess, "check_output", fake_check_output)
    video, real_frames = video_reader.load_video_1p5(
        "video;touch injected.mp4",
        video_length=1,
        video_fps=1,
        video_height=2,
        video_width=2,
    )

    assert isinstance(observed["command"], list)
    assert observed["command"][2] == "video;touch injected.mp4"
    assert "shell" not in observed["kwargs"]
    assert video.shape == (1, 1, 2, 2, 3)
    assert real_frames == 1


def test_uvq_weights_are_rendered_in_models_catalog():
    from ayase.models_doc import generate_models_doc

    document = generate_models_doc(fetch_licenses=False)

    assert "## Local Weight Files" in document
    assert "`uvq1p5/content_net.pth`" in document
    assert "`uvq1p5/distortion_net.pth`" in document
    assert "`uvq1p5/aggregation_net.pth`" in document
