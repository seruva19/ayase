"""Tests for the facesim module (FaceSim-Cur / FaceSim-Arc, ConsisID and OpenS2V protocols)."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_facesim_basics():
    from ayase.modules.facesim import FaceSimModule
    _test_module_basics(FaceSimModule, "facesim")


def test_facesim_protocol_frame_counts():
    from ayase.modules.facesim import FaceSimModule
    assert FaceSimModule({"protocol": "consisid"}).num_frames == 16
    assert FaceSimModule({"protocol": "opens2v"}).num_frames == 32
    with pytest.raises(ValueError):
        FaceSimModule({"protocol": "other"})


def test_facesim_without_backend_leaves_fields_unset(tmp_path):
    from ayase.modules.facesim import FaceSimModule
    m = FaceSimModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.facesim_cur is None


def test_facesim_rejects_video_reference(tmp_path):
    from ayase.modules.facesim import FaceSimModule
    m = FaceSimModule()
    m._backend, m._arc = "consisid", object()  # past setup; the reference check comes first
    sample = Sample(path=tmp_path / "v.mp4", is_video=True, reference_path=tmp_path / "ref.mp4")
    assert m.process(sample).quality_metrics is None or sample.quality_metrics.facesim_cur is None
