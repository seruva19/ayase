"""Tests for the aed_apd_deep3d published-protocol stub."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_aed_apd_deep3d_basics():
    from ayase.modules.aed_apd_deep3d import AedApdDeep3DModule
    _test_module_basics(AedApdDeep3DModule, "aed_apd_deep3d")


def test_aed_apd_deep3d_external_backend():
    from ayase.modules.aed_apd_deep3d import AedApdDeep3DModule
    assert AedApdDeep3DModule.requires_external_backend is True
    meta = AedApdDeep3DModule.get_metadata()
    assert meta["provenance"]["aed"] == "published"
    assert "Deep3DFaceRecon" in meta["sources"]["aed"]


def test_aed_apd_deep3d_process_is_passthrough(tmp_path):
    from ayase.modules.aed_apd_deep3d import AedApdDeep3DModule
    m = AedApdDeep3DModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    out = m.process(sample)
    assert out is sample
    assert out.quality_metrics.aed is None
    assert out.quality_metrics.apd is None
