"""Tests for ssimulacra2 module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ssimulacra2_basics():
    from ayase.modules.ssimulacra2 import SSIMULACRA2Module
    _test_module_basics(SSIMULACRA2Module, "ssimulacra2")

def test_ssimulacra2_no_reference(image_sample):
    from ayase.modules.ssimulacra2 import SSIMULACRA2Module
    m = SSIMULACRA2Module()
    result = m.process(image_sample)
    assert result is image_sample


def _patched_module(score):
    """SSIMULACRA2Module with a stubbed compute function (no real backend)."""
    from ayase.modules.ssimulacra2 import SSIMULACRA2Module

    m = SSIMULACRA2Module()
    m._ml_available = True
    m._compute_fn = lambda ref_rgb, dist_rgb: score
    return m


def test_ssimulacra2_warns_on_low_score(image_sample, tmp_path):
    """SSIMULACRA 2: 100 = identical (higher=better). A LOW score means
    visible distortion and must raise a warning."""
    import cv2
    import numpy as np

    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), np.zeros((32, 32, 3), dtype=np.uint8))
    image_sample.reference_path = ref
    image_sample.quality_metrics = QualityMetrics()

    m = _patched_module(10.0)  # low score = heavy distortion
    out = m.process(image_sample)
    assert out.quality_metrics.ssimulacra2 == 10.0
    assert any("SSIMULACRA" in i.message for i in out.validation_issues)


def test_ssimulacra2_no_warning_on_high_score(image_sample, tmp_path):
    """A near-identical score (>= threshold) must not warn."""
    import cv2
    import numpy as np

    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), np.zeros((32, 32, 3), dtype=np.uint8))
    image_sample.reference_path = ref
    image_sample.quality_metrics = QualityMetrics()

    m = _patched_module(95.0)
    out = m.process(image_sample)
    assert out.quality_metrics.ssimulacra2 == 95.0
    assert not any("SSIMULACRA" in i.message for i in out.validation_issues)
