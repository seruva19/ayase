"""Tests for exposure module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_exposure_basics():
    from ayase.modules.exposure import ExposureModule
    _test_module_basics(ExposureModule, "exposure")

def test_exposure_image(image_sample):
    from ayase.modules.exposure import ExposureModule
    image_sample.quality_metrics = QualityMetrics()
    m = ExposureModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_exposure_video(video_sample):
    from ayase.modules.exposure import ExposureModule
    video_sample.quality_metrics = QualityMetrics()
    m = ExposureModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def _write_gray(tmp_path, value):
    import cv2
    import numpy as np
    from ayase.models import Sample

    path = tmp_path / f"gray_{value}.png"
    img = np.full((32, 32, 3), value, dtype=np.uint8)
    img[:8, :8] = 128  # 1/16 of pixels are mid-gray
    cv2.imwrite(str(path), img)
    return Sample(path=path, is_video=False)


def test_exposure_writes_pixel_ratios(tmp_path):
    from ayase.modules.exposure import ExposureModule

    m = ExposureModule()
    m.on_mount()

    dark = m.process(_write_gray(tmp_path, 5))
    assert dark.quality_metrics.underexposed_pixel_ratio == 15 / 16
    assert dark.quality_metrics.overexposed_pixel_ratio == 0.0
    assert any("Underexposure" in i.message for i in dark.validation_issues)

    bright = m.process(_write_gray(tmp_path, 250))
    assert bright.quality_metrics.overexposed_pixel_ratio == 15 / 16
    assert bright.quality_metrics.underexposed_pixel_ratio == 0.0

    mid = m.process(_write_gray(tmp_path, 128))
    assert mid.quality_metrics.underexposed_pixel_ratio == 0.0
    assert mid.quality_metrics.overexposed_pixel_ratio == 0.0
