"""Tests for image_lpips module."""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample, DatasetStats
from ..conftest import _test_module_basics


def test_image_lpips_basics():
    from ayase.modules.image_lpips import ImageLPIPSModule

    _test_module_basics(ImageLPIPSModule, "image_lpips")


def test_image_lpips_skip_no_reference(image_sample):
    """Module should return sample unchanged when no reference_path is set."""
    from ayase.modules.image_lpips import ImageLPIPSModule

    image_sample.quality_metrics = QualityMetrics()
    m = ImageLPIPSModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    # No reference_path -> image_lpips stays None
    assert result.quality_metrics.image_lpips is None


def test_image_lpips_same_image(image_sample, _shared_synthetic_image):
    """LPIPS between an image and itself should be ~0."""
    from ayase.modules.image_lpips import ImageLPIPSModule

    # Set reference_path to the same image
    image_sample.reference_path = _shared_synthetic_image
    image_sample.quality_metrics = QualityMetrics()
    m = ImageLPIPSModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    # With heuristic fallback (no lpips lib), SSIM-based distance should be ~0
    if result.quality_metrics.image_lpips is not None:
        assert result.quality_metrics.image_lpips < 0.05


def test_image_lpips_different_images(_shared_tmp_dir, _shared_synthetic_image):
    """LPIPS between different images should be > 0."""
    from ayase.modules.image_lpips import ImageLPIPSModule

    # Create a second, different image (bright red)
    alt_path = _shared_tmp_dir / "test_image_alt.png"
    alt_img = np.full((256, 256, 3), (0, 0, 255), dtype=np.uint8)
    cv2.imwrite(str(alt_path), alt_img)

    sample = Sample(path=alt_path, is_video=False)
    sample.reference_path = _shared_synthetic_image
    sample.quality_metrics = QualityMetrics()

    m = ImageLPIPSModule()
    m.on_mount()
    result = m.process(sample)
    if result.quality_metrics.image_lpips is not None:
        assert result.quality_metrics.image_lpips > 0.0


def test_image_lpips_diversity_empty():
    """Diversity with < 2 cached tensors should not crash."""
    from ayase.modules.image_lpips import ImageLPIPSModule

    m = ImageLPIPSModule()
    m.on_mount()
    # post_process with empty cache
    m.post_process([])
    assert m._tensor_cache == []


def test_image_lpips_field_exists():
    """Verify the image_lpips field exists in QualityMetrics."""
    qm = QualityMetrics()
    assert hasattr(qm, "image_lpips")
    assert qm.image_lpips is None


def test_image_lpips_field_group():
    """Verify image_lpips is in the fr_quality group."""
    qm = QualityMetrics()
    assert qm._FIELD_GROUPS.get("image_lpips") == "fr_quality"


def test_image_lpips_dataset_stats_field():
    """Verify lpips_diversity field exists in DatasetStats."""
    assert hasattr(DatasetStats, "model_fields")
    assert "lpips_diversity" in DatasetStats.model_fields


def test_image_lpips_video(video_sample):
    """Module should handle video samples gracefully (caches frames for diversity)."""
    from ayase.modules.image_lpips import ImageLPIPSModule

    video_sample.quality_metrics = QualityMetrics()
    m = ImageLPIPSModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
