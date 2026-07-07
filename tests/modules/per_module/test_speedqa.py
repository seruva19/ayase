"""Tests for speedqa module (deterministic SpEED-QA port)."""

import cv2
import numpy as np

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_speedqa_basics():
    from ayase.modules.speedqa import SpEEDQAModule
    _test_module_basics(SpEEDQAModule, "speedqa")


def test_speedqa_video(video_sample):
    """No reference_path -> module is a no-op and leaves speedqa_score unset."""
    from ayase.modules.speedqa import SpEEDQAModule
    video_sample.quality_metrics = QualityMetrics()
    m = SpEEDQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.speedqa_score is None


def _write_video(path, noise_std=0.0, seed=0):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (256, 256))
    for i in range(16):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        cx = 128 + int(50 * np.sin(i * 0.2))
        cy = 128 + int(50 * np.cos(i * 0.2))
        cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)
        for x in range(256):
            frame[:, x, 0] = min(255, x + i)
        if noise_std > 0:
            noisy = frame.astype(np.float64) + rng.normal(0, noise_std, frame.shape)
            frame = np.clip(noisy, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


def test_speedqa_identical_is_near_zero(tmp_dir):
    """Reference == distorted -> SpEED distortion index ~ 0."""
    from ayase.modules.speedqa import SpEEDQAModule

    ref = tmp_dir / "ref.mp4"
    _write_video(ref, noise_std=0.0)

    m = SpEEDQAModule()
    m.on_mount()
    assert m._backend == "port"

    sample = Sample(path=ref, is_video=True, reference_path=ref)
    sample.quality_metrics = QualityMetrics()
    m.process(sample)

    score = sample.quality_metrics.speedqa_score
    assert score is not None
    assert np.isfinite(score)
    assert score < 1e-6  # exact same frames -> zero difference


def test_speedqa_monotonic_with_distortion(tmp_dir):
    """Score is finite and increases monotonically with added distortion."""
    from ayase.modules.speedqa import SpEEDQAModule

    ref = tmp_dir / "clean.mp4"
    low = tmp_dir / "low.mp4"
    high = tmp_dir / "high.mp4"
    _write_video(ref, noise_std=0.0)
    _write_video(low, noise_std=12.0, seed=1)
    _write_video(high, noise_std=45.0, seed=2)

    m = SpEEDQAModule()
    m.on_mount()

    def _score(dist_path):
        s = Sample(path=dist_path, is_video=True, reference_path=ref)
        s.quality_metrics = QualityMetrics()
        m.process(s)
        return s.quality_metrics.speedqa_score

    s_low = _score(low)
    s_high = _score(high)

    assert s_low is not None and s_high is not None
    assert np.isfinite(s_low) and np.isfinite(s_high)
    assert s_low > 0.0
    assert s_high > s_low  # heavier distortion -> larger SpEED index


def test_speedqa_no_reference_graceful(tmp_dir):
    """Missing reference file -> graceful no-op, score stays None."""
    from ayase.modules.speedqa import SpEEDQAModule

    vid = tmp_dir / "only.mp4"
    _write_video(vid, noise_std=0.0)

    m = SpEEDQAModule()
    m.on_mount()
    sample = Sample(path=vid, is_video=True)  # no reference_path
    sample.quality_metrics = QualityMetrics()
    m.process(sample)
    assert sample.quality_metrics.speedqa_score is None
