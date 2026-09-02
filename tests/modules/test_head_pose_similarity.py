"""Tests for head-motion manner similarity.

The properties held here are the ones that separate this metric from
``head_motion_dynamics``: it needs a reference, it reports angles and rates apart
rather than folded together, and it says how much of the clip was usable.
"""

from pathlib import Path

import numpy as np

from tests.modules.conftest import _test_module_basics


def _series(values, fps=25.0):
    """A head-angle track: time column followed by pitch, yaw and roll."""
    values = np.asarray(values, dtype=np.float64)
    times = np.arange(len(values), dtype=np.float64) / fps
    return np.column_stack([times, values])


def _module(**config):
    from ayase.modules.head_pose_similarity import HeadPoseSimilarityModule

    return HeadPoseSimilarityModule(config or None)


def test_head_pose_similarity_basics():
    from ayase.modules.head_pose_similarity import HeadPoseSimilarityModule

    _test_module_basics(HeadPoseSimilarityModule, "head_pose_similarity")


def test_identical_distributions_agree_completely():
    from ayase.modules.head_pose_similarity import _agreement

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _agreement(values, values) == 1.0


def test_shifted_distribution_agrees_less():
    from ayase.modules.head_pose_similarity import _agreement

    still = np.array([0.1, 0.0, -0.1, 0.05, 0.0])
    restless = np.array([9.0, -8.0, 7.5, -9.5, 8.0])
    assert _agreement(still, restless) < _agreement(still, still)


def test_reference_is_required():
    from ayase.models import Sample

    module = _module()
    module._available = True
    sample = Sample(path=Path("clip.mp4"), is_video=True)
    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_angles_and_rates_reported_apart(monkeypatch):
    """Angles carry camera placement, rates survive it -- folding them loses that."""
    from ayase.models import Sample

    module = _module()
    module._available = True
    rng = np.random.default_rng(0)
    track = _series(rng.normal(size=(40, 3)))

    monkeypatch.setattr(type(module), "_series", lambda self, path: (track, 1.0))
    sample = Sample(path=Path("clip.mp4"), is_video=True, reference_path=Path("ref.mp4"))
    sample = module.process(sample)

    qm = sample.quality_metrics
    assert qm is not None
    assert qm.head_pose_angle_agreement == 1.0
    assert qm.head_pose_rate_agreement == 1.0
    assert qm.head_pose_similarity == 1.0
    assert qm.head_pose_similarity_coverage == 1.0


def test_rate_is_per_second_not_per_frame(monkeypatch):
    """The same motion sampled at two frame rates must yield the same rates."""
    from ayase.models import Sample

    module = _module()
    module._available = True
    step = np.linspace(0.0, 10.0, 40)
    slow = _series(np.column_stack([step, step, step]), fps=16.0)
    fast = _series(np.column_stack([step, step, step]), fps=25.0)

    def series(self, path):
        return (slow, 1.0) if "clip" in str(path) else (fast, 1.0)

    monkeypatch.setattr(type(module), "_series", series)
    sample = Sample(path=Path("clip.mp4"), is_video=True, reference_path=Path("ref.mp4"))
    sample = module.process(sample)

    # Same displacement per frame at different frame rates is a different rate per
    # second, so the two clips must NOT be reported as identical in rate.
    assert sample.quality_metrics.head_pose_rate_agreement < 1.0


def test_short_track_yields_no_score(monkeypatch):
    from ayase.models import Sample

    module = _module()
    module._available = True
    monkeypatch.setattr(type(module), "_series", lambda self, path: (None, 0.05))
    sample = Sample(path=Path("clip.mp4"), is_video=True, reference_path=Path("ref.mp4"))
    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_disabled_backend_leaves_sample_untouched():
    from ayase.models import Sample

    module = _module()
    sample = Sample(path=Path("clip.mp4"), is_video=True, reference_path=Path("ref.mp4"))
    assert module.process(sample) is sample
    assert sample.quality_metrics is None
