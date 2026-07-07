"""Tests for chronomagic module."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_chronomagic_basics():
    from ayase.modules.chronomagic import ChronoMagicModule
    _test_module_basics(ChronoMagicModule, "chronomagic")

def test_chronomagic_video(video_sample):
    from ayase.modules.chronomagic import ChronoMagicModule
    video_sample.quality_metrics = QualityMetrics()
    m = ChronoMagicModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def _cotracker_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    try:
        torch.hub.load("facebookresearch/co-tracker", "cotracker2")
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _cotracker_available(),
    reason="CoTracker2 (torch.hub) not available; CHScore real backend cannot run",
)
def test_chronomagic_chscore_real(video_sample, monkeypatch):
    """Real CoTracker2 CHScore path (skips if the backbone can't be fetched).

    Verifies the actual InternVideo2-free path: CoTracker2 loads, CHScore
    (TSI_score = 1/TSI_sum) is a real finite number, and MTScore stays None
    because InternVideo2 is not configured (no fabrication).
    """
    from ayase.modules.chronomagic import ChronoMagicModule
    from ayase.pipeline import PipelineModule

    # Force the real backend to load even when the suite runs in test mode.
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    monkeypatch.setattr(PipelineModule, "_global_test_mode", False, raising=False)

    m = ChronoMagicModule({"ch_grid_size": 10, "ch_size": 128})
    assert m.test_mode is False
    m.setup()
    assert m._backend == "real"
    assert m._ch_available is True
    assert m._mt_available is False  # InternVideo2 not configured

    video_sample.quality_metrics = QualityMetrics()
    result = m.process(video_sample)
    ch = result.quality_metrics.chronomagic_ch_score
    assert ch is not None
    assert ch > 0.0
    # MTScore must stay None — no heuristic substitute.
    assert result.quality_metrics.chronomagic_mt_score is None
