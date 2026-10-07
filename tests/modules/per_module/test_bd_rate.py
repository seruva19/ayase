"""Tests for bd_rate module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics
import pytest


def test_bd_rate_basics():
    from ayase.modules.bd_rate import BDRateModule
    _test_module_basics(BDRateModule, "bd_rate")

def test_bd_rate_extract(video_sample):
    from ayase.modules.bd_rate import BDRateModule
    m = BDRateModule()
    feat = m.extract_features(video_sample)
    # May be None for non-video or missing deps
    assert video_sample is not None


def test_bd_rate_configured_reference_curve():
    from ayase.modules.bd_rate import BDRateModule

    reference = [[500, 30], [1000, 35], [2000, 40], [4000, 45]]
    candidate = [(350, 30), (700, 35), (1400, 40), (2800, 45)]
    module = BDRateModule({"reference_curve": reference})

    score = module.compute_distribution_metric(candidate)

    assert score is not None
    assert score == pytest.approx(-30.0, abs=1e-8)


def test_bd_rate_rejects_incomplete_reference_curve():
    from ayase.modules.bd_rate import BDRateModule

    module = BDRateModule({"reference_curve": [[500, 30], [1000, 35]]})

    assert module.compute_distribution_metric(
        [(350, 30), (700, 35), (1400, 40), (2800, 45)]
    ) is None


@pytest.mark.parametrize(
    ("rates", "qualities"),
    [([1, 2, 3, 4], [30, 30, 35, 40]),
     ([0, 2, 3, 4], [30, 35, 40, 45]),
     ([1, 2, float("nan"), 4], [30, 35, 40, 45]),
     ([1, 2, 3], [30, 35, 40, 45])],
)
def test_bd_rate_undefined_curve_has_no_score(rates, qualities):
    from ayase.modules.bd_rate import _bd_rate

    assert _bd_rate(rates, qualities, [1, 2, 3, 4], [30, 35, 40, 45]) is None


def test_bd_rate_configured_curve_reaches_pipeline_stats(tmp_path):
    from ayase.modules.bd_rate import BDRateModule
    from ayase.models import Sample, VideoMetadata
    from ayase.pipeline import Pipeline

    module = BDRateModule({"reference_curve": [[500, 30], [1000, 35],
                                              [2000, 40], [4000, 45]]})
    pipeline = Pipeline([module])
    pipeline.start()
    try:
        for index, (bitrate, quality) in enumerate([(350, 30), (700, 35),
                                                   (1400, 40), (2800, 45)]):
            sample = Sample(
                path=tmp_path / f"encode-{index}.mp4", is_video=True,
                video_metadata=VideoMetadata(width=16, height=16, frame_count=30,
                                             fps=30, duration=1, bitrate=bitrate,
                                             file_size=100),
                quality_metrics=QualityMetrics(vmaf=quality),
            )
            pipeline.process_sample(sample)
    finally:
        pipeline.stop()
    assert pipeline.stats.bd_rate == pytest.approx(-30.0, abs=1e-8)
    assert pipeline.stats.metric_provenance["bd_rate"] == "published"
