from ayase.models import QualityMetrics


def test_tonal_dynamic_range_basics():
    from ayase.modules.tonal_dynamic_range import TonalDynamicRangeModule
    from .conftest import _test_module_basics

    _test_module_basics(TonalDynamicRangeModule, "tonal_dynamic_range")


def test_tonal_dynamic_range_config():
    from ayase.modules.tonal_dynamic_range import TonalDynamicRangeModule

    m = TonalDynamicRangeModule()
    assert "low_percentile" in m.default_config
    assert "high_percentile" in m.default_config
    assert "subsample" in m.default_config


def test_tonal_dynamic_range_image(image_sample):
    from ayase.modules.tonal_dynamic_range import TonalDynamicRangeModule

    m = TonalDynamicRangeModule()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    score = result.quality_metrics.tonal_dynamic_range
    assert score is not None
    assert 0.0 <= score <= 100.0


def test_tonal_dynamic_range_video(video_sample):
    from ayase.modules.tonal_dynamic_range import TonalDynamicRangeModule

    m = TonalDynamicRangeModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    score = result.quality_metrics.tonal_dynamic_range
    assert score is not None
    assert 0.0 <= score <= 100.0


def test_tonal_dynamic_range_custom_percentiles(image_sample):
    from ayase.modules.tonal_dynamic_range import TonalDynamicRangeModule

    m_wide = TonalDynamicRangeModule({"low_percentile": 1, "high_percentile": 99})
    m_narrow = TonalDynamicRangeModule({"low_percentile": 10, "high_percentile": 90})
    r_wide = m_wide.process(image_sample)
    # Reset metrics for second pass
    image_sample.quality_metrics = QualityMetrics()
    r_narrow = m_narrow.process(image_sample)
    # Wider percentile range should give >= score than narrower
    assert r_wide.quality_metrics.tonal_dynamic_range >= r_narrow.quality_metrics.tonal_dynamic_range
