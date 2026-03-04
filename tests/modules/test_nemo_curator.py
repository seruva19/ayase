from ayase.models import CaptionMetadata, QualityMetrics


def test_nemo_curator_basics():
    from ayase.modules.nemo_curator import NemoCuratorModule
    from .conftest import _test_module_basics

    _test_module_basics(NemoCuratorModule, "nemo_curator")


def test_nemo_curator_config():
    from ayase.modules.nemo_curator import NemoCuratorModule

    m = NemoCuratorModule()
    assert "backend" in m.default_config
    assert "model_name" in m.default_config
    assert "min_length" in m.default_config


def test_nemo_curator_skip_without_caption(image_sample):
    from ayase.modules.nemo_curator import NemoCuratorModule

    m = NemoCuratorModule({"backend": "heuristic"})
    m.setup()
    result = m.process(image_sample)
    # No caption → metrics untouched
    assert result.quality_metrics is None or result.quality_metrics.nemo_quality_score is None


def test_nemo_curator_heuristic_scoring(image_sample):
    from ayase.modules.nemo_curator import NemoCuratorModule

    m = NemoCuratorModule({"backend": "heuristic"})
    m.setup()
    image_sample.caption = CaptionMetadata(
        text="A beautifully composed sunset over the ocean, with vibrant orange and purple hues reflecting off the calm water.",
        length=113,
    )
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.nemo_quality_score is not None
    assert 0.0 <= result.quality_metrics.nemo_quality_score <= 1.0
    assert result.quality_metrics.nemo_quality_label in ("Low", "Medium", "High")


def test_nemo_curator_low_quality(image_sample):
    from ayase.modules.nemo_curator import NemoCuratorModule

    m = NemoCuratorModule({"backend": "heuristic"})
    m.setup()
    image_sample.caption = CaptionMetadata(text="ok", length=2)
    result = m.process(image_sample)
    assert result.quality_metrics.nemo_quality_score < 0.5


def test_nemo_curator_repetition_penalty(image_sample):
    from ayase.modules.nemo_curator import NemoCuratorModule

    m = NemoCuratorModule({"backend": "heuristic"})
    m.setup()

    image_sample.caption = CaptionMetadata(
        text="dog dog dog dog dog dog dog dog dog dog", length=39
    )
    result = m.process(image_sample)
    bad_score = result.quality_metrics.nemo_quality_score

    image_sample.quality_metrics = QualityMetrics()
    image_sample.caption = CaptionMetadata(
        text="A well-composed photograph showing diverse elements in natural lighting.",
        length=71,
    )
    result = m.process(image_sample)
    good_score = result.quality_metrics.nemo_quality_score

    assert good_score > bad_score


def test_nemo_curator_setup_not_called_warns(image_sample, caplog):
    from ayase.modules.nemo_curator import NemoCuratorModule

    m = NemoCuratorModule()
    # Deliberately skip setup()
    image_sample.caption = CaptionMetadata(text="Some caption text here.", length=23)
    result = m.process(image_sample)
    # Should still produce a score via heuristic fallback
    assert result.quality_metrics is not None
    assert result.quality_metrics.nemo_quality_score is not None
    assert "setup() was not called" in caplog.text
