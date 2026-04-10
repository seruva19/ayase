from ayase.models import CaptionMetadata


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

    m = NemoCuratorModule()
    result = m.process(image_sample)
    # No caption → metrics untouched
    assert result.quality_metrics is None or result.quality_metrics.nemo_quality_score is None


def test_nemo_curator_setup_not_called_skips(image_sample):
    from ayase.modules.nemo_curator import NemoCuratorModule

    m = NemoCuratorModule()
    # Deliberately skip setup() — without ML backend, module skips gracefully
    image_sample.caption = CaptionMetadata(text="Some caption text here.", length=23)
    result = m.process(image_sample)
    # Without ML backend, module returns sample unchanged (metrics stay None)
    assert result.quality_metrics is None or result.quality_metrics.nemo_quality_score is None
