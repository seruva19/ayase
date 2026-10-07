from ayase.models import CaptionMetadata, QualityMetrics


def test_tifa_basics():
    from ayase.modules.tifa import TIFAModule
    from .conftest import _test_module_basics

    _test_module_basics(TIFAModule, "tifa")


def test_tifa_config():
    from ayase.modules.tifa import TIFAModule

    m = TIFAModule()
    assert "vqa_model" in m.default_config
    assert "question_generator" in m.default_config
    assert "filter_questions" in m.default_config
    assert "subsample" in m.default_config


def test_tifa_skip_without_caption(image_sample):
    from ayase.modules.tifa import TIFAModule

    m = TIFAModule()
    result = m.process(image_sample)
    # No caption → should skip
    assert result.quality_metrics is None or result.quality_metrics.tifa_score is None


def test_tifa_unavailable_without_backends(image_sample):
    # Without the official pipeline deps (torch/transformers checkpoints),
    # setup() must degrade gracefully and process() must be a no-op.
    from ayase.modules.tifa import TIFAModule

    m = TIFAModule()
    m.on_mount()
    if not m._ml_available:
        image_sample.quality_metrics = QualityMetrics()
        result = m.process(image_sample)
        assert result.quality_metrics.tifa_score is None


def test_tifa_field_exists():
    qm = QualityMetrics()
    assert hasattr(qm, "tifa_score")
    assert qm.tifa_score is None


def test_tifa_field_group():
    from ayase.pipeline import ModuleRegistry

    ModuleRegistry.discover_modules()
    groups = QualityMetrics._FIELD_GROUPS
    assert groups.get("tifa_score") == "alignment"
