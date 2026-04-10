from ayase.models import CaptionMetadata, QualityMetrics


def test_tifa_basics():
    from ayase.modules.tifa import TIFAModule
    from .conftest import _test_module_basics

    _test_module_basics(TIFAModule, "tifa")


def test_tifa_config():
    from ayase.modules.tifa import TIFAModule

    m = TIFAModule()
    assert "vqa_model" in m.default_config
    assert "num_questions" in m.default_config
    assert "subsample" in m.default_config


def test_tifa_skip_without_caption(image_sample):
    from ayase.modules.tifa import TIFAModule

    m = TIFAModule()
    result = m.process(image_sample)
    # No caption → should skip
    assert result.quality_metrics is None or result.quality_metrics.tifa_score is None


def test_tifa_question_generation():
    from ayase.modules.tifa import _generate_questions

    questions = _generate_questions("A red dog sitting on a green chair")
    assert len(questions) > 0
    # Should have questions about colors and objects
    q_texts = [q[0].lower() for q in questions]
    assert any("red" in q for q in q_texts)


def test_tifa_color_detection():
    from ayase.modules.tifa import _generate_questions

    questions = _generate_questions("A blue bird on a white fence")
    q_texts = [q[0].lower() for q in questions]
    assert any("blue" in q for q in q_texts)


def test_tifa_empty_caption():
    from ayase.modules.tifa import _generate_questions

    questions = _generate_questions("")
    assert len(questions) == 0


def test_tifa_field_exists():
    qm = QualityMetrics()
    assert hasattr(qm, "tifa_score")
    assert qm.tifa_score is None


def test_tifa_field_group():
    qm = QualityMetrics()
    groups = qm._FIELD_GROUPS
    assert groups.get("tifa_score") == "alignment"
