import numpy as np
import pytest

from ayase.models import CaptionMetadata, QualityMetrics, Sample


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
    m._backend = "heuristic"
    result = m.process(image_sample)
    # No caption → should skip
    assert result.quality_metrics is None or result.quality_metrics.tifa_score is None


def test_tifa_heuristic_scoring():
    from ayase.modules.tifa import TIFAModule

    m = TIFAModule()
    m._backend = "heuristic"
    # Short caption → higher score
    short = m._compute_heuristic("a red car")
    long = m._compute_heuristic(
        "a large red car driving down a busy street with many pedestrians "
        "crossing at the intersection near the tall glass building"
    )
    assert short > long
    assert 0.0 <= short <= 1.0
    assert 0.0 <= long <= 1.0


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


def test_tifa_heuristic_with_sample(image_sample):
    from ayase.modules.tifa import TIFAModule

    m = TIFAModule()
    m._backend = "heuristic"
    image_sample.caption = CaptionMetadata(text="a red car", length=9)
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.tifa_score is not None
    assert 0.0 <= result.quality_metrics.tifa_score <= 1.0


def test_tifa_field_exists():
    qm = QualityMetrics()
    assert hasattr(qm, "tifa_score")
    assert qm.tifa_score is None


def test_tifa_field_group():
    qm = QualityMetrics()
    groups = qm._FIELD_GROUPS
    assert groups.get("tifa_score") == "alignment"
