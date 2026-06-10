"""Tests for EvoQuality module."""

from ..conftest import _test_module_basics


def test_evoquality_basics():
    from ayase.modules.evoquality import EvoQualityModule

    _test_module_basics(EvoQualityModule, "evoquality")


def test_evoquality_no_backend_is_noop(image_sample):
    from ayase.modules.evoquality import EvoQualityModule

    result = EvoQualityModule().process(image_sample)

    assert result is image_sample
    if result.quality_metrics is not None:
        assert result.quality_metrics.evoquality_score is None


def test_evoquality_parse_boxed_output():
    from ayase.modules.evoquality import parse_evoquality_output

    text = (
        "The picture shows mild blur and compression artifacts, but the "
        "composition is clear. \\boxed{3.25}"
    )
    assert parse_evoquality_output(text) == 3.25


def test_evoquality_parse_last_boxed_wins():
    from ayase.modules.evoquality import parse_evoquality_output

    text = "First guess \\boxed{2.00}, but on reflection \\boxed{4.50}"
    assert parse_evoquality_output(text) == 4.5


def test_evoquality_parse_boxed_clamped_to_range():
    from ayase.modules.evoquality import parse_evoquality_output

    assert parse_evoquality_output("\\boxed{7.5}") == 5.0
    assert parse_evoquality_output("\\boxed{0.2}") == 1.0


def test_evoquality_parse_fallback_plain_number():
    from ayase.modules.evoquality import parse_evoquality_output

    assert parse_evoquality_output("Overall rating: 4.12") == 4.12


def test_evoquality_parse_no_score():
    from ayase.modules.evoquality import parse_evoquality_output

    assert parse_evoquality_output("The image quality is excellent.") is None
    assert parse_evoquality_output("") is None
    assert parse_evoquality_output(None) is None


def test_evoquality_quality_metrics_field():
    from ayase.models import QualityMetrics

    qm = QualityMetrics(evoquality_score=4.25)
    assert qm.evoquality_score == 4.25
