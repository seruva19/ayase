"""Tests for concept_presence module."""

from ayase.models import QualityMetrics
from ..conftest import _test_module_basics


def test_concept_presence_basics():
    from ayase.modules.concept_presence import ConceptPresenceModule

    _test_module_basics(ConceptPresenceModule, "concept_presence")


def test_concept_presence_skip_no_ml(image_sample):
    """Module should return sample unchanged when _ml_available is False."""
    from ayase.modules.concept_presence import ConceptPresenceModule

    image_sample.quality_metrics = QualityMetrics()
    m = ConceptPresenceModule()
    # Do NOT call on_mount() / setup(), so _ml_available stays False
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.concept_presence is None
    assert result.quality_metrics.concept_count is None
    assert result.quality_metrics.concept_face_count is None


def test_concept_presence_image_with_mount(image_sample):
    """Module should process image sample after on_mount (test mode = heuristic)."""
    from ayase.modules.concept_presence import ConceptPresenceModule

    image_sample.quality_metrics = QualityMetrics()
    m = ConceptPresenceModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample


def test_concept_presence_video(video_sample):
    """Module should handle video sample gracefully."""
    from ayase.modules.concept_presence import ConceptPresenceModule

    video_sample.quality_metrics = QualityMetrics()
    m = ConceptPresenceModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_concept_presence_auto_mode():
    """Auto mode should detect face-related keywords."""
    from ayase.modules.concept_presence import ConceptPresenceModule

    m = ConceptPresenceModule()
    # Without any backends, auto_detect_mode falls back to "clip"
    assert m._auto_detect_mode(["a sunset over mountains"]) == "clip"
    assert m._auto_detect_mode(["a person smiling"]) == "clip"  # No face backend yet

    # Simulate having a face backend
    m._face_backend = "haar"
    mode = m._auto_detect_mode(["a person smiling"])
    assert mode in ("face", "combined")

    # Non-face concepts with face backend still returns "clip"
    assert m._auto_detect_mode(["a sunset over mountains"]) == "clip"


def test_concept_presence_fields():
    """Verify all concept_presence fields exist in QualityMetrics."""
    qm = QualityMetrics()
    assert hasattr(qm, "concept_presence")
    assert hasattr(qm, "concept_count")
    assert hasattr(qm, "concept_face_count")
    assert qm.concept_presence is None
    assert qm.concept_count is None
    assert qm.concept_face_count is None


def test_concept_presence_field_groups():
    """Verify field group assignments."""
    qm = QualityMetrics()
    groups = qm._FIELD_GROUPS
    assert groups.get("concept_presence") == "scene"
    assert groups.get("concept_count") == "scene"
    assert groups.get("concept_face_count") == "face"


def test_concept_presence_face_keywords():
    """Test that face keywords set is populated correctly."""
    from ayase.modules.concept_presence import _FACE_KEYWORDS

    assert "face" in _FACE_KEYWORDS
    assert "person" in _FACE_KEYWORDS
    assert "portrait" in _FACE_KEYWORDS
    assert "sunset" not in _FACE_KEYWORDS
