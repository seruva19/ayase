"""Tests for the anatomy_check module.

Covers module basics, graceful behaviour when no pose backend is installed,
and unit tests of the anatomy rules over synthetic keypoint sets (no ML deps).
"""

from ..conftest import _test_module_basics
from ayase.modules.anatomy_check import (
    AnatomyCheckModule,
    check_person_anatomy,
    score_person_frames,
)


def _normal_person():
    """A well-proportioned, symmetric standing skeleton: 2 hands, 1 head."""
    body = {
        "nose": (0.50, 0.10, 0.9), "neck": (0.50, 0.20, 0.9),
        "r_shoulder": (0.42, 0.22, 0.9), "l_shoulder": (0.58, 0.22, 0.9),
        "r_elbow": (0.40, 0.38, 0.9), "l_elbow": (0.60, 0.38, 0.9),
        "r_wrist": (0.39, 0.54, 0.9), "l_wrist": (0.61, 0.54, 0.9),
        "r_hip": (0.45, 0.55, 0.9), "l_hip": (0.55, 0.55, 0.9),
        "r_knee": (0.44, 0.75, 0.9), "l_knee": (0.56, 0.75, 0.9),
        "r_ankle": (0.43, 0.95, 0.9), "l_ankle": (0.57, 0.95, 0.9),
    }
    return {
        "body": body,
        "hands": [(0.39, 0.54, 0.9), (0.61, 0.54, 0.9)],
        "heads": [(0.50, 0.10, 0.9)],
    }


def test_anatomy_check_basics():
    _test_module_basics(AnatomyCheckModule, "anatomy_check")


def test_backend_unavailable_is_graceful(image_sample):
    """With no pose backend mounted, the module must not set anatomy_score."""
    m = AnatomyCheckModule()
    assert m._backend == "unavailable"
    assert m._ml_available is False
    out = m.process(image_sample)
    assert out.quality_metrics is None or out.quality_metrics.anatomy_score is None


def test_normal_skeleton_is_plausible():
    plausible, reasons = check_person_anatomy(_normal_person())
    assert plausible is True
    assert reasons == []


def test_three_hands_is_implausible():
    """>2 high-confidence hand clusters trips the extra-hands rule."""
    person = _normal_person()
    person["hands"] = person["hands"] + [(0.50, 0.50, 0.9)]  # a spurious third hand
    plausible, reasons = check_person_anatomy(person)
    assert plausible is False
    assert any(r.startswith("extra_hands") for r in reasons)


def test_two_heads_is_implausible():
    """>1 high-confidence head cluster trips the extra-heads rule."""
    person = _normal_person()
    person["heads"] = person["heads"] + [(0.30, 0.10, 0.9)]
    plausible, reasons = check_person_anatomy(person)
    assert plausible is False
    assert any(r.startswith("extra_heads") for r in reasons)


def test_low_confidence_hands_do_not_count():
    """Extra hand clusters below HAND_CONF are ignored (no false positive)."""
    person = _normal_person()
    person["hands"] = person["hands"] + [(0.50, 0.50, 0.05)]  # low-conf ghost
    plausible, reasons = check_person_anatomy(person)
    assert plausible is True, reasons


def test_impossible_limb_ratio_is_implausible():
    """A forearm ~3.5x the upper arm violates the arm-ratio rule."""
    person = _normal_person()
    person["body"] = dict(person["body"])
    person["body"]["l_wrist"] = (0.61, 0.95, 0.9)  # stretch left forearm far past the hand
    plausible, reasons = check_person_anatomy(person)
    assert plausible is False
    assert any("arm_ratio" in r or "asymmetry" in r for r in reasons)


def test_score_is_fraction_of_plausible_person_frames():
    good = _normal_person()
    bad = _normal_person()
    bad["hands"] = bad["hands"] + [(0.5, 0.5, 0.9)]
    # One plausible person-frame, one implausible -> 0.5
    assert score_person_frames([[good], [bad]]) == 0.5


def test_no_persons_scores_none():
    """No detected person in any frame -> None (not 1.0)."""
    assert score_person_frames([[], []]) is None
    assert score_person_frames([]) is None
