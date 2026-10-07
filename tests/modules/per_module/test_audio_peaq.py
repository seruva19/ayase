"""PEAQ BASIC via peaqb-fast module tests.

If the peaqb output yields only one of ODG/DI, the missing value must stay
``None`` — substituting ``0.0`` fabricates a published-metric number.
"""

from ayase.modules.audio_peaq import _parse_peaqb_output


def test_parse_both_values():
    out = _parse_peaqb_output("Objective Difference Grade: ODG = -1.234\nDistortion Index: DI = 5.6\n")
    assert out == (-1.234, 5.6)


def test_missing_di_stays_none():
    out = _parse_peaqb_output("ODG: -2.5\n")
    assert out is not None
    assert out[0] == -2.5
    assert out[1] is None


def test_missing_odg_stays_none():
    out = _parse_peaqb_output("Distortion Index: DI = 3.1\n")
    assert out is not None
    assert out[0] is None
    assert out[1] == 3.1


def test_no_values_returns_none():
    assert _parse_peaqb_output("peaqb: error: cannot decode input") is None
