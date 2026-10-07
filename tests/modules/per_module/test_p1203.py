"""ITU-T P.1203 module tests.

The upstream ``itu_p1203`` API is
``P1203Standalone(input_json).calculate_complete()`` where ``input_json`` is
a dict with ``I13`` (video segments), ``I23`` (stalling) and ``IGen``
(display/device) sections — not a bare segment list. The pooled MOS is
returned under ``O46``.
"""

import sys
from types import SimpleNamespace

import pytest

from ayase.models import QualityMetrics, Sample, VideoMetadata
from ayase.modules.p1203 import P1203Module


class _FakeP1203Standalone:
    """Records the input JSON and answers like the real implementation."""

    def __init__(self, input_json):
        self.input_json = input_json

    def calculate_complete(self):
        return {"O21": 4.1, "O22": 3.8, "O46": 4.0}


def _sample(tmp_path, codec="h264"):
    s = Sample(path=tmp_path / "v.mp4", is_video=True)
    s.video_metadata = VideoMetadata(
        width=1920,
        height=1080,
        frame_count=240,
        fps=24.0,
        duration=10.0,
        codec=codec,
        bitrate=5_000_000,
        file_size=6_250_000,
    )
    return s


def test_p1203_official_input_schema(monkeypatch, tmp_path):
    captured = {}

    class Rec(_FakeP1203Standalone):
        def __init__(self, input_json):
            captured["input"] = input_json
            super().__init__(input_json)

    monkeypatch.setitem(sys.modules, "itu_p1203", SimpleNamespace(P1203Standalone=Rec))
    m = P1203Module()
    m.setup()
    out = m.process(_sample(tmp_path))

    spec = captured["input"]
    assert isinstance(spec, dict), "P1203Standalone takes a JSON dict, not a list"
    assert "I13" in spec and spec["I13"]["segments"], "missing video stream I13"
    seg = spec["I13"]["segments"][0]
    assert seg["codec"] == "h264"
    assert seg["bitrate"] == pytest.approx(5000.0)  # kbit/s
    assert spec["IGen"]["device"] in ("mobile", "handheld", "pc")
    assert out.quality_metrics is not None
    assert out.quality_metrics.p1203_mos == pytest.approx(4.0)


def test_p1203_non_h264_returns_none(monkeypatch, tmp_path):
    """Mode 0 supports only h264 — no fabricated score for other codecs."""
    monkeypatch.setitem(
        sys.modules, "itu_p1203", SimpleNamespace(P1203Standalone=_FakeP1203Standalone)
    )
    m = P1203Module()
    m.setup()
    out = m.process(_sample(tmp_path, codec="vp9"))
    assert out.quality_metrics is None or out.quality_metrics.p1203_mos is None
